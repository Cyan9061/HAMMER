"""
优化版RAG Prompt构建器 - 实现真正的批量embedding和检索
深度优化：先批量embedding所有queries并且筛选coreset，再分别进行RAG检索和prompt构建
"""

import time
import typing as T
from dataclasses import dataclass,field
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.base.embeddings.base import BaseEmbedding

from hammer.logger import logger
from hammer.flows import Flow, RAGFlow
from hammer.utils.coreset import select_coreset_from_embeddings, CoresetResult
from hammer.tuner.main_tuner_mcts import GPU_QUERY_EMBED_LIST, GPU_BATCHSIZE, DEVICE_ID
@dataclass
class BatchRAGPromptResult:
    """批量RAG Prompt构建结果"""
    final_prompts: List[str]
    queries: List[str]
    retrieved_nodes_list: List[List[NodeWithScore]]
    processed_nodes_list: List[List[NodeWithScore]]
    context_strs: List[str]
    processing_time: float
    embedding_time: float
    retrieval_time: float
    success_count: int
    failed_indices: List[int]
    error_messages: List[str]
    coreset_result: Optional[CoresetResult] = field(default=None, repr=False)
    coreset_train_indices: List[int] = field(default_factory=list, repr=False)
    query_embeddings: Optional[np.ndarray] = field(default=None, repr=False)

class OptimizedRAGPromptBuilder:
    """
    优化版RAG Prompt构建器
    核心优化：批量embedding -> 批量检索 -> 批量prompt构建
    """
    
    def __init__(self, flow: Flow, max_workers: int = None):
        """
        初始化优化版RAG Prompt构建器
        
        Args:
            flow: Flow对象，包含RAG配置
            max_workers: 最大并发数，用于LLM API调用
        """
        self.flow = flow
        self.query_engine = flow.query_engine
        self.max_workers = max_workers  # 🔧 添加max_workers支持
        
        # 提取query_engine的组件
        self._extract_components()
        
        # 🚀 初始化并行reranker
        self._init_parallel_reranker()
        
    def _init_parallel_reranker(self):
        """初始化批处理reranker"""
        self.parallel_reranker = None
        
        # 🎯 定义所有专用reranker模型（非通用LLM）
        SPECIALIZED_RERANKERS = {
            "EchoRank",           # EchoRank重排序器
            "flashrank",          # 快速ONNX模型
            "TransformerRanker",  # 通用Transformer重排序器
            "MonoT5",             # MonoT5重排序器
            "RankT5",             # RankT5重排序器
            "MonoBERT",           # MonoBERT重排序器
            "InRanker",           # InRanker模型
            "ColbertRanker",      # ColBERT重排序器
            "TWOLAR",             # TWOLAR重排序器
            "twolar",
            "monobert_ranker",    # MonoBERT变体
            "inranker",           # InRanker变体
            "echorank",           # EchoRank变体
            "transformer_ranker", # TransformerRanker变体
            "monot5",             # MonoT5变体
            "rankt5",             # RankT5变体
            "listt5",             # ListT5变体
            "Flashrank",          # Flashrank变体
            "FlashRank"
        }
        
        try:
            # 检查是否启用reranker
            if hasattr(self.flow, 'params') and self.flow.params:
                reranker_enabled = self.flow.params.get('reranker_enabled', False)
                if reranker_enabled:
                    reranker_llm = self.flow.params.get('reranker_llm', 'flashrank')
                    reranker_top_k = self.flow.params.get('reranker_top_k', 5)
                    
                    # 🚀 关键改进：使用真正的批处理reranker
                    if reranker_llm in SPECIALIZED_RERANKERS:
                        logger.info(f"🎯 检测到专用reranker模型 '{reranker_llm}'，创建批处理reranker")
                        self.parallel_reranker = BatchReranker(
                            reranker_name=reranker_llm,
                            top_k=reranker_top_k
                        )
                    else:
                        # 通用LLM暂时禁用（根据用户要求抛弃LLM支持）
                        logger.warning(f"⚠️ 通用LLM reranker '{reranker_llm}' 暂不支持，跳过处理")
                        self.parallel_reranker = None
                else:
                    logger.info("⚪ Reranker未启用")
            
        except Exception as e:
            logger.error(f"❌ 批处理reranker初始化失败: {e}")
            self.parallel_reranker = None
        
    def _extract_components(self):
        """从query_engine提取关键组件，包括embedding模型"""
        try:
            # 处理可能的TransformQueryEngine包装（如HyDE）
            base_engine = self.query_engine
            self.query_transforms = []
            
            while isinstance(base_engine, TransformQueryEngine):
                self.query_transforms.append(base_engine._query_transform)
                base_engine = base_engine._query_engine
            
            # 现在base_engine应该是RetrieverQueryEngine
            if isinstance(base_engine, RetrieverQueryEngine):
                self.retriever = base_engine._retriever
                self.node_postprocessors = base_engine._node_postprocessors or []
                self.response_synthesizer = base_engine._response_synthesizer
            else:
                raise ValueError(f"不支持的query_engine类型: {type(base_engine)}")
            
            # 🔑 关键优化：根据配置智能提取embedding模型
            self.embedding_model = self._smart_extract_embedding_model()
            
            # 提取prompt模板
            self.prompt_template = self._extract_prompt_template()
            
            logger.debug(f"✅ 成功提取RAG组件: retriever={type(self.retriever).__name__}, "
                        f"embedding_model={type(self.embedding_model).__name__ if self.embedding_model else 'None'}, "
                        f"postprocessors={len(self.node_postprocessors)}, "
                        f"transforms={len(self.query_transforms)}")
                        
        except Exception as e:
            logger.error(f"❌ 提取RAG组件失败: {e}")
            raise
    
    def _smart_extract_embedding_model(self) -> Optional[BaseEmbedding]:
        """
        智能提取embedding模型：根据配置决定是否需要embedding模型
        sparse模式（如BM25）不需要embedding模型，可以跳过搜索
        """
        # 1. 首先检查flow配置
        rag_method = None
        if hasattr(self.flow, 'params') and self.flow.params:
            rag_method = self.flow.params.get('rag_method', '').lower()
        
        # 2. 根据rag_method决定是否需要embedding模型
        if rag_method == 'sparse':
            logger.info("🔧 检测到sparse模式（如BM25），跳过embedding模型搜索")
            return None
        elif rag_method in ['dense', 'hybrid']:
            logger.info(f"🔧 检测到{rag_method}模式，需要embedding模型，开始搜索...")
            return self._extract_embedding_model()
        else:
            # 3. 未明确配置或配置不明确时，尝试搜索embedding模型
            logger.info(f"🔧 rag_method配置不明确('{rag_method}')，尝试自动检测embedding模型...")
            return self._extract_embedding_model()
    
    #
    # 在 optimized_rag_prompt_builder.py 文件中，替换这个方法
    #
    def _extract_embedding_model(self) -> Optional[BaseEmbedding]:
        """
        深度提取embedding模型。
        通过递归搜索，能够穿透复杂的复合检索器（如FusionRetriever）
        找到底层的embedding模型实例。
        """
        
        def _find_recursively(obj: Any) -> Optional[BaseEmbedding]:
            """递归搜索函数"""
            # 1. Base Case: 直接在当前对象上查找
            if hasattr(obj, '_embed_model') and obj._embed_model is not None:
                logger.debug(f"✅ 在 {type(obj).__name__} 上直接找到 _embed_model")
                return obj._embed_model
            
            # 2. 检查常见的llama-index内部属性
            if hasattr(obj, '_service_context') and hasattr(obj._service_context, 'embed_model'):
                logger.debug(f"✅ 在 {type(obj).__name__} 的 _service_context 中找到 embed_model")
                return obj._service_context.embed_model

            if hasattr(obj, '_vector_store') and hasattr(obj._vector_store, '_embed_model'):
                logger.debug(f"✅ 在 {type(obj).__name__} 的 _vector_store 中找到 _embed_model")
                return obj._vector_store._embed_model

            # 3. Recursive Step: 深入搜索嵌套的检索器
            #    这对于 FusionRetriever 或类似的复合检索器至关重要
            if hasattr(obj, '_retrievers') and isinstance(obj._retrievers, list):
                logger.debug(f"🕵️‍♂️ 在 {type(obj).__name__} 中发现 _retrievers 列表，正在深入搜索...")
                for sub_retriever in obj._retrievers:
                    found_model = _find_recursively(sub_retriever)
                    if found_model:
                        return found_model # 只要找到一个就立刻返回
            
            # 4. Recursive Step: 深入搜索QueryEngine中的retriever
            if hasattr(obj, '_query_engine') and obj._query_engine is not None:
                logger.debug(f"🕵️‍♂️ 在 {type(obj).__name__} 中发现 _query_engine，正在深入搜索...")
                return _find_recursively(obj._query_engine)
                
            if hasattr(obj, '_retriever') and obj._retriever is not None:
                logger.debug(f"🕵️‍♂️ 在 {type(obj).__name__} 中发现 _retriever，正在深入搜索...")
                return _find_recursively(obj._retriever)

            # 5. 如果所有尝试都失败
            return None

        logger.info("🔍 开始深度搜索Embedding模型...")
        model = _find_recursively(self.retriever)
        
        if model:
            model_class = type(model).__name__
            model_path = "未知路径"
            
            # 🔥 提取真实的模型路径信息
            try:
                if hasattr(model, 'model_name') and model.model_name:
                    model_path = model.model_name
                elif hasattr(model, '_model_name') and model._model_name:
                    model_path = model._model_name
                elif hasattr(model, 'model') and hasattr(model.model, 'name_or_path'):
                    model_path = model.model.name_or_path
                elif hasattr(model, '_model') and hasattr(model._model, 'name_or_path'):
                    model_path = model._model.name_or_path
                elif hasattr(model, 'embedding_model_name'):
                    model_path = model.embedding_model_name
                else:
                    # 尝试从配置中获取
                    if hasattr(model, '__dict__'):
                        for attr_name, attr_value in model.__dict__.items():
                            if 'model' in attr_name.lower() and isinstance(attr_value, str) and ('/' in attr_value or attr_value.startswith('/mnt')):
                                model_path = attr_value
                                break
            except Exception as e:
                logger.warning(f"⚠️ 提取模型路径时出错: {e}")
            
            logger.info(f"🎉 成功找到Embedding模型: {model_class}")
            logger.info(f"📍 实际使用的模型路径: {model_path}")
            return model
        else:
            # 只有在所有递归尝试都失败后才发出警告
            logger.error("❌ 深度搜索后仍无法提取embedding模型！批量Embedding将失效。")
            logger.error("请检查您的Retriever结构，确保其中至少有一个子检索器持有有效的embedding模型。")
            return None
    
    def _extract_prompt_template(self) -> BasePromptTemplate:
        """提取prompt模板"""
        try:
            if hasattr(self.response_synthesizer, '_text_qa_template'):
                return self.response_synthesizer._text_qa_template
            elif hasattr(self.flow, 'prompt_template') and self.flow.prompt_template:
                return self.flow.prompt_template
            else:
                return DEFAULT_TEXT_QA_PROMPT
        except Exception as e:
            logger.warning(f"⚠️ 提取prompt模板失败，使用默认模板: {e}")
            return DEFAULT_TEXT_QA_PROMPT
    
    def _batch_embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        批量embed所有queries
        核心优化：一次性计算所有query embeddings
        改进版：添加详细进度显示和错误处理
        """
        if not self.embedding_model:
            logger.warning("⚠️ 无embedding模型，为支持Coreset功能，使用默认BGE-M3模型")
            
            try:
                from hammer.huggingface_helper import get_embedding_model
                
                default_model_path = "/mnt/data/wangshu/llm_lm/bge-m3"
                logger.info(f"🔧 正在初始化默认embedding模型: {default_model_path}")
                
                # ✅ 简化版：使用最基本的参数
                self.embedding_model, _ = get_embedding_model(
                    default_model_path,
                    timeout_config=None,        # 使用默认超时配置
                    device = f'cuda:{DEVICE_ID}',            
                    use_hf_endpoint_models=False,  # 使用本地模型
                )
                
                logger.info(f"✅ 默认embedding模型初始化成功: {type(self.embedding_model).__name__}")
                
            except Exception as e:
                logger.error(f"❌ 默认embedding模型初始化失败: {e}")
                logger.warning("⚠️ Sparse模式下无法使用Coreset功能")
                return []
        
        total_queries = len(queries)
        logger.info(f"🚀 开始批量embedding: {total_queries}个queries")
        logger.info(f"📊 Embedding模型信息: {type(self.embedding_model).__name__}")
        
        try:
            # 🔑 关键修改：使用BatchLLMCaller替代逐个LLM调用
            logger.info("📝 开始批量查询转换...")
            transform_start = time.time()
            transformed_queries = []
            failed_transforms = 0
            
            # 🔥 修复检测逻辑：优先检查Flow配置中的HyDE和查询分解参数
            has_hyde_flow_config = False
            has_decomposition_flow_config = False
            
            if hasattr(self.flow, 'params') and self.flow.params:
                has_hyde_flow_config = self.flow.params.get('hyde_enabled', False)
                has_decomposition_flow_config = self.flow.params.get('rag_query_decomposition_enabled', False)
                logger.info(f"🔧 Flow配置检查: HyDE={has_hyde_flow_config}, 查询分解={has_decomposition_flow_config}")
            
            # 原检测逻辑（注释保留作为备用）
            # has_hyde = any("hyde" in str(type(transform)).lower() for transform in self.query_transforms)
            # has_decomposition = any("decomposition" in str(type(transform)).lower() for transform in self.query_transforms)
            
            # 🔥 核心修复：优先使用Flow配置
            has_hyde = has_hyde_flow_config
            has_decomposition = has_decomposition_flow_config
            
            if has_hyde or has_decomposition:
                # 动态导入BatchLLMCaller（避免循环导入）
                from hammer.utils.batch_api_evaluator import get_batch_llm_caller
                
                # 🔥 关键修复：从Flow配置中获取正确的模型名称
                hyde_model_name = "Qwen2-7b"  # 默认值
                decomp_model_name = "Qwen2-7b"  # 默认值
                
                if hasattr(self.flow, 'params') and self.flow.params:
                    if has_hyde:
                        hyde_model_name = self.flow.params.get('hyde_llm_name', self.flow.params.get('hyde_llm', 'Qwen2-7b'))
                        logger.info(f"🔧 HyDE将使用模型: {hyde_model_name}")
                    if has_decomposition:
                        decomp_model_name = self.flow.params.get('rag_query_decomposition_llm_name', 
                                                               self.flow.params.get('query_decomposition_llm', 'Qwen2-7b'))
                        logger.info(f"🔧 查询分解将使用模型: {decomp_model_name}")
                
                # 优先使用HyDE模型，如果没有HyDE则使用查询分解模型
                selected_model = hyde_model_name if has_hyde else decomp_model_name
                
                # 🔧 重要修复：传递正确的model_name和max_workers参数
                batch_caller = get_batch_llm_caller(model_name=selected_model, max_workers=self.max_workers)
                
                if has_hyde:
                    logger.info("🔄 执行批量HyDE转换...")
                    hyde_results = batch_caller.batch_hyde_transform(queries)
                    transformed_queries = hyde_results
                    # 统计失败数
                    failed_transforms = sum(1 for result in hyde_results if "[HyDE失败]" in result or "[HyDE异常]" in result)
                    
                elif has_decomposition:
                    logger.info("🔄 Query decomposition检测到，将在检索阶段统一处理，embedding阶段使用原查询")
                    # 🚀 新策略：跳过embedding阶段的查询分解，在检索阶段统一进行两阶段高并发处理
                    transformed_queries = queries  # 直接使用原查询进行embedding
                else:
                    # 其他转换类型：保持原有逻辑作为回退
                    logger.info("🔄 应用其他类型的查询转换...")
                    for i, query in enumerate(queries):
                        transformed_query = query
                        for j, transform in enumerate(self.query_transforms):
                            try:
                                query_bundle = QueryBundle(query_str=transformed_query)
                                transformed_bundle = transform(query_bundle)
                                transformed_query = transformed_bundle.query_str
                            except Exception as e:
                                logger.warning(f"⚠️ 转换器{j+1}失败 (query {i+1}): {e}")
                                failed_transforms += 1
                        transformed_queries.append(transformed_query)
            else:
                # 无需转换，直接使用原查询
                transformed_queries = queries
                logger.info("ℹ️ 无需查询转换，直接使用原查询")
            
            transform_duration = time.time() - transform_start
            logger.info(f"📝 批量查询转换完成: 耗时{transform_duration:.2f}s, 失败{failed_transforms}个")
            
            if failed_transforms > 0:
                logger.warning(f"⚠️ 共有{failed_transforms}个查询转换失败")
            
            # 🚀 批量embedding - 核心优化点
            logger.info(f"🔄 开始批量embedding计算...")
            logger.info(f"📊 将处理{len(transformed_queries)}个转换后的查询")
            embedding_start = time.time()
            
            # 检查embedding模型的可用方法
            has_batch_method = hasattr(self.embedding_model, 'get_text_embeddings')
            has_single_method = hasattr(self.embedding_model, 'get_text_embedding') 
            has_encode_method = hasattr(self.embedding_model, 'encode')
            
            logger.info(f"📊 Embedding方法检测: batch={has_batch_method}, single={has_single_method}, encode={has_encode_method}")
            
            if has_batch_method:
                # 使用批量接口，分批处理避免CUDA OOM
                logger.info("🚀 使用智能分批embedding接口")
                batch_size = 32  # 减小批量大小避免OOM
                embeddings = []
                
                try:
                    # 尝试分批处理
                    for i in range(0, len(transformed_queries), batch_size):
                        batch_queries = transformed_queries[i:i+batch_size]
                        batch_embeddings = self.embedding_model.get_text_embeddings(batch_queries)
                        embeddings.extend(batch_embeddings)
                        if i == 0:
                            logger.info(f"✅ 首批embedding成功，维度: {len(batch_embeddings[0]) if batch_embeddings else 'None'}")
                    
                    logger.info(f"✅ 智能分批embedding成功，返回{len(embeddings)}个结果")
                    
                except Exception as e:
                    logger.error(f"❌ 智能分批embedding失败: {e}")
                    logger.info("🔄 切换到多GPU并行方案...")
                    # 使用多GPU并行方案
                    embeddings = self._multi_gpu_parallel_embed(transformed_queries)
                    if embeddings:
                        logger.info(f"✅ 多GPU并行embedding成功，返回{len(embeddings)}个结果")
                    else:
                        raise Exception("多GPU并行embedding也失败")
                    
            # elif has_single_method:
            #     # 🔄 串行embedding方法已被注释，优先使用多GPU并行方案
            #     # 回退到单个embedding，但添加进度显示
            #     logger.info("🔄 回退到单个embedding接口，逐个处理...")
            #     embeddings = []
            #     failed_embeddings = 0
            #     
            #     for i, query in enumerate(transformed_queries):
            #         if i == 0:
            #             logger.info(f"⚡ 开始处理第一个embedding: {query[:50]}...")
            #         elif i > 0 and i % 100 == 0:  # 进度显示
            #             elapsed = time.time() - embedding_start
            #             eta = elapsed / i * (total_queries - i) if i > 0 else 0
            #             speed = i / elapsed if elapsed > 0 else 0
            #             logger.info(f"⚡ Embedding进度: {i}/{total_queries} ({i/total_queries*100:.1f}%) "
            #                        f"- 速度{speed:.1f} queries/s, 已用时{elapsed:.1f}s, 预计剩余{eta:.1f}s")
            #         
            #         try:
            #             embedding = self.embedding_model.get_text_embedding(query)
            #             embeddings.append(embedding)
            #             
            #             if i == 0:
            #                 logger.info(f"✅ 第一个embedding成功，维度: {len(embedding) if embedding else 'None'}")
            #                 
            #         except Exception as e:
            #             logger.warning(f"⚠️ 单个embedding失败 (query {i+1}): {e}")
            #             embeddings.append([])  # 空embedding作为占位符
            #             failed_embeddings += 1
            #     
            #     if failed_embeddings > 0:
            #         logger.warning(f"⚠️ 共有{failed_embeddings}个embedding计算失败")
                    
            else:
                # 🚀 如果批量接口不可用，直接使用多GPU并行方案
                logger.info("🔄 批量接口不可用，直接使用多GPU并行方案...")
                embeddings = self._multi_gpu_parallel_embed(transformed_queries)
                if not embeddings:
                    logger.error("❌ 多GPU并行embedding失败，无法继续")
                    return []
            
            embedding_duration = time.time() - embedding_start
            successful_embeddings = sum(1 for emb in embeddings if emb)
            total_duration = transform_duration + embedding_duration
            
            logger.info(f"🚀 批量embedding完成: {successful_embeddings}/{total_queries}个成功")
            logger.info(f"⏱️ 详细耗时: 转换{transform_duration:.2f}s + embedding{embedding_duration:.2f}s = 总计{total_duration:.2f}s")
            logger.info(f"📊 处理速度: {total_queries/total_duration:.1f} queries/s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 批量embedding失败: {e}")
            logger.error(f"💡 调试信息:")
            logger.error(f"  - embedding_model类型: {type(self.embedding_model)}")
            logger.error(f"  - queries数量: {len(queries)}")
            logger.error(f"  - query_transforms数量: {len(self.query_transforms) if hasattr(self, 'query_transforms') else 'None'}")
            if self.query_transforms:
                for i, transform in enumerate(self.query_transforms):
                    logger.error(f"  - 转换器{i+1}: {type(transform).__name__}")
            return []
    
    def _multi_gpu_parallel_embed(self, queries: List[str]) -> List[List[float]]:
        """多GPU并行embedding - 核心性能优化"""
        available_gpus = GPU_QUERY_EMBED_LIST#[4, 5, 6, 7]  # 用户指定的可用GPU
        batch_size = GPU_BATCHSIZE  # 每个GPU的批量大小
        
        logger.info(f"🔥 启动多GPU并行embedding: {len(available_gpus)}个GPU并行处理{len(queries)}个查询")
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import torch
            
            # 分配查询到不同GPU
            gpu_tasks = []
            queries_per_gpu = len(queries) // len(available_gpus) + 1
            
            for i, gpu_id in enumerate(available_gpus):
                start_idx = i * queries_per_gpu
                end_idx = min((i + 1) * queries_per_gpu, len(queries))
                if start_idx < len(queries):
                    gpu_queries = queries[start_idx:end_idx]
                    gpu_tasks.append((gpu_id, gpu_queries, start_idx))
            
            results = [None] * len(queries)
            
            def gpu_worker(gpu_id, gpu_queries, start_idx):
                """GPU工作线程"""
                try:
                    # 为每个GPU创建独立的embedding模型实例
                    from hammer.huggingface_helper import get_hf_embedding_model
                    from hammer.studies import TimeoutConfig
                    
                    gpu_model = get_hf_embedding_model(
                        self.embedding_model.model_name,
                        timeout_config=TimeoutConfig(),
                        total_chunks=len(gpu_queries),
                        device=f"cuda:{gpu_id}"
                    )
                    
                    # 分批处理避免单个GPU内存溢出
                    gpu_embeddings = []
                    for i in range(0, len(gpu_queries), batch_size):
                        batch = gpu_queries[i:i+batch_size]
                        try:
                            batch_emb = gpu_model.get_text_embeddings(batch)
                            gpu_embeddings.extend(batch_emb)
                        except:
                            # GPU内存不足时，回退到单个embedding
                            for query in batch:
                                gpu_embeddings.append(gpu_model.get_text_embedding(query))
                    
                    return (start_idx, gpu_embeddings)
                    
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 工作失败: {e}")
                    return (start_idx, [])
            
            # 并行执行
            with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
                futures = [executor.submit(gpu_worker, gpu_id, gpu_queries, start_idx) 
                          for gpu_id, gpu_queries, start_idx in gpu_tasks]
                
                for future in as_completed(futures):
                    start_idx, gpu_embeddings = future.result()
                    for i, emb in enumerate(gpu_embeddings):
                        if start_idx + i < len(queries):
                            results[start_idx + i] = emb
            
            # 过滤None结果
            valid_embeddings = [emb for emb in results if emb is not None]
            logger.info(f"✅ 多GPU并行embedding完成: {len(valid_embeddings)}/{len(queries)}个成功")
            
            return valid_embeddings if len(valid_embeddings) > 0 else []
            
        except Exception as e:
            logger.error(f"❌ 多GPU并行embedding失败: {e}")
            return []

    def _batch_retrieve_with_embeddings(self, queries: List[str], query_embeddings: List[List[float]]) -> List[List[NodeWithScore]]:
        """
        [新优化] 两阶段高并发查询分解检索
        阶段1：批量分解 (用默认并发数)  
        阶段2：扁平化并发检索 (32并发)
        """
        # 🔥 修复检测逻辑：优先检查Flow配置中的查询分解参数
        from hammer.mcts.hierarchical_search import query_decomposition_on
        
        # 首先检查Flow配置中的查询分解设置
        flow_query_decomp_enabled = False
        if hasattr(self.flow, 'params') and self.flow.params:
            flow_query_decomp_enabled = self.flow.params.get('rag_query_decomposition_enabled', False)
            logger.info(f"🔧 Flow配置中查询分解状态: {flow_query_decomp_enabled}")
        
        # 检查全局控制变量
        global_query_decomp_enabled = query_decomposition_on
        logger.info(f"🔧 全局查询分解控制: {global_query_decomp_enabled}")
        
        # 🔥 核心修复：优先使用Flow配置，其次使用全局控制变量
        query_decomp_needed = flow_query_decomp_enabled or global_query_decomp_enabled
        
        # 原检测逻辑（注释保留作为备用）
        # has_decomposition = any("decomposition" in str(type(transform)).lower() for transform in self.query_transforms)
        # needs_query_fusion = self._detect_query_fusion_retriever()
        
        if query_decomp_needed:
            logger.info(f"🚀 检测到查询分解需求（Flow配置={flow_query_decomp_enabled}，全局控制={global_query_decomp_enabled}），启动两阶段高并发流程")
            return self._two_stage_decomposition_retrieval(queries)
        else:
            logger.info(f"🔧 未启用查询分解，使用传统检索流程")
            # 传统流程：无query decomposition
            return self._traditional_batch_retrieve(queries)
    
    def _detect_query_fusion_retriever(self) -> bool:
        """检测是否使用QueryFusionRetriever"""
        try:
            if hasattr(self.retriever, '_retrievers') and self.retriever._retrievers:
                for sub_retriever in self.retriever._retrievers:
                    if "QueryFusionRetriever" in str(type(sub_retriever)):
                        return True
            return False
        except:
            return False
    
    def _get_num_queries(self) -> int:
        """获取查询分解的数量参数"""
        try:
            # 🔥 修复：优先从Flow配置中获取查询分解数量
            if hasattr(self.flow, 'params') and self.flow.params:
                num_queries = self.flow.params.get('rag_query_decomposition_num_queries', 4)
                logger.debug(f"🔧 从Flow配置获取查询分解数量: {num_queries}")
                return num_queries
            
            # # 原来的组件检测逻辑（保留作为备用）
            # if hasattr(self.retriever, '_retrievers') and self.retriever._retrievers:
            #     for sub_retriever in self.retriever._retrievers:
            #         if "QueryFusionRetriever" in str(type(sub_retriever)):
            #             if hasattr(sub_retriever, '_num_queries'):
            #                 logger.debug(f"🔧 从QueryFusionRetriever获取查询分解数量: {sub_retriever._num_queries}")
            #                 return sub_retriever._num_queries
            
            logger.debug("🔧 使用默认查询分解数量: 4")
            return 4  # 默认值
        except:
            logger.debug("🔧 获取查询分解数量失败，使用默认值: 4")
            return 4
    
    def _two_stage_decomposition_retrieval(self, queries: List[str]) -> List[List[NodeWithScore]]:
        """两阶段高并发查询分解检索"""
        # 🚀 阶段1：批量查询分解 (使用BatchLLMCaller默认并发数)
        logger.info(f"🔄 阶段1：批量分解{len(queries)}个查询...")
        from hammer.utils.batch_api_evaluator import get_batch_llm_caller
        
        # 🔥 修复：从Flow配置中获取正确的查询分解模型名称
        decomp_model_name = "Qwen2-7b"  # 默认值
        if hasattr(self.flow, 'params') and self.flow.params:
            decomp_model_name = self.flow.params.get('rag_query_decomposition_llm_name', 
                                                   self.flow.params.get('query_decomposition_llm', 'Qwen2-7b'))
        
        batch_caller = get_batch_llm_caller(model_name=decomp_model_name, max_workers=self.max_workers)
        num_queries = self._get_num_queries()
        
        decomp_results = batch_caller.batch_query_decomposition(queries, num_queries=num_queries)
        logger.info(f"✅ [Fusion阶段1] 查询分解完成: 原始{len(queries)}个查询 → 期望{num_queries}个子查询/原始查询")
        
        # 🚀 阶段2：构建扁平化检索映射
        flatten_queries = []
        query_mapping = []  # [(原始索引, 子查询索引), ...]
        
        for orig_idx, sub_queries in enumerate(decomp_results):
            effective_sub_queries = sub_queries if sub_queries else [queries[orig_idx]]
            for sub_idx, sub_query in enumerate(effective_sub_queries):
                flatten_queries.append(sub_query)
                query_mapping.append((orig_idx, sub_idx))
        
        logger.info(f"🔄 [Fusion阶段2] 查询映射构建完成: {len(queries)}个原始查询 → {len(flatten_queries)}个子查询")
        logger.info(f"📊 [Fusion统计] 平均分解倍数: {len(flatten_queries)/len(queries):.1f}x")
        
        logger.info(f"🔄 阶段2：高并发检索{len(flatten_queries)}个子查询...")
        
        # 🚀 阶段3：高并发批量检索 (32并发)
        MAX_RETRIEVAL_WORKERS = 32  # 大幅提升并发数
        
        all_results = [[] for _ in flatten_queries]
        with ThreadPoolExecutor(max_workers=MAX_RETRIEVAL_WORKERS) as executor:
            future_to_index = {
                executor.submit(self._direct_retrieve_without_decomposition, QueryBundle(query)): i
                for i, query in enumerate(flatten_queries)
            }
            
            with tqdm(total=len(flatten_queries), desc="高并发子查询检索") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        all_results[index] = future.result()
                    except Exception as e:
                        logger.warning(f"检索失败 query_idx={index}: {e}")
                        all_results[index] = []
                    finally:
                        pbar.update(1)
        
        # 🚀 阶段4：重组和融合结果
        logger.info(f"🔄 [Fusion阶段3] 开始结果重组和融合...")
        return self._regroup_and_fuse_results(all_results, query_mapping, len(queries))
    
    def _regroup_and_fuse_results(self, all_results: List[List[NodeWithScore]], 
                                 query_mapping: List[tuple], 
                                 num_original_queries: int) -> List[List[NodeWithScore]]:
        """重组：从扁平化结果重构到分组结果并融合"""
        logger.info(f"🔗 [Fusion详细] 开始重组: {len(all_results)}个子查询结果 → {num_original_queries}个原始查询组")
        
        # 按原始查询分组
        grouped_results = [[] for _ in range(num_original_queries)]
        
        for result_idx, (orig_idx, sub_idx) in enumerate(query_mapping):
            grouped_results[orig_idx].append(all_results[result_idx])
        
        logger.info(f"🔀 [Fusion详细] 重组完成，开始融合算法处理...")
        
        # 融合：每组内的多个检索结果融合为1个
        final_results = []
        total_nodes_before = 0
        total_nodes_after = 0
        
        for group_idx, group_results in enumerate(grouped_results):
            nodes_before = sum(len(nodes) for nodes in group_results)
            fused = self._fuse_subquery_results(group_results)
            nodes_after = len(fused)
            
            total_nodes_before += nodes_before
            total_nodes_after += nodes_after
            final_results.append(fused)
        
        logger.info(f"✅ [Fusion完成] 融合统计: {total_nodes_before}个节点 → {total_nodes_after}个节点 (压缩率: {total_nodes_after/total_nodes_before*100:.1f}%)")
        
        return final_results
    
    def _fuse_subquery_results(self, sub_results: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """高效融合算法：去重 + 相对分数融合 + Top-K选择"""
        if not sub_results:
            return []
        
        # 统计融合前信息
        total_input_nodes = sum(len(nodes) for nodes in sub_results)
        logger.debug(f"🔀 [融合算法] 输入: {len(sub_results)}个子查询, 共{total_input_nodes}个节点")
        
        # 使用字典实现O(1)去重
        node_score_map = {}
        
        for sub_idx, nodes in enumerate(sub_results):
            # 子查询权重：后面的子查询权重略低
            query_weight = 1.0 - (sub_idx * 0.1)
            
            for node in nodes:
                node_id = getattr(node.node, 'node_id', hash(node.node.get_content()[:100]))
                
                # 相对分数融合：取最高分数
                weighted_score = node.score * query_weight
                
                if node_id not in node_score_map:
                    node_score_map[node_id] = (node, weighted_score)
                else:
                    existing_node, existing_score = node_score_map[node_id]
                    if weighted_score > existing_score:
                        node_score_map[node_id] = (node, weighted_score)
        
        logger.debug(f"🔀 [融合算法] 去重后: {len(node_score_map)}个唯一节点 (去重率: {(1-len(node_score_map)/total_input_nodes)*100:.1f}%)")
        
        # 按分数排序并返回Top-K
        sorted_nodes = sorted(
            [(node, score) for node, score in node_score_map.values()], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_k = getattr(self.retriever, '_similarity_top_k', 20)
        return [node for node, _ in sorted_nodes[:top_k]]
    
    def _traditional_batch_retrieve(self, queries: List[str]) -> List[List[NodeWithScore]]:
        """
        🔑 修复后的传统批量检索：严格遵循TPE两阶段架构
        阶段1：批量查询分解（如需要）
        阶段2：扁平化高并发检索
        """
        logger.info(f"🚀 开始传统并发检索 {len(queries)} 个查询...")
        
        # 🔥 修复：统一使用Flow配置检测逻辑，与第一阶段保持一致
        flow_query_decomp_enabled = False
        if hasattr(self.flow, 'params') and self.flow.params:
            flow_query_decomp_enabled = self.flow.params.get('rag_query_decomposition_enabled', False)
            logger.info(f"🔧 检索阶段Flow配置检查: 查询分解={flow_query_decomp_enabled}")
        
        needs_decomposition = flow_query_decomp_enabled
        
        if needs_decomposition:
            logger.info(f"🔧 检测到查询分解需求（Flow配置），使用两阶段架构")
            return self._two_stage_decomposition_retrieval(queries)
        else:
            # 🚀 无查询分解：直接并发检索
            logger.info(f"🔧 无查询分解需求，直接并发检索")
            return self._simple_batch_retrieve(queries)
            
    def _simple_batch_retrieve(self, queries: List[str]) -> List[List[NodeWithScore]]:
        """简单批量检索（无查询分解）- 🔑 优化：检索和rerank分离并行"""
        MAX_RETRIEVAL_WORKERS = 16  # 传统模式使用适中并发数
        
        # 🚀 阶段1：纯检索并行（不包含后处理）
        retrieved_nodes_list = [[] for _ in queries]  # 预分配结果列表以保证顺序

        with ThreadPoolExecutor(max_workers=MAX_RETRIEVAL_WORKERS) as executor:
            # 创建future到索引的映射，以便将结果放回正确的位置
            future_to_index = {
                # 提交任务：对每个查询调用直接检索
                executor.submit(self._direct_retrieve_without_decomposition, QueryBundle(query)): i 
                for i, query in enumerate(queries)
            }
            
            # 使用tqdm显示进度条
            with tqdm(total=len(queries), desc="传统并发检索") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        nodes = future.result()
                        # 🔑 关键优化：只收集检索结果，延迟后处理
                        retrieved_nodes_list[index] = nodes
                    except Exception as e:
                        logger.error(f"❌ 传统检索失败 (查询索引 {index}): {e}")
                        # 即使失败，也保持列表结构
                        retrieved_nodes_list[index] = []
                    finally:
                        pbar.update(1)
        
        # ✅ 返回纯检索结果，后处理将在主流程中统一进行
        logger.info(f"✅ 传统检索完成，返回{len([r for r in retrieved_nodes_list if r])}个结果")
        return retrieved_nodes_list
    
    def _single_retrieve(self, query: str) -> List[NodeWithScore]:
        """单个查询的检索（带重试机制以应对API限流）"""
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                query_bundle = QueryBundle(query_str=query)
                return self.retriever.retrieve(query_bundle)
            except Exception as e:
                error_msg = str(e).lower()
                # 检查是否是API限流相关错误
                is_api_error = any(keyword in error_msg for keyword in [
                    '500', 'timeout', 'maximum retries', 'rate limit', 'server error'
                ])
                
                if is_api_error and retry < max_retries - 1:
                    logger.warning(f"⚠️ API检索失败 (query: {query[:50]}..., retry {retry+1}): {e}")
                    logger.info(f"🔄 等待{retry_delay}s后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.warning(f"⚠️ 检索失败: {e}")
                    return []
        
        return []
    
    def _single_retrieve_serialized(self, query: str) -> List[NodeWithScore]:
        """
        串行化的单个查询检索，使用BatchLLMCaller统一管理API调用
        🔑 核心修复：绕过QueryFusionRetriever的直接API调用，通过BatchLLMCaller处理查询分解
        """
        max_retries = 3
        base_delay = 1.0
        
        for retry in range(max_retries):
            try:
                # 🔑 关键修复：检查是否需要查询分解，使用BatchLLMCaller处理
                decomposed_queries = self._handle_query_decomposition_with_batch_caller(query)
                
                # 🔑 关键修复：使用分解后的查询进行检索，绕过QueryFusionRetriever的直接API调用
                all_retrieved_nodes = []
                for sub_query in decomposed_queries:
                    query_bundle = QueryBundle(query_str=sub_query)
                    # 直接调用底层检索器，避免QueryFusionRetriever的查询分解逻辑
                    sub_nodes = self._direct_retrieve_without_decomposition(query_bundle)
                    all_retrieved_nodes.extend(sub_nodes)
                
                # 融合和去重检索结果
                retrieved_nodes = self._fuse_retrieval_results(all_retrieved_nodes, decomposed_queries)
                
                # 应用后处理器
                processed_nodes = self._apply_postprocessors_serialized(retrieved_nodes, query)
                
                return processed_nodes
                
            except Exception as e:
                error_msg = str(e).lower()
                is_api_error = any(keyword in error_msg for keyword in [
                    '500', '502', '503', '504', 'timeout', 'maximum retries', 
                    'rate limit', 'server error', 'overloaded', 'internal server error',
                    'connection error', 'read timeout', 'connection pool'
                ])
                
                if is_api_error and retry < max_retries - 1:
                    delay = base_delay * (1.5 ** retry)
                    logger.warning(f"⚠️ 串行检索失败 (query: {query[:50]}..., retry {retry+1}/{max_retries}): {str(e)[:100]}...")
                    logger.info(f"🔄 等待{delay:.1f}s后重试...")
                    time.sleep(delay)
                else:
                    logger.warning(f"⚠️ 串行检索彻底失败: {str(e)[:100]}...")
                    return []
        
        return []
    
    def _handle_query_decomposition_with_batch_caller(self, query: str) -> List[str]:
        """
        🔑 基于配置的查询分解处理，使用BatchLLMCaller实现
        """
        logger.info(f"🔧 开始处理查询分解，原查询: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # 检查配置是否启用查询分解
        if not hasattr(self.flow, 'params') or not self.flow.params:
            logger.info("🔧 未找到Flow配置，跳过查询分解")
            return [query]
        
        query_decomp_enabled = self.flow.params.get('rag_query_decomposition_enabled', False)
        logger.info(f"🔧 查询分解配置状态: {query_decomp_enabled}")
        
        if not query_decomp_enabled:
            logger.info("🔧 配置中未启用查询分解，使用原查询")
            return [query]
        
        # 从配置获取查询分解参数
        num_queries = self.flow.params.get('rag_query_decomposition_num_queries', 4)
        decomp_llm = self.flow.params.get('rag_query_decomposition_llm_name', 'Qwen2_5-7b')
        
        logger.info(f"🔧 开始查询分解: 目标子查询数量={num_queries}, 使用模型={decomp_llm}")
        
        try:
            # 使用BatchLLMCaller进行查询分解
            from hammer.utils.batch_api_evaluator import get_batch_llm_caller
            batch_caller = get_batch_llm_caller(max_workers=self.max_workers)
            
            # 批量查询分解（单个查询的批量处理）
            decomp_results = batch_caller.batch_query_decomposition([query], num_queries=num_queries)
            
            if decomp_results and decomp_results[0] and len(decomp_results[0]) > 0:
                decomposed_queries = decomp_results[0]
                logger.info(f"✅ 查询分解成功完成!")
                logger.info(f"   原查询: {query}")
                logger.info(f"   分解为{len(decomposed_queries)}个子查询:")
                for i, sub_query in enumerate(decomposed_queries, 1):
                    logger.info(f"     子查询{i}: {sub_query}")
                logger.info(f"📊 查询分解统计: 请求{num_queries}个子查询，实际生成{len(decomposed_queries)}个")
                return decomposed_queries
            else:
                logger.warning("⚠️ 查询分解返回空结果，使用原查询")
                logger.warning(f"   分解结果: {decomp_results}")
                return [query]
                
        except Exception as e:
            logger.error(f"❌ 查询分解过程发生异常: {e}")
            logger.info("🔄 异常处理: 回退使用原查询")
            return [query]
    
    def _direct_retrieve_without_decomposition(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        🔑 核心修复：直接调用底层检索器，绕过QueryFusionRetriever的查询分解逻辑
        """
        try:
            # 如果是复合检索器，找到非QueryFusionRetriever的检索器
            if hasattr(self.retriever, '_retrievers') and self.retriever._retrievers:
                for sub_retriever in self.retriever._retrievers:
                    # 跳过QueryFusionRetriever，使用其他检索器
                    if "QueryFusionRetriever" not in str(type(sub_retriever)):
                        logger.debug(f"🔧 使用底层检索器: {type(sub_retriever).__name__}")
                        return sub_retriever.retrieve(query_bundle)
                
                # 如果只有QueryFusionRetriever，则直接调用但设置为不分解模式
                fusion_retriever = self.retriever._retrievers[0]
                if hasattr(fusion_retriever, '_retrievers') and fusion_retriever._retrievers:
                    # 直接使用QueryFusionRetriever的第一个子检索器
                    base_retriever = fusion_retriever._retrievers[0]
                    logger.debug(f"🔧 使用QueryFusion的底层检索器: {type(base_retriever).__name__}")
                    return base_retriever.retrieve(query_bundle)
                    
            # 回退到原始检索器
            logger.debug(f"🔧 回退到原始检索器: {type(self.retriever).__name__}")
            return self.retriever.retrieve(query_bundle)
            
        except Exception as e:
            logger.warning(f"⚠️ 底层检索失败: {e}")
            return []
    
    def _fuse_retrieval_results(self, all_nodes: List[NodeWithScore], queries: List[str]) -> List[NodeWithScore]:
        """
        🔑 融合多个查询的检索结果，实现简化版的相对分数融合
        """
        if not all_nodes:
            return []
        
        # 简化版融合：去重并按分数排序
        try:
            # 按节点ID去重
            seen_node_ids = set()
            unique_nodes = []
            
            for node in all_nodes:
                node_id = node.node.node_id if hasattr(node.node, 'node_id') else str(hash(node.node.get_content()[:100]))
                if node_id not in seen_node_ids:
                    seen_node_ids.add(node_id)
                    unique_nodes.append(node)
            
            # 按分数排序（分数越高越好）
            unique_nodes.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
            
            # 限制返回数量（基于retriever的top_k设置）
            top_k = 20  # 默认值，可以从retriever配置中获取
            if hasattr(self.retriever, '_similarity_top_k'):
                top_k = self.retriever._similarity_top_k
            elif hasattr(self.retriever, '_top_k'):
                top_k = self.retriever._top_k
            
            result = unique_nodes[:top_k]
            logger.debug(f"🔧 结果融合完成: {len(all_nodes)}个原始结果 -> {len(unique_nodes)}个去重结果 -> {len(result)}个最终结果")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 结果融合失败，返回原始结果: {e}")
            return all_nodes[:20]  # 简单截断
    
    def _apply_postprocessors_serialized(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """
        🔑 关键修改：使用BatchLLMCaller进行后处理器处理
        特别优化LLMRerank的批量处理，避免LLM实例共享冲突
        """
        processed_nodes = nodes
        
        # 分离LLM和非LLM后处理器
        llm_postprocessors = []
        non_llm_postprocessors = []
        
        for postprocessor in self.node_postprocessors:
            if "LLMRerank" in str(type(postprocessor)):
                llm_postprocessors.append(postprocessor)
            else:
                non_llm_postprocessors.append(postprocessor)
        
        # 🔑 关键优化：使用BatchLLMCaller处理LLMRerank
        if llm_postprocessors:
            try:
                # 动态导入BatchLLMCaller（避免循环导入）
                from hammer.utils.batch_api_evaluator import get_batch_llm_caller
                batch_caller = get_batch_llm_caller(max_workers=self.max_workers)
                
                for llm_postprocessor in llm_postprocessors:
                    # 提取文档文本
                    documents = [node.node.get_content() for node in processed_nodes]
                    
                    if documents:
                        # 获取top_k配置
                        top_k = getattr(llm_postprocessor, 'top_n', 5)
                        
                        # 批量重排序（单个查询的批量处理）
                        logger.debug(f"🔄 执行批量重排序: {len(documents)}个文档，top_k={top_k}")
                        rankings = batch_caller.batch_rerank([(query, documents)], top_k=top_k)
                        
                        if rankings and rankings[0]:
                            # 应用新的排序
                            ranking = rankings[0]
                            reranked_nodes = [processed_nodes[i] for i in ranking if i < len(processed_nodes)]
                            processed_nodes = reranked_nodes[:top_k]  # 限制到top_k
                            logger.debug(f"✅ 重排序完成: 保留{len(processed_nodes)}个文档")
                        else:
                            logger.warning(f"⚠️ 批量重排序失败，保持原顺序")
                            processed_nodes = processed_nodes[:getattr(llm_postprocessor, 'top_n', len(processed_nodes))]
                    
            except Exception as e:
                logger.warning(f"⚠️ 批量LLM后处理失败，保持原顺序: {e}")
        
        # 处理非LLM后处理器（保持原有逻辑）
        for postprocessor in non_llm_postprocessors:
            max_retries = 3
            base_delay = 0.5
            success = False
            
            for retry in range(max_retries):
                try:
                    query_bundle = QueryBundle(query_str=query)
                    processed_nodes = postprocessor.postprocess_nodes(
                        processed_nodes, query_bundle=query_bundle
                    )
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    is_api_error = any(keyword in error_msg for keyword in [
                        '500', '502', '503', '504', 'timeout', 'maximum retries',
                        'rate limit', 'server error', 'overloaded', 'internal server error',
                        'connection error', 'read timeout'
                    ])
                    
                    if is_api_error and retry < max_retries - 1:
                        delay = base_delay * (2.0 ** retry)
                        logger.warning(f"⚠️ 非LLM后处理失败 {type(postprocessor).__name__} "
                                     f"(retry {retry+1}/{max_retries}): {str(e)[:80]}...")
                        time.sleep(delay)
                    else:
                        logger.warning(f"⚠️ 非LLM后处理彻底失败 {type(postprocessor).__name__}: {str(e)[:80]}...")
                        break
            
            if not success:
                logger.warning(f"⚠️ 跳过失败的后处理器 {type(postprocessor).__name__}")
                continue
                
        return processed_nodes
    
    def _batch_build_context_strings(self, processed_nodes_list: List[List[NodeWithScore]]) -> List[str]:
        """批量构建上下文字符串"""
        context_strs = []
        
        for nodes in processed_nodes_list:
            context_parts = []
            for i, node in enumerate(nodes):
                try:
                    text = node.node.get_content()
                    if text.strip():
                        context_parts.append(f"Context {i+1}:\n{text.strip()}")
                except Exception as e:
                    logger.warning(f"⚠️ 处理节点{i}失败: {e}")
                    continue
            
            context_str = "\n\n".join(context_parts)
            context_strs.append(context_str)
        
        return context_strs
    
    def _batch_build_final_prompts(self, queries: List[str], context_strs: List[str]) -> List[str]:
        """批量构建最终prompts"""
        final_prompts = []
        
        for query, context_str in zip(queries, context_strs):
            try:
                # 使用prompt模板构建最终prompt
                if hasattr(self.prompt_template, 'format'):
                    try:
                        final_prompt = self.prompt_template.format(
                            context_str=context_str,
                            query_str=query
                        )
                    except (KeyError, TypeError):
                        try:
                            final_prompt = self.prompt_template.format(
                                context=context_str,
                                query=query
                            )
                        except (KeyError, TypeError):
                            # 备用方案：简单字符串替换
                            template_str = str(self.prompt_template)
                            final_prompt = template_str.replace("{context_str}", context_str)
                            final_prompt = final_prompt.replace("{query_str}", query)
                else:
                    # 最简单的备用方案
                    final_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                
                final_prompts.append(final_prompt)
            except Exception as e:
                logger.warning(f"⚠️ 构建prompt失败: {e}")
                final_prompts.append(f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:")
        
        return final_prompts
    
    def batch_build_prompts(
        self, 
        queries: List[str],
        is_coreset: bool = False,
        coreset_size: int = 100,
        train_data_size: int = 0
    ) -> BatchRAGPromptResult:
        """
        批量构建RAG Prompts - 核心优化入口
        新版本：对全量数据进行一次性处理，并在训练集部分选择Coreset
        """
        start_time = time.time()
        logger.info(f"🏗️ 开始批量RAG构建: {len(queries)}个查询。Coreset启用: {is_coreset}, 训练集大小: {train_data_size}")

        query_embeddings_np = None
        coreset_result_obj = None
        coreset_train_indices = list(range(min(len(queries), train_data_size)))  # 🔧 修复索引越界：使用实际查询数量和训练集大小的最小值
        embedding_time = 0.0

        if self.embedding_model:
            logger.info("✅ 检测到Embedding模型，执行Embedding优化路径。")
            
            # 🚀 阶段1: 对所有queries进行一次性批量Embedding
            embedding_start = time.time()
            query_embeddings_list = self._batch_embed_queries(queries)
            embedding_time = time.time() - embedding_start
            query_embeddings_np = np.array(query_embeddings_list)
            logger.info(f"⚡ 批量Embedding完成: 耗时{embedding_time:.3f}s. Shape: {query_embeddings_np.shape}")

            # 🎯 阶段2: 仅在训练集部分选择Coreset
            indices_to_process = list(range(len(queries))) # 默认处理所有
            
            if is_coreset and train_data_size > 0 and query_embeddings_np.shape[0] >= train_data_size:
                logger.info(f"🎯 开始在 {train_data_size} 条训练数据中选择 {coreset_size} 条Coreset...")
                train_embeddings_np = query_embeddings_np[:train_data_size]
                
                try:
                    coreset_result_obj = select_coreset_from_embeddings(
                        embeddings=train_embeddings_np,
                        coreset_size=min(coreset_size, train_data_size)
                    )
                    logger.info(f"🎯 Coreset选择完成，选出 {len(coreset_result_obj.coreset_indices)} 条样本。")
                    
                    # --- NEW: 创建需要处理的索引列表 (Coreset索引 + 测试集索引) ---
                    coreset_train_indices = coreset_result_obj.coreset_indices
                    test_indices = list(range(train_data_size, len(queries)))
                    indices_to_process = sorted(coreset_train_indices + test_indices)
                    logger.info(f"  - 将处理 {len(indices_to_process)}/{len(queries)} 个样本。")

                except Exception as e:
                    logger.error(f"💥 Coreset选择失败: {e}。回退到使用完整训练集。", exc_info=True)
                    # 失败时，indices_to_process 保持为全部索引
            work_queries = [queries[i] for i in indices_to_process]
            work_query_embeddings_list = [query_embeddings_list[i] for i in indices_to_process]

            # 🔍 阶段3: 仅在子集上进行一次性批量检索
            retrieval_start = time.time()
            work_retrieved_nodes_list = self._batch_retrieve_with_embeddings(work_queries, work_query_embeddings_list)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"🔍 批量检索完成 (在子集上): 耗时{retrieval_time:.3f}s")
            
            # --- NEW: 重构完整尺寸的结果列表 ---
            retrieved_nodes_list = [[] for _ in queries]
            for i, original_idx in enumerate(indices_to_process):
                retrieved_nodes_list[original_idx] = work_retrieved_nodes_list[i]

        else:
            logger.info("🔧 检测到sparse模式，开始执行coreset，跳过检索部分的embedding启用")
            embedding_start = time.time()
            query_embeddings_list = self._batch_embed_queries(queries)
            embedding_time = time.time() - embedding_start
            query_embeddings_np = np.array(query_embeddings_list)
            logger.info(f"⚡ 批量Embedding完成: 耗时{embedding_time:.3f}s. Shape: {query_embeddings_np.shape}")

            indices_to_process = list(range(len(queries))) # 默认处理所有
            
            if is_coreset and train_data_size > 0 and query_embeddings_np.shape[0] >= train_data_size:
                logger.info(f"🎯 开始在 {train_data_size} 条训练数据中选择 {coreset_size} 条Coreset...")
                train_embeddings_np = query_embeddings_np[:train_data_size]
                
                try:
                    coreset_result_obj = select_coreset_from_embeddings(
                        embeddings=train_embeddings_np,
                        coreset_size=min(coreset_size, train_data_size)
                    )
                    logger.info(f"🎯 Coreset选择完成，选出 {len(coreset_result_obj.coreset_indices)} 条样本。")
                    
                    coreset_train_indices = coreset_result_obj.coreset_indices
                    test_indices = list(range(train_data_size, len(queries)))
                    indices_to_process = sorted(coreset_train_indices + test_indices)
                    logger.info(f"  - 将处理 {len(indices_to_process)}/{len(queries)} 个样本。")

                except Exception as e:
                    logger.error(f"💥 Coreset选择失败: {e}。回退到使用完整训练集。", exc_info=True)
            work_queries = [queries[i] for i in indices_to_process]
            work_query_embeddings_list = [query_embeddings_list[i] for i in indices_to_process]

            retrieval_start = time.time()
            work_retrieved_nodes_list = self._traditional_batch_retrieve(work_queries) # Sparse模式仍然处理全部
            retrieval_time = time.time() - retrieval_start
            logger.info(f"🔍 批量检索完成: 耗时{retrieval_time:.3f}s")
            retrieved_nodes_list = [[] for _ in queries]
            for i, original_idx in enumerate(indices_to_process):
                retrieved_nodes_list[original_idx] = work_retrieved_nodes_list[i]

        # 🔄 阶段4: 批处理后处理（真正的批处理reranking）
        rerank_phase_start = time.time()
        logger.info("🚀 开始阶段4: 批处理后处理（真正的批处理reranking）")
        
        # 统计检索结果
        total_retrieved = sum(len(nodes) for nodes in retrieved_nodes_list)
        avg_retrieved = total_retrieved / len(queries) if queries else 0
        logger.info(f"📊 检索结果统计: 总文档数={total_retrieved}, 平均每查询={avg_retrieved:.1f}个文档")
        
        # 4.1 批处理reranking（如果启用）
        if self.parallel_reranker:
            logger.info(f"🎯 启动批处理reranking: {self.parallel_reranker.reranker_name}...")
            rerank_start = time.time()
            processed_nodes_list = self.parallel_reranker.batch_rerank_all_queries(retrieved_nodes_list, queries)
            rerank_time = time.time() - rerank_start
            
            # 统计rerank结果
            total_reranked = sum(len(nodes) for nodes in processed_nodes_list)
            logger.info(f"✅ 批处理reranking完成: 耗时{rerank_time:.2f}s")
            logger.info(f"📊 Rerank结果: 输出文档数={total_reranked}, 平均每查询={total_reranked/len(queries):.1f}个")
        else:
            # 如果没有reranker，直接返回原始结果（不截断）
            logger.info("⚪ 未启用reranker，使用原始检索结果")
            processed_nodes_list = retrieved_nodes_list
        
        rerank_phase_time = time.time() - rerank_phase_start
        logger.info(f"✅ 阶段4完成: 总耗时{rerank_phase_time:.2f}s")
        
        # 4.2 构建最终结果
        context_strs = self._batch_build_context_strings(processed_nodes_list)
        final_prompts = self._batch_build_final_prompts(queries, context_strs)
        
        processing_time = time.time() - start_time
        # 成功计数现在基于最终生成的prompts，跳过的样本prompt为空
        success_count = sum(1 for prompt in final_prompts if prompt.strip())
        
        logger.info(f"🎉 批量RAG构建完成: 总耗时={processing_time:.3f}s, 实际处理={success_count}/{len(queries)}")
        
        return BatchRAGPromptResult(
            final_prompts=final_prompts,
            queries=queries,
            retrieved_nodes_list=retrieved_nodes_list,
            processed_nodes_list=processed_nodes_list,
            context_strs=context_strs,
            processing_time=processing_time,
            embedding_time=embedding_time,
            retrieval_time=retrieval_time,
            success_count=success_count,
            failed_indices=[i for i, p in enumerate(final_prompts) if not p.strip()],
            error_messages=[],
            coreset_result=coreset_result_obj,
            coreset_train_indices=coreset_train_indices,
            query_embeddings=query_embeddings_np
        )

class BatchReranker:
    """批处理Reranker - 真正的批处理优化，解决GPU竞争问题"""
    
    def __init__(self, reranker_name: str, top_k: int):
        self.reranker_name = reranker_name
        self.top_k = top_k
        
        logger.info(f"🚀 初始化批处理Reranker: {reranker_name}, top_k={top_k}")
        
        # 直接创建批处理reranker实例
        try:
            from hammer.rerankers.integrations import BatchSentenceTransformerReranker
            self.batch_reranker = BatchSentenceTransformerReranker(
                model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                top_n=top_k,
                batch_size=32
            )
            logger.info("✅ 批处理reranker初始化成功")
        except Exception as e:
            logger.error(f"❌ 批处理reranker初始化失败: {e}")
            raise
    
    def batch_rerank_all_queries(self, all_retrieved_nodes: List[List[NodeWithScore]], 
                                queries: List[str]) -> List[List[NodeWithScore]]:
        """真正的批处理reranking - 一次性处理所有query-document对"""
        
        import time
        start_time = time.time()
        
        logger.info(f"🚀 BatchReranker开始处理: {len(queries)}个查询")
        logger.info(f"📊 Reranker配置: 模型={self.reranker_name}, top_k={self.top_k}")
        
        # 统计输入数据
        total_nodes = sum(len(nodes) for nodes in all_retrieved_nodes)
        avg_nodes = total_nodes / len(queries) if queries else 0
        logger.info(f"📊 输入统计: 总节点数={total_nodes}, 平均每查询={avg_nodes:.1f}个节点")
        
        try:
            # 直接调用批处理reranker的批量处理方法
            results = self.batch_reranker.batch_rerank_all_queries(queries, all_retrieved_nodes)
            
            total_time = time.time() - start_time
            
            # 统计输出结果
            output_nodes = sum(len(nodes) for nodes in results)
            successful_queries = len([r for r in results if r])
            
            logger.info(f"✅ BatchReranker处理完成")
            logger.info(f"📊 输出统计: 成功查询数={successful_queries}/{len(queries)}, 输出节点数={output_nodes}")
            logger.info(f"📊 处理效率: 总耗时={total_time:.2f}s, 处理速度={len(queries)/total_time:.1f} queries/s")
            
            return results
        except Exception as e:
            logger.error(f"❌ BatchReranker处理失败: {e}")
            logger.error(f"🔍 失败上下文: queries数量={len(queries)}, nodes数量={len(all_retrieved_nodes)}")
            raise  # 不提供fallback，直接报错

def create_optimized_rag_prompt_builder(flow: Flow, max_workers: int = None) -> OptimizedRAGPromptBuilder:
    """
    创建优化版RAG Prompt构建器的便捷函数
    
    Args:
        flow: Flow对象
        max_workers: 最大并发数，用于LLM API调用
        
    Returns:
        OptimizedRAGPromptBuilder实例
    """
    return OptimizedRAGPromptBuilder(flow, max_workers)

# 使用示例
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass