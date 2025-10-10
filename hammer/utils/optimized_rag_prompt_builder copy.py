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
    
    def __init__(self, flow: Flow):
        """
        初始化优化版RAG Prompt构建器
        
        Args:
            flow: Flow对象，包含RAG配置
        """
        self.flow = flow
        self.query_engine = flow.query_engine
        
        # 提取query_engine的组件
        self._extract_components()
        
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
            logger.info(f"🎉 成功找到Embedding模型: {type(model).__name__}")
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
                    device='cuda',              # 自动选择设备
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
            
            # 检查需要应用的转换类型
            has_hyde = any("hyde" in str(type(transform)).lower() for transform in self.query_transforms)
            has_decomposition = any("decomposition" in str(type(transform)).lower() for transform in self.query_transforms)
            
            if has_hyde or has_decomposition:
                # 动态导入BatchLLMCaller（避免循环导入）
                from hammer.utils.batch_api_evaluator import get_batch_llm_caller
                batch_caller = get_batch_llm_caller()
                
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
                # 使用批量接口
                logger.info("🚀 使用批量embedding接口 (get_text_embeddings)")
                try:
                    embeddings = self.embedding_model.get_text_embeddings(transformed_queries)
                    logger.info(f"✅ 批量embedding调用成功，返回{len(embeddings)}个结果")
                except Exception as e:
                    logger.error(f"❌ 批量embedding调用失败: {e}")
                    logger.info("🔄 回退到单个embedding方法...")
                    raise e
                    
            elif has_single_method:
                # 回退到单个embedding，但添加进度显示
                logger.info("🔄 回退到单个embedding接口，逐个处理...")
                embeddings = []
                failed_embeddings = 0
                
                for i, query in enumerate(transformed_queries):
                    if i == 0:
                        logger.info(f"⚡ 开始处理第一个embedding: {query[:50]}...")
                    elif i > 0 and i % 100 == 0:  # 进度显示
                        elapsed = time.time() - embedding_start
                        eta = elapsed / i * (total_queries - i) if i > 0 else 0
                        speed = i / elapsed if elapsed > 0 else 0
                        logger.info(f"⚡ Embedding进度: {i}/{total_queries} ({i/total_queries*100:.1f}%) "
                                   f"- 速度{speed:.1f} queries/s, 已用时{elapsed:.1f}s, 预计剩余{eta:.1f}s")
                    
                    try:
                        embedding = self.embedding_model.get_text_embedding(query)
                        embeddings.append(embedding)
                        
                        if i == 0:
                            logger.info(f"✅ 第一个embedding成功，维度: {len(embedding) if embedding else 'None'}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ 单个embedding失败 (query {i+1}): {e}")
                        embeddings.append([])  # 空embedding作为占位符
                        failed_embeddings += 1
                
                if failed_embeddings > 0:
                    logger.warning(f"⚠️ 共有{failed_embeddings}个embedding计算失败")
                    
            else:
                # 最后的回退方案
                logger.info("🔄 使用encode方法回退方案...")
                embeddings = []
                for i, query in enumerate(transformed_queries):
                    if i > 0 and i % 50 == 0:
                        logger.info(f"🔄 Encode进度: {i}/{len(transformed_queries)}")
                    try:
                        if has_encode_method:
                            embeddings.append(self.embedding_model.encode(query))
                        else:
                            logger.error(f"❌ Embedding模型没有可用的方法!")
                            embeddings.append([])
                    except Exception as e:
                        logger.warning(f"⚠️ encode方法失败 (query {i+1}): {e}")
                        embeddings.append([])
            
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
    
    def _batch_retrieve_with_embeddings(self, queries: List[str], query_embeddings: List[List[float]]) -> List[List[NodeWithScore]]:
        """
        [新优化] 两阶段高并发查询分解检索
        阶段1：批量分解 (用默认并发数)  
        阶段2：扁平化并发检索 (32并发)
        """
        # 检测是否启用query decomposition
        has_decomposition = any("decomposition" in str(type(transform)).lower() for transform in self.query_transforms)
        needs_query_fusion = self._detect_query_fusion_retriever()
        
        if has_decomposition or needs_query_fusion:
            logger.info(f"🚀 检测到查询分解需求，启动两阶段高并发流程")
            return self._two_stage_decomposition_retrieval(queries)
        else:
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
            # 从QueryFusionRetriever获取num_queries参数
            if hasattr(self.retriever, '_retrievers') and self.retriever._retrievers:
                for sub_retriever in self.retriever._retrievers:
                    if "QueryFusionRetriever" in str(type(sub_retriever)):
                        if hasattr(sub_retriever, '_num_queries'):
                            return sub_retriever._num_queries
            return 4  # 默认值
        except:
            return 4
    
    def _two_stage_decomposition_retrieval(self, queries: List[str]) -> List[List[NodeWithScore]]:
        """两阶段高并发查询分解检索"""
        # 🚀 阶段1：批量查询分解 (使用BatchLLMCaller默认并发数)
        logger.info(f"🔄 阶段1：批量分解{len(queries)}个查询...")
        from hammer.utils.batch_api_evaluator import get_batch_llm_caller
        batch_caller = get_batch_llm_caller()
        num_queries = self._get_num_queries()
        
        decomp_results = batch_caller.batch_query_decomposition(queries, num_queries=num_queries)
        
        # 🚀 阶段2：构建扁平化检索映射
        flatten_queries = []
        query_mapping = []  # [(原始索引, 子查询索引), ...]
        
        for orig_idx, sub_queries in enumerate(decomp_results):
            effective_sub_queries = sub_queries if sub_queries else [queries[orig_idx]]
            for sub_idx, sub_query in enumerate(effective_sub_queries):
                flatten_queries.append(sub_query)
                query_mapping.append((orig_idx, sub_idx))
        
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
        return self._regroup_and_fuse_results(all_results, query_mapping, len(queries))
    
    def _regroup_and_fuse_results(self, all_results: List[List[NodeWithScore]], 
                                 query_mapping: List[tuple], 
                                 num_original_queries: int) -> List[List[NodeWithScore]]:
        """重组：从扁平化结果重构到分组结果并融合"""
        # 按原始查询分组
        grouped_results = [[] for _ in range(num_original_queries)]
        
        for result_idx, (orig_idx, sub_idx) in enumerate(query_mapping):
            grouped_results[orig_idx].append(all_results[result_idx])
        
        # 融合：每组内的多个检索结果融合为1个
        final_results = []
        for group_results in grouped_results:
            fused = self._fuse_subquery_results(group_results)
            final_results.append(fused)
        
        return final_results
    
    def _fuse_subquery_results(self, sub_results: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """高效融合算法：去重 + 相对分数融合 + Top-K选择"""
        if not sub_results:
            return []
        
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
        
        # 🔍 检测是否需要查询分解
        needs_decomposition = self._detect_query_fusion_retriever()
        
        if needs_decomposition:
            logger.info(f"🔧 检测到查询分解需求，使用TPE两阶段架构")
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
        
        # 🚀 阶段2：批量并发后处理（包含rerank）
        logger.info(f"🔧 开始批量并发后处理（含Rerank）...")
        return self._batch_apply_postprocessors(retrieved_nodes_list, queries)
    
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
        🔑 核心修复：使用BatchLLMCaller处理查询分解，避免QueryFusionRetriever的直接API调用
        """
        try:
            # 检查retriever是否需要查询分解
            if hasattr(self.retriever, '_retrievers') and self.retriever._retrievers:
                # 这是一个复合检索器，检查是否包含QueryFusionRetriever
                for sub_retriever in self.retriever._retrievers:
                    if "QueryFusionRetriever" in str(type(sub_retriever)):
                        # 发现QueryFusionRetriever，需要手动处理查询分解
                        if hasattr(sub_retriever, '_num_queries'):
                            num_queries = sub_retriever._num_queries
                        else:
                            num_queries = 4  # 默认值
                        
                        logger.info(f"🔧 检测到QueryFusionRetriever，使用BatchLLMCaller进行查询分解，子查询数量: {num_queries}")
                        
                        # 动态导入BatchLLMCaller
                        from hammer.utils.batch_api_evaluator import get_batch_llm_caller
                        batch_caller = get_batch_llm_caller()
                        
                        # 批量查询分解（单个查询的批量处理）
                        decomp_results = batch_caller.batch_query_decomposition([query], num_queries=num_queries)
                        
                        if decomp_results and decomp_results[0]:
                            decomposed_queries = decomp_results[0]
                            logger.info(f"✅ 查询分解成功: {len(decomposed_queries)}个子查询")
                            return decomposed_queries
                        else:
                            logger.warning("⚠️ 查询分解失败，使用原查询")
                            return [query]
                            
            # 无需查询分解，直接返回原查询
            return [query]
            
        except Exception as e:
            logger.warning(f"⚠️ 查询分解处理失败，回退到原查询: {e}")
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
                batch_caller = get_batch_llm_caller()
                
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
    
    def _single_apply_postprocessors(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """辅助函数：为单个查询应用所有后处理器"""
        processed_nodes = nodes
        for postprocessor in self.node_postprocessors:
            try:
                query_bundle = QueryBundle(query_str=query)
                processed_nodes = postprocessor.postprocess_nodes(
                    processed_nodes, query_bundle=query_bundle
                )
            except Exception as e:
                logger.warning(f"⚠️ 后处理器 {type(postprocessor).__name__} 失败 (query: {query[:50]}...): {e}")
                continue # 即使一个后处理器失败，也继续处理下一个
        return processed_nodes

    def _batch_apply_postprocessors(self, retrieved_nodes_list: List[List[NodeWithScore]], queries: List[str]) -> List[List[NodeWithScore]]:
        """[已优化] 并发应用后处理器（如Reranker）"""
        if not self.node_postprocessors:
            logger.info("ℹ️ 无后处理器，跳过后处理阶段。")
            return retrieved_nodes_list

        logger.info(f"🚀 开始并发应用 {len(self.node_postprocessors)} 个后处理器到 {len(queries)} 个结果集...")
        MAX_POSTPROCESS_WORKERS = 16  # 🔑 关键优化：启用Rerank并行处理，大幅提升效率
        
        processed_nodes_list = [[] for _ in queries]

        with ThreadPoolExecutor(max_workers=MAX_POSTPROCESS_WORKERS) as executor:
            future_to_index = {
                executor.submit(self._single_apply_postprocessors, nodes, query): i
                for i, (nodes, query) in enumerate(zip(retrieved_nodes_list, queries))
            }

            with tqdm(total=len(queries), desc="并发后处理") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        processed_nodes = future.result()
                        processed_nodes_list[index] = processed_nodes
                    except Exception as e:
                        logger.error(f"❌ 并发后处理失败 (查询索引 {index}): {e}")
                        processed_nodes_list[index] = retrieved_nodes_list[index] # 失败则返回原始节点
                    finally:
                        pbar.update(1)

        return processed_nodes_list
    
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
        coreset_train_indices = list(range(len(queries[train_data_size])))  # 初始化为全列表
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

        # 🔄 阶段4: 后续处理（在重构后的完整尺寸列表上进行）
        logger.info("ℹ️ 后处理已集成到检索阶段。")
        processed_nodes_list = retrieved_nodes_list
        
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

def create_optimized_rag_prompt_builder(flow: Flow) -> OptimizedRAGPromptBuilder:
    """
    创建优化版RAG Prompt构建器的便捷函数
    
    Args:
        flow: Flow对象
        
    Returns:
        OptimizedRAGPromptBuilder实例
    """
    return OptimizedRAGPromptBuilder(flow)

# 使用示例
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass