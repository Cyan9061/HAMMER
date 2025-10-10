"""
批量API调用和评估器 - 高并发API调用和评估
集成hammer/utils/utils_getAPI.py的批量API调用能力
"""

import time
import typing as T
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from hammer.utils.utils_getAPI import batch_predict, get_api_keys, get_api_keys_for_model, MODEL_SPECIFIC_API_KEYS
# from hammer.utils.rag_prompt_builder import RAGPromptResult
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from hammer.utils.optimized_rag_prompt_builder import BatchRAGPromptResult
from hammer.logger import logger

@dataclass
class RAGPromptResult:
    """RAG Prompt构建结果"""
    final_prompt: str
    retrieved_nodes: List[NodeWithScore]
    processed_nodes: List[NodeWithScore]
    query: str
    context_str: str
    processing_time: float
    success: bool
    error_message: str = ""

@dataclass
class QAEvaluationItem:
    """单个QA评估项"""
    question: str
    answer: str
    qa_pair: Any  # 原始qa_pair对象
    index: int
    prompt: str = ""
    response: str = ""
    api_latency: float = 0.0
    tokens: int = 0
    success: bool = False
    error_message: str = ""
    contexts: List[str] = field(default_factory=list)  # 🔥 添加contexts字段

@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    total_count: int
    success_count: int
    failed_count: int
    api_call_time: float
    evaluation_time: float
    total_time: float
    
    # 评估指标
    answer_ems: List[float]
    answer_f1s: List[float]
    joint_ems: List[float]
    joint_f1s: List[float]
    run_times: List[float]
    
    # 🔥 新增统一评估指标
    lexical_acs: List[float] = field(default_factory=list)
    lexical_ffs: List[float] = field(default_factory=list)
    mrrs: List[float] = field(default_factory=list)
    rouge_ls: List[float] = field(default_factory=list)  # 🔥 新增ROUGE-L指标
    
    # 详细结果
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # API统计
    api_success_count: int = 0
    api_failed_count: int = 0
    avg_api_latency: float = 0.0
    total_tokens: int = 0
    
    # QA执行日志 (用于MCTS知识库)
    qa_execution_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # 🔥 新增便捷方法
    def get_optimization_score(self) -> float:
        """获取优化目标分数 - 统一使用F1"""
        return sum(self.answer_f1s) / len(self.answer_f1s) if self.answer_f1s else 0.0
    
    def get_metric_summary(self) -> Dict[str, float]:
        """获取所有指标的汇总"""
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0
            
        return {
            'f1': safe_mean(self.answer_f1s),
            'em': safe_mean(self.answer_ems), 
            'lexical_ac': safe_mean(self.lexical_acs),
            'lexical_ff': safe_mean(self.lexical_ffs),
            'mrr': safe_mean(self.mrrs),
            'optimization_target': self.get_optimization_score()
        }
    
    # def __post_init__(self):
    #     """初始化后处理，确保qa_execution_logs不为None"""
    #     if self.qa_execution_logs is None:
    #         self.qa_execution_logs = []

class BatchAPIEvaluator:
    """
    批量API调用和评估器
    结合RAG Prompt构建器和批量API调用，实现高效的批量评估
    """
    
    # 🔧 hammer内部名称到API名称的映射 - 修复模型名称映射问题
    MODEL_API_MAPPING = {
        # hammer内部名称 -> API调用名称
        "Qwen2_5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2-7b": "Qwen/Qwen2-7B-Instruct",  # 🔥 修复主要问题
        "DeepSeek-R1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
        "Qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "gpt-4o-mini": "gpt-4o-mini",  # gaochao API支持
    }
    
    def __init__(self, 
                 model_name: str,
                 max_workers: Optional[int] = None,
                 multihop_evaluator=None):
        """
        初始化批量API评估器
        
        Args:
            model_name: hammer内部模型名称，会自动映射为API名称
            max_workers: 最大并发数，默认为API key数量×3
            multihop_evaluator: 多跳评估器实例
        """
        # 🔧 修复模型名称映射问题：将hammer内部名称映射为API名称
        if model_name in self.MODEL_API_MAPPING:
            self.model_name = self.MODEL_API_MAPPING[model_name]
            self.hammer_name = model_name
            logger.info(f"🔧 BatchAPIEvaluator模型映射: hammer名称='{model_name}' -> API名称='{self.model_name}'")
        else:
            # 向后兼容：如果传入的已经是API格式，直接使用
            self.model_name = model_name
            self.hammer_name = model_name
            logger.warning(f"⚠️ BatchAPIEvaluator使用未映射的模型名称: '{model_name}'，请检查是否需要添加映射")
        self.api_keys = get_api_keys_for_model(self.model_name)
        # 🔑 修复None比较错误：当max_workers为None时，使用默认值
        default_workers = len(self.api_keys) * 1
        self.max_workers = min(max_workers, default_workers) if max_workers is not None else default_workers
        self.multihop_evaluator = multihop_evaluator
        
        self.current_flow: Optional[Any] = None
        self.current_config: Optional[Dict[str, Any]] = None

        # 显示API key使用信息
        is_model_specific = self.model_name in MODEL_SPECIFIC_API_KEYS
        key_type = "模型特定" if is_model_specific else "默认"
        logger.info(f"🚀 初始化批量API评估器: 模型={model_name}, "
                    f"API密钥数={len(self.api_keys)}({key_type}), 并发度={self.max_workers}")
        if is_model_specific:
            logger.info(f"🔑 模型 {self.model_name} 使用专用API keys")
    
    def _prepare_evaluation_items_from_batch_result(self, 
                                                  batch_rag_result: BatchRAGPromptResult, 
                                                  qa_pairs: List[Any]) -> List[QAEvaluationItem]:
        """从BatchRAGPromptResult准备评估项目（优化版批量RAG结果）"""
        items = []
        
        for i, (final_prompt, qa_pair) in enumerate(zip(batch_rag_result.final_prompts, qa_pairs)):
            # 检查是否在失败列表中
            is_failed = i in batch_rag_result.failed_indices
            
            item = QAEvaluationItem(
                question=qa_pair.question,
                answer=qa_pair.answer,
                qa_pair=qa_pair,
                index=i + 1,
                prompt=final_prompt,
                success=not is_failed,
                contexts=[batch_rag_result.context_strs[i]] if i < len(batch_rag_result.context_strs) else []  # 🔥 添加contexts
            )
            
            if is_failed:
                error_idx = batch_rag_result.failed_indices.index(i) if i in batch_rag_result.failed_indices else 0
                error_msg = batch_rag_result.error_messages[error_idx] if error_idx < len(batch_rag_result.error_messages) else "RAG构建失败"
                item.error_message = error_msg
            
            items.append(item)
        
        return items
    
    def _prepare_evaluation_items(self, 
                                  rag_results: List[RAGPromptResult], 
                                  qa_pairs: List[Any]) -> List[QAEvaluationItem]:
        """准备评估项目"""
        items = []
        
        for i, (rag_result, qa_pair) in enumerate(zip(rag_results, qa_pairs)):
            item = QAEvaluationItem(
                question=qa_pair.question,
                answer=qa_pair.answer,
                qa_pair=qa_pair,
                index=i + 1,
                prompt=rag_result.final_prompt,
                success=rag_result.success,
                contexts=[rag_result.context_str] if hasattr(rag_result, 'context_str') and rag_result.context_str else []  # 🔥 添加contexts
            )
            
            if not rag_result.success:
                item.error_message = f"RAG构建失败: {rag_result.error_message}"
            
            items.append(item)
        
        return items
    
    def _batch_api_call(self, evaluation_items: List[QAEvaluationItem]) -> List[QAEvaluationItem]:
        """批量API调用"""
        logger.info(f"🚀 开始批量API调用: {len(evaluation_items)}个请求, 并发度={self.max_workers}")
        
        # 提取所有prompts
        prompts = [item.prompt for item in evaluation_items]
        
        # 记录API调用开始时间
        api_start_time = time.time()
        
        try:
            # 使用batch_predict进行批量API调用
            api_results = batch_predict(
                input_texts=prompts,
                model_name=self.model_name,
                max_workers=self.max_workers
            )
            
            api_call_time = time.time() - api_start_time
            logger.info(f"✅ 批量API调用完成: 耗时{api_call_time:.2f}s, 获得{len(api_results)}个响应")
            
            # 处理API结果
            api_success_count = 0
            api_failed_count = 0
            total_tokens = 0
            total_latency = 0
            
            for i, (api_result, item) in enumerate(zip(api_results, evaluation_items)):
                try:
                    input_text, response_text, latency, tokens = api_result
                    
                    if response_text is not None and latency is not None:
                        # API调用成功
                        item.response = response_text
                        item.api_latency = latency
                        item.tokens = tokens or 0
                        item.success = item.success and True  # 保持之前的状态
                        api_success_count += 1
                        total_tokens += item.tokens
                        total_latency += latency
                    else:
                        # API调用失败
                        item.response = ""
                        item.success = False
                        item.error_message = "API调用失败"
                        api_failed_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ 处理API结果失败 (项目{i+1}): {e}")
                    item.response = ""
                    item.success = False
                    item.error_message = f"处理API结果失败: {e}"
                    api_failed_count += 1
            
            # 记录API统计
            avg_latency = total_latency / api_success_count if api_success_count > 0 else 0
            
            # 🔥 关键修复：将RAG tokens记录到简单token追踪系统
            if total_tokens > 0:
                from hammer.utils.simple_token_tracker import record_token_usage
                # RAG批量调用token记录
                record_token_usage(
                    llm_name=f"rag_batch_{self.model_name}",
                    total_tokens=total_tokens
                )
                logger.info(f"🔢 [RAG_BATCH] 记录RAG批量token: {total_tokens} tokens")
            
            logger.info(f"📊 API调用统计: 成功={api_success_count}, 失败={api_failed_count}, "
                        f"平均延迟={avg_latency:.0f}ms, 总token={total_tokens}")
            
            return evaluation_items
            
        except Exception as e:
            logger.error(f"❌ 批量API调用失败: {e}")
            # 标记所有项目为失败
            for item in evaluation_items:
                item.success = False
                item.error_message = f"批量API调用失败: {e}"
            return evaluation_items
    
    # ▼▼▼【修复 1/3】修改方法签名，增加 batch_rag_result 参数，并设为可选以兼容旧代码 ▼▼▼
    def _batch_evaluate(self, evaluation_items: List[QAEvaluationItem], batch_rag_result: Optional[BatchRAGPromptResult] = None) -> BatchEvaluationResult:
        """批量评估"""
        logger.info(f"📊 开始批量评估: {len(evaluation_items)}个响应")
        
        eval_start_time = time.time()
        
        # 初始化结果
        answer_ems = []
        answer_f1s = []
        joint_ems = []
        joint_f1s = []
        # 🔥 新增统一评估指标初始化
        lexical_acs = []
        lexical_ffs = []
        mrrs = []
        rouge_ls = []  # 🔥 新增ROUGE-L列表初始化
        run_times = []
        detailed_results = []
        
        success_count = 0
        failed_count = 0
        
        for item in evaluation_items:
            eval_item_start = time.time()
            
            if not item.success or not item.response:
                # 失败项目，使用默认分数
                result = {
                    'answer_em': 0.0,
                    'answer_f1': 0.0,
                    'joint_em': 0.0,
                    'joint_f1': 0.0,
                    # 🔥 新增统一评估指标默认值
                    'lexical_ac': 0.0,
                    'lexical_ff': 0.0,
                    'mrr': 0.0,
                    'rouge_l': 0.0,  # 🔥 新增ROUGE-L默认值
                    'run_time': 0.0,
                    'index': item.index,
                    'success': False,
                    'error': item.error_message,
                    'api_latency': item.api_latency,
                    'tokens': item.tokens,
                    'question': item.question,
                    'answer': item.answer
                }
                failed_count += 1
            else:
                # 成功项目，执行评估
                try:
                    if self.multihop_evaluator:
                        multihop_result = self.multihop_evaluator._evaluate(
                            query=item.question,
                            response=item.response,
                            reference=item.answer,
                            qa_pair=item.qa_pair,
                            contexts=item.contexts  # 🔥 添加contexts参数
                        )
                        
                        eval_item_time = time.time() - eval_item_start
                        
                        result = {
                            'answer_em': multihop_result.answer_em,
                            'answer_f1': multihop_result.answer_f1,
                            'joint_em': multihop_result.joint_em,
                            'joint_f1': multihop_result.joint_f1,
                            # 🔥 新增统一评估指标
                            'lexical_ac': multihop_result.lexical_ac,
                            'lexical_ff': multihop_result.lexical_ff,
                            'mrr': multihop_result.mrr,
                            'rouge_l': multihop_result.rouge_l,  # 🔥 新增ROUGE-L结果提取
                            'run_time': eval_item_time,
                            'index': item.index,
                            'success': True,
                            'api_latency': item.api_latency,
                            'tokens': item.tokens,
                            'question': item.question,
                            'answer': item.answer
                        }
                        success_count += 1
                    else:
                        # 没有评估器，使用默认分数
                        result = {
                            'answer_em': 0.0,
                            'answer_f1': 0.0,
                            'joint_em': 0.0,
                            'joint_f1': 0.0,
                            # 🔥 新增统一评估指标默认值
                            'lexical_ac': 0.0,
                            'lexical_ff': 0.0,
                            'mrr': 0.0,
                            'rouge_l': 0.0,  # 🔥 新增ROUGE-L默认值
                            'run_time': 0.0,
                            'index': item.index,
                            'success': False,
                            'error': '缺少评估器',
                            'api_latency': item.api_latency,
                            'tokens': item.tokens,
                            'question': item.question,
                            'answer': item.answer
                        }
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ 评估失败 (项目{item.index}): {e}")
                    result = {
                        'answer_em': 0.0,
                        'answer_f1': 0.0,
                        'joint_em': 0.0,
                        'joint_f1': 0.0,
                        # 🔥 新增统一评估指标默认值
                        'lexical_ac': 0.0,
                        'lexical_ff': 0.0,
                        'mrr': 0.0,
                        'rouge_l': 0.0,  # 🔥 新增ROUGE-L默认值
                        'run_time': 0.0,
                        'index': item.index,
                        'success': False,
                        'error': f'评估失败: {e}',
                        'api_latency': item.api_latency,
                        'tokens': item.tokens,
                        'question': item.question,
                        'answer': item.answer
                    }
                    failed_count += 1
            
            # 收集结果
            answer_ems.append(result['answer_em'])
            answer_f1s.append(result['answer_f1'])
            joint_ems.append(result['joint_em'])
            joint_f1s.append(result['joint_f1'])
            # 🔥 新增统一评估指标收集
            lexical_acs.append(result['lexical_ac'])
            lexical_ffs.append(result['lexical_ff'])
            mrrs.append(result['mrr'])
            rouge_ls.append(result['rouge_l'])  # 🔥 新增ROUGE-L收集
            run_times.append(result['run_time'])
            detailed_results.append(result)
        
        evaluation_time = time.time() - eval_start_time
        
        # 计算API统计
        api_success_items = [item for item in evaluation_items if item.response]
        api_success_count = len(api_success_items)
        api_failed_count = len(evaluation_items) - api_success_count
        avg_api_latency = sum(item.api_latency for item in api_success_items) / api_success_count if api_success_count > 0 else 0
        total_tokens = sum(item.tokens for item in evaluation_items)
        
        qa_execution_logs = []
        
        # ▼▼▼【修复 1/3, 续】只有在 batch_rag_result 存在时，才生成详细日志 ▼▼▼
        if batch_rag_result:
            # 从batch_rag_result中提取RAG配置信息
            rag_config = self._extract_rag_config_from_result(batch_rag_result)
            
            for i, (item, result) in enumerate(zip(evaluation_items, detailed_results)):
                # 提取详细的RAG执行信息
                qa_log = {
                    'qa_id': f'batch_qa_{i}',
                    'question': result.get('question', item.question),
                    'ground_truth': result.get('answer', item.answer),
                    'predicted_answer': item.response or '',
                    'f1_score': result.get('answer_f1', 0.0),
                    'exact_match': result.get('answer_em', 0.0) > 0,
                    'execution_time': result.get('run_time', 0.0),
                    'total_execution_time': result.get('run_time', 0.0) + item.api_latency,
                    
                    # 从真实RAG结果中提取完整流程信息
                    'retrieval_method': rag_config.get('retrieval_method', 'sparse'),
                    'embedding_model': rag_config.get('embedding_model', '/mnt/data/wangshu/llm_lm/bge-m3'),
                    'template_name': rag_config.get('template_name', 'CoT'),
                    'response_synthesizer_llm': rag_config.get('response_synthesizer_llm', 'Qwen/Qwen2-7B-Instruct'),
                    
                    # Query decomposition信息 - 修复参数名一致性
                    'query_decomposition_enabled': rag_config.get('rag_query_decomposition_enabled', rag_config.get('query_decomposition_enabled', False)),
                    'decomposed_queries': self._extract_decomposed_queries(batch_rag_result, i, rag_config),
                    'query_decomposition_llm': rag_config.get('rag_query_decomposition_llm_name', rag_config.get('query_decomposition_llm', '')),
                    'decomposition_time': 0.0,  # 暂时使用默认值
                    
                    # HyDE信息 - 从实际执行结果中提取
                    'hyde_enabled': rag_config.get('hyde_enabled', False),
                    'hyde_query': self._extract_hyde_query(batch_rag_result, i),
                    'hyde_llm': rag_config.get('hyde_llm', ''),
                    'hyde_time': 0.0,  # 暂时使用默认值
                    
                    # 检索信息
                    'retrieval_top_k': rag_config.get('rag_top_k', rag_config.get('retrieval_top_k', 10)),
                    'hybrid_bm25_weight': rag_config.get('rag_hybrid_bm25_weight', rag_config.get('hybrid_bm25_weight', 0.5)),
                    'initial_retrieved_docs': self._extract_initial_retrieved_docs(batch_rag_result, i),
                    'final_retrieved_docs': self._extract_final_retrieved_docs(batch_rag_result, i),
                    'retrieval_time': batch_rag_result.retrieval_time / len(evaluation_items) if evaluation_items else 0.0,
                    'retrieval_precision': result.get('retrieval_precision', 0.7),
                    'retrieval_recall': result.get('retrieval_recall', 0.7),
                    'context_overlap': result.get('context_overlap', 0.6),
                    'ground_truth_context': self._extract_ground_truth_context(item, i),
                    
                    # Fusion信息
                    'fusion_enabled': rag_config.get('fusion_enabled', rag_config.get('rag_query_decomposition_enabled', False)),  # 查询分解启用时通常也启用融合
                    'fusion_mode': rag_config.get('rag_fusion_mode', rag_config.get('fusion_mode', 'simple')),
                    'fused_docs': self._extract_fused_docs(batch_rag_result, i),
                    'fusion_time': 0.0,
                    
                    # Reranker信息 - 从实际执行结果中提取
                    'reranker_enabled': rag_config.get('reranker_enabled', False),
                    'reranker_llm': rag_config.get('reranker_llm', ''),
                    'reranker_top_k': rag_config.get('reranker_top_k', 5),
                    'reranker_results': self._extract_reranker_results(batch_rag_result, i),
                    'reranking_time': 0.0,
                    
                    # Additional context信息
                    'additional_context_enabled': rag_config.get('additional_context_enabled', False),
                    'additional_context_num_nodes': rag_config.get('additional_context_num_nodes', 0),
                    'additional_context_docs': self._extract_additional_context_docs(batch_rag_result, i),
                    'additional_context_time': 0.0,
                    
                    # Context assembly信息
                    'final_context': self._extract_context_str(batch_rag_result, i),
                    'context_assembly_time': batch_rag_result.processing_time / len(evaluation_items) if evaluation_items else 0.0,
                    
                    # Few-shot信息
                    'few_shot_enabled': rag_config.get('few_shot_enabled', False),
                    'few_shot_examples': [],
                    'few_shot_retrieval_time': 0.0,
                    
                    # Synthesis信息
                    'final_prompt': item.prompt or '',
                    'synthesis_time': item.api_latency,
                    'answer_relevance': result.get('answer_relevance', 0.6)
                }
                qa_execution_logs.append(qa_log)
        else:
            logger.warning("⚠️ batch_rag_result未提供，跳过详细QA日志生成。")

        logger.info(f"📊 批量评估完成: 成功={success_count}, 失败={failed_count}, 耗时={evaluation_time:.2f}s")
        logger.info(f"📝 生成了 {len(qa_execution_logs)} 条完整QA执行日志")
        
        return BatchEvaluationResult(
            total_count=len(evaluation_items),
            success_count=success_count,
            failed_count=failed_count,
            api_call_time=0,  # 这个在外层设置
            evaluation_time=evaluation_time,
            total_time=0,  # 这个在外层设置
            answer_ems=answer_ems,
            answer_f1s=answer_f1s,
            joint_ems=joint_ems,
            joint_f1s=joint_f1s,
            # 🔥 新增统一评估指标
            lexical_acs=lexical_acs,
            lexical_ffs=lexical_ffs,
            mrrs=mrrs,
            rouge_ls=rouge_ls,  # 🔥 新增ROUGE-L结果
            run_times=run_times,
            detailed_results=detailed_results,
            api_success_count=api_success_count,
            api_failed_count=api_failed_count,
            avg_api_latency=avg_api_latency,
            total_tokens=total_tokens,
            qa_execution_logs=qa_execution_logs  # 添加QA执行日志
        )
    
    def _extract_rag_config_from_result(self, batch_rag_result) -> Dict[str, Any]:
        """从BatchRAGPromptResult中提取RAG配置信息"""
        config = {}
        
        try:
            # 🔑 关键修复：优先从当前的Flow配置中提取真实参数
            if hasattr(self, 'current_config') and self.current_config:
                config.update(self.current_config)
                logger.info(f"📋 从current_config提取到RAG配置: {len(config)} 个参数")
                
                # 记录关键参数以确保正确性
                key_params = {
                    'rag_query_decomposition_enabled': config.get('rag_query_decomposition_enabled'),
                    'rag_fusion_mode': config.get('rag_fusion_mode'),
                    'hyde_enabled': config.get('hyde_enabled'),
                    'reranker_enabled': config.get('reranker_enabled')
                }
                logger.debug(f"🔑 关键Flow参数: {key_params}")
            
            # 尝试从flow对象中提取配置
            elif hasattr(self, 'current_flow') and self.current_flow:
                flow = self.current_flow
                if hasattr(flow, 'params') and flow.params:
                    config.update(flow.params)
                    logger.info(f"📋 从current_flow.params提取到RAG配置: {len(config)} 个参数")
            
            # 如果还没有配置，尝试从batch_rag_result中提取
            elif hasattr(batch_rag_result, 'flow') and batch_rag_result.flow:
                flow = batch_rag_result.flow
                if hasattr(flow, 'params') and flow.params:
                    config.update(flow.params)
                    logger.info(f"📋 从batch_rag_result.flow提取到RAG配置: {len(config)} 个参数")
            
            # 设置一些合理的默认值（如果没有找到配置）
            if not config:
                logger.warning("⚠️ 未找到RAG配置，使用默认值")
                config = {
                    'retrieval_method': 'sparse',
                    'embedding_model': '/mnt/data/wangshu/llm_lm/bge-m3',
                    'template_name': 'CoT',
                    'response_synthesizer_llm': 'Qwen/Qwen2-7B-Instruct',
                    'retrieval_top_k': 10,
                    'rag_query_decomposition_enabled': True,
                    'hyde_enabled': True,
                    'reranker_enabled': False
                }
                logger.info(f"✅ 使用默认RAG配置: {config}")
            
            # 确保关键参数存在，使用flow.params中的正确参数名
            config.setdefault('retrieval_method', 'sparse')
            config.setdefault('embedding_model', '/mnt/data/wangshu/llm_lm/bge-m3')
            config.setdefault('template_name', 'CoT')
            config.setdefault('response_synthesizer_llm', 'Qwen/Qwen2-7B-Instruct')
            
            logger.info(f"✅ 最终RAG配置包含: {list(config.keys())}")
            return config
            
        except Exception as e:
            logger.error(f"❌ 提取RAG配置失败: {e}")
            # 返回基本默认配置
            return {
                'retrieval_method': 'sparse',
                'embedding_model': '/mnt/data/wangshu/llm_lm/bge-m3',
                'template_name': 'CoT',
                'response_synthesizer_llm': 'Qwen/Qwen2-7B-Instruct',
                'retrieval_top_k': 10
            }
    
    def _extract_reranker_results(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取重排序结果"""
        try:
            # 检查是否有处理过的节点（可能经过重排序）
            if hasattr(batch_rag_result, 'processed_nodes_list') and batch_rag_result.processed_nodes_list:
                if index < len(batch_rag_result.processed_nodes_list):
                    processed_nodes = batch_rag_result.processed_nodes_list[index]
                    # 提取处理后的文档信息
                    reranker_docs = []
                    for i, node in enumerate(processed_nodes[:5]):  # 通常reranker只保留top-k个结果
                        try:
                            if hasattr(node, 'node') and hasattr(node.node, 'text'):
                                doc_info = {
                                    'doc_id': i,
                                    'text': node.node.text[:200],  # 限制长度
                                    'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                    'reranked': True,
                                    'node_type': str(type(node).__name__)
                                }
                                reranker_docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理重排序节点{i}失败: {node_error}")
                    
                    if reranker_docs:
                        logger.info(f"✅ 成功提取{len(reranker_docs)}个重排序文档 (index={index})")
                    
                    return reranker_docs
            return []
        except Exception as e:
            logger.error(f"❌ 提取重排序结果失败 (index={index}): {e}")
            return []
    
    def _extract_hyde_query(self, batch_rag_result, index: int) -> str:
        """从批量RAG结果中提取HyDE查询"""
        try:
            # 尝试从HyDE转换结果中提取
            if hasattr(batch_rag_result, 'queries') and batch_rag_result.queries:
                if index < len(batch_rag_result.queries):
                    original_query = batch_rag_result.queries[index]
                    
                    # 如果启用了HyDE，检查是否有转换过的查询
                    if hasattr(self, 'current_flow') and self.current_flow:
                        flow = self.current_flow
                        if hasattr(flow, 'query_engine') and hasattr(flow.query_engine, '_query_transform'):
                            # 可能存在HyDE转换，但由于是批量处理，我们无法单独提取HyDE结果
                            # 这里返回原始查询，表示HyDE已处理但未单独记录
                            return original_query
                    
                    return original_query
            return ''
        except Exception as e:
            logger.error(f"❌ 提取HyDE查询失败 (index={index}): {e}")
            return ''
    
    def _extract_retrieved_docs(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取检索到的文档信息 - 改进版"""
        try:
            if hasattr(batch_rag_result, 'retrieved_nodes_list') and batch_rag_result.retrieved_nodes_list:
                if index < len(batch_rag_result.retrieved_nodes_list):
                    nodes = batch_rag_result.retrieved_nodes_list[index]
                    # 提取文档的文本内容和元数据
                    docs = []
                    for i, node in enumerate(nodes[:10]):  # 限制前10个文档
                        try:
                            doc_info = {
                                'doc_id': i,
                                'text': node.node.text[:500] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:500],  # 限制长度
                                'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                'metadata': dict(node.node.metadata) if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {},
                                'node_type': str(type(node).__name__)
                            }
                            docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理检索节点{i}失败: {node_error}")
                            # 添加一个错误占位符
                            docs.append({
                                'doc_id': i,
                                'text': f'Error processing node: {str(node_error)[:100]}',
                                'score': 0.0,
                                'metadata': {},
                                'node_type': 'error'
                            })
                    
                    if docs:
                        logger.info(f"✅ 成功提取{len(docs)}个检索文档 (index={index})")
                        return docs
                    else:
                        logger.warning(f"⚠️ 检索节点列表为空 (index={index})")
                        
            logger.warning(f"⚠️ 无法访问retrieved_nodes_list (index={index})")
            return []
        except Exception as e:
            logger.error(f"❌ 提取检索文档失败 (index={index}): {e}")
            return []
    
    def _extract_context_str(self, batch_rag_result, index: int) -> str:
        """从批量RAG结果中提取上下文字符串"""
        try:
            if hasattr(batch_rag_result, 'context_strs') and batch_rag_result.context_strs:
                if index < len(batch_rag_result.context_strs):
                    return batch_rag_result.context_strs[index] or ''
            
            # 如果没有context_strs，尝试从final_prompts中提取
            if hasattr(batch_rag_result, 'final_prompts') and batch_rag_result.final_prompts:
                if index < len(batch_rag_result.final_prompts):
                    prompt = batch_rag_result.final_prompts[index]
                    # 从prompt中提取context部分（通常在Context information is below之后）
                    if 'Context information is below' in prompt:
                        context_start = prompt.find('Context information is below')
                        if context_start != -1:
                            context_part = prompt[context_start:context_start+2000]  # 限制长度
                            return context_part
                    return prompt[:1000]  # 返回prompt的前1000字符作为context
            
            return ''
        except Exception as e:
            logger.error(f"❌ 提取上下文字符串失败 (index={index}): {e}")
            return ''
    
    def evaluate_batch(self, 
                       rag_results: List[RAGPromptResult], 
                       qa_pairs: List[Any]) -> BatchEvaluationResult:
        """
        执行批量评估
        
        Args:
            rag_results: RAG prompt构建结果列表
            qa_pairs: QA数据对列表
            
        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        total_start_time = time.time()
        
        logger.info(f"🎯 开始批量评估: {len(rag_results)}个项目")
        
        # 准备评估项目
        evaluation_items = self._prepare_evaluation_items(rag_results, qa_pairs)
        
        # 批量API调用
        api_start_time = time.time()
        evaluation_items = self._batch_api_call(evaluation_items)
        api_call_time = time.time() - api_start_time
        
        # 批量评估
        # ▼▼▼【修复 2/3】在调用时传递 None，因为此路径没有 batch_rag_result ▼▼▼
        batch_result = self._batch_evaluate(evaluation_items, None)
        
        # 设置时间统计
        batch_result.api_call_time = api_call_time
        batch_result.total_time = time.time() - total_start_time
        
        logger.info(f"🎉 批量评估完成: 总耗时={batch_result.total_time:.2f}s, "
                    f"API调用={batch_result.api_call_time:.2f}s, "
                    f"评估={batch_result.evaluation_time:.2f}s")
        
        return batch_result
    
    def evaluate_batch_optimized(self, 
                                 batch_rag_result: BatchRAGPromptResult, 
                                 qa_pairs: List[Any]) -> BatchEvaluationResult:
        """
        执行优化版批量评估（使用BatchRAGPromptResult）
        核心优化：先批量embedding，再批量RAG，最后批量API调用
        
        Args:
            batch_rag_result: 优化版批量RAG构建结果
            qa_pairs: QA数据对列表
            
        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        total_start_time = time.time()
        
        logger.info(f"🎯 开始优化版批量评估: {len(batch_rag_result.final_prompts)}个项目")
        logger.info(f"⚡ RAG优化统计: embedding耗时={batch_rag_result.embedding_time:.3f}s, "
                    f"检索耗时={batch_rag_result.retrieval_time:.3f}s, "
                    f"成功数={batch_rag_result.success_count}/{len(batch_rag_result.final_prompts)}")
        
        # 准备评估项目（使用优化版批量结果）
        evaluation_items = self._prepare_evaluation_items_from_batch_result(batch_rag_result, qa_pairs)
        
        # 批量API调用
        api_start_time = time.time()
        evaluation_items = self._batch_api_call(evaluation_items)
        api_call_time = time.time() - api_start_time
        
        # 批量评估
        # ▼▼▼【修复 3/3】在调用时，将 batch_rag_result 变量传递下去 ▼▼▼
        batch_result = self._batch_evaluate(evaluation_items, batch_rag_result)
        
        # 设置时间统计（包含RAG构建时间）
        batch_result.api_call_time = api_call_time
        batch_result.total_time = time.time() - total_start_time
        
        # 记录优化版特有的统计信息
        logger.info(f"🎉 优化版批量评估完成: 总耗时={batch_result.total_time:.2f}s, "
                    f"RAG构建={batch_rag_result.processing_time:.2f}s (embedding={batch_rag_result.embedding_time:.3f}s), "
                    f"API调用={batch_result.api_call_time:.2f}s, "
                    f"评估={batch_result.evaluation_time:.2f}s")
        
        return batch_result

    def _extract_decomposed_queries(self, batch_rag_result, index: int, rag_config: Dict[str, Any]) -> List[str]:
        """从批量RAG结果中提取分解的查询"""
        try:
            # 检查是否启用了查询分解
            if not rag_config.get('rag_query_decomposition_enabled', rag_config.get('query_decomposition_enabled', False)):
                # 如果未启用查询分解，返回原始查询
                if hasattr(batch_rag_result, 'queries') and batch_rag_result.queries and index < len(batch_rag_result.queries):
                    return [batch_rag_result.queries[index]]
                return []
            
            # 尝试从分解结果中提取
            if hasattr(batch_rag_result, 'decomposed_queries') and batch_rag_result.decomposed_queries:
                if index < len(batch_rag_result.decomposed_queries):
                    decomposed = batch_rag_result.decomposed_queries[index]
                    if isinstance(decomposed, list) and decomposed:
                        return decomposed
            
            # 如果没有找到分解结果，返回原始查询
            if hasattr(batch_rag_result, 'queries') and batch_rag_result.queries and index < len(batch_rag_result.queries):
                return [batch_rag_result.queries[index]]
            
            return []
        except Exception as e:
            logger.error(f"❌ 提取分解查询失败 (index={index}): {e}")
            return []

    def _extract_fused_docs(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取融合后的文档"""
        try:
            # 检查是否有融合处理后的文档
            if hasattr(batch_rag_result, 'fused_nodes_list') and batch_rag_result.fused_nodes_list:
                if index < len(batch_rag_result.fused_nodes_list):
                    fused_nodes = batch_rag_result.fused_nodes_list[index]
                    fused_docs = []
                    for i, node in enumerate(fused_nodes[:10]):  # 限制前10个文档
                        try:
                            doc_info = {
                                'doc_id': i,
                                'text': node.node.text[:500] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:500],
                                'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                'fusion_method': 'query_fusion',
                                'node_type': str(type(node).__name__)
                            }
                            fused_docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理融合节点{i}失败: {node_error}")
                    
                    if fused_docs:
                        logger.info(f"✅ 成功提取{len(fused_docs)}个融合文档 (index={index})")
                    return fused_docs
            
            # 如果没有融合文档，尝试从处理后的节点中提取
            if hasattr(batch_rag_result, 'processed_nodes_list') and batch_rag_result.processed_nodes_list:
                if index < len(batch_rag_result.processed_nodes_list):
                    processed_nodes = batch_rag_result.processed_nodes_list[index]
                    return self._convert_nodes_to_docs(processed_nodes, 'processed_fusion')
            
            # 最后回退到检索文档
            return self._extract_retrieved_docs(batch_rag_result, index)
            
        except Exception as e:
            logger.error(f"❌ 提取融合文档失败 (index={index}): {e}")
            return []

    def _convert_nodes_to_docs(self, nodes, doc_type: str = 'unknown') -> List[Dict]:
        """将节点列表转换为文档字典列表"""
        docs = []
        for i, node in enumerate(nodes[:10]):  # 限制前10个
            try:
                doc_info = {
                    'doc_id': i,
                    'text': node.node.text[:500] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:500],
                    'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                    'doc_type': doc_type,
                    'node_type': str(type(node).__name__)
                }
                docs.append(doc_info)
            except Exception as e:
                logger.warning(f"⚠️ 转换节点{i}失败: {e}")
        return docs

    def _extract_initial_retrieved_docs(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取初始检索到的文档信息"""
        try:
            if hasattr(batch_rag_result, 'retrieved_nodes_list') and batch_rag_result.retrieved_nodes_list:
                if index < len(batch_rag_result.retrieved_nodes_list):
                    nodes = batch_rag_result.retrieved_nodes_list[index]
                    # 提取文档的文本内容和元数据
                    docs = []
                    for i, node in enumerate(nodes[:10]):  # 限制前10个文档
                        try:
                            doc_info = {
                                'doc_id': i,
                                'text': node.node.text[:500] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:500],  # 限制长度
                                'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                'metadata': dict(node.node.metadata) if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {},
                                'node_type': str(type(node).__name__),
                                'retrieval_stage': 'initial'
                            }
                            docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理初始检索节点{i}失败: {node_error}")
                            # 添加一个错误占位符
                            docs.append({
                                'doc_id': i,
                                'text': f'Error processing node: {str(node_error)[:100]}',
                                'score': 0.0,
                                'metadata': {},
                                'node_type': 'error',
                                'retrieval_stage': 'initial'
                            })
                    
                    if docs:
                        logger.info(f"✅ 成功提取{len(docs)}个初始检索文档 (index={index})")
                        return docs
                    else:
                        logger.warning(f"⚠️ 初始检索节点列表为空 (index={index})")
                        
            logger.warning(f"⚠️ 无法访问retrieved_nodes_list (index={index})")
            return []
        except Exception as e:
            logger.error(f"❌ 提取初始检索文档失败 (index={index}): {e}")
            return []
    
    def _extract_final_retrieved_docs(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取最终处理后的文档信息"""
        try:
            # 优先从处理后的节点中提取
            if hasattr(batch_rag_result, 'processed_nodes_list') and batch_rag_result.processed_nodes_list:
                if index < len(batch_rag_result.processed_nodes_list):
                    processed_nodes = batch_rag_result.processed_nodes_list[index]
                    final_docs = []
                    for i, node in enumerate(processed_nodes[:10]):
                        try:
                            doc_info = {
                                'doc_id': i,
                                'text': node.node.text[:500] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:500],
                                'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                'metadata': dict(node.node.metadata) if hasattr(node, 'node') and hasattr(node.node, 'metadata') else {},
                                'node_type': str(type(node).__name__),
                                'retrieval_stage': 'final'
                            }
                            final_docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理最终文档节点{i}失败: {node_error}")
                    
                    if final_docs:
                        logger.info(f"✅ 成功提取{len(final_docs)}个最终文档 (index={index})")
                        return final_docs
            
            # 如果没有处理后的节点，回退到初始检索文档
            return self._extract_initial_retrieved_docs(batch_rag_result, index)
            
        except Exception as e:
            logger.error(f"❌ 提取最终检索文档失败 (index={index}): {e}")
            return []
    
    def _extract_additional_context_docs(self, batch_rag_result, index: int) -> List[Dict]:
        """从批量RAG结果中提取额外上下文文档"""
        try:
            # 检查是否有额外上下文相关的节点
            if hasattr(batch_rag_result, 'additional_context_nodes') and batch_rag_result.additional_context_nodes:
                if index < len(batch_rag_result.additional_context_nodes):
                    additional_nodes = batch_rag_result.additional_context_nodes[index]
                    additional_docs = []
                    for i, node in enumerate(additional_nodes[:5]):  # 限制前5个额外文档
                        try:
                            doc_info = {
                                'doc_id': i,
                                'text': node.node.text[:300] if hasattr(node, 'node') and hasattr(node.node, 'text') else str(node)[:300],
                                'score': float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                                'context_type': 'additional',
                                'node_type': str(type(node).__name__)
                            }
                            additional_docs.append(doc_info)
                        except Exception as node_error:
                            logger.warning(f"⚠️ 处理额外上下文节点{i}失败: {node_error}")
                    
                    if additional_docs:
                        logger.info(f"✅ 成功提取{len(additional_docs)}个额外上下文文档 (index={index})")
                    return additional_docs
            
            return []
            
        except Exception as e:
            logger.error(f"❌ 提取额外上下文文档失败 (index={index}): {e}")
            return []
    
    def _extract_ground_truth_context(self, item, index: int) -> str:
        """从QA对中提取真实上下文信息"""
        try:
            # 检查qa_pair是否有上下文信息
            if hasattr(item, 'qa_pair') and item.qa_pair:
                qa_pair = item.qa_pair
                
                # 检查常见的上下文字段
                context_fields = ['context', 'supporting_facts', 'evidence', 'passages', 'documents']
                for field in context_fields:
                    if hasattr(qa_pair, field):
                        context_value = getattr(qa_pair, field)
                        if context_value:
                            if isinstance(context_value, list):
                                # 如果是列表，连接成字符串
                                return ' '.join(str(c) for c in context_value)[:1000]  # 限制长度
                            else:
                                return str(context_value)[:1000]  # 限制长度
                
                # 如果没有找到上下文字段，检查其他可能的字段
                if hasattr(qa_pair, '__dict__'):
                    for key, value in qa_pair.__dict__.items():
                        if 'context' in key.lower() or 'passage' in key.lower():
                            if value:
                                return str(value)[:1000]
            
            return ''
            
        except Exception as e:
            logger.error(f"❌ 提取真实上下文失败 (index={index}): {e}")
            return ''

def create_batch_api_evaluator(model_name: str,
                                 max_workers: Optional[int] = None,
                                 multihop_evaluator=None) -> BatchAPIEvaluator:
    """
    创建批量API评估器的便捷函数
    
    Args:
        model_name: API模型名称
        max_workers: 最大并发数
        multihop_evaluator: 多跳评估器实例
        
    Returns:
        BatchAPIEvaluator实例
    """
    return BatchAPIEvaluator(
        model_name=model_name,
        max_workers=max_workers,
        multihop_evaluator=multihop_evaluator
    )

"""
批量LLM调用器 - 统一的批量LLM调用接口，替代SiliconFlow LLM实例
"""

from typing import List, Dict, Any, Tuple
import re
from hammer.utils.utils_getAPI import batch_predict
from hammer.logger import logger

class BatchLLMCaller:
    """统一的批量LLM调用接口，替代SiliconFlow LLM实例"""
    
    # 🔧 hammer内部名称到API名称的映射
    MODEL_API_MAPPING = {
        # hammer内部名称 -> API调用名称
        "Qwen2_5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2-7b": "Qwen/Qwen2-7B-Instruct",      # 🔥 新增Qwen2-7b支持
        "DeepSeek-R1-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
        "Qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "gpt-4o-mini": "gpt-4o-mini",  # gaochao API支持
    }
    
    def __init__(self, model_name: str = "Qwen2-7b", max_workers: Optional[int] = None):
        # 🔧 统一使用hammer内部名称格式，自动转换为API格式
        if model_name in self.MODEL_API_MAPPING:
            self.model_name = self.MODEL_API_MAPPING[model_name]
            self.hammer_name = model_name
        else:
            # 向后兼容：如果传入的是旧格式，先转换
            legacy_mapping = {v: k for k, v in self.MODEL_API_MAPPING.items()}
            if model_name in legacy_mapping:
                self.hammer_name = legacy_mapping[model_name]
                self.model_name = model_name
            else:
                self.hammer_name = model_name
                self.model_name = model_name
        
        # 🔧 设置max_workers，如果未指定则使用默认值
        from hammer.utils.utils_getAPI import get_api_keys_for_provider, get_provider_for_model
        provider = get_provider_for_model(self.model_name)
        api_keys = get_api_keys_for_provider(provider)
        self.max_workers = max_workers if max_workers is not None else 3#len(api_keys)
        
        logger.info(f"🔧 BatchLLMCaller初始化: hammer名称='{self.hammer_name}' -> API名称='{self.model_name}', max_workers={self.max_workers}")
    
    def batch_hyde_transform(self, queries: List[str]) -> List[str]:
        """批量HyDE查询转换"""
        prompts = [f"""Please write a passage to answer the question: {query}
Try to include as much detail as possible, write the passage as if you are an expert on the topic.
Passage:""" for query in queries]
        
        try:
            results = batch_predict(prompts, self.model_name, max_workers=self.max_workers)
            return [result[1] if result[1] else f"[HyDE失败] {query}" 
                   for result, query in zip(results, queries)]
        except Exception as e:
            logger.error(f"❌ 批量HyDE转换失败: {e}")
            return [f"[HyDE异常] {query}" for query in queries]
    
    def batch_query_decomposition(self, queries: List[str], num_queries: int = 4) -> List[List[str]]:
        """改进的批量查询分解"""
        logger.info(f"🔧 BatchLLMCaller开始批量查询分解: {len(queries)}个查询，目标{num_queries}个子查询/每个")
        
        prompts = []
        for i, query in enumerate(queries):
            logger.debug(f"   准备查询{i+1}: {query[:80]}{'...' if len(query) > 80 else ''}")
            prompt = f"""Please decompose the following complex question into {num_queries} simpler, more specific sub-questions.

Original question: {query}

Requirements:
1. Each sub-question should be independent and answerable
2. Sub-questions should cover different aspects of the original question
3. Avoid redundant or overlapping sub-questions
4. Use clear and concise language

Please output in the following format:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question]
4. [Fourth sub-question]"""
            prompts.append(prompt)
        
        try:
            logger.info(f"🚀 调用API进行批量查询分解，最大并发数: {self.max_workers}")
            results = batch_predict(prompts, self.model_name, max_workers=self.max_workers)
            
            decomposed_queries = []
            successful_decompositions = 0
            total_sub_queries = 0
            
            for i, (result, original_query) in enumerate(zip(results, queries)):
                if result[1] and len(result[1].strip()) > 10:
                    logger.debug(f"   处理查询{i+1}的分解结果: {len(result[1])}字符")
                    # 改进的解析逻辑
                    sub_queries = self._parse_numbered_list(result[1])
                    if len(sub_queries) > 0:
                        # 确保不超过请求的数量
                        final_sub_queries = sub_queries[:num_queries]
                        decomposed_queries.append(final_sub_queries)
                        successful_decompositions += 1
                        total_sub_queries += len(final_sub_queries)
                        logger.debug(f"     成功解析出{len(final_sub_queries)}个子查询")
                    else:
                        # 如果解析失败，使用原查询
                        decomposed_queries.append([original_query])
                        logger.warning(f"     查询{i+1}解析失败，使用原查询")
                else:
                    # 如果结果为空，使用原查询
                    decomposed_queries.append([original_query])
                    logger.warning(f"     查询{i+1}返回空结果，使用原查询")
            
            # 输出统计信息
            logger.info(f"✅ 批量查询分解完成统计:")
            logger.info(f"   处理查询数: {len(queries)}")
            logger.info(f"   成功分解: {successful_decompositions}")
            logger.info(f"   失败/回退: {len(queries) - successful_decompositions}")
            logger.info(f"   总子查询数: {total_sub_queries}")
            logger.info(f"   平均子查询数: {total_sub_queries / successful_decompositions if successful_decompositions > 0 else 0:.1f}")
            
            return decomposed_queries
        except Exception as e:
            logger.error(f"❌ 批量查询分解失败: {e}")
            logger.info(f"🔄 异常处理: 所有查询回退到原查询")
            # 出错时返回原查询
            return [[query] for query in queries]
    
    def batch_rerank(self, query_doc_pairs: List[Tuple[str, List[str]]], top_k: int = 5) -> List[List[int]]:
        """批量文档重排序"""
        prompts = []
        for query, documents in query_doc_pairs:
            doc_list = "\n".join([f"Document {i+1}: {doc[:200]}..." 
                                 for i, doc in enumerate(documents[:top_k])])
            prompt = f"""Given a query and documents, rank them by relevance (numbers only):
Query: {query}
Documents:
{doc_list}
Ranking:"""
            prompts.append(prompt)
        
        try:
            results = batch_predict(prompts, self.model_name, max_workers=self.max_workers)
            
            rankings = []
            for result, (query, docs) in zip(results, query_doc_pairs):
                if result[1]:
                    ranking = self._parse_ranking(result[1], len(docs))
                    rankings.append(ranking)
                else:
                    rankings.append(list(range(len(docs))))  # 默认顺序
            
            return rankings
        except Exception as e:
            logger.error(f"❌ 批量重排序失败: {e}")
            return [list(range(len(docs))) for _, docs in query_doc_pairs]
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """改进的编号列表解析方法"""
        lines = text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 匹配多种编号格式: "1.", "1)", "(1)", "1 -", "1:"
            patterns = [
                r'^\d+\.\s*(.+)',      # 1. question
                r'^\d+\)\s*(.+)',      # 1) question  
                r'^\(\d+\)\s*(.+)',    # (1) question
                r'^\d+\s*-\s*(.+)',    # 1 - question
                r'^\d+:\s*(.+)',       # 1: question
                r'^\[\w+.*?\]\s*(.+)', # [First sub-question] question
            ]
            
            extracted = None
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    break
            
            # 如果没有匹配到编号格式，但是行内容看起来像问题，也包含进来
            if not extracted and len(line) > 10 and ('?' in line or line.endswith('.') or 'what' in line.lower() or 'how' in line.lower() or 'why' in line.lower() or 'when' in line.lower() or 'where' in line.lower()):
                extracted = line
            
            if extracted and len(extracted) > 5:  # 基本质量检查
                sub_queries.append(extracted)
        
        return sub_queries
    
    def _parse_ranking(self, text: str, doc_count: int) -> List[int]:
        """解析排序结果"""
        numbers = re.findall(r'\d+', text)
        try:
            ranking = [int(n)-1 for n in numbers if 1 <= int(n) <= doc_count]  # 转为0索引
            if not ranking:
                return list(range(doc_count))
            return ranking[:doc_count]
        except:
            return list(range(doc_count))

# 全局实例管理
_batch_llm_caller = None

def get_batch_llm_caller(model_name: str = "Qwen2-7b", max_workers: Optional[int] = None) -> BatchLLMCaller:
    """获取BatchLLMCaller实例（单例模式）"""
    global _batch_llm_caller
    # 🔧 重要修改：每次都重新创建实例以确保max_workers参数生效
    _batch_llm_caller = BatchLLMCaller(model_name, max_workers)
    return _batch_llm_caller

