"""
#hammer/mcts/optimization_engine.py
真正的MCTS优化引擎实现

主要修改：
1. 去掉GPT模拟评估，改为真实评估回调
2. 简化MCTS调用逻辑
3. 保持现有的图记忆系统
4. 删除不必要的复杂代码
"""

import time
import typing as T
from abc import ABC, abstractmethod
from pathlib import Path

import optuna
from hammer.logger import logger
import os
import json
import hashlib

# 导入增强的三层图记忆系统
from .kb_manager import (
    GraphMemoryRAGMCTS,
    QAExecutionNode,
    InsightAgent,
)

class OptimizationEngine(ABC):
    """优化引擎的抽象基类"""
    
    @abstractmethod
    def suggest_parameters(self, trial: optuna.Trial, components: T.List[str]) -> T.Dict[str, T.Any]:
        """建议参数配置"""
        pass

class EnhancedMCTSOptimizationEngine(OptimizationEngine):
    """Enhanced MCTS optimization engine with true MCTS implementation"""
    
    def __init__(self, api_key: str, api_base: str, experiment_id: str = None, iteration: int = 50,
                existing_knowledge_base: T.Dict[str, T.Any] = None):
        self.api_key = api_key
        self.api_base = api_base
        self.experiment_id = experiment_id or self._generate_experiment_id()
        
        # 初始化三层图记忆系统
        graph_memory_path = f"Experiment/graph_memory/{self.experiment_id}"
        self.graph_memory = GraphMemoryRAGMCTS(storage_path=graph_memory_path)
        
        # 迁移旧知识库（如果提供）
        if existing_knowledge_base:
            self._migrate_old_knowledge_base(existing_knowledge_base)
        
        # 初始化洞察智能体
        self.insight_agent = InsightAgent(
            api_key=self.api_key,
            api_base=self.api_base
        )
        
        # MCTS配置参数
        self.mcts_iterations = 1000
        self.current_search_idx = 0
        self.max_searches = 1000
        
        # 真实评估回调 - 由外部设置
        self.evaluation_callback = None
        
        logger.info(f"🚀 Enhanced MCTS Optimization Engine initialized")
        logger.info(f"🎯 MCTS iterations per search: {self.mcts_iterations}")
        logger.info(f"🧠 Graph memory: {self.graph_memory.get_memory_stats()}")

    def set_evaluation_callback(self, callback: T.Callable[[T.Dict[str, T.Any]], float]):
        """设置真实评估回调函数"""
        self.evaluation_callback = callback
        logger.info("✅ Real evaluation callback set")

    def suggest_parameters(self) -> T.Dict[str, T.Any]:
        """使用真正的MCTS建议参数配置"""
        if self.evaluation_callback is None:
            logger.error("❌ Evaluation callback not set! Cannot perform MCTS search.")
            return self._get_default_config()
        
        if self.current_search_idx >= self.max_searches:
            logger.warning(f"已完成所有{self.max_searches}次搜索，返回默认配置")
            return self._get_default_config()
        
        logger.info(f"🔄 开始第{self.current_search_idx+1}/{self.max_searches}次真正的MCTS参数搜索")
        
        # 执行真正的MCTS搜索
        best_config = self._execute_true_mcts_search()
        
        # 应用参数增强（保持与现有系统的兼容性）
        enhanced_params = self._ensure_required_params(best_config)
        
        # 更新状态
        self.current_search_idx += 1
        
        return enhanced_params
    
    def _execute_true_mcts_search(self) -> T.Dict[str, T.Any]:
        """执行真正的MCTS搜索"""
        from .hierarchical_search import TrueMCTS, RAGSearchSpace
        
        # 创建搜索空间
        search_space = self._build_search_space_from_config()
        
        # Debug: 记录当前内存状态
        memory_stats = self.graph_memory.get_memory_stats()
        logger.info(f"🧠 Pre-search memory state: {memory_stats['config_layer']['configurations']} configs, "
                f"{memory_stats['insight_layer']['insights']} insights")
        
        # 创建真正的MCTS实例，传入知识库和洞察智能体
        mcts_instance = TrueMCTS(
            search_space=search_space,
            evaluation_callback=self.evaluation_callback,
            exploration_constant=1.414,
            max_iterations=self.mcts_iterations,
            graph_memory=self.graph_memory,  # 🔥 传入知识库
            insight_agent=self.insight_agent  # 🔥 传入洞察智能体
        )
        
        # 执行搜索
        best_config = mcts_instance.search()
        
        logger.info(f"✅ GPT-guided MCTS搜索完成")
        return best_config
    
    def record_complete_evaluation(self, params: T.Dict[str, T.Any], metrics: T.Dict[str, T.Any], 
                                qa_execution_logs: T.List[T.Dict[str, T.Any]]):
        """记录完整的评估结果"""
        logger.info(f"🧠 ===== 开始记录完整评估到三层图记忆系统 =====")
        logger.info(f"📊 输入数据统计: {len(qa_execution_logs)} QA executions")
        
        # 处理空日志情况
        if len(qa_execution_logs) == 0:
            logger.warning("⚠️ QA execution logs are empty, will create placeholder QA records for config tracking")
            
            placeholder_qa_log = {
                'question': 'Placeholder question due to evaluation failure',
                'ground_truth_answer': 'Placeholder ground truth',
                'predicted_answer': 'Evaluation failed',
                'f1_score': 0.0,
                'exact_match': False,
                'retrieval_precision': 0.0,
                'retrieval_recall': 0.0,
                'context_overlap': 0.0,
                'answer_relevance': 0.0,
                'retrieval_method': params.get('retrieval_method', 'unknown'),
                'embedding_model': params.get('embedding_model', 'unknown'),
                'template_name': params.get('template_name', 'unknown'),
                'reranker_enabled': params.get('reranker_enabled', False),
                'hyde_enabled': params.get('hyde_enabled', False),
                'total_execution_time': 0.0,
            }
            qa_execution_logs = [placeholder_qa_log]
            logger.info("✅ Created 1 placeholder QA record for config tracking")
        
        # Debug: 记录内存状态
        memory_stats_before = self.graph_memory.get_memory_stats()
        logger.info(f"📊 ===== 记录前图记忆状态 =====")
        logger.info(f"   Query Layer: {memory_stats_before['query_layer']['qa_executions']} QA executions")
        logger.info(f"   Config Layer: {memory_stats_before['config_layer']['configurations']} configurations")
        logger.info(f"                {memory_stats_before['config_layer']['config_relationships']} relationships")  
        logger.info(f"   Insight Layer: {memory_stats_before['insight_layer']['insights']} insights")
        logger.info(f"                 {memory_stats_before['insight_layer']['insight_relationships']} relationships")
        logger.info(f"📊 ===== 记录前状态统计完成 =====")
        
        # 转换QA日志为QAExecutionNode对象
        qa_executions = []
        config_id = self._generate_config_id(params)
        logger.info(f"🔑 生成config_id: {config_id}")
        logger.info(f"🔑 配置参数摘要: {json.dumps({k: v for k, v in params.items() if k in ['retrieval_method', 'template_name', 'reranker_enabled', 'hyde_enabled']}, ensure_ascii=False)}")
        
        logger.info(f"🔄 开始转换 {len(qa_execution_logs)} 个QA日志为QAExecutionNode...")
        for i, qa_log in enumerate(qa_execution_logs):
            qa_execution = self._convert_qa_log_to_node(qa_log, config_id, i)
            qa_executions.append(qa_execution)
            if i < 3:  # 只显示前3个的详细信息
                logger.info(f"   QA{i+1}: {qa_execution.qa_id} -> F1={qa_execution.f1_score:.3f}, question='{qa_execution.question[:50]}...'")
        logger.info(f"✅ QA日志转换完成: {len(qa_executions)} QAExecutionNodes")
        
        # 添加到三层图记忆系统
        logger.info(f"💾 开始添加到图记忆系统...")
        config_node = self.graph_memory.add_complete_evaluation(params, qa_executions)
        logger.info(f"✅ 图记忆系统更新完成")
        
        # Debug: 记录内存状态
        memory_stats_after = self.graph_memory.get_memory_stats()
        logger.info(f"📊 ===== 记录后图记忆状态 =====")
        logger.info(f"   Query Layer: {memory_stats_after['query_layer']['qa_executions']} QA executions (+{memory_stats_after['query_layer']['qa_executions'] - memory_stats_before['query_layer']['qa_executions']})")
        logger.info(f"   Config Layer: {memory_stats_after['config_layer']['configurations']} configurations (+{memory_stats_after['config_layer']['configurations'] - memory_stats_before['config_layer']['configurations']})")
        logger.info(f"                {memory_stats_after['config_layer']['config_relationships']} relationships (+{memory_stats_after['config_layer']['config_relationships'] - memory_stats_before['config_layer']['config_relationships']})")
        logger.info(f"   Insight Layer: {memory_stats_after['insight_layer']['insights']} insights (+{memory_stats_after['insight_layer']['insights'] - memory_stats_before['insight_layer']['insights']})")
        logger.info(f"                 {memory_stats_after['insight_layer']['insight_relationships']} relationships (+{memory_stats_after['insight_layer']['insight_relationships'] - memory_stats_before['insight_layer']['insight_relationships']})")
        logger.info(f"📊 ===== 记录后状态统计完成 =====")
        
        # 简化的insight生成
        existing_insights = list(self.graph_memory.insight_layer.nodes.values())
        logger.info(f"🧠 ===== 开始Insight生成流程 =====")
        logger.info(f"📚 当前已有insights: {len(existing_insights)}")
        logger.info(f"🎯 开始为config {config_id} 生成新insights...")
        
        # 生成新洞察
        new_insights = self.insight_agent.extract_insights_from_evaluation(
            config_node, qa_executions, existing_insights
        )
        
        # 添加新洞察到洞察层
        if new_insights:
            logger.info(f"💡 InsightAgent返回了 {len(new_insights)} 个新insights")
            for i, insight in enumerate(new_insights):
                logger.info(f"   Insight{i+1}: {insight.title} (置信度: {insight.confidence_score:.2f})")
            
            self.graph_memory.insight_layer.add_insights(new_insights)
            logger.info(f"✅ 已将 {len(new_insights)} 个新insights添加到insight层")
        else:
            logger.warning(f"⚠️ InsightAgent未生成任何新insights")
        
        # 最终内存状态检查
        memory_stats_final = self.graph_memory.get_memory_stats()
        logger.info(f"📊 ===== 最终图记忆状态 =====")
        logger.info(f"   Total QAs: {memory_stats_final['query_layer']['qa_executions']}")
        logger.info(f"   Total Configs: {memory_stats_final['config_layer']['configurations']}")  
        logger.info(f"   Total Insights: {memory_stats_final['insight_layer']['insights']}")
        logger.info(f"   Config F1: {config_node.avg_f1_score:.4f}")
        logger.info(f"   New insights: {len(new_insights) if new_insights else 0}")
        logger.info(f"📊 ===== 最终状态统计完成 =====")
        
        logger.info(f"✅ 完整评估记录完成: Config F1={config_node.avg_f1_score:.4f}, {len(new_insights) if new_insights else 0} new insights extracted")
        logger.info(f"🧠 ===== 三层图记忆系统记录完成 =====")

    def _migrate_old_knowledge_base(self, old_kb: T.Dict[str, T.Any]):
        """迁移旧知识库格式到新的三层系统"""
        logger.info(f"🔄 Migrating old knowledge base...")
        
        try:
            configs = old_kb.get('configs', [])
            
            # 处理旧的字典格式和新的列表格式
            if isinstance(configs, dict):
                configs = list(configs.values())
            
            # 转换旧记录到新格式
            for i, old_record in enumerate(configs):
                if isinstance(old_record, dict) and 'config' in old_record:
                    # 创建简化的QA执行记录
                    qa_execution = QAExecutionNode(
                        qa_id=f"migrated_{self.experiment_id}_{i}",
                        config_id=self._generate_config_id(old_record['config']),
                        question="Migrated historical question",
                        ground_truth_answer="Unknown",
                        f1_score=old_record.get('train_f1', 0.0),
                        retrieval_precision=0.7,
                        retrieval_recall=0.7,
                        retrieval_method=old_record['config'].get('retrieval_method', 'unknown'),
                        embedding_model=old_record['config'].get('embedding_model', 'unknown'),
                        template_name=old_record['config'].get('template_name', 'unknown'),
                        reranker_enabled=old_record['config'].get('reranker_enabled', False),
                        hyde_enabled=old_record['config'].get('hyde_enabled', False)
                    )
                    
                    # 添加到图记忆
                    self.graph_memory.query_layer.add_qa_execution(qa_execution)
            
            logger.info(f"✅ Successfully migrated {len(configs)} old records to new format")
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate old knowledge base: {e}")
    
    def _convert_qa_log_to_node(self, qa_log: T.Dict[str, T.Any], config_id: str, index: int) -> QAExecutionNode:
        """转换QA执行日志为QAExecutionNode"""
        # 确保QA ID全局唯一
        current_qa_count = len(self.graph_memory.query_layer.nodes)
        global_qa_index = current_qa_count + index
        
        return QAExecutionNode(
            qa_id=f"{self.experiment_id}_qa_{global_qa_index}",
            config_id=config_id,
            
            # 基本QA信息
            question=qa_log.get('question', 'Unknown question'),
            ground_truth_answer=qa_log.get('ground_truth', 'Unknown answer'),
            predicted_answer=qa_log.get('predicted_answer', 'Unknown prediction'),
            
            # 性能指标
            f1_score=qa_log.get('f1_score', 0.0),
            exact_match=qa_log.get('exact_match', False),
            retrieval_precision=qa_log.get('retrieval_precision', 0.0),
            retrieval_recall=qa_log.get('retrieval_recall', 0.0),
            context_overlap=qa_log.get('context_overlap', 0.0),
            answer_relevance=qa_log.get('answer_relevance', 0.0),
            
            # RAG流程详情
            raw_query=qa_log.get('raw_query', None),
            
            # 查询分解
            query_decomposition_enabled=qa_log.get('query_decomposition_enabled', False),
            decomposed_queries=qa_log.get('decomposed_queries', []),
            query_decomposition_llm=qa_log.get('query_decomposition_llm', None),
            decomposition_time=qa_log.get('decomposition_time', 0.0),
            
            # HyDE增强
            hyde_enabled=qa_log.get('hyde_enabled', False),
            hyde_query=qa_log.get('hyde_query', None),
            hyde_llm=qa_log.get('hyde_llm', None),
            hyde_time=qa_log.get('hyde_time', 0.0),
            
            # 检索配置
            embedding_model=qa_log.get('embedding_model', 'unknown'),
            retrieval_method=qa_log.get('retrieval_method', 'unknown'),
            retrieval_top_k=qa_log.get('retrieval_top_k', 10),
            hybrid_bm25_weight=qa_log.get('hybrid_bm25_weight', 0.5),
            retrieval_time=qa_log.get('retrieval_time', 0.0),
            
            # 融合处理
            fusion_enabled=qa_log.get('fusion_enabled', False),
            fusion_mode=qa_log.get('fusion_mode', None),
            fusion_time=qa_log.get('fusion_time', 0.0),
            
            # 重排序
            reranker_enabled=qa_log.get('reranker_enabled', False),
            reranker_llm=qa_log.get('reranker_llm', None),
            reranker_top_k=qa_log.get('reranker_top_k', 5),
            reranking_time=qa_log.get('reranking_time', 0.0),
            
            # 额外上下文
            additional_context_enabled=qa_log.get('additional_context_enabled', False),
            additional_context_num_nodes=qa_log.get('additional_context_num_nodes', 0),
            additional_context_time=qa_log.get('additional_context_time', 0.0),
            
            # 最终结果
            final_context=qa_log.get('final_context', ''),
            context_assembly_time=qa_log.get('context_assembly_time', 0.0),
            
            # Few-shot
            few_shot_enabled=qa_log.get('few_shot_enabled', False),
            few_shot_examples=qa_log.get('few_shot_examples', []),
            few_shot_retrieval_time=qa_log.get('few_shot_retrieval_time', 0.0),
            
            # 合成
            response_synthesizer_llm=qa_log.get('response_synthesizer_llm', 'unknown'),
            template_name=qa_log.get('template_name', 'unknown'),
            final_prompt=qa_log.get('final_prompt', ''),
            synthesis_time=qa_log.get('synthesis_time', 0.0),
            
            total_execution_time=qa_log.get('execution_time', 0.0)
        )
    
    def save_knowledge_base(self, file_path: str):
        """保存MCTS知识库到指定文件路径"""
        logger.info(f"🧠 保存图记忆系统到: {file_path}")
        
        try:
            # 图记忆系统有自己的持久化机制
            self.graph_memory.save_all_layers()
            
            # 为了兼容性，也可以生成legacy格式
            legacy_kb_data = self._get_legacy_knowledge_base_format()
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(legacy_kb_data, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 兼容性知识库已保存到: {file_path}")
            logger.info(f"💾 图记忆系统已保存到: {self.graph_memory.storage_path}")
        except Exception as e:
            logger.error(f"❌ 保存知识库时发生错误: {e}", exc_info=True)

    def _get_legacy_knowledge_base_format(self) -> T.Dict[str, T.Any]:
        """将图记忆数据转换为legacy知识库格式以保持兼容性"""
        try:
            configs = {}
            for config_id, config_node in self.graph_memory.config_layer.nodes.items():
                best_score = config_node.avg_f1_score
                worst_score = config_node.avg_f1_score
                
                if config_node.qa_execution_ids:
                    qa_scores = []
                    for qa_id in config_node.qa_execution_ids:
                        if qa_id in self.graph_memory.query_layer.nodes:
                            qa_scores.append(self.graph_memory.query_layer.nodes[qa_id].f1_score)
                    
                    if qa_scores:
                        best_score = max(qa_scores)
                        worst_score = min(qa_scores)
                
                configs[config_id] = {
                    'average_score': config_node.avg_f1_score,
                    'exploration_count': config_node.total_evaluations,
                    'best_score': best_score,
                    'worst_score': worst_score,
                    'config_hash': config_id,
                    'parameters': config_node.config_params,
                    'timestamp': config_node.evaluation_timestamps[-1] if config_node.evaluation_timestamps else time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            return {
                'experiment_id': self.experiment_id,
                'configs': configs,
                'metadata': {
                    'total_configs': len(configs),
                    'total_explorations': sum(c['exploration_count'] for c in configs.values()),
                    'graph_memory_stats': self.graph_memory.get_memory_stats()
                }
            }
        except Exception as e:
            logger.error(f"❌ 转换legacy格式失败: {e}")
            return {
                'experiment_id': self.experiment_id,
                'configs': {},
                'metadata': {'total_configs': 0, 'total_explorations': 0}
            }
    
    @property
    def knowledge_base(self) -> T.Dict[str, T.Any]:
        """提供向后兼容的knowledge_base属性"""
        return self._get_legacy_knowledge_base_format()

    def _build_search_space_from_config(self):
        """构建MCTS搜索空间"""
        from .hierarchical_search import RAGSearchSpace
        
        # 使用默认搜索空间
        return RAGSearchSpace()
    
    def _ensure_required_params(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        """确保所有必需参数都存在，使用与TPE相同的逻辑"""
        # 设置默认rag_mode
        final_params = {"rag_mode": "rag"}
        final_params.update(params)
        
        # 使用优化配置作为默认值
        defaults = {
            "enforce_full_evaluation": True,
            "template_name": final_params.get("template_name", "CoT"),
            "response_synthesizer_llm": final_params.get("response_synthesizer_llm", "Qwen2-7b"),
            "rag_embedding_model": final_params.get("embedding_model", "/mnt/data/wangshu/llm_lm/bge-m3"),
            "rag_method": final_params.get("retrieval_method", "sparse"),
            "rag_top_k": final_params.get("retrieval_top_k", 9),
            "splitter_method": final_params.get("splitter_method", "sentence"),
            "splitter_chunk_overlap_frac": final_params.get("splitter_overlap", 0.1),
        }
        
        # 处理chunk size参数
        if "splitter_chunk_size" in final_params:
            import math
            chunk_size = final_params["splitter_chunk_size"]
            if isinstance(chunk_size, (int, float)) and chunk_size > 0:
                chunk_exp = int(math.log2(chunk_size))
                if 2 ** chunk_exp != chunk_size:
                    chunk_exp = round(math.log2(chunk_size))
                defaults["splitter_chunk_exp"] = chunk_exp
            else:
                defaults["splitter_chunk_exp"] = 8
        else:
            defaults["splitter_chunk_exp"] = 8
        
        # 添加条件参数
        if final_params.get("retrieval_method") == "hybrid":
            defaults["rag_hybrid_bm25_weight"] = final_params.get("hybrid_bm25_weight", 0.5)
            
        # 查询分解参数（强制开启）
        if final_params.get("query_decomposition_enabled", True):
            defaults["rag_query_decomposition_enabled"] = True
            defaults["rag_query_decomposition_num_queries"] = final_params.get("query_decomposition_num_queries", 4)
            defaults["rag_query_decomposition_llm_name"] = final_params.get("query_decomposition_llm", "Qwen2-7b")
            defaults["rag_fusion_mode"] = final_params.get("fusion_mode", "simple")
        else:
            defaults["rag_query_decomposition_enabled"] = False
            
        # Hyde参数（🔥 强制关闭HyDE以减少搜索空间）
        # # Hyde参数（强制开启）
        # if final_params.get("hyde_enabled", True):
        #     defaults["hyde_enabled"] = True
        #     defaults["hyde_llm_name"] = final_params.get("hyde_llm", "Qwen2-7b")
        # else:
        #     defaults["hyde_enabled"] = False
        defaults["hyde_enabled"] = False
            
        # Reranker参数
        defaults["reranker_enabled"] = True
        defaults["reranker_llm_name"] = final_params.get("reranker_llm", "Qwen2-7b")
        defaults["reranker_top_k"] = final_params.get("reranker_top_k", 5)
            
        # 额外上下文参数（强制开启）
        if final_params.get("additional_context_enabled", True):
            defaults["additional_context_enabled"] = True
            defaults["additional_context_num_nodes"] = final_params.get("additional_context_num_nodes", 5)
        else:
            defaults["additional_context_enabled"] = False
            
        # Few-shot参数
        defaults["few_shot_enabled"] = False
        
        # 更新默认值
        for key, value in defaults.items():
            if key not in final_params:
                final_params[key] = value
                
        return final_params
    
    def _get_default_config(self) -> T.Dict[str, T.Any]:
        """获取默认配置"""
        return {
            "rag_mode": "rag",
            "splitter_method": "sentence",
            "splitter_chunk_size": 256,
            "splitter_overlap": 0.1,
            "embedding_model": "/mnt/data/wangshu/llm_lm/bge-m3",
            "retrieval_method": "sparse",
            "retrieval_top_k": 9,
            "query_decomposition_enabled": True,
            "query_decomposition_num_queries": 4,
            "query_decomposition_llm": "Qwen2-7b",
            "fusion_mode": "simple",
            # "hyde_enabled": True,  
            "hyde_enabled": False,#关闭HyDE
            "hyde_llm": "Qwen2-7b",
            "reranker_enabled": True,
            "additional_context_enabled": True,
            "additional_context_num_nodes": 5,
            "response_synthesizer_llm": "Qwen2-7b",
            "template_name": "CoT",
            "few_shot_enabled": False,
        }
    
    def _generate_experiment_id(self) -> str:
        """生成实验标识符"""
        import sys
        import os
        current_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
        
        cmd_args = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'default'
        experiment_id = f"true_mcts_{timestamp}_{hash(cmd_args) % 10000:04d}"
        return experiment_id
    
    def _generate_config_id(self, config_params: T.Dict) -> str:
        """Generate unique config ID"""
        config_str = json.dumps(config_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def get_graph_memory_stats(self) -> T.Dict[str, T.Any]:
        """Get current graph memory statistics"""
        return self.graph_memory.get_memory_stats()
        
    # 保持原方法以兼容性
    def record_real_evaluation(self, params: T.Dict[str, T.Any], metrics: T.Dict[str, T.Any]):
        """Legacy method for backward compatibility"""
        logger.warning("⚠️ Using legacy record_real_evaluation method. Consider upgrading to record_complete_evaluation.")
        
        # 创建简化的QA执行日志
        qa_log = {
            'question': 'Legacy evaluation',
            'ground_truth': 'Unknown',
            'predicted_answer': 'Unknown',
            'f1_score': metrics.get('train_joint_f1', 0.0),
            'retrieval_precision': 0.7,
            'retrieval_recall': 0.7,
            'execution_time': 1.0
        }
        
        self.record_complete_evaluation(params, metrics, [qa_log])

# # 保持向后兼容性
# class MCTSOptimizationEngine(EnhancedMCTSOptimizationEngine):
#     """Legacy MCTS optimization engine for backward compatibility"""
    
#     def __init__(self, api_key: str, api_base: str, experiment_id: str = None, 
#                  existing_knowledge_base: T.Dict[str, T.Any] = None):
#         logger.warning("⚠️ Using legacy MCTSOptimizationEngine. Consider upgrading to EnhancedMCTSOptimizationEngine.")
#         super().__init__(api_key, api_base, experiment_id, existing_knowledge_base)