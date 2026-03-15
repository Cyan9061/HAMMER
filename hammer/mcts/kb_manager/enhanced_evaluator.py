"""
#hammer/mcts/kb_manager/enhanced_evaluator.py
最新版修改
修复后的enhanced_evaluator.py完整版

主要修复：
1. 完整提取Query层所有有价值信息（包括decomposed_queries等）
2. 使用参数匹配相似度替代向量相似度
3. 删除废弃的向量计算代码
4. 确保与Config层正确连接
"""
#from hammer.tuner.main_tuner_mcts import MODEL_SIMUL
MODEL_SIMUL = "Qwen/Qwen2.5-72B-Instruct-128K"
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

import openai
import tiktoken
from hammer.logger import logger
from hammer.mcts.kb_manager.graph_memory import GraphMemoryRAGMCTS, QAExecutionNode, ConfigNode, InsightNode
from hammer.mcts.kb_manager.insight_agent import RAGInsightPrompts, RAGConfigurationAnalyzer, InsightAgent

class EnhancedGPTSimulationEvaluator:
    """修复后的增强GPT评估器 - 完整版"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, 
                 graph_memory: Optional[GraphMemoryRAGMCTS] = None,
                 model_name: str = "Qwen/Qwen2.5-72B-Instruct-128K"):
        # 使用显式传参，必要时回退到知识库专用环境变量。
        api_key = (
            api_key
            or os.getenv("OPENAI_KB_API_KEY")
            or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("SILICONFLOW_API_KEYS", "").split(",")[0]
            or ""
        )
        api_base = api_base or os.getenv("OPENAI_KB_API_BASE") or "https://api.siliconflow.cn/v1"

        self.gpt_client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.use_fallback = False  # 完全禁用fallback机制
        
        self.graph_memory = graph_memory or GraphMemoryRAGMCTS()
        self.prompts = RAGInsightPrompts()  # 🔥 使用新的英文prompt类
        self.analyzer = RAGConfigurationAnalyzer()
        
        # 🔥 新增：创建insight_agent实例用于处理insights
        # from .insight_agent import InsightAgent
        self.insight_agent = InsightAgent(api_key=api_key, api_base=api_base, model_name=self.model_name)

        logger.info(f"🚀 Enhanced GPT Evaluator initialized with {self.model_name}")
        
        # 初始化tokenizer用于token计数
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Token计数方法"""
        if not self.tokenizer:
            return len(text) // 4
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"⚠️ Token counting error: {e}")
            return len(text) // 4
    
    def evaluate_configuration(self, params: Dict[str, Any], query: str, predict_score:float = 0) -> float:
        """增强配置评估的主入口"""
        logger.info(f"🧠 Starting enhanced evaluation for query: {query[:50]}...")
        
        # Step 1: 多维度上下文检索
        evaluation_context = self._build_intelligent_context(params, query)
        
        # Step 2: 增强GPT评估（只使用GPT，无fallback）
        try:
            score = self._enhanced_gpt_evaluation(params, query, evaluation_context, predict_score)
            logger.info(f"✅ GPT evaluation complete: {score:.4f}")
            return score
        except Exception as e:
            logger.error(f"❌ GPT evaluation failed: {e}")
            raise e  # 直接抛出异常，不使用fallback
    
    def _build_intelligent_context(self, params: Dict[str, Any], query: str) -> Dict[str, Any]:
        """构建智能评估上下文"""
        # logger.info("🔍 Building intelligent evaluation context...")
        
        # 检查内存状态
        memory_stats = self.graph_memory.get_memory_stats()
        total_configs = memory_stats['config_layer']['configurations']
        total_qas = memory_stats['query_layer']['qa_executions']
        total_insights = memory_stats['insight_layer']['insights']
        
        logger.info(f"📊 Memory state: {total_configs} configs, {total_qas} QAs, {total_insights} insights")
        
        # 🔥 修复后的三步检索
        all_insights = self._retrieve_all_insights()
        # logger.info(f"✅ Found {len(all_insights)} total insights")
        
        top_3_configs = self._retrieve_top_3_similar_configurations(params)
        # logger.info(f"✅ Found {len(top_3_configs)} similar configs")
        
        representative_queries = self._extract_complete_query_details(top_3_configs)
        logger.info(f"✅Found {len(all_insights)} total insights，Found {len(top_3_configs)} similar configs, Extracted {len(representative_queries)} complete query details")
        
        context = {
            'all_insights': all_insights,
            'top_3_configs': top_3_configs,
            'representative_queries': representative_queries,
        }
        
        # 增强日志记录
        # logger.info(f"📊 Context built summary:")
        # logger.info(f"  📝 All Insights: {len(all_insights)} items")
        # logger.info(f"  ⚙️  Top Configs: {len(top_3_configs)} items")
        # logger.info(f"  ❓ Complete Query Details: {len(representative_queries)} items")
        
        return context
    
    def _retrieve_all_insights(self) -> List[Dict]:
        """获取全部洞察 - 适配新的insight格式"""
        all_insights = list(self.graph_memory.insight_layer.nodes.values())
        
        formatted_insights = []
        for insight in all_insights:
            formatted_insights.append({
                'insight': insight,
                'title': insight.title,
                'description': insight.description,
                'confidence': insight.confidence_score,
                'type': insight.insight_type,
                'recommendation': insight.recommendation,
                'supporting_config_ids': insight.supporting_config_ids  # 🔥 新增：用于简化查找
            })
        
        logger.info(f"📝 Retrieved {len(formatted_insights)} insights for evaluation context")
        return formatted_insights
    
    def _retrieve_top_3_similar_configurations(self, params: Dict[str, Any]) -> List[Dict]:
        """
        🔥 核心修复：使用参数匹配而非向量相似度计算配置相似度
        """
        total_configs = len(self.graph_memory.config_layer.nodes)
        logger.info(f"🔍 Finding top 3 configs among {total_configs} available configs using parameter matching")
        
        if total_configs == 0:
            logger.warning("⚠️ No configurations available in memory")
            return []
        
        config_similarities = []
        
        # 创建临时的ConfigNode用于相似度计算
        temp_config_node = ConfigNode(
            config_id="temp",
            config_params=params,
            avg_f1_score=0.0,
            avg_retrieval_precision=0.0,
            avg_retrieval_recall=0.0,
            avg_execution_time=0.0
        )
        
        for config_id, config_node in self.graph_memory.config_layer.nodes.items():
            try:
                # 🔥 关键修改：使用config_layer的参数匹配方法
                similarity = self.graph_memory.config_layer._compute_config_similarity(
                    temp_config_node, config_node
                )
                
                config_similarities.append({
                    'config_node': config_node,
                    'config_id': config_id,
                    'similarity': similarity,
                    'performance_summary': self._summarize_config_performance(config_node)
                })
                
            except Exception as e:
                logger.error(f"❌ Error computing similarity for config {config_id}: {e}")
                continue
        
        # 按相似度排序，取前3个（无阈值限制）
        config_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_3_configs = config_similarities[:3]
        
        # 详细日志记录
        logger.info(f"🎯 Selected top {len(top_3_configs)} configs by parameter similarity:")
        for i, config_ctx in enumerate(top_3_configs, 1):
            logger.info(f"  Config {i}: similarity={config_ctx['similarity']:.4f}, "
                       f"F1={config_ctx['config_node'].avg_f1_score:.4f}")
        
        return top_3_configs
    
    def _extract_complete_query_details(self, top_configs: List[Dict]) -> List[Dict]:
        """
        🔥 核心修复：提取完整的RAG执行详情，包含所有信息且无截断
        """
        if not top_configs:
            return []
        
        all_complete_queries = []
        
        # 从每个config收集其关联的QA执行详情
        for config_ctx in top_configs:
            config_node = config_ctx['config_node']
            config_id = config_ctx['config_id']
            
            for qa_id in config_node.qa_execution_ids:
                if qa_id in self.graph_memory.query_layer.nodes:
                    qa_node = self.graph_memory.query_layer.nodes[qa_id]
                    
                    # 🔥 关键改进：提取所有信息，完全无截断
                    complete_query_details = {
                        'qa_execution': qa_node,
                        'config_id': config_id,
                        'config_similarity': config_ctx['similarity'],
                        'f1_score': qa_node.f1_score,
                        
                        # ===== 基本QA信息（完整） =====
                        'qa_id': qa_node.qa_id,
                        'question': qa_node.question,
                        'ground_truth_answer': qa_node.ground_truth_answer,
                        'ground_truth_context': qa_node.ground_truth_context,
                        'predicted_answer': qa_node.predicted_answer,
                        'exact_match': qa_node.exact_match,
                        
                        # ===== 查询处理详情（完整） =====
                        'raw_query': qa_node.raw_query,
                        'query_decomposition_enabled': qa_node.query_decomposition_enabled,
                        'decomposed_queries': qa_node.decomposed_queries,
                        'query_decomposition_llm': qa_node.query_decomposition_llm,
                        'decomposition_time': qa_node.decomposition_time,
                        'hyde_enabled': qa_node.hyde_enabled,
                        'hyde_query': qa_node.hyde_query,
                        'hyde_llm': qa_node.hyde_llm,
                        'hyde_time': qa_node.hyde_time,
                        
                        # ===== 检索配置详情（完整） =====
                        'embedding_model': qa_node.embedding_model,
                        'retrieval_method': qa_node.retrieval_method,
                        'retrieval_top_k': qa_node.retrieval_top_k,
                        'hybrid_bm25_weight': qa_node.hybrid_bm25_weight,
                        'initial_retrieved_docs': qa_node.initial_retrieved_docs,  # 🔥 完整文档，无截断
                        'retrieval_time': qa_node.retrieval_time,
                        
                        # ===== 融合处理详情（完整） =====
                        'fusion_enabled': qa_node.fusion_enabled,
                        'fusion_mode': qa_node.fusion_mode,
                        'fused_docs': qa_node.fused_docs,  # 🔥 完整文档，无截断
                        'fusion_time': qa_node.fusion_time,
                        
                        # ===== 重排序详情（完整） =====
                        'reranker_enabled': qa_node.reranker_enabled,
                        'reranker_llm': qa_node.reranker_llm,
                        'reranker_top_k': qa_node.reranker_top_k,
                        'reranker_results': qa_node.reranker_results,  # 🔥 完整结果，无截断
                        'reranking_time': qa_node.reranking_time,
                        
                        # ===== 额外上下文详情（完整） =====
                        'additional_context_enabled': qa_node.additional_context_enabled,
                        'additional_context_num_nodes': qa_node.additional_context_num_nodes,
                        'additional_context_docs': qa_node.additional_context_docs,  # 🔥 完整文档，无截断
                        'additional_context_time': qa_node.additional_context_time,
                        
                        # ===== 最终结果详情（完整） =====
                        'final_retrieved_docs': qa_node.final_retrieved_docs,  # 🔥 完整文档，无截断
                        'final_context': qa_node.final_context,  # 🔥 完整上下文，无截断
                        'context_assembly_time': qa_node.context_assembly_time,
                        'final_prompt': qa_node.final_prompt,  # 🔥 完整prompt，无截断
                        'synthesis_time': qa_node.synthesis_time,
                        
                        # ===== 性能指标（完整） =====
                        'retrieval_precision': qa_node.retrieval_precision,
                        'retrieval_recall': qa_node.retrieval_recall,
                        'context_overlap': qa_node.context_overlap,
                        'answer_relevance': qa_node.answer_relevance,
                        'total_execution_time': qa_node.total_execution_time,
                        
                        # ===== 模板和LLM信息（完整） =====
                        'template_name': qa_node.template_name,
                        'response_synthesizer_llm': qa_node.response_synthesizer_llm,
                        
                        # ===== Few-shot信息（完整） =====
                        'few_shot_enabled': qa_node.few_shot_enabled,
                        'few_shot_examples': qa_node.few_shot_examples,  # 🔥 完整示例，无截断
                        'few_shot_retrieval_time': qa_node.few_shot_retrieval_time,
                    }
                    
                    all_complete_queries.append(complete_query_details)
        
        if not all_complete_queries:
            logger.warning("⚠️ No complete query details found from top configs")
            return []
        
        # 🔥 智能选择策略：选择多样化的代表性样本
        all_complete_queries.sort(key=lambda x: x['f1_score'], reverse=True)
        
        representative_queries = []
        total_queries = len(all_complete_queries)
        
        if total_queries >= 10:
            # 丰富样本：高分3个 + 低分3个 + 中分4个
            representative_queries.extend(all_complete_queries[:3])  # 高分3个
            representative_queries.extend(all_complete_queries[-3:])  # 低分3个
            middle_start = total_queries // 3
            middle_end = 2 * total_queries // 3
            middle_queries = all_complete_queries[middle_start:middle_end]
            representative_queries.extend(middle_queries[:4])  # 中分4个
        elif total_queries >= 5:
            # 标准策略：高分2个 + 低分2个 + 中分1个
            representative_queries.extend(all_complete_queries[:2])
            representative_queries.extend(all_complete_queries[-2:])
            middle_index = total_queries // 2
            representative_queries.append(all_complete_queries[middle_index])
        else:
            # 数量不足，全部选择
            representative_queries = all_complete_queries
        
        # 详细日志记录完整信息
        logger.info(f"🎯 Extracted {len(representative_queries)} complete query details (no truncation):")
        for i, query_ctx in enumerate(representative_queries, 1):
            logger.info(f"  Query {i}: F1={query_ctx['f1_score']:.4f}")
            logger.info(f"    Question: {query_ctx['question']}")
            logger.info(f"    Ground Truth: {query_ctx['ground_truth_answer']}")
            logger.info(f"    Predicted: {query_ctx['predicted_answer']}")
            
            # 显示关键RAG配置
            rag_config_summary = []
            if query_ctx['query_decomposition_enabled']:
                rag_config_summary.append(f"QueryDecomp({len(query_ctx['decomposed_queries'])})")
            if query_ctx['hyde_enabled']:
                rag_config_summary.append("HyDE")
            rag_config_summary.append(f"Retrieval({query_ctx['retrieval_method']})")
            if query_ctx['fusion_enabled']:
                rag_config_summary.append(f"Fusion({query_ctx['fusion_mode']})")
            if query_ctx['reranker_enabled']:
                rag_config_summary.append(f"Rerank(top-{query_ctx['reranker_top_k']})")
            rag_config_summary.append(f"Template({query_ctx['template_name']})")
            
            logger.info(f"    RAG Pipeline: {' → '.join(rag_config_summary)}")
            logger.info(f"    Docs: initial={len(query_ctx['initial_retrieved_docs'])}, "
                    f"final={len(query_ctx['final_retrieved_docs'])}")
        
        return representative_queries
    
    def _format_complete_docs_list(self, docs: List[Dict], doc_type: str = "docs") -> str:
        """格式化完整文档列表，无截断"""
        if not docs:
            return f"No {doc_type} available."
        
        formatted = f"\n**{doc_type.title()} ({len(docs)} items):**\n"
        for i, doc in enumerate(docs, 1):
            formatted += f"  **Doc {i}:**\n"
            
            # 基本信息
            if isinstance(doc, dict):
                formatted += f"    - Doc ID: {doc.get('doc_id', 'N/A')}\n"
                formatted += f"    - Score: {doc.get('score', 'N/A')}\n"
                formatted += f"    - Node Type: {doc.get('node_type', 'N/A')}\n"
                
                # 元数据
                metadata = doc.get('metadata', {})
                if isinstance(metadata, dict) and metadata:
                    formatted += f"    - Title: {metadata.get('title', 'N/A')}\n"
                    for key, value in metadata.items():
                        if key != 'title':
                            formatted += f"    - {key}: {value}\n"
                
                # 完整文本内容，无截断
                text = doc.get('text', '')
                if text:
                    formatted += f"    - Full Text: {text}\n"
                
                # 其他字段
                for key, value in doc.items():
                    if key not in ['doc_id', 'score', 'node_type', 'metadata', 'text']:
                        formatted += f"    - {key}: {value}\n"
            else:
                formatted += f"    - Content: {str(doc)}\n"
            
            formatted += "\n"
        
        return formatted

    def _extract_doc_summaries(self, docs: List[Dict], max_docs: int) -> List[Dict]:
        """提取文档摘要信息"""
        if not docs:
            return []
        
        summaries = []
        for doc in docs[:max_docs]:
            if isinstance(doc, dict):
                text = doc.get('text', '')
                truncated_text = self._truncate_text(text, 150)
                
                summary = {
                    'text': truncated_text,
                    'score': doc.get('score', 'N/A'),
                    'title': doc.get('metadata', {}).get('title', 'N/A') if isinstance(doc.get('metadata'), dict) else 'N/A'
                }
                summaries.append(summary)
        
        return summaries
    
    def _build_evaluation_prompt(self, params: Dict[str, Any], query: str, 
                                context: Dict[str, Any], predict_score: float) -> str:
        """
        🔥 核心修复：构建完整的评估prompt，包含所有信息且无截断
        """
        
        # 格式化top3配置的完整参数信息（保持不变）
        similar_configs_text = ""
        for i, config_ctx in enumerate(context['top_3_configs'], 1):
            config_node = config_ctx['config_node']
            similar_configs_text += f"""
    **Similar Configuration {i}** (Parameter Similarity: {config_ctx['similarity']:.4f})
    ```json
    {json.dumps(config_node.config_params, indent=2)}
    ```
    **Performance Summary:**
    - Average F1 Score: {config_node.avg_f1_score:.4f}
    - Average Retrieval Precision: {config_node.avg_retrieval_precision:.4f}
    - Average Retrieval Recall: {config_node.avg_retrieval_recall:.4f}
    - Total Evaluations: {config_node.total_evaluations}
    - Question Type Performance: {json.dumps(config_node.question_type_performance, indent=2)}
    """
        
        # 🔥 格式化完整的RAG执行详情，无任何截断
        representative_queries_text = ""
        for i, query_ctx in enumerate(context['representative_queries'], 1):
            representative_queries_text += f"""
    **Representative Query {i}** (F1: {query_ctx['f1_score']:.4f}, Config Similarity: {query_ctx['config_similarity']:.4f})

    **=== Basic QA Information ===**
    - QA ID: {query_ctx['qa_id']}
    - Question: "{query_ctx['question']}"
    - Ground Truth Answer: "{query_ctx['ground_truth_answer']}"
    - Ground Truth Context: {json.dumps(query_ctx['ground_truth_context'], indent=2)}
    - Predicted Answer: "{query_ctx['predicted_answer']}"
    - Exact Match: {query_ctx['exact_match']}
    - F1 Score: {query_ctx['f1_score']:.4f}

    ---
    """
        
        # 🔥 关键修改：使用简化的insight接口
        config_ids = [config_ctx['config_id'] for config_ctx in context['top_3_configs']]
        all_insights = [insight_ctx['insight'] for insight_ctx in context['all_insights']]
        
        # 使用insight_agent的新接口获取config关联的insights
        config_insights_map = self.insight_agent.get_config_linked_insights(config_ids, all_insights)
        
        # 格式化insights用于prompt
        config_linked_insights_text = self.insight_agent.format_insights_for_evaluation(config_insights_map)
        
        # 🔥 使用英文优化的prompt模板,EVALUATE_CONFIG_USER_HYBRID是知识库+真实评估，EVALUATE_CONFIG_USER是纯知识库
        return self.prompts.EVALUATE_CONFIG_USER_HYBRID.format(
            true_score=predict_score,
            query=query,
            config_json=json.dumps(params, indent=2),
            similar_configs_with_insights=similar_configs_text + "\n" + config_linked_insights_text,
            representative_queries=representative_queries_text or "No representative queries found."
        )
    
    def _format_doc_list(self, docs: List[Dict]) -> str:
        """格式化文档列表"""
        if not docs:
            return "No documents retrieved."
        
        formatted = ""
        for i, doc in enumerate(docs, 1):
            formatted += f"  Doc {i} (Score: {doc.get('score', 'N/A')}, Title: {doc.get('title', 'N/A')}):\n"
            formatted += f"    {doc.get('text', 'No text available')}\n"
        
        return formatted
    
    def _enhanced_gpt_evaluation(self, params: Dict[str, Any], query: str, 
                                context: Dict[str, Any], predict_score:float) -> float:
        """增强GPT评估"""
        evaluation_prompt = self._build_evaluation_prompt(params, query, context, predict_score)
        
        system_prompt = self.prompts.EVALUATE_CONFIG_SYSTEM
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(evaluation_prompt)
        total_tokens = system_tokens + user_tokens
        
        response = self.gpt_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.,
            max_tokens=4096
        )
        
        # 🔥 记录token使用到全局统计
        from hammer.utils.simple_token_tracker import record_openai_response
        record_openai_response(response, self.model_name)
        
        content = response.choices[0].message.content.strip()
        score = self._extract_score_enhanced(content)
        
        logger.info(f"🤖 GPT evaluation: tokens={total_tokens}, score={score:.4f}, \nprompt={evaluation_prompt}, \n\nrespnose={content}")
        
        return max(0.0, min(1.0, score))
    
    def _extract_score_enhanced(self, content: str) -> float:
        """从回复中提取评分"""
        try:
            import re
            pattern = r'Predict_Score:\s*(\d+(?:\.\d+)?)\s*'
            matches = re.findall(pattern, content)
            if matches:
                score = float(matches[-1])
                if score > 1.0:
                    score = score / 100.0
                return score
            else:
                logger.warning(f"无法从响应中提取评分: {content}")
                return 0.5
        except Exception as e:
            logger.error(f"评分提取失败: {e}")
            return 0.5
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """文本截断工具"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + '...'
    
    def _summarize_config_performance(self, config_node: ConfigNode) -> str:
        """总结配置性能"""
        return (f"F1: {config_node.avg_f1_score:.4f}, "
                f"Precision: {config_node.avg_retrieval_precision:.4f}, "
                f"Evaluations: {config_node.total_evaluations}")

#被删除的内容：
"""
    **=== Query Processing Pipeline ===**
    - Raw Query: {query_ctx['raw_query']}
    - Query Decomposition Enabled: {query_ctx['query_decomposition_enabled']}
    - Decomposed Queries: {json.dumps(query_ctx['decomposed_queries'], indent=2)}
    - Query Decomposition LLM: {query_ctx['query_decomposition_llm']}
    - Decomposition Time: {query_ctx['decomposition_time']}s
    - HyDE Enabled: {query_ctx['hyde_enabled']}
    - HyDE Query: {query_ctx['hyde_query']}
    - HyDE LLM: {query_ctx['hyde_llm']}
    - HyDE Time: {query_ctx['hyde_time']}s

    **=== Retrieval Configuration ===**
    - Embedding Model: {query_ctx['embedding_model']}
    - Retrieval Method: {query_ctx['retrieval_method']}
    - Retrieval Top-K: {query_ctx['retrieval_top_k']}
    - Hybrid BM25 Weight: {query_ctx['hybrid_bm25_weight']}
    - Retrieval Time: {query_ctx['retrieval_time']}s

    **=== Initial Retrieved Documents ===**
    {self._format_complete_docs_list(query_ctx['initial_retrieved_docs'], "Initial Retrieved Documents")}

    **=== Fusion Processing ===**
    - Fusion Enabled: {query_ctx['fusion_enabled']}
    - Fusion Mode: {query_ctx['fusion_mode']}
    - Fusion Time: {query_ctx['fusion_time']}s

    **=== Fused Documents ===**
    {self._format_complete_docs_list(query_ctx['fused_docs'], "Fused Documents")}

    **=== Reranking Processing ===**
    - Reranker Enabled: {query_ctx['reranker_enabled']}
    - Reranker LLM: {query_ctx['reranker_llm']}
    - Reranker Top-K: {query_ctx['reranker_top_k']}
    - Reranking Time: {query_ctx['reranking_time']}s

    **=== Reranker Results ===**
    {self._format_complete_docs_list(query_ctx['reranker_results'], "Reranker Results")}

    **=== Additional Context Processing ===**
    - Additional Context Enabled: {query_ctx['additional_context_enabled']}
    - Additional Context Num Nodes: {query_ctx['additional_context_num_nodes']}
    - Additional Context Time: {query_ctx['additional_context_time']}s

    **=== Additional Context Documents ===**
    {self._format_complete_docs_list(query_ctx['additional_context_docs'], "Additional Context Documents")}

    **=== Final Results ===**
    - Context Assembly Time: {query_ctx['context_assembly_time']}s

    **=== Final Retrieved Documents ===**
    {self._format_complete_docs_list(query_ctx['final_retrieved_docs'], "Final Retrieved Documents")}

    **=== Final Context (Complete, No Truncation) ===**
    {query_ctx['final_context']}

    **=== Few-shot Configuration ===**
    - Few-shot Enabled: {query_ctx['few_shot_enabled']}
    - Few-shot Examples: {json.dumps(query_ctx['few_shot_examples'], indent=2)}
    - Few-shot Retrieval Time: {query_ctx['few_shot_retrieval_time']}s

    **=== Response Synthesis ===**
    - Response Synthesizer LLM: {query_ctx['response_synthesizer_llm']}
    - Template Name: {query_ctx['template_name']}
    - Synthesis Time: {query_ctx['synthesis_time']}s

    **=== Final Prompt (Complete, No Truncation) ===**
    {query_ctx['final_prompt']}

    **=== Performance Metrics ===**
    - F1 Score: {query_ctx['f1_score']:.4f}
    - Retrieval Precision: {query_ctx['retrieval_precision']:.4f}
    - Retrieval Recall: {query_ctx['retrieval_recall']:.4f}
    - Context Overlap: {query_ctx['context_overlap']:.4f}
    - Answer Relevance: {query_ctx['answer_relevance']:.4f}
    - Total Execution Time: {query_ctx['total_execution_time']}s
"""
