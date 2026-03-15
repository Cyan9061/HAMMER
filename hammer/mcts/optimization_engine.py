"""
True MCTS optimization engine implementation.
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

# Import the enhanced three-layer graph-memory system.
from .kb_manager import (
    GraphMemoryRAGMCTS,
    QAExecutionNode,
    InsightAgent,
)

class OptimizationEngine(ABC):
    """Abstract base class for optimization engines."""
    
    @abstractmethod
    def suggest_parameters(self, trial: optuna.Trial, components: T.List[str]) -> T.Dict[str, T.Any]:
        """Suggest a parameter configuration."""
        pass

class EnhancedMCTSOptimizationEngine(OptimizationEngine):
    """Enhanced MCTS optimization engine with true MCTS implementation"""
    
    def __init__(self, api_key: str, api_base: str, experiment_id: str = None, iteration: int = 50,
                existing_knowledge_base: T.Dict[str, T.Any] = None):
        self.api_key = api_key
        self.api_base = api_base
        self.experiment_id = experiment_id or self._generate_experiment_id()
        
        # Initialize the three-layer graph-memory system.
        graph_memory_path = f"Experiment/graph_memory/{self.experiment_id}"
        self.graph_memory = GraphMemoryRAGMCTS(storage_path=graph_memory_path)
        
        # Migrate an old-format knowledge base if one is provided.
        if existing_knowledge_base:
            self._migrate_old_knowledge_base(existing_knowledge_base)
        
        # Initialize the insight agent.
        self.insight_agent = InsightAgent(
            api_key=self.api_key,
            api_base=self.api_base
        )
        
        # MCTS configuration.
        self.mcts_iterations = 1000
        self.current_search_idx = 0
        self.max_searches = 1000
        
        # Real-evaluation callback registered externally.
        self.evaluation_callback = None
        
        logger.info(f"🚀 Enhanced MCTS Optimization Engine initialized")
        logger.info(f"🎯 MCTS iterations per search: {self.mcts_iterations}")
        logger.info(f"🧠 Graph memory: {self.graph_memory.get_memory_stats()}")

    def set_evaluation_callback(self, callback: T.Callable[[T.Dict[str, T.Any]], float]):
        """Register the real-evaluation callback."""
        self.evaluation_callback = callback
        logger.info("✅ Real evaluation callback set")

    def suggest_parameters(self) -> T.Dict[str, T.Any]:
        """Suggest parameters using the real MCTS search loop."""
        if self.evaluation_callback is None:
            logger.error("❌ Evaluation callback not set! Cannot perform MCTS search.")
            return self._get_default_config()
        
        if self.current_search_idx >= self.max_searches:
            logger.warning("All %s searches are complete; returning the default configuration", self.max_searches)
            return self._get_default_config()
        
        logger.info(
            "Starting real MCTS parameter search %s/%s",
            self.current_search_idx + 1,
            self.max_searches,
        )
        
        # Run the MCTS search.
        best_config = self._execute_true_mcts_search()
        
        # Add compatibility defaults expected by the rest of the system.
        enhanced_params = self._ensure_required_params(best_config)
        
        # Advance search state.
        self.current_search_idx += 1
        
        return enhanced_params
    
    def _execute_true_mcts_search(self) -> T.Dict[str, T.Any]:
        """Execute the real MCTS search."""
        from .hierarchical_search import TrueMCTS, RAGSearchSpace
        
        # Build the search space.
        search_space = self._build_search_space_from_config()
        
        # Debug: record the current memory state.
        memory_stats = self.graph_memory.get_memory_stats()
        logger.info(f"🧠 Pre-search memory state: {memory_stats['config_layer']['configurations']} configs, "
                f"{memory_stats['insight_layer']['insights']} insights")
        
        # Build the TrueMCTS instance and pass in the knowledge base and insight agent.
        mcts_instance = TrueMCTS(
            search_space=search_space,
            evaluation_callback=self.evaluation_callback,
            exploration_constant=1.414,
            max_iterations=self.mcts_iterations,
            graph_memory=self.graph_memory,
            insight_agent=self.insight_agent
        )
        
        # Execute search.
        best_config = mcts_instance.search()
        
        logger.info("GPT-guided MCTS search completed")
        return best_config
    
    def record_complete_evaluation(self, params: T.Dict[str, T.Any], metrics: T.Dict[str, T.Any], 
                                qa_execution_logs: T.List[T.Dict[str, T.Any]]):
        """Record the complete evaluation into the graph-memory system."""
        logger.info("===== Recording complete evaluation into the three-layer graph-memory system =====")
        logger.info("Input QA execution count: %s", len(qa_execution_logs))
        
        # Handle empty-log cases.
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
        
        # Debug: record pre-update memory stats.
        memory_stats_before = self.graph_memory.get_memory_stats()
        logger.info("===== Pre-record graph-memory state =====")
        logger.info("   Query Layer: %s QA executions", memory_stats_before['query_layer']['qa_executions'])
        logger.info("   Config Layer: %s configurations", memory_stats_before['config_layer']['configurations'])
        logger.info("                %s relationships", memory_stats_before['config_layer']['config_relationships'])
        logger.info("   Insight Layer: %s insights", memory_stats_before['insight_layer']['insights'])
        logger.info("                 %s relationships", memory_stats_before['insight_layer']['insight_relationships'])
        logger.info("===== Finished pre-record state summary =====")
        
        # Convert QA logs into QAExecutionNode objects.
        qa_executions = []
        config_id = self._generate_config_id(params)
        logger.info("Generated config_id: %s", config_id)
        logger.info(
            "Config summary: %s",
            json.dumps({k: v for k, v in params.items() if k in ['retrieval_method', 'template_name', 'reranker_enabled', 'hyde_enabled']}, ensure_ascii=False),
        )
        
        logger.info("Converting %s QA logs into QAExecutionNode objects", len(qa_execution_logs))
        for i, qa_log in enumerate(qa_execution_logs):
            qa_execution = self._convert_qa_log_to_node(qa_log, config_id, i)
            qa_executions.append(qa_execution)
            if i < 3:
                logger.info(
                    "   QA%s: %s -> F1=%.3f, question='%s...'",
                    i + 1,
                    qa_execution.qa_id,
                    qa_execution.f1_score,
                    qa_execution.question[:50],
                )
        logger.info("Converted QA logs successfully: %s QAExecutionNodes", len(qa_executions))
        
        # Add data into the graph-memory system.
        logger.info("Adding evaluation data into graph memory")
        config_node = self.graph_memory.add_complete_evaluation(params, qa_executions)
        logger.info("Graph memory update completed")
        
        # Debug: record post-update memory stats.
        memory_stats_after = self.graph_memory.get_memory_stats()
        logger.info("===== Post-record graph-memory state =====")
        logger.info(f"   Query Layer: {memory_stats_after['query_layer']['qa_executions']} QA executions (+{memory_stats_after['query_layer']['qa_executions'] - memory_stats_before['query_layer']['qa_executions']})")
        logger.info(f"   Config Layer: {memory_stats_after['config_layer']['configurations']} configurations (+{memory_stats_after['config_layer']['configurations'] - memory_stats_before['config_layer']['configurations']})")
        logger.info(f"                {memory_stats_after['config_layer']['config_relationships']} relationships (+{memory_stats_after['config_layer']['config_relationships'] - memory_stats_before['config_layer']['config_relationships']})")
        logger.info(f"   Insight Layer: {memory_stats_after['insight_layer']['insights']} insights (+{memory_stats_after['insight_layer']['insights'] - memory_stats_before['insight_layer']['insights']})")
        logger.info(f"                 {memory_stats_after['insight_layer']['insight_relationships']} relationships (+{memory_stats_after['insight_layer']['insight_relationships'] - memory_stats_before['insight_layer']['insight_relationships']})")
        logger.info("===== Finished post-record state summary =====")
        
        # Generate insights.
        existing_insights = list(self.graph_memory.insight_layer.nodes.values())
        logger.info("===== Starting insight generation =====")
        logger.info("Existing insights: %s", len(existing_insights))
        logger.info("Generating new insights for config %s", config_id)
        
        # Generate new insights.
        new_insights = self.insight_agent.extract_insights_from_evaluation(
            config_node, qa_executions, existing_insights
        )
        
        # Add new insights to the insight layer.
        if new_insights:
            logger.info("InsightAgent returned %s new insights", len(new_insights))
            for i, insight in enumerate(new_insights):
                logger.info("   Insight%s: %s (confidence: %.2f)", i + 1, insight.title, insight.confidence_score)
            
            self.graph_memory.insight_layer.add_insights(new_insights)
            logger.info("Added %s new insights to the insight layer", len(new_insights))
        else:
            logger.warning("InsightAgent did not generate any new insights")
        
        # Final memory-state summary.
        memory_stats_final = self.graph_memory.get_memory_stats()
        logger.info("===== Final graph-memory state =====")
        logger.info(f"   Total QAs: {memory_stats_final['query_layer']['qa_executions']}")
        logger.info(f"   Total Configs: {memory_stats_final['config_layer']['configurations']}")  
        logger.info(f"   Total Insights: {memory_stats_final['insight_layer']['insights']}")
        logger.info(f"   Config F1: {config_node.avg_f1_score:.4f}")
        logger.info(f"   New insights: {len(new_insights) if new_insights else 0}")
        logger.info("===== Finished final state summary =====")
        
        logger.info(
            "Complete evaluation recorded: Config F1=%.4f, %s new insights extracted",
            config_node.avg_f1_score,
            len(new_insights) if new_insights else 0,
        )
        logger.info("===== Finished recording into the three-layer graph-memory system =====")

    def _migrate_old_knowledge_base(self, old_kb: T.Dict[str, T.Any]):
        """Migrate an old knowledge-base format into the new three-layer system."""
        logger.info(f"🔄 Migrating old knowledge base...")
        
        try:
            configs = old_kb.get('configs', [])
            
            # Handle both dict-based and list-based legacy layouts.
            if isinstance(configs, dict):
                configs = list(configs.values())
            
            # Convert legacy records into the new format.
            for i, old_record in enumerate(configs):
                if isinstance(old_record, dict) and 'config' in old_record:
                    # Create a simplified QA execution record.
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
                    
                    # Add it to graph memory.
                    self.graph_memory.query_layer.add_qa_execution(qa_execution)
            
            logger.info(f"✅ Successfully migrated {len(configs)} old records to new format")
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate old knowledge base: {e}")
    
    def _convert_qa_log_to_node(self, qa_log: T.Dict[str, T.Any], config_id: str, index: int) -> QAExecutionNode:
        """Convert a QA execution log into a QAExecutionNode."""
        # Ensure the QA id is globally unique.
        current_qa_count = len(self.graph_memory.query_layer.nodes)
        global_qa_index = current_qa_count + index
        
        return QAExecutionNode(
            qa_id=f"{self.experiment_id}_qa_{global_qa_index}",
            config_id=config_id,
            
            # Basic QA fields.
            question=qa_log.get('question', 'Unknown question'),
            ground_truth_answer=qa_log.get('ground_truth', 'Unknown answer'),
            predicted_answer=qa_log.get('predicted_answer', 'Unknown prediction'),
            
            # Metrics.
            f1_score=qa_log.get('f1_score', 0.0),
            exact_match=qa_log.get('exact_match', False),
            retrieval_precision=qa_log.get('retrieval_precision', 0.0),
            retrieval_recall=qa_log.get('retrieval_recall', 0.0),
            context_overlap=qa_log.get('context_overlap', 0.0),
            answer_relevance=qa_log.get('answer_relevance', 0.0),
            
            # RAG pipeline details.
            raw_query=qa_log.get('raw_query', None),
            
            # Query decomposition.
            query_decomposition_enabled=qa_log.get('query_decomposition_enabled', False),
            decomposed_queries=qa_log.get('decomposed_queries', []),
            query_decomposition_llm=qa_log.get('query_decomposition_llm', None),
            decomposition_time=qa_log.get('decomposition_time', 0.0),
            
            # HyDE settings.
            hyde_enabled=qa_log.get('hyde_enabled', False),
            hyde_query=qa_log.get('hyde_query', None),
            hyde_llm=qa_log.get('hyde_llm', None),
            hyde_time=qa_log.get('hyde_time', 0.0),
            
            # Retrieval config.
            embedding_model=qa_log.get('embedding_model', 'unknown'),
            retrieval_method=qa_log.get('retrieval_method', 'unknown'),
            retrieval_top_k=qa_log.get('retrieval_top_k', 10),
            hybrid_bm25_weight=qa_log.get('hybrid_bm25_weight', 0.5),
            retrieval_time=qa_log.get('retrieval_time', 0.0),
            
            # Fusion settings.
            fusion_enabled=qa_log.get('fusion_enabled', False),
            fusion_mode=qa_log.get('fusion_mode', None),
            fusion_time=qa_log.get('fusion_time', 0.0),
            
            # Reranking.
            reranker_enabled=qa_log.get('reranker_enabled', False),
            reranker_llm=qa_log.get('reranker_llm', None),
            reranker_top_k=qa_log.get('reranker_top_k', 5),
            reranking_time=qa_log.get('reranking_time', 0.0),
            
            # Additional context.
            additional_context_enabled=qa_log.get('additional_context_enabled', False),
            additional_context_num_nodes=qa_log.get('additional_context_num_nodes', 0),
            additional_context_time=qa_log.get('additional_context_time', 0.0),
            
            # Final assembled output.
            final_context=qa_log.get('final_context', ''),
            context_assembly_time=qa_log.get('context_assembly_time', 0.0),
            
            # Few-shot
            few_shot_enabled=qa_log.get('few_shot_enabled', False),
            few_shot_examples=qa_log.get('few_shot_examples', []),
            few_shot_retrieval_time=qa_log.get('few_shot_retrieval_time', 0.0),
            
            # Response synthesis.
            response_synthesizer_llm=qa_log.get('response_synthesizer_llm', 'unknown'),
            template_name=qa_log.get('template_name', 'unknown'),
            final_prompt=qa_log.get('final_prompt', ''),
            synthesis_time=qa_log.get('synthesis_time', 0.0),
            
            total_execution_time=qa_log.get('execution_time', 0.0)
        )
    
    def save_knowledge_base(self, file_path: str):
        """Save the MCTS knowledge base to the requested file path."""
        logger.info("Saving graph memory to %s", file_path)
        
        try:
            # Graph memory has its own persistence mechanism.
            self.graph_memory.save_all_layers()
            
            # Also emit a legacy-compatible format for compatibility.
            legacy_kb_data = self._get_legacy_knowledge_base_format()
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(legacy_kb_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved compatibility knowledge base to %s", file_path)
            logger.info("Saved graph memory to %s", self.graph_memory.storage_path)
        except Exception as e:
            logger.error("Error while saving the knowledge base: %s", e, exc_info=True)

    def _get_legacy_knowledge_base_format(self) -> T.Dict[str, T.Any]:
        """Convert graph-memory data into the legacy knowledge-base format."""
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
            logger.error("Failed to convert to the legacy format: %s", e)
            return {
                'experiment_id': self.experiment_id,
                'configs': {},
                'metadata': {'total_configs': 0, 'total_explorations': 0}
            }
    
    @property
    def knowledge_base(self) -> T.Dict[str, T.Any]:
        """Backward-compatible `knowledge_base` property."""
        return self._get_legacy_knowledge_base_format()

    def _build_search_space_from_config(self):
        """Build the MCTS search space."""
        from .hierarchical_search import RAGSearchSpace
        
        # Use the default search space.
        return RAGSearchSpace()
    
    def _ensure_required_params(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        """Ensure all required parameters exist, using the same logic as TPE."""
        # Set the default rag_mode.
        final_params = {"rag_mode": "rag"}
        final_params.update(params)
        
        # Use the optimization config as the default source.
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
        
        # Normalize the chunk-size parameter.
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
        
        # Add conditional parameters.
        if final_params.get("retrieval_method") == "hybrid":
            defaults["rag_hybrid_bm25_weight"] = final_params.get("hybrid_bm25_weight", 0.5)
            
        # Query decomposition parameters (forced on by default).
        if final_params.get("query_decomposition_enabled", True):
            defaults["rag_query_decomposition_enabled"] = True
            defaults["rag_query_decomposition_num_queries"] = final_params.get("query_decomposition_num_queries", 4)
            defaults["rag_query_decomposition_llm_name"] = final_params.get("query_decomposition_llm", "Qwen2-7b")
            defaults["rag_fusion_mode"] = final_params.get("fusion_mode", "simple")
        else:
            defaults["rag_query_decomposition_enabled"] = False
            
        # HyDE parameters. Keep HyDE disabled here to reduce the search space.
        # # HyDE parameters (forced on)
        # if final_params.get("hyde_enabled", True):
        #     defaults["hyde_enabled"] = True
        #     defaults["hyde_llm_name"] = final_params.get("hyde_llm", "Qwen2-7b")
        # else:
        #     defaults["hyde_enabled"] = False
        defaults["hyde_enabled"] = False
            
        # Reranker parameters.
        defaults["reranker_enabled"] = True
        defaults["reranker_llm_name"] = final_params.get("reranker_llm", "Qwen2-7b")
        defaults["reranker_top_k"] = final_params.get("reranker_top_k", 5)
            
        # Additional-context parameters (forced on by default).
        if final_params.get("additional_context_enabled", True):
            defaults["additional_context_enabled"] = True
            defaults["additional_context_num_nodes"] = final_params.get("additional_context_num_nodes", 5)
        else:
            defaults["additional_context_enabled"] = False
            
        # Few-shot parameters.
        defaults["few_shot_enabled"] = False
        
        # Apply default values.
        for key, value in defaults.items():
            if key not in final_params:
                final_params[key] = value
                
        return final_params
    
    def _get_default_config(self) -> T.Dict[str, T.Any]:
        """Return the default configuration."""
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
            "hyde_enabled": False,  # Keep HyDE disabled by default.
            "hyde_llm": "Qwen2-7b",
            "reranker_enabled": True,
            "additional_context_enabled": True,
            "additional_context_num_nodes": 5,
            "response_synthesizer_llm": "Qwen2-7b",
            "template_name": "CoT",
            "few_shot_enabled": False,
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate the experiment identifier."""
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
        
    # Keep the legacy method for compatibility.
    def record_real_evaluation(self, params: T.Dict[str, T.Any], metrics: T.Dict[str, T.Any]):
        """Legacy method for backward compatibility"""
        logger.warning("⚠️ Using legacy record_real_evaluation method. Consider upgrading to record_complete_evaluation.")
        
        # Create a simplified QA execution log.
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

# # Keep backward compatibility
# class MCTSOptimizationEngine(EnhancedMCTSOptimizationEngine):
#     """Legacy MCTS optimization engine for backward compatibility"""
    
#     def __init__(self, api_key: str, api_base: str, experiment_id: str = None, 
#                  existing_knowledge_base: T.Dict[str, T.Any] = None):
#         logger.warning("⚠️ Using legacy MCTSOptimizationEngine. Consider upgrading to EnhancedMCTSOptimizationEngine.")
#         super().__init__(api_key, api_base, experiment_id, existing_knowledge_base)
