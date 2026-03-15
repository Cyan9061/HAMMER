"""

Three-Layer Graph Memory System for RAG-MCTS Optimization

This module implements the enhanced three-layer memory architecture:
1. Query Layer: Detailed QA execution records with complete RAG flow
2. Config Layer: Configuration performance analysis and relationships  
3. Insight Layer: Extracted insights and rules for optimization guidance
"""

import json
import time
import hashlib
import numpy as np
import networkx as nx
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from hammer.logger import logger

def embed_text(text: str) -> np.ndarray:
    """Simple text embedding using hash-based approach for consistency"""
    # Deterministic embedding based on text hash
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2**32))
    return np.random.normal(0, 1, 384)  # Standard embedding dimension

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norms if norms > 0 else 0.0

@dataclass
class QAExecutionNode:
    """Complete RAG execution record with all intermediate results"""
    qa_id: str
    config_id: str
    
    # Input information
    question: str
    ground_truth_answer: str
    ground_truth_context: List[str] = field(default_factory=list)
    
    # RAG Flow Intermediate Results
    raw_query: str = None
    
    # Step 1: Query Decomposition
    query_decomposition_enabled: bool = False
    decomposed_queries: List[str] = field(default_factory=list)
    query_decomposition_llm: str = None
    decomposition_time: float = 0.0
    
    # Step 2: HyDE Query Transform
    hyde_enabled: bool = False
    hyde_query: str = None
    hyde_llm: str = None
    hyde_time: float = 0.0
    
    # Step 3: Initial Retrieval
    embedding_model: str = None
    retrieval_method: str = None
    retrieval_top_k: int = None
    hybrid_bm25_weight: float = None
    initial_retrieved_docs: List[Dict] = field(default_factory=list)
    retrieval_time: float = 0.0
    
    # Step 4: Fusion
    fusion_enabled: bool = False
    fusion_mode: str = None
    fused_docs: List[Dict] = field(default_factory=list)
    fusion_time: float = 0.0
    
    # Step 5: Reranking
    reranker_enabled: bool = False
    reranker_llm: str = None
    reranker_top_k: int = None
    reranker_results: List[Dict] = field(default_factory=list)
    reranking_time: float = 0.0
    
    # Step 6: Additional Context
    additional_context_enabled: bool = False
    additional_context_num_nodes: int = 0
    additional_context_docs: List[Dict] = field(default_factory=list)
    additional_context_time: float = 0.0
    
    # Step 7: Final Context Assembly
    final_retrieved_docs: List[Dict] = field(default_factory=list)
    final_context: str = None
    context_assembly_time: float = 0.0
    
    # Step 8: Few-shot Examples
    few_shot_enabled: bool = False
    few_shot_examples: List[str] = field(default_factory=list)
    few_shot_retrieval_time: float = 0.0
    
    # Step 9: Response Synthesis
    response_synthesizer_llm: str = None
    template_name: str = None
    final_prompt: str = None
    predicted_answer: str = None
    synthesis_time: float = 0.0
    
    # Performance Metrics
    f1_score: float = 0.0
    exact_match: bool = False
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    context_overlap: float = 0.0
    answer_relevance: float = 0.0
    
    # Execution metadata
    total_execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def extract_execution_pattern(self) -> str:
        """Extract structured execution pattern description"""
        pattern_parts = []
        
        if self.query_decomposition_enabled:
            pattern_parts.append(f"Decomposed into {len(self.decomposed_queries)} sub-queries")
        
        if self.hyde_enabled:
            pattern_parts.append("Applied HyDE query transformation")
            
        retrieval_desc = f"Retrieved {len(self.initial_retrieved_docs)} docs via {self.retrieval_method}"
        if self.retrieval_method == "hybrid":
            retrieval_desc += f" (BM25 weight: {self.hybrid_bm25_weight})"
        pattern_parts.append(retrieval_desc)
        
        if self.fusion_enabled:
            pattern_parts.append(f"Fused results using {self.fusion_mode}")
            
        if self.reranker_enabled:
            pattern_parts.append(f"Reranked to top-{self.reranker_top_k} docs")
            
        if self.additional_context_enabled:
            pattern_parts.append(f"Added {self.additional_context_num_nodes} context nodes")
            
        pattern_parts.append(f"Synthesized using {self.template_name} template")
        
        return " → ".join(pattern_parts)

    def compute_semantic_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for similarity search"""
        return {
            'question_embedding': embed_text(self.question),
            'pattern_embedding': embed_text(self.extract_execution_pattern()),
            'context_embedding': embed_text(self.final_context or ""),
            'config_embedding': embed_text(self.extract_config_signature())
        }
    
    def extract_config_signature(self) -> str:
        """Extract configuration signature for similarity matching"""
        config_elements = [
            f"retrieval={self.retrieval_method}",
            f"embedding={self.embedding_model.split('/')[-1] if self.embedding_model else 'unknown'}",
            f"template={self.template_name}",
            f"reranker={'enabled' if self.reranker_enabled else 'disabled'}",
            f"hyde={'enabled' if self.hyde_enabled else 'disabled'}"
        ]
        return " | ".join(config_elements)

@dataclass
class ConfigNode:
    """配置节点：存储配置和性能分析"""
    config_id: str
    config_params: Dict[str, Any]
    
    # 聚合性能指标
    avg_f1_score: float
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_execution_time: float
    
    # 详细性能分析
    question_type_performance: Dict[str, float] = field(default_factory=dict)
    component_analysis: Dict[str, float] = field(default_factory=dict)
    
    # 关联的QA执行记录
    qa_execution_ids: List[str] = field(default_factory=list)
    
    # 统计信息
    total_evaluations: int = 0
    evaluation_timestamps: List[str] = field(default_factory=list)
    
    def compute_config_signature(self) -> str:
        """计算配置语义签名，包含全面参数"""
        key_params = {
            # 核心组件
            'splitter_method': self.config_params.get('splitter_method'),
            'retrieval_method': self.config_params.get('retrieval_method'), 
            'embedding_model': self.config_params.get('embedding_model'),
            'template_name': self.config_params.get('template_name'),
            
            # RAG增强
            'reranker_enabled': self.config_params.get('reranker_enabled'),
            'hyde_enabled': self.config_params.get('hyde_enabled'),
            
            # 检索详情
            'retrieval_top_k': self.config_params.get('retrieval_top_k'),
            'hybrid_bm25_weight': self.config_params.get('hybrid_bm25_weight'),
            
            # 查询处理
            'query_decomposition_enabled': self.config_params.get('query_decomposition_enabled'),
            'query_decomposition_num_queries': self.config_params.get('query_decomposition_num_queries'),
            'fusion_mode': self.config_params.get('fusion_mode'),
            
            # 上下文和分块
            'splitter_chunk_size': self.config_params.get('splitter_chunk_size'),
            'splitter_overlap': self.config_params.get('splitter_overlap'),
            'additional_context_num_nodes': self.config_params.get('additional_context_num_nodes'),
            
            # 模型
            'response_synthesizer_llm': self.config_params.get('response_synthesizer_llm'),
        }
        
        # 清理None值使签名更可读
        clean_params = {k: v for k, v in key_params.items() if v is not None}
        return json.dumps(clean_params, sort_keys=True)

@dataclass
class InsightNode:
    """Insight node: stores extracted insights and rules"""
    insight_id: str
    insight_type: str  # 'parameter_pattern', 'performance_rule', 'failure_mode'
    
    # Insight content
    title: str
    description: str
    confidence_score: float  # 0-1, based on supporting evidence
    
    # Supporting evidence
    supporting_config_ids: List[str] = field(default_factory=list)
    supporting_qa_ids: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Application guidance
    recommendation: str = ""
    applicable_conditions: List[str] = field(default_factory=list)
    
    # Metadata
    discovery_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    last_updated: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    validation_count: int = 0  # Number of times validated by subsequent data

class QueryLayer:
    """Query layer: manages all QA execution records"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.nodes: Dict[str, QAExecutionNode] = {}
        # self.similarity_graph = nx.Graph()
        
        self._load_from_disk()
    
    def add_qa_execution(self, qa_execution: QAExecutionNode):
        """Add new QA execution record with config_id validation"""
        # 🔥 验证config_id是否正确设置
        if not qa_execution.config_id:
            logger.error(f"❌ QA execution {qa_execution.qa_id} missing config_id")
            raise ValueError(f"QA execution {qa_execution.qa_id} must have a valid config_id")
        
        # 检查是否已存在相同的QA记录
        if qa_execution.qa_id in self.nodes:
            existing_qa = self.nodes[qa_execution.qa_id]
            if existing_qa.config_id != qa_execution.config_id:
                logger.warning(f"⚠️ QA {qa_execution.qa_id} config_id changed: "
                            f"{existing_qa.config_id} -> {qa_execution.config_id}")
            else:
                logger.info(f"🔄 Updating existing QA {qa_execution.qa_id}")
        
        self.nodes[qa_execution.qa_id] = qa_execution
        # self._update_similarity_graph(qa_execution)
        # self._save_to_disk()
        
        logger.debug(f"📝 Added QA execution: {qa_execution.qa_id} -> config: {qa_execution.config_id}")
    
    # def _update_similarity_graph(self, new_qa: QAExecutionNode):
    #     """Update similarity graph: find similar historical QAs"""
    #     self.similarity_graph.add_node(new_qa.qa_id)
        
    #     new_embeddings = new_qa.compute_semantic_embeddings()
        
    #     for existing_qa_id, existing_qa in self.nodes.items():
    #         if existing_qa_id != new_qa.qa_id:
    #             existing_embeddings = existing_qa.compute_semantic_embeddings()
                
    #             # Compute combined similarity
    #             question_sim = cosine_similarity(
    #                 new_embeddings['question_embedding'],
    #                 existing_embeddings['question_embedding']
    #             )
    #             pattern_sim = cosine_similarity(
    #                 new_embeddings['pattern_embedding'],
    #                 existing_embeddings['pattern_embedding']
    #             )
                
    #             combined_sim = 0.6 * question_sim + 0.4 * pattern_sim
                
    #             if combined_sim > 0.7:  # Similarity threshold
    #                 self.similarity_graph.add_edge(
    #                     new_qa.qa_id, existing_qa_id,
    #                     weight=combined_sim,
    #                     relation_type='semantic_similar'
    #                 )
    
    def find_similar_executions(self, query_embedding: np.ndarray, top_k: int = 10) -> List[QAExecutionNode]:
        """Find similar QA executions based on query embedding"""
        similarities = []
        for qa_id, qa_node in self.nodes.items():
            qa_embeddings = qa_node.compute_semantic_embeddings()
            sim = cosine_similarity(query_embedding, qa_embeddings['question_embedding'])
            similarities.append((sim, qa_node))
        
        similarities.sort(reverse=True)
        return [qa_node for _, qa_node in similarities[:top_k]]
    
    def _save_to_disk(self):
        """Save query layer to disk"""
        query_data = {
            'nodes': {qa_id: asdict(qa) for qa_id, qa in self.nodes.items()},
            # 'similarity_edges': list(self.similarity_graph.edges(data=True))
        }
        
        query_file = self.storage_path / "query_layer.json"
        with open(query_file, 'w', encoding='utf-8') as f:
            json.dump(query_data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """Load query layer from disk"""
        query_file = self.storage_path / "query_layer.json"
        if query_file.exists():
            try:
                with open(query_file, 'r', encoding='utf-8') as f:
                    query_data = json.load(f)
                
                # Restore nodes
                for qa_id, qa_dict in query_data.get('nodes', {}).items():
                    self.nodes[qa_id] = QAExecutionNode(**qa_dict)
                
                # Restore similarity graph
                # for edge_data in query_data.get('similarity_edges', []):
                #     if len(edge_data) == 3:
                #         node1, node2, attrs = edge_data
                #         self.similarity_graph.add_edge(node1, node2, **attrs)
                
                logger.info(f"✅ Loaded {len(self.nodes)} QA executions from disk")
            except Exception as e:
                logger.error(f"❌ Failed to load query layer: {e}")

class ConfigLayer:
    """Config layer: manages configuration relationships and patterns - 修复版"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.nodes: Dict[str, ConfigNode] = {}
        self.config_graph = nx.DiGraph()
        
        self._load_from_disk()
    
    def add_config_evaluation(self, config_params: Dict, qa_executions: List[QAExecutionNode]):
        """
        🔥 核心修复：累积更新ConfigNode而不是覆盖
        """
        config_id = self._generate_config_id(config_params)
        
        # 计算新的聚合指标
        new_avg_f1 = np.mean([qa.f1_score for qa in qa_executions])
        new_avg_precision = np.mean([qa.retrieval_precision for qa in qa_executions])
        new_avg_recall = np.mean([qa.retrieval_recall for qa in qa_executions])
        new_avg_time = np.mean([qa.total_execution_time for qa in qa_executions])
        
        # 分析问题类型性能
        new_question_type_perf = self._analyze_question_type_performance(qa_executions)
        
        new_qa_ids = [qa.qa_id for qa in qa_executions]
        new_timestamps = [qa.timestamp for qa in qa_executions]
        
        if config_id in self.nodes:
            # 🔥 关键修复：累积更新现有ConfigNode
            existing_config = self.nodes[config_id]
            
            logger.info(f"📊 Updating existing config {config_id}: "
                       f"adding {len(qa_executions)} new QA executions to existing {existing_config.total_evaluations}")
            
            # 累积QA执行记录
            all_qa_ids = existing_config.qa_execution_ids + new_qa_ids
            all_timestamps = existing_config.evaluation_timestamps + new_timestamps
            
            # 重新计算加权平均指标
            old_count = existing_config.total_evaluations
            new_count = len(qa_executions)
            total_count = old_count + new_count
            
            # 加权平均计算
            combined_avg_f1 = (existing_config.avg_f1_score * old_count + new_avg_f1 * new_count) / total_count
            combined_avg_precision = (existing_config.avg_retrieval_precision * old_count + new_avg_precision * new_count) / total_count
            combined_avg_recall = (existing_config.avg_retrieval_recall * old_count + new_avg_recall * new_count) / total_count
            combined_avg_time = (existing_config.avg_execution_time * old_count + new_avg_time * new_count) / total_count
            
            # 合并问题类型性能
            combined_question_type_perf = self._merge_question_type_performance(
                existing_config.question_type_performance, new_question_type_perf, old_count, new_count
            )
            
            # 更新现有ConfigNode
            existing_config.avg_f1_score = combined_avg_f1
            existing_config.avg_retrieval_precision = combined_avg_precision
            existing_config.avg_retrieval_recall = combined_avg_recall
            existing_config.avg_execution_time = combined_avg_time
            existing_config.question_type_performance = combined_question_type_perf
            existing_config.qa_execution_ids = all_qa_ids
            existing_config.total_evaluations = total_count
            existing_config.evaluation_timestamps = all_timestamps
            
            config_node = existing_config
            
        else:
            # 🔥 创建新的ConfigNode
            logger.info(f"📊 Creating new config {config_id} with {len(qa_executions)} QA executions")
            
            config_node = ConfigNode(
                config_id=config_id,
                config_params=config_params,
                avg_f1_score=new_avg_f1,
                avg_retrieval_precision=new_avg_precision,
                avg_retrieval_recall=new_avg_recall,
                avg_execution_time=new_avg_time,
                question_type_performance=new_question_type_perf,
                qa_execution_ids=new_qa_ids,
                total_evaluations=len(qa_executions),
                evaluation_timestamps=new_timestamps
            )
            
            self.nodes[config_id] = config_node
        
        # 更新配置关系图
        self._update_config_relationships(config_node)
        self._save_to_disk()
        
        logger.info(f"✅ Config evaluation recorded: {config_id}, "
                   f"Total F1={config_node.avg_f1_score:.3f}, Total QAs={config_node.total_evaluations}")
        
        return config_node
    
    def _merge_question_type_performance(self, existing_perf: Dict[str, float], 
                                        new_perf: Dict[str, float], 
                                        old_count: int, new_count: int) -> Dict[str, float]:
        """合并问题类型性能指标"""
        merged = {}
        
        all_types = set(existing_perf.keys()) | set(new_perf.keys())
        
        for qtype in all_types:
            existing_score = existing_perf.get(qtype, 0.0)
            new_score = new_perf.get(qtype, 0.0)
            
            if qtype in existing_perf and qtype in new_perf:
                # 两边都有，计算加权平均
                merged[qtype] = (existing_score * old_count + new_score * new_count) / (old_count + new_count)
            elif qtype in existing_perf:
                # 只有旧的有
                merged[qtype] = existing_score
            else:
                # 只有新的有
                merged[qtype] = new_score
        
        return merged
    
    def _analyze_question_type_performance(self, qa_executions: List[QAExecutionNode]) -> Dict[str, float]:
        """分析问题类型性能"""
        type_scores = defaultdict(list)
        
        for qa in qa_executions:
            question_type = self._classify_question_type(qa.question)
            type_scores[question_type].append(qa.f1_score)
        
        return {qtype: np.mean(scores) for qtype, scores in type_scores.items()}
    
    def _classify_question_type(self, question: str) -> str:
        """简单问题类型分类"""
        question_lower = question.lower()
        if 'when' in question_lower and ('die' in question_lower or 'death' in question_lower):
            return 'death_date'
        elif 'who' in question_lower and ('mother' in question_lower or 'father' in question_lower):
            return 'family_relation'
        elif 'where' in question_lower:
            return 'location'
        elif 'what' in question_lower:
            return 'definition'
        elif 'are' in question_lower and 'both' in question_lower:
            return 'comparison'
        else:
            return 'other'
    
    def _update_config_relationships(self, new_config: ConfigNode):
        """Update the configuration relationship graph."""
        logger.info("===== Starting config-relationship graph update =====")
        logger.info("Creating relationships for new config %s", new_config.config_id)
        logger.info("Existing config count: %s", len(self.nodes))
        
        self.config_graph.add_node(new_config.config_id, **asdict(new_config))
        logger.info("Added config node to the graph")
        
        # Find similar configs and connect them with edges.
        relationships_added = 0
        logger.info("Starting similarity analysis")
        
        for existing_id, existing_config in self.nodes.items():
            if existing_id != new_config.config_id:
                similarity = self._compute_config_similarity(new_config, existing_config)
                logger.info("   Similarity to %s...: %.3f", existing_id[:8], similarity)
                
                if similarity > 0.5:
                    performance_delta = new_config.avg_f1_score - existing_config.avg_f1_score
                    self.config_graph.add_edge(
                        existing_id, new_config.config_id,
                        weight=similarity,
                        relation_type='parameter_similar',
                        performance_delta=performance_delta
                    )
                    relationships_added += 1
                    logger.info(
                        "   Added relationship: similarity=%.3f, performance_delta=%+.3f",
                        similarity,
                        performance_delta,
                    )
                    
                    # Summarize key parameter differences.
                    key_diffs = []
                    for key in ['retrieval_method', 'template_name', 'reranker_enabled', 'hyde_enabled']:
                        val1 = new_config.config_params.get(key)
                        val2 = existing_config.config_params.get(key) 
                        if val1 != val2:
                            key_diffs.append(f"{key}: {val2}->{val1}")
                    if key_diffs:
                        logger.info("     Key differences: %s", ", ".join(key_diffs))
                else:
                    logger.debug("   Similarity too low; no relationship added")
        
        logger.info("Relationship update completed: %s new edges", relationships_added)
        logger.info("Total relationship count: %s", self.config_graph.number_of_edges())
        logger.info("===== Config-relationship graph update finished =====")
    
    def _compute_config_similarity(self, config1: ConfigNode, config2: ConfigNode) -> float:
        """
        🔥 核心方法：基于参数匹配的配置相似度计算
        确保结果始终为非负数
        """
        params1, params2 = config1.config_params, config2.config_params
        
        # 🔥 扩展的参数权重设置
        key_param_weights = {
            # 核心组件权重
            'splitter_method': 0.15,
            'retrieval_method': 0.15,
            'embedding_model': 0.12,
            'template_name': 0.12,
            
            # 数值参数权重（需要特殊处理）
            'retrieval_top_k': 0.08,
            'splitter_chunk_size': 0.08,
            'splitter_overlap': 0.06,
            
            # 布尔参数权重
            'reranker_enabled': 0.08,
            'hyde_enabled': 0.08,
            'query_decomposition_enabled': 0.08
        }
        
        similarity = 0.0
        
        for param, weight in key_param_weights.items():
            if param in params1 and param in params2:
                val1, val2 = params1[param], params2[param]
                
                if param in ['retrieval_top_k', 'splitter_chunk_size']:
                    # 数值参数：基于相对差异计算相似度
                    if val1 == val2:
                        similarity += weight
                    elif val1 and val2:
                        diff = abs(val1 - val2) / max(val1, val2)
                        similarity += weight * max(0, 1 - diff)
                
                elif param in ['splitter_overlap']:
                    # 浮点参数：基于绝对差异计算相似度
                    if abs(val1 - val2) < 0.01:  # 认为基本相等
                        similarity += weight
                    else:
                        diff = abs(val1 - val2)
                        similarity += weight * max(0, 1 - diff * 2)  # 差异0.5以内有部分相似度
                
                else:
                    # 字符串和布尔参数：精确匹配
                    if val1 == val2:
                        similarity += weight
        
        # 确保相似度在[0, 1]范围内
        return max(0.0, min(1.0, similarity))
    
    def _generate_config_id(self, config_params: Dict) -> str:
        """生成唯一配置ID"""
        config_str = json.dumps(config_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _save_to_disk(self):
        """保存配置层到磁盘"""
        config_data = {
            'nodes': {config_id: asdict(config) for config_id, config in self.nodes.items()},
            'graph_edges': list(self.config_graph.edges(data=True))
        }
        
        config_file = self.storage_path / "config_layer.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """从磁盘加载配置层"""
        config_file = self.storage_path / "config_layer.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 恢复节点
                for config_id, config_dict in config_data.get('nodes', {}).items():
                    self.nodes[config_id] = ConfigNode(**config_dict)
                
                # 恢复图
                for edge_data in config_data.get('graph_edges', []):
                    if len(edge_data) == 3:
                        node1, node2, attrs = edge_data
                        self.config_graph.add_edge(node1, node2, **attrs)
                
                logger.info(f"✅ Loaded {len(self.nodes)} config nodes from disk")
            except Exception as e:
                logger.error(f"❌ Failed to load config layer: {e}")

class InsightLayer:
    """Insight layer: manages insights and rules"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.nodes: Dict[str, InsightNode] = {}
        self.insight_graph = nx.DiGraph()
        
        self._load_from_disk()
    
    def add_insights(self, new_insights: List[InsightNode]):
        """Add new insights and update relationship graph"""
        for insight in new_insights:
            self.nodes[insight.insight_id] = insight
            self._update_insight_relationships(insight)
        
        self._save_to_disk()
        logger.info(f"🧠 Added {len(new_insights)} new insights")
    
    def _update_insight_relationships(self, new_insight: InsightNode):
        """Update insight relationship graph"""
        self.insight_graph.add_node(new_insight.insight_id, **asdict(new_insight))
        
        # Find related insights and establish relationships
        for existing_id, existing_insight in self.nodes.items():
            if existing_id != new_insight.insight_id:
                relationship = self._detect_insight_relationship(new_insight, existing_insight)
                if relationship:
                    self.insight_graph.add_edge(
                        existing_id, new_insight.insight_id,
                        relation_type=relationship['type'],
                        strength=relationship['strength']
                    )
    
    def _detect_insight_relationship(self, insight1: InsightNode, insight2: InsightNode) -> Optional[Dict]:
        """Detect relationship between two insights"""
        # Simple keyword-based relationship detection
        words1 = set(insight1.description.lower().split())
        words2 = set(insight2.description.lower().split())
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union > 0:
            jaccard_sim = overlap / union
            if jaccard_sim > 0.3:
                return {
                    'type': 'semantic_related',
                    'strength': jaccard_sim
                }
        
        return None
    
    def query_relevant_insights(self, config_params: Dict[str, Any], 
                              question_type: str = None) -> List[InsightNode]:
        """Query insights relevant to given configuration and question type"""
        relevant_insights = []
        
        for insight in self.nodes.values():
            relevance = self._compute_insight_relevance(insight, config_params, question_type)
            if relevance > 0.5:
                relevant_insights.append((relevance, insight))
        
        relevant_insights.sort(reverse=True)
        return [insight for _, insight in relevant_insights[:5]]
    
    def _compute_insight_relevance(self, insight: InsightNode, config_params: Dict[str, Any], 
                                 question_type: str = None) -> float:
        """Compute insight relevance score"""
        relevance = insight.confidence_score * 0.5  # Base relevance from confidence
        
        # Check if insight applies to the configuration
        insight_text = insight.description.lower()
        
        # Configuration-based relevance
        if config_params.get('retrieval_method') == 'hybrid' and 'hybrid' in insight_text:
            relevance += 0.3
        if config_params.get('template_name') == 'CoT' and 'cot' in insight_text:
            relevance += 0.2
        if config_params.get('reranker_enabled') and 'rerank' in insight_text:
            relevance += 0.2
        
        # Question type relevance
        if question_type and question_type in insight_text:
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _save_to_disk(self):
        """Save insight layer to disk"""
        insight_data = {
            'nodes': {insight_id: asdict(insight) for insight_id, insight in self.nodes.items()},
            'graph_edges': list(self.insight_graph.edges(data=True))
        }
        
        insight_file = self.storage_path / "insight_layer.json"
        with open(insight_file, 'w', encoding='utf-8') as f:
            json.dump(insight_data, f, ensure_ascii=False, indent=2)
    
    def _load_from_disk(self):
        """Load insight layer from disk"""
        insight_file = self.storage_path / "insight_layer.json"
        if insight_file.exists():
            try:
                with open(insight_file, 'r', encoding='utf-8') as f:
                    insight_data = json.load(f)
                
                # Restore nodes
                for insight_id, insight_dict in insight_data.get('nodes', {}).items():
                    self.nodes[insight_id] = InsightNode(**insight_dict)
                
                # Restore graph
                for edge_data in insight_data.get('graph_edges', []):
                    if len(edge_data) == 3:
                        node1, node2, attrs = edge_data
                        self.insight_graph.add_edge(node1, node2, **attrs)
                
                logger.info(f"✅ Loaded {len(self.nodes)} insights from disk")
            except Exception as e:
                logger.error(f"❌ Failed to load insight layer: {e}")

class GraphMemoryRAGMCTS:
    """Complete three-layer graph memory system"""
    
    def __init__(self, storage_path: str = "Experiment/graph_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize three layers
        self.query_layer = QueryLayer(str(self.storage_path / "query"))
        self.config_layer = ConfigLayer(str(self.storage_path / "config"))
        self.insight_layer = InsightLayer(str(self.storage_path / "insight"))
        
        logger.info("Initialized Graph Memory System at %s", self.storage_path)
        logger.info(
            "Current state: %s QAs, %s configs, %s insights",
            len(self.query_layer.nodes),
            len(self.config_layer.nodes),
            len(self.insight_layer.nodes),
        )
        
    def save_all_layers(self):
        """Save all three layers to disk."""
        logger.info("===== Saving all three graph-memory layers =====")
        logger.info("Pre-save stats: %s", self.get_memory_stats())
        
        try:
            # Save Query Layer.
            logger.info("Saving Query Layer to %s", self.query_layer.storage_path)
            qa_count_before = len(self.query_layer.nodes)
            self.query_layer._save_to_disk()
            logger.info("Saved Query Layer: %s QA executions", qa_count_before)
            
            # Save Config Layer.
            logger.info("Saving Config Layer to %s", self.config_layer.storage_path)
            config_count_before = len(self.config_layer.nodes)
            relationship_count_before = self.config_layer.config_graph.number_of_edges()
            self.config_layer._save_to_disk()
            logger.info("Saved Config Layer: %s configurations, %s relationships", config_count_before, relationship_count_before)
            
            # Save Insight Layer.
            logger.info("Saving Insight Layer to %s", self.insight_layer.storage_path)
            insight_count_before = len(self.insight_layer.nodes)
            insight_relationship_count_before = self.insight_layer.insight_graph.number_of_edges()
            self.insight_layer._save_to_disk()
            logger.info("Saved Insight Layer: %s insights, %s relationships", insight_count_before, insight_relationship_count_before)
            
            logger.info(f"💾 All three layers saved to {self.storage_path}")
            logger.info("===== Finished saving all three graph-memory layers =====")
        except Exception as e:
            logger.error(f"❌ Failed to save all layers: {e}")
            logger.error("Storage path: %s", self.storage_path)
            logger.error("Query Layer path: %s", self.query_layer.storage_path)
            logger.error("Config Layer path: %s", self.config_layer.storage_path)
            logger.error("Insight Layer path: %s", self.insight_layer.storage_path)
            raise

    def add_complete_evaluation(self, config_params: Dict[str, Any],  
                                qa_executions: List[QAExecutionNode]) -> ConfigNode:
        """Add complete evaluation results with enhanced validation"""
        
        # Generate config_id.
        logger.info("Generating config_id from params: %s", config_params)
        standardized_config_id = self.config_layer._generate_config_id(config_params)
        
        # Ensure every QA execution carries the standardized config id.
        for qa_execution in qa_executions:
            if qa_execution.config_id != standardized_config_id:
                logger.warning(f"⚠️ Correcting QA {qa_execution.qa_id} config_id: "
                            f"{qa_execution.config_id} -> {standardized_config_id}")
                qa_execution.config_id = standardized_config_id
        
        # 1. Add records to the query layer.
        for i, qa_execution in enumerate(qa_executions):
            try:
                self.query_layer.add_qa_execution(qa_execution)
            except Exception as e:
                logger.error(f"❌ Error adding QA execution at index {i}: {qa_execution.qa_id}")
                logger.error(f"Error details: {e}")
                # Skip malformed QA records and continue.
                continue
        self.query_layer._save_to_disk()
        # 2. Add/update the config layer.
        config_node = self.config_layer.add_config_evaluation(config_params, qa_executions)
        
        logger.info(f"✅ Added complete evaluation: config_id={config_node.config_id}, "
                f"{len(qa_executions)} QAs, avg_F1={config_node.avg_f1_score:.3f}")
        
        return config_node

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        return {
            'query_layer': {
                'qa_executions': len(self.query_layer.nodes),
                # 'similarity_edges': self.query_layer.similarity_graph.number_of_edges()
            },
            'config_layer': {
                'configurations': len(self.config_layer.nodes),
                'config_relationships': self.config_layer.config_graph.number_of_edges()
            },
            'insight_layer': {
                'insights': len(self.insight_layer.nodes),
                'insight_relationships': self.insight_layer.insight_graph.number_of_edges()
            }
        }

    def validate_config_qa_consistency(self):
        """验证配置和QA记录的一致性"""
        logger.info("🔍 Validating config-QA consistency...")
        
        total_config_qas = 0
        total_query_qas = len(self.query_layer.nodes)
        
        for config_id, config_node in self.config_layer.nodes.items():
            total_config_qas += len(config_node.qa_execution_ids)
            
            # 检查QA记录是否存在
            missing_qas = []
            for qa_id in config_node.qa_execution_ids:
                if qa_id not in self.query_layer.nodes:
                    missing_qas.append(qa_id)
                else:
                    # 检查config_id是否匹配
                    qa_node = self.query_layer.nodes[qa_id]
                    if qa_node.config_id != config_id:
                        logger.warning(f"⚠️ Config-QA mismatch: QA {qa_id} "
                                    f"has config_id {qa_node.config_id} but linked to {config_id}")
            
            if missing_qas:
                logger.warning(f"⚠️ Config {config_id} references missing QAs: {missing_qas}")
        
        logger.info(f"✅ Validation complete: {len(self.config_layer.nodes)} configs, "
                f"{total_config_qas} config-referenced QAs, {total_query_qas} total QAs")
