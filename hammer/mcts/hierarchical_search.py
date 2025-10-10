"""
MCTS搜索实现
"""
import json
import math
import random
import time
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from typing import Dict, List, Optional, Any, Tuple, Callable

from hammer.logger import logger
import numpy as np
import pandas as pd

generative_LLM = "Qwen2-7b"#"Qwen/Qwen2-7B-Instruct"#"DeepSeek-R1-32b"

def _load_embedding_models_from_env() -> List[str]:
    """从环境变量加载embedding models"""
    # 从.env文件读取Optional_EMBED_MODEL
    env_models = []
    
    # 读取Optional_EMBED_MODEL环境变量
    optional_embed_model = os.getenv('Optional_EMBED_MODEL', '').strip()
    if optional_embed_model:
        # 如果包含多个模型，用逗号分隔
        env_models = [model.strip() for model in optional_embed_model.split(',') if model.strip()]

    return env_models

# 🔥 全局控制变量：强制启用查询分解功能
query_decomposition_on = True

class ParameterLayer(Enum):
    """RAG参数分层结构 - 保持不变"""
    # 第1层：文本处理
    SPLITTER_METHOD = "splitter_method"
    SPLITTER_CHUNK_SIZE = "splitter_chunk_size"
    SPLITTER_OVERLAP = "splitter_overlap"
    
    # 第2层：嵌入模型
    EMBEDDING_MODEL = "embedding_model"
    
    # 第3层：检索方法
    RETRIEVAL_METHOD = "retrieval_method"
    RETRIEVAL_TOP_K = "retrieval_top_k"
    
    # 第4层：混合检索权重
    HYBRID_BM25_WEIGHT = "hybrid_bm25_weight"
    
    # 第5层：查询分解
    QUERY_DECOMPOSITION_NUM_QUERIES = "query_decomposition_num_queries"
    QUERY_DECOMPOSITION_LLM = "query_decomposition_llm"
    FUSION_MODE = "fusion_mode"
    
    # 第6层：可选组件
    # HYDE_LLM = "hyde_llm"  # 🔥 关闭HyDE：从搜索空间中移除
    RERANKER_LLM = "reranker_models"
    RERANKER_TOP_K = "reranker_top_k"
    ADDITIONAL_CONTEXT_NUM_NODES = "additional_context_num_nodes"
    
    # 第7层：响应合成
    RESPONSE_SYNTHESIZER_LLM = "response_synthesizer_llm"
    
    # 第8层：模板选择
    TEMPLATE_NAME = "template_name"
    
    # 第9层：Few-shot配置
    FEW_SHOT_EMBEDDING_MODEL = "few_shot_embedding_model"
    FEW_SHOT_TOP_K = "few_shot_top_k"

@dataclass
class ParameterChoice:
    """单个参数选择"""
    layer: ParameterLayer
    name: str
    value: Any
    
    def __hash__(self):
        return hash((self.layer, self.name, str(self.value)))

@dataclass
class RAGSearchSpace:
    """RAG参数搜索空间定义 - 保持不变"""
    
    # 文本分割器选项 - 🔧 修复：重叠比例与TPE对齐
    splitter_methods: List[str] = field(default_factory=lambda: ["recursive", "sentence", "token"])
    splitter_chunk_sizes: List[int] = field(default_factory=lambda: [2**i for i in range(7, 11)])
    # 🔧 修复：overlap范围从0.55改为0.5以与TPE对齐
    splitter_overlaps: List[float] = field(default_factory=lambda: [round(x, 2) for x in np.arange(0.0, 0.51, 0.05)])
    
    # 嵌入模型 - 🔧 从环境变量加载，与TPE配置完全对齐
    embedding_models: List[str] = field(default_factory=_load_embedding_models_from_env)
    #  thenlper-gte-base
    # 检索配置
    retrieval_methods: List[str] = field(default_factory=lambda: ["dense", "sparse", "hybrid"])#dense
    retrieval_top_k_options: List[int] = field(default_factory=lambda: list(range(1, 11, 1)))
    hybrid_bm25_weights: List[float] = field(default_factory=lambda: [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)])
    
    # 查询分解
    query_decomposition_num_queries_options: List[int] = field(default_factory=lambda: [2, 3, 4, 5])

    query_decomposition_llms: List[str] = field(default_factory=lambda: ["Qwen2-7b"])  # 🔥 添加Qwen2-7b支持
    fusion_modes: List[str] = field(default_factory=lambda: ["simple", "reciprocal_rerank", "relative_score", "dist_based_score"])
    
    # 增强模块 - 🔧 与TPE配置完全对齐
    # hyde_llms: List[str] = field(default_factory=lambda: ["Qwen2-7b"])  # 🔥 关闭HyDE：移除HyDE LLM选项
    reranker_llms: List[str] = field(default_factory=lambda: [
        # 🔑 与TPE完全对齐的增强Reranker模型（7种）
        "flashrank",           # 快速ONNX推理
        "TransformerRanker",   # 通用Transformer
        "MonoT5",             # MonoT5重排序 
        "RankT5",             # RankT5变体
        "MonoBERT",           # BERT重排序
        "InRanker",           # InRanker模型
        "EchoRank",           # EchoRank模型
    ])
    # 🔧 修复：Reranker top_k与TPE对齐 (2-32, step=1)
    reranker_top_k_options: List[int] = field(default_factory=lambda: list(range(2, 33)))
    additional_context_num_nodes_options: List[int] = field(default_factory=lambda: list(range(2, 21, 2)))
    
    # 输出配置  
    response_synthesizer_llms: List[str] = field(default_factory=lambda: ["Qwen2-7b"])  # 🔥 添加Qwen2-7b选项
    # 🔧 模板顺序与TPE配置对齐：['default', 'concise', 'CoT']
    template_names: List[str] = field(default_factory=lambda: ["default", "concise", "CoT"])
    
    # Few-shot 配置
    few_shot_embedding_models: List[str] = field(default_factory=list)
    few_shot_top_k_options: List[int] = field(default_factory=list)

    def get_layer_choices(self, layer: ParameterLayer, current_params: Dict[str, Any]) -> List[ParameterChoice]:
        """获取指定层的所有选择 - 保持不变"""
        choices = []
        
        if layer == ParameterLayer.SPLITTER_METHOD:
            for method in self.splitter_methods:
                choices.append(ParameterChoice(layer, "splitter_method", method))
        elif layer == ParameterLayer.SPLITTER_CHUNK_SIZE:
            for size in self.splitter_chunk_sizes:
                choices.append(ParameterChoice(layer, "splitter_chunk_size", size))
        elif layer == ParameterLayer.SPLITTER_OVERLAP:
            for overlap in self.splitter_overlaps:
                choices.append(ParameterChoice(layer, "splitter_overlap", overlap))
        elif layer == ParameterLayer.EMBEDDING_MODEL:
            for model in self.embedding_models:
                choices.append(ParameterChoice(layer, "embedding_model", model))
        elif layer == ParameterLayer.RETRIEVAL_METHOD:
            for method in self.retrieval_methods:
                choices.append(ParameterChoice(layer, "retrieval_method", method))
        elif layer == ParameterLayer.RETRIEVAL_TOP_K:
            for top_k in self.retrieval_top_k_options:
                choices.append(ParameterChoice(layer, "retrieval_top_k", top_k))
        elif layer == ParameterLayer.HYBRID_BM25_WEIGHT:
            for weight in self.hybrid_bm25_weights:
                choices.append(ParameterChoice(layer, "hybrid_bm25_weight", weight))
        elif layer == ParameterLayer.QUERY_DECOMPOSITION_NUM_QUERIES:
            for num in self.query_decomposition_num_queries_options:
                choices.append(ParameterChoice(layer, "query_decomposition_num_queries", num))
        elif layer == ParameterLayer.QUERY_DECOMPOSITION_LLM:
            for llm in self.query_decomposition_llms:
                choices.append(ParameterChoice(layer, "query_decomposition_llm", llm))
        elif layer == ParameterLayer.FUSION_MODE:
            for mode in self.fusion_modes:
                choices.append(ParameterChoice(layer, "fusion_mode", mode))
        # elif layer == ParameterLayer.HYDE_LLM:  # 🔥 关闭HyDE：移除HyDE LLM选择逻辑
        #     for llm in self.hyde_llms:
        #         choices.append(ParameterChoice(layer, "hyde_llm", llm))
        elif layer == ParameterLayer.RERANKER_LLM:
            for llm in self.reranker_llms:
                choices.append(ParameterChoice(layer, "reranker_llm", llm))
        elif layer == ParameterLayer.RERANKER_TOP_K:
            for top_k in self.reranker_top_k_options:
                choices.append(ParameterChoice(layer, "reranker_top_k", top_k))
        elif layer == ParameterLayer.ADDITIONAL_CONTEXT_NUM_NODES:
            for num in self.additional_context_num_nodes_options:
                choices.append(ParameterChoice(layer, "additional_context_num_nodes", num))
        elif layer == ParameterLayer.RESPONSE_SYNTHESIZER_LLM:
            for llm in self.response_synthesizer_llms:
                choices.append(ParameterChoice(layer, "response_synthesizer_llm", llm))
        elif layer == ParameterLayer.TEMPLATE_NAME:
            for template in self.template_names:
                choices.append(ParameterChoice(layer, "template_name", template))
        elif layer == ParameterLayer.FEW_SHOT_EMBEDDING_MODEL:
            for model in self.few_shot_embedding_models:
                choices.append(ParameterChoice(layer, "few_shot_embedding_model", model))
        elif layer == ParameterLayer.FEW_SHOT_TOP_K:
            for top_k in self.few_shot_top_k_options:
                choices.append(ParameterChoice(layer, "few_shot_top_k", top_k))
        
        return choices

    def get_parameter_layers(self) -> List[ParameterLayer]:
        """获取参数层序列"""
        return [
            ParameterLayer.SPLITTER_METHOD,
            ParameterLayer.SPLITTER_CHUNK_SIZE,
            ParameterLayer.SPLITTER_OVERLAP,
            ParameterLayer.EMBEDDING_MODEL,
            ParameterLayer.RETRIEVAL_METHOD,
            ParameterLayer.RETRIEVAL_TOP_K,
            ParameterLayer.HYBRID_BM25_WEIGHT,
            ParameterLayer.QUERY_DECOMPOSITION_NUM_QUERIES,
            ParameterLayer.QUERY_DECOMPOSITION_LLM,
            ParameterLayer.FUSION_MODE,
            # ParameterLayer.HYDE_LLM,  # 🔥 关闭HyDE：从参数层序列中移除
            ParameterLayer.RERANKER_LLM,
            ParameterLayer.RERANKER_TOP_K,
            ParameterLayer.ADDITIONAL_CONTEXT_NUM_NODES,
            ParameterLayer.RESPONSE_SYNTHESIZER_LLM,
            ParameterLayer.TEMPLATE_NAME,
            ParameterLayer.FEW_SHOT_EMBEDDING_MODEL,
            ParameterLayer.FEW_SHOT_TOP_K,
        ]

@dataclass
class MCTSNode:
    """MCTS树节点"""
    parameter_path: List[ParameterChoice] = field(default_factory=list)
    depth: int = 0  # 当前深度（对应参数层）
    
    visits: int = 0
    value_sum: float = 0.0
    
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = id(self)
    
    def get_current_params(self) -> Dict[str, Any]:
        """获取当前参数配置"""
        params = {}
        for choice in self.parameter_path:
            params[choice.name] = choice.value
        return params
    
    def is_terminal(self, total_layers: int) -> bool:
        """检查是否为终端节点（所有参数都已选择）"""
        return self.depth >= total_layers
    
    # def is_fully_expanded(self, available_choices: List[ParameterChoice]) -> bool:
    #     """检查是否所有选择都已扩展"""
    #     return len(self.children) >= len(available_choices)
    
    def get_ucb_score(self, exploration_constant: float = 1.414) -> float:
        """计算UCB1分数"""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return self.value_sum / self.visits
        
        exploitation = self.value_sum / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def update(self, value: float):
        """更新节点统计"""
        self.visits += 1
        self.value_sum += value
        
    def get_average_value(self) -> float:
        """获取平均值"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_fully_expanded(self, search_space: 'RAGSearchSpace') -> bool:
        """检查此节点是否已完全扩展"""
        if self.is_terminal(len(search_space.get_parameter_layers())):
            return True # 终端节点视为完全扩展
            
        current_layer = search_space.get_parameter_layers()[self.depth]
        available_choices = search_space.get_layer_choices(current_layer, self.get_current_params())
        return len(self.children) >= len(available_choices)

class TrueMCTS:
    """真正的MCTS实现"""
    
    def __init__(self, 
                 search_space: RAGSearchSpace,
                 evaluation_callback: Callable[[Dict[str, Any]], float],
                 exploration_constant: float = 1.414,
                 max_iterations: int = 100,
                 graph_memory=None,  # 🔥 新增：知识库访问
                 insight_agent=None):  # 🔥 新增：洞察智能体
        
        self.search_space = search_space
        self.evaluate = evaluation_callback
        self.exploration_constant = exploration_constant#1e10#
        self.max_iterations = max_iterations
        
        # 🔥 新增：知识库和智能体
        self.graph_memory = graph_memory
        self.insight_agent = insight_agent
        
        # 获取参数层序列
        self.parameter_layers = search_space.get_parameter_layers()
        self.total_layers = len(self.parameter_layers)
        
        # 初始化根节点
        self.root = MCTSNode(depth=0)
        
        logger.info(f"🌳 True MCTS initialized: {self.total_layers} layers, {max_iterations} iterations")
        if graph_memory:
            logger.info(f"🧠 GPT-guided rollout enabled with knowledge base")

    def search(self) -> Dict[str, Any]:
        """执行MCTS搜索，返回最佳配置"""
        logger.info(f"🎯 Starting True MCTS search...")
        
        for iteration in range(self.max_iterations):
            # 1. Selection - 选择最有前景的叶节点
            leaf_node = self._select()
            logger.info(f"🔍 Selected leaf node {leaf_node} at depth {leaf_node.depth}")
            # 2. Expansion - 扩展一个子节点
            if not leaf_node.is_terminal(self.total_layers):
                leaf_node = self._expand(leaf_node)
                logger.info(f"🌱 Expanded node {leaf_node} at depth {leaf_node.depth}")
            
            # 3. Simulation - 评估当前配置
            value = self._simulate(leaf_node)
            
            # 4. Backpropagation - 反向传播结果
            self._backpropagate(leaf_node, value)
            
            if (iteration + 1) % 1 == 0:
                logger.info(f"🔄 MCTS iteration {iteration + 1}/{self.max_iterations}, "
                           f"root visits: {self.root.visits}, avg value: {self.root.get_average_value():.4f}")
                self.debug_tree_state()
        
        # 选择最佳路径
        best_config = self._get_best_configuration()
        logger.info(f"✅ MCTS search completed, best config selected")
        
        return best_config
    
    def _select(self) -> MCTSNode:
        """Selection: 使用UCB1选择最有前景的叶节点 (带有详细日志)"""
        node = self.root
        
        while node.is_fully_expanded(self.search_space):
            
            # --- 开始注入日志 ---

            # 1. 打印当前决策节点的上下文信息
            logger.info(f"--- MCTS Selection at Depth {node.depth} (Parent Visits: {node.visits}) ---")
            
            # 2. 遍历所有候选子节点，计算并打印它们的详细分数
            candidate_details = []
            for child in node.children:
                param_choice = child.parameter_path[-1]
                visits = child.visits
                
                # 单独计算 利用项 (exploitation)
                exploitation_score = child.get_average_value()
                
                # 单独计算 探索项 (exploration)
                exploration_score = 0.0
                if visits == 0:
                    exploration_score = float('inf')
                elif node.visits > 0:
                    # 公式: C * sqrt(log(N) / n)
                    exploration_score = self.exploration_constant * math.sqrt(
                        math.log(node.visits) / visits
                    )
                
                # 获取最终的UCB1分数
                ucb_score = child.get_ucb_score(self.exploration_constant)
                
                # 格式化输出字符串，为了对齐，给参数值留出足够宽度
                # The str(param_choice.value):<40 pads the string to 40 characters
                log_msg = (
                    f"  - Candidate: {param_choice.name:<30} = {str(param_choice.value):<50} | "
                    f"AvgValue: {exploitation_score:.4f} | "
                    f"Visits: {visits:<4} | "
                    f"ExplScore: {exploration_score:.4f} | "
                    f"UCB1: {ucb_score:.4f}"
                )
                logger.info(log_msg)
                candidate_details.append({'ucb': ucb_score, 'node': child})

            # --- 日志注入结束 ---
            
            # 原始代码：选择UCB1分数最高的子节点
            # best_child = max(node.children, key=lambda child: child.get_ucb_score(self.exploration_constant))
            # 使用我们已经计算好的值，避免重复计算
            if not candidate_details:
                # 如果没有候选，理论上不应该进入这个循环，但作为保护
                break
            best_child = max(candidate_details, key=lambda x: x['ucb'])['node']
            node = best_child
            # --- 再次注入日志 ---

            # 3. 打印最终的选择结果
            selected_param = best_child.parameter_path[-1]
            logger.info(
                f"==> Selected: {selected_param.name} = {selected_param.value} (UCB1: {best_child.get_ucb_score(self.exploration_constant):.4f})\n"
            )
            
            # --- 日志注入结束 ---
            if not node.children:
                # 如果一个节点被标记为完全扩展但没有子节点，那它就是终端节点
                return node
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion: 如果节点未被完全扩展，则为其添加一个新的子节点。
        """
        if node.is_fully_expanded(self.search_space):
            # 如果节点已经完全扩展，不应该再尝试扩展，直接返回自身或一个最优子节点用于模拟
            return node

        # 获取所有可能的、但尚未被创建的子节点选项
        current_layer = self.parameter_layers[node.depth]
        available_choices = self.search_space.get_layer_choices(current_layer, node.get_current_params())
        
        existing_choices_values = {child.parameter_path[-1].value for child in node.children}
        
        untried_choices = [
            choice for choice in available_choices 
            if choice.value not in existing_choices_values
        ]
        
        # 随机选择一个未尝试的选项来创建新节点
        choice_to_add = random.choice(untried_choices)
        
        new_child = MCTSNode(
            parameter_path=node.parameter_path + [choice_to_add],
            depth=node.depth + 1,
            parent=node
        )
        node.children.append(new_child)
        
        return new_child

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation: 评估当前配置"""
        if node.is_terminal(self.total_layers):
            # 终端节点：直接评估完整配置
            complete_config = node.get_current_params()
        else:
            # 非终端节点：GPT引导的rollout
            if random.random() < 0.5:
                logger.info("🔄 Using random rollout for simulation")
                complete_config = self._random_rollout(node)
            else:
                # 使用GPT引导的rollout
                logger.info("🧠 Using GPT-guided rollout for simulation")
                complete_config = self._gpt_guided_rollout(node)  # 🔥 改为GPT引导
        
        # 调用真实评估
        try:
            score = self.evaluate(complete_config)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return 0.0
    
    def _random_rollout(self, node: MCTSNode) -> Dict[str, Any]:
        """随机完成剩余参数"""
        current_params = node.get_current_params()
        
        # 完成剩余层级的参数选择
        for layer_idx in range(node.depth, self.total_layers):
            layer = self.parameter_layers[layer_idx]
            choices = self.search_space.get_layer_choices(layer, current_params)
            
            if choices:
                choice = random.choice(choices)
                current_params[choice.name] = choice.value
        
        return current_params

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation: 反向传播结果"""
        while node is not None:
            node.update(value)
            node = node.parent
    
    def _get_best_configuration(self) -> Dict[str, Any]:
        """获取最佳配置"""
        # 找到访问次数最多的完整路径
        best_config = self._find_best_terminal_path(self.root)
        
        if not best_config:
            # 如果没有完整路径，从根节点开始选择最佳路径
            best_config = self._construct_best_path()
        
        return best_config
    
    def _find_best_terminal_path(self, node: MCTSNode) -> Optional[Dict[str, Any]]:
        """递归查找最佳终端路径"""
        if node.is_terminal(self.total_layers):
            return node.get_current_params()
        
        if not node.children:
            return None
        
        # 选择访问次数最多的子节点
        best_child = max(node.children, key=lambda child: child.visits)
        return self._find_best_terminal_path(best_child)
    
    def _construct_best_path(self) -> Dict[str, Any]:
        """构造最佳路径"""
        current_params = {}
        node = self.root
        
        for layer_idx in range(self.total_layers):
            if node.children:
                # 选择访问次数最多的子节点
                best_child = max(node.children, key=lambda child: child.visits)
                if best_child.parameter_path:
                    last_choice = best_child.parameter_path[-1]
                    current_params[last_choice.name] = last_choice.value
                node = best_child
            else:
                # 没有子节点，随机选择
                layer = self.parameter_layers[layer_idx]
                choices = self.search_space.get_layer_choices(layer, current_params)
                if choices:
                    choice = random.choice(choices)
                    current_params[choice.name] = choice.value
        
        return current_params

    def _gpt_guided_rollout(self, node: MCTSNode) -> Dict[str, Any]:
        """GPT引导的rollout策略 - 简化版"""
        current_params = node.get_current_params()
        
        # 如果没有知识库或洞察智能体，回退到随机rollout
        if not self.graph_memory or not self.insight_agent:
            logger.info("🔄 No knowledge base available, using random rollout")
            return self._random_rollout(node)
        
        try:
            # 为每个剩余层级请求GPT建议
            for layer_idx in range(node.depth, self.total_layers):
                layer = self.parameter_layers[layer_idx]
                choices = self.search_space.get_layer_choices(layer, current_params)
                
                if not choices:
                    continue
                    
                # 🔥 直接调用简化的GPT参数选择
                best_choice = self._ask_gpt_for_parameter_choice(layer, current_params, choices)
                
                if best_choice:
                    current_params[best_choice.name] = best_choice.value
                else:
                    # GPT无法决定时，随机选择
                    choice = random.choice(choices)
                    current_params[choice.name] = choice.value
                    
            logger.info(f"🧠 GPT引导rollout完成，参数路径深度: {node.depth} → {self.total_layers}")
            return current_params
            
        except Exception as e:
            logger.warning(f"⚠️ GPT引导rollout失败: {e}，回退到随机rollout")
            return self._random_rollout(node)

    def _ask_gpt_for_parameter_choice(self, layer: ParameterLayer, current_params: Dict[str, Any], 
                                    available_choices: List) -> Optional:
        """向GPT请求参数选择建议 - 完全集成版"""
        
        try:
            # 🔥 添加graph_memory调试信息
            logger.debug(f"🔍 开始参数选择 {layer.value}, graph_memory类型: {type(self.graph_memory)}")
            
            # 🔥 直接在这里获取知识库上下文
            historical_insights, similar_configs = self.insight_agent.get_knowledge_base_context(
                current_params, self.graph_memory
            )
            
            # 🔥 新增：调试信息 - 显示加载的知识库信息
            logger.info(f"🧠 参数选择 {layer.value}: 加载了 {len(historical_insights)} 条Insight, {len(similar_configs)} 个相似Config")
            logger.info(f"示例historical_insights[0]={historical_insights[0][:100] + '...' if historical_insights else 'N/A'}similar_configs[0]={similar_configs[0] if similar_configs else 'N/A'}")
            
            if historical_insights:
                logger.debug(f"   Insights预览: {[insight[:50] + '...' for insight in historical_insights[:2]]}")
            else:
                logger.debug("   无可用的历史Insights")
                
            if similar_configs:
                config_summary = [f"F1={config['f1_score']:.3f}(相似度={config['similarity']:.2f})" for config in similar_configs[:2]]
                logger.debug(f"   相似Config预览: {config_summary}")
            else:
                logger.debug("   无相似的历史Config")
            
            # 🔥 提取选项值
            available_options = [choice.value for choice in available_choices]
            logger.debug(f"   可选参数: {available_options}")
            
            # 🔥 调用insight_agent的参数选择方法
            chosen_index = self.insight_agent.select_parameter_choice(
                parameter_name=layer.value,
                current_params=current_params,
                available_options=available_options,
                historical_insights=historical_insights,
                similar_configs=similar_configs
            )
            
            # 🔥 返回对应的ParameterChoice对象
            if 0 <= chosen_index < len(available_choices):
                return available_choices[chosen_index]
            else:
                # 默认选择第一个选项
                logger.info(f"🔄 参数{layer.value}使用默认选项0: {available_choices[0].value}")
                return available_choices[0] if available_choices else None
                
        except Exception as e:
            logger.error(f"❌ 参数选择失败: {e}", exc_info=True)
            # 失败时返回第一个选项
            if available_choices:
                logger.info(f"🔄 参数{layer.value}选择失败，使用默认选项0: {available_choices[0].value}")
                return available_choices[0]
            return None

    def debug_tree_state(self):
        """打印MCTS树顶层节点的UCB分数和统计信息"""
        if not self.root.children:
            logger.info("Tree has no children yet.")
            return

        data = []
        for child in self.root.children:
            choice = child.parameter_path[-1]
            exploitation = child.get_average_value()
            
            if child.visits > 0 and self.root.visits > 0:
                exploration = self.exploration_constant * math.sqrt(math.log(self.root.visits) / child.visits)
            else:
                exploration = float('inf')
                
            ucb_score = child.get_ucb_score(self.exploration_constant)
            
            data.append({
                "Parameter": f"{choice.name}={choice.value}",
                "Visits": child.visits,
                "AvgScore (Exploitation)": f"{exploitation:.4f}",
                "Exploration_Term": f"{exploration:.4f}" if exploration != float('inf') else "inf",
                "UCB1_Score": f"{ucb_score:.4f}" if ucb_score != float('inf') else "inf",
            })
        
        # 使用Pandas打印一个清晰的表格
        df = pd.DataFrame(data)
        logger.info("\n" + df.to_string())