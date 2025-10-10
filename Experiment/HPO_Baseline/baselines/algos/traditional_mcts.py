"""
Traditional Monte Carlo Tree Search (MCTS) Baseline Algorithm

This implementation provides a pure MCTS algorithm without:
- LLM-guided rollout 
- Memory components (graph_memory, insight_agent)
- Advanced heuristics

Features classic MCTS with:
- UCB1 selection strategy
- Random expansion
- Random rollout simulation  
- Standard backpropagation

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset 2wikimultihopqa --metric joint_f1 --max_evals 75 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, musique, finqa, medqa, bioasq, eli5, convfinqa, popqa, web_questions, quartz
"""

import sys
import csv
import time
import math
import random
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.logger import logger
from ..search_space import build_hyperopt_space_from_rag_search_space, get_mcts_search_space
from ..objective import make_evaluate_fn

class ParameterLayer(Enum):
    """RAG参数分层结构"""
    SPLITTER_METHOD = "splitter_method"
    SPLITTER_CHUNK_SIZE = "splitter_chunk_size"
    SPLITTER_OVERLAP = "splitter_overlap"
    EMBEDDING_MODEL = "embedding_model"
    RETRIEVAL_METHOD = "retrieval_method"
    RETRIEVAL_TOP_K = "retrieval_top_k"
    HYBRID_BM25_WEIGHT = "hybrid_bm25_weight"
    QUERY_DECOMPOSITION_NUM_QUERIES = "query_decomposition_num_queries"
    QUERY_DECOMPOSITION_LLM = "query_decomposition_llm"
    FUSION_MODE = "fusion_mode"
    HYDE_LLM = "hyde_llm"
    RERANKER_LLM = "reranker_models"
    RERANKER_TOP_K = "reranker_top_k"
    ADDITIONAL_CONTEXT_NUM_NODES = "additional_context_num_nodes"
    RESPONSE_SYNTHESIZER_LLM = "response_synthesizer_llm"
    TEMPLATE_NAME = "template_name"

@dataclass
class ParameterChoice:
    """单个参数选择"""
    layer: ParameterLayer
    name: str
    value: Any
    
    def __hash__(self):
        return hash((self.layer, self.name, str(self.value)))

@dataclass
class TraditionalMCTSNode:
    """传统MCTS树节点 - 简化版"""
    parameter_path: List[ParameterChoice] = field(default_factory=list)
    depth: int = 0
    
    visits: int = 0
    value_sum: float = 0.0
    
    parent: Optional['TraditionalMCTSNode'] = None
    children: List['TraditionalMCTSNode'] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = id(self)
    
    def get_current_params(self) -> Dict[str, Any]:
        """获取当前参数配置"""
        params = {}
        for choice in self.parameter_path:
            params[choice.name] = choice.value
        return params
    
    def is_terminal(self, total_layers: int) -> bool:
        """检查是否为终端节点"""
        return self.depth >= total_layers
    
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

    def is_fully_expanded(self, search_space) -> bool:
        """检查此节点是否已完全扩展"""
        if self.is_terminal(len(search_space.get_parameter_layers())):
            return True
            
        current_layer = search_space.get_parameter_layers()[self.depth]
        available_choices = search_space.get_layer_choices(current_layer, self.get_current_params())
        return len(self.children) >= len(available_choices)

class TraditionalMCTS:
    """传统MCTS实现 - 无LLM指导和记忆组件"""
    
    def __init__(self, 
                 search_space,
                 evaluation_callback: Callable[[Dict[str, Any]], float],
                 exploration_constant: float = 1.414,
                 max_iterations: int = 100):
        
        self.search_space = search_space
        self.evaluate = evaluation_callback
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        
        # 获取参数层序列
        self.parameter_layers = search_space.get_parameter_layers()
        self.total_layers = len(self.parameter_layers)
        
        # 初始化根节点
        self.root = TraditionalMCTSNode(depth=0)
        
        logger.info(f"🌳 Traditional MCTS initialized: {self.total_layers} layers, {max_iterations} iterations")

    def search(self) -> Dict[str, Any]:
        """执行传统MCTS搜索"""
        logger.info(f"🎯 Starting Traditional MCTS search...")
        
        for iteration in range(self.max_iterations):
            # 1. Selection - 选择最有前景的叶节点
            leaf_node = self._select()
            
            # 2. Expansion - 扩展一个子节点
            if not leaf_node.is_terminal(self.total_layers):
                leaf_node = self._expand(leaf_node)
            
            # 3. Simulation - 随机rollout评估
            value = self._simulate(leaf_node)
            
            # 4. Backpropagation - 反向传播结果
            self._backpropagate(leaf_node, value)
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"🔄 Traditional MCTS iteration {iteration + 1}/{self.max_iterations}, "
                           f"root visits: {self.root.visits}, avg value: {self.root.get_average_value():.4f}")
        
        # 选择最佳路径
        best_config = self._get_best_configuration()
        logger.info(f"✅ Traditional MCTS search completed, best config selected")
        
        return best_config
    
    def _select(self) -> TraditionalMCTSNode:
        """Selection: 使用UCB1选择最有前景的叶节点"""
        node = self.root
        
        while node.is_fully_expanded(self.search_space):
            if not node.children:
                return node
            
            # 选择UCB1分数最高的子节点
            best_child = max(node.children, key=lambda child: child.get_ucb_score(self.exploration_constant))
            node = best_child
        
        return node
    
    def _expand(self, node: TraditionalMCTSNode) -> TraditionalMCTSNode:
        """Expansion: 随机扩展一个未访问的子节点"""
        if node.is_fully_expanded(self.search_space):
            return node

        # 获取当前层的所有可能选择
        current_layer = self.parameter_layers[node.depth]
        available_choices = self.search_space.get_layer_choices(current_layer, node.get_current_params())
        
        # 找出尚未扩展的选择
        existing_choices_values = {child.parameter_path[-1].value for child in node.children}
        untried_choices = [
            choice for choice in available_choices 
            if choice.value not in existing_choices_values
        ]
        
        if not untried_choices:
            return node
        
        # 随机选择一个未尝试的选项
        choice_to_add = random.choice(untried_choices)
        
        new_child = TraditionalMCTSNode(
            parameter_path=node.parameter_path + [choice_to_add],
            depth=node.depth + 1,
            parent=node
        )
        node.children.append(new_child)
        
        return new_child
    
    def _simulate(self, node: TraditionalMCTSNode) -> float:
        """Simulation: 完全随机rollout评估"""
        if node.is_terminal(self.total_layers):
            # 终端节点：直接评估完整配置
            complete_config = node.get_current_params()
        else:
            # 非终端节点：随机完成剩余参数
            complete_config = self._random_rollout(node)
        
        # 调用真实评估
        try:
            score = self.evaluate(complete_config)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return 0.0
    
    def _random_rollout(self, node: TraditionalMCTSNode) -> Dict[str, Any]:
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

    def _backpropagate(self, node: TraditionalMCTSNode, value: float):
        """Backpropagation: 反向传播结果"""
        while node is not None:
            node.update(value)
            node = node.parent
    
    def _get_best_configuration(self) -> Dict[str, Any]:
        """获取最佳配置 - 选择访问次数最多的路径"""
        node = self.root
        
        while node.children:
            # 选择访问次数最多的子节点（exploitation策略）
            best_child = max(node.children, key=lambda child: child.visits)
            node = best_child
        
        return node.get_current_params()

def run_traditional_mcts(ss, qa_train, qa_test, max_evals=10, seed=42, metric='joint_f1', dataset_name='unknown'):
    """
    传统Monte Carlo Tree Search算法baseline
    
    特点：
    - 经典的四阶段MCTS算法：Selection, Expansion, Simulation, Backpropagation
    - UCB1选择策略
    - 完全随机rollout（无LLM指导）
    - 无记忆组件
    
    Args:
        ss: 搜索空间 (未使用但保持API兼容性)
        qa_train: 训练数据集用于优化
        qa_test: 测试数据集用于评估(不用于优化)
        max_evals: 最大评估次数
        seed: 随机种子
        metric: 优化指标
        dataset_name: 数据集名称用于CSV文件命名
    """
    logger.info(f"🌳 开始Traditional MCTS搜索: max_evals={max_evals}, seed={seed}, metric={metric}, dataset={dataset_name}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_traditional_mcts_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_traditional_mcts_{dataset_name}_{timestamp}.csv"
    
    # 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    # 构建搜索空间和评估函数
    search_space = get_mcts_search_space()
    
    # 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    
    # 用于优化的单一指标评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)
    
    logger.info(f"🔍 搜索空间: {len(search_space.get_parameter_layers())}个参数层")
    
    # 封装目标函数以记录结果
    iteration_count = [0]
    
    def objective_wrapper(params):
        """包装目标函数以记录结果"""
        try:
            iteration_count[0] += 1
            
            # 在训练集上评估(用于优化)
            train_start = time.time()
            train_score = train_evaluate_single(params)
            train_metrics = train_evaluate(params)
            train_time = time.time() - train_start
            
            # 在测试集上评估(仅用于记录)
            test_start = time.time()
            test_metrics = test_evaluate(params)
            test_time = time.time() - test_start
            
            # 保存训练集结果到CSV
            import datetime
            current_timestamp = datetime.datetime.now().isoformat()
            train_row = [iteration_count[0]]
            train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
            train_row.append(train_time)
            train_row.append(train_metrics.get('total_tokens', 0))
            train_row.append(train_metrics.get('training_samples', 0))
            train_row.append(current_timestamp)
            train_row.append(json.dumps(params, ensure_ascii=False, separators=(',', ':')))
            
            with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(train_row)
                
            # 保存测试集结果到CSV
            test_row = [iteration_count[0]]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))
            test_row.append(test_metrics.get('training_samples', 0))
            test_row.append(current_timestamp)
            test_row.append(json.dumps(params, ensure_ascii=False, separators=(',', ':')))
            
            with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(test_row)
            
            logger.info(f"   迭代 {iteration_count[0]}: 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            
            return train_score
        except Exception as e:
            logger.error(f"❌ Traditional MCTS评估参数时出错: {e}")
            return 0.0
    
    try:
        # 创建Traditional MCTS实例
        mcts = TraditionalMCTS(
            search_space=search_space,
            evaluation_callback=objective_wrapper,
            exploration_constant=1.414,  # 标准UCB1探索常数
            max_iterations=max_evals
        )
        
        # 执行MCTS搜索
        logger.info(f"🚀 开始Traditional MCTS优化")
        best_params = mcts.search()
        
        logger.info(f"✅ Traditional MCTS搜索完成, 总迭代次数: {iteration_count[0]}")
        logger.info(f"🎯 最佳参数: {best_params}")
        logger.info(f"📋 结果已保存到:")
        logger.info(f"   训练集: {train_csv_file}")
        logger.info(f"   测试集: {test_csv_file}")
        
    except Exception as e:
        logger.error(f"❌ Traditional MCTS搜索过程中出错: {e}")
        # 如果MCTS完全失败，返回随机配置作为fallback
        logger.warning("🔄 Traditional MCTS失败，使用随机搜索作为备选")
        from .random import run_random
        return run_random(ss, qa_train, qa_test, min(max_evals, 5), seed, metric, dataset_name)
    
    return best_params