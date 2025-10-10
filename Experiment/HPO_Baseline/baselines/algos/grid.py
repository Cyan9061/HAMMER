"""
Grid Search Baseline Algorithm

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset 2wikimultihopqa --metric joint_f1 --max_evals 50 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# MuSiQue dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset musique --metric joint_em --max_evals 40 --seed 456 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset medqa --metric answer_em --max_evals 35 --seed 101 &

# BioASQ dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset bioasq --metric mrr --max_evals 45 --seed 202 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, musique, finqa, medqa, bioasq
"""

import sys
import itertools
import csv
import time
import random
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, List

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.logger import logger
from ..search_space import build_hyperopt_space_from_rag_search_space as build_hyperopt_space
from ..objective import make_evaluate_fn

def _convert_hyperopt_space(hyperopt_space: Dict[str, Any]) -> (Dict[str, List[Any]], List[str]):
    """
    Helper function to convert a hyperopt space into a standard Python dictionary
    and preserve the original parameter order.
    """
    param_options: Dict[str, List[Any]] = {}
    param_order: List[str] = list(hyperopt_space.keys())

    for name, expression in hyperopt_space.items():
        if expression.name == 'switch':  # 🔥 hyperopt的choice实际上是switch
            try:
                # 🔥 从pos_args中提取选项值（跳过第一个索引参数）
                options = []
                for i in range(1, len(expression.pos_args)):
                    arg = expression.pos_args[i]
                    if hasattr(arg, 'obj'):
                        options.append(arg.obj)
                    else:
                        options.append(arg)
                param_options[name] = options
                logger.debug(f"  提取参数 '{name}': {len(options)} 个选项 - {options}")
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not extract options for param '{name}': {e}. Skipping.")
        else:
            logger.warning(f"Unknown expression type '{expression.name}' for param '{name}'. Skipping.")
    
    ordered_params = [p for p in param_order if p in param_options]
    logger.info(f"🔧 成功解析 {len(param_options)} 个参数，总共 {len(ordered_params)} 个有效参数")
    return param_options, ordered_params

def _estimate_total_combinations(grid_values: Dict[str, List]) -> int:
    """
    估算总的参数组合数量
    """
    import math
    param_counts = [len(values) for values in grid_values.values()]
    return math.prod(param_counts)

def _generate_all_combinations(grid_values: Dict[str, List], max_combinations: int = 10000) -> List[Dict[str, Any]]:
    """
    生成参数组合，如果组合数过多则使用随机采样
    
    Args:
        grid_values: 参数值字典
        max_combinations: 最大允许的组合数，超过则使用采样
    """
    param_names = list(grid_values.keys())
    param_values = [grid_values[name] for name in param_names]
    
    # 估算总组合数
    total_combinations = _estimate_total_combinations(grid_values)
    logger.info(f"📊 预估总组合数: {total_combinations:,}")
    
    if total_combinations <= max_combinations:
        # 如果组合数合理，生成所有组合
        logger.info(f"✅ 组合数合理，生成所有 {total_combinations} 个组合")
        all_combinations = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            all_combinations.append(config)
        return all_combinations
    else:
        # 如果组合数过多，使用随机采样
        logger.warning(f"⚠️ 组合数过多 ({total_combinations:,})，使用随机采样 {max_combinations} 个组合")
        sampled_combinations = []
        
        # 使用随机采样生成指定数量的组合
        for _ in range(max_combinations):
            combination = []
            for values in param_values:
                combination.append(random.choice(values))
            config = dict(zip(param_names, combination))
            
            # 避免重复组合
            if config not in sampled_combinations:
                sampled_combinations.append(config)
        
        logger.info(f"🎯 实际生成 {len(sampled_combinations)} 个不重复组合")
        return sampled_combinations

def run_grid(ss: Any, qa_train: List[Dict[str, Any]], qa_test: List[Dict[str, Any]], max_evals: int, seed: int = 42, metric: str = 'joint_f1', dataset_name: str = 'unknown') -> Dict[str, Any]:
    """
    Grid search baseline
    Systematically evaluates all parameter combinations (or a random subset if max_evals is limited)
    
    Args:
        ss: Search space (unused but kept for API compatibility)
        qa_train: Training dataset for optimization
        qa_test: Test dataset for evaluation (not used for optimization)
        max_evals: Maximum evaluations (if less than total combinations, random sampling is used)
        seed: Random seed
        metric: Optimization metric
        dataset_name: Dataset name for CSV file naming
        
    Returns:
        Best parameter configuration found
    """
    logger.info(f"🔍 开始网格搜索: max_evals={max_evals}, seed={seed}, metric={metric}, dataset={dataset_name}")
    random.seed(seed)
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_grid_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_grid_{dataset_name}_{timestamp}.csv"
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']  # 🔥 将parameters移到最后一列
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # 构建搜索空间
    hyperopt_space = build_hyperopt_space()
    
    # 🔥 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    
    # 🔥 用于优化的单一指标评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)
    
    # 转换为网格参数值
    grid_values, param_order = _convert_hyperopt_space(hyperopt_space)
    logger.info(f"📊 网格搜索空间参数:")
    for param, values in grid_values.items():
        logger.info(f"  {param}: {len(values)} 个选择")
    
    # 生成组合，如果max_evals较小则限制组合数
    max_combinations = max(max_evals * 2, 10000) if max_evals else 10000
    all_combinations = _generate_all_combinations(grid_values, max_combinations)
    total_combinations = len(all_combinations)
    
    logger.info(f"🎯 总共 {total_combinations} 种参数组合")
    
    # 如果指定了max_evals，随机采样组合
    if max_evals is not None and max_evals < total_combinations:
        logger.info(f"⚡ 限制评估次数为 {max_evals}，随机选择组合")
        random.shuffle(all_combinations)
        combinations_to_evaluate = all_combinations[:max_evals]
    else:
        # 评估所有组合，但打乱顺序
        combinations_to_evaluate = all_combinations.copy()
        random.shuffle(combinations_to_evaluate)
        if max_evals is not None:
            combinations_to_evaluate = combinations_to_evaluate[:max_evals]
    
    logger.info(f"🚀 开始评估 {len(combinations_to_evaluate)} 种组合")
    
    # 执行网格搜索
    best_score = -float('inf')
    best_params = None
    iteration_count = 0
    
    for config in combinations_to_evaluate:
        iteration_count += 1
        logger.info(f"🔧 评估组合 {iteration_count}/{len(combinations_to_evaluate)}: {config}")
        
        try:
            # 🔥 在训练集上评估(用于优化)
            train_start = time.time()
            train_score = train_evaluate_single(config)
            train_metrics = train_evaluate(config)
            train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            test_start = time.time()
            test_metrics = test_evaluate(config)
            test_time = time.time() - test_start
            
            # 🔥 保存训练集结果到CSV
            train_row = [iteration_count]
            train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
            train_row.append(train_time)
            train_row.append(train_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            train_row.append(train_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            import datetime
            import json
            train_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            train_row.append(json.dumps(config, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(train_row)
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            test_row.append(json.dumps(config, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(test_row)
            
            logger.info(f"   迭代 {iteration_count}: 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            
            # 🔥 只用训练集结果作为优化依据
            if train_score > best_score:
                best_score = train_score
                best_params = config.copy()
                logger.info(f"🏆 发现更佳配置: 得分 {best_score:.4f}")
        
        except Exception as e:
            logger.error(f"❌ 评估组合 {iteration_count} 时出错: {e}")
            continue
    
    if best_params is None:
        logger.error("❌ 网格搜索失败：没有找到有效的配置")
        # 返回第一个配置作为fallback
        best_params = combinations_to_evaluate[0] if combinations_to_evaluate else {}
    else:
        logger.info(f"✅ 网格搜索完成")
        logger.info(f"🏆 最佳得分: {best_score:.4f}")
        logger.info(f"🎯 最佳配置: {best_params}")
        
    logger.info(f"📋 结果已保存到:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    
    return best_params