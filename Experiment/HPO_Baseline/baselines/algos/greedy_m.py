"""
Greedy-M (Model-First) Baseline Algorithm

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset 2wikimultihopqa --metric joint_f1 --max_evals 50 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# MuSiQue dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset musique --metric joint_em --max_evals 40 --seed 456 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset medqa --metric answer_em --max_evals 35 --seed 101 &

# BioASQ dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset bioasq --metric mrr --max_evals 45 --seed 202 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, musique, finqa, medqa, bioasq
"""
import sys
import random
import json
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List

# Add hammer package to path
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
        if expression.name == 'switch':  # 🔥 修改：hyperopt使用'switch'而不是'choice'
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
            except (AttributeError, IndexError):
                logger.warning(f"Could not extract options for param '{name}'. Skipping.")
        elif expression.name == 'choice':  # 🔥 保留原有choice处理
            try:
                param_options[name] = expression.pos_args[1].obj
            except (AttributeError, IndexError):
                logger.warning(f"Could not extract options for param '{name}'. Skipping.")
    
    ordered_params = [p for p in param_order if p in param_options]
    return param_options, ordered_params

def run_greedy_m(ss: Any, qa_train: List[Dict[str, Any]], qa_test: List[Dict[str, Any]], max_evals: int, seed: int = 42, metric: str = 'joint_f1', dataset_name: str = 'unknown') -> Dict[str, Any]:
    """
    Greedy-M search baseline (Model-First).
    Optimizes parameters sequentially, prioritizing model selection first.
    
    Args:
        ss: Search space (unused but kept for API compatibility)
        qa_train: Training dataset for optimization
        qa_test: Test dataset for evaluation (not used for optimization)
        max_evals: Maximum evaluations
        seed: Random seed
        metric: Optimization metric
        dataset_name: Dataset name for CSV file naming
        
    Returns:
        Best parameter configuration found
    """
    logger.info(f"🚀 开始 Greedy-M 贪心搜索 (模型优先): seed={seed}, metric={metric}, dataset={dataset_name}, max_evals={max_evals}")
    random.seed(seed)
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_greedy_m_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_greedy_m_{dataset_name}_{timestamp}.csv"
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']  # 🔥 将parameters移到最后一列
    
    logger.info(f"📝 初始化CSV结果文件:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    logger.info(f"✅ CSV文件头部已写入，包含指标: {metric_names}")

    hyperopt_space = build_hyperopt_space()
    
    # 🔥 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    
    # 🔥 用于优化的单一指标评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)
    
    search_space, original_order = _convert_hyperopt_space(hyperopt_space)

    # Define the specific parameter order for Greedy-M
    # This order prioritizes model selection, as per the paper.
    greedy_m_core_order = [
        'response_synthesizer_llm',
        'embedding_model', # In your space, it's 'embedding_model', not 'rag_embedding_model'
        'splitter_chunk_size',
        'splitter_overlap',
        'retrieval_top_k',
    ]

    # Construct the final, full optimization order
    processed_params = set()
    param_order = []
    for param in greedy_m_core_order:
        if param in search_space and param not in processed_params:
            param_order.append(param)
            processed_params.add(param)
    
    for param in original_order:
        if param not in processed_params:
            param_order.append(param)
            processed_params.add(param)
    
    logger.info(f"🔍 搜索空间已转换，将按 Greedy-M 顺序优化 {len(param_order)} 个参数: {param_order}")

    fixed_params: Dict[str, Any] = {}
    remaining_params = list(param_order)
    iteration_count = 0

    for i, param_to_optimize in enumerate(param_order):
        logger.info(f"\n[步骤 {i+1}/{len(param_order)}] 正在优化参数: '{param_to_optimize}'")
        remaining_params.remove(param_to_optimize)
        
        best_score_for_current_param = -1.0
        best_value_for_current_param = None
        possible_values = search_space[param_to_optimize]
        
        for value in possible_values:
            # Check max_evals limit before proceeding
            if iteration_count >= max_evals:
                logger.info(f"🛑 达到最大评估次数限制 {max_evals}，提前终止优化")
                break
                
            iteration_count += 1
            temp_params = deepcopy(fixed_params)
            temp_params[param_to_optimize] = value
            
            # 🔥 为剩余参数随机选择值
            for param in remaining_params:
                temp_params[param] = random.choice(search_space[param])
            
            # 🔥 在训练集上评估(用于优化)
            train_start = time.time()
            train_score = train_evaluate_single(temp_params)
            train_metrics = train_evaluate(temp_params)
            train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            test_start = time.time()
            test_metrics = test_evaluate(temp_params)
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
            train_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            try:
                with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(train_row)
                logger.info(f"✅ 迭代 {iteration_count}: 训练集结果已保存到CSV文件: {train_csv_file}")
            except Exception as csv_error:
                logger.error(f"❌ 迭代 {iteration_count}: 训练集CSV保存失败: {csv_error}")
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            test_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            try:
                with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_row)
                logger.info(f"✅ 迭代 {iteration_count}: 测试集结果已保存到CSV文件: {test_csv_file}")
            except Exception as csv_error:
                logger.error(f"❌ 迭代 {iteration_count}: 测试集CSV保存失败: {csv_error}")
            
            logger.info(f"   迭代 {iteration_count}: {param_to_optimize}={value}, 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            print(f"   📊 迭代 {iteration_count}: CSV保存完成，训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            
            # 🔥 只用训练集结果作为优化依据
            if train_score > best_score_for_current_param:
                best_score_for_current_param = train_score
                best_value_for_current_param = value
        
        # Check if we need to break outer loop due to max_evals limit
        if iteration_count >= max_evals:
            logger.info(f"🛑 达到最大评估次数限制 {max_evals}，终止所有参数优化")
            # Use best value found so far, or first value if none tested
            if best_value_for_current_param is None and possible_values:
                best_value_for_current_param = possible_values[0]
                logger.info(f"⚠️ 参数 '{param_to_optimize}' 未完成优化，使用默认值: {best_value_for_current_param}")
            break
        
        fixed_params[param_to_optimize] = best_value_for_current_param
        logger.info(f"✅ 参数 '{param_to_optimize}' 的最佳值已确定为: {json.dumps(best_value_for_current_param)}")
        logger.info(f"   (获得训练集分数: {best_score_for_current_param:.4f})")
        logger.info(f"   当前已固定的参数: {json.dumps(fixed_params)}")

    logger.info(f"\n🏆 Greedy-M 搜索完成！")
    logger.info(f"   总迭代次数: {iteration_count}")
    logger.info(f"   最终找到的最佳参数组合: {json.dumps(fixed_params)}")
    logger.info(f"📋 结果已保存到:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    
    return fixed_params
