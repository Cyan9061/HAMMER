"""
Tree-structured Parzen Estimators (TPE) Baseline Algorithm

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset 2wikimultihopqa --metric joint_f1 --max_evals 50 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# MuSiQue dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset musique --metric joint_em --max_evals 40 --seed 456 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset medqa --metric answer_em --max_evals 35 --seed 101 &

# BioASQ dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset bioasq --metric mrr --max_evals 45 --seed 202 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, musique, finqa, medqa, bioasq
"""

import sys
import csv
import time
from pathlib import Path
from hyperopt import fmin, tpe, space_eval, Trials, STATUS_OK
import numpy as np
from functools import partial

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.logger import logger
from ..search_space import build_hyperopt_space_from_rag_search_space
from ..objective import make_evaluate_fn

def run_tpe(ss, qa_train, qa_test, max_evals=10, seed=42, metric='joint_f1', dataset_name='unknown'):
    """
    Tree-structured Parzen Estimators (TPE) 算法baseline
    
    TPE是一种序列化模型优化(SMBO)算法，它：
    1. 维护历史评估结果
    2. 使用Parzen估计器建模p(x|y)
    3. 将观察结果分为"好"和"坏"两类
    4. 选择最大化EI(期望改进)的下一个点
    
    Args:
        ss: 搜索空间 (未使用但保持API兼容性)
        qa_train: 训练数据集用于优化
        qa_test: 测试数据集用于评估(不用于优化)
        max_evals: 最大评估次数
        seed: 随机种子
        metric: 优化指标
        dataset_name: 数据集名称用于CSV文件命名
    """
    logger.info(f"🧠 开始TPE搜索: max_evals={max_evals}, seed={seed}, metric={metric}, dataset={dataset_name}")
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_tpe_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_tpe_{dataset_name}_{timestamp}.csv"
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']  # 🔥 将parameters移到最后一列
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    # 构建搜索空间和评估函数
    space = build_hyperopt_space_from_rag_search_space()
    
    # 🔥 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    
    # 🔥 用于优化的单一指标评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)
    
    logger.info(f"🔍 搜索空间: {len(space)}个参数")
    
    # 创建RandomState对象
    try:
        rstate = np.random.default_rng(seed)  # 现代numpy版本
    except AttributeError:
        rstate = np.random.RandomState(seed)  # 兼容旧版本
    
    # 创建Trials对象来跟踪历史评估
    trials = Trials()
    
    # 配置TPE算法参数
    tpe_algo = partial(
        tpe.suggest,
        # gamma控制好/坏样本的分割点，默认0.25表示前25%为"好"样本
        gamma=0.25,
        # n_startup_jobs控制随机初始化的试验次数
        n_startup_jobs=max(1, max_evals // 4),  # 25%的试验用于随机初始化
        # n_EI_candidates控制每次优化时考虑的候选点数量
        n_EI_candidates=24,
    )
    
    # 🔥 封装目标函数以记录结果
    iteration_count = [0]  # 使用列表以便在函数内修改
    
    def objective_wrapper(params):
        """包装目标函数以适配hyperopt格式并记录结果"""
        try:
            iteration_count[0] += 1
            
            # 🔥 在训练集上评估(用于优化)
            train_start = time.time()
            train_score = train_evaluate_single(params)
            train_metrics = train_evaluate(params)
            train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            test_start = time.time()
            test_metrics = test_evaluate(params)
            test_time = time.time() - test_start
            
            # 🔥 保存训练集结果到CSV
            import datetime
            import json
            current_timestamp = datetime.datetime.now().isoformat()
            train_row = [iteration_count[0]]
            train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
            train_row.append(train_time)
            train_row.append(train_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            train_row.append(train_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            train_row.append(current_timestamp)  # 🔥 添加时间戳
            train_row.append(json.dumps(params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(train_row)
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count[0]]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(current_timestamp)  # 🔥 添加时间戳
            test_row.append(json.dumps(params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(test_row)
            
            logger.info(f"   迭代 {iteration_count[0]}: 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            
            # hyperopt最小化，但我们要最大化F1分数，所以取负值
            return {
                'loss': -train_score,
                'status': STATUS_OK,
                'eval_time': train_time,
                'params': params.copy(),
            }
        except Exception as e:
            logger.error(f"❌ TPE评估参数时出错: {e}")
            # 返回很差的分数，但不中断搜索
            return {
                'loss': float('inf'),
                'status': STATUS_OK,
                'params': params.copy(),
            }
    
    try:
        # 执行TPE优化
        logger.info(f"🚀 开始TPE优化，随机初始化: {max(1, max_evals // 4)} 次")
        
        best = fmin(
            fn=objective_wrapper,
            space=space,
            algo=tpe_algo,
            max_evals=max_evals,
            trials=trials,
            rstate=rstate,
            verbose=True,
            show_progressbar=True
        )
        
        # 转换最佳参数
        best_params = space_eval(space, best)
        
        # 获取最佳分数
        best_trial = trials.best_trial
        best_score = -best_trial['result']['loss'] if best_trial else 0.0
        
        logger.info(f"✅ TPE搜索完成, 总迭代次数: {iteration_count[0]}")
        logger.info(f"🏆 最佳得分: {best_score:.4f}")
        logger.info(f"🎯 最佳参数: {best_params}")
        logger.info(f"📊 总评估次数: {len(trials.trials)}")
        logger.info(f"📋 结果已保存到:")
        logger.info(f"   训练集: {train_csv_file}")
        logger.info(f"   测试集: {test_csv_file}")
        
    except Exception as e:
        logger.error(f"❌ TPE搜索过程中出错: {e}")
        # 如果TPE完全失败，返回第一个随机配置作为fallback
        logger.warning("🔄 TPE失败，使用随机搜索作为备选")
        from .random import run_random
        return run_random(ss, qa_train, qa_test, min(max_evals, 5), seed, metric, dataset_name)
    
    return best_params