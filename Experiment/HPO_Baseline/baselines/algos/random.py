"""
Random Search Baseline Algorithm

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset 2wikimultihopqa --metric joint_f1 --max_evals 50 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# MuSiQue dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset musique --metric joint_em --max_evals 40 --seed 456 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset medqa --metric answer_f1 --max_evals 100 --seed 101 &

# BioASQ dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset bioasq --metric mrr --max_evals 45 --seed 202 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, musique, finqa, medqa, bioasq
"""

import sys
import csv
import time
from pathlib import Path
from hyperopt import fmin, rand, space_eval, Trials
import numpy as np   # 新增

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.logger import logger
from ..search_space import build_hyperopt_space_from_rag_search_space
from ..objective import make_evaluate_fn

def run_random(ss, qa_train, qa_test, max_evals=10, seed=42, metric='joint_f1', dataset_name='unknown'):
    """
    随机搜索baseline
    
    Args:
        ss: 搜索空间 (未使用但保持API兼容性)
        qa_train: 训练数据集用于优化
        qa_test: 测试数据集用于评估(不用于优化)
        max_evals: 最大评估次数
        seed: 随机种子
        metric: 优化指标
        dataset_name: 数据集名称用于CSV文件命名
    """
    logger.info(f"🎲 开始随机搜索: max_evals={max_evals}, seed={seed}, metric={metric}, dataset={dataset_name}")
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_random_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_random_{dataset_name}_{timestamp}.csv"
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']  # 🔥 将parameters移到最后一列
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    space = build_hyperopt_space_from_rag_search_space()
    
    # 🔥 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    
    # 🔥 用于优化的单一指标评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)

    logger.info(f"🔍 搜索空间: {len(space)}个参数")
    
    # 🔥 封装目标函数以记录结果
    iteration_count = [0]  # 使用列表以便在函数内修改
    
    def objective_with_logging(params):
        iteration_count[0] += 1
        
        try:
            # 🔧 详细日志开始
            print(f"🔍 迭代 {iteration_count[0]} 开始评估: {params}")
            logger.info(f"🔍 迭代 {iteration_count[0]} 开始评估: {params}")
            
            # 🔥 在训练集上评估(用于优化)
            print(f"📊 开始训练集评估...")
            train_start = time.time()
            
            try:
                train_score = train_evaluate_single(params)
                train_metrics = train_evaluate(params)
                print(f"✅ 训练集评估成功: {metric}={train_score:.4f}")
                print(f"📈 训练集所有指标: {train_metrics}")
            except Exception as train_error:
                print(f"❌ 训练集评估失败: {train_error}")
                logger.error(f"❌ 训练集评估失败: {train_error}", exc_info=True)
                # 使用默认值
                train_score = 0.0
                train_metrics = {m: 0.0 for m in metric_names}
            
            train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            print(f"📊 开始测试集评估...")
            test_start = time.time()
            
            try:
                test_metrics = test_evaluate(params)
                print(f"✅ 测试集评估成功")
                print(f"📈 测试集所有指标: {test_metrics}")
            except Exception as test_error:
                print(f"❌ 测试集评估失败: {test_error}")
                logger.error(f"❌ 测试集评估失败: {test_error}", exc_info=True)
                # 使用默认值
                test_metrics = {m: 0.0 for m in metric_names}
                
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
            
            try:
                with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(train_row)
                logger.info(f"✅ 迭代 {iteration_count[0]}: 训练集结果已保存到CSV")
                print(f"✅ 迭代 {iteration_count[0]}: 训练集结果已保存到CSV")
            except Exception as csv_error:
                logger.error(f"❌ 迭代 {iteration_count[0]}: 训练集CSV保存失败: {csv_error}")
                print(f"❌ 迭代 {iteration_count[0]}: 训练集CSV保存失败: {csv_error}")
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count[0]]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(current_timestamp)  # 🔥 添加时间戳
            test_row.append(json.dumps(params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            try:
                with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_row)
                logger.info(f"✅ 迭代 {iteration_count[0]}: 测试集结果已保存到CSV")
                print(f"✅ 迭代 {iteration_count[0]}: 测试集结果已保存到CSV")
            except Exception as csv_error:
                logger.error(f"❌ 迭代 {iteration_count[0]}: 测试集CSV保存失败: {csv_error}")
                print(f"❌ 迭代 {iteration_count[0]}: 测试集CSV保存失败: {csv_error}")
            
            # 🔧 综合结果日志
            result_msg = f"🎯 迭代 {iteration_count[0]} 完成: 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}, 训练耗时={train_time:.2f}s, 测试耗时={test_time:.2f}s"
            print(result_msg)
            logger.info(result_msg)
            
            return -train_score  # hyperopt最小化，所以取负值
            
        except Exception as e:
            error_msg = f"💥 迭代 {iteration_count[0]} 发生严重异常: {e}"
            print(error_msg)
            logger.error(error_msg, exc_info=True)
            
            # 🔧 即使异常也要记录CSV以保持数据完整性
            try:
                failed_metrics = {m: 0.0 for m in metric_names}
                
                train_row = [iteration_count[0]]
                train_row.extend([0.0] * len(metric_names))
                train_row.append(0.0)
                train_row.append(0)  # 🔥 Failed情况下token=0
                train_row.append(0)  # 🔥 Failed情况下samples=0
                train_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
                train_row.append(json.dumps({"error": "evaluation_failed"}, ensure_ascii=False))  # 🔥 失败情况记录错误信息
                
                with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(train_row)
                    
                test_row = [iteration_count[0]]
                test_row.extend([0.0] * len(metric_names))
                test_row.append(0.0)
                test_row.append(0)  # 🔥 Failed情况下token=0
                test_row.append(0)  # 🔥 Failed情况下samples=0
                test_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
                test_row.append(json.dumps({"error": "evaluation_failed"}, ensure_ascii=False))  # 🔥 失败情况记录错误信息
                
                with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_row)
            except Exception as csv_error:
                print(f"💥💥 CSV记录也失败: {csv_error}")
                
            return 0.0  # 异常时返回0，避免hyperopt崩溃

    # 生成合适的 rstate 对象（优先使用 numpy Generator）
    try:
        rstate = np.random.default_rng(seed)  # 现代 numpy（有 .integers）
    except Exception:
        # 兼容旧环境：回退到 RandomState（旧方法可能使用 .randint）
        rstate = np.random.RandomState(seed)

    best = fmin(
        fn=objective_with_logging,
        space=space,
        algo=rand.suggest,
        max_evals=max_evals,
        rstate=rstate,
        verbose=True
    )

    best_params = space_eval(space, best)
    logger.info(f"🏆 随机搜索完成，总迭代次数: {iteration_count[0]}")
    logger.info(f"🏆 最佳参数: {best_params}")
    logger.info(f"📋 结果已保存到:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")

    return best_params
