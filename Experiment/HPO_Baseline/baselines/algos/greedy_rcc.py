"""
Enhanced Greedy-R-CC (RAG Pipeline Order with Context Correctness) Baseline Algorithm

This algorithm implements an enhanced Greedy-R-CC approach based on comprehensive search space analysis:
1. Intelligent parameter categorization: Parameters grouped by their actual impact on retrieval vs generation
2. Context-aware metric switching: Uses MRR for retrieval parameters that affect context quality,
   Answer Correctness for query understanding and answer generation parameters
3. Full search space coverage: Utilizes our complete parameter space rather than paper's limited scope
4. Efficiency: Optimizes retrieval quality first, then answer generation quality

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset 2wikimultihopqa --metric joint_f1 --max_evals 50 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset hotpotqa --metric answer_f1 --max_evals 30 --seed 123 &

# MuSiQue dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset musique --metric joint_em --max_evals 40 --seed 456 &

# FinQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset finqa --metric lexical_ac --max_evals 50 --seed 789 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset medqa --metric answer_em --max_evals 35 --seed 101 &

# BioASQ dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset bioasq --metric mrr --max_evals 45 --seed 202 &

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
    
    logger.info(f"🔍 开始转换hyperopt搜索空间，包含{len(hyperopt_space)}个参数")

    for name, expression in hyperopt_space.items():
        logger.info(f"   处理参数 '{name}': type={type(expression)}, name='{expression.name if hasattr(expression, 'name') else 'N/A'}'")
        
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
                logger.info(f"   ✅ 成功提取switch参数 '{name}': {len(options)}个选项 = {options[:3]}...")
            except (AttributeError, IndexError) as e:
                logger.error(f"   ❌ 无法提取switch参数 '{name}': {e}")
                logger.warning(f"Could not extract options for param '{name}'. Skipping.")
        elif expression.name == 'choice':  # 🔥 保留原有choice处理
            try:
                param_options[name] = expression.pos_args[1].obj
                logger.info(f"   ✅ 成功提取choice参数 '{name}': {len(expression.pos_args[1].obj)}个选项")
            except (AttributeError, IndexError) as e:
                logger.error(f"   ❌ 无法提取choice参数 '{name}': {e}")
                logger.warning(f"Could not extract options for param '{name}'. Skipping.")
        else:
            logger.warning(f"   ⚠️ 未知参数类型 '{name}': {expression.name}")
    
    ordered_params = [p for p in param_order if p in param_options]
    logger.info(f"🎯 搜索空间转换完成: {len(param_options)}/{len(param_order)} 参数成功转换")
    logger.info(f"   转换成功的参数: {ordered_params}")
    return param_options, ordered_params

def run_greedy_rcc(ss: Any, qa_train: List[Dict[str, Any]], qa_test: List[Dict[str, Any]], max_evals: int, seed: int = 42, metric: str = 'joint_f1', dataset_name: str = 'unknown') -> Dict[str, Any]:
    """
    Enhanced Greedy-R-CC search baseline with intelligent parameter categorization.
    
    Implements our improved approach:
    1. Intelligent parameter grouping based on comprehensive search space analysis
    2. Context-aware metric switching: MRR for retrieval quality, Answer Correctness for generation quality
    3. Full search space utilization: Covers all parameters in our complete RAG pipeline
    4. Two-phase optimization: Retrieval optimization → Answer generation optimization
    
    Parameter Categories:
    - Retrieval params (MRR): embedding_model, splitter_*, retrieval_*, reranker_*, additional_context_*
    - Generation params (Answer Correctness): query_decomposition_*, fusion_mode, hyde_*, response_synthesizer_llm, template_name
    
    Args:
        ss: Search space (unused but kept for API compatibility)
        qa_train: Training dataset for optimization
        qa_test: Test dataset for evaluation (not used for optimization)
        max_evals: Maximum evaluations 
        seed: Random seed
        metric: Final optimization metric (used for generation parameters)
        dataset_name: Dataset name for CSV file naming
        
    Returns:
        Best parameter configuration found
    """
    logger.info(f"🚀 开始增强版 Greedy-R-CC 搜索 (基于完整搜索空间的智能参数分类): seed={seed}, metric={metric}, dataset={dataset_name}, max_evals={max_evals}")
    logger.info(f"📊 数据集大小: 训练集={len(qa_train)}条, 测试集={len(qa_test)}条")
    random.seed(seed)
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径
    output_dir = Path("Experiment/HPO_Baseline/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_greedy_rcc_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_greedy_rcc_{dataset_name}_{timestamp}.csv"
    
    logger.info(f"📝 CSV结果文件:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration', 'optimization_metric'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']  # 🔥 保留optimization_metric，将parameters移到最后
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    logger.info(f"🔧 开始构建搜索空间...")
    hyperopt_space = build_hyperopt_space()
    logger.info(f"✅ 搜索空间构建完成")
    
    # 🔥 创建训练集和测试集的评估函数
    logger.info(f"🎯 创建评估函数 (优化指标: {metric}, 数据集: {dataset_name})...")
    try:
        train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
        test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
        logger.info(f"✅ 评估函数创建成功")
    except Exception as e:
        logger.error(f"❌ 评估函数创建失败: {e}")
        raise
    
    logger.info(f"🔄 开始转换搜索空间...")
    search_space, original_order = _convert_hyperopt_space(hyperopt_space)
    
    # 🔥 重新设计参数分组（基于我们完整搜索空间的深度分析）
    # 检索相关参数：使用MRR (Context Correctness)指标优化 - 这些参数直接影响检索到的上下文质量
    retrieval_params = [
        # 文本处理和嵌入 - 影响文档表示质量
        'embedding_model',           # 嵌入模型决定向量表示质量，直接影响检索准确性
        'splitter_method',          # 分割方法影响文档块的语义完整性
        'splitter_chunk_size',      # 块大小影响上下文窗口和检索粒度
        'splitter_overlap',         # 重叠度影响信息连续性
        
        # 核心检索策略 - 决定如何找到相关文档
        'retrieval_method',         # dense/sparse/hybrid决定检索算法
        'retrieval_top_k',          # 检索数量直接影响召回率
        'hybrid_bm25_weight',       # 混合检索权重平衡语义和关键词匹配
        
        # 检索后处理 - 改善检索结果质量
        'reranker_llm',            # 重排序模型类型影响排序准确性
        'reranker_top_k',          # 重排序后保留数量
        'reranker_enabled',        # 是否启用重排序
        'additional_context_num_nodes',  # 额外上下文节点数
        'additional_context_enabled'     # 是否启用额外上下文
    ]
    
    # 查询理解和答案生成相关参数：使用Answer Correctness指标优化 - 这些参数影响如何理解查询并生成答案
    generation_params = [
        # 查询理解和分解 - 影响如何理解和处理用户查询
        'query_decomposition_enabled',    # 是否启用查询分解
        'query_decomposition_num_queries', # 分解查询数量影响查询理解深度
        'query_decomposition_llm',        # 查询分解使用的LLM
        'fusion_mode',                   # 多查询结果融合策略
        
        # 查询增强 - 改善查询理解
        'hyde_enabled',                  # 是否启用假设文档生成
        'hyde_llm',                     # HyDE使用的LLM
        
        # 答案生成和合成 - 决定最终答案质量
        'response_synthesizer_llm',      # 最终答案生成的LLM
        'template_name',                # 答案合成使用的提示模板
        
        # 高级策略 - 答案生成策略
        'few_shot_enabled'              # Few-shot学习策略
    ]
    
    # 过滤出实际存在于搜索空间的参数
    retrieval_params = [p for p in retrieval_params if p in search_space]
    generation_params = [p for p in generation_params if p in search_space]
    
    logger.info(f"🔍 智能分类 - 检索参数 (使用MRR优化): {retrieval_params}")
    logger.info(f"🎯 智能分类 - 生成参数 (使用{metric}优化): {generation_params}")
    
    # 🔥 检查参数分类的完整性
    total_classified = len(retrieval_params) + len(generation_params)
    total_available = len(search_space)
    if total_classified != total_available:
        missing_params = set(search_space.keys()) - set(retrieval_params) - set(generation_params)
        logger.warning(f"⚠️ 参数分类不完整: {total_classified}/{total_available} 参数已分类")
        logger.warning(f"   未分类的参数: {missing_params}")

    fixed_params: Dict[str, Any] = {}
    iteration_count = 0
    
    # ===== 阶段1：优化检索相关参数，使用MRR指标 =====
    logger.info(f"\n📊 === 阶段1: 智能优化检索参数 (使用MRR指标评估上下文质量) ===")
    
    for i, param_to_optimize in enumerate(retrieval_params):
        logger.info(f"\n[检索阶段 {i+1}/{len(retrieval_params)}] 🎯 正在优化参数: '{param_to_optimize}'")
        
        best_score_for_current_param = -1.0
        best_value_for_current_param = None
        possible_values = search_space[param_to_optimize]
        
        logger.info(f"   参数选项: {len(possible_values)}个 = {possible_values[:5]}{'...' if len(possible_values) > 5 else ''}")
        
        for j, value in enumerate(possible_values):
            # Check max_evals limit before proceeding
            if iteration_count >= max_evals:
                logger.info(f"🛑 达到最大评估次数限制 {max_evals}，提前终止优化")
                break
                
            iteration_count += 1
            logger.info(f"\n   🔧 迭代 {iteration_count}: 测试 {param_to_optimize}={value} ({j+1}/{len(possible_values)})")
            
            temp_params = deepcopy(fixed_params)
            temp_params[param_to_optimize] = value
            
            # 🔥 为尚未确定的参数随机选择值
            logger.info(f"   🎲 为其他参数选择随机值...")
            random_params_count = 0
            for param in retrieval_params + generation_params:
                if param not in temp_params:
                    random_value = random.choice(search_space[param])
                    temp_params[param] = random_value
                    random_params_count += 1
            logger.info(f"   📝 参数配置完整: {len(temp_params)}个参数 ({random_params_count}个随机选择)")
            
            # 🔥 在训练集上评估(使用MRR作为优化指标)
            logger.info(f"   🚀 开始训练集评估 (使用MRR指标)...")
            train_start = time.time()
            try:
                train_metrics = train_evaluate(temp_params)
                train_mrr_score = train_metrics.get('mrr', 0.0)
                train_time = time.time() - train_start
                
                # 🔥 MRR为0的备用策略：使用answer_f1或joint_f1
                if train_mrr_score == 0.0:
                    backup_score = max(
                        train_metrics.get('answer_f1', 0.0),
                        train_metrics.get('joint_f1', 0.0),
                        train_metrics.get('lexical_ac', 0.0)
                    )
                    if backup_score > 0:
                        logger.warning(f"   ⚠️ MRR=0，使用备用指标: {backup_score:.4f}")
                        train_mrr_score = backup_score
                
                logger.info(f"   ✅ 训练集评估完成: MRR={train_mrr_score:.4f}, 耗时={train_time:.1f}s")
                logger.info(f"      其他指标: joint_f1={train_metrics.get('joint_f1', 0):.4f}, answer_f1={train_metrics.get('answer_f1', 0):.4f}")
            except Exception as e:
                logger.error(f"   ❌ 训练集评估失败: {e}")
                train_metrics = {m: 0.0 for m in metric_names}
                train_mrr_score = 0.0
                train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            logger.info(f"   🧪 开始测试集评估 (仅记录)...")
            test_start = time.time()
            try:
                test_metrics = test_evaluate(temp_params)
                test_time = time.time() - test_start
                logger.info(f"   ✅ 测试集评估完成: MRR={test_metrics.get('mrr', 0.0):.4f}, 耗时={test_time:.1f}s")
            except Exception as e:
                logger.error(f"   ❌ 测试集评估失败: {e}")
                test_metrics = {m: 0.0 for m in metric_names}
                test_time = time.time() - test_start
            
            # 🔥 保存训练集结果到CSV
            train_row = [iteration_count, 'mrr']
            train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
            train_row.append(train_time)
            train_row.append(train_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            train_row.append(train_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            import datetime
            import json
            train_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            train_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(train_row)
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count, 'mrr']
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            test_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(test_row)
            
            logger.info(f"   📊 结果已保存到CSV文件")
            
            # 🔥 只用训练集的MRR结果作为优化依据
            if train_mrr_score > best_score_for_current_param:
                logger.info(f"   🎯 发现更优参数值: {param_to_optimize}={value} (MRR: {best_score_for_current_param:.4f} -> {train_mrr_score:.4f})")
                best_score_for_current_param = train_mrr_score
                best_value_for_current_param = value
            else:
                logger.info(f"   📉 当前值未超过最优: {train_mrr_score:.4f} <= {best_score_for_current_param:.4f}")
        
        # Check if we need to break outer loop due to max_evals limit
        if iteration_count >= max_evals:
            logger.info(f"🛑 达到最大评估次数限制 {max_evals}，终止所有参数优化")
            # Use best value found so far, or first value if none tested
            if best_value_for_current_param is None and possible_values:
                best_value_for_current_param = possible_values[0]
                logger.info(f"⚠️ 参数 '{param_to_optimize}' 未完成优化，使用默认值: {best_value_for_current_param}")
            fixed_params[param_to_optimize] = best_value_for_current_param
            break
        
        fixed_params[param_to_optimize] = best_value_for_current_param
        logger.info(f"✅ 检索参数 '{param_to_optimize}' 优化完成")
        logger.info(f"   最佳值: {json.dumps(best_value_for_current_param)}")
        logger.info(f"   最佳训练集MRR分数: {best_score_for_current_param:.4f}")
        logger.info(f"   当前固定参数数量: {len(fixed_params)}")
    
    # ===== 阶段2：优化生成相关参数，使用Answer Correctness指标 =====
    logger.info(f"\n🎯 === 阶段2: 智能优化生成参数 (使用{metric}指标评估答案质量) ===")
    
    # 🔥 创建用于优化生成参数的评估函数
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)
    
    # Check if we still have evaluations left after phase 1
    if iteration_count >= max_evals:
        logger.info(f"🛑 第一阶段已消耗完所有评估次数，跳过第二阶段")
    else:
        for i, param_to_optimize in enumerate(generation_params):
            logger.info(f"\n[生成阶段 {i+1}/{len(generation_params)}] 正在优化参数: '{param_to_optimize}'")
            
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
            
            # 🔥 为尚未确定的生成参数随机选择值
            for param in generation_params:
                if param not in temp_params:
                    temp_params[param] = random.choice(search_space[param])
            
            # 🔥 在训练集上评估(使用目标指标作为优化指标)
            train_start = time.time()
            train_score = train_evaluate_single(temp_params)
            train_metrics = train_evaluate(temp_params)
            train_time = time.time() - train_start
            
            # 🔥 在测试集上评估(仅用于记录)
            test_start = time.time()
            test_metrics = test_evaluate(temp_params)
            test_time = time.time() - test_start
            
            # 🔥 保存训练集结果到CSV
            train_row = [iteration_count, metric]
            train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
            train_row.append(train_time)
            train_row.append(train_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            train_row.append(train_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            train_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            train_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(train_row)
                
            # 🔥 保存测试集结果到CSV
            test_row = [iteration_count, metric]
            test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
            test_row.append(test_time)
            test_row.append(test_metrics.get('total_tokens', 0))  # 🔥 添加token信息
            test_row.append(test_metrics.get('training_samples', 0))  # 🔥 添加样本数信息
            test_row.append(datetime.datetime.now().isoformat())  # 🔥 添加时间戳
            test_row.append(json.dumps(temp_params, ensure_ascii=False, separators=(',', ':')))  # 🔥 添加完整参数组合
            
            with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(test_row)
            
            logger.info(f"   迭代 {iteration_count}: {param_to_optimize}={value}, 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
            
            # 🔥 只用训练集结果作为优化依据
            if train_score > best_score_for_current_param:
                best_score_for_current_param = train_score
                best_value_for_current_param = value
        
        fixed_params[param_to_optimize] = best_value_for_current_param
        logger.info(f"✅ 生成参数 '{param_to_optimize}' 最佳值: {json.dumps(best_value_for_current_param)}")
        logger.info(f"   (获得训练集{metric}分数: {best_score_for_current_param:.4f})")

    logger.info(f"\n🏆 增强版 Greedy-R-CC 搜索完成！")
    logger.info(f"   总迭代次数: {iteration_count}")
    logger.info(f"   阶段1(智能检索参数优化): {len(retrieval_params)}个参数，使用MRR指标")
    logger.info(f"   阶段2(智能生成参数优化): {len(generation_params)}个参数，使用{metric}指标")
    logger.info(f"   基于完整搜索空间分析的最佳参数组合: {json.dumps(fixed_params)}")
    logger.info(f"📋 结果已保存到:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    
    return fixed_params