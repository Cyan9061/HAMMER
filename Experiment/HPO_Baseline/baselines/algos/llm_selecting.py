"""
LLM-Guided Parameter Selection Baseline Algorithm

This algorithm uses Large Language Models (LLMs) to intelligently select RAG hyperparameters
by analyzing historical configurations and their performance scores.

Usage Examples:
# 2WikiMultiHopQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo llm_selecting --dataset 2wikimultihopqa --metric joint_f1 --max_evals 20 --seed 42 &

# HotpotQA dataset 
nohup python Experiment/HPO_Baseline/run_baselines.py --algo llm_selecting --dataset hotpotqa --metric answer_f1 --max_evals 15 --seed 123 &

# MedQA dataset
nohup python Experiment/HPO_Baseline/run_baselines.py --algo llm_selecting --dataset medqa --metric answer_f1 --max_evals 10 --seed 456 &

Available metrics: joint_f1, answer_f1, answer_em, joint_em, lexical_ac, lexical_ff, mrr, rouge_l
Available datasets: 2wikimultihopqa, hotpotqa, medqa, eli5, fiqa, popqa, quartz, webquestions
"""

import sys
import csv
import time
import json
import random
import openai
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.logger import logger
from ..search_space import build_hyperopt_space_from_rag_search_space
from ..objective import make_evaluate_fn

class LLMParameterSelector:
    """LLM-guided parameter selector based on historical performance"""
    
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize LLM client for parameter selection"""
        self.model_name = model_name
        
        # 🔥 使用与insight_agent.py完全相同的gaochao站点配置
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.ai-gaochao.cn/v1")
        
        self.llm_client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.historical_configs = []  # Store (config, score) pairs
        
        logger.info(f"🤖 LLM客户端初始化完成: model={model_name}")
        logger.info(f"   API Base URL: {api_base}")
        logger.info(f"   API密钥: {'✅ 已配置gaochao站点密钥' if 'gaochao' in api_base else '✅ 已配置自定义密钥'}")
        
    def _convert_hyperopt_space(self, hyperopt_space: Dict[str, Any]) -> Tuple[Dict[str, List[Any]], List[str]]:
        """
        Convert hyperopt space to standard Python dictionary format
        Returns: (param_options, param_order)
        """
        param_options: Dict[str, List[Any]] = {}
        param_order: List[str] = list(hyperopt_space.keys())

        for name, expression in hyperopt_space.items():
            if expression.name == 'switch':  # hyperopt uses 'switch' for choices
                try:
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
            elif expression.name == 'choice':
                try:
                    param_options[name] = expression.pos_args[1].obj
                except (AttributeError, IndexError):
                    logger.warning(f"Could not extract options for param '{name}'. Skipping.")
        
        ordered_params = [p for p in param_order if p in param_options]
        return param_options, ordered_params

    def _get_parameter_selection_order(self) -> List[str]:
        """
        Define the order for parameter selection, prioritizing important parameters
        Based on LLM-guided optimization principles
        """
        return [
            # Core model selection (most important)
            'response_synthesizer_llm',
            'embedding_model',
            
            # Retrieval configuration
            'retrieval_method',
            'retrieval_top_k',
            'hybrid_bm25_weight',
            
            # Text processing
            'splitter_method',
            'splitter_chunk_size',
            'splitter_overlap',
            
            # Query processing
            'query_decomposition_num_queries',
            'query_decomposition_llm',
            'fusion_mode',
            
            # Enhancement modules
            'reranker_llm',
            'reranker_top_k',
            'additional_context_num_nodes',
            
            # Template and formatting
            'template_name',
            
            # Boolean switches
            'query_decomposition_enabled',
            'hyde_enabled',
            'reranker_enabled',
            'additional_context_enabled',
            'few_shot_enabled',
        ]

    def _build_selection_prompt(self, 
                               parameter_name: str,
                               current_params: Dict[str, Any],
                               available_options: List[Any],
                               historical_configs: List[Tuple[Dict[str, Any], float]]) -> str:
        """Build prompt for LLM parameter selection"""
        
        # Format available options with indices
        options_text = ""
        for i, option in enumerate(available_options):
            options_text += f"{i}: {option}\n"
        
        # Format historical configurations (top 5 best)
        historical_text = ""
        if historical_configs:
            # Sort by score (descending) and take top 5
            sorted_configs = sorted(historical_configs, key=lambda x: x[1], reverse=True)[:5]
            for i, (config, score) in enumerate(sorted_configs, 1):
                param_value = config.get(parameter_name, "N/A")
                historical_text += f"- Config {i}: {parameter_name}={param_value}, Score={score:.4f}\n"
        else:
            historical_text = "- No historical configurations available yet\n"
        
        # Current configuration context
        current_config_text = json.dumps(current_params, indent=2) if current_params else "{}"
        
        prompt = f"""You are an expert in RAG (Retrieval-Augmented Generation) system optimization. Your task is to select the best parameter value based on historical performance data.

## Current Configuration:
```json
{current_config_text}
```

## Parameter to Select: {parameter_name}

## Available Options:
{options_text}

## Historical Performance Data:
{historical_text}

## Task:
Based on the historical performance patterns and RAG system optimization principles, select the option number that would likely achieve the highest performance score.

Consider these factors:
1. **Parameter Interactions**: How this parameter works with already selected parameters
2. **Historical Patterns**: Which values performed well in similar configurations
3. **RAG Optimization Principles**: General best practices for this parameter type
4. **Performance Trends**: What the data suggests about parameter effectiveness

## Response Format:
Respond with ONLY the option number (0, 1, 2, etc.). Do not include any explanation or additional text.

Your selection:"""
        
        return prompt

    def select_parameter_value(self,
                             parameter_name: str,
                             current_params: Dict[str, Any],
                             available_options: List[Any],
                             historical_configs: List[Tuple[Dict[str, Any], float]]) -> Any:
        """
        Use LLM to select the best parameter value
        Returns the selected value, or first option if LLM fails
        """
        logger.info(f"🤖 LLM Parameter Selection: {parameter_name}")
        logger.info(f"   Available options: {len(available_options)}")
        logger.info(f"   Historical configs: {len(historical_configs)}")
        
        try:
            prompt = self._build_selection_prompt(
                parameter_name, current_params, available_options, historical_configs
            )
            
            logger.info(f"🚀 Sending LLM request for {parameter_name}...")
            start_time = time.time()
            
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent selection
                max_tokens=10,    # Very short response needed
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            chosen_response = response.choices[0].message.content.strip()
            logger.info(f"✅ LLM response: '{chosen_response}' (time: {response_time:.2f}s)")
            
            # Parse the response
            try:
                chosen_index = int(chosen_response)
                if 0 <= chosen_index < len(available_options):
                    selected_value = available_options[chosen_index]
                    logger.info(f"🎯 Selected: {parameter_name}={selected_value}")
                    return selected_value
                else:
                    logger.warning(f"⚠️ LLM returned out-of-range index: {chosen_index}")
                    return available_options[0]
            except ValueError:
                logger.warning(f"⚠️ LLM returned non-numeric response: '{chosen_response}'")
                return available_options[0]
                
        except Exception as e:
            logger.error(f"❌ LLM parameter selection failed: {e}")
            return available_options[0]  # Default to first option

    def record_configuration(self, config: Dict[str, Any], score: float):
        """Record a configuration and its performance score"""
        self.historical_configs.append((deepcopy(config), score))
        logger.info(f"📝 Recorded config with score {score:.4f} (total: {len(self.historical_configs)})")

def run_llm_selecting(ss, qa_train, qa_test, max_evals=10, seed=42, metric='joint_f1', dataset_name='unknown'):
    """
    LLM-guided parameter selection baseline
    
    Args:
        ss: Search space (used for parameter extraction)
        qa_train: Training dataset for optimization
        qa_test: Test dataset for evaluation (not used for optimization)
        max_evals: Maximum evaluations
        seed: Random seed
        metric: Optimization metric
        dataset_name: Dataset name for CSV file naming
    """
    logger.info(f"🤖 开始LLM-guided参数选择: max_evals={max_evals}, seed={seed}, metric={metric}, dataset={dataset_name}")
    random.seed(seed)
    
    # 🔥 生成时间戳用于CSV文件命名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🔥 创建CSV文件路径 - 保存到ablation消融实验目录（与Memory消融实验一致）
    output_dir = Path("Experiment/ablation/llm_selection/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_file = output_dir / f"train_llm_selecting_{dataset_name}_{timestamp}.csv"
    test_csv_file = output_dir / f"test_llm_selecting_{dataset_name}_{timestamp}.csv"
    
    # 🔥 初始化CSV文件
    metric_names = ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']
    csv_headers = ['iteration'] + metric_names + ['eval_time', 'total_tokens', 'training_samples', 'timestamp', 'parameters']
    
    with open(train_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # Initialize LLM selector and search space
    llm_selector = LLMParameterSelector()
    hyperopt_space = build_hyperopt_space_from_rag_search_space()
    
    # 🔥 创建训练集和测试集的评估函数
    train_evaluate = make_evaluate_fn(qa_train, metric, return_all_metrics=True, dataset_name=dataset_name)
    test_evaluate = make_evaluate_fn(qa_test, metric, return_all_metrics=True, dataset_name=dataset_name)
    train_evaluate_single = make_evaluate_fn(qa_train, metric, return_all_metrics=False, dataset_name=dataset_name)

    search_space, original_order = llm_selector._convert_hyperopt_space(hyperopt_space)
    parameter_order = llm_selector._get_parameter_selection_order()
    
    # Filter to only include parameters that exist in search space
    parameter_order = [p for p in parameter_order if p in search_space]
    # Add any remaining parameters not in the predefined order
    remaining_params = [p for p in original_order if p not in parameter_order]
    parameter_order.extend(remaining_params)
    
    logger.info(f"🔍 搜索空间转换完成，将按LLM-guided顺序选择 {len(parameter_order)} 个参数")
    logger.info(f"📋 参数选择顺序: {parameter_order}")
    
    best_score = -1.0
    best_config = {}
    iteration_count = 0
    
    # Main optimization loop
    for eval_num in range(max_evals):
        iteration_count += 1
        logger.info(f"\n🔄 === 第 {iteration_count}/{max_evals} 次评估 ===")
        
        # Build configuration using LLM guidance
        current_config = {}
        
        for param_name in parameter_order:
            if param_name in search_space:
                available_options = search_space[param_name]
                
                # LLM selects the parameter value
                selected_value = llm_selector.select_parameter_value(
                    param_name, 
                    current_config, 
                    available_options, 
                    llm_selector.historical_configs
                )
                
                current_config[param_name] = selected_value
                logger.info(f"   ✅ {param_name} = {selected_value}")
        
        logger.info(f"🎯 LLM构建的完整配置: {json.dumps(current_config, separators=(',', ':'))}")
        
        # 🔥 在训练集上评估(用于优化)
        train_start = time.time()
        try:
            train_score = train_evaluate_single(current_config)
            train_metrics = train_evaluate(current_config)
        except Exception as e:
            logger.error(f"❌ 训练集评估失败: {e}")
            train_score = 0.0
            train_metrics = {m: 0.0 for m in metric_names}
        train_time = time.time() - train_start
        
        # 🔥 在测试集上评估(仅用于记录)
        test_start = time.time()
        try:
            test_metrics = test_evaluate(current_config)
        except Exception as e:
            logger.error(f"❌ 测试集评估失败: {e}")
            test_metrics = {m: 0.0 for m in metric_names}
        test_time = time.time() - test_start
        
        # Record this configuration for future LLM guidance
        llm_selector.record_configuration(current_config, train_score)
        
        # Update best configuration
        if train_score > best_score:
            best_score = train_score
            best_config = deepcopy(current_config)
            logger.info(f"🏆 新的最佳配置! {metric}={best_score:.4f}")
        
        # 🔥 保存训练集结果到CSV
        import datetime
        current_timestamp = datetime.datetime.now().isoformat()
        train_row = [iteration_count]
        train_row.extend([train_metrics.get(m, 0.0) for m in metric_names])
        train_row.append(train_time)
        train_row.append(train_metrics.get('total_tokens', 0))
        train_row.append(train_metrics.get('training_samples', 0))
        train_row.append(current_timestamp)
        train_row.append(json.dumps(current_config, ensure_ascii=False, separators=(',', ':')))
        
        with open(train_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(train_row)
            
        # 🔥 保存测试集结果到CSV
        test_row = [iteration_count]
        test_row.extend([test_metrics.get(m, 0.0) for m in metric_names])
        test_row.append(test_time)
        test_row.append(test_metrics.get('total_tokens', 0))
        test_row.append(test_metrics.get('training_samples', 0))
        test_row.append(current_timestamp)
        test_row.append(json.dumps(current_config, ensure_ascii=False, separators=(',', ':')))
        
        with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(test_row)
        
        logger.info(f"📊 评估完成: 训练{metric}={train_score:.4f}, 测试{metric}={test_metrics.get(metric, 0.0):.4f}")
        logger.info(f"⏱️ 训练耗时={train_time:.2f}s, 测试耗时={test_time:.2f}s")

    logger.info(f"\n🏆 LLM-guided搜索完成！")
    logger.info(f"   总迭代次数: {iteration_count}")
    logger.info(f"   最佳训练得分: {best_score:.4f}")
    logger.info(f"   最佳配置: {json.dumps(best_config)}")
    logger.info(f"📋 结果已保存到:")
    logger.info(f"   训练集: {train_csv_file}")
    logger.info(f"   测试集: {test_csv_file}")
    logger.info(f"💾 LLM Selection消融实验结果: Experiment/ablation/llm_selection/csv/")
    
    return best_config