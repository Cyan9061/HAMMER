"""hammer/tuner/main_tuner_mcts.py
Main tuner for RAG system optimization using True MCTS algorithm.

Usage:
# 基本用法示例 - 使用默认优化目标 train_answer_f1

nohup python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_2wikimultihopqa_f1_50iter.csv > Experiment/log/mcts_2wikimultihopqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset hotpotqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_hotpotqa_f1_50iter.csv > Experiment/log/mcts_hotpotqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset MedQA --iterations 50 --optimization-target train_answer_f1 --train-size 267 --csv-file Experiment/MCTS_csv/mcts_MedQA_f1_50iter.csv > Experiment/log/mcts_MedQA_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset eli5 --iterations 50 --optimization-target train_answer_f1 --train-size 317 --csv-file Experiment/MCTS_csv/mcts_eli5_f1_50iter.csv > Experiment/log/mcts_eli5_f1_50iter.log 2>&1 &

nohup python -m hammer.tuner.main_tuner_mcts --dataset fiqa --iterations 50 --optimization-target train_answer_f1 --train-size 105 --csv-file Experiment/MCTS_csv/mcts_fiqa_f1_50iter.csv > Experiment/log/mcts_fiqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset quartz --iterations 50 --optimization-target train_answer_f1 --train-size 192 --csv-file Experiment/MCTS_csv/mcts_quartz_f1_50iter.csv > Experiment/log/mcts_quartz_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset webquestions --iterations 50 --optimization-target train_answer_f1 --train-size 426 --csv-file Experiment/MCTS_csv/mcts_webquestions_f1_50iter.csv > Experiment/log/mcts_webquestions_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset popqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_popqa_f1_50iter.csv > Experiment/log/mcts_popqa_f1_50iter.log 2>&1 &

nohup python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_2wikimultihopqa_f1_50iter.csv > Experiment/log/mcts_2wikimultihopqa_f1_50iter_Qwen7b_debug.log 2>&1 &

# 优化目标说明:
# train_answer_f1: 训练集答案F1分数（推荐）
# train_joint_f1: 训练集联合F1分数（默认）
# train_lexical_ac: 训练集词汇答案覆盖度
# train_lexical_ff: 训练集词汇忠实度
# train_mrr: 训练集平均倒数排名
# train_answer_em: 训练集答案精确匹配
# train_joint_em: 训练集联合精确匹配
    
"""

MODEL_MAXWORKERS = 6
DEFAULT_TRAIN_SIZE = 210
"""
默认训练集大小（可通过--train-size参数修改）:
210 2wikimultihopqa
210 hotpotqa
267 MedQA
105 fiqa
192 quartz
426 webquestions
317 eli5
210 popqa

"""
USE_CORESET = False
DEFAULT_CORESET_RATIO = 1
SAVE_CSV_TPE = False
MODEL_NAME="Qwen2-7b"  # 🔥 修改为支持Qwen2-7b模型
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

MODEL_SIMUL="gpt-4o-mini"  # 保留但不使用
# MCTS_CSV_FILE="Experiment/mcts_MedQA_ff.csv"

import csv
import argparse
import json
import os
import sys
import typing as T
from datetime import datetime, timezone
from pathlib import Path

# 🔧 修复CUDA设备冲突：正确处理CUDA_VISIBLE_DEVICES映射
# 当设置CUDA_VISIBLE_DEVICES时，PyTorch会将可见设备重新编号为0,1,2...
# 所以我们应该使用逻辑设备号0，而不是物理设备号
DEVICE_ID = 0 #if os.environ.get('CUDA_VISIBLE_DEVICES') else 0
GPU_QUERY_EMBED_LIST=[DEVICE_ID]#[4,5,6,7]
GPU_BATCHSIZE=128
GPU_TEXT_EMBED= DEVICE_ID

# Import dataset-specific prompts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from docs.dataset.dataset_main_prompt import get_dataset_prompt, validate_dataset_name

import optuna
from hammer.logger import logger
from hammer.flows import Flow

# Import dataset loader
from hammer.mcts.mcts_dataset_loader import create_simple_dataset

# Import shared components from TPE tuner (only what we need)
from hammer.tuner.main_tuner_tpe import (
    FlowBuilder,
    # EvaluationManager,
    # TrialManager,
    prepare_worker
)

from hammer.tuner.cuda_cleaner import CUDACleaner

# Import enhanced MCTS optimization engine
from hammer.mcts.optimization_engine import EnhancedMCTSOptimizationEngine
from hammer.mcts.kb_manager.graph_memory import GraphMemoryRAGMCTS

# Simple dataset configuration to replace StudyConfig
class SimpleSearchSpace:
    """简化的搜索空间类 - 兼容FlowBuilder"""
    
    def is_few_shot(self, params: T.Dict) -> bool:
        """检查是否启用few-shot"""
        return params.get("few_shot_enabled", False)

class SimpleTimeoutConfig:
    """简化的超时配置类 - 兼容build_rag_retriever"""
    
    def __init__(self):
        self.embedding_timeout_active = False
        self.embedding_max_time = 3600 * 4
        self.embedding_min_chunks_to_process = 100
        self.embedding_min_time_to_process = 120
        self.eval_timeout = 3600 * 10
        self.single_eval_timeout = 3600 * 2
        self.onnx_timeout = 600

class SimpleOptimizationConfig:
    """简化的优化配置类 - 兼容build_rag_retriever和optimization.py"""
    
    def __init__(self):
        self.embedding_device = GPU_TEXT_EMBED  # GPU设备ID
        self.use_hf_embedding_models = False  # 是否使用HuggingFace embedding models
        self.num_trials = 100
        self.cpus_per_trial = 2
        self.gpus_per_trial = 0.0
        
        # 添加optimization.py需要的额外属性
        self.objective_1_name = "answer_f1"  # 主要目标
        self.objective_2_name = None  # 单目标优化，所以为None
        self.seeder_timeout = 300
        self.method = "expanding"
        self.blocks = []  # 空的优化块列表
        self.shuffle_blocks = False
        self.max_concurrent_trials = 10
        self.raise_on_failed_trial = False
        self.pareto_eval_success_rate = 0.8

class SimpleDatasetConfig:
    """简化的数据集配置类 - 替代StudyConfig"""
    
    def __init__(self, dataset_name: str, train_size: int = DEFAULT_TRAIN_SIZE):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.name = f"mcts-{dataset_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 🔥 # 🔥 使用Query Selection优化后的数据集文件路径
        # 确保除了数据划分方式外，两个系统使用相同的数据集内容
        self.corpus_file = f"docs/dataset/unified_query_selection/{dataset_name}_corpus_unified.json"
        self.qa_file = f"docs/dataset/unified_query_selection/{dataset_name}_qa_unified.json"
        
        # 创建数据集对象
        self.dataset = create_simple_dataset(self.corpus_file, self.qa_file, dataset_name)
        
        # 基本配置
        self.max_workers = MODEL_MAXWORKERS
        
        # ✅ 添加兼容TPE FlowBuilder的search_space属性
        self.search_space = SimpleSearchSpace()
        
        # ✅ 添加兼容build_rag_retriever的model_config属性 (用于debug日志)
        self.model_config = {"extra": "forbid", "yaml_file": None}
        
        # ✅ 添加兼容build_rag_retriever的其他必需属性
        self.toy_mode = False  # 非玩具模式
        
        # 创建简化的timeouts配置
        self.timeouts = SimpleTimeoutConfig()
        
        # 创建简化的optimization配置  
        self.optimization = SimpleOptimizationConfig()
        
        logger.info(f"📊 数据集配置: {dataset_name}")
        logger.info(f"  语料库: {self.corpus_file}")
        logger.info(f"  问答对: {self.qa_file}")
        logger.info(f"  训练集大小: {self.train_size}")

    def iter_examples(self, partition="test"):
        """兼容原始StudyConfig接口 - 添加partition参数兼容性"""
        # 在MCTS中我们不区分partition，都使用相同的数据集
        return self.dataset.load_qa_pairs()

# MCTS专用评估管理器 - 简化版本
class MCTSEvaluationManager:
    """MCTS专用评估管理器，使用简化的数据集配置"""
    
    def __init__(self, dataset_config: SimpleDatasetConfig, save_csv_tpe: bool = True, optimization_target: str = 'train_joint_f1'):
        self.dataset_config = dataset_config
        self.save_csv_tpe = save_csv_tpe
        self.optimization_target = optimization_target
        logger.info(f"🎯 MCTS评估管理器初始化，训练集大小: {dataset_config.train_size}")
        logger.info(f"🎯 优化目标: {optimization_target}")
    
    def evaluate_flow(self, flow: Flow) -> T.Tuple[float, T.Dict[str, T.Any]]:
        """评估流程并返回目标值和详细结果"""
        # 使用MCTS专用的多跳评估策略
        evaluation_strategy = MCTSMultiHopEvaluationStrategy()
        results = evaluation_strategy.evaluate_flow(flow, self.dataset_config, self.save_csv_tpe)
        
        # 🔥 根据配置的优化目标提取目标值
        objective_value = self._extract_objective_value(results)
        
        return objective_value, results
    
    def _extract_objective_value(self, results: T.Dict[str, T.Any]) -> float:
        """根据配置的优化目标提取目标值"""
        # 🎯 可用的训练相关优化目标
        available_targets = {
            'train_answer_em': 'Train Answer Exact Match',
            'train_answer_f1': 'Train Answer F1 Score', 
            'train_joint_em': 'Train Joint Exact Match',
            'train_joint_f1': 'Train Joint F1 Score',
            'train_lexical_ac': 'Train Lexical Answer Coverage',
            'train_lexical_ff': 'Train Lexical Faithfulness',
            'train_mrr': 'Train Mean Reciprocal Rank',
            'train_rouge_l': 'Train ROUGE-L Score'  # 🔥 新增ROUGE-L优化目标
        }
        
        if self.optimization_target not in available_targets:
            logger.warning(f"⚠️ 未知的优化目标: {self.optimization_target}，使用默认目标 train_joint_f1")
            self.optimization_target = 'train_joint_f1'
        
        # 直接使用配置的优化目标
        target_key = self.optimization_target
        if target_key not in results:
            logger.warning(f"⚠️ 目标键 {self.optimization_target} 不存在于结果中")
            # 列出可用的键供调试
            available_keys = [k for k in results.keys() if 'train' in k or any(metric in k for metric in ['f1', 'em', 'ac', 'ff', 'mrr'])]
            logger.warning(f"⚠️ 可用的相关键: {available_keys}")
            # 使用默认值
            target_key = 'train_joint_f1' if 'train_joint_f1' in results else 'joint_f1'
        
        objective_value = results.get(target_key, 0.0)
        target_description = available_targets.get(self.optimization_target, self.optimization_target)
        
        logger.info(f"🎯 优化目标 '{target_description}' ({target_key}): {objective_value:.4f}")
        return objective_value

# 在main_tuner_mcts.py中修改MCTSMultiHopEvaluationStrategy类

class MCTSMultiHopEvaluationStrategy:
    """MCTS专用多跳评估策略，确保正确提取QA执行日志并支持Coreset加权"""
    
    def evaluate_flow(self, flow: Flow, dataset_config: SimpleDatasetConfig, save_csv_tpe=False) -> T.Dict[str, T.Any]:
        """MCTS版本的流程评估，支持Coreset加权计算训练集指标"""
        
        from hammer.utils.simple_token_tracker import get_token_statistics, clear_token_usage, print_debug_info
        from hammer.multihop_evaluation import MultiHopQAEvaluator
        from hammer.utils.optimized_rag_prompt_builder import create_optimized_rag_prompt_builder
        from hammer.utils.batch_api_evaluator import create_batch_api_evaluator
        import time
        import numpy as np
        import json
        
        # 🔥 注释掉token清空 - 保留MCTS搜索阶段记录的Agent token
        # clear_token_usage()  # 不再清空，让Agent和RAG token累积
        
        if hasattr(flow, 'params') and flow.params:
            logger.info(f"🔧 MCTS评估开始 | 配置: {json.dumps(flow.params, ensure_ascii=False, separators=(',', ':'))}")
        else:
            logger.info("🔧 MCTS评估开始 | ⚠️ 无配置参数")

        eval_start = time.time()
        
        # 使用MCTS专用的训练集大小
        all_qa_pairs = list(dataset_config.iter_examples())
        total_qa_count = len(all_qa_pairs)
        train_qa_pairs = all_qa_pairs[:min(dataset_config.train_size, total_qa_count)]
        test_qa_pairs = all_qa_pairs[dataset_config.train_size:] if total_qa_count > dataset_config.train_size else []
        
        logger.info(f"🎯 MCTS数据划分：总共{total_qa_count}个QA对，训练集{len(train_qa_pairs)}个(MCTS目标)，测试集{len(test_qa_pairs)}个")
        
        # 阶段一：批量embedding和RAG构建（全部数据）
        logger.info("🏗️ 阶段一：开始批量embedding,coreset和RAG构建...")
        batch_rag_result = None
        
        try:
            all_questions = [qa.question for qa in all_qa_pairs]
            optimized_rag_builder = create_optimized_rag_prompt_builder(flow, max_workers=MODEL_MAXWORKERS)
            full_rag_result = optimized_rag_builder.batch_build_prompts(
                all_questions,
                is_coreset=USE_CORESET,
                coreset_size=min(int(dataset_config.train_size * DEFAULT_CORESET_RATIO), len(train_qa_pairs)),
                train_data_size=len(train_qa_pairs)
            )
            batch_rag_result = full_rag_result
            
            # 🔥 关键修改：提取Coreset信息以备后续加权计算
            coreset_weights = None
            coreset_used = False
            original_train_size = len(train_qa_pairs)
            
            if (USE_CORESET and 
                hasattr(full_rag_result, 'coreset_result') and 
                full_rag_result.coreset_result is not None):
                
                coreset_result = full_rag_result.coreset_result
                coreset_weights = np.array(coreset_result.weights, dtype=np.float64)
                coreset_used = True
                
                original_coreset_indices = full_rag_result.coreset_train_indices
                sorted_coreset_indices = sorted(original_coreset_indices)
                
                logger.info(f"🎯 Coreset信息:")
                logger.info(f"   原始索引: {original_coreset_indices}")
                logger.info(f"   排序索引: {sorted_coreset_indices}")
                logger.info(f"   权重数组: {coreset_weights}")
                logger.info(f"   权重总和: {np.sum(coreset_weights)}")
                logger.info(f"   原始训练集大小: {original_train_size}")
                
                # 🔥 关键修复：同步调整QA pairs和prompts
                # 1. 调整训练集QA pairs
                train_qa_pairs = [train_qa_pairs[i] for i in sorted_coreset_indices]
                
                # 2. 🔥 同步调整prompts - 只保留选中的训练prompts + 所有测试prompts
                selected_train_prompts = [batch_rag_result.final_prompts[i] for i in sorted_coreset_indices]
                test_prompts = batch_rag_result.final_prompts[len(all_qa_pairs[:dataset_config.train_size]):]  # 原始测试集prompts
                
                # 3. 重构prompts数组以匹配新的qa_pairs
                batch_rag_result.final_prompts = selected_train_prompts + test_prompts
                
                # 4. 重构all_qa_pairs
                all_qa_pairs = train_qa_pairs + test_qa_pairs
                
                logger.info(f"🎯 Coreset重构完成：")
                logger.info(f"   训练集：{len(train_qa_pairs)}个QA pairs，{len(selected_train_prompts)}个prompts")
                logger.info(f"   测试集：{len(test_qa_pairs)}个QA pairs，{len(test_prompts)}个prompts")
                logger.info(f"   总计：{len(all_qa_pairs)}个QA pairs，{len(batch_rag_result.final_prompts)}个prompts")
                
                # 🔥 关键验证：确保数量匹配
                if len(all_qa_pairs) != len(batch_rag_result.final_prompts):
                    raise ValueError(f"数量不匹配：QA pairs({len(all_qa_pairs)}) vs prompts({len(batch_rag_result.final_prompts)})")
                
                # 验证权重与重构后的训练集大小匹配
                if len(coreset_weights) != len(train_qa_pairs):
                    raise ValueError(f"权重数量不匹配：weights({len(coreset_weights)}) vs train_qa_pairs({len(train_qa_pairs)})")
                    
            else:
                logger.info(f"🎯 未使用Coreset（USE_CORESET={USE_CORESET}），保持原始数据顺序")

        except Exception as e:
            logger.error(f"❌ 阶段一失败: {e}")
            return {
                "failed": True,
                "exception_message": f"MCTS RAG构建失败: {e}",
                "flow_start": eval_start,
                "flow_end": time.time(),
                "flow_duration": time.time() - eval_start,
                "joint_f1": 0.0,
                "answer_f1": 0.0,
                "train_joint_f1": 0.0,
                "train_answer_f1": 0.0,
                "accuracy": 0.0,
                "qa_execution_logs": [],
            }
        
        # 确保阶段一成功
        if batch_rag_result is None:
            logger.error("❌ 阶段一失败，batch_rag_result为None")
            return {
                "failed": True,
                "exception_message": "MCTS RAG构建失败，batch_rag_result为None",
                "flow_start": eval_start,
                "flow_end": time.time(),
                "flow_duration": time.time() - eval_start,
                "joint_f1": 0.0,
                "answer_f1": 0.0,
                "train_joint_f1": 0.0,
                "train_answer_f1": 0.0,
                "accuracy": 0.0,
                "qa_execution_logs": [],
            }
        
        # 阶段二：批量API调用和评估
        logger.info("🚀 阶段二：开始批量API调用和评估...")
        
        batch_result = None
        exception_message = ""
        
        try:
            # 🔥 获取corpus mapping用于修复指标计算
            corpus_mapping = dataset_config.dataset._load_corpus_mapping()
            logger.info(f"📚 获取corpus mapping，共 {len(corpus_mapping)} 个文档")
            
            multihop_evaluator = MultiHopQAEvaluator(corpus_lookup=corpus_mapping)
            batch_evaluator = create_batch_api_evaluator(
                model_name=MODEL_NAME,
                max_workers=MODEL_MAXWORKERS,
                multihop_evaluator=multihop_evaluator
            )
            
            # 将Flow参数传递给批量评估器，确保RAG配置能被正确提取
            batch_evaluator.current_flow = flow
            batch_evaluator.current_config = getattr(flow, 'params', {})
            
            batch_result = batch_evaluator.evaluate_batch_optimized(batch_rag_result, all_qa_pairs)
            logger.info(f"✅ 阶段二完成：评估了{batch_result.total_count}个响应")
            
        except Exception as e:
            exception_message = f"MCTS批量API评估失败: {e}"
            logger.error(f"❌ 阶段二失败: {e}", exc_info=True)

        # 统一处理阶段二的失败情况
        if batch_result is None:
            return {
                "failed": True,
                "exception_message": exception_message or "MCTS批量API评估失败，batch_result为None",
                "flow_start": eval_start,
                "flow_end": time.time(),
                "flow_duration": time.time() - eval_start,
                "joint_f1": 0.0,
                "answer_f1": 0.0,
                "train_joint_f1": 0.0,
                "train_answer_f1": 0.0,
                "accuracy": 0.0,
                "qa_execution_logs": [],
            }

        eval_end = time.time()
        eval_duration = eval_end - eval_start
        
        # 🎯 按要求：只统计RAG_train_token和Agent_token
        agent_tokens, rag_tokens = get_token_statistics()
        
        # 计算训练集RAG token（按训练集在总数据中的实际比例）
        train_ratio = len(train_qa_pairs) / total_qa_count if total_qa_count > 0 else 0.0
        RAG_train_token = int(rag_tokens["total"] * train_ratio)
        Agent_token = agent_tokens["total"]
        
        logger.info(f"🎯 Token统计 (按要求):")
        logger.info(f"   Agent_token: {Agent_token} tokens ({agent_tokens['calls']}次调用)")
        logger.info(f"   RAG总token: {rag_tokens['total']} tokens ({rag_tokens['calls']}次调用)")
        logger.info(f"   训练集比例: {train_ratio:.4f} (训练集{len(train_qa_pairs)}/总数据{total_qa_count})")
        logger.info(f"   RAG_train_token: {RAG_train_token} tokens")

        # 使用MCTS版本的数据划分计算指标
        train_count = len(train_qa_pairs)
        test_count = len(test_qa_pairs)
        
        # 分离训练集和测试集的结果
        train_answer_ems = batch_result.answer_ems[:train_count] if batch_result.answer_ems else []
        train_answer_f1s = batch_result.answer_f1s[:train_count] if batch_result.answer_f1s else []
        train_joint_ems = batch_result.joint_ems[:train_count] if batch_result.joint_ems else []
        train_joint_f1s = batch_result.joint_f1s[:train_count] if batch_result.joint_f1s else []
        
        # 🔥 新增：分离统一评估指标
        train_lexical_acs = batch_result.lexical_acs[:train_count] if batch_result.lexical_acs else []
        train_lexical_ffs = batch_result.lexical_ffs[:train_count] if batch_result.lexical_ffs else []
        train_mrrs = batch_result.mrrs[:train_count] if batch_result.mrrs else []
        train_rouge_ls = batch_result.rouge_ls[:train_count] if batch_result.rouge_ls else []  # 🔥 新增ROUGE-L分离
        
        test_answer_ems = batch_result.answer_ems[train_count:] if batch_result.answer_ems and test_count > 0 else []
        test_answer_f1s = batch_result.answer_f1s[train_count:] if batch_result.answer_f1s and test_count > 0 else []
        test_joint_ems = batch_result.joint_ems[train_count:] if batch_result.joint_ems and test_count > 0 else []
        test_joint_f1s = batch_result.joint_f1s[train_count:] if batch_result.joint_f1s and test_count > 0 else []
        
        # 🔥 新增：测试集统一评估指标
        test_lexical_acs = batch_result.lexical_acs[train_count:] if batch_result.lexical_acs and test_count > 0 else []
        test_lexical_ffs = batch_result.lexical_ffs[train_count:] if batch_result.lexical_ffs and test_count > 0 else []
        test_mrrs = batch_result.mrrs[train_count:] if batch_result.mrrs and test_count > 0 else []
        test_rouge_ls = batch_result.rouge_ls[train_count:] if batch_result.rouge_ls and test_count > 0 else []  # 🔥 新增ROUGE-L测试集分离
        
        logger.info(f"train_answer_f1s = {train_answer_f1s}")
        logger.info(f"test_answer_f1s = {test_answer_f1s}")
        
        # 🔥 关键修改：按Coreset权重计算训练集指标（包括统一评估指标）
        if coreset_used and coreset_weights is not None:
            logger.info("🎯 使用Coreset权重计算训练集指标")
            
            # 验证数据一致性
            if len(coreset_weights) != len(train_answer_f1s):
                logger.error(f"❌ 权重与结果数量不匹配: weights({len(coreset_weights)}) vs results({len(train_answer_f1s)})")
                raise ValueError(f"权重与结果数量不匹配")
            
            # 归一化权重
            normalized_weights = coreset_weights / np.sum(coreset_weights)
            logger.info(f"🎯 归一化权重: {normalized_weights}")
            logger.info(f"🎯 权重总和验证: {np.sum(normalized_weights):.6f}")
            
            # 加权平均计算训练集指标
            train_answer_em = np.average(train_answer_ems, weights=normalized_weights) if train_answer_ems else 0.0
            train_answer_f1 = np.average(train_answer_f1s, weights=normalized_weights) if train_answer_f1s else 0.0
            train_joint_em = np.average(train_joint_ems, weights=normalized_weights) if train_joint_ems else 0.0
            train_joint_f1 = np.average(train_joint_f1s, weights=normalized_weights) if train_joint_f1s else 0.0
            
            # 🔥 新增：加权计算统一评估指标
            train_lexical_ac = np.average(train_lexical_acs, weights=normalized_weights) if train_lexical_acs else 0.0
            train_lexical_ff = np.average(train_lexical_ffs, weights=normalized_weights) if train_lexical_ffs else 0.0
            train_mrr = np.average(train_mrrs, weights=normalized_weights) if train_mrrs else 0.0
            train_rouge_l = np.average(train_rouge_ls, weights=normalized_weights) if train_rouge_ls else 0.0  # 🔥 新增ROUGE-L加权计算
            
            logger.info(f"📊 Coreset加权训练集指标:")
            logger.info(f"   加权answer_f1={train_answer_f1:.4f} (vs 简单均值={np.mean(train_answer_f1s):.4f})")
            logger.info(f"   加权joint_f1={train_joint_f1:.4f} (vs 简单均值={np.mean(train_joint_f1s):.4f})")
            logger.info(f"   加权lexical_ac={train_lexical_ac:.4f} (vs 简单均值={np.mean(train_lexical_acs) if train_lexical_acs else 0:.4f})")
            logger.info(f"   加权lexical_ff={train_lexical_ff:.4f} (vs 简单均值={np.mean(train_lexical_ffs) if train_lexical_ffs else 0:.4f})")
            logger.info(f"   加权mrr={train_mrr:.4f} (vs 简单均值={np.mean(train_mrrs) if train_mrrs else 0:.4f})")
            logger.info(f"   加权rouge_l={train_rouge_l:.4f} (vs 简单均值={np.mean(train_rouge_ls) if train_rouge_ls else 0:.4f})")  # 🔥 新增ROUGE-L日志
            
        else:
            logger.info("🎯 使用简单均值计算训练集指标（未使用Coreset）")
            # 原始计算方式：简单平均
            train_answer_em = np.mean(train_answer_ems) if train_answer_ems else 0.0
            train_answer_f1 = np.mean(train_answer_f1s) if train_answer_f1s else 0.0
            train_joint_em = np.mean(train_joint_ems) if train_joint_ems else 0.0
            train_joint_f1 = np.mean(train_joint_f1s) if train_joint_f1s else 0.0
            
            # 🔥 新增：简单均值计算统一评估指标
            train_lexical_ac = np.mean(train_lexical_acs) if train_lexical_acs else 0.0
            train_lexical_ff = np.mean(train_lexical_ffs) if train_lexical_ffs else 0.0
            train_mrr = np.mean(train_mrrs) if train_mrrs else 0.0
            train_rouge_l = np.mean(train_rouge_ls) if train_rouge_ls else 0.0  # 🔥 新增ROUGE-L简单均值计算
        
        # 计算测试集指标（验证性能）- 始终使用简单平均
        test_answer_em = np.mean(test_answer_ems) if test_answer_ems else 0.0
        test_answer_f1 = np.mean(test_answer_f1s) if test_answer_f1s else 0.0
        test_joint_em = np.mean(test_joint_ems) if test_joint_ems else 0.0
        test_joint_f1 = np.mean(test_joint_f1s) if test_joint_f1s else 0.0
        
        # 🔥 新增：测试集统一评估指标
        test_lexical_ac = np.mean(test_lexical_acs) if test_lexical_acs else 0.0
        test_lexical_ff = np.mean(test_lexical_ffs) if test_lexical_ffs else 0.0
        test_mrr = np.mean(test_mrrs) if test_mrrs else 0.0
        test_rouge_l = np.mean(test_rouge_ls) if test_rouge_ls else 0.0  # 🔥 新增ROUGE-L测试集计算
        
        logger.info(f"📊 MCTS训练集指标({len(train_qa_pairs)}条): answer_f1={train_answer_f1:.4f}, joint_f1={train_joint_f1:.4f}, lexical_ac={train_lexical_ac:.4f}, lexical_ff={train_lexical_ff:.4f}, mrr={train_mrr:.4f}, rouge_l={train_rouge_l:.4f}")
        logger.info(f"📊 测试集指标({test_count}条): answer_f1={test_answer_f1:.4f}, joint_f1={test_joint_f1:.4f}, lexical_ac={test_lexical_ac:.4f}, lexical_ff={test_lexical_ff:.4f}, mrr={test_mrr:.4f}, rouge_l={test_rouge_l:.4f}")

        # ===================== 最终修复：将独立指标合并回日志 =====================
        logger.info("🔧 开始将所有独立指标分数合并到 qa_execution_logs 中...")

        # 获取完整的日志和指标列表
        full_qa_logs = batch_result.qa_execution_logs
        full_f1s = batch_result.answer_f1s or []
        full_ems = batch_result.answer_ems or []
        full_lexical_acs = batch_result.lexical_acs or []
        full_lexical_ffs = batch_result.lexical_ffs or []
        full_rouge_ls = batch_result.rouge_ls or []

        # 使用循环和索引，将每个指标值添加到对应的log字典中
        num_logs = len(full_qa_logs)
        for i in range(num_logs):
            log_item = full_qa_logs[i]

            # 使用我们之前在pdb中确认的真实键名
            # 注意：这里我们用 f1_score, exact_match, 因为这是qa_log内部的命名
            # 但对于AC/FF/ROUGE等，它们本来就不在log里，可以直接添加
            if i < len(full_f1s): log_item['f1_score'] = full_f1s[i]
            if i < len(full_ems): log_item['exact_match'] = full_ems[i]
            if i < len(full_lexical_acs): log_item['lexical_ac'] = full_lexical_acs[i]
            if i < len(full_lexical_ffs): log_item['lexical_ff'] = full_lexical_ffs[i]
            if i < len(full_rouge_ls): log_item['rouge_l'] = full_rouge_ls[i]

        logger.info("✅ 所有独立指标已成功合并到 qa_execution_logs。")
        # ===================== 修复结束 =====================

        # 确保QA执行日志被正确提取
        qa_execution_logs = batch_result.qa_execution_logs
        if not qa_execution_logs:
            logger.warning("⚠️ 批量评估未返回QA执行日志，这可能影响知识库构建")
        else:
            # 只保存训练集的QA执行日志用于知识库构建
            train_qa_logs = qa_execution_logs[:train_count]
            logger.info(f"📝 提取了{len(train_qa_logs)}条训练集QA执行日志用于知识库构建")
            qa_execution_logs = train_qa_logs

        # 构建返回结果
        processed_results = {
            # 基本统计
            'num_total': batch_result.total_count,
            'num_success': batch_result.success_count,
            'num_errors': batch_result.failed_count,
            'train_count': train_count,
            'test_count': test_count,
            'eval_start': eval_start,
            'eval_end': eval_end,
            'eval_duration': eval_duration,
            
            # Coreset相关信息
            'coreset_used': coreset_used,
            'coreset_size': len(train_qa_pairs) if coreset_used else 0,
            'original_train_size': original_train_size,
            'coreset_weights_sum': float(np.sum(coreset_weights)) if coreset_weights is not None else 0.0,
            
            # 时间统计
            'rag_embedding_time': batch_rag_result.embedding_time,
            'rag_retrieval_time': batch_rag_result.retrieval_time,
            'rag_total_time': batch_rag_result.processing_time,
            'api_call_time': batch_result.api_call_time,
            'evaluation_time': batch_result.evaluation_time,
            'total_processing_time': batch_result.total_time,
            
            # API统计
            'api_success_count': batch_result.api_success_count,
            'api_failed_count': batch_result.api_failed_count,
            'avg_api_latency': batch_result.avg_api_latency,
            'total_tokens': batch_result.total_tokens,
            
            # MCTS训练集指标（优化目标）- 现在支持Coreset加权
            'train_answer_em': train_answer_em,
            'train_answer_f1': train_answer_f1, 
            'train_joint_em': train_joint_em,
            'train_joint_f1': train_joint_f1,
            'train_lexical_ac': train_lexical_ac,
            'train_lexical_ff': train_lexical_ff,
            'train_mrr': train_mrr,
            'train_rouge_l': train_rouge_l,  # 🔥 新增ROUGE-L训练集结果
            
            # 测试集指标（验证用）
            'test_answer_em': test_answer_em,
            'test_answer_f1': test_answer_f1,
            'test_joint_em': test_joint_em,
            'test_joint_f1': test_joint_f1,
            'test_lexical_ac': test_lexical_ac,
            'test_lexical_ff': test_lexical_ff,
            'test_mrr': test_mrr,
            'test_rouge_l': test_rouge_l,  # 🔥 新增ROUGE-L测试集结果
            
            # 保持向后兼容的总体指标（使用训练集）
            'answer_em': train_answer_em,
            'answer_f1': train_answer_f1,
            'joint_em': train_joint_em,
            'joint_f1': train_joint_f1,
            
            # 时间指标
            'min_time': np.min(batch_result.run_times) if batch_result.run_times else 0,
            'max_time': np.max(batch_result.run_times) if batch_result.run_times else 0,
            'mean_time': np.mean(batch_result.run_times) if batch_result.run_times else 0,
            'std_time': np.std(batch_result.run_times) if batch_result.run_times else 0,
            
            # 🎯 按要求：只统计RAG_train_token和Agent_token
            'RAG_train_token': RAG_train_token,
            'Agent_token': Agent_token,

            # 确保QA执行日志被包含在结果中
            'qa_execution_logs': qa_execution_logs,
        }
        
        # 保存配置信息
        if hasattr(flow, 'params') and flow.params:
            config_str = json.dumps(flow.params, ensure_ascii=False, separators=(',', ':'))
            processed_results['configuration'] = config_str
        else:
            processed_results['configuration'] = "{}"

        logger.info(f"🎉 MCTS评估完成: 总耗时={eval_duration:.2f}s, "
                   f"训练集F1={train_joint_f1:.4f}{'(Coreset加权)' if coreset_used else ''}, "
                   f"测试集F1={test_joint_f1:.4f}, "
                   f"QA日志={len(qa_execution_logs)}条")
        
        return processed_results

class EnhancedMCTSRAGOptimizer:
    """Enhanced MCTS RAG Optimizer with True MCTS implementation"""
    
    def __init__(self, dataset_config: SimpleDatasetConfig, api_key: str = None, api_base: str = None, 
                 existing_knowledge_base: T.Optional[T.Dict[str, T.Any]] = None, 
                 optimization_target: str = 'train_joint_f1', csv_file: str = "Experiment/mcts_results.csv"):
        # Initialize components with simplified config
        self.dataset_config = dataset_config
        self.optimization_target = optimization_target
        self.csv_file = csv_file
        self.flow_builder = FlowBuilder(dataset_config)
        self.evaluation_manager = MCTSEvaluationManager(dataset_config, save_csv_tpe=SAVE_CSV_TPE, optimization_target=optimization_target)
        
        # API configuration
        self.api_key = api_key or self._get_default_api_key()
        self.api_base = api_base or self._get_default_api_base()
        self.experiment_id = self._generate_experiment_id()
        
        # Enhanced MCTS optimization engine with true MCTS implementation
        self.optimization_engine = EnhancedMCTSOptimizationEngine(
            api_key=self.api_key,
            api_base=self.api_base,
            experiment_id=self.experiment_id,
            existing_knowledge_base=existing_knowledge_base
        )
        
        # 🔥 核心修改：设置真实评估回调
        self.optimization_engine.set_evaluation_callback(self._real_evaluation_callback)
        
        # 使用optimization_engine中的图记忆系统（避免重复初始化）
        self.graph_memory = self.optimization_engine.graph_memory
        
        # 初始化洞察智能体（使用共享的图记忆系统）
        from hammer.mcts.kb_manager.insight_agent import InsightAgent
        self.insight_agent = InsightAgent(
            api_key=self.api_key,
            api_base=self.api_base
        )

        # 初始化模拟评估器（使用共享的图记忆系统）
        from hammer.mcts.kb_manager.enhanced_evaluator import EnhancedGPTSimulationEvaluator
        self.simulation_evaluator = EnhancedGPTSimulationEvaluator(
            api_key=self.api_key,
            api_base=self.api_base,
            graph_memory=self.graph_memory
        )

    def set_mcts_iterations(self, iterations: int):
        """
        设置MCTS rollout次数
        
        Args:
            iterations: 要执行的MCTS rollout次数，每次rollout都是一次完整的MCTS搜索
        
        Note: 
            iterations=50 意味着进行MCTS内部执行50次rollout：
        """
        # self.optimization_engine.max_searches = iterations  # rollout次数
        self.optimization_engine.mcts_iterations = iterations  # 每次MCTS搜索的内部迭代数（固定为合理值）
        logger.info(f"🎯 MCTS配置更新: 将执行{iterations}次MCTS rollout")        
        # Setup full dataset evaluation callback
        # self._full_dataset_evaluation = None
        self._csv_path = self.csv_file
        
        logger.info("🚀 Enhanced True MCTS RAG Optimizer initialized")
        logger.info(f"⚙️  Configuration: max_workers={MODEL_MAXWORKERS}, train_size={self.dataset_config.train_size}")
        logger.info(f"🧠 Graph Memory Stats: {self.optimization_engine.get_graph_memory_stats()}")

    def evaluate_single_flow(self, params: T.Dict[str, T.Any], use_simulation: bool = True) -> T.Tuple[float, T.Dict[str, T.Any], str]:
        """
        评估单个流程配置 - 适配MCTS版本
        - use_simulation=False: 执行标准真实评估。
        - use_simulation=True: 执行真实评估，更新知识库，然后执行GPT模拟评估，并返回模拟分数。
        """
        prepare_worker()
        logger.info("🔧 MCTS评估配置: %s", json.dumps(params, ensure_ascii=False, separators=(',', ':')))
        
        flow = None
        context = {"flow_start": datetime.now(timezone.utc).timestamp()}
        
        try:
            # 步骤 1: 总是执行真实评估，以获取真实的性能数据和日志
            logger.info("🚀 [Phase 1/2] 开始真实评估...")
            flow = self.flow_builder.build_flow(params)
            real_objective_value, results = self.evaluation_manager.evaluate_flow(flow)
            flow_json = json.dumps(params)
            logger.info(f"✅ [Phase 1/2] 真实评估完成. 真实 F1: {real_objective_value:.4f}")

            # 步骤 2: 根据模式决定后续操作
            if use_simulation:
                logger.info("🚀 [Phase 2/2] 开始知识库更新与模拟评估...")
                
                # 2.1 更新知识库
                qa_logs = results.get('qa_execution_logs', [])
                if qa_logs:
                    self._record_evaluation_to_knowledge_graph(params, qa_logs)
                else:
                    logger.warning("⚠️ 未找到 'qa_execution_logs'，无法更新知识库。")

                # 2.2 执行GPT模拟评估
                # 使用数据集专用prompt
                try:
                    
                    main_query = get_dataset_prompt(self.dataset_config.dataset_name)
                    logger.info(f"✅ 使用数据集专用prompt进行模拟评估,main_query: {main_query}")
                except KeyError as e:
                    logger.warning(f"⚠️ 未找到数据集 '{self.dataset_config.dataset_name}' 的专用prompt，使用通用prompt: {e}")
                    # 如果找不到专用prompt，使用通用的fallback
                    main_query = f"""Dataset Content:
The {self.dataset_config.dataset_name} dataset requires specialized domain knowledge for accurate question answering.

Considerations for RAG Tasks:
Core Challenge: A RAG system must handle domain-specific reasoning and knowledge retrieval tailored to the characteristics of this dataset.
Retriever: The retrieval component needs to identify relevant information appropriate to the domain and question type.
Generator: The generator must synthesize retrieved information accurately while following domain-specific conventions and requirements."""
                
                simulated_score = self.simulation_evaluator.evaluate_configuration(params, main_query, predict_score=real_objective_value)
                logger.info(f"✅ [Phase 2/2] 模拟评估完成. 模拟分数: {simulated_score:.4f}")
                
                # 在模拟模式下，MCTS优化的目标是模拟分数
                objective_value = simulated_score
            else:
                # 在真实模式下，MCTS优化的目标是真实分数
                objective_value = real_objective_value

            # 填充最终的 metrics
            results.update({
                "failed": False,
                "flow_start": context["flow_start"],
                "flow_end": datetime.now(timezone.utc).timestamp(),
                "simulated_score": objective_value if use_simulation else None,
                "real_f1_score": real_objective_value
            })
            results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
            
            logger.info(f"🎉 评估流程结束. 返回目标值: {objective_value:.4f}")
            return objective_value, results, flow_json

        except Exception as ex:
            logger.exception("❌ Flow评估失败: %s", ex)
            results = {
                "failed": True,
                "exception_message": str(ex),
                "flow_start": context["flow_start"],
                "flow_end": datetime.now(timezone.utc).timestamp(),
            }
            flow_json = json.dumps(params)
            raise ex
        finally:
            # 资源回收
            if flow:
                cleaner = CUDACleaner(device_id=DEVICE_ID)
                cleanup_result = cleaner.cleanup_and_delete_flow(flow, aggressive=True)
                freed_memory = cleanup_result.get('freed_allocated', 0)
                logger.info(f"🧹 清理完成 - 释放显存: {freed_memory:.2f} MB")

    def _record_evaluation_to_knowledge_graph(self, params: T.Dict[str, T.Any], qa_execution_logs: T.List[T.Dict[str, T.Any]]):
        """记录评估结果到知识图谱"""
        try:
            self.optimization_engine.record_complete_evaluation(params, {}, qa_execution_logs)
            logger.info("✅ 成功记录评估结果到知识图谱")
        except Exception as e:
            logger.error(f"❌ 记录知识图谱失败: {e}")

    def _generate_experiment_id(self) -> str:
        """生成实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        return f"{timestamp}"

    def _real_evaluation_callback(self, params: T.Dict[str, T.Any]) -> float:
        """真实评估回调函数 - 供MCTS调用"""
        try:
            logger.info(f"🔧 MCTS真实评估回调: {json.dumps(params, ensure_ascii=False, separators=(',', ':'))}")
            
            #确保参数完整性
            enhanced_params = self._ensure_required_params_for_callback(params)
            
            # 进行真实评估
            # objective_value, results, flow_json = self.evaluate_single_flow(enhanced_params, is_simul=False)
            
            objective_value, results, flow_json = self.evaluate_single_flow(enhanced_params, use_simulation=True)
            
            # 🔥 新增：保存评估结果到知识库
            qa_execution_logs = []
            if isinstance(results, dict) and 'qa_execution_logs' in results and results['qa_execution_logs']:
                qa_execution_logs = results['qa_execution_logs']
                logger.info(f"📝 ✅ 成功提取{len(qa_execution_logs)}条真实QA执行日志（来自True MCTS评估）")
            else:
                logger.error("❌ 无法获取真实QA执行日志，知识库构建将受到影响")
                qa_execution_logs = []
            
            # 🔥 关键修复：将评估结果保存到知识库
            try:
                logger.info(f"💾 开始保存MCTS rollout评估结果到知识库...")
                self.optimization_engine.record_complete_evaluation(
                    params=enhanced_params, 
                    metrics=results, 
                    qa_execution_logs=qa_execution_logs
                )
                logger.info(f"✅ MCTS rollout评估结果已保存到知识库")
            except Exception as save_e:
                logger.error(f"❌ 保存MCTS rollout评估结果到知识库失败: {save_e}")
            
            # Save test results to CSV
            self._save_test_results_to_csv(results, params)

            logger.info(f"✅ rollout評估评估完成: F1={objective_value:.4f}")
            return objective_value
            
        except Exception as e:
            logger.error(f"❌ rollout評估评估失败: {e}")
            return 0.0

    def _ensure_required_params_for_callback(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        """为评估回调确保参数完整性 - 精简版本"""
        # 🔧 修复LLM名称中的引号问题
        def clean_llm_name(name):
            """清理LLM名称中的多余引号"""
            if isinstance(name, str):
                # 去掉前后的双引号和单引号
                return name.strip().strip('"').strip("'")
            return name
        
        # 设置基本必需参数
        enhanced_params = {
            "rag_mode": "rag",  # 🔥 修复KeyError: 'rag_mode'
            "enforce_full_evaluation": True,
        }
        
        # 更新传入的参数
        enhanced_params.update(params)
        
        # 🔧 清理所有LLM相关参数中的引号
        llm_param_keys = [
            "response_synthesizer_llm", "query_decomposition_llm", "hyde_llm", "reranker_llm"
        ]
        for key in llm_param_keys:
            if key in enhanced_params:
                enhanced_params[key] = clean_llm_name(enhanced_params[key])
        
        # 确保关键参数有默认值
        defaults = {
            "template_name": enhanced_params.get("template_name", "CoT"),
            "response_synthesizer_llm": enhanced_params.get("response_synthesizer_llm", "Qwen2-7b"),
            "rag_embedding_model": enhanced_params.get("embedding_model", "/mnt/data/wangshu/llm_lm/bge-m3"),
            "rag_method": enhanced_params.get("retrieval_method", "sparse"),
            "rag_top_k": enhanced_params.get("retrieval_top_k", 9),
            "splitter_method": enhanced_params.get("splitter_method", "sentence"),
            "splitter_chunk_overlap_frac": enhanced_params.get("splitter_overlap", 0.1),
        }
        
        # 处理chunk size参数
        if "splitter_chunk_size" in enhanced_params:
            import math
            chunk_size = enhanced_params["splitter_chunk_size"]
            if isinstance(chunk_size, (int, float)) and chunk_size > 0:
                try:
                    chunk_exp = int(math.log2(chunk_size))
                    if 2 ** chunk_exp != chunk_size:
                        chunk_exp = round(math.log2(chunk_size))
                    defaults["splitter_chunk_exp"] = chunk_exp
                except (ValueError, OverflowError):
                    defaults["splitter_chunk_exp"] = 8
            else:
                defaults["splitter_chunk_exp"] = 8
        else:
            defaults["splitter_chunk_exp"] = 8
        
        # 条件参数
        if enhanced_params.get("retrieval_method") == "hybrid":
            defaults["rag_hybrid_bm25_weight"] = enhanced_params.get("hybrid_bm25_weight", 0.5)
            
        # 查询分解参数
        defaults["rag_query_decomposition_enabled"] = enhanced_params.get("query_decomposition_enabled", True)
        if defaults["rag_query_decomposition_enabled"]:
            defaults["rag_query_decomposition_num_queries"] = enhanced_params.get("query_decomposition_num_queries", 4)
            defaults["rag_query_decomposition_llm_name"] = clean_llm_name(enhanced_params.get("query_decomposition_llm", "Qwen2-7b"))
            defaults["rag_fusion_mode"] = enhanced_params.get("fusion_mode", "simple")
            
        # Hyde参数（🔥 强制关闭HyDE以减少搜索空间）
        defaults["hyde_enabled"] = False  # enhanced_params.get("hyde_enabled", True)
        # if defaults["hyde_enabled"]:
        #     defaults["hyde_llm_name"] = clean_llm_name(enhanced_params.get("hyde_llm", "Qwen2-7b"))
            
        # Reranker参数
        defaults["reranker_enabled"] = enhanced_params.get("reranker_enabled", True)
        if defaults["reranker_enabled"]:
            defaults["reranker_llm_name"] = clean_llm_name(enhanced_params.get("reranker_llm", "Qwen2-7b"))
            defaults["reranker_top_k"] = enhanced_params.get("reranker_top_k", 5)
            
        # 额外上下文参数
        defaults["additional_context_enabled"] = enhanced_params.get("additional_context_enabled", True)
        if defaults["additional_context_enabled"]:
            defaults["additional_context_num_nodes"] = enhanced_params.get("additional_context_num_nodes", 5)
            
        # Few-shot参数
        defaults["few_shot_enabled"] = enhanced_params.get("few_shot_enabled", False)
        
        # 应用默认值
        for key, value in defaults.items():
            if key not in enhanced_params:
                enhanced_params[key] = value
                
        return enhanced_params

    def _get_default_api_key(self) -> str:
        """获取默认API密钥"""
        import os
        return os.getenv('OPENAI_API_KEY', '')
    
    def _get_default_api_base(self) -> str:
        """获取默认API基础URL"""
        import os
        return os.getenv('OPENAI_API_BASE', 'https://api.ai-gaochao.cn/v1')

    # def create_objective_function(self):
    #     """Create objective function - Enhanced True MCTS workflow"""
    #     def objective_function(trial: optuna.Trial, study_config: StudyConfig, components: T.List[str]) -> float:
    #         """Enhanced True MCTS objective function with complete evaluation logging"""
    #         logger.debug("Starting Enhanced True MCTS trial with executable: %s", sys.executable)
            
    #         context = self.trial_manager.create_trial_context({})
            
    #         # 🔥 核心修改：MCTS在suggest_parameters内部已经进行了真实评估
    #         # 这里返回的params已经是MCTS选择的最佳配置
    #         params = self.optimization_engine.suggest_parameters(trial, study_config, components)
            
            # try:
            #     # 再次进行完整的数据集评估以获取详细日志（用于记录和分析）
            #     objective_value, metrics, flow_json, qa_execution_logs = self._trigger_enhanced_dataset_evaluation(params)
                
            #     # 记录完整评估结果到三层图记忆系统
            #     self.optimization_engine.record_complete_evaluation(params, metrics, qa_execution_logs)

            #     self.trial_manager.record_trial_success(trial, context, metrics, params)
            #     logger.info("🎯 Enhanced True MCTS Trial %d completed, F1 score: %.4f", trial.number, objective_value)
            #     return objective_value
                
            # except Exception as ex:
            #     self.trial_manager.record_trial_failure(trial, context, ex, params)
            #     raise ex
        
        # return objective_function

    # def _trigger_enhanced_dataset_evaluation(self, params: T.Dict[str, T.Any]) -> T.Tuple[float, T.Dict[str, T.Any], str, T.List[T.Dict[str, T.Any]]]:
    #     """Enhanced dataset evaluation with complete QA execution logging"""
    #     logger.info(f"🚀 Starting enhanced True MCTS dataset evaluation ({FIXED_TRAIN_SIZE} training samples)")
        
    #     # 进行评估并获取详细日志
    #     objective_value, results, flow_json = self.evaluate_single_flow(params, is_simul=False)
        
    #     # 正确检查并提取QA执行日志
    #     if isinstance(results, dict) and 'qa_execution_logs' in results and results['qa_execution_logs']:
    #         qa_execution_logs = results['qa_execution_logs']
    #         logger.info(f"📝 ✅ 成功提取{len(qa_execution_logs)}条真实QA执行日志（来自True MCTS评估）")
            
    #         # 验证QA日志的完整性
    #         if qa_execution_logs and len(qa_execution_logs) > 0:
    #             sample_log = qa_execution_logs[0]
    #             required_fields = ['question', 'ground_truth', 'f1_score', 'retrieval_method', 'embedding_model']
    #             missing_fields = [field for field in required_fields if field not in sample_log]
                
    #             if missing_fields:
    #                 logger.warning(f"⚠️ QA日志缺少关键字段: {missing_fields}")
    #             else:
    #                 logger.info("✅ QA执行日志完整性验证通过")
            
    #     else:
    #         # 如果真实日志不可用，记录错误并使用空列表
    #         logger.error("❌ 无法获取真实QA执行日志，知识库构建将受到影响")
    #         logger.error(f"Results类型: {type(results)}")
    #         logger.error(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")
    #         qa_execution_logs = []
        
    #     # Save test results to CSV
    #     self._save_test_results_to_csv(results, params)
        
    #     logger.info("✅ Enhanced True MCTS dataset evaluation completed")
    #     return objective_value, results, flow_json, qa_execution_logs

    def _save_test_results_to_csv(self, results: T.Dict[str, T.Any], params: T.Dict[str, T.Any]):
        """保存测试结果到CSV - 完整版本包含所有评估指标"""
        try:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            
            file_exists = os.path.exists(self._csv_path)
            
            # 🔥 完整的CSV数据结构
            csv_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'train_answer_f1': float(results.get('train_answer_f1', 0.0)),
                'test_answer_f1': float(results.get('test_answer_f1', 0.0)),
                'train_answer_em': float(results.get('train_answer_em', 0.0)),
                'test_answer_em': float(results.get('test_answer_em', 0.0)),
                'train_lexical_ac': float(results.get('train_lexical_ac', 0.0)),
                'test_lexical_ac': float(results.get('test_lexical_ac', 0.0)),
                'train_lexical_ff': float(results.get('train_lexical_ff', 0.0)),
                'test_lexical_ff': float(results.get('test_lexical_ff', 0.0)),
                'train_mrr': float(results.get('train_mrr', 0.0)),
                'test_mrr': float(results.get('test_mrr', 0.0)),
                'train_rouge_l': float(results.get('train_rouge_l', 0.0)),  # 🔥 新增ROUGE-L CSV保存
                'test_rouge_l': float(results.get('test_rouge_l', 0.0)),   # 🔥 新增ROUGE-L CSV保存
                # 🎯 按要求：只统计RAG_train_token和Agent_token
                'RAG_train_token': int(results.get('RAG_train_token', 0)),
                'Agent_token': int(results.get('Agent_token', 0)),
                'dataset_name': getattr(self.dataset_config, 'dataset_name', 'unknown'),
                'configuration': json.dumps(params, ensure_ascii=False, separators=(',', ':'))
            }
            
            # 🎯 按要求：只包含RAG_train_token和Agent_token字段
            header = [
                'timestamp', 'train_answer_f1', 'test_answer_f1', 'train_answer_em', 'test_answer_em',
                'train_lexical_ac', 'test_lexical_ac', 'train_lexical_ff', 'test_lexical_ff',
                'train_mrr', 'test_mrr', 'train_rouge_l', 'test_rouge_l',  # 🔥 新增ROUGE-L列
                'RAG_train_token', 'Agent_token',
                'dataset_name', 'configuration'
            ]
            
            with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            logger.info(f"💾 完整测试结果已保存到: {self._csv_path}")
            logger.info(f"💾 保存的指标: F1={csv_data['train_answer_f1']:.4f}/{csv_data['test_answer_f1']:.4f}, "
                       f"EM={csv_data['train_answer_em']:.4f}/{csv_data['test_answer_em']:.4f}, "
                       f"AC={csv_data['train_lexical_ac']:.4f}/{csv_data['test_lexical_ac']:.4f}, "
                       f"FF={csv_data['train_lexical_ff']:.4f}/{csv_data['test_lexical_ff']:.4f}, "
                       f"MRR={csv_data['train_mrr']:.4f}/{csv_data['test_mrr']:.4f}, "
                       f"ROUGE-L={csv_data['train_rouge_l']:.4f}/{csv_data['test_rouge_l']:.4f}")  # 🔥 新增ROUGE-L日志
            logger.info(f"💾 按要求统计Token: RAG_train_token={csv_data['RAG_train_token']}, Agent_token={csv_data['Agent_token']}")
            
        except Exception as e:
            logger.error(f"💥 保存测试结果失败: {e}")
            logger.error(f"💥 Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")
            logger.error(f"💥 Params keys: {list(params.keys()) if isinstance(params, dict) else 'N/A'}")

# 知识库加载函数 - 保持不变
def load_knowledge_base_by_id(experiment_id: str) -> T.Optional[T.Dict[str, T.Any]]:
    """根据实验ID加载并验证现有的知识库"""
    
    def _validate_knowledge_base_format(kb_data: dict) -> bool:
        """验证知识库数据格式"""
        required_keys = ['experiment_id', 'configs', 'metadata']
        if not all(key in kb_data for key in required_keys):
            return False
        
        configs = kb_data.get('configs', {})
        if not isinstance(configs, dict):
            return False
        
        metadata = kb_data.get('metadata', {})
        required_metadata_keys = ['total_configs', 'total_explorations']
        if not all(key in metadata for key in required_metadata_keys):
            return False
        
        return True

    try:
        kb_dir = Path("Experiment/mcts_knowledgebase")
        kb_file = kb_dir / f"{experiment_id}_knowledge_base.json"
        
        if not kb_file.exists():
            logger.error(f"❌ 知识库文件不存在: {kb_file}")
            return None
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            existing_kb = json.load(f)
        
        if not _validate_knowledge_base_format(existing_kb):
            logger.error(f"❌ 知识库格式无效: {kb_file}")
            return None
        
        if existing_kb.get('experiment_id') != experiment_id:
            logger.warning(f"⚠️ 知识库experiment_id不匹配: 文件内ID为 {existing_kb.get('experiment_id')}, 请求ID为 {experiment_id}")
        
        existing_configs_len = len(existing_kb.get('configs', {}))
        existing_explorations = existing_kb.get('metadata', {}).get('total_explorations', 0)
        
        logger.info(f"✅ 成功加载现有知识库:")
        logger.info(f"   实验ID: {existing_kb.get('experiment_id', 'unknown')}")
        logger.info(f"   配置数量: {existing_configs_len}")
        logger.info(f"   已完成探索: {existing_explorations}")
        logger.info(f"   文件位置: {kb_file}")
        
        configs = existing_kb.get('configs', {})
        if configs:
            best_config_data = max(configs.values(), key=lambda x: x.get('average_score', 0))
            best_score = best_config_data.get('average_score', 0)
            best_count = best_config_data.get('exploration_count', 1)
            logger.info(f"   🏆 历史最佳得分: {best_score:.4f} (探索{best_count}次)")
            
        return existing_kb
        
    except Exception as e:
        logger.error(f"❌ 加载现有知识库失败: {e}", exc_info=True)
        return None

def run_optimization(dataset_name: str, iterations: int = 50, api_key: str = None, 
                   api_base: str = None, kb_id: str = None, optimization_target: str = 'train_joint_f1', train_size: int = DEFAULT_TRAIN_SIZE, csv_file: str = "Experiment/mcts_results.csv") -> None:
    """运行True MCTS优化"""
    logger.info(f"🚀 Running True MCTS optimization on dataset: {dataset_name}")
    logger.info(f"🎯 Optimization target: {optimization_target}")
    logger.info(f"📊 Training set size: {train_size}")
    
    # 创建数据集配置
    dataset_config = SimpleDatasetConfig(dataset_name, train_size=train_size)
    
    # 加载现有知识库（如果指定）
    existing_knowledge_base = None
    if kb_id:
        logger.info(f"正在尝试加载知识库ID: {kb_id}...")
        existing_knowledge_base = load_knowledge_base_by_id(kb_id)
    else:
        logger.info("未提供知识库ID，将从零开始优化。")
    
    # 创建True MCTS优化器，传入优化目标和CSV文件路径
    optimizer = EnhancedMCTSRAGOptimizer(
        dataset_config=dataset_config,
        api_key=api_key,
        api_base=api_base,
        existing_knowledge_base=existing_knowledge_base,
        optimization_target=optimization_target,
        csv_file=csv_file
    )
    
    # 设置MCTS迭代次数
    optimizer.set_mcts_iterations(iterations)
    best_params = optimizer.optimization_engine.suggest_parameters()
    logger.info(f"🎯 MCTS建议的最佳参数: {best_params}")
    logger.info("🎉 MCTS执行完成")

# def run_study(dataset_name: str, iterations: int = 50,
#               api_key: str = None, api_base: str = None,
#               kb_id: str = None) -> None:
#     """运行完整的True MCTS研究流程"""
#     logger.info(f"📊 启动MCTS研究 - 数据集: {dataset_name}, rollout次数: {iterations}")

#     # 运行优化
#     if not skip_optimization:
#         run_optimization(
#             dataset_name=dataset_name,
#             iterations=iterations,
#             api_key=api_key,
#             api_base=api_base,
#             kb_id=kb_id
#         )
#     else:
#         logger.info("跳过True MCTS优化")

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="RAG System Optimization Framework - True MCTS Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 在2wikimultihopqa数据集上运行MCTS优化 (50次rollout)，使用默认的train_joint_f1作为优化目标
    python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50
    
    # 新增支持的fiqa数据集，使用计算得出的训练集大小
    python -m hammer.tuner.main_tuner_mcts --dataset fiqa --iterations 50 --train-size 105
    
    # 使用修复命名后的webquestions数据集
    python -m hammer.tuner.main_tuner_mcts --dataset webquestions --iterations 50 --train-size 426
    
    # 使用修复命名后的popqa数据集
    python -m hammer.tuner.main_tuner_mcts --dataset popqa --iterations 50 --train-size 210

    # 使用现有知识库继续优化，优化目标为lexical_ac（答案覆盖度）
    python -m hammer.tuner.main_tuner_mcts --dataset hotpotqa --iterations 20 --kb-id my_exp_01 --optimization-target train_lexical_ac --train-size 210
    
    # 使用lexical_ff（忠实度）作为优化目标，MedQA数据集
    python -m hammer.tuner.main_tuner_mcts --dataset MedQA --iterations 30 --optimization-target train_lexical_ff --train-size 267
    
    # 使用MRR（平均倒数排序）作为优化目标
    python -m hammer.tuner.main_tuner_mcts --dataset bioasq --iterations 40 --optimization-target train_mrr
    
    # 注意：每次rollout都会执行完整的数据集评估，计算成本较高
    # 
    # 可用数据集 (unified_query_selection): 
    #   2wikimultihopqa(210), hotpotqa(210), MedQA(267), fiqa(105), 
    #   quartz(192), webquestions(426), eli5(317), popqa(210)
    # 
    # 其他支持的数据集: musique, FinQA, bioasq, ConvFinQA
    # 
    # 可用LLM模型: Qwen2-7b, DeepSeek-R1-32b, Qwen2.5-72b, gpt-4o-mini
    # 
    # 可用优化目标: train_answer_em, train_answer_f1, train_joint_em, train_joint_f1, 
    #              train_lexical_ac, train_lexical_ff, train_mrr, train_rouge_l
        """
    )
    
    # 获取支持的数据集列表
    from docs.dataset.dataset_main_prompt import get_available_datasets
    supported_datasets = get_available_datasets()
    
    parser.add_argument(
        "--dataset",
        help=f"Dataset name ({', '.join(supported_datasets)})",
        choices=supported_datasets,
        required=True,
    )
    parser.add_argument(
        "--iterations",
        help="Number of MCTS rollouts/iterations (default: 50, each rollout triggers full dataset evaluation)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key for GPT evaluation",
        default=None,
    )
    parser.add_argument(
        "--api-base",
        help="OpenAI API base URL",
        default=None,
    )
    parser.add_argument(
        "--kb-id",
        help="ID of the existing knowledge base to load and continue optimization.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--optimization-target",
        help="MCTS optimization target metric (default: train_joint_f1)",
        choices=[
            "train_answer_em", "train_answer_f1", 
            "train_joint_em", "train_joint_f1",
            "train_lexical_ac", "train_lexical_ff", "train_mrr"
        ],
        default="train_joint_f1",
    )
    parser.add_argument(
        "--train-size",
        help=f"Training set size (default: {DEFAULT_TRAIN_SIZE})",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
    )
    parser.add_argument(
        "--csv-file",
        help="Path to CSV file for saving results (default: Experiment/mcts_results.csv)",
        type=str,
        default="Experiment/mcts_results.csv",
    )
    
    args = parser.parse_args()
    
    try:
        run_optimization(
            dataset_name=args.dataset,
            iterations=args.iterations,
            api_key=args.api_key,
            api_base=args.api_base,
            kb_id=args.kb_id,
            optimization_target=args.optimization_target,
            train_size=args.train_size,
            csv_file=args.csv_file,
        )
    except Exception as e:
        logger.error("True MCTS study execution failed: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()