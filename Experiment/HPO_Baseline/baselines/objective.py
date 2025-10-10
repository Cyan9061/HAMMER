
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Add hammer package to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# ==============================================================================
# 🔥 UNIFIED: 导入完全统一的MCTS数据流和评估组件
# ==============================================================================
from hammer.logger import logger
from hammer.flows import Flow
from hammer.tuner.main_tuner_tpe import FlowBuilder
from hammer.utils.optimized_rag_prompt_builder import create_optimized_rag_prompt_builder
from hammer.utils.batch_api_evaluator import create_batch_api_evaluator
from hammer.multihop_evaluation import MultiHopQAEvaluator

# 🔥 UNIFIED: 导入完全统一的MCTS数据流组件
from .data import SUPPORTED_DATASETS
from hammer.mcts.mcts_dataset_loader import SimpleDataset
from hammer.storage import QAPair

# ==============================================================================
# 🔥 NEW: 使用与MCTS一致的常量
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # LLM for answer generation
MODEL_MAXWORKERS = 2  # 🔧 减少并发数以避免API速率限制
API_RETRY_COUNT = 3  # API重试次数
API_RETRY_DELAY = 10  # API重试间隔(秒)

# ==============================================================================
# 🔥 NEW: 使用MCTS的评估逻辑
# ==============================================================================
def _run_mcts_style_evaluation(flow: Flow, qa_pairs: List[QAPair], dataset_name: str) -> Dict[str, Any]:
    """
    🔥 NEW: 使用MCTS风格的评估流程
    完全复用MCTS的评估逻辑，确保结果一致性
    
    Args:
        flow: Flow对象
        qa_pairs: QAPair对象列表（来自MCTS数据流）
        dataset_name: 数据集名称
    
    Returns:
        Dict[str, Any]: 评估结果字典
    """
    eval_start_time = time.time()
    
    logger.info("🏗️ MCTS风格评估阶段1: 批量RAG构建...")
    print("🏗️ MCTS风格评估阶段1: 批量RAG构建...")
    try:
        all_questions = [qa.question for qa in qa_pairs]
        logger.info(f"📊 准备评估 {len(all_questions)} 个问题")
        print(f"📊 准备评估 {len(all_questions)} 个问题")
        
        optimized_rag_builder = create_optimized_rag_prompt_builder(flow)
        batch_rag_result = optimized_rag_builder.batch_build_prompts(all_questions)
        
        logger.info(f"✅ RAG构建完成，生成 {batch_rag_result.success_count} 个提示")
        print(f"✅ RAG构建完成，生成 {batch_rag_result.success_count} 个提示")
    except Exception as e:
        error_msg = f"❌ MCTS风格评估阶段1 (RAG Build) 失败: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        return {"failed": True, "exception_message": str(e)}

    logger.info("🚀 MCTS风格评估阶段2: 批量API调用和评估...")
    print("🚀 MCTS风格评估阶段2: 批量API调用和评估...")
    try:
        # 🔥 关键：创建MCTS风格的数据集对象以获取corpus_mapping
        if dataset_name in SUPPORTED_DATASETS:
            dataset_config = SUPPORTED_DATASETS[dataset_name]
            project_root = Path(__file__).parent.parent.parent.parent
            corpus_file = project_root / dataset_config['corpus_file']
            qa_file = project_root / dataset_config['qa_file']
            
            # 使用MCTS的SimpleDataset来获取corpus_mapping
            simple_dataset = SimpleDataset(
                corpus_file=str(corpus_file),
                qa_file=str(qa_file), 
                dataset_name=dataset_name
            )
            corpus_mapping = simple_dataset._load_corpus_mapping()
            logger.info(f"📚 使用MCTS数据流获取corpus mapping，共 {len(corpus_mapping)} 个文档")
            print(f"📚 使用MCTS数据流获取corpus mapping，共 {len(corpus_mapping)} 个文档")
        else:
            corpus_mapping = {}
            logger.warning(f"⚠️ 未找到数据集配置: {dataset_name}")
        
        # 🔥 使用与MCTS完全一致的评估器
        multihop_evaluator = MultiHopQAEvaluator(corpus_lookup=corpus_mapping)
        batch_evaluator = create_batch_api_evaluator(
            model_name=MODEL_NAME,
            max_workers=MODEL_MAXWORKERS,
            multihop_evaluator=multihop_evaluator
        )
        batch_evaluator.current_flow = flow
        batch_evaluator.current_config = getattr(flow, 'params', {})
        batch_result = batch_evaluator.evaluate_batch_optimized(batch_rag_result, qa_pairs)
        
        success_msg = f"✅ MCTS风格评估阶段2完成: 评估了 {batch_result.total_count} 个响应"
        logger.info(success_msg)
        print(success_msg)
        
        # 🔥 详细展示低分问题的例子（与MCTS一致）
        if hasattr(batch_result, 'joint_f1s') and batch_result.joint_f1s:
            zero_f1_indices = [i for i, f1 in enumerate(batch_result.joint_f1s) if f1 == 0.0]
            if zero_f1_indices:
                print(f"🔍 发现 {len(zero_f1_indices)} 个F1=0的问题，展示前3个:")
                logger.info(f"🔍 发现 {len(zero_f1_indices)} 个F1=0的问题，展示前3个:")
                
                for idx in zero_f1_indices[:3]:  # 只展示前3个
                    if idx < len(qa_pairs) and idx < len(batch_rag_result.final_prompts):
                        qa = qa_pairs[idx]
                        rag_result = batch_rag_result.final_prompts[idx]
                        
                        debug_info = f"""
📋 问题 {idx+1}: {qa.question[:100]}...
🎯 正确答案: {qa.answer[:100]}...
🤖 生成提示: {rag_result[:100] if rag_result else 'N/A'}...
📊 得分情况: joint_f1={batch_result.joint_f1s[idx]:.3f}, answer_f1={batch_result.answer_f1s[idx]:.3f}
                        """
                        print(debug_info)
                        logger.info(debug_info.replace('\n', ' | '))
                        
    except Exception as e:
        error_msg = f"❌ MCTS风格评估阶段2 (API Eval) 失败: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        return {"failed": True, "exception_message": str(e)}

    # 🔥 使用与MCTS完全一致的指标提取逻辑
    def safe_get_attr(obj, attr_name, default_value=[0.0], is_optional=False):
        """安全地获取对象属性，与MCTS逻辑一致"""
        if hasattr(obj, attr_name):
            attr_value = getattr(obj, attr_name)
            
            # 对于None值，直接使用默认值
            if attr_value is None:
                if not is_optional:
                    logger.warning(f"⚠️ BatchEvaluationResult属性 {attr_name} 为None，使用默认值")
                return default_value
            
            # 对于列表类型，检查是否为空
            if isinstance(attr_value, list):
                if len(attr_value) > 0:
                    return attr_value
                else:
                    if not is_optional:
                        logger.warning(f"⚠️ BatchEvaluationResult属性 {attr_name} 为空列表，使用默认值")
                    return default_value
            
            # 对于非列表类型（如数字、字符串等），直接返回
            else:
                return attr_value
                
        else:
            if not is_optional:
                logger.warning(f"⚠️ BatchEvaluationResult缺少属性: {attr_name}，使用默认值")
            return default_value
    
    metric_values = {
        'joint_f1': safe_get_attr(batch_result, 'joint_f1s'),
        'answer_f1': safe_get_attr(batch_result, 'answer_f1s'),
        'answer_em': safe_get_attr(batch_result, 'answer_ems'),
        'joint_em': safe_get_attr(batch_result, 'joint_ems'),
        # 🔥 与MCTS一致的统一评估指标
        'lexical_ac': safe_get_attr(batch_result, 'lexical_acs', default_value=[0.0]),
        'lexical_ff': safe_get_attr(batch_result, 'lexical_ffs', default_value=[0.0]),
        'mrr': safe_get_attr(batch_result, 'mrrs', default_value=[0.0], is_optional=True),
        'rouge_l': safe_get_attr(batch_result, 'rouge_ls', default_value=[0.0]),
    }
    
    # 计算所有指标的平均值
    metric_averages = {k: np.mean(v) for k, v in metric_values.items()}
    
    # 🔥 与MCTS一致的详细指标报告
    metrics_report = f"📊 MCTS风格评估结果: " + " | ".join([f"{k}={v:.4f}" for k, v in metric_averages.items()])
    logger.info(metrics_report)
    print(metrics_report)
    
    # 🔥 与MCTS一致的token使用统计
    total_tokens_attr = safe_get_attr(batch_result, 'total_tokens', 0)
    if isinstance(total_tokens_attr, list):
        training_token_usage = total_tokens_attr[0] if total_tokens_attr else 0
    else:
        training_token_usage = total_tokens_attr if total_tokens_attr is not None else 0
    
    # 🔥 添加训练集规模信息用于验证
    training_sample_count = len(qa_pairs)
    
    logger.info(f"🔢 Token统计: 训练集样本数={training_sample_count}, 总token使用={training_token_usage}")
    print(f"🔢 Token统计: 训练集样本数={training_sample_count}, 总token使用={training_token_usage}")
    
    return { 
        "failed": False, 
        "eval_duration": time.time() - eval_start_time,
        "total_tokens": int(training_token_usage),
        "training_samples": training_sample_count,
        **metric_averages  # 返回所有指标
    }

# ==================== ARCHITECTURE NOTES ====================
# 🔥 UNIFIED: 所有旧的向后兼容代码已完全移除
# ✅ 现在完全使用MCTS架构：SimpleDataset -> QAPair -> _run_mcts_style_evaluation
# ✅ 与MCTS的唯一差异：数据源(unified_1 vs unified_query_selection)和划分方式

def make_evaluate_fn(qa_pairs: List[QAPair], optimization_metric: str = 'joint_f1', return_all_metrics: bool = False, dataset_name: str = 'unknown'):
    """
    🔥 UNIFIED: 完全统一为MCTS架构的评估函数创建器
    直接使用QAPair对象，与MCTS完全一致
    
    Args:
        qa_pairs: QAPair对象列表（来自load_qa_pairs()）
        optimization_metric: 优化目标指标
        return_all_metrics: 是否返回所有指标
        dataset_name: 数据集名称
    
    Returns:
        评估函数
    """
    
    # 🔥 VALIDATION: 确保输入为QAPair对象列表
    if not qa_pairs:
        raise ValueError("QAPair列表不能为空")
    if not isinstance(qa_pairs[0], QAPair):
        raise TypeError(f"必须传入QAPair对象列表，收到: {type(qa_pairs[0])}。请使用data.load_qa_pairs()获取数据")
    
    logger.info(f"✅ MCTS统一架构评估: 接收到{len(qa_pairs)}个QAPair对象")
    
    try:
        # 🔥 UNIFIED: 使用完全相同的MCTS数据集对象创建方式
        if dataset_name in SUPPORTED_DATASETS:
            dataset_config = SUPPORTED_DATASETS[dataset_name]
            project_root = Path(__file__).parent.parent.parent.parent
            corpus_file = project_root / dataset_config['corpus_file']
            qa_file = project_root / dataset_config['qa_file']
            
            # 🔥 KEY: 与MCTS完全一致的SimpleDataset创建
            simple_dataset = SimpleDataset(
                corpus_file=str(corpus_file),
                qa_file=str(qa_file),
                dataset_name=dataset_name
            )
        else:
            raise ValueError(f"不支持的数据集名称: {dataset_name}. 支持的有: {list(SUPPORTED_DATASETS.keys())}")
        
        # 🔥 KEY: 直接复用MCTS的SimpleDatasetConfig，最大化代码复用
        from hammer.tuner.main_tuner_mcts import SimpleDatasetConfig
        
        # 创建HPO_Baseline专用的数据集配置，继承MCTS的SimpleDatasetConfig但使用unified_1数据源
        class HPOBaselineDatasetConfig(SimpleDatasetConfig):
            """HPO_Baseline专用配置类 - 继承MCTS的SimpleDatasetConfig，只修改数据源为unified_1"""
            def __init__(self, dataset_name: str):
                # 手动设置属性以使用unified_1数据源，而不是调用父类初始化
                self.dataset_name = dataset_name
                self.train_size = 210  # 临时设置，实际使用70%/30%划分
                self.name = f"hpo_baseline_{dataset_name}"
                
                # 🔥 修改数据源路径为unified_1（与MCTS的unified_query_selection差异）
                self.corpus_file = dataset_config['corpus_file']  # 使用当前函数的dataset_config
                self.qa_file = dataset_config['qa_file']
                
                # 创建数据集对象 - 直接使用已有的simple_dataset
                self.dataset = simple_dataset
                
                # 🔥 KEY: 直接复用MCTS的所有配置属性
                self.max_workers = 2  # HPO_Baseline使用较小的并发数
                
                # 复用MCTS的search_space
                from hammer.tuner.main_tuner_mcts import SimpleSearchSpace, SimpleTimeoutConfig, SimpleOptimizationConfig
                self.search_space = SimpleSearchSpace()
                self.timeouts = SimpleTimeoutConfig()
                self.optimization = SimpleOptimizationConfig()
                self.toy_mode = False
                
                # 兼容属性
                self.model_config = {"extra": "forbid", "yaml_file": None}
        
        study_config = HPOBaselineDatasetConfig(dataset_name)
        flow_builder = FlowBuilder(study_config)
        
    except Exception as e:
        logger.error(f"❌ MCTS统一架构评估环境初始化失败: {e}", exc_info=True)
        def failed_evaluate(params: Dict[str, Any]):
            if return_all_metrics:
                return {metric: 0.0 for metric in ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']}
            else:
                return 0.0
        return failed_evaluate

    def evaluate(params: Dict[str, Any]):
        """🔥 UNIFIED: 使用完全统一的MCTS架构评估"""
        try:
            logger.info(f"🔧 开始MCTS统一架构基线评估 | 配置: {json.dumps(params, ensure_ascii=False, separators=(',', ':'))}")
            enhanced_params = _ensure_required_params(params)
            flow = flow_builder.build_flow(enhanced_params)
            
            # 🔥 UNIFIED: 使用完全统一的MCTS评估流程
            results = _run_mcts_style_evaluation(flow, qa_pairs, dataset_name)

            if not results.get("failed"):
                # 🔥 提取所有可用指标
                all_metrics = {k: v for k, v in results.items() if k not in ['failed', 'eval_duration']}
                metric_score = results.get(optimization_metric, 0.0)
                duration = results.get('eval_duration', -1)
                
                logger.info(f"✅ MCTS统一架构基线评估完成: {optimization_metric}={metric_score:.4f}, Duration={duration:.2f}s")
                logger.info(f"📊 所有指标: {json.dumps(all_metrics, separators=(',', ':'))}")
                
                # 🔥 根据return_all_metrics决定返回格式
                if return_all_metrics:
                    return all_metrics
                else:
                    return float(metric_score)
            else:
                error_msg = results.get("exception_message", "Unknown error")
                logger.error(f"❌ MCTS统一架构基线评估失败: {error_msg}")
                if return_all_metrics:
                    return {metric: 0.0 for metric in ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']}
                else:
                    return 0.0
        except Exception as e:
            logger.error(f"❌ MCTS统一架构评估函数出现严重异常: {e}", exc_info=True)
            if return_all_metrics:
                return {metric: 0.0 for metric in ['joint_f1', 'answer_f1', 'answer_em', 'joint_em', 'lexical_ac', 'lexical_ff', 'mrr', 'rouge_l']}
            else:
                return 0.0
    
    return evaluate

def _ensure_required_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """🔥 UNCHANGED: 参数完整性确保（与MCTS兼容）"""
    enhanced_params = { "rag_mode": "rag", "enforce_full_evaluation": True, }
    enhanced_params.update(params)
    defaults = {
        "template_name": enhanced_params.get("template_name", "CoT"),
        "response_synthesizer_llm": enhanced_params.get("response_synthesizer_llm", "Qwen2-7b"),
        "rag_embedding_model": enhanced_params.get("embedding_model", "/mnt/data/wangshu/llm_lm/bge-m3"),
        "rag_method": enhanced_params.get("retrieval_method", "sparse"),
        "rag_top_k": enhanced_params.get("retrieval_top_k", 9),
        "splitter_method": enhanced_params.get("splitter_method", "sentence"),
        "splitter_chunk_overlap_frac": enhanced_params.get("splitter_overlap", 0.1),
    }
    if "splitter_chunk_size" in enhanced_params:
        import math
        chunk_size = enhanced_params["splitter_chunk_size"]
        defaults["splitter_chunk_exp"] = round(math.log2(chunk_size)) if chunk_size > 0 else 8
    else:
        defaults["splitter_chunk_exp"] = 8
    if enhanced_params.get("retrieval_method") == "hybrid":
        defaults["rag_hybrid_bm25_weight"] = enhanced_params.get("hybrid_bm25_weight", 0.5)
    defaults["rag_query_decomposition_enabled"] = enhanced_params.get("query_decomposition_enabled", True)
    if defaults["rag_query_decomposition_enabled"]:
        defaults["rag_query_decomposition_num_queries"] = enhanced_params.get("query_decomposition_num_queries", 4)
        defaults["rag_query_decomposition_llm_name"] = enhanced_params.get("query_decomposition_llm", "Qwen2-7b")
        defaults["rag_fusion_mode"] = enhanced_params.get("fusion_mode", "simple")
    defaults["hyde_enabled"] = False  # 🔥 强制关闭HyDE
    if defaults["hyde_enabled"]:
        defaults["hyde_llm_name"] = enhanced_params.get("hyde_llm", "Qwen2-7b")
    defaults["reranker_enabled"] = enhanced_params.get("reranker_enabled", True)
    if defaults["reranker_enabled"]:
        defaults["reranker_llm_name"] = enhanced_params.get("reranker_llm", "TransformerRanker")
        defaults["reranker_top_k"] = enhanced_params.get("reranker_top_k", 5)
    defaults["additional_context_enabled"] = enhanced_params.get("additional_context_enabled", True)
    if defaults["additional_context_enabled"]:
        defaults["additional_context_num_nodes"] = enhanced_params.get("additional_context_num_nodes", 5)
    defaults["few_shot_enabled"] = enhanced_params.get("few_shot_enabled", False)
    for key, value in defaults.items():
        if key not in enhanced_params:
            enhanced_params[key] = value
    return enhanced_params