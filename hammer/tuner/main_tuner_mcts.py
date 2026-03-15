"""hammer/tuner/main_tuner_mcts.py
Main tuner for RAG system optimization using True MCTS algorithm.

Usage:
# Basic usage examples with the default optimization target `train_answer_f1`

nohup python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_2wikimultihopqa_f1_50iter.csv > Experiment/log/mcts_2wikimultihopqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset hotpotqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_hotpotqa_f1_50iter.csv > Experiment/log/mcts_hotpotqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset MedQA --iterations 50 --optimization-target train_answer_f1 --train-size 267 --csv-file Experiment/MCTS_csv/mcts_MedQA_f1_50iter.csv > Experiment/log/mcts_MedQA_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset eli5 --iterations 50 --optimization-target train_answer_f1 --train-size 317 --csv-file Experiment/MCTS_csv/mcts_eli5_f1_50iter.csv > Experiment/log/mcts_eli5_f1_50iter.log 2>&1 &

nohup python -m hammer.tuner.main_tuner_mcts --dataset fiqa --iterations 50 --optimization-target train_answer_f1 --train-size 105 --csv-file Experiment/MCTS_csv/mcts_fiqa_f1_50iter.csv > Experiment/log/mcts_fiqa_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset quartz --iterations 50 --optimization-target train_answer_f1 --train-size 192 --csv-file Experiment/MCTS_csv/mcts_quartz_f1_50iter.csv > Experiment/log/mcts_quartz_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset webquestions --iterations 50 --optimization-target train_answer_f1 --train-size 426 --csv-file Experiment/MCTS_csv/mcts_webquestions_f1_50iter.csv > Experiment/log/mcts_webquestions_f1_50iter.log 2>&1 &
nohup python -m hammer.tuner.main_tuner_mcts --dataset popqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_popqa_f1_50iter.csv > Experiment/log/mcts_popqa_f1_50iter.log 2>&1 &

nohup python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50 --optimization-target train_answer_f1 --train-size 210 --csv-file Experiment/MCTS_csv/mcts_2wikimultihopqa_f1_50iter.csv > Experiment/log/mcts_2wikimultihopqa_f1_50iter_Qwen7b_debug.log 2>&1 &

# Optimization targets:
# train_answer_f1: training answer F1 score (recommended)
# train_joint_f1: training joint F1 score (default)
# train_lexical_ac: training lexical answer coverage
# train_lexical_ff: training lexical faithfulness
# train_mrr: training mean reciprocal rank
# train_answer_em: training answer exact match
# train_joint_em: training joint exact match
    
"""

MODEL_MAXWORKERS = 6
DEFAULT_TRAIN_SIZE = 210
"""
Default training set sizes (override with `--train-size`):
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
MODEL_NAME="Qwen2-7b"  # Keep the default runnable model on Qwen2-7b.
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

MODEL_SIMUL="gpt-4o-mini"  # Kept for compatibility, currently unused.
# MCTS_CSV_FILE="Experiment/mcts_MedQA_ff.csv"

import csv
import argparse
import json
import os
import sys
import typing as T
from datetime import datetime, timezone
from pathlib import Path

# When CUDA_VISIBLE_DEVICES is set, PyTorch renumbers visible GPUs to 0..N-1.
# Use the logical device id instead of a physical device id.
DEVICE_ID = 0 #if os.environ.get('CUDA_VISIBLE_DEVICES') else 0
GPU_QUERY_EMBED_LIST=[DEVICE_ID]#[4,5,6,7]
GPU_BATCHSIZE=128
GPU_TEXT_EMBED= DEVICE_ID

# Import dataset-specific prompts
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.dataset.dataset_main_prompt import (
    get_available_datasets,
    get_dataset_prompt,
    resolve_dataset_files,
    validate_dataset_name,
)

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
    """Minimal search-space adapter compatible with FlowBuilder."""
    
    def is_few_shot(self, params: T.Dict) -> bool:
        """Return whether few-shot prompting is enabled."""
        return params.get("few_shot_enabled", False)

class SimpleTimeoutConfig:
    """Minimal timeout config compatible with build_rag_retriever."""
    
    def __init__(self):
        self.embedding_timeout_active = False
        self.embedding_max_time = 3600 * 4
        self.embedding_min_chunks_to_process = 100
        self.embedding_min_time_to_process = 120
        self.eval_timeout = 3600 * 10
        self.single_eval_timeout = 3600 * 2
        self.onnx_timeout = 600

class SimpleOptimizationConfig:
    """Minimal optimization config compatible with build_rag_retriever and optimization.py."""
    
    def __init__(self):
        self.embedding_device = GPU_TEXT_EMBED  # Embedding device id.
        self.use_hf_embedding_models = False  # Whether to use HuggingFace embedding models.
        self.num_trials = 100
        self.cpus_per_trial = 2
        self.gpus_per_trial = 0.0
        
        # Additional attributes expected by optimization.py.
        self.objective_1_name = "answer_f1"  # Primary objective.
        self.objective_2_name = None  # Single-objective optimization.
        self.seeder_timeout = 300
        self.method = "expanding"
        self.blocks = []  # No block-based optimization in this compatibility layer.
        self.shuffle_blocks = False
        self.max_concurrent_trials = 10
        self.raise_on_failed_trial = False
        self.pareto_eval_success_rate = 0.8

class SimpleDatasetConfig:
    """Simplified dataset config used in place of StudyConfig."""
    
    def __init__(self, dataset_name: str, train_size: int = DEFAULT_TRAIN_SIZE):
        canonical_dataset_name, corpus_file, qa_file = resolve_dataset_files(dataset_name)
        self.dataset_name = canonical_dataset_name
        self.train_size = train_size
        self.name = f"mcts-{self.dataset_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prefer dataset files that actually exist in this repository while keeping legacy layout compatibility.
        self.corpus_file = corpus_file
        self.qa_file = qa_file
        
        # Build the dataset wrapper.
        self.dataset = create_simple_dataset(self.corpus_file, self.qa_file, self.dataset_name)
        
        # Core runtime settings.
        self.max_workers = MODEL_MAXWORKERS
        
        # Compatibility attribute expected by the TPE FlowBuilder.
        self.search_space = SimpleSearchSpace()
        
        # Compatibility attribute expected by build_rag_retriever debug logging.
        self.model_config = {"extra": "forbid", "yaml_file": None}
        
        # Additional compatibility attributes expected downstream.
        self.toy_mode = False  # Always use the full runtime path.
        
        # Build simplified timeout config.
        self.timeouts = SimpleTimeoutConfig()
        
        # Build simplified optimization config.
        self.optimization = SimpleOptimizationConfig()
        
        logger.info("Dataset config: %s", self.dataset_name)
        logger.info("  Corpus: %s", self.corpus_file)
        logger.info("  QA file: %s", self.qa_file)
        logger.info("  Train size: %s", self.train_size)

    def iter_examples(self, partition="test"):
        """Compatibility shim for the original StudyConfig interface."""
        # MCTS does not distinguish partitions here; always use the same dataset.
        return self.dataset.load_qa_pairs()

# Simplified evaluation manager used by the MCTS entrypoint.
class MCTSEvaluationManager:
    """MCTS-specific evaluation manager built around the simplified dataset config."""
    
    def __init__(self, dataset_config: SimpleDatasetConfig, save_csv_tpe: bool = True, optimization_target: str = 'train_joint_f1'):
        self.dataset_config = dataset_config
        self.save_csv_tpe = save_csv_tpe
        self.optimization_target = optimization_target
        logger.info("Initialized MCTS evaluation manager with train_size=%s", dataset_config.train_size)
        logger.info("Optimization target: %s", optimization_target)
    
    def evaluate_flow(self, flow: Flow) -> T.Tuple[float, T.Dict[str, T.Any]]:
        """Evaluate a flow and return both the objective value and the detailed results."""
        # Use the MCTS-specific multihop evaluation strategy.
        evaluation_strategy = MCTSMultiHopEvaluationStrategy()
        results = evaluation_strategy.evaluate_flow(flow, self.dataset_config, self.save_csv_tpe)
        
        # Extract the configured optimization target.
        objective_value = self._extract_objective_value(results)
        
        return objective_value, results
    
    def _extract_objective_value(self, results: T.Dict[str, T.Any]) -> float:
        """Extract the configured optimization target from the evaluation result dict."""
        # Training targets available for optimization.
        available_targets = {
            'train_answer_em': 'Train Answer Exact Match',
            'train_answer_f1': 'Train Answer F1 Score', 
            'train_joint_em': 'Train Joint Exact Match',
            'train_joint_f1': 'Train Joint F1 Score',
            'train_lexical_ac': 'Train Lexical Answer Coverage',
            'train_lexical_ff': 'Train Lexical Faithfulness',
            'train_mrr': 'Train Mean Reciprocal Rank',
            'train_rouge_l': 'Train ROUGE-L Score'
        }
        
        if self.optimization_target not in available_targets:
            logger.warning("Unknown optimization target %s; falling back to train_joint_f1", self.optimization_target)
            self.optimization_target = 'train_joint_f1'
        
        # Use the configured target directly.
        target_key = self.optimization_target
        if target_key not in results:
            logger.warning("Target key %s is missing from results", self.optimization_target)
            # List nearby keys for debugging.
            available_keys = [k for k in results.keys() if 'train' in k or any(metric in k for metric in ['f1', 'em', 'ac', 'ff', 'mrr'])]
            logger.warning("Available related keys: %s", available_keys)
            # Fall back to a sensible default.
            target_key = 'train_joint_f1' if 'train_joint_f1' in results else 'joint_f1'
        
        objective_value = results.get(target_key, 0.0)
        target_description = available_targets.get(self.optimization_target, self.optimization_target)
        
        logger.info("Optimization target '%s' (%s): %.4f", target_description, target_key, objective_value)
        return objective_value

class MCTSMultiHopEvaluationStrategy:
    """MCTS-specific multihop evaluation strategy with QA-log extraction and coreset weighting support."""
    
    def evaluate_flow(self, flow: Flow, dataset_config: SimpleDatasetConfig, save_csv_tpe=False) -> T.Dict[str, T.Any]:
        """Evaluate a flow with the MCTS-specific pipeline, including coreset-aware train metrics."""
        
        from hammer.utils.simple_token_tracker import get_token_statistics, clear_token_usage, print_debug_info
        from hammer.multihop_evaluation import MultiHopQAEvaluator
        from hammer.utils.optimized_rag_prompt_builder import create_optimized_rag_prompt_builder
        from hammer.utils.batch_api_evaluator import create_batch_api_evaluator
        import time
        import numpy as np
        import json
        
        # Keep token counts accumulated so MCTS search-phase agent tokens are preserved.
        # clear_token_usage()
        
        if hasattr(flow, 'params') and flow.params:
            logger.info("Starting MCTS evaluation with config: %s", json.dumps(flow.params, ensure_ascii=False, separators=(',', ':')))
        else:
            logger.info("Starting MCTS evaluation without explicit flow params")

        eval_start = time.time()
        
        # Use the MCTS-specific train/test split.
        all_qa_pairs = list(dataset_config.iter_examples())
        total_qa_count = len(all_qa_pairs)
        train_qa_pairs = all_qa_pairs[:min(dataset_config.train_size, total_qa_count)]
        test_qa_pairs = all_qa_pairs[dataset_config.train_size:] if total_qa_count > dataset_config.train_size else []
        
        logger.info(
            "MCTS split: total=%s QA pairs, train=%s (optimization target), test=%s",
            total_qa_count,
            len(train_qa_pairs),
            len(test_qa_pairs),
        )
        
        # Phase 1: batch embedding and RAG prompt construction over the full dataset.
        logger.info("Phase 1: starting batch embedding, coreset selection, and RAG construction")
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
            
            # Extract coreset metadata for later weighted metric computation.
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
                
                logger.info("Coreset details:")
                logger.info("  Original indices: %s", original_coreset_indices)
                logger.info("  Sorted indices: %s", sorted_coreset_indices)
                logger.info("  Weights: %s", coreset_weights)
                logger.info("  Weight sum: %s", np.sum(coreset_weights))
                logger.info("  Original train size: %s", original_train_size)
                
                # Keep QA pairs and prompts aligned after coreset selection.
                # 1. Filter the training QA pairs.
                train_qa_pairs = [train_qa_pairs[i] for i in sorted_coreset_indices]
                
                # 2. Rebuild prompts as selected-train prompts plus all test prompts.
                selected_train_prompts = [batch_rag_result.final_prompts[i] for i in sorted_coreset_indices]
                test_prompts = batch_rag_result.final_prompts[len(all_qa_pairs[:dataset_config.train_size]):]
                
                # 3. Rebuild the prompt array.
                batch_rag_result.final_prompts = selected_train_prompts + test_prompts
                
                # 4. Rebuild the full QA sequence.
                all_qa_pairs = train_qa_pairs + test_qa_pairs
                
                logger.info("Coreset reconstruction completed:")
                logger.info("  Train: %s QA pairs, %s prompts", len(train_qa_pairs), len(selected_train_prompts))
                logger.info("  Test: %s QA pairs, %s prompts", len(test_qa_pairs), len(test_prompts))
                logger.info("  Total: %s QA pairs, %s prompts", len(all_qa_pairs), len(batch_rag_result.final_prompts))
                
                # Sanity-check array lengths after reconstruction.
                if len(all_qa_pairs) != len(batch_rag_result.final_prompts):
                    raise ValueError(
                        f"Length mismatch: QA pairs({len(all_qa_pairs)}) vs prompts({len(batch_rag_result.final_prompts)})"
                    )
                
                # Ensure weights still match the new training-set size.
                if len(coreset_weights) != len(train_qa_pairs):
                    raise ValueError(
                        f"Weight count mismatch: weights({len(coreset_weights)}) vs train_qa_pairs({len(train_qa_pairs)})"
                    )
                    
            else:
                logger.info("Coreset disabled or unavailable (USE_CORESET=%s); keeping original data order", USE_CORESET)

        except Exception as e:
            logger.error("Phase 1 failed: %s", e)
            return {
                "failed": True,
                "exception_message": f"MCTS RAG build failed: {e}",
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
        
        # Ensure phase 1 produced a batch result.
        if batch_rag_result is None:
            logger.error("Phase 1 failed because batch_rag_result is None")
            return {
                "failed": True,
                "exception_message": "MCTS RAG build failed because batch_rag_result is None",
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
        
        # Phase 2: batch API calls and evaluation.
        logger.info("Phase 2: starting batch API calls and evaluation")
        
        batch_result = None
        exception_message = ""
        
        try:
            # Load the corpus mapping used by downstream metric computation.
            corpus_mapping = dataset_config.dataset._load_corpus_mapping()
            logger.info("Loaded corpus mapping with %s documents", len(corpus_mapping))
            
            multihop_evaluator = MultiHopQAEvaluator(corpus_lookup=corpus_mapping)
            batch_evaluator = create_batch_api_evaluator(
                model_name=MODEL_NAME,
                max_workers=MODEL_MAXWORKERS,
                multihop_evaluator=multihop_evaluator
            )
            
            # Pass the flow params through so the evaluator can recover the active RAG config.
            batch_evaluator.current_flow = flow
            batch_evaluator.current_config = getattr(flow, 'params', {})
            
            batch_result = batch_evaluator.evaluate_batch_optimized(batch_rag_result, all_qa_pairs)
            logger.info("Phase 2 completed with %s evaluated responses", batch_result.total_count)
            
        except Exception as e:
            exception_message = f"MCTS batch API evaluation failed: {e}"
            logger.error("Phase 2 failed: %s", e, exc_info=True)

        # Uniform failure handling for phase 2.
        if batch_result is None:
            return {
                "failed": True,
                "exception_message": exception_message or "MCTS batch API evaluation failed because batch_result is None",
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
        
        # Track only RAG_train_token and Agent_token as requested.
        agent_tokens, rag_tokens = get_token_statistics()
        
        # Approximate train-set RAG tokens by the train ratio over the full run.
        train_ratio = len(train_qa_pairs) / total_qa_count if total_qa_count > 0 else 0.0
        RAG_train_token = int(rag_tokens["total"] * train_ratio)
        Agent_token = agent_tokens["total"]
        
        logger.info("Token summary:")
        logger.info("  Agent_token: %s tokens (%s calls)", Agent_token, agent_tokens["calls"])
        logger.info("  RAG total: %s tokens (%s calls)", rag_tokens["total"], rag_tokens["calls"])
        logger.info("  Train ratio: %.4f (%s/%s)", train_ratio, len(train_qa_pairs), total_qa_count)
        logger.info("  RAG_train_token: %s tokens", RAG_train_token)

        # Compute metrics using the MCTS split.
        train_count = len(train_qa_pairs)
        test_count = len(test_qa_pairs)
        
        # Split result arrays into train/test slices.
        train_answer_ems = batch_result.answer_ems[:train_count] if batch_result.answer_ems else []
        train_answer_f1s = batch_result.answer_f1s[:train_count] if batch_result.answer_f1s else []
        train_joint_ems = batch_result.joint_ems[:train_count] if batch_result.joint_ems else []
        train_joint_f1s = batch_result.joint_f1s[:train_count] if batch_result.joint_f1s else []
        
        # Split unified evaluation metrics as well.
        train_lexical_acs = batch_result.lexical_acs[:train_count] if batch_result.lexical_acs else []
        train_lexical_ffs = batch_result.lexical_ffs[:train_count] if batch_result.lexical_ffs else []
        train_mrrs = batch_result.mrrs[:train_count] if batch_result.mrrs else []
        train_rouge_ls = batch_result.rouge_ls[:train_count] if batch_result.rouge_ls else []
        
        test_answer_ems = batch_result.answer_ems[train_count:] if batch_result.answer_ems and test_count > 0 else []
        test_answer_f1s = batch_result.answer_f1s[train_count:] if batch_result.answer_f1s and test_count > 0 else []
        test_joint_ems = batch_result.joint_ems[train_count:] if batch_result.joint_ems and test_count > 0 else []
        test_joint_f1s = batch_result.joint_f1s[train_count:] if batch_result.joint_f1s and test_count > 0 else []
        
        # Test-set unified evaluation metrics.
        test_lexical_acs = batch_result.lexical_acs[train_count:] if batch_result.lexical_acs and test_count > 0 else []
        test_lexical_ffs = batch_result.lexical_ffs[train_count:] if batch_result.lexical_ffs and test_count > 0 else []
        test_mrrs = batch_result.mrrs[train_count:] if batch_result.mrrs and test_count > 0 else []
        test_rouge_ls = batch_result.rouge_ls[train_count:] if batch_result.rouge_ls and test_count > 0 else []
        
        logger.info(f"train_answer_f1s = {train_answer_f1s}")
        logger.info(f"test_answer_f1s = {test_answer_f1s}")
        
        # Compute training metrics with coreset weights when available.
        if coreset_used and coreset_weights is not None:
            logger.info("Computing training metrics with coreset weights")
            
            # Validate array lengths before weighted aggregation.
            if len(coreset_weights) != len(train_answer_f1s):
                logger.error(
                    "Weight/result length mismatch: weights(%s) vs results(%s)",
                    len(coreset_weights),
                    len(train_answer_f1s),
                )
                raise ValueError("Weight/result length mismatch")
            
            # Normalize weights.
            normalized_weights = coreset_weights / np.sum(coreset_weights)
            logger.info("Normalized weights: %s", normalized_weights)
            logger.info("Normalized weight sum: %.6f", np.sum(normalized_weights))
            
            # Weighted training metrics.
            train_answer_em = np.average(train_answer_ems, weights=normalized_weights) if train_answer_ems else 0.0
            train_answer_f1 = np.average(train_answer_f1s, weights=normalized_weights) if train_answer_f1s else 0.0
            train_joint_em = np.average(train_joint_ems, weights=normalized_weights) if train_joint_ems else 0.0
            train_joint_f1 = np.average(train_joint_f1s, weights=normalized_weights) if train_joint_f1s else 0.0
            
            # Weighted unified evaluation metrics.
            train_lexical_ac = np.average(train_lexical_acs, weights=normalized_weights) if train_lexical_acs else 0.0
            train_lexical_ff = np.average(train_lexical_ffs, weights=normalized_weights) if train_lexical_ffs else 0.0
            train_mrr = np.average(train_mrrs, weights=normalized_weights) if train_mrrs else 0.0
            train_rouge_l = np.average(train_rouge_ls, weights=normalized_weights) if train_rouge_ls else 0.0
            
            logger.info("Coreset-weighted training metrics:")
            logger.info("  Weighted answer_f1=%.4f (simple mean=%.4f)", train_answer_f1, np.mean(train_answer_f1s))
            logger.info("  Weighted joint_f1=%.4f (simple mean=%.4f)", train_joint_f1, np.mean(train_joint_f1s))
            logger.info(
                "  Weighted lexical_ac=%.4f (simple mean=%.4f)",
                train_lexical_ac,
                np.mean(train_lexical_acs) if train_lexical_acs else 0,
            )
            logger.info(
                "  Weighted lexical_ff=%.4f (simple mean=%.4f)",
                train_lexical_ff,
                np.mean(train_lexical_ffs) if train_lexical_ffs else 0,
            )
            logger.info("  Weighted mrr=%.4f (simple mean=%.4f)", train_mrr, np.mean(train_mrrs) if train_mrrs else 0)
            logger.info(
                "  Weighted rouge_l=%.4f (simple mean=%.4f)",
                train_rouge_l,
                np.mean(train_rouge_ls) if train_rouge_ls else 0,
            )
            
        else:
            logger.info("Computing training metrics with simple means (no coreset weighting)")
            # Original behavior: simple averages.
            train_answer_em = np.mean(train_answer_ems) if train_answer_ems else 0.0
            train_answer_f1 = np.mean(train_answer_f1s) if train_answer_f1s else 0.0
            train_joint_em = np.mean(train_joint_ems) if train_joint_ems else 0.0
            train_joint_f1 = np.mean(train_joint_f1s) if train_joint_f1s else 0.0
            
            # Unified evaluation metrics via simple means.
            train_lexical_ac = np.mean(train_lexical_acs) if train_lexical_acs else 0.0
            train_lexical_ff = np.mean(train_lexical_ffs) if train_lexical_ffs else 0.0
            train_mrr = np.mean(train_mrrs) if train_mrrs else 0.0
            train_rouge_l = np.mean(train_rouge_ls) if train_rouge_ls else 0.0
        
        # Test metrics are always computed with simple averages.
        test_answer_em = np.mean(test_answer_ems) if test_answer_ems else 0.0
        test_answer_f1 = np.mean(test_answer_f1s) if test_answer_f1s else 0.0
        test_joint_em = np.mean(test_joint_ems) if test_joint_ems else 0.0
        test_joint_f1 = np.mean(test_joint_f1s) if test_joint_f1s else 0.0
        
        # Test-set unified evaluation metrics.
        test_lexical_ac = np.mean(test_lexical_acs) if test_lexical_acs else 0.0
        test_lexical_ff = np.mean(test_lexical_ffs) if test_lexical_ffs else 0.0
        test_mrr = np.mean(test_mrrs) if test_mrrs else 0.0
        test_rouge_l = np.mean(test_rouge_ls) if test_rouge_ls else 0.0
        
        logger.info(
            "MCTS training metrics (%s items): answer_f1=%.4f, joint_f1=%.4f, lexical_ac=%.4f, lexical_ff=%.4f, mrr=%.4f, rouge_l=%.4f",
            len(train_qa_pairs),
            train_answer_f1,
            train_joint_f1,
            train_lexical_ac,
            train_lexical_ff,
            train_mrr,
            train_rouge_l,
        )
        logger.info(
            "Test metrics (%s items): answer_f1=%.4f, joint_f1=%.4f, lexical_ac=%.4f, lexical_ff=%.4f, mrr=%.4f, rouge_l=%.4f",
            test_count,
            test_answer_f1,
            test_joint_f1,
            test_lexical_ac,
            test_lexical_ff,
            test_mrr,
            test_rouge_l,
        )

        # Merge standalone metrics back into the QA execution logs.
        logger.info("Merging standalone metric scores back into qa_execution_logs")

        # Collect the full log and metric arrays.
        full_qa_logs = batch_result.qa_execution_logs
        full_f1s = batch_result.answer_f1s or []
        full_ems = batch_result.answer_ems or []
        full_lexical_acs = batch_result.lexical_acs or []
        full_lexical_ffs = batch_result.lexical_ffs or []
        full_rouge_ls = batch_result.rouge_ls or []

        # Attach metric values to the corresponding QA log dict.
        num_logs = len(full_qa_logs)
        for i in range(num_logs):
            log_item = full_qa_logs[i]

            # Use the field names already present in qa_log for EM/F1 and append the rest directly.
            if i < len(full_f1s): log_item['f1_score'] = full_f1s[i]
            if i < len(full_ems): log_item['exact_match'] = full_ems[i]
            if i < len(full_lexical_acs): log_item['lexical_ac'] = full_lexical_acs[i]
            if i < len(full_lexical_ffs): log_item['lexical_ff'] = full_lexical_ffs[i]
            if i < len(full_rouge_ls): log_item['rouge_l'] = full_rouge_ls[i]

        logger.info("Merged standalone metrics into qa_execution_logs successfully")

        # Keep only the QA execution logs needed for knowledge-base updates.
        qa_execution_logs = batch_result.qa_execution_logs
        if not qa_execution_logs:
            logger.warning("Batch evaluation did not return QA execution logs; knowledge-base updates may be affected")
        else:
            # Persist only the training QA execution logs for knowledge-base construction.
            train_qa_logs = qa_execution_logs[:train_count]
            logger.info("Extracted %s training QA execution logs for knowledge-base construction", len(train_qa_logs))
            qa_execution_logs = train_qa_logs

        # Build the final result payload.
        processed_results = {
            # Basic counts.
            'num_total': batch_result.total_count,
            'num_success': batch_result.success_count,
            'num_errors': batch_result.failed_count,
            'train_count': train_count,
            'test_count': test_count,
            'eval_start': eval_start,
            'eval_end': eval_end,
            'eval_duration': eval_duration,
            
            # Coreset metadata.
            'coreset_used': coreset_used,
            'coreset_size': len(train_qa_pairs) if coreset_used else 0,
            'original_train_size': original_train_size,
            'coreset_weights_sum': float(np.sum(coreset_weights)) if coreset_weights is not None else 0.0,
            
            # Timing metrics.
            'rag_embedding_time': batch_rag_result.embedding_time,
            'rag_retrieval_time': batch_rag_result.retrieval_time,
            'rag_total_time': batch_rag_result.processing_time,
            'api_call_time': batch_result.api_call_time,
            'evaluation_time': batch_result.evaluation_time,
            'total_processing_time': batch_result.total_time,
            
            # API metrics.
            'api_success_count': batch_result.api_success_count,
            'api_failed_count': batch_result.api_failed_count,
            'avg_api_latency': batch_result.avg_api_latency,
            'total_tokens': batch_result.total_tokens,
            
            # MCTS training metrics (optimization targets).
            'train_answer_em': train_answer_em,
            'train_answer_f1': train_answer_f1, 
            'train_joint_em': train_joint_em,
            'train_joint_f1': train_joint_f1,
            'train_lexical_ac': train_lexical_ac,
            'train_lexical_ff': train_lexical_ff,
            'train_mrr': train_mrr,
            'train_rouge_l': train_rouge_l,
            
            # Test metrics used for validation.
            'test_answer_em': test_answer_em,
            'test_answer_f1': test_answer_f1,
            'test_joint_em': test_joint_em,
            'test_joint_f1': test_joint_f1,
            'test_lexical_ac': test_lexical_ac,
            'test_lexical_ff': test_lexical_ff,
            'test_mrr': test_mrr,
            'test_rouge_l': test_rouge_l,
            
            # Backward-compatible aggregate metrics based on the training split.
            'answer_em': train_answer_em,
            'answer_f1': train_answer_f1,
            'joint_em': train_joint_em,
            'joint_f1': train_joint_f1,
            
            # Runtime summary metrics.
            'min_time': np.min(batch_result.run_times) if batch_result.run_times else 0,
            'max_time': np.max(batch_result.run_times) if batch_result.run_times else 0,
            'mean_time': np.mean(batch_result.run_times) if batch_result.run_times else 0,
            'std_time': np.std(batch_result.run_times) if batch_result.run_times else 0,
            
            # Keep only the requested token counters.
            'RAG_train_token': RAG_train_token,
            'Agent_token': Agent_token,

            # Include QA execution logs for downstream knowledge-base updates.
            'qa_execution_logs': qa_execution_logs,
        }
        
        # Save the evaluated configuration.
        if hasattr(flow, 'params') and flow.params:
            config_str = json.dumps(flow.params, ensure_ascii=False, separators=(',', ':'))
            processed_results['configuration'] = config_str
        else:
            processed_results['configuration'] = "{}"

        logger.info(
            "MCTS evaluation completed: total_time=%.2fs, train_f1=%.4f%s, test_f1=%.4f, qa_logs=%s",
            eval_duration,
            train_joint_f1,
            " (coreset-weighted)" if coreset_used else "",
            test_joint_f1,
            len(qa_execution_logs),
        )
        
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
        
        # Register the real-evaluation callback used by the MCTS engine.
        self.optimization_engine.set_evaluation_callback(self._real_evaluation_callback)
        
        # Reuse the graph-memory instance owned by the optimization engine.
        self.graph_memory = self.optimization_engine.graph_memory
        
        # Initialize the insight agent against the shared graph memory.
        from hammer.mcts.kb_manager.insight_agent import InsightAgent
        self.insight_agent = InsightAgent(
            api_key=self.api_key,
            api_base=self.api_base
        )

        # Initialize the simulation evaluator against the shared graph memory.
        from hammer.mcts.kb_manager.enhanced_evaluator import EnhancedGPTSimulationEvaluator
        self.simulation_evaluator = EnhancedGPTSimulationEvaluator(
            api_key=self.api_key,
            api_base=self.api_base,
            graph_memory=self.graph_memory
        )

    def set_mcts_iterations(self, iterations: int):
        """
        Configure the number of MCTS rollouts.
        
        Args:
            iterations: Number of MCTS rollouts to execute.
        
        Note: 
            `iterations=50` means the internal MCTS loop will perform 50 rollouts.
        """
        # self.optimization_engine.max_searches = iterations
        self.optimization_engine.mcts_iterations = iterations
        logger.info("Updated MCTS config to run %s rollouts", iterations)
        # Setup full dataset evaluation callback
        # self._full_dataset_evaluation = None
        self._csv_path = self.csv_file
        
        logger.info("🚀 Enhanced True MCTS RAG Optimizer initialized")
        logger.info(f"⚙️  Configuration: max_workers={MODEL_MAXWORKERS}, train_size={self.dataset_config.train_size}")
        logger.info(f"🧠 Graph Memory Stats: {self.optimization_engine.get_graph_memory_stats()}")

    def evaluate_single_flow(self, params: T.Dict[str, T.Any], use_simulation: bool = True) -> T.Tuple[float, T.Dict[str, T.Any], str]:
        """
        Evaluate a single flow configuration in the MCTS runtime.
        - `use_simulation=False`: run the standard real evaluation only.
        - `use_simulation=True`: run real evaluation, update the knowledge base, then run GPT-based simulation evaluation.
        """
        prepare_worker()
        logger.info("MCTS evaluation config: %s", json.dumps(params, ensure_ascii=False, separators=(',', ':')))
        
        flow = None
        context = {"flow_start": datetime.now(timezone.utc).timestamp()}
        
        try:
            # Step 1: always run the real evaluation to collect ground-truth metrics and logs.
            logger.info("[Phase 1/2] Starting real evaluation")
            flow = self.flow_builder.build_flow(params)
            real_objective_value, results = self.evaluation_manager.evaluate_flow(flow)
            flow_json = json.dumps(params)
            logger.info("[Phase 1/2] Real evaluation finished with real F1=%.4f", real_objective_value)

            # Step 2: optionally update the KB and run simulation evaluation.
            if use_simulation:
                logger.info("[Phase 2/2] Starting knowledge-base update and simulation evaluation")
                
                # 2.1 Update the knowledge base.
                qa_logs = results.get('qa_execution_logs', [])
                if qa_logs:
                    self._record_evaluation_to_knowledge_graph(params, qa_logs)
                else:
                    logger.warning("Could not find 'qa_execution_logs'; skipping knowledge-base update")

                # 2.2 Run GPT-based simulation evaluation.
                # Prefer a dataset-specific prompt.
                try:
                    
                    main_query = get_dataset_prompt(self.dataset_config.dataset_name)
                    logger.info("Using dataset-specific simulation prompt: %s", main_query)
                except KeyError as e:
                    logger.warning(
                        "Dataset-specific prompt for '%s' is unavailable; using the generic fallback: %s",
                        self.dataset_config.dataset_name,
                        e,
                    )
                    # Fall back to a generic dataset description.
                    main_query = f"""Dataset Content:
The {self.dataset_config.dataset_name} dataset requires specialized domain knowledge for accurate question answering.

Considerations for RAG Tasks:
Core Challenge: A RAG system must handle domain-specific reasoning and knowledge retrieval tailored to the characteristics of this dataset.
Retriever: The retrieval component needs to identify relevant information appropriate to the domain and question type.
Generator: The generator must synthesize retrieved information accurately while following domain-specific conventions and requirements."""
                
                simulated_score = self.simulation_evaluator.evaluate_configuration(params, main_query, predict_score=real_objective_value)
                logger.info("[Phase 2/2] Simulation evaluation finished with score %.4f", simulated_score)
                
                # In simulation mode, optimize against the simulated score.
                objective_value = simulated_score
            else:
                # In real-only mode, optimize against the real score.
                objective_value = real_objective_value

            # Fill the final metric payload.
            results.update({
                "failed": False,
                "flow_start": context["flow_start"],
                "flow_end": datetime.now(timezone.utc).timestamp(),
                "simulated_score": objective_value if use_simulation else None,
                "real_f1_score": real_objective_value
            })
            results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
            
            logger.info("Evaluation flow finished with objective value %.4f", objective_value)
            return objective_value, results, flow_json

        except Exception as ex:
            logger.exception("Flow evaluation failed: %s", ex)
            results = {
                "failed": True,
                "exception_message": str(ex),
                "flow_start": context["flow_start"],
                "flow_end": datetime.now(timezone.utc).timestamp(),
            }
            flow_json = json.dumps(params)
            raise ex
        finally:
            # Release resources after each evaluation.
            if flow:
                cleaner = CUDACleaner(device_id=DEVICE_ID)
                cleanup_result = cleaner.cleanup_and_delete_flow(flow, aggressive=True)
                freed_memory = cleanup_result.get('freed_allocated', 0)
                logger.info("Cleanup finished; released %.2f MB of allocated GPU memory", freed_memory)

    def _record_evaluation_to_knowledge_graph(self, params: T.Dict[str, T.Any], qa_execution_logs: T.List[T.Dict[str, T.Any]]):
        """Record evaluation results into the knowledge graph."""
        try:
            self.optimization_engine.record_complete_evaluation(params, {}, qa_execution_logs)
            logger.info("Recorded evaluation results to the knowledge graph")
        except Exception as e:
            logger.error("Failed to record evaluation results to the knowledge graph: %s", e)

    def _generate_experiment_id(self) -> str:
        """Generate the experiment id."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        return f"{timestamp}"

    def _real_evaluation_callback(self, params: T.Dict[str, T.Any]) -> float:
        """Real-evaluation callback used by the MCTS engine."""
        try:
            logger.info("MCTS real-evaluation callback: %s", json.dumps(params, ensure_ascii=False, separators=(',', ':')))
            
            # Ensure all required params are present.
            enhanced_params = self._ensure_required_params_for_callback(params)
            
            objective_value, results, flow_json = self.evaluate_single_flow(enhanced_params, use_simulation=True)
            
            # Save QA execution logs for the knowledge base.
            qa_execution_logs = []
            if isinstance(results, dict) and 'qa_execution_logs' in results and results['qa_execution_logs']:
                qa_execution_logs = results['qa_execution_logs']
                logger.info("Extracted %s real QA execution logs from the True MCTS evaluation", len(qa_execution_logs))
            else:
                logger.error("Failed to extract real QA execution logs; knowledge-base construction may be affected")
                qa_execution_logs = []
            
            # Persist the evaluation into the knowledge base.
            try:
                logger.info("Saving MCTS rollout evaluation results to the knowledge base")
                self.optimization_engine.record_complete_evaluation(
                    params=enhanced_params, 
                    metrics=results, 
                    qa_execution_logs=qa_execution_logs
                )
                logger.info("Saved MCTS rollout evaluation results to the knowledge base")
            except Exception as save_e:
                logger.error("Failed to save MCTS rollout evaluation results to the knowledge base: %s", save_e)
            
            # Save test results to CSV
            self._save_test_results_to_csv(results, params)

            logger.info("Rollout evaluation completed with F1=%.4f", objective_value)
            return objective_value
            
        except Exception as e:
            logger.error("Rollout evaluation failed: %s", e)
            return 0.0

    def _ensure_required_params_for_callback(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        """Ensure the evaluation callback receives a complete parameter set."""
        # Strip stray quotes from LLM parameter values.
        def clean_llm_name(name):
            """Strip extra quotes from an LLM name."""
            if isinstance(name, str):
                # Remove surrounding single or double quotes.
                return name.strip().strip('"').strip("'")
            return name
        
        # Base required parameters.
        enhanced_params = {
            "rag_mode": "rag",
            "enforce_full_evaluation": True,
        }
        
        # Merge the user-provided params.
        enhanced_params.update(params)
        
        # Clean all LLM-related parameter values.
        llm_param_keys = [
            "response_synthesizer_llm", "query_decomposition_llm", "hyde_llm", "reranker_llm"
        ]
        for key in llm_param_keys:
            if key in enhanced_params:
                enhanced_params[key] = clean_llm_name(enhanced_params[key])
        
        # Fill required defaults.
        defaults = {
            "template_name": enhanced_params.get("template_name", "CoT"),
            "response_synthesizer_llm": enhanced_params.get("response_synthesizer_llm", "Qwen2-7b"),
            "rag_embedding_model": enhanced_params.get("embedding_model", "/mnt/data/wangshu/llm_lm/bge-m3"),
            "rag_method": enhanced_params.get("retrieval_method", "sparse"),
            "rag_top_k": enhanced_params.get("retrieval_top_k", 9),
            "splitter_method": enhanced_params.get("splitter_method", "sentence"),
            "splitter_chunk_overlap_frac": enhanced_params.get("splitter_overlap", 0.1),
        }
        
        # Normalize the chunk-size parameter into chunk_exp.
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
        
        # Conditional parameters.
        if enhanced_params.get("retrieval_method") == "hybrid":
            defaults["rag_hybrid_bm25_weight"] = enhanced_params.get("hybrid_bm25_weight", 0.5)
            
        # Query decomposition parameters.
        defaults["rag_query_decomposition_enabled"] = enhanced_params.get("query_decomposition_enabled", True)
        if defaults["rag_query_decomposition_enabled"]:
            defaults["rag_query_decomposition_num_queries"] = enhanced_params.get("query_decomposition_num_queries", 4)
            defaults["rag_query_decomposition_llm_name"] = clean_llm_name(enhanced_params.get("query_decomposition_llm", "Qwen2-7b"))
            defaults["rag_fusion_mode"] = enhanced_params.get("fusion_mode", "simple")
            
        # HyDE settings. Keep it disabled here to reduce search-space size.
        defaults["hyde_enabled"] = False  # enhanced_params.get("hyde_enabled", True)
        # if defaults["hyde_enabled"]:
        #     defaults["hyde_llm_name"] = clean_llm_name(enhanced_params.get("hyde_llm", "Qwen2-7b"))
            
        # Reranker parameters.
        defaults["reranker_enabled"] = enhanced_params.get("reranker_enabled", True)
        if defaults["reranker_enabled"]:
            defaults["reranker_llm_name"] = clean_llm_name(enhanced_params.get("reranker_llm", "Qwen2-7b"))
            defaults["reranker_top_k"] = enhanced_params.get("reranker_top_k", 5)
            
        # Additional-context parameters.
        defaults["additional_context_enabled"] = enhanced_params.get("additional_context_enabled", True)
        if defaults["additional_context_enabled"]:
            defaults["additional_context_num_nodes"] = enhanced_params.get("additional_context_num_nodes", 5)
            
        # Few-shot parameters.
        defaults["few_shot_enabled"] = enhanced_params.get("few_shot_enabled", False)
        
        # Apply defaults for missing keys.
        for key, value in defaults.items():
            if key not in enhanced_params:
                enhanced_params[key] = value
                
        return enhanced_params

    def _get_default_api_key(self) -> str:
        """Return the default API key."""
        import os
        return os.getenv('OPENAI_API_KEY', '')
    
    def _get_default_api_base(self) -> str:
        """Return the default API base URL."""
        import os
        return os.getenv('OPENAI_API_BASE', 'https://api.ai-gaochao.cn/v1')

    # def create_objective_function(self):
    #     """Create objective function - Enhanced True MCTS workflow"""
    #     def objective_function(trial: optuna.Trial, study_config: StudyConfig, components: T.List[str]) -> float:
    #         """Enhanced True MCTS objective function with complete evaluation logging"""
    #         logger.debug("Starting Enhanced True MCTS trial with executable: %s", sys.executable)
            
    #         context = self.trial_manager.create_trial_context({})
            
    #         # Core design: MCTS already performs real evaluation inside suggest_parameters.
    #         # The returned params are therefore the best configuration chosen by MCTS.
    #         params = self.optimization_engine.suggest_parameters(trial, study_config, components)
            
            # try:
            #     # Re-run the full dataset evaluation to capture detailed logs.
            #     objective_value, metrics, flow_json, qa_execution_logs = self._trigger_enhanced_dataset_evaluation(params)
                
            #     # Record the complete evaluation into the three-layer graph-memory system.
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
        
    #     # Run evaluation and collect detailed logs.
    #     objective_value, results, flow_json = self.evaluate_single_flow(params, is_simul=False)
        
    #     # Check and extract QA execution logs.
    #     if isinstance(results, dict) and 'qa_execution_logs' in results and results['qa_execution_logs']:
    #         qa_execution_logs = results['qa_execution_logs']
    #         logger.info(f"Extracted {len(qa_execution_logs)} real QA execution logs from True MCTS evaluation")
            
    #         # Validate QA log completeness.
    #         if qa_execution_logs and len(qa_execution_logs) > 0:
    #             sample_log = qa_execution_logs[0]
    #             required_fields = ['question', 'ground_truth', 'f1_score', 'retrieval_method', 'embedding_model']
    #             missing_fields = [field for field in required_fields if field not in sample_log]
                
    #             if missing_fields:
    #                 logger.warning(f"QA logs are missing required fields: {missing_fields}")
    #             else:
    #                 logger.info("QA execution log validation passed")
            
    #     else:
    #         # If real logs are unavailable, log the issue and continue with an empty list.
    #         logger.error("Failed to extract real QA execution logs; knowledge-base construction may be affected")
    #         logger.error(f"Results type: {type(results)}")
    #         logger.error(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")
    #         qa_execution_logs = []
        
    #     # Save test results to CSV
    #     self._save_test_results_to_csv(results, params)
        
    #     logger.info("✅ Enhanced True MCTS dataset evaluation completed")
    #     return objective_value, results, flow_json, qa_execution_logs

    def _save_test_results_to_csv(self, results: T.Dict[str, T.Any], params: T.Dict[str, T.Any]):
        """Save test results to CSV with the full metric set."""
        try:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            
            file_exists = os.path.exists(self._csv_path)
            
            # Full CSV payload.
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
                'train_rouge_l': float(results.get('train_rouge_l', 0.0)),
                'test_rouge_l': float(results.get('test_rouge_l', 0.0)),
                # Keep only the requested token counters.
                'RAG_train_token': int(results.get('RAG_train_token', 0)),
                'Agent_token': int(results.get('Agent_token', 0)),
                'dataset_name': getattr(self.dataset_config, 'dataset_name', 'unknown'),
                'configuration': json.dumps(params, ensure_ascii=False, separators=(',', ':'))
            }
            
            # Keep only the requested token fields.
            header = [
                'timestamp', 'train_answer_f1', 'test_answer_f1', 'train_answer_em', 'test_answer_em',
                'train_lexical_ac', 'test_lexical_ac', 'train_lexical_ff', 'test_lexical_ff',
                'train_mrr', 'test_mrr', 'train_rouge_l', 'test_rouge_l',
                'RAG_train_token', 'Agent_token',
                'dataset_name', 'configuration'
            ]
            
            with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            logger.info("Saved full test results to %s", self._csv_path)
            logger.info(
                "Saved metrics: F1=%.4f/%.4f, EM=%.4f/%.4f, AC=%.4f/%.4f, FF=%.4f/%.4f, MRR=%.4f/%.4f, ROUGE-L=%.4f/%.4f",
                csv_data['train_answer_f1'],
                csv_data['test_answer_f1'],
                csv_data['train_answer_em'],
                csv_data['test_answer_em'],
                csv_data['train_lexical_ac'],
                csv_data['test_lexical_ac'],
                csv_data['train_lexical_ff'],
                csv_data['test_lexical_ff'],
                csv_data['train_mrr'],
                csv_data['test_mrr'],
                csv_data['train_rouge_l'],
                csv_data['test_rouge_l'],
            )
            logger.info(
                "Saved requested token counters: RAG_train_token=%s, Agent_token=%s",
                csv_data['RAG_train_token'],
                csv_data['Agent_token'],
            )
            
        except Exception as e:
            logger.error("Failed to save test results: %s", e)
            logger.error("Results keys: %s", list(results.keys()) if isinstance(results, dict) else 'N/A')
            logger.error("Params keys: %s", list(params.keys()) if isinstance(params, dict) else 'N/A')

# Knowledge-base loading helper.
def load_knowledge_base_by_id(experiment_id: str) -> T.Optional[T.Dict[str, T.Any]]:
    """Load and validate an existing knowledge base by experiment id."""
    
    def _validate_knowledge_base_format(kb_data: dict) -> bool:
        """Validate the knowledge-base payload format."""
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
            logger.error("Knowledge-base file does not exist: %s", kb_file)
            return None
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            existing_kb = json.load(f)
        
        if not _validate_knowledge_base_format(existing_kb):
            logger.error("Knowledge-base format is invalid: %s", kb_file)
            return None
        
        if existing_kb.get('experiment_id') != experiment_id:
            logger.warning(
                "Knowledge-base experiment_id mismatch: file has %s, request asked for %s",
                existing_kb.get('experiment_id'),
                experiment_id,
            )
        
        existing_configs_len = len(existing_kb.get('configs', {}))
        existing_explorations = existing_kb.get('metadata', {}).get('total_explorations', 0)
        
        logger.info("Loaded existing knowledge base successfully:")
        logger.info("  Experiment ID: %s", existing_kb.get('experiment_id', 'unknown'))
        logger.info("  Config count: %s", existing_configs_len)
        logger.info("  Completed explorations: %s", existing_explorations)
        logger.info("  File path: %s", kb_file)
        
        configs = existing_kb.get('configs', {})
        if configs:
            best_config_data = max(configs.values(), key=lambda x: x.get('average_score', 0))
            best_score = best_config_data.get('average_score', 0)
            best_count = best_config_data.get('exploration_count', 1)
            logger.info("  Historical best score: %.4f (%s explorations)", best_score, best_count)
            
        return existing_kb
        
    except Exception as e:
        logger.error("Failed to load existing knowledge base: %s", e, exc_info=True)
        return None

def run_optimization(dataset_name: str, iterations: int = 50, api_key: str = None, 
                   api_base: str = None, kb_id: str = None, optimization_target: str = 'train_joint_f1', train_size: int = DEFAULT_TRAIN_SIZE, csv_file: str = "Experiment/mcts_results.csv") -> None:
    """Run True MCTS optimization."""
    dataset_name = validate_dataset_name(dataset_name)
    logger.info(f"🚀 Running True MCTS optimization on dataset: {dataset_name}")
    logger.info(f"🎯 Optimization target: {optimization_target}")
    logger.info(f"📊 Training set size: {train_size}")
    
    # Build the dataset config.
    dataset_config = SimpleDatasetConfig(dataset_name, train_size=train_size)
    
    # Load an existing knowledge base when requested.
    existing_knowledge_base = None
    if kb_id:
        logger.info("Attempting to load knowledge-base id %s", kb_id)
        existing_knowledge_base = load_knowledge_base_by_id(kb_id)
    else:
        logger.info("No knowledge-base id provided; starting optimization from scratch")
    
    # Build the optimizer with the target metric and CSV output path.
    optimizer = EnhancedMCTSRAGOptimizer(
        dataset_config=dataset_config,
        api_key=api_key,
        api_base=api_base,
        existing_knowledge_base=existing_knowledge_base,
        optimization_target=optimization_target,
        csv_file=csv_file
    )
    
    # Configure the rollout count.
    optimizer.set_mcts_iterations(iterations)
    best_params = optimizer.optimization_engine.suggest_parameters()
    logger.info("Best parameters suggested by MCTS: %s", best_params)
    logger.info("MCTS execution completed")

# def run_study(dataset_name: str, iterations: int = 50,
#               api_key: str = None, api_base: str = None,
#               kb_id: str = None) -> None:
#     """Run the full True MCTS study workflow."""
#     logger.info(f"Starting MCTS study for dataset={dataset_name}, rollouts={iterations}")

#     # Run optimization.
#     if not skip_optimization:
#         run_optimization(
#             dataset_name=dataset_name,
#             iterations=iterations,
#             api_key=api_key,
#             api_base=api_base,
#             kb_id=kb_id
#         )
#     else:
#         logger.info("Skipping True MCTS optimization")

def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="RAG System Optimization Framework - True MCTS Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run MCTS optimization on 2wikimultihopqa with 50 rollouts and the default train_joint_f1 target.
    python -m hammer.tuner.main_tuner_mcts --dataset 2wikimultihopqa --iterations 50
    
    # Run the supported fiqa dataset with the computed train-size value.
    python -m hammer.tuner.main_tuner_mcts --dataset fiqa --iterations 50 --train-size 105
    
    # Run the normalized webquestions dataset name.
    python -m hammer.tuner.main_tuner_mcts --dataset webquestions --iterations 50 --train-size 426
    
    # Run the normalized popqa dataset name.
    python -m hammer.tuner.main_tuner_mcts --dataset popqa --iterations 50 --train-size 210

    # Continue from an existing knowledge base and optimize lexical answer coverage.
    python -m hammer.tuner.main_tuner_mcts --dataset hotpotqa --iterations 20 --kb-id my_exp_01 --optimization-target train_lexical_ac --train-size 210
    
    # Optimize lexical faithfulness on the MedQA dataset.
    python -m hammer.tuner.main_tuner_mcts --dataset MedQA --iterations 30 --optimization-target train_lexical_ff --train-size 267
    
    # Optimize mean reciprocal rank.
    python -m hammer.tuner.main_tuner_mcts --dataset bioasq --iterations 40 --optimization-target train_mrr
    
    # Each rollout executes a full dataset evaluation, so the cost can be high.
    # 
    # Available datasets (unified_query_selection):
    #   2wikimultihopqa(210), hotpotqa(210), MedQA(267), fiqa(105), 
    #   quartz(192), webquestions(426), eli5(317), popqa(210)
    # 
    # Other supported datasets: musique, FinQA, bioasq, ConvFinQA
    # 
    # Available LLM models: Qwen2-7b, DeepSeek-R1-32b, Qwen2.5-72b, gpt-4o-mini
    # 
    # Available optimization targets: train_answer_em, train_answer_f1, train_joint_em, train_joint_f1,
    #              train_lexical_ac, train_lexical_ff, train_mrr, train_rouge_l
        """
    )
    
    # Read the list of supported datasets.
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
