#!/usr/bin/env python3
"""
HPO Baseline Algorithms Runner

This script provides a unified interface to run different baseline optimization algorithms
for RAG hyperparameter optimization, including the new LLMSelecting algorithm.

Usage:
    python run_baselines.py --algo {random,tpe,greedy_m,greedy_r,greedy_rcc,grid,traditional_mcts,llm_selecting} 
                           --dataset {2wikimultihopqa,hotpotqa,eli5,medqa,popqa,quartz,webquestions,fiqa} 
                           --metric {joint_f1,answer_f1,lexical_ac} 
                           --max_evals INT 
                           --seed INT
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import baseline algorithms
from baselines.algos import (
    random, tpe, greedy_m, greedy_r, greedy_rcc, 
    grid, traditional_mcts, llm_selecting
)

# Import data loading and search space functions
from baselines.data import load_train_test_split
from baselines.search_space import create_mcts_compatible_search_space

# Algorithm mapping
ALGO_MAP = {
    'random': random.run_random,
    'tpe': tpe.run_tpe, 
    'greedy_m': greedy_m.run_greedy_m,
    'greedy_r': greedy_r.run_greedy_r,
    'greedy_rcc': greedy_rcc.run_greedy_rcc,
    'grid': grid.run_grid,
    'traditional_mcts': traditional_mcts.run_traditional_mcts,
    'llm_selecting': llm_selecting.run_llm_selecting
}

SUPPORTED_DATASETS = [
    '2wikimultihopqa', 'hotpotqa', 'eli5', 'medqa', 
    'popqa', 'quartz', 'webquestions', 'fiqa'
]

SUPPORTED_METRICS = ['joint_f1', 'answer_f1', 'lexical_ac']

def setup_automatic_logging(args):
    """Setup automatic logging for baseline algorithms"""
    # LLM Selection通过外部脚本重定向处理日志，不需要自动日志配置
    if args.algo == "llm_selecting":
        # 对于LLM Selection，只设置基本的控制台输出，日志通过脚本重定向处理
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"=== Starting {args.algo} experiment (logging via script redirection) ===")
        return logger
    
    # 其他算法使用标准的HPO_Baseline日志目录
    log_dir = Path("Experiment/HPO_Baseline/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.algo}_{args.dataset}_{args.metric}_{args.max_evals}_{args.seed}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting {args.algo} experiment ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Max evaluations: {args.max_evals}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Log file: {log_path}")
    
    return logger

def main():
    parser = argparse.ArgumentParser(description='Run HPO baseline algorithms')
    parser.add_argument('--algo', choices=list(ALGO_MAP.keys()), required=True,
                       help='Algorithm to run')
    parser.add_argument('--dataset', choices=SUPPORTED_DATASETS, required=True,
                       help='Dataset to use')
    parser.add_argument('--metric', choices=SUPPORTED_METRICS, required=True,
                       help='Metric to optimize')
    parser.add_argument('--max_evals', type=int, required=True,
                       help='Maximum number of evaluations')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_automatic_logging(args)
    
    try:
        # Load data using the unified MCTS architecture
        logger.info("Loading data using MCTS unified architecture...")
        qa_train, qa_test = load_train_test_split(args.dataset)
        
        logger.info(f"Loaded {len(qa_train)} training samples and {len(qa_test)} test samples")
        
        # Get search space
        logger.info("Creating MCTS-compatible search space...")
        search_space = create_mcts_compatible_search_space()
        
        logger.info(f"Search space created with MCTS architecture")
        
        # Get algorithm function
        algo_func = ALGO_MAP[args.algo]
        
        # Run algorithm
        logger.info(f"Running {args.algo} algorithm...")
        
        # Call the algorithm with consistent API
        results = algo_func(
            ss=search_space,
            qa_train=qa_train,
            qa_test=qa_test,
            max_evals=args.max_evals,
            seed=args.seed,
            metric=args.metric,
            dataset_name=args.dataset
        )
        
        logger.info(f"Algorithm completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Error running {args.algo}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()