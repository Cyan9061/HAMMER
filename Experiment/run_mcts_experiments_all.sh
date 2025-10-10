#!/bin/bash
# 合并后的MCTS实验脚本 - 包含所有MCTS相关实验
# 替代: run_mcts_experiments_part1.sh, run_mcts_experiments_part2.sh, run_mcts_experiments_part3.sh, run_mcts_experiments_part_test.sh

echo "Starting MCTS Experiments..."

# Part 1 - HotpotQA experiments
echo "Running MCTS experiments - Part 1 (HotpotQA)"
python -m hammer.tuner.main_tuner_mcts \
    --dataset hotpotqa \
    --max_iterations 50 \
    --experiment_name mcts_hotpotqa_f1_50iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset 2wikimultihopqa \
    --max_iterations 50 \
    --experiment_name mcts_2wikimultihopqa_f1_50iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset popqa \
    --max_iterations 10 \
    --experiment_name mcts_popqa_f1_10iter \
    --metric joint_f1 &

wait

# Part 2 - Additional datasets
echo "Running MCTS experiments - Part 2 (Additional datasets)"
python -m hammer.tuner.main_tuner_mcts \
    --dataset eli5 \
    --max_iterations 50 \
    --experiment_name mcts_eli5_f1_50iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset quartz \
    --max_iterations 50 \
    --experiment_name mcts_quartz_f1_50iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset MedQA \
    --max_iterations 20 \
    --experiment_name mcts_MedQA_f1_20iter \
    --metric joint_f1 &

wait

# Part 3 - Extended experiments
echo "Running MCTS experiments - Part 3 (Extended)"
python -m hammer.tuner.main_tuner_mcts \
    --dataset hotpotqa \
    --max_iterations 30 \
    --experiment_name mcts_hotpotqa_deepseek_f1_30iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset 2wikimultihopqa \
    --max_iterations 30 \
    --experiment_name mcts_2wikimultihopqa_deepseek_f1_30iter \
    --metric joint_f1 &

wait

# Test experiments
echo "Running MCTS experiments - Test batch"
python -m hammer.tuner.main_tuner_mcts \
    --dataset hotpotqa \
    --max_iterations 1 \
    --experiment_name test_mcts_hotpotqa_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset 2wikimultihopqa \
    --max_iterations 1 \
    --experiment_name test_mcts_2wikimultihopqa_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset eli5 \
    --max_iterations 1 \
    --experiment_name test_mcts_eli5_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset quartz \
    --max_iterations 1 \
    --experiment_name test_mcts_quartz_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset popqa \
    --max_iterations 1 \
    --experiment_name test_mcts_popqa_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset MedQA \
    --max_iterations 1 \
    --experiment_name test_mcts_MedQA_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset fiqa \
    --max_iterations 1 \
    --experiment_name test_mcts_fiqa_f1_1iter \
    --metric joint_f1 &

python -m hammer.tuner.main_tuner_mcts \
    --dataset webquestions \
    --max_iterations 1 \
    --experiment_name test_mcts_webquestions_f1_1iter \
    --metric joint_f1 &

wait

echo "All MCTS experiments completed!"