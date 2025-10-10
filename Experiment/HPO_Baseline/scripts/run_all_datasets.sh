#!/bin/bash
# 合并后的HPO Baseline按数据集运行脚本
# 替代所有dataset_order/run_*.sh脚本

echo "Starting HPO Baseline Dataset Order Experiments..."

# 算法配置
ALGORITHMS=("greedy_m" "greedy_r" "greedy_rcc" "tpe" "random" "grid" "traditional_mcts")

# 函数：为指定数据集运行所有算法
run_dataset_experiments() {
    local dataset=$1
    local max_evals=$2
    local seed=$3
    
    echo "Running experiments for dataset: $dataset"
    
    for algo in "${ALGORITHMS[@]}"; do
        echo "  Running $algo on $dataset"
        
        if [ "$algo" = "traditional_mcts" ]; then
            # Traditional MCTS with higher max_evals
            python Experiment/HPO_Baseline/run_baselines.py \
                --algo $algo \
                --dataset $dataset \
                --metric joint_f1 \
                --max_evals 15 \
                --seed $seed
        else
            # Other algorithms
            python Experiment/HPO_Baseline/run_baselines.py \
                --algo $algo \
                --dataset $dataset \
                --metric joint_f1 \
                --max_evals $max_evals \
                --seed $seed
        fi
    done
    
    echo "Completed experiments for dataset: $dataset"
}

# 数据集特定配置
echo "=== Running 2wikimultihopqa experiments ==="
run_dataset_experiments "2wikimultihopqa" 5 101

echo "=== Running hotpotqa experiments ==="  
run_dataset_experiments "hotpotqa" 5 102

echo "=== Running medqa experiments ==="
run_dataset_experiments "medqa" 5 103

echo "=== Running eli5 experiments ==="
run_dataset_experiments "eli5" 5 104

echo "=== Running fiqa experiments ==="
run_dataset_experiments "fiqa" 5 105

echo "=== Running popqa experiments ==="
# popqa has special config with max_evals=1
echo "Running experiments for dataset: popqa"
for algo in "${ALGORITHMS[@]}"; do
    echo "  Running $algo on popqa"
    if [ "$algo" = "traditional_mcts" ]; then
        python Experiment/HPO_Baseline/run_baselines.py \
            --algo $algo \
            --dataset popqa \
            --metric joint_f1 \
            --max_evals 15 \
            --seed 106
    else
        python Experiment/HPO_Baseline/run_baselines.py \
            --algo $algo \
            --dataset popqa \
            --metric joint_f1 \
            --max_evals 1 \
            --seed 106
    fi
done

echo "=== Running webquestions experiments ==="
run_dataset_experiments "webquestions" 5 107

echo "=== Running quartz experiments ==="
# quartz has special config with max_evals=15 for some algorithms
echo "Running experiments for dataset: quartz"
for algo in "${ALGORITHMS[@]}"; do
    echo "  Running $algo on quartz"
    if [ "$algo" = "traditional_mcts" ]; then
        python Experiment/HPO_Baseline/run_baselines.py \
            --algo $algo \
            --dataset quartz \
            --metric joint_f1 \
            --max_evals 15 \
            --seed 108
    elif [ "$algo" = "greedy_m" ] || [ "$algo" = "greedy_r" ] || [ "$algo" = "greedy_rcc" ]; then
        python Experiment/HPO_Baseline/run_baselines.py \
            --algo $algo \
            --dataset quartz \
            --metric joint_f1 \
            --max_evals 15 \
            --seed 108
    else
        python Experiment/HPO_Baseline/run_baselines.py \
            --algo $algo \
            --dataset quartz \
            --metric joint_f1 \
            --max_evals 5 \
            --seed 108
    fi
done

echo "All dataset order experiments completed!"

# 并行运行所有数据集 (原run_all_by_datasets.sh的功能)
echo "Starting parallel execution of all dataset experiments..."

# 创建日志目录
mkdir -p Experiment/HPO_Baseline/logs/dataset_order

# 并行启动所有数据集实验
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments 2wikimultihopqa 5 101
" > Experiment/HPO_Baseline/logs/dataset_order/2wikimultihopqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments hotpotqa 5 102
" > Experiment/HPO_Baseline/logs/dataset_order/hotpotqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments medqa 5 103
" > Experiment/HPO_Baseline/logs/dataset_order/medqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments eli5 5 104
" > Experiment/HPO_Baseline/logs/dataset_order/eli5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments fiqa 5 105
" > Experiment/HPO_Baseline/logs/dataset_order/fiqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    # popqa特殊处理
    for algo in greedy_m greedy_r greedy_rcc tpe random grid; do
        python Experiment/HPO_Baseline/run_baselines.py --algo \$algo --dataset popqa --metric joint_f1 --max_evals 1 --seed 106
    done
    python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset popqa --metric joint_f1 --max_evals 15 --seed 106
" > Experiment/HPO_Baseline/logs/dataset_order/popqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    ./run_dataset_experiments webquestions 5 107
" > Experiment/HPO_Baseline/logs/dataset_order/webquestions.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash -c "
    # quartz特殊处理
    for algo in greedy_m greedy_r greedy_rcc; do
        python Experiment/HPO_Baseline/run_baselines.py --algo \$algo --dataset quartz --metric joint_f1 --max_evals 15 --seed 108
    done
    for algo in tpe random grid; do
        python Experiment/HPO_Baseline/run_baselines.py --algo \$algo --dataset quartz --metric joint_f1 --max_evals 5 --seed 108
    done
    python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset quartz --metric joint_f1 --max_evals 15 --seed 108
" > Experiment/HPO_Baseline/logs/dataset_order/quartz.log 2>&1 &

echo "All dataset experiments started in parallel. Check logs in Experiment/HPO_Baseline/logs/dataset_order/"