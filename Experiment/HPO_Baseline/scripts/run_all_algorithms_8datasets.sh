#!/bin/bash
# 合并后的HPO Baseline算法实验脚本 - 统一的8数据集实验
# 替代所有*_8datasets.sh脚本

echo "Starting HPO Baseline 8 Datasets Experiments..."

# 定义数据集列表
DATASETS=("2wikimultihopqa" "medqa" "eli5" "fiqa" "popqa" "quartz" "webquestions" "hotpotqa")

# 定义算法列表及其配置
declare -A ALGOS=(
    ["greedy_m"]="5"
    ["greedy_r"]="5" 
    ["greedy_rcc"]="5"
    ["tpe"]="5"
    ["random"]="5"
    ["grid"]="5"
    ["traditional_mcts"]="15"
)

# 函数：运行单个算法在所有数据集上
run_algorithm() {
    local algo=$1
    local max_evals=$2
    local base_seed=$3
    
    echo "Running algorithm: $algo (max_evals=$max_evals)"
    
    for i in "${!DATASETS[@]}"; do
        local dataset=${DATASETS[$i]}
        local seed=$((base_seed + i + 1))
        
        echo "  Running $algo on $dataset (seed=$seed)"
        
        if [ "$algo" = "traditional_mcts" ]; then
            # traditional_mcts runs in background
            python Experiment/HPO_Baseline/run_baselines.py \
                --algo $algo \
                --dataset $dataset \
                --metric joint_f1 \
                --max_evals $max_evals \
                --seed $seed &
        else
            # Other algorithms run sequentially
            python Experiment/HPO_Baseline/run_baselines.py \
                --algo $algo \
                --dataset $dataset \
                --metric joint_f1 \
                --max_evals $max_evals \
                --seed $seed
        fi
    done
    
    # Wait for background processes if any
    if [ "$algo" = "traditional_mcts" ]; then
        wait
    fi
    
    echo "Completed algorithm: $algo"
}

# 运行所有算法
seed_base=100
for algo in "${!ALGOS[@]}"; do
    max_evals=${ALGOS[$algo]}
    run_algorithm $algo $max_evals $seed_base
    seed_base=$((seed_base + 100))
done

echo "All HPO Baseline 8 Datasets experiments completed!"

# 运行合并的all experiments (原run_all_8datasets_experiments.sh的功能)
echo "Starting background execution of all experiments..."

# 启动所有算法的后台执行
nohup bash -c "
    echo 'Running Greedy M 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((300 + \$i + 3))
        python Experiment/HPO_Baseline/run_baselines.py --algo greedy_m --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/greedy_m_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running Greedy R 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((400 + \$i + 3))
        python Experiment/HPO_Baseline/run_baselines.py --algo greedy_r --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/greedy_r_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running Greedy RCC 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((500 + \$i + 5))
        python Experiment/HPO_Baseline/run_baselines.py --algo greedy_rcc --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/greedy_rcc_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running TPE 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((200 + \$i + 1))
        python Experiment/HPO_Baseline/run_baselines.py --algo tpe --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/tpe_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running Random 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((100 + \$i + 5))
        python Experiment/HPO_Baseline/run_baselines.py --algo random --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/random_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running Grid 8 datasets...'
    for i in {0..7}; do
        dataset=\${DATASETS[\$i]}
        seed=\$((600 + \$i + 6))
        python Experiment/HPO_Baseline/run_baselines.py --algo grid --dataset \$dataset --metric joint_f1 --max_evals 5 --seed \$seed
    done
" > Experiment/HPO_Baseline/logs/grid_8datasets.log 2>&1 &

nohup bash -c "
    echo 'Running Traditional MCTS 8 datasets...'
    python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset fiqa --metric joint_f1 --max_evals 15 --seed 705 &
    python Experiment/HPO_Baseline/run_baselines.py --algo traditional_mcts --dataset quartz --metric joint_f1 --max_evals 15 --seed 707 &
    wait
" > Experiment/HPO_Baseline/logs/traditional_mcts_8datasets.log 2>&1 &

echo "All experiments started in background. Check logs in Experiment/HPO_Baseline/logs/"