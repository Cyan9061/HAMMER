#!/bin/bash
# 统一的终止实验脚本
# 替代: kill_mcts_experiments.sh, run_mcts_vary_models_kill.sh, run_mcts_vary_coreset_kill.sh, HPO_Baseline/scripts/kill_all_experiments.sh

echo "Killing all running experiments..."

# 终止所有MCTS相关进程
echo "Killing MCTS experiments..."
pkill -f "main_tuner_mcts"
pkill -f "main_tuner_mcts_vary_model" 
pkill -f "main_tuner_mcts_vary_coreset"
pkill -f "main_tuner_mcts_simulationByLLM"
pkill -f "main_tuner_mcts_selectByLLM"
pkill -f "main_tuner_mcts_withoutMemory"

# 终止所有HPO Baseline相关进程
echo "Killing HPO Baseline experiments..."
pkill -f "run_baselines.py"
pkill -f "HPO_Baseline"

# 终止所有nohup后台进程
echo "Killing nohup background processes..."
pkill -f "nohup.*experiment"
pkill -f "nohup.*mcts"
pkill -f "nohup.*baseline"

# 终止所有Python实验进程
echo "Killing Python experiment processes..."
pkill -f "python.*experiment"
pkill -f "python.*tuner"
pkill -f "python.*mcts"

# 显示剩余的相关进程
echo "Checking for remaining experiment processes..."
ps aux | grep -E "(mcts|baseline|tuner|experiment)" | grep -v grep | grep -v "kill_all_experiments"

echo "Kill operation completed. Please verify no experiment processes are running."

# 可选：强制终止顽固进程
read -p "Force kill remaining processes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Force killing remaining processes..."
    pkill -9 -f "main_tuner"
    pkill -9 -f "run_baselines"
    pkill -9 -f "experiment"
    echo "Force kill completed."
fi