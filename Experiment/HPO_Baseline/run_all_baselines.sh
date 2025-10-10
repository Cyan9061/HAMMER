#\!/bin/bash

# ==============================================================================
# 全自动Baseline实验执行脚本
# ==============================================================================
# 功能: 串行执行所有baseline算法在所有数据集上的实验
# 配置: max_evals=50, seed=42, 两轮实验 (lexical_ac + lexical_ff)
# ==============================================================================

set -e  # 遇到错误立即退出
set -u  # 遇到未定义变量立即退出

# 实验配置
MAX_EVALS=50
SEED=42
METRICS=("lexical_ac" "lexical_ff")
ALGORITHMS=("random" "grid" "tpe" "greedy_m" "greedy_r" "greedy_rcc")
DATASETS=("2wikimultihopqa" "hotpotqa" "musique" "finqa" "medqa" "bioasq")

# 日志配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="${LOG_DIR}/run_all_baselines_${TIMESTAMP}.log"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 日志函数
log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo "$msg" | tee -a "$MAIN_LOG"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$msg" | tee -a "$MAIN_LOG" >&2
}

log_section() {
    local msg="
================================================================================
🎯 $1
================================================================================"
    echo "$msg" | tee -a "$MAIN_LOG"
}

# 检查Python环境
check_python_env() {
    log_info "检查Python环境..."
    if \! command -v python &> /dev/null; then
        log_error "Python未找到，请确保Python已安装"
        exit 1
    fi
    
    local python_version=$(python --version 2>&1)
    log_info "Python版本: $python_version"
    
    # 检查run_baselines.py是否存在
    if [[ \! -f "${SCRIPT_DIR}/run_baselines.py" ]]; then
        log_error "run_baselines.py文件未找到"
        exit 1
    fi
    
    log_info "Python环境检查完成 ✅"
}

# 执行单个实验
run_single_experiment() {
    local algo="$1"
    local dataset="$2"
    local metric="$3"
    local max_evals="$4"
    local seed="$5"
    
    local exp_name="${algo}_${dataset}_${metric}_${max_evals}_${seed}"
    local exp_log="${LOG_DIR}/${exp_name}_${TIMESTAMP}.log"
    
    log_info "开始实验: ${exp_name}"
    log_info "日志文件: ${exp_log}"
    
    local start_time=$(date +%s)
    
    # 构建命令
    local cmd="python ${SCRIPT_DIR}/run_baselines.py --algo ${algo} --dataset ${dataset} --metric ${metric} --max_evals ${max_evals} --seed ${seed}"
    
    # 执行实验
    if $cmd > "${exp_log}" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_info "实验完成: ${exp_name} (耗时: ${duration}秒) ✅"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "实验失败: ${exp_name} (耗时: ${duration}秒) ❌"
        log_error "详细错误信息请查看: ${exp_log}"
        return 1
    fi
}

# 主实验流程
main() {
    log_section "全自动Baseline实验开始"
    log_info "实验配置:"
    log_info "  算法: ${ALGORITHMS[*]}"
    log_info "  数据集: ${DATASETS[*]}"
    log_info "  指标: ${METRICS[*]}"
    log_info "  最大评估次数: ${MAX_EVALS}"
    log_info "  随机种子: ${SEED}"
    log_info "  主日志: ${MAIN_LOG}"
    
    # 检查环境
    check_python_env
    
    local total_experiments=$((${#ALGORITHMS[@]} * ${#DATASETS[@]} * ${#METRICS[@]}))
    local current_exp=0
    local failed_experiments=0
    local start_time=$(date +%s)
    
    log_info "总实验数量: ${total_experiments}"
    
    # 双重循环：先按指标，再按算法和数据集
    for metric in "${METRICS[@]}"; do
        log_section "指标轮次: ${metric}"
        
        for algo in "${ALGORITHMS[@]}"; do
            log_section "算法: ${algo} (指标: ${metric})"
            
            for dataset in "${DATASETS[@]}"; do
                current_exp=$((current_exp + 1))
                
                log_info "进度: ${current_exp}/${total_experiments}"
                
                if \! run_single_experiment "$algo" "$dataset" "$metric" "$MAX_EVALS" "$SEED"; then
                    failed_experiments=$((failed_experiments + 1))
                    log_error "实验失败，继续下一个实验..."
                fi
                
                # 添加短暂延迟避免系统负载过高
                sleep 2
            done
        done
    done
    
    # 实验总结
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local successful_experiments=$((total_experiments - failed_experiments))
    
    log_section "实验总结"
    log_info "总实验数: ${total_experiments}"
    log_info "成功实验: ${successful_experiments}"
    log_info "失败实验: ${failed_experiments}"
    log_info "总耗时: ${total_duration}秒 ($(($total_duration / 60))分钟)"
    
    if [[ $failed_experiments -eq 0 ]]; then
        log_info "🎉 所有实验成功完成！"
        return 0
    else
        log_error "⚠️ 有 ${failed_experiments} 个实验失败"
        log_error "请查看相应的日志文件了解详情"
        return 1
    fi
}

# 信号处理：Ctrl+C时优雅退出
cleanup() {
    log_info "接收到中断信号，正在清理..."
    log_info "已完成的实验结果已保存"
    exit 1
}

trap cleanup SIGINT SIGTERM

# 执行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
SCRIPT_EOF < /dev/null
