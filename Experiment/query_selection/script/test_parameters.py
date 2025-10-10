#!/usr/bin/env python3
"""
测试query selection参数的脚本，仅处理小数据集
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('../query_selection')

# 导入批量处理脚本的函数
from script.batch_query_selection import process_dataset

def test_parameters():
    """测试参数设置，仅处理小数据集"""
    print("="*60)
    print("参数测试 - 处理小数据集")
    print("="*60)
    
    # 选择较小的数据集进行测试
    test_datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']
    
    local_model_path = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
    
    results = []
    
    for dataset_name in test_datasets:
        try:
            print(f"\n{'='*40}")
            print(f"测试数据集: {dataset_name}")
            print(f"{'='*40}")
            
            result = process_dataset(dataset_name, model_name=local_model_path)
            if result:
                results.append(result)
                
                # 检查是否满足50%的限制
                if result['selection_ratio'] > 0.5:
                    print(f"警告: {dataset_name} 的选择比例 {result['selection_ratio']:.2%} 超过了50%的限制!")
                else:
                    print(f"✓ {dataset_name} 的选择比例 {result['selection_ratio']:.2%} 在50%限制内")
                    
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印测试结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    print(f"{'数据集':15} | {'原始':5} | {'选择':5} | {'最终':5} | {'比例':6} | {'状态':6}")
    print("-"*60)
    
    all_pass = True
    for result in results:
        status = "✓通过" if result['selection_ratio'] <= 0.5 else "✗超限"
        if result['selection_ratio'] > 0.5:
            all_pass = False
            
        print(f"{result['dataset']:15} | {result['original_size']:5} | {result['selected_size']:5} | "
              f"{result['final_size']:5} | {result['selection_ratio']:.2%} | {status}")
    
    print("-"*60)
    
    if all_pass:
        print("✓ 所有测试数据集都通过了50%的限制检查")
        print("可以继续处理大数据集")
    else:
        print("✗ 部分数据集超过了50%的限制，需要调整参数")
    
    return all_pass

if __name__ == '__main__':
    test_parameters()