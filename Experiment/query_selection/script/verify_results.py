#!/usr/bin/env python3
"""
验证query selection结果的脚本
"""

import json
from pathlib import Path

def count_jsonl_lines(file_path):
    """统计JSONL文件的行数"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def verify_results():
    """验证query selection的结果"""
    print("="*60)
    print("验证Query Selection结果")
    print("="*60)
    
    unified_dir = Path('../../docs/dataset/unified')
    result_dir = Path('../../docs/dataset/unified_query_selection')
    temp_dir = result_dir / 'temp'
    
    datasets = [
        '2wikimultihopqa',
        'FinQA',
        'MedQA', 
        'bioasq',
        'hotpotqa',
        'musique'
    ]
    
    print(f"{'数据集':15} | {'原始':5} | {'最终':5} | {'选择':5} | {'剩余训练':6} | {'测试':4} | {'比例':6}")
    print("-"*68)
    
    total_original = 0
    total_final = 0
    total_selected = 0
    
    for dataset_name in datasets:
        # 原始数据量
        original_file = unified_dir / f'{dataset_name}_qa_unified.json'
        original_count = count_jsonl_lines(original_file)
        
        # 最终结果数据量
        result_file = result_dir / f'{dataset_name}_qa_unified.json'
        final_count = count_jsonl_lines(result_file)
        
        # 选择的数据量
        selected_file = temp_dir / f'{dataset_name}_selected.jsonl'
        selected_count = count_jsonl_lines(selected_file)
        
        # 剩余训练数据量
        remaining_train_file = temp_dir / f'{dataset_name}_remaining_train.jsonl'
        remaining_train_count = count_jsonl_lines(remaining_train_file)
        
        # 测试数据量
        test_file = temp_dir / f'{dataset_name}_test.jsonl'
        test_count = count_jsonl_lines(test_file)
        
        # 计算选择比例
        selection_ratio = selected_count / original_count
        
        # 验证数据完整性
        expected_final = selected_count + remaining_train_count + test_count
        integrity_check = "✓" if final_count == expected_final == original_count else "✗"
        
        print(f"{dataset_name:15} | {original_count:5} | {final_count:5} | {selected_count:5} | "
              f"{remaining_train_count:6} | {test_count:4} | {selection_ratio:.2%} {integrity_check}")
        
        total_original += original_count
        total_final += final_count
        total_selected += selected_count
        
        # 验证选择比例是否在50%以下
        if selection_ratio > 0.5:
            print(f"  ⚠️  警告: {dataset_name} 的选择比例超过50%！")
    
    print("-"*68)
    print(f"{'总计':15} | {total_original:5} | {total_final:5} | {total_selected:5} | "
          f"{'':6} | {'':4} | {total_selected/total_original:.2%}")
    
    print("\n" + "="*60)
    print("验证结果:")
    print("="*60)
    
    # 整体检查
    if total_original == total_final:
        print("✓ 数据完整性检查通过：原始数据量 = 最终数据量")
    else:
        print(f"✗ 数据完整性检查失败：原始数据量({total_original}) ≠ 最终数据量({total_final})")
    
    overall_ratio = total_selected / total_original
    if overall_ratio <= 0.5:
        print(f"✓ 选择比例检查通过：总体选择比例({overall_ratio:.2%}) ≤ 50%")
    else:
        print(f"✗ 选择比例检查失败：总体选择比例({overall_ratio:.2%}) > 50%")
    
    # 检查所有文件是否存在
    all_files_exist = True
    for dataset_name in datasets:
        result_file = result_dir / f'{dataset_name}_qa_unified.json'
        if not result_file.exists():
            print(f"✗ 文件不存在: {result_file}")
            all_files_exist = False
    
    if all_files_exist:
        print("✓ 所有结果文件都已正确生成")
    
    print("\n🎉 Query Selection任务完成!")
    print(f"📊 处理了 {len(datasets)} 个数据集，共 {total_original:,} 条原始数据")
    print(f"🎯 选择了 {total_selected:,} 条高质量数据 (选择比例: {overall_ratio:.2%})")
    print(f"💾 结果已保存到: {result_dir}")

if __name__ == '__main__':
    verify_results()