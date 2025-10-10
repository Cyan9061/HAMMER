#!/usr/bin/env python3
"""
快速修正脚本：重新合并选择结果和测试数据，不包括剩余训练数据
"""

import json
import os
from pathlib import Path

def load_jsonl(file_path):
    """加载JSONL格式文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """保存为JSONL格式文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def fix_dataset_merge(dataset_name):
    """修正单个数据集的合并逻辑"""
    print(f"\n修正数据集: {dataset_name}")
    
    temp_dir = Path('../../docs/dataset/unified_query_selection/temp')
    result_dir = Path('../../docs/dataset/unified_query_selection')
    
    # 加载选择结果和测试数据
    selected_file = temp_dir / f'{dataset_name}_selected.jsonl'
    test_file = temp_dir / f'{dataset_name}_test.jsonl'
    
    if not selected_file.exists() or not test_file.exists():
        print(f"文件不存在: {selected_file} 或 {test_file}")
        return None
        
    selected_data = load_jsonl(selected_file)
    test_data = load_jsonl(test_file)
    
    print(f"选择数据: {len(selected_data)} 条")
    print(f"测试数据: {len(test_data)} 条")
    
    # 正确的合并逻辑：只合并选择结果和测试数据
    final_data = selected_data + test_data
    
    print(f"最终数据集大小: {len(final_data)} 条")
    
    # 保存修正后的结果
    output_file = result_dir / f'{dataset_name}_qa_unified.json'
    save_jsonl(final_data, output_file)
    
    print(f"修正结果已保存到: {output_file}")
    
    # 计算统计信息
    unified_dir = Path('../../docs/dataset/unified')
    original_file = unified_dir / f'{dataset_name}_qa_unified.json'
    original_count = len(load_jsonl(original_file))
    
    return {
        'dataset': dataset_name,
        'original_size': original_count,
        'selected_size': len(selected_data),
        'test_size': len(test_data),
        'final_size': len(final_data),
        'selection_ratio': len(selected_data) / original_count,
        'final_ratio': len(final_data) / original_count
    }

def main():
    """主函数"""
    print("="*60)
    print("修正Query Selection数据合并逻辑")
    print("="*60)
    
    datasets = [
        '2wikimultihopqa',
        'FinQA',
        'MedQA', 
        'bioasq',
        'hotpotqa',
        'musique'
    ]
    
    results = []
    
    for dataset_name in datasets:
        try:
            result = fix_dataset_merge(dataset_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {e}")
    
    # 保存修正后的统计信息
    stats_file = '../../docs/dataset/unified_query_selection/temp/corrected_selection_results.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印修正后的统计
    print("\n" + "="*70)
    print("修正后的Query Selection处理结果汇总:")
    print("="*70)
    print(f"{'数据集':15} | {'原始':5} | {'选择':5} | {'测试':4} | {'最终':5} | {'选择比例':8} | {'最终比例':8}")
    print("-"*70)
    
    total_original = 0
    total_selected = 0
    total_test = 0
    total_final = 0
    
    for result in results:
        print(f"{result['dataset']:15} | {result['original_size']:5} | {result['selected_size']:5} | "
              f"{result['test_size']:4} | {result['final_size']:5} | {result['selection_ratio']:8.2%} | {result['final_ratio']:8.2%}")
        total_original += result['original_size']
        total_selected += result['selected_size'] 
        total_test += result['test_size']
        total_final += result['final_size']
    
    print("-"*70)
    print(f"{'总计':15} | {total_original:5} | {total_selected:5} | {total_test:4} | {total_final:5} | "
          f"{total_selected/total_original:8.2%} | {total_final/total_original:8.2%}")
    
    print(f"\n修正后的统计已保存到: {stats_file}")
    
    # 验证结果
    print("\n" + "="*60)
    print("验证修正结果:")
    print("="*60)
    
    overall_selection_ratio = total_selected / total_original
    overall_final_ratio = total_final / total_original
    
    print(f"✓ 选择比例: {overall_selection_ratio:.2%} (目标: ~21%)")
    print(f"✓ 最终数据集比例: {overall_final_ratio:.2%} (目标: ~51%)")
    
    if 0.20 <= overall_selection_ratio <= 0.22:
        print("✅ 选择比例符合预期")
    else:
        print("⚠️  选择比例不在预期范围内")
        
    if 0.50 <= overall_final_ratio <= 0.52:
        print("✅ 最终数据集比例符合预期")
    else:
        print("⚠️  最终数据集比例不在预期范围内")

if __name__ == '__main__':
    main()