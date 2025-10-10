#!/usr/bin/env python3
"""
将数据集按70%训练集和30%测试集进行划分，并从训练集中提取30%用于query selection
"""

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

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

def split_dataset(dataset_name):
    """划分单个数据集"""
    print(f"\n处理数据集: {dataset_name}")
    
    # 文件路径
    unified_dir = Path('../../docs/dataset/unified')
    temp_dir = Path('../../docs/dataset/unified_query_selection/temp')
    
    qa_file = unified_dir / f'{dataset_name}_qa_unified.json'
    
    if not qa_file.exists():
        print(f"文件不存在: {qa_file}")
        return
    
    # 加载数据
    data = load_jsonl(qa_file)
    print(f"加载了 {len(data)} 条数据")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算划分点
    total_size = len(data)
    train_size = int(total_size * 0.7)
    test_size = total_size - train_size
    
    # 划分数据
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 从训练集中提取30%用于query selection
    selection_input_size = int(train_size * 0.3)
    selection_input_data = train_data[:selection_input_size]
    remaining_train_data = train_data[selection_input_size:]
    
    print(f"用于query selection的输入: {len(selection_input_data)} 条")
    print(f"剩余训练数据: {len(remaining_train_data)} 条")
    
    # 保存划分结果
    save_jsonl(train_data, temp_dir / f'{dataset_name}_train.jsonl')
    save_jsonl(test_data, temp_dir / f'{dataset_name}_test.jsonl')
    save_jsonl(selection_input_data, temp_dir / f'{dataset_name}_selection_input.jsonl')
    save_jsonl(remaining_train_data, temp_dir / f'{dataset_name}_remaining_train.jsonl')
    
    # 返回统计信息
    return {
        'dataset': dataset_name,
        'total_size': total_size,
        'train_size': train_size,
        'test_size': test_size,
        'selection_input_size': selection_input_size,
        'remaining_train_size': len(remaining_train_data)
    }

def main():
    """主函数"""
    print("="*60)
    print("数据集划分处理")
    print("="*60)
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 数据集列表
    datasets = [
        '2wikimultihopqa',
        'FinQA', 
        'MedQA',
        'bioasq',
        'hotpotqa',
        'musique'
    ]
    
    split_stats = []
    
    for dataset_name in datasets:
        stats = split_dataset(dataset_name)
        if stats:
            split_stats.append(stats)
    
    # 保存划分统计信息
    stats_file = '../../docs/dataset/unified_query_selection/temp/split_stats.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("划分统计汇总:")
    print("="*60)
    print(f"{'数据集':15} | {'总计':5} | {'训练':4} | {'测试':4} | {'选择输入':6} | {'剩余训练':6}")
    print("-"*60)
    
    for stats in split_stats:
        print(f"{stats['dataset']:15} | {stats['total_size']:5} | {stats['train_size']:4} | "
              f"{stats['test_size']:4} | {stats['selection_input_size']:6} | {stats['remaining_train_size']:6}")
    
    print(f"\n划分统计已保存到: {stats_file}")

if __name__ == '__main__':
    main()