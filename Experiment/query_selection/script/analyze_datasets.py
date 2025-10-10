#!/usr/bin/env python3
"""
分析docs/dataset/unified目录下各个QA数据集的query数量
"""

import json
import os
from pathlib import Path

def count_queries_in_file(file_path):
    """统计单个文件中的query数量"""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data = json.loads(line)
                    if 'query' in data:
                        count += 1
        return count
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():

    print("="*60)
    print("数据集Query数量统计")
    print("="*60)
    
    total_queries = 0
    dataset_stats = []
    
    # 获取所有qa文件
    qa_files = [f for f in unified_dir.glob('*_qa_unified.json')]
    qa_files.sort()  # 按名称排序
    
    for qa_file in qa_files:
        dataset_name = qa_file.stem.replace('_qa_unified', '')
        query_count = count_queries_in_file(qa_file)
        
        # 计算70%训练集和30%测试集的大小
        train_size = int(query_count * 0.7)
        test_size = query_count - train_size
        
        # 计算30%的训练集用于选择
        selection_input_size = int(train_size * 0.3)
        
        dataset_stats.append({
            'dataset': dataset_name,
            'total': query_count,
            'train_size': train_size,
            'test_size': test_size,
            'selection_input_size': selection_input_size
        })
        
        total_queries += query_count
        
        print(f"{dataset_name:25} | 总计: {query_count:5} | 训练: {train_size:4} | 测试: {test_size:4} | 选择输入: {selection_input_size:3}")
    
    print("-"*60)
    print(f"{'总计':25} | 总计: {total_queries:5}")
    print("="*60)
    
    # 保存统计结果到JSON文件
    stats_file = '../../docs/dataset/unified_query_selection/temp/dataset_stats.json'
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存到: {stats_file}")
    
    return dataset_stats

if __name__ == '__main__':
    main()