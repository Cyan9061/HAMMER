#!/usr/bin/env python3
"""
对graph_unified目录下的QA数据集进行随机打乱
直接覆盖原文件
"""

import json
import random
import os

def shuffle_qa_dataset(file_path: str):
    """
    对JSONL格式的QA数据集进行随机打乱
    """
    print(f"📖 读取文件: {file_path}")
    
    # 读取所有数据
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    original_count = len(data)
    print(f"📊 原始数据量: {original_count}")
    
    # 随机打乱
    random.shuffle(data)
    print(f"🔀 数据已随机打乱")
    
    # 覆盖原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存到原文件: {file_path}")
    return original_count

def main():
    """
    主函数：对所有QA文件进行随机打乱
    """
    # 设置随机种子以确保可重现性（可选）
    random.seed(42)
    
    base_dir = "docs/dataset/graph_unified"
    
    print("🔀 GraphRAG-Benchmark QA数据集随机打乱工具")
    print("=" * 60)
    
    # 要处理的QA文件列表
    qa_files = [
        "graph_medical_qa_unified.json",
        "graph_novel_qa_unified.json"
    ]
    
    total_processed = 0
    
    for qa_file in qa_files:
        file_path = os.path.join(base_dir, qa_file)
        
        if os.path.exists(file_path):
            print(f"\n🔄 处理文件: {qa_file}")
            count = shuffle_qa_dataset(file_path)
            total_processed += count
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"🎉 随机打乱完成!")
    print(f"📊 总计处理: {total_processed} 条QA记录")
    print(f"📁 目录: {base_dir}")

if __name__ == "__main__":
    main()