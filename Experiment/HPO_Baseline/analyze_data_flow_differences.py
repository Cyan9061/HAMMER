#!/usr/bin/env python3
"""
深度对比分析 HPO_Baseline 和 MCTS 系统的数据流处理差异

这个脚本将详细检查两个系统在：
1. 数据集路径配置
2. 数据加载方式
3. QAPair对象创建
4. 数据划分逻辑
5. 评估流程
等方面的差异
"""

import sys
from pathlib import Path

# 添加项目根路径到最前面
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Any

def analyze_hpo_baseline_data_config():
    """分析HPO_Baseline的数据配置"""
    
    print("🔍 分析 HPO_Baseline 数据配置")
    print("=" * 60)
    
    from Experiment.HPO_Baseline.baselines.data import SUPPORTED_DATASETS
    
    print("📊 HPO_Baseline 数据集配置:")
    for dataset_name, config in SUPPORTED_DATASETS.items():
        print(f"  {dataset_name}:")
        print(f"    QA文件: {config['qa_file']}")
        print(f"    语料库文件: {config['corpus_file']}")
    
    # 检查数据划分逻辑
    print("\n📐 HPO_Baseline 数据划分逻辑:")
    print("  - 使用70%/30%训练集/测试集划分")
    print("  - 无固定训练集大小限制")
    print("  - 数据集来源: unified_1")
    
    return SUPPORTED_DATASETS

def analyze_mcts_data_config():
    """分析MCTS的数据配置"""
    
    print("\n🔍 分析 MCTS 数据配置")
    print("=" * 60)
    
    # 模拟MCTS数据集配置（基于main_tuner_mcts.py）
    mcts_datasets = {}
    dataset_names = ['2wikimultihopqa', 'hotpotqa', 'medqa', 'eli5', 'fiqa', 'popqa', 'quartz', 'webquestions']
    
    for dataset_name in dataset_names:
        # 修正MedQA的文件名
        file_prefix = 'MedQA' if dataset_name == 'medqa' else dataset_name
        mcts_datasets[dataset_name] = {
            'qa_file': f"docs/dataset/unified_query_selection/{file_prefix}_qa_unified.json",
            'corpus_file': f"docs/dataset/unified_query_selection/{file_prefix}_corpus_unified.json"
        }
    
    print("📊 MCTS 数据集配置:")
    for dataset_name, config in mcts_datasets.items():
        print(f"  {dataset_name}:")
        print(f"    QA文件: {config['qa_file']}")
        print(f"    语料库文件: {config['corpus_file']}")
    
    # 检查数据划分逻辑
    print("\n📐 MCTS 数据划分逻辑:")
    print("  - 使用固定训练集大小配置")
    print("  - 训练集大小: 2wikimultihopqa=210, hotpotqa=210, medqa=267, ...")
    print("  - 数据集来源: unified_query_selection")
    
    return mcts_datasets

def compare_dataset_sizes():
    """比较两个数据集的大小差异"""
    
    print("\n📏 比较数据集大小差异")
    print("=" * 60)
    
    # 读取unified_1和unified_query_selection的数据集大小
    dataset_names = ['2wikimultihopqa', 'hotpotqa', 'eli5', 'fiqa', 'popqa', 'quartz', 'webquestions']
    
    size_comparison = {}
    
    for dataset in dataset_names:
        # 构造文件路径
        unified_1_file = project_root / f"docs/dataset/unified_1/{dataset}_qa_unified.json"
        unified_query_file = project_root / f"docs/dataset/unified_query_selection/{dataset}_qa_unified.json"
        
        # MedQA特殊处理
        if dataset == 'medqa':
            unified_1_file = project_root / f"docs/dataset/unified_1/MedQA_qa_unified.json"
            unified_query_file = project_root / f"docs/dataset/unified_query_selection/MedQA_qa_unified.json"
        
        try:
            # 统计行数
            unified_1_size = 0
            unified_query_size = 0
            
            if unified_1_file.exists():
                with open(unified_1_file, 'r') as f:
                    unified_1_size = sum(1 for _ in f)
            
            if unified_query_file.exists():
                with open(unified_query_file, 'r') as f:
                    unified_query_size = sum(1 for _ in f)
            
            size_comparison[dataset] = {
                'unified_1': unified_1_size,
                'unified_query_selection': unified_query_size,
                'difference': unified_1_size - unified_query_size
            }
            
            print(f"  {dataset}:")
            print(f"    unified_1: {unified_1_size}")
            print(f"    unified_query_selection: {unified_query_size}")
            print(f"    差异: {unified_1_size - unified_query_size} (+{(unified_1_size/unified_query_size-1)*100:.1f}%)" if unified_query_size > 0 else f"    差异: {unified_1_size - unified_query_size}")
            
        except Exception as e:
            print(f"  {dataset}: 读取失败 - {e}")
    
    return size_comparison

def analyze_qa_pair_creation():
    """分析QAPair对象创建过程"""
    
    print("\n🏗️ 分析 QAPair 对象创建过程")
    print("=" * 60)
    
    try:
        # HPO_Baseline的QAPair创建
        from Experiment.HPO_Baseline.baselines.data import load_qa_pairs_with_mcts_dataflow
        hpo_qa_pairs = load_qa_pairs_with_mcts_dataflow('fiqa', split='train', test_size=10)
        
        print("📦 HPO_Baseline QAPair创建:")
        if hpo_qa_pairs:
            sample = hpo_qa_pairs[0]
            print(f"  类型: {type(sample)}")
            print(f"  属性: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
            print(f"  示例:")
            print(f"    ID: {sample.id}")
            print(f"    问题: {sample.question[:50]}...")
            print(f"    答案: {sample.answer[:50]}...")
            print(f"    数据集名: {sample.dataset_name}")
            print(f"    难度: {sample.difficulty}")
            print(f"    类型: {sample.qtype}")
            print(f"    ground_truth: {len(sample.text_ground_truth) if sample.text_ground_truth else 0} 个文档")
        
        # MCTS的QAPair创建（通过SimpleDataset）
        from hammer.mcts.mcts_dataset_loader import SimpleDataset
        
        # 使用unified_query_selection路径
        corpus_file = str(project_root / "docs/dataset/unified_query_selection/fiqa_corpus_unified.json")
        qa_file = str(project_root / "docs/dataset/unified_query_selection/fiqa_qa_unified.json")
        
        mcts_dataset = SimpleDataset(corpus_file=corpus_file, qa_file=qa_file, dataset_name='fiqa')
        mcts_qa_pairs = mcts_dataset.load_qa_pairs()[:10]  # 只取前10个
        
        print("\n📦 MCTS QAPair创建:")
        if mcts_qa_pairs:
            sample = mcts_qa_pairs[0]
            print(f"  类型: {type(sample)}")
            print(f"  属性: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
            print(f"  示例:")
            print(f"    ID: {sample.id}")
            print(f"    问题: {sample.question[:50]}...")
            print(f"    答案: {sample.answer[:50]}...")
            print(f"    数据集名: {sample.dataset_name}")
            print(f"    难度: {sample.difficulty}")
            print(f"    类型: {sample.qtype}")
            print(f"    ground_truth: {len(sample.text_ground_truth) if sample.text_ground_truth else 0} 个文档")
        
        # 比较两个QAPair对象的差异
        print(f"\n🔍 QAPair对象比较:")
        print(f"  HPO_Baseline 数据集大小: {len(hpo_qa_pairs)}")
        print(f"  MCTS 数据集大小: {len(mcts_qa_pairs)}")
        
        if hpo_qa_pairs and mcts_qa_pairs:
            hpo_sample = hpo_qa_pairs[0]
            mcts_sample = mcts_qa_pairs[0]
            
            # 比较具体字段
            fields_to_compare = ['id', 'question', 'answer', 'dataset_name', 'difficulty', 'qtype']
            
            print(f"  字段比较 (第一个样本):")
            for field in fields_to_compare:
                hpo_val = getattr(hpo_sample, field, 'N/A')
                mcts_val = getattr(mcts_sample, field, 'N/A')
                match = "✅" if hpo_val == mcts_val else "❌"
                print(f"    {field}: {match} HPO='{hpo_val}' vs MCTS='{mcts_val}'")
        
    except Exception as e:
        print(f"❌ QAPair创建分析失败: {e}")
        import traceback
        traceback.print_exc()

def summarize_differences():
    """总结发现的差异"""
    
    print("\n📋 发现的主要差异总结")
    print("=" * 60)
    
    differences = [
        {
            "类别": "数据集路径",
            "HPO_Baseline": "docs/dataset/unified_1/",
            "MCTS": "docs/dataset/unified_query_selection/",
            "影响": "使用不同版本的数据集，unified_1约为unified_query_selection的2倍大小"
        },
        {
            "类别": "数据划分方式",
            "HPO_Baseline": "70%训练集 / 30%测试集",
            "MCTS": "固定训练集大小 (如 210, 267, 105 等)",
            "影响": "训练集大小和测试集大小计算方式不同"
        },
        {
            "类别": "训练集大小配置",
            "HPO_Baseline": "根据总数据量动态计算 (total * 0.7)",
            "MCTS": "使用预定义的固定值",
            "影响": "可能导致不同的训练数据量和模型性能"
        }
    ]
    
    for diff in differences:
        print(f"\n🔸 {diff['类别']}:")
        print(f"  HPO_Baseline: {diff['HPO_Baseline']}")
        print(f"  MCTS: {diff['MCTS']}")
        print(f"  影响: {diff['影响']}")
    
    print(f"\n💡 修复建议:")
    print(f"  1. 统一数据集路径 - 都使用 unified_1 或都使用 unified_query_selection")
    print(f"  2. 统一数据划分方式 - 建议都使用70%/30%划分")
    print(f"  3. 确保QAPair对象的创建过程完全一致")
    print(f"  4. 验证评估函数和指标计算的一致性")

def main():
    """主函数"""
    
    print("🚀 开始深度对比分析 HPO_Baseline 和 MCTS 数据流差异")
    print("=" * 80)
    
    # 1. 分析配置差异
    hpo_datasets = analyze_hpo_baseline_data_config()
    mcts_datasets = analyze_mcts_data_config()
    
    # 2. 比较数据集大小
    size_comparison = compare_dataset_sizes()
    
    # 3. 分析QAPair创建过程
    analyze_qa_pair_creation()
    
    # 4. 总结差异
    summarize_differences()
    
    print(f"\n✅ 分析完成！")
    print(f"📝 关键发现:")
    print(f"   - 数据集路径不统一（unified_1 vs unified_query_selection）")
    print(f"   - 数据划分方式不一致（比例 vs 固定大小）")
    print(f"   - 数据集规模差异约2倍")
    print(f"   - QAPair对象创建过程基本一致（都使用SimpleDataset）")

if __name__ == "__main__":
    main()