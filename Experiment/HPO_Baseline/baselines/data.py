
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 🔥 ONLY: 直接使用MCTS的数据流组件 - 完全统一架构
from hammer.mcts.mcts_dataset_loader import SimpleDataset
from hammer.storage import QAPair

# 🔥 数据集配置：使用unified_1路径（与MCTS的数据源差异是设计决策）
SUPPORTED_DATASETS = {
    '2wikimultihopqa': {
        'name': '2WikiMultiHopQA',
        'description': 'Multi-hop reasoning over Wikipedia',
        'qa_file': 'docs/dataset/unified_1/2wikimultihopqa_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/2wikimultihopqa_corpus_unified.json'
    },
    'hotpotqa': {
        'name': 'HotpotQA', 
        'description': 'Multi-hop reasoning QA',
        'qa_file': 'docs/dataset/unified_1/hotpotqa_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/hotpotqa_corpus_unified.json'
    },
    'medqa': {
        'name': 'MedQA-USMLE',
        'description': 'Medical question answering',
        'qa_file': 'docs/dataset/unified_1/MedQA_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/MedQA_corpus_unified.json'
    },
    'eli5': {
        'name': 'ELI5',
        'description': 'Explain Like I\'m 5 - long-form QA',
        'qa_file': 'docs/dataset/unified_1/eli5_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/eli5_corpus_unified.json'
    },
    'fiqa': {
        'name': 'FiQA',
        'description': 'Financial question answering dataset',
        'qa_file': 'docs/dataset/unified_1/fiqa_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/fiqa_corpus_unified.json'
    },
    'popqa': {
        'name': 'PopQA',
        'description': 'Popular entity questions',
        'qa_file': 'docs/dataset/unified_1/popqa_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/popqa_corpus_unified.json'
    },
    'quartz': {
        'name': 'QuALITY Reading Comprehension',
        'description': 'Science reasoning questions',
        'qa_file': 'docs/dataset/unified_1/quartz_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/quartz_corpus_unified.json'
    },
    'webquestions': {
        'name': 'WebQuestions',
        'description': 'Open-domain QA from web search',
        'qa_file': 'docs/dataset/unified_1/webquestions_qa_unified.json',
        'corpus_file': 'docs/dataset/unified_1/webquestions_corpus_unified.json'
    }
}

# ===============================================================================
# 🔥 CORE: 完全统一为MCTS架构的数据加载接口
# ===============================================================================

def load_qa_pairs(dataset_name: str, split: str = "test", limit: Optional[int] = None) -> List[QAPair]:
    """
    🔥 UNIFIED: 完全使用MCTS架构的唯一数据加载接口
    直接返回QAPair对象，与MCTS完全一致
    
    Args:
        dataset_name: 数据集名称
        split: 'train'或'test'或'all'
        limit: 限制返回数据量（可选）
    
    Returns:
        List[QAPair]: 与MCTS完全一致的QAPair对象列表
    """
    if dataset_name not in SUPPORTED_DATASETS:
        available = list(SUPPORTED_DATASETS.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: {available}")
    
    dataset_config = SUPPORTED_DATASETS[dataset_name]
    
    # 构建文件路径
    project_root = Path(__file__).parent.parent.parent.parent
    qa_file = project_root / dataset_config['qa_file']
    corpus_file = project_root / dataset_config['corpus_file']
    
    if not qa_file.exists():
        raise FileNotFoundError(f"QA数据文件未找到: {qa_file}")
    if not corpus_file.exists():
        raise FileNotFoundError(f"语料库文件未找到: {corpus_file}")
    
    # 🔥 KEY: 完全使用MCTS的SimpleDataset - 架构完全一致
    simple_dataset = SimpleDataset(
        corpus_file=str(corpus_file), 
        qa_file=str(qa_file), 
        dataset_name=dataset_name
    )
    
    # 🔥 KEY: 与MCTS完全一致的数据加载方式
    all_qa_pairs = simple_dataset.load_qa_pairs()
    
    print(f"📊 MCTS统一架构数据加载 - {dataset_config['name']}: 总共{len(all_qa_pairs)}条记录")
    
    # 🔥 HPO_Baseline特有的70%/30%数据划分逻辑（与MCTS的固定train_size划分的设计差异）
    if split == "all":
        result_qa_pairs = all_qa_pairs
        print(f"   📋 返回全部数据: {len(result_qa_pairs)}条记录")
        
    elif split == "train":
        total_samples = len(all_qa_pairs)
        train_size = int(total_samples * 0.7)  # 70%作为训练集
        result_qa_pairs = all_qa_pairs[:train_size]
        print(f"   📈 训练集(70%): {len(result_qa_pairs)}条记录")
        
    elif split == "test":
        total_samples = len(all_qa_pairs)
        train_size = int(total_samples * 0.7)
        result_qa_pairs = all_qa_pairs[train_size:]  # 30%作为测试集
        print(f"   📊 测试集(30%): {len(result_qa_pairs)}条记录")
        
    else:
        raise ValueError(f"不支持的split类型: {split}，支持'train'、'test'或'all'")
    
    # 如果指定了limit，则截取
    if limit and limit < len(result_qa_pairs):
        result_qa_pairs = result_qa_pairs[:limit]
        print(f"   🔧 数据截取到: {len(result_qa_pairs)}条记录")
    
    return result_qa_pairs

def load_train_test_split(dataset_name: str, 
                         train_limit: Optional[int] = None, 
                         test_limit: Optional[int] = None) -> Tuple[List[QAPair], List[QAPair]]:
    """
    🔥 UNIFIED: 为HPO_Baseline实验加载训练和测试数据集
    完全使用MCTS架构，70%/30%划分
    
    Args:
        dataset_name: 数据集名称
        train_limit: 训练集大小限制（可选）
        test_limit: 测试集大小限制（可选）
    
    Returns:
        (train_qa_pairs, test_qa_pairs): QAPair对象元组
    """
    print(f"📊 HPO_Baseline: 使用MCTS架构 + 70%/30%划分 - {dataset_name}")
    
    # 🔥 KEY: 直接获取QAPair对象，无格式转换
    train_qa_pairs = load_qa_pairs(dataset_name, "train", train_limit)
    test_qa_pairs = load_qa_pairs(dataset_name, "test", test_limit)
    
    print(f"🎯 最终数据集: 训练集={len(train_qa_pairs)}, 测试集={len(test_qa_pairs)}")
    print(f"✅ 架构完全统一: 直接使用QAPair对象，无格式转换")
    
    return train_qa_pairs, test_qa_pairs

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    🔥 UTILITY: 获取数据集信息
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        Dict: 数据集配置信息
    """
    if dataset_name not in SUPPORTED_DATASETS:
        available = list(SUPPORTED_DATASETS.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: {available}")
    
    return SUPPORTED_DATASETS[dataset_name].copy()

def list_supported_datasets() -> List[str]:
    """
    🔥 UTILITY: 列出所有支持的数据集名称
    
    Returns:
        List[str]: 支持的数据集名称列表
    """
    return list(SUPPORTED_DATASETS.keys())

# ===============================================================================
# 🔥 REMOVED: 完全移除的向后兼容代码
# ===============================================================================

# ❌ REMOVED: load_unified_dataset() - 不再支持字典格式
# ❌ REMOVED: load_dataset() - 不再支持字典格式  
# ❌ REMOVED: load_2wikimultihop_dataset() - 不再支持字典格式
# ❌ REMOVED: load_dataset_for_baseline() - 不再支持字典格式
# ❌ REMOVED: convert_unified_to_baseline_format() - 不再进行格式转换
# ❌ REMOVED: 所有字典格式转换逻辑

# ===============================================================================
# 🎯 ARCHITECTURE: 完全统一为MCTS架构
# ===============================================================================
# 
# HPO_Baseline现在与MCTS使用完全相同的架构：
# 
# 1. 数据加载：SimpleDataset (from hammer.mcts.mcts_dataset_loader)
# 2. 数据对象：QAPair (from hammer.storage)  
# 3. 数据流：直接QAPair对象，无格式转换
# 4. 评估流：使用相同的MultiHopQAEvaluator和BatchAPIEvaluator
# 
# 唯一的设计差异（有意保留）：
# - 数据源：HPO_Baseline使用unified_1，MCTS使用unified_query_selection
# - 数据划分：HPO_Baseline使用70%/30%，MCTS使用固定train_size
# 
# ===============================================================================