
import sys
from pathlib import Path
from hyperopt import hp

# 添加hammer包到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from hammer.mcts.hierarchical_search import RAGSearchSpace

def build_hyperopt_space_from_rag_search_space():
    """
    直接从hierarchical_search.py的RAGSearchSpace构建hyperopt搜索空间
    """
    rag_space = RAGSearchSpace()
    space = {}
    
    # 文本分割器参数
    space['splitter_method'] = hp.choice('splitter_method', rag_space.splitter_methods)
    space['splitter_chunk_size'] = hp.choice('splitter_chunk_size', rag_space.splitter_chunk_sizes)
    space['splitter_overlap'] = hp.choice('splitter_overlap', rag_space.splitter_overlaps)
    
    # 嵌入模型
    space['embedding_model'] = hp.choice('embedding_model', rag_space.embedding_models)
    
    # 检索配置
    space['retrieval_method'] = hp.choice('retrieval_method', rag_space.retrieval_methods)
    space['retrieval_top_k'] = hp.choice('retrieval_top_k', rag_space.retrieval_top_k_options)
    space['hybrid_bm25_weight'] = hp.choice('hybrid_bm25_weight', rag_space.hybrid_bm25_weights)
    
    # 查询分解
    space['query_decomposition_num_queries'] = hp.choice('query_decomposition_num_queries', rag_space.query_decomposition_num_queries_options)
    space['query_decomposition_llm'] = hp.choice('query_decomposition_llm', rag_space.query_decomposition_llms)
    space['fusion_mode'] = hp.choice('fusion_mode', rag_space.fusion_modes)
    
    # 增强模块
    # space['hyde_llm'] = hp.choice('hyde_llm', rag_space.hyde_llms)
    space['reranker_llm'] = hp.choice('reranker_llm', rag_space.reranker_llms)
    space['reranker_top_k'] = hp.choice('reranker_top_k', rag_space.reranker_top_k_options)
    space['additional_context_num_nodes'] = hp.choice('additional_context_num_nodes', rag_space.additional_context_num_nodes_options)
    
    # 输出配置
    space['response_synthesizer_llm'] = hp.choice('response_synthesizer_llm', rag_space.response_synthesizer_llms)
    space['template_name'] = hp.choice('template_name', rag_space.template_names)
    
    # 启用/禁用开关（🔧 与TPE配置对齐）
    space['query_decomposition_enabled'] = hp.choice('query_decomposition_enabled', [True])  # TPE默认开启
    space['hyde_enabled'] = hp.choice('hyde_enabled', [False])  # 🔥 强制关闭HyDE
    space['reranker_enabled'] = hp.choice('reranker_enabled', [True])  # 🔥 改为开启以测试reranker模型
    space['additional_context_enabled'] = hp.choice('additional_context_enabled', [True])  # TPE默认开启
    space['few_shot_enabled'] = hp.choice('few_shot_enabled', [False])  # TPE默认关闭
    
    return space

def get_mcts_search_space():
    """获取MCTS的原始搜索空间对象"""
    return RAGSearchSpace()

# 兼容性函数
def build_hyperopt_space(ss):
    """为了兼容性保留，实际不使用ss参数"""
    return build_hyperopt_space_from_rag_search_space()

def create_mcts_compatible_search_space():
    """返回MCTS搜索空间对象"""
    return get_mcts_search_space()