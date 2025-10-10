
import typing as T
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from hammer.llm import LLMs, get_llm
from hammer.logger import logger

def get_reranker(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """
    增强版Reranker工厂函数。
    根据提供的名称和 top_k 参数，实例化并返回一个 Reranker 后处理器。
    """
    logger.info(f"🔧 正在尝试创建 Reranker: name='{name}', top_k={top_k}")

    # 检查是否为自定义的 reranker 模型
    if name in ["ColbertRanker", "colbert_ranker"]:
        try:
            from llama_index.postprocessor.colbert_rerank import ColbertRerank
            return ColbertRerank(top_n=top_k)
        except ImportError:
            logger.warning("⚠️ ColbertRanker not available. Try: pip install llama-index-postprocessor-colbert-rerank")

    elif name in ["Flashrank", "flashrank"]:
        try:
            from llama_index.postprocessor.flashrank import FlashrankRerank
            return FlashrankRerank(top_n=top_k)
        except ImportError:
            logger.warning("⚠️ Flashrank not available. Try: pip install llama-index-postprocessor-flashrank")
    
    # 添加更多基础reranker支持
    elif name in ["TransformerRanker", "transformer_ranker"]:
        try:
            from hammer.rerankers.integrations import BatchSentenceTransformerReranker
            logger.info(f"🚀 创建批处理TransformerRanker: top_k={top_k}")
            return BatchSentenceTransformerReranker(
                model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                top_n=top_k,
                batch_size=32  # 优化的批处理大小
            )
        except ImportError:
            logger.warning("⚠️ SentenceTransformers not available for TransformerRanker")
    
    # 🔑 改进T5系列模型的处理逻辑
    elif name in ["MonoT5", "monot5", "RankT5", "rankt5", "ListT5", "listt5"]:
        logger.warning(f"⚠️ {name} not available in user's framework. Attempting LLM fallback.")
        # 尝试作为LLM处理
        return _try_llm_fallback(name, top_k)
    
    # 添加对其他模型的支持  
    elif name in ["TWOLAR", "twolar", "MonoBERT", "monobert_ranker", "InRanker", "inranker", "EchoRank", "echorank"]:
        logger.warning(f"⚠️ {name} not available in user's framework. Attempting LLM fallback.")
        # 尝试作为LLM处理
        return _try_llm_fallback(name, top_k)

    # 检查它是否为一个已知的 LLM
    elif name in LLMs:
        # 🚀 优化：不再创建串行LLMRerank，让OptimizedRAGPromptBuilder处理并行reranking
        logger.info(f"   -> 检测到 '{name}' 是一个 LLM, 跳过创建LLMRerank（将使用并行处理）")
        return None  # 返回None，让并行处理器处理
        
        # try:
        #     from llama_index.core.postprocessor import LLMRerank
        #     logger.info(f"   -> 检测到 '{name}' 是一个 LLM, 创建 LLMRerank 实例。")
        #     llm = get_llm(name)
        #     return LLMRerank(top_n=top_k, llm=llm)
        # except Exception as e:
        #     logger.error(f"❌ 无法创建LLMRerank for {name}: {e}")
            
    # 如果名称未被识别，但尝试作为LLM处理
    else:
        logger.warning(f"⚠️ 未知的 Reranker 或 LLM 名称: '{name}'。将使用并行处理。")
        return None  # 返回None，让并行处理器处理

def _try_llm_fallback(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """尝试将reranker名称作为LLM处理的回退函数"""
    # 🚀 优化：不再创建串行LLMRerank，让OptimizedRAGPromptBuilder处理并行reranking
    logger.info(f"✅ 将使用并行处理替代传统LLMRerank for {name}")
    return None  # 返回None，让并行处理器处理
    
    # try:
    #     from llama_index.core.postprocessor import LLMRerank
    #     llm = get_llm(name)
    #     logger.info(f"✅ Successfully created LLM fallback for {name}")
    #     return LLMRerank(top_n=top_k, llm=llm)
    # except Exception as e:
    #     logger.error(f"❌ 无法创建任何类型的Reranker: '{name}'. Error: {e}")
    #     return None