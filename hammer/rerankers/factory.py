
import typing as T
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from hammer.llm import LLMs, get_llm
from hammer.logger import logger

def get_reranker(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """
    Compatibility reranker factory.
    Instantiate and return a reranker postprocessor for the requested name/top_k pair.
    """
    logger.info("Attempting to create reranker '%s' with top_k=%s", name, top_k)

    # Handle explicit specialized rerankers first.
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
    
    # Additional built-in reranker support.
    elif name in ["TransformerRanker", "transformer_ranker"]:
        try:
            from hammer.rerankers.integrations import BatchSentenceTransformerReranker
            logger.info("Creating batch TransformerRanker with top_k=%s", top_k)
            return BatchSentenceTransformerReranker(
                model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                top_n=top_k,
                batch_size=32,
            )
        except ImportError:
            logger.warning("⚠️ SentenceTransformers not available for TransformerRanker")
    
    # T5-family rerankers currently fall back to the parallel LLM path.
    elif name in ["MonoT5", "monot5", "RankT5", "rankt5", "ListT5", "listt5"]:
        logger.warning(f"⚠️ {name} not available in user's framework. Attempting LLM fallback.")
        return _try_llm_fallback(name, top_k)
    
    # Other framework rerankers also fall back to the parallel path for now.
    elif name in ["TWOLAR", "twolar", "MonoBERT", "monobert_ranker", "InRanker", "inranker", "EchoRank", "echorank"]:
        logger.warning(f"⚠️ {name} not available in user's framework. Attempting LLM fallback.")
        return _try_llm_fallback(name, top_k)

    # Known LLMs are handled by the optimized prompt-builder path.
    elif name in LLMs:
        logger.info(
            "Detected LLM reranker '%s'; skipping serial LLMRerank and deferring to parallel processing",
            name,
        )
        return None
        
        # try:
        #     from llama_index.core.postprocessor import LLMRerank
        #     logger.info(f"Detected LLM '{name}', creating LLMRerank.")
        #     llm = get_llm(name)
        #     return LLMRerank(top_n=top_k, llm=llm)
        # except Exception as e:
        #     logger.error(f"Failed to create LLMRerank for {name}: {e}")
            
    # Unknown names also fall back to the parallel path.
    else:
        logger.warning("Unknown reranker or LLM name '%s'; deferring to parallel processing", name)
        return None

def _try_llm_fallback(name: str, top_k: int) -> T.Optional[BaseNodePostprocessor]:
    """Fallback that routes reranking to the parallel LLM path."""
    logger.info("Using parallel reranking instead of serial LLMRerank for %s", name)
    return None
    
    # try:
    #     from llama_index.core.postprocessor import LLMRerank
    #     llm = get_llm(name)
    #     logger.info(f"Successfully created LLM fallback for {name}")
    #     return LLMRerank(top_n=top_k, llm=llm)
    # except Exception as e:
    #     logger.error(f"Failed to create any reranker for '{name}'. Error: {e}")
    #     return None
