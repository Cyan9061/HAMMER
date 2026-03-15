"""Compatibility helpers for legacy imports.

The original TPE tuner module is not present in this checkout, but several MCTS
and baseline paths still import ``FlowBuilder`` and ``prepare_worker`` from it.
This lightweight compatibility module provides the subset required to build and
evaluate RAG flows in the current repository.
"""

from __future__ import annotations

import os
import typing as T

from hammer.flows import Flow, RAGFlow
from hammer.llm import get_llm
from hammer.logger import logger
from hammer.retrievers.build import build_rag_retriever
from hammer.templates import get_template


def prepare_worker() -> None:
    """Apply a couple of safe worker defaults before evaluation."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class FlowBuilder:
    """Minimal flow builder compatible with the historical tuner interface."""

    def __init__(self, study_config: T.Any):
        self.study_config = study_config

    def _normalize_params(self, params: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        normalized = dict(params)
        normalized.setdefault("rag_mode", "rag")
        normalized.setdefault("template_name", "CoT")
        normalized.setdefault("response_synthesizer_llm", "Qwen2-7b")
        normalized.setdefault("splitter_method", "sentence")
        normalized.setdefault("splitter_chunk_exp", 8)
        normalized.setdefault("splitter_chunk_overlap_frac", 0.1)

        if "retrieval_method" in normalized and "rag_method" not in normalized:
            normalized["rag_method"] = normalized["retrieval_method"]
        if "retrieval_top_k" in normalized and "rag_top_k" not in normalized:
            normalized["rag_top_k"] = normalized["retrieval_top_k"]
        if "embedding_model" in normalized and "rag_embedding_model" not in normalized:
            normalized["rag_embedding_model"] = normalized["embedding_model"]
        if "splitter_overlap" in normalized and "splitter_chunk_overlap_frac" not in normalized:
            normalized["splitter_chunk_overlap_frac"] = normalized["splitter_overlap"]
        if (
            "query_decomposition_enabled" in normalized
            and "rag_query_decomposition_enabled" not in normalized
        ):
            normalized["rag_query_decomposition_enabled"] = normalized["query_decomposition_enabled"]
        if (
            "query_decomposition_num_queries" in normalized
            and "rag_query_decomposition_num_queries" not in normalized
        ):
            normalized["rag_query_decomposition_num_queries"] = normalized["query_decomposition_num_queries"]
        if (
            "query_decomposition_llm" in normalized
            and "rag_query_decomposition_llm_name" not in normalized
        ):
            normalized["rag_query_decomposition_llm_name"] = normalized["query_decomposition_llm"]
        if "fusion_mode" in normalized and "rag_fusion_mode" not in normalized:
            normalized["rag_fusion_mode"] = normalized["fusion_mode"]
        if "hybrid_bm25_weight" in normalized and "rag_hybrid_bm25_weight" not in normalized:
            normalized["rag_hybrid_bm25_weight"] = normalized["hybrid_bm25_weight"]

        normalized.setdefault("rag_method", "sparse")
        normalized.setdefault("rag_top_k", 9)
        normalized.setdefault("rag_embedding_model", "BAAI/bge-small-en-v1.5")
        normalized.setdefault("rag_query_decomposition_enabled", True)
        normalized.setdefault("rag_query_decomposition_num_queries", 4)
        normalized.setdefault("rag_query_decomposition_llm_name", "Qwen2-7b")
        normalized.setdefault("rag_fusion_mode", "simple")
        normalized.setdefault("hyde_enabled", False)
        normalized.setdefault("few_shot_enabled", False)
        normalized.setdefault("enforce_full_evaluation", False)
        normalized.setdefault("reranker_enabled", False)
        normalized.setdefault("additional_context_enabled", True)
        normalized.setdefault("additional_context_num_nodes", 5)

        if "reranker_llm_name" in normalized and "reranker_llm" not in normalized:
            normalized["reranker_llm"] = normalized["reranker_llm_name"]
        if "reranker_llm" in normalized and "reranker_llm_name" not in normalized:
            normalized["reranker_llm_name"] = normalized["reranker_llm"]

        if normalized["few_shot_enabled"]:
            logger.warning("few-shot flow building is not wired in this compatibility layer; disabling it.")
            normalized["few_shot_enabled"] = False

        return normalized

    def build_flow(self, params: T.Dict[str, T.Any]) -> Flow:
        normalized = self._normalize_params(params)
        response_llm = get_llm(str(normalized["response_synthesizer_llm"]))
        template = get_template(
            str(normalized["template_name"]),
            with_context=normalized["rag_mode"] == "rag",
            with_few_shot_prompt=False,
        )

        if normalized["rag_mode"] != "rag":
            return Flow(
                response_synthesizer_llm=response_llm,
                template=template,
                params=normalized,
                enforce_full_evaluation=normalized["enforce_full_evaluation"],
            )

        retriever, docstore = build_rag_retriever(self.study_config, normalized)
        hyde_llm = None
        if normalized.get("hyde_enabled") and normalized.get("hyde_llm_name"):
            hyde_llm = get_llm(str(normalized["hyde_llm_name"]))

        additional_context_num_nodes = (
            int(normalized.get("additional_context_num_nodes", 0))
            if normalized.get("additional_context_enabled")
            else 0
        )

        return RAGFlow(
            response_synthesizer_llm=response_llm,
            template=template,
            retriever=retriever,
            docstore=docstore,
            hyde_llm=hyde_llm,
            reranker_llm=None,
            reranker_top_k=normalized.get("reranker_top_k"),
            additional_context_num_nodes=additional_context_num_nodes,
            params=normalized,
            enforce_full_evaluation=normalized["enforce_full_evaluation"],
        )
