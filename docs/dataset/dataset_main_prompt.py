"""Compatibility helpers for dataset prompts and file resolution.

This repository currently stores dataset files under ``docs/unified_1`` and, when
available, ``docs/unified_query_selection``. Older code still imports
``docs.dataset.dataset_main_prompt`` and expects a couple of helper functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

_DATASET_ALIASES: Dict[str, str] = {
    "2wikimultihopqa": "2wikimultihopqa",
    "hotpotqa": "hotpotqa",
    "medqa": "MedQA",
    "MedQA": "MedQA",
    "fiqa": "fiqa",
    "quartz": "quartz",
    "webquestions": "webquestions",
    "eli5": "eli5",
    "popqa": "popqa",
}

_DATASET_PROMPTS: Dict[str, str] = {
    "2wikimultihopqa": (
        "2WikiMultiHopQA focuses on multi-hop reasoning across multiple entities. "
        "Prefer configurations that keep retrieval recall high and preserve evidence "
        "chains across several supporting documents."
    ),
    "hotpotqa": (
        "HotpotQA requires multi-document reasoning and answer grounding. "
        "Prefer configurations that retrieve complementary evidence and synthesize "
        "it into a concise answer."
    ),
    "MedQA": (
        "MedQA is a medical QA benchmark. Prefer configurations that preserve "
        "terminology accuracy, avoid hallucinations, and keep retrieval focused "
        "on high-precision medical evidence."
    ),
    "fiqa": (
        "FiQA contains finance-domain questions. Prefer configurations that retrieve "
        "domain-specific evidence and answer with precise financial terminology."
    ),
    "quartz": (
        "QuaRTz emphasizes science question answering with structured reasoning. "
        "Prefer configurations that keep context compact and reasoning explicit."
    ),
    "webquestions": (
        "WebQuestions emphasizes short factual answers grounded in web-retrieved "
        "evidence. Prefer high-precision retrieval and concise generation."
    ),
    "eli5": (
        "ELI5 benefits from broader context and coherent long-form generation. "
        "Prefer configurations that retrieve diverse supporting evidence and "
        "synthesize it into a readable explanation."
    ),
    "popqa": (
        "PopQA focuses on short factual knowledge questions. Prefer precise "
        "retrieval, low-noise contexts, and concise direct answers."
    ),
}


def validate_dataset_name(dataset_name: str) -> str:
    """Return the canonical dataset name or raise for unsupported values."""
    canonical = _DATASET_ALIASES.get(dataset_name)
    if canonical is None:
        supported = ", ".join(sorted(_DATASET_ALIASES))
        raise KeyError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return canonical


def get_available_datasets() -> list[str]:
    """Return user-facing dataset choices accepted by the CLI."""
    return [
        "2wikimultihopqa",
        "hotpotqa",
        "MedQA",
        "medqa",
        "fiqa",
        "quartz",
        "webquestions",
        "eli5",
        "popqa",
    ]


def get_dataset_prompt(dataset_name: str) -> str:
    canonical = validate_dataset_name(dataset_name)
    return _DATASET_PROMPTS.get(
        canonical,
        (
            f"{canonical} requires accurate retrieval-augmented question answering. "
            "Prefer configurations that maximize evidence quality and answer faithfulness."
        ),
    )


def resolve_dataset_files(dataset_name: str) -> Tuple[str, str, str]:
    """Resolve dataset files across current and legacy directory layouts."""
    canonical = validate_dataset_name(dataset_name)
    candidate_roots = [
        REPO_ROOT / "docs" / "unified_query_selection",
        REPO_ROOT / "docs" / "unified_1",
    ]
    qa_suffixes = ["_qa_unified.json", "_qa_final.json"]
    corpus_suffixes = ["_corpus_unified.json"]

    for root in candidate_roots:
        for corpus_suffix in corpus_suffixes:
            corpus_path = root / f"{canonical}{corpus_suffix}"
            if not corpus_path.exists():
                continue
            for qa_suffix in qa_suffixes:
                qa_path = root / f"{canonical}{qa_suffix}"
                if qa_path.exists():
                    return canonical, str(corpus_path), str(qa_path)

    searched = []
    for root in candidate_roots:
        searched.extend(str(root / f"{canonical}{suffix}") for suffix in corpus_suffixes + qa_suffixes)
    raise FileNotFoundError(
        f"Could not resolve dataset files for '{dataset_name}'. Looked for: {searched}"
    )
