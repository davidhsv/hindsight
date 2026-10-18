"""Context admission: which recalled memories enter an agent's context, and when none should.

Retrieval, reranking and the adaptive-memory lifecycle decide what is a candidate and what is
believed. This stage decides what is worth injecting for the task at hand, a separate question:
semantic similarity is not relevance, a stable important memory can be useless here, and the right
answer can be no memory at all. It runs on a recall's top-ranked candidates only, after the
``min_scores`` floors and before the prefer-observations dedup, and only for external recalls in a
bank whose ``context_admission_mode`` is not ``off`` (with it off nothing here runs).
"""

from .models import (
    PURPOSES,
    REJECTION_REASONS,
    TASK_CONTEXT_KEYS,
    AdmissionMode,
    CandidateFeatures,
    ContextAdmissionDecision,
    ContextAdmissionOutcome,
    Purpose,
    Reason,
    ScopeMatch,
    ScorerKind,
)
from .pipeline import ContextAdmissionRecall, ObserveOutcome, clean_task_context
from .scoring import JEV_QUESTION, jev_client, memory_metadata
from .settings import ContextAdmissionSettings, mode_of
from .signals import detect_reference_signal

__all__ = [
    "JEV_QUESTION",
    "PURPOSES",
    "REJECTION_REASONS",
    "TASK_CONTEXT_KEYS",
    "AdmissionMode",
    "CandidateFeatures",
    "ContextAdmissionDecision",
    "ContextAdmissionOutcome",
    "ContextAdmissionRecall",
    "ContextAdmissionSettings",
    "ObserveOutcome",
    "Purpose",
    "Reason",
    "ScopeMatch",
    "ScorerKind",
    "clean_task_context",
    "detect_reference_signal",
    "jev_client",
    "memory_metadata",
    "mode_of",
]
