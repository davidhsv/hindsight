"""Candidate features and the two scorers: deterministic (logistic) and Jev (decision model).

Features come from what recall already computed (reranker, arms, rank, metadata, the adaptive
state it loaded); nothing here reads the database or calls a generative model. The Jev scorer
asks the decision model one Noul per candidate, all in one request per recall.
"""

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from ...config import DecisionModelConfig, get_config
from ..decision_model import DecisionModel, DecisionQuestion
from ..search.types import ScoredResult
from ..token_encoding import count_tokens as _count_tokens
from ..token_encoding import truncate_to_tokens
from .models import CandidateFeatures, Purpose, ScopeMatch

logger = logging.getLogger(__name__)

_RELEVANCE_TERMS = frozenset({"bias", "reranker", "semantic", "bm25", "arm_agreement", "rank"})
_SCOPE_PREFIXES = ("scope_match.", "scope_mismatch.")


def memory_metadata(value: object) -> Mapping[str, object]:
    """A candidate's metadata as a mapping. The SQL arms hand it over as stored, which may be the
    JSON text of the column rather than a dict; anything unreadable is no metadata."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalized(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().casefold().rstrip("/")
    return text or None


def _scope_values(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return {n for v in value if (n := _normalized(v)) is not None}
    normalized = _normalized(value)
    return {normalized} if normalized is not None else set()


def scope_matches(
    task_context: Mapping[str, str], metadata: Mapping[str, object] | None, keys: Sequence[str]
) -> dict[str, ScopeMatch]:
    """Per scope key: match / mismatch when the request and the memory both name a value, else
    unknown. Compared case-insensitively, ignoring surrounding whitespace and a trailing slash.

    A memory may name several values for a key (an observation, whose values are its source
    facts'): it matches when the request's value is among them and mismatches when none is it.
    """
    out: dict[str, ScopeMatch] = {}
    for key in keys:
        wanted = _normalized(task_context.get(key))
        stored = _scope_values((metadata or {}).get(key))
        if wanted is None or not stored:
            out[key] = ScopeMatch.UNKNOWN
        else:
            out[key] = ScopeMatch.MATCH if wanted in stored else ScopeMatch.MISMATCH
    return out


def observation_scope(
    own: Mapping[str, object], sources: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> dict[str, object]:
    """The scope values an observation is compared on, per key: its own metadata's value when it
    names one, else every distinct value its source facts name (none: the key stays unknown).

    Consolidation writes no metadata on an observation, so without this every observation reads
    unknown on every scope key and hard scope could never reject one consolidated from another
    repository's facts.
    """
    out: dict[str, object] = {}
    for key in keys:
        if _normalized(own.get(key)) is not None:
            out[key] = own[key]
            continue
        values = tuple(dict.fromkeys(v for md in sources if (v := _normalized(md.get(key))) is not None))
        if values:
            out[key] = values
    return out


def candidate_features(
    sr: ScoredResult,
    *,
    rank: int,
    task_context: Mapping[str, str],
    scope_keys: Sequence[str],
    count_tokens: Callable[[str], int],
    scope_metadata: Mapping[str, object] | None = None,
    rank_based_relevance: bool = False,
) -> CandidateFeatures:
    """``scope_metadata`` replaces the memory's own metadata for the scope comparison (an
    observation's, derived from its sources by :func:`observation_scope`)."""
    lifecycle = sr.candidate.lifecycle
    if scope_metadata is None:
        scope_metadata = memory_metadata(sr.retrieval.metadata)
    return CandidateFeatures(
        memory_id=sr.id,
        rank=rank,
        base_retrieval_score=sr.weight,
        reranker=sr.cross_encoder_score_normalized,
        semantic=sr.candidate.arm_scores.semantic,
        bm25=sr.candidate.arm_scores.keyword is not None,
        arms=len(sr.candidate.source_ranks),
        scope=scope_matches(task_context, scope_metadata, scope_keys),
        truth_status=lifecycle.truth_status if lifecycle is not None else "active",
        retrievability=sr.retrievability,
        importance=sr.importance,
        fact_type=sr.retrieval.fact_type,
        tokens=count_tokens(sr.retrieval.text or ""),
        rank_based_relevance=rank_based_relevance,
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class DeterministicScore:
    """The logistic score and its three parts, each the sigmoid of its own logit sum."""

    task_relevance: float
    scope_score: float
    expected_utility: float
    admission_score: float


def deterministic_score(vector: Mapping[str, float], weights: Mapping[str, float]) -> DeterministicScore:
    """``sigmoid(sum(weight * feature))`` over the features ``weights`` names (others weigh 0)."""
    relevance = scope = utility = 0.0
    for key, value in vector.items():
        contribution = weights.get(key, 0.0) * value
        if key in _RELEVANCE_TERMS:
            relevance += contribution
        elif key.startswith(_SCOPE_PREFIXES):
            scope += contribution
        else:
            utility += contribution
    return DeterministicScore(
        task_relevance=_sigmoid(relevance),
        scope_score=_sigmoid(scope),
        expected_utility=_sigmoid(utility),
        admission_score=_sigmoid(relevance + scope + utility),
    )


# --------------------------------------------------------------------------- the Jev scorer

JEV_QUESTION = "Would including this memory in the assistant's context help with the current task?"

# Every request carries the task once and each candidate's text; both are capped so thirty
# candidates stay well inside one request's token ceiling.
_MAX_TASK_TOKENS = 1_000
_MAX_MEMORY_TOKENS = 300
_METADATA_KEYS_SHOWN = 8


@lru_cache(maxsize=4)
def _client(setup: DecisionModelConfig) -> DecisionModel:
    # One client per setup for the process, so its connection pool and concurrency cap are shared.
    return DecisionModel.from_config(setup)


def jev_client() -> DecisionModel | None:
    """The decision model configured for context admission, or None when there is none."""
    setup = get_config().decision_model_config("context_admission")
    return _client(setup) if setup is not None else None


def _capped(text: str, ceiling: int) -> str:
    return truncate_to_tokens(text, ceiling).text if _count_tokens(text) > ceiling else text


def _memory_metadata(sr: ScoredResult) -> str:
    retrieval = sr.retrieval
    parts = [f"type={retrieval.fact_type}"]
    when = retrieval.occurred_start or retrieval.mentioned_at
    if when is not None:
        parts.append(f"date={when.date().isoformat()}")
    if retrieval.context:
        parts.append(f"context={retrieval.context[:200]}")
    for key, value in list(memory_metadata(retrieval.metadata).items())[:_METADATA_KEYS_SHOWN]:
        parts.append(f"{key}={str(value)[:120]}")
    if sr.candidate.lifecycle is not None:
        parts.append(f"status={sr.candidate.lifecycle.truth_status}")
    return "; ".join(parts)


@dataclass(frozen=True)
class JevRequest:
    """The one decision-model request of a recall: the shared state and a question per candidate."""

    state: dict[str, str]
    #: Keyed ``c<slot>``, in candidate order.
    questions: dict[str, DecisionQuestion]


def jev_request(
    candidates: Sequence[ScoredResult], *, query: str, purpose: Purpose, task_context: Mapping[str, str]
) -> JevRequest:
    """The request that asks the decision model about every candidate at once."""
    state = {"task": _capped(query, _MAX_TASK_TOKENS), "purpose": purpose.value}
    if task_context:
        state["task_context"] = "; ".join(f"{k}={v}" for k, v in task_context.items())
    questions = {
        f"c{slot}": DecisionQuestion(
            type="noul",
            instructions={
                "question": JEV_QUESTION,
                "memory": _capped(sr.retrieval.text or "", _MAX_MEMORY_TOKENS),
                "memory_metadata": _memory_metadata(sr),
            },
        )
        for slot, sr in enumerate(candidates)
    }
    return JevRequest(state=state, questions=questions)


async def jev_probabilities(
    client: DecisionModel,
    candidates: Sequence[ScoredResult],
    *,
    query: str,
    purpose: Purpose,
    task_context: Mapping[str, str],
) -> dict[str, float]:
    """One decision-model request for every candidate: memory id -> probability it helps.

    Raises what the request raises (after the client's own retries); a candidate the model left
    unanswered is absent from the result.
    """
    if not candidates:
        return {}
    request = jev_request(candidates, query=query, purpose=purpose, task_context=task_context)
    response = await client.ask(request.state, request.questions)
    out: dict[str, float] = {}
    for slot, sr in enumerate(candidates):
        answer = response.answers.get(f"c{slot}")
        if answer is not None and answer.noul is not None and math.isfinite(answer.noul):
            out[sr.id] = min(1.0, max(0.0, float(answer.noul)))
    return out
