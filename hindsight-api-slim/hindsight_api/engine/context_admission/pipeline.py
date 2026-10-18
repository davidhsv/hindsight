"""One recall's context-admission work: score the top candidates, gate them, report.

Created by recall only for an external recall in a bank whose mode is not off. Recall calls
:meth:`ContextAdmissionRecall.decide` after the ``min_scores`` floors and before the
prefer-observations dedup, then :meth:`apply`, and after its token budget :meth:`finish`.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from ..adaptive.scoring import CachedTokenCounter, final_selection
from ..decision_model import DecisionModel
from ..search.types import ScoredResult
from .models import (
    AdmissionMode,
    ContextAdmissionDecision,
    ContextAdmissionOutcome,
    Purpose,
    Reason,
    ScorerKind,
)
from .scoring import candidate_features, deterministic_score, jev_probabilities, memory_metadata, observation_scope
from .selection import GateRules, apply_budget, gate, readmit_twins, suppress_redundant
from .settings import ContextAdmissionSettings

logger = logging.getLogger(__name__)

LoadEmbeddings = Callable[[list[str]], Awaitable[Mapping[str, Sequence[float]]]]
#: Observation candidates -> each one's source facts' metadata, keyed by observation id.
LoadSourceMetadata = Callable[[list[ScoredResult]], Awaitable[Mapping[str, Sequence[Mapping[str, object]]]]]

#: Longest a task-context value may be, and how many keys a request may carry; the rest is dropped.
MAX_TASK_CONTEXT_VALUE_CHARS = 256
MAX_TASK_CONTEXT_KEYS = 32


def clean_task_context(task_context: Mapping[str, object] | None) -> dict[str, str]:
    """The usable part of a request's task context: non-empty string keys and values, trimmed and
    bounded. Anything else is dropped rather than refused: the context only ever adds evidence."""
    out: dict[str, str] = {}
    for key, value in (task_context or {}).items():
        if len(out) >= MAX_TASK_CONTEXT_KEYS:
            break
        if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip():
            out[key.strip()] = value.strip()[:MAX_TASK_CONTEXT_VALUE_CHARS]
    return out


@dataclass(frozen=True)
class ObservationScopes:
    """Observation candidates' scope values (by memory id), or why they could not be read."""

    by_id: Mapping[str, Mapping[str, object]]
    error: str | None = None


@dataclass(frozen=True)
class ObserveOutcome:
    """Observe mode: what the recall would have returned with admission applied."""

    shadow_ids: list[str]
    would_drop: int
    would_abstain: bool
    would_change_top_k: bool


@dataclass
class ContextAdmissionRecall:
    settings: ContextAdmissionSettings
    purpose: Purpose
    task_context: Mapping[str, str]
    query: str
    reference_signal: str | None
    count_tokens: CachedTokenCounter
    outcome: ContextAdmissionOutcome | None = None
    #: Observe mode: the admitted candidates the tail is replayed over in :meth:`finish`.
    shadow: list[ScoredResult] | None = None
    historical: bool = False

    @property
    def mode(self) -> AdmissionMode:
        return self.settings.mode

    @property
    def threshold(self) -> float:
        return self.settings.threshold(self.purpose, reference_signal=self.reference_signal is not None)

    async def decide(
        self,
        scored_results: Sequence[ScoredResult],
        *,
        historical: bool,
        load_embeddings: LoadEmbeddings,
        jev: DecisionModel | None,
        load_source_metadata: LoadSourceMetadata | None = None,
        rank_based_relevance: bool = False,
    ) -> ContextAdmissionOutcome:
        """Judge the top ``candidates`` of ``scored_results`` (recall order) and keep the outcome.

        The decision model is asked only when the scorer needs it, in one request; when it is not
        configured or the request fails, every candidate falls back to the deterministic score and
        ``jev_error`` says why. A failed embedding read skips redundancy suppression (nothing is
        dropped as a duplicate) and sets ``embeddings_error``.

        An observation is compared on the scope of its source facts (``load_source_metadata``,
        one read, only when the request names a scope key and an observation is among the
        candidates); a failed read leaves observations unknown and sets ``scope_error``.
        ``rank_based_relevance``: the recall's reranker scored by rank, so the deterministic
        scorer reads the semantic similarity in its place (``relevance_source``).
        """
        settings = self.settings
        self.historical = historical
        candidates = list(scored_results[: settings.candidates])
        scope_keys = settings.compared_scope_keys
        observed_scopes = await self._observation_scopes(candidates, scope_keys, load_source_metadata)
        features = [
            candidate_features(
                sr,
                rank=rank,
                task_context=self.task_context,
                scope_keys=scope_keys,
                count_tokens=self.count_tokens,
                scope_metadata=observed_scopes.by_id.get(sr.id),
                rank_based_relevance=rank_based_relevance,
            )
            for rank, sr in enumerate(candidates)
        ]

        jev_error: str | None = None
        probabilities: Mapping[str, float] = {}
        if settings.scorer is not ScorerKind.DETERMINISTIC and candidates:
            if jev is None:
                jev_error = "not_configured"
            else:
                try:
                    probabilities = await jev_probabilities(
                        jev, candidates, query=self.query, purpose=self.purpose, task_context=self.task_context
                    )
                except Exception as e:
                    logger.warning(f"[CONTEXT ADMISSION] decision model request failed ({type(e).__name__}: {e})")
                    jev_error = type(e).__name__

        decisions: list[ContextAdmissionDecision] = []
        for feature in features:
            vector = feature.vector()
            det = deterministic_score(vector, settings.weights)
            relevance, score = det.task_relevance, det.admission_score
            probability = probabilities.get(feature.memory_id)
            if probability is not None:
                relevance = probability
                score = (
                    probability
                    if settings.scorer is ScorerKind.JEV
                    else settings.jev_weight * probability + (1.0 - settings.jev_weight) * det.admission_score
                )
            decisions.append(
                ContextAdmissionDecision(
                    memory_id=feature.memory_id,
                    base_retrieval_score=feature.base_retrieval_score,
                    scope_score=det.scope_score,
                    task_relevance=relevance,
                    expected_utility=det.expected_utility,
                    truth_status=feature.truth_status,
                    retrievability=feature.retrievability,
                    admission_score=score,
                    admitted=False,
                    reason=Reason.BELOW_THRESHOLD,
                    scope=feature.scope,
                    features=vector,
                    tokens=feature.tokens,
                    uncertain=feature.truth_status == "uncertain",
                )
            )

        full = self.mode in (AdmissionMode.STRICT, AdmissionMode.OBSERVE)
        rules = GateRules(
            threshold=self.threshold,
            historical=historical,
            uncertain_margin=settings.uncertain_margin,
            full=full,
            hard_scope_keys=settings.hard_scope_keys,
        )
        for decision in decisions:
            gate(decision, rules)

        embeddings_error: str | None = None
        if full:
            admitted_ids = [d.memory_id for d in decisions if d.admitted]
            embeddings: Mapping[str, Sequence[float]] = {}
            if len(admitted_ids) > 1:
                try:
                    embeddings = await load_embeddings(admitted_ids)
                except Exception as e:
                    logger.warning(f"[CONTEXT ADMISSION] embedding read failed ({type(e).__name__}: {e})")
                    embeddings_error = type(e).__name__
            redundancy = suppress_redundant(
                decisions, embeddings, mmr_lambda=settings.mmr_lambda, dup_similarity=settings.dup_similarity
            )
            apply_budget(
                decisions, redundancy.values, max_memories=settings.max_memories, max_tokens=settings.max_tokens
            )
            readmit_twins(decisions, redundancy, max_memories=settings.max_memories, max_tokens=settings.max_tokens)

        self.outcome = ContextAdmissionOutcome(
            admitted_ids=[d.memory_id for d in decisions if d.admitted],
            decisions=decisions,
            threshold=rules.threshold,
            reference_signal=self.reference_signal,
            scorer=settings.scorer,
            jev_error=jev_error,
            embeddings_error=embeddings_error,
            relevance_source="semantic" if rank_based_relevance else "reranker",
            scope_error=observed_scopes.error,
        )
        return self.outcome

    async def _observation_scopes(
        self,
        candidates: Sequence[ScoredResult],
        scope_keys: tuple[str, ...],
        load_source_metadata: LoadSourceMetadata | None,
    ) -> "ObservationScopes":
        """Each observation candidate's scope values, derived from its source facts."""
        observations = [sr for sr in candidates if sr.retrieval.fact_type == "observation"]
        if load_source_metadata is None or not observations or not any(k in self.task_context for k in scope_keys):
            return ObservationScopes(by_id={})
        try:
            sources = await load_source_metadata(observations)
        except Exception as e:
            logger.warning(f"[CONTEXT ADMISSION] observation source read failed ({type(e).__name__}: {e})")
            return ObservationScopes(by_id={}, error=type(e).__name__)
        return ObservationScopes(
            by_id={
                sr.id: observation_scope(memory_metadata(sr.retrieval.metadata), sources.get(sr.id, ()), scope_keys)
                for sr in observations
            }
        )

    def reset(self) -> None:
        """Forget a previous attempt's outcome (recall retries the whole search on a lost connection)."""
        self.outcome = None
        self.shadow = None
        self.historical = False

    def apply(self, scored_results: list[ScoredResult]) -> list[ScoredResult]:
        """The list the recall tail continues with.

        Filter and strict: the admitted candidates, in recall order; nothing past the judged top
        ``candidates`` survives, since an unjudged candidate must not backfill a rejected one.
        Observe: ``scored_results`` unchanged; the admitted ones are kept as the shadow list.
        """
        if self.outcome is None:
            return scored_results
        admitted = set(self.outcome.admitted_ids)
        selected = [sr for sr in scored_results[: self.settings.candidates] if sr.id in admitted]
        if self.mode.changes_results:
            return selected
        self.shadow = selected
        return scored_results

    def finish(
        self, final_ids: list[str], *, prefer_observations: bool, rerank_limit: int, max_tokens: int
    ) -> ObserveOutcome | None:
        """Observe mode: replay the recall tail over the shadow list (the ``min_scores`` floors
        already ran before admission) and compare it with what the recall returned."""
        if self.shadow is None:
            return None
        selection = final_selection(
            self.shadow,
            min_reranker=None,
            min_final=None,
            prefer_observations=prefer_observations,
            rerank_limit=rerank_limit,
            max_tokens=max_tokens,
            count_tokens=self.count_tokens,
        )
        shadow_ids = selection.ids
        kept = set(shadow_ids)
        return ObserveOutcome(
            shadow_ids=shadow_ids,
            would_drop=sum(1 for i in final_ids if i not in kept),
            would_abstain=not shadow_ids and bool(final_ids),
            would_change_top_k=shadow_ids != final_ids,
        )
