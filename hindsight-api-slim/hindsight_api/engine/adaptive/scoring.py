"""Adaptive memory in recall: the ranking factor, and the recall tail it is measured through.

Recall's combined-scoring branch calls into this module only when the effective mode loads state
(observe, rank, lifecycle); with the mode off none of it runs.

- :func:`attach_adaptive` records each candidate's retention values and factor on its
  ``ScoredResult``; :func:`apply_adaptive_factor` multiplies the factor into ``weight`` (rank and
  lifecycle only — observe reports and changes nothing).
- The recall tail that turns a sorted candidate list into the returned facts — the
  ``min_scores`` floors, the prefer-observations dedup, the ``budget × 2`` truncation and the
  token budget — is written here as small pure stages. The real tail in
  ``MemoryEngine._search_with_retries`` calls the same stages in place (it interleaves a source
  read and phase timings between them), and :func:`final_selection` composes them for the
  counterfactual: observe asks "what would rank mode have returned?", rank asks "what would
  the factor-free order have returned?". Sharing the stages is what keeps the counterfactual an
  exact replay rather than an approximation of the tail.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from ..fact_budget import select_facts_within_budget
from ..search.types import ScoredResult
from .models import AdaptiveState
from .retention import adaptive_factor, effective_retention, memory_floor, retention_factor
from .settings import AdaptiveSettings

_RAW_FACT_TYPES = ("world", "experience")


def attach_adaptive(
    scored_results: Sequence[ScoredResult],
    states: Mapping[str, AdaptiveState],
    settings: AdaptiveSettings,
    now: datetime,
) -> None:
    """Set each result's adaptive attributes from its state (or lazy defaults) at ``now``.

    ``adaptive_base_score`` is the weight as it stands when this runs — after combined scoring
    and the strategy boosts — so it is the factor-free score the trace compares against.
    ``weight`` itself is not touched.
    """
    alpha = settings.score_alpha
    for sr in scored_results:
        retrieval = sr.retrieval
        effective = effective_retention(
            states.get(sr.id),
            fact_type=retrieval.fact_type,
            mentioned_at=retrieval.mentioned_at,
            occurred_start=retrieval.occurred_start,
            event_date=retrieval.event_date,
            settings=settings,
            now=now,
        )
        floor_i = memory_floor(settings.score_floor, effective.importance)
        sr.adaptive_base_score = sr.weight
        sr.retrievability = effective.retrievability
        sr.stability_days = effective.stability_days
        sr.importance = effective.importance
        sr.reinforcement_count = effective.reinforcement_count
        sr.retention_factor = retention_factor(effective.retrievability, floor_i)
        sr.adaptive_factor = adaptive_factor(effective.retrievability, alpha, floor_i)


def apply_adaptive_factor(scored_results: Sequence[ScoredResult]) -> None:
    """``weight *= adaptive_factor`` for every result :func:`attach_adaptive` scored. No sort."""
    for sr in scored_results:
        if sr.adaptive_factor is not None:
            sr.weight *= sr.adaptive_factor


# --------------------------------------------------------------------------- the recall tail


def filter_min_scores(
    scored_results: Sequence[ScoredResult], *, min_reranker: float | None, min_final: float | None
) -> list[ScoredResult]:
    """The post-rank ``min_scores`` floors: inclusive (``>=``), AND-ed, ``None`` is no floor."""
    return [
        sr
        for sr in scored_results
        if (min_reranker is None or sr.cross_encoder_score_normalized >= min_reranker)
        and (min_final is None or sr.weight >= min_final)
    ]


def observations_in_window(scored_results: Sequence[ScoredResult], window: int) -> list[ScoredResult]:
    """The observations among the top ``window`` results: the only ones that may supersede a
    raw fact (a far-down observation must not suppress a top raw fact it happens to cite)."""
    return [sr for sr in scored_results[:window] if sr.retrieval.fact_type == "observation"]


def superseded_source_ids(observations: Sequence[ScoredResult]) -> set[str]:
    """The source memory ids the given observations carry (``None`` sources count as none)."""
    return {str(sid) for sr in observations for sid in (sr.retrieval.source_memory_ids or [])}


def drop_superseded_raw_facts(scored_results: Sequence[ScoredResult], superseded_ids: set[str]) -> list[ScoredResult]:
    """Drop the raw facts an observation in the window was consolidated from."""
    return [sr for sr in scored_results if not (sr.retrieval.fact_type in _RAW_FACT_TYPES and sr.id in superseded_ids)]


@dataclass(frozen=True)
class FinalSelection:
    """The ids the recall tail returns, in rank order."""

    ids: list[str]
    total_tokens: int
    truncated: bool


def final_selection(
    scored_results: Sequence[ScoredResult],
    *,
    min_reranker: float | None,
    min_final: float | None,
    prefer_observations: bool,
    rerank_limit: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> FinalSelection:
    """Replay the recall tail over an already-sorted list (no I/O).

    ``prefer_observations`` is the resolved dedup switch (the caller asked for it AND requested
    observations alongside a raw type). Observation sources are read off
    ``retrieval.source_memory_ids``; the caller makes sure the ones in this list's window were
    resolved, as the real tail resolves its own.
    """
    selected = (
        filter_min_scores(scored_results, min_reranker=min_reranker, min_final=min_final)
        if min_reranker is not None or min_final is not None
        else list(scored_results)
    )
    if prefer_observations:
        superseded = superseded_source_ids(observations_in_window(selected, rerank_limit))
        if superseded:
            selected = drop_superseded_raw_facts(selected, superseded)
    top = selected[:rerank_limit]
    budget = select_facts_within_budget(
        fact_ids_ordered=[sr.id for sr in top],
        text_by_id={sr.id: sr.retrieval.text for sr in top},
        max_tokens=max_tokens,
        count_tokens=count_tokens,
    )
    kept = set(budget.ids)
    return FinalSelection(
        ids=[sr.id for sr in top if sr.id in kept],
        total_tokens=budget.total_tokens,
        truncated=budget.truncated,
    )


def reweighted(
    scored_results: Sequence[ScoredResult], weight_of: Callable[[ScoredResult], float]
) -> list[ScoredResult]:
    """Shallow copies with a substitute weight, sorted the way recall sorts (stable, descending).

    The copies share their candidate (and so their retrieval) with the originals, which is what
    lets a source list resolved on one be read from the other.
    """
    copies = [replace(sr, weight=weight_of(sr)) for sr in scored_results]
    copies.sort(key=lambda sr: sr.weight, reverse=True)
    return copies


class CachedTokenCounter:
    """``count_tokens`` memoised for one recall, so replaying the tail does not re-tokenise."""

    def __init__(self, count_tokens: Callable[[str], int]) -> None:
        self._count_tokens = count_tokens
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is None:
            cached = self._cache[text] = self._count_tokens(text)
        return cached


@dataclass(frozen=True)
class AdaptiveStateLoad:
    """The result of loading a recall's adaptive state: the rows, or why they could not be read."""

    states: Mapping[str, AdaptiveState] | None
    error: Exception | None
    seconds: float


@dataclass(frozen=True)
class CounterfactualOutcome:
    """The counterfactual tail's ids and whether they differ from what recall returned."""

    ids: list[str]
    changed: bool


@dataclass
class AdaptiveRecall:
    """One recall's adaptive-memory work, created only when the effective mode loads state.

    Holds what the engine threads through the recall pipeline: the loaded states (or the load
    error), the counterfactual list built at scoring time, and timings for the diagnostic phases.
    """

    settings: AdaptiveSettings
    now: datetime
    count_tokens: CachedTokenCounter
    states: Mapping[str, AdaptiveState] | None = None
    load_error: Exception | None = None
    load_seconds: float = 0.0
    scoring_seconds: float = 0.0
    counterfactual: list[ScoredResult] | None = field(default=None)

    def receive(self, load: AdaptiveStateLoad) -> None:
        self.states = load.states
        self.load_error = load.error
        self.load_seconds = load.seconds

    def score(self, scored_results: list[ScoredResult]) -> bool:
        """Attach the factor (and apply it in rank/lifecycle); build the counterfactual list.

        Must run before the caller's sort, so the counterfactual is sorted from the same
        pre-sort order the real list is — ties then break identically. Returns False (and does
        nothing) when the state could not be loaded: the recall continues with factor 1.
        """
        if self.states is None:
            return False
        attach_adaptive(scored_results, self.states, self.settings, self.now)
        if self.settings.effective_mode.applies_factor:
            self.counterfactual = reweighted(scored_results, lambda sr: sr.weight)
            apply_adaptive_factor(scored_results)
        else:
            self.counterfactual = reweighted(
                scored_results,
                lambda sr: sr.weight * (sr.adaptive_factor if sr.adaptive_factor is not None else 1.0),
            )
        return True

    def counterfactual_window_observations(
        self, *, min_reranker: float | None, min_final: float | None, window: int
    ) -> list[ScoredResult]:
        """The observations the counterfactual tail's dedup will read sources for."""
        if self.counterfactual is None:
            return []
        filtered = (
            filter_min_scores(self.counterfactual, min_reranker=min_reranker, min_final=min_final)
            if min_reranker is not None or min_final is not None
            else self.counterfactual
        )
        return observations_in_window(filtered, window)

    def finish(
        self,
        final_ids: list[str],
        *,
        min_reranker: float | None,
        min_final: float | None,
        prefer_observations: bool,
        rerank_limit: int,
        max_tokens: int,
    ) -> CounterfactualOutcome | None:
        """Replay the tail over the counterfactual; None when scoring did not run."""
        if self.counterfactual is None:
            return None
        selection = final_selection(
            self.counterfactual,
            min_reranker=min_reranker,
            min_final=min_final,
            prefer_observations=prefer_observations,
            rerank_limit=rerank_limit,
            max_tokens=max_tokens,
            count_tokens=self.count_tokens,
        )
        return CounterfactualOutcome(ids=selection.ids, changed=selection.ids != final_ids)
