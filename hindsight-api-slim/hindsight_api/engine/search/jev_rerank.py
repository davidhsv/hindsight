"""The decision model reorders the reranker's top results, told when each memory is from.

A recall check (``recall_jev_rerank``, per bank, off by default; ``recall_checks.py`` resolves it
with the other checks, and it runs on the same recalls they do). The cross-encoder reranks and
the recency, temporal and proof boosts score as always; then the top :data:`POOL_SIZE` results go
to the recall-checks decision model as one listwise Choice, the TypeSafe reranker's own rank
question (``TypeSafeCrossEncoder.rank``) with two additions: the state says the date the question
is asked (``Current date: YYYY-MM-DD``, recall's clock, so ``query_timestamp`` when the request
has one) and each option starts with its memory's date (``[Memory date: YYYY-MM-DD]``: when it
was said, else when it happened). The answer's order is final for those results, as a decision
model reranking recall's whole pool is: no boost re-sorts them. Everything below them keeps the
local order, after them.

Why the dates: on its own, Jev ranks what answers a question better than the local cross-encoder
but cannot tell a current value from an outdated one. On the recorded LoCoMo and LongMemEval
pools (jev-1.14.0) this order put a gold memory in the top five for 0.961 of LoCoMo questions
against 0.902 for the local order (114 gained, 23 lost, exact McNemar p = 1e-15), and all of a
temporal-reasoning question's evidence in the top five for 0.827 against 0.701 (18 gained, 2
lost). On knowledge-update questions the latest value came first for 0.764 against 0.819 for the
local order: the dates win back most of what Jev without them loses there (0.597), not all of it.

A request that fails, times out or answers out of shape leaves the local order: the recall never
fails for it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..cross_encoder import TypeSafeCrossEncoder
from .reranking import rerank_document
from .types import RetrievalResult, ScoredResult

logger = logging.getLogger(__name__)

# The local top 100, as measured: every such pool fit one Choice, with room to spare.
POOL_SIZE = 100
_OPTION_PREFIX = "[Memory date: {date}] "


def _utc_date(moment: datetime) -> str:
    """``YYYY-MM-DD`` of ``moment`` in UTC; a naive one is read as UTC, as recall's clock is."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date().isoformat()


def memory_date(retrieval: RetrievalResult) -> str | None:
    """The memory's date as the options carry it: when it was said, else when it happened."""
    moment = retrieval.mentioned_at or retrieval.occurred_start
    return _utc_date(moment) if moment is not None else None


def jev_option(retrieval: RetrievalResult) -> str:
    """One option of the Choice: the memory's date, then the text the cross-encoder scored."""
    date = memory_date(retrieval)
    document = rerank_document(retrieval)
    return f"{_OPTION_PREFIX.format(date=date)}{document}" if date else document


@dataclass(frozen=True)
class JevRerank:
    """One recall's results after the rerank, and what the decision model did to them."""

    results: list[ScoredResult]
    #: The results it ordered, best first; empty when it ordered none.
    ranked_ids: list[str] = field(default_factory=list)
    #: Why it ordered none although there was something to order, e.g. ``"TimeoutError"``.
    error: str | None = None


async def jev_rerank(
    reranker: TypeSafeCrossEncoder, query: str, scored_results: list[ScoredResult], now: datetime
) -> JevRerank:
    """``scored_results`` (sorted, best first) with the top :data:`POOL_SIZE` in the decision
    model's order, or unchanged when it does not answer within the reranker's timeout.

    The reordered results keep their final scores as a set, handed out by the new rank: the
    result ranked first gets the highest final score of the pool, and so on. The final score then
    still sorts the whole list (the ones below the pool score lower than all of it), and every
    result keeps its own reranker score and score components.
    """
    pool = scored_results[:POOL_SIZE]
    if len(pool) < 2:
        # Nothing to order, and Jev rejects a Choice with fewer than two options.
        return JevRerank(results=scored_results)
    options = [jev_option(sr.retrieval) for sr in pool]
    try:
        async with asyncio.timeout(reranker.timeout):
            order = await reranker.rank(query, options, current_date=_utc_date(now))
        if sorted(order) != list(range(len(pool))):
            raise ValueError(f"the answer ranks {len(order)} positions for {len(pool)} options")
    except Exception as e:
        logger.warning(f"Recall Jev rerank: the decision model did not answer, the local order stays: {e!r}")
        return JevRerank(results=scored_results, error=type(e).__name__)
    ranked = [pool[position] for position in order]
    for sr, weight in zip(ranked, sorted((sr.weight for sr in pool), reverse=True)):
        sr.weight = weight
    return JevRerank(results=ranked + scored_results[len(pool) :], ranked_ids=[sr.id for sr in ranked])


def ranked_first(results: list[ScoredResult], ranked_ids: list[str]) -> list[ScoredResult]:
    """``results`` with the ones in ``ranked_ids`` first, in that order, and the rest after them
    in the order they had.

    Applies one recall's Jev order to another list of the same candidates (adaptive memory's
    counterfactual), without another request.
    """
    position = {memory_id: rank for rank, memory_id in enumerate(ranked_ids)}
    head = sorted((sr for sr in results if sr.id in position), key=lambda sr: position[sr.id])
    return head + [sr for sr in results if sr.id not in position]
