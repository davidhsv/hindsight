"""Claim-lifecycle views in recall (adaptive memory ``lifecycle`` mode).

A lifecycle-mode recall reads the claim of every merged candidate it may hand to the reranker,
BEFORE the reranker trim, and removes the ones its view does not show. Removing them there (not
after scoring) means the cross-encoder or decision model never scores a retired claim and the
trim backfills from the fusion tail, so a recall in a bank full of superseded history still fills
its candidate budget. It is a removal, so it applies on every ranking branch, the decision-model
and interleave orders included.

Views:

- ``current``: superseded and refuted claims are removed.
- ``historical``: refuted claims and merge-retired ones (a rewording folded into another
  observation, not an earlier state) are removed; superseded claims stay, annotated.
- ``as_of`` (an instant T): a claim stays iff ``valid_from`` is unset or ``<= T`` and ``valid_to`` is
  unset or ``> T``; refuted and merge-retired claims never do. An unset ``valid_from`` means "since
  before anything recorded" — it is never defaulted from ``mentioned_at``.
- ``all-with-status`` (internal; consolidation's pool recall): nothing is removed, every candidate
  is annotated, so the consolidation model sees retired beliefs labelled.

A memory with no claim row is ``active`` with an open interval, so it passes every view.

Everything here is pure except :func:`filter_candidates_by_claims`, which takes the claim loader as
an argument.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ..adaptive.models import ClaimState, TruthStatus
from ..response_models import RecallLifecycle
from .recall_boost import boosted_rrf_score
from .types import MergedCandidate

LifecycleViewName = Literal["current", "historical", "as_of", "all-with-status"]

#: The internal view consolidation's pool recall uses. Never accepted from HTTP or MCP.
ALL_WITH_STATUS: Literal["all-with-status"] = "all-with-status"
#: The views ``recall_async(lifecycle_view=...)`` accepts by name. ``as_of`` is chosen by passing an
#: instant instead.
_NAMED: dict[str, LifecycleViewName] = {
    "current": "current",
    "historical": "historical",
    ALL_WITH_STATUS: ALL_WITH_STATUS,
}
NAMED_VIEWS: tuple[str, ...] = tuple(_NAMED)

#: How many candidates past the trim size the first claim read covers, so a few removals do not
#: need a second read.
CLAIM_LOAD_SLACK = 64
#: Further reads after the first when removals leave the trim short. After the last, the unread tail
#: is dropped rather than passed on unchecked (fail closed).
MAX_REFILL_ROUNDS = 3

_MERGES = "merges"


class ClaimStateUnavailable(RuntimeError):
    """A lifecycle-mode recall could not read its candidates' claims, so it fails closed.

    Showing the candidates unchecked would return refuted memories as current, so the recall
    raises instead; the HTTP API answers 503 with ``Retry-After`` (the read is retried by the
    caller, like any transient store failure). Propagated unwrapped through the recall pipeline.
    """

    #: Seconds a caller should wait before retrying.
    retry_after: int = 5


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None or value.utcoffset() is None else value


@dataclass(frozen=True)
class LifecycleView:
    """The claim view one recall applies. ``as_of`` is set exactly when ``name == "as_of"``."""

    name: LifecycleViewName
    as_of: datetime | None = None
    #: The historical-cue pattern that chose this view, when one did.
    cue: str | None = None

    def keeps(self, claim: ClaimState | None) -> bool:
        """Whether a candidate with this claim (None: no row, plain active) is shown."""
        if claim is None or self.name == ALL_WITH_STATUS:
            return True
        status = claim.truth_status
        if self.name == "current":
            return status not in (TruthStatus.SUPERSEDED, TruthStatus.REFUTED)
        if status is TruthStatus.REFUTED or claim.retired_via == _MERGES:
            return False
        if self.name == "historical":
            return True
        instant = self.as_of
        if instant is None:  # unreachable by construction (resolve_view), kept fail-closed
            return False
        if claim.valid_from is not None and _utc(claim.valid_from) > instant:
            return False
        return claim.valid_to is None or _utc(claim.valid_to) > instant


def resolve_view(
    *,
    requested: str | None,
    as_of: datetime | None,
    default_view: str,
    cue_name: str | None = None,
    cue_as_of: datetime | None = None,
) -> LifecycleView:
    """The view a lifecycle-mode recall applies.

    An explicit ``as_of`` wins over a named view (it is the more specific request), a named view
    over a cue, and a cue over the bank's ``default_view``. ``cue_name`` is only passed when the
    caller set neither ``requested`` nor ``as_of``.
    """
    if as_of is not None:
        return LifecycleView(name="as_of", as_of=_utc(as_of))
    if requested is not None:
        name = _NAMED.get(requested)
        if name is None:
            raise ValueError(f"Unknown lifecycle_view {requested!r}; expected one of {', '.join(NAMED_VIEWS)}")
        return LifecycleView(name=name)
    if cue_name is not None:
        if cue_as_of is not None:
            return LifecycleView(name="as_of", as_of=_utc(cue_as_of), cue=cue_name)
        return LifecycleView(name="historical", cue=cue_name)
    return LifecycleView(name="historical" if default_view == "historical" else "current")


def lifecycle_annotation(claim: ClaimState | None) -> RecallLifecycle | None:
    """The result annotation for a claim that is not plain active; None for active or no row."""
    if claim is None or claim.truth_status is TruthStatus.ACTIVE:
        return None
    return RecallLifecycle(
        truth_status=claim.truth_status.value,
        confidence=claim.confidence,
        valid_from=_utc(claim.valid_from).isoformat() if claim.valid_from is not None else None,
        valid_to=_utc(claim.valid_to).isoformat() if claim.valid_to is not None else None,
        superseded_by=claim.superseded_by_id,
    )


def in_trim_order(
    candidates: list[MergedCandidate], max_candidates: int, boosts: dict[str, str]
) -> list[MergedCandidate]:
    """The candidates in the order :func:`.recall_boost.trim_merged_candidates` keeps them.

    Under the cap the trim keeps the list as fusion wrote it; over it, it sorts by the boosted RRF
    score (stable). Reading claims in this order is what makes "refill from the tail" refill with
    exactly the candidates the trim would have kept next.
    """
    if len(candidates) <= max_candidates:
        return candidates
    return sorted(candidates, key=lambda mc: boosted_rrf_score(mc, boosts), reverse=True)


@dataclass(frozen=True)
class RemovedCandidate:
    """A candidate the view removed, with the claim that removed it."""

    memory_id: str
    claim: ClaimState


@dataclass(frozen=True)
class ClaimFilterResult:
    """What :func:`filter_candidates_by_claims` kept and removed."""

    #: The shown candidates, in the order they were read (the trim's order), annotated.
    kept: list[MergedCandidate]
    #: The removed candidates, in the same order.
    removed: list[RemovedCandidate]
    #: Candidates whose claim was read.
    claims_checked: int
    #: Reads after the first.
    refill_rounds: int
    #: Candidates past the last read: dropped, never passed on without a claim check.
    unchecked: int


async def filter_candidates_by_claims(
    candidates: Sequence[MergedCandidate],
    *,
    view: LifecycleView,
    target: int,
    load_claims: Callable[[list[str]], Awaitable[Mapping[str, ClaimState]]],
    slack: int = CLAIM_LOAD_SLACK,
    max_refills: int = MAX_REFILL_ROUNDS,
) -> ClaimFilterResult:
    """Read the claims of ``candidates`` front to back and keep what ``view`` shows.

    The first read covers ``target + slack`` candidates (``target`` is the reranker trim size). When
    removals leave fewer than ``target`` kept and candidates remain, the next read covers the
    shortfall plus ``slack``, at most ``max_refills`` times. Each kept candidate gets its
    :attr:`~.types.MergedCandidate.lifecycle` annotation. ``load_claims`` raises on failure; the
    caller fails the recall closed rather than show unchecked claims.
    """
    kept: list[MergedCandidate] = []
    removed: list[RemovedCandidate] = []
    position = 0
    rounds = 0
    size = max(target, 0) + slack
    while position < len(candidates):
        batch = list(candidates[position : position + size])
        position += len(batch)
        claims = await load_claims([mc.id for mc in batch])
        for mc in batch:
            claim = claims.get(mc.id)
            if claim is not None and not view.keeps(claim):
                removed.append(RemovedCandidate(memory_id=mc.id, claim=claim))
                continue
            mc.lifecycle = lifecycle_annotation(claim)
            kept.append(mc)
        if len(kept) >= target or position >= len(candidates) or rounds >= max_refills:
            break
        rounds += 1
        size = (target - len(kept)) + slack
    return ClaimFilterResult(
        kept=kept,
        removed=removed,
        claims_checked=position,
        refill_rounds=rounds,
        unchecked=len(candidates) - position,
    )
