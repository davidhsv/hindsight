"""Retention math: retrievability, stability and reinforcement (DESIGN.md §4.1, §4.3).

Everything here is pure and takes ``now`` explicitly; nothing reads a clock. Naive datetimes are
read as UTC, the convention the rest of recall scoring uses.

Model: a memory's retrievability decays exponentially with time since it was last reinforced,
``R = exp(-Δt / S)``, where ``S`` (stability, in days) grows each time independent evidence
reinforces it — more when the evidence arrives after the memory has faded (spacing), less with
every reinforcement already counted (diminishing returns), never past ``max_stability_days``.

This is not forgetting and never deletes anything: recall turns ``R`` into a bounded,
penalty-only ranking factor (:func:`adaptive_factor`), so a strong cue still recovers an old
memory.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from .models import (
    FINAL_EXTERNAL_RECALL,
    AdaptiveState,
    EventOutcome,
    EventOutcomeKind,
    ReinforcementEvent,
    RetentionState,
)
from .settings import AdaptiveSettings

#: Floor on stability, in days. Below half a day a memory would decay within a single session,
#: which no source weight or importance is meant to express.
MIN_STABILITY_DAYS = 0.5

#: Importance of a memory nobody rated: the neutral point, where it scales nothing.
DEFAULT_IMPORTANCE = 0.5

#: Spacing quality never drops below this. An event that lands while the memory is still fully
#: retrievable (R≈1), or out of order (before the last reinforcement), still counts a little:
#: it is independent evidence, just not evidence that the memory had faded.
MIN_SPACING_QUALITY = 0.05

_SECONDS_PER_DAY = 86_400.0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def default_anchor(
    mentioned_at: datetime | None, occurred_start: datetime | None, event_date: datetime | None
) -> datetime | None:
    """Where a memory with no retention row starts decaying from.

    ``mentioned_at`` first: it is when the memory entered the bank, and a raw fact's own
    ``mentioned_at`` never moves. The event dates are fallbacks for rows that lack it.
    """
    for candidate in (mentioned_at, occurred_start, event_date):
        if candidate is not None:
            return _as_utc(candidate)
    return None


def initial_stability(fact_type: str, importance: float, settings: AdaptiveSettings) -> float:
    """Stability of a memory nobody has reinforced yet: ``base × type_mult × (0.5 + importance)``.

    The type multiplier makes an experience fade faster than a world fact and an observation
    (already a consolidation of several facts) slower; importance 0.5 is neutral (×1.0).
    Clamped to ``[MIN_STABILITY_DAYS, max_stability_days]``.
    """
    multiplier = settings.type_stability.get(fact_type, 1.0)
    stability = settings.base_stability_days * multiplier * (0.5 + _clamp(importance, 0.0, 1.0))
    return _clamp(stability, MIN_STABILITY_DAYS, max(MIN_STABILITY_DAYS, settings.max_stability_days))


def rescale_for_importance(
    stability_days: float, old_importance: float, new_importance: float, settings: AdaptiveSettings
) -> float:
    """Stability after an explicit importance change: ``S × (0.5 + new) / (0.5 + old)``.

    The same multiplier :func:`initial_stability` applies, as a ratio, so a memory keeps what it
    learned and only its importance factor changes: a fresh row (neutral importance, ``S0``) set to
    importance ``i`` ends at ``initial_stability(fact_type, i)``. Clamped to
    ``[MIN_STABILITY_DAYS, max_stability_days]``.
    """
    upper = max(MIN_STABILITY_DAYS, settings.max_stability_days)
    base = stability_days if math.isfinite(stability_days) else upper
    ratio = (0.5 + _clamp(new_importance, 0.0, 1.0)) / (0.5 + _clamp(old_importance, 0.0, 1.0))
    return _clamp(base * ratio, MIN_STABILITY_DAYS, upper)


def memory_floor(floor: float, importance: float) -> float:
    """The per-memory retention floor: ``floor + (1 − floor)·max(0, importance − 0.5)``.

    Importance above neutral lifts the floor (an important memory is penalised less however long
    it goes unreinforced); importance below neutral never lowers it below the bank's floor.
    """
    return floor + (1.0 - floor) * max(0.0, _clamp(importance, 0.0, 1.0) - 0.5)


def retrievability(
    last_reinforced_at: datetime | None, stability_days: float, now: datetime, *, exempt: bool = False
) -> float:
    """``R = exp(-Δt / S)``, in (0, 1].

    1 when the memory is exempt from forgetting or has no anchor at all. ``Δt`` is clamped at 0,
    so a memory reinforced "after" ``now`` (a historical ``query_timestamp``, clock skew) is fully
    retrievable rather than more-than-fully. ``S`` is clamped to at least
    :data:`MIN_STABILITY_DAYS`; a non-finite ``S`` reads as that floor.
    """
    if exempt or last_reinforced_at is None:
        return 1.0
    elapsed_days = max(0.0, (_as_utc(now) - _as_utc(last_reinforced_at)).total_seconds() / _SECONDS_PER_DAY)
    stability = stability_days if math.isfinite(stability_days) else MIN_STABILITY_DAYS
    return math.exp(-elapsed_days / max(MIN_STABILITY_DAYS, stability))


@dataclass(frozen=True)
class EffectiveRetention:
    """A memory's retention as recall scores it: from its row, or lazily from its own fields."""

    retrievability: float
    stability_days: float
    importance: float
    reinforcement_count: int
    last_reinforced_at: datetime | None
    exempt: bool
    materialized: bool
    """True when a retention row exists; False when every value above is a lazy default."""


def effective_retention(
    state: AdaptiveState | None,
    *,
    fact_type: str,
    mentioned_at: datetime | None,
    occurred_start: datetime | None,
    event_date: datetime | None,
    settings: AdaptiveSettings,
    now: datetime,
) -> EffectiveRetention:
    """Retention for one memory at ``now``.

    With a row, its values (stability clamped to the bank's current bounds: lowering
    ``max_stability_days`` applies at once, without rewriting rows). Without one, the lazy
    defaults: neutral importance, :func:`initial_stability`, anchored on :func:`default_anchor`.
    """
    upper = max(MIN_STABILITY_DAYS, settings.max_stability_days)
    row = state.retention if state is not None else None
    if row is not None:
        stability = (
            _clamp(row.stability_days, MIN_STABILITY_DAYS, upper) if math.isfinite(row.stability_days) else upper
        )
        last = _as_utc(row.last_reinforced_at)
        return EffectiveRetention(
            retrievability=retrievability(last, stability, now, exempt=row.forgetting_exempt),
            stability_days=stability,
            importance=_clamp(row.importance, 0.0, 1.0),
            reinforcement_count=row.reinforcement_count,
            last_reinforced_at=last,
            exempt=row.forgetting_exempt,
            materialized=True,
        )
    stability = initial_stability(fact_type, DEFAULT_IMPORTANCE, settings)
    anchor = default_anchor(mentioned_at, occurred_start, event_date)
    return EffectiveRetention(
        retrievability=retrievability(anchor, stability, now),
        stability_days=stability,
        importance=DEFAULT_IMPORTANCE,
        reinforcement_count=0,
        last_reinforced_at=anchor,
        exempt=False,
        materialized=False,
    )


def retention_factor(r: float, floor_i: float) -> float:
    """``floor_i + (1 − floor_i)·R``: retrievability mapped onto ``[floor_i, 1]``."""
    return floor_i + (1.0 - floor_i) * r


def adaptive_factor(r: float, alpha: float, floor_i: float) -> float:
    """The ranking multiplier ``(1 − α) + α·retention_factor``, in ``[1 − α·(1 − floor_i), 1]``.

    Penalty-only and bounded: at the defaults (α 0.1, floor 0.3) the worst case is 0.93, so
    relevance still dominates and a precise cue recovers an old memory. ``α = 0`` disables it.
    """
    return (1.0 - alpha) + alpha * retention_factor(r, floor_i)


@dataclass(frozen=True)
class AppliedEvents:
    """The state after :func:`apply_events`, and one outcome per input event (in time order)."""

    state: RetentionState
    outcomes: list[EventOutcome]


def apply_events(
    state: RetentionState,
    events: Sequence[ReinforcementEvent],
    settings: AdaptiveSettings,
    *,
    initial: bool = False,
) -> AppliedEvents:
    """Apply reinforcement events to one memory's retention state (DESIGN.md §4.3).

    Events are applied in ``event_time`` order (ties keep their input order). For each:

    - **cooldown**: within ``reinforcement_cooldown_hours`` of the last reinforcement, on either
      side, it gains nothing. A burst of events with no independent provenance thus counts once,
      whatever order it is processed in.
    - otherwise ``q = clamp(1 − R_before, MIN_SPACING_QUALITY, 1)`` (spacing quality: evidence
      arriving after the memory faded counts more), and
      ``gain = g · w · w_indep · q / √(1 + count)`` (diminishing returns); then
      ``S ← min(S·(1 + gain), max_stability_days)``, ``last ← max(last, t)``, ``count += 1``.
      A late event (``t`` before the last reinforcement) has ``q = MIN_SPACING_QUALITY`` and
      never moves ``last`` back.

    ``final_external_recall`` events are also accesses: they bump ``access_count`` and
    ``last_recalled_at`` whether or not they reinforced (cooldown included), so the access count
    counts recalls, not gains.

    ``initial=True`` is the creating evidence of a new observation: every outcome is marked
    ``initial``; the first event only anchors (``last ← t``, no gain, no count) and the rest are
    applied as above, so independent creating evidence counts once each.
    """
    cooldown = timedelta(hours=settings.reinforcement_cooldown_hours)
    upper = max(MIN_STABILITY_DAYS, settings.max_stability_days)
    stability = (
        _clamp(state.stability_days, MIN_STABILITY_DAYS, upper) if math.isfinite(state.stability_days) else upper
    )
    last = _as_utc(state.last_reinforced_at)
    count = state.reinforcement_count
    access_count = state.access_count
    last_recalled = _as_utc(state.last_recalled_at) if state.last_recalled_at is not None else None

    outcomes: list[EventOutcome] = []
    ordered = sorted(events, key=lambda e: _as_utc(e.event_time))
    for index, event in enumerate(ordered):
        t = _as_utc(event.event_time)
        if event.source_type == FINAL_EXTERNAL_RECALL:
            access_count += 1
            last_recalled = t if last_recalled is None else max(last_recalled, t)

        if initial and index == 0:
            last = t
            outcomes.append(EventOutcome(event=event, outcome="initial", applied_gain=0.0))
            continue

        kind: EventOutcomeKind = "initial" if initial else "applied"
        if abs(t - last) < cooldown:
            outcomes.append(EventOutcome(event=event, outcome="initial" if initial else "cooldown", applied_gain=0.0))
            continue

        if t < last:
            quality = MIN_SPACING_QUALITY
        else:
            quality = _clamp(1.0 - retrievability(last, stability, t), MIN_SPACING_QUALITY, 1.0)
        gain = settings.reinforcement_gain * event.weight * event.independence_weight * quality / math.sqrt(1.0 + count)
        gain = max(0.0, gain) if math.isfinite(gain) else 0.0
        stability = min(stability * (1.0 + gain), upper)
        last = max(last, t)
        count += 1
        outcomes.append(EventOutcome(event=event, outcome=kind, applied_gain=gain))

    return AppliedEvents(
        state=replace(
            state,
            stability_days=stability,
            last_reinforced_at=last,
            reinforcement_count=count,
            access_count=access_count,
            last_recalled_at=last_recalled,
        ),
        outcomes=outcomes,
    )
