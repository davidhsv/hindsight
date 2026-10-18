"""Retention math (engine/adaptive/retention.py): forgetting and reinforcement, pure and clock-free.

Every function under test takes ``now`` explicitly, so each case pins its own instants (day 0, 1,
30, 365 from a fixed origin) and nothing here sleeps, patches a clock or touches a database.
"""

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hindsight_api.engine.adaptive import clock
from hindsight_api.engine.adaptive.models import AdaptiveState, ReinforcementEvent, RetentionState
from hindsight_api.engine.adaptive.retention import (
    DEFAULT_IMPORTANCE,
    MIN_SPACING_QUALITY,
    MIN_STABILITY_DAYS,
    adaptive_factor,
    apply_events,
    default_anchor,
    effective_retention,
    initial_stability,
    memory_floor,
    retention_factor,
    retrievability,
)
from hindsight_api.engine.adaptive.settings import AdaptiveSettings

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


# Static (server-level) fields the settings read; None means "the default", which is also how test
# helpers that build HindsightConfig by introspection leave the map fields.
_STATIC_FIELDS = (
    "adaptive_memory_recall_reinforcement_enabled",
    "adaptive_memory_reflect_reinforcement_enabled",
    "adaptive_memory_reinforcement_cooldown_hours",
    "adaptive_memory_reinforcement_gain",
    "adaptive_memory_source_weights",
    "adaptive_memory_type_stability",
    "adaptive_memory_source_trust",
    "adaptive_memory_conflict_policy",
    "adaptive_memory_archive_enabled",
    "adaptive_memory_archive_threshold",
    "adaptive_memory_archive_grace_days",
    "adaptive_memory_archive_recall_min_score",
    "adaptive_memory_archive_protect_importance",
    "adaptive_memory_prune_enabled",
    "adaptive_memory_prune_after_days",
)


def _settings(*, static: dict | None = None, **bank) -> AdaptiveSettings:
    values = {name: None for name in _STATIC_FIELDS}
    values.update({f"adaptive_memory_{k}": v for k, v in (static or {}).items()})
    return AdaptiveSettings.resolve(
        {"adaptive_memory_mode": "rank", **{f"adaptive_memory_{k}": v for k, v in bank.items()}},
        SimpleNamespace(**values),
        store_supports=True,
    )


def _state(*, last: datetime = DAY0, stability: float = 30.0, count: int = 0, **overrides) -> RetentionState:
    base = RetentionState(
        memory_id="m1",
        importance=0.5,
        stability_days=stability,
        last_reinforced_at=last,
        reinforcement_count=count,
        access_count=0,
        forgetting_exempt=False,
        retention_state="active",
        last_recalled_at=None,
        below_threshold_since=None,
        archived_at=None,
    )
    return replace(base, **overrides)


def _event(
    t: datetime,
    *,
    source: str = "explicit_confirmation",
    key: str | None = None,
    weight: float = 1.0,
    independence: float = 1.0,
) -> ReinforcementEvent:
    return ReinforcementEvent(
        memory_id="m1",
        source_type=source,  # type: ignore[arg-type]
        source_key=key or f"k:{t.isoformat()}",
        event_time=t,
        weight=weight,
        independence_weight=independence,
    )


# --------------------------------------------------------------------------- forgetting


def test_retrievability_is_one_at_the_moment_of_reinforcement():
    assert retrievability(DAY0, 30.0, DAY0) == 1.0


@pytest.mark.parametrize(
    "days, expected", [(0, 1.0), (1, math.exp(-1 / 30)), (30, math.exp(-1)), (365, math.exp(-365 / 30))]
)
def test_retrievability_decays_exponentially_with_elapsed_days(days, expected):
    assert retrievability(DAY0, 30.0, day(days)) == pytest.approx(expected)


def test_retrievability_is_strictly_monotonic_over_time():
    values = [retrievability(DAY0, 30.0, day(d)) for d in (0, 1, 30, 365)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_larger_stability_decays_slower():
    assert retrievability(DAY0, 90.0, day(30)) > retrievability(DAY0, 30.0, day(30))


def test_a_future_reinforcement_clamps_to_fully_retrievable():
    # A historical query_timestamp earlier than the last reinforcement, or clock skew.
    assert retrievability(day(10), 30.0, DAY0) == 1.0


def test_naive_datetimes_are_read_as_utc():
    naive_last = DAY0.replace(tzinfo=None)
    naive_now = day(30).replace(tzinfo=None)
    aware = retrievability(DAY0, 30.0, day(30))
    assert retrievability(naive_last, 30.0, day(30)) == pytest.approx(aware)
    assert retrievability(DAY0, 30.0, naive_now) == pytest.approx(aware)
    # The same instant expressed in another offset is the same instant.
    plus_five = timezone(timedelta(hours=5))
    assert retrievability(DAY0.astimezone(plus_five), 30.0, day(30)) == pytest.approx(aware)


def test_exempt_memories_never_fade():
    assert retrievability(DAY0, 30.0, day(365), exempt=True) == 1.0


def test_a_memory_without_any_anchor_is_fully_retrievable():
    assert retrievability(None, 30.0, day(365)) == 1.0


@pytest.mark.parametrize("bad", [0.0, -5.0, 0.1, float("nan")])
def test_stability_below_the_floor_reads_as_the_floor(bad):
    assert retrievability(DAY0, bad, day(1)) == pytest.approx(math.exp(-1 / MIN_STABILITY_DAYS))


def test_default_anchor_prefers_mentioned_at_then_occurred_start_then_event_date():
    assert default_anchor(day(3), day(2), day(1)) == day(3)
    assert default_anchor(None, day(2), day(1)) == day(2)
    assert default_anchor(None, None, day(1)) == day(1)
    assert default_anchor(None, None, None) is None
    assert default_anchor(DAY0.replace(tzinfo=None), None, None) == DAY0


@pytest.mark.parametrize(
    "fact_type, importance, expected",
    [
        ("world", 0.5, 30.0),
        ("experience", 0.5, 15.0),
        ("observation", 0.5, 60.0),
        ("world", 1.0, 45.0),
        ("world", 0.0, 15.0),
        ("unknown-type", 0.5, 30.0),  # unknown types get multiplier 1
    ],
)
def test_initial_stability(fact_type, importance, expected):
    assert initial_stability(fact_type, importance, _settings()) == pytest.approx(expected)


def test_initial_stability_is_clamped_to_the_bank_bounds():
    assert initial_stability("observation", 1.0, _settings(max_stability_days=40.0)) == 40.0
    tiny = _settings(static={"type_stability": {"experience": 0.001}})
    assert initial_stability("experience", 0.5, tiny) == MIN_STABILITY_DAYS


def test_missing_state_falls_back_to_lazy_defaults():
    effective = effective_retention(
        None,
        fact_type="world",
        mentioned_at=DAY0,
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(30),
    )
    assert effective.materialized is False
    assert effective.importance == DEFAULT_IMPORTANCE
    assert effective.stability_days == pytest.approx(30.0)
    assert effective.reinforcement_count == 0
    assert effective.last_reinforced_at == DAY0
    assert effective.retrievability == pytest.approx(math.exp(-1))


def test_a_state_with_only_a_claim_still_uses_lazy_retention():
    effective = effective_retention(
        AdaptiveState(retention=None, claim=None),
        fact_type="observation",
        mentioned_at=DAY0,
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(60),
    )
    assert effective.materialized is False
    assert effective.stability_days == pytest.approx(60.0)
    assert effective.retrievability == pytest.approx(math.exp(-1))


def test_a_materialized_row_wins_over_the_memory_timestamps():
    row = _state(last=day(100), stability=200.0, count=3, importance=0.9)
    effective = effective_retention(
        AdaptiveState(retention=row, claim=None),
        fact_type="world",
        mentioned_at=DAY0,
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(300),
    )
    assert effective.materialized is True
    assert effective.last_reinforced_at == day(100)
    assert effective.reinforcement_count == 3
    assert effective.importance == 0.9
    assert effective.retrievability == pytest.approx(math.exp(-1))


def test_a_stored_stability_outside_the_current_bounds_is_clamped_on_read():
    settings = _settings(max_stability_days=100.0)
    for stored, expected in ((5000.0, 100.0), (0.01, MIN_STABILITY_DAYS)):
        effective = effective_retention(
            AdaptiveState(retention=_state(stability=stored), claim=None),
            fact_type="world",
            mentioned_at=None,
            occurred_start=None,
            event_date=None,
            settings=settings,
            now=day(1),
        )
        assert effective.stability_days == expected


def test_an_exempt_row_is_fully_retrievable():
    effective = effective_retention(
        AdaptiveState(retention=_state(forgetting_exempt=True), claim=None),
        fact_type="world",
        mentioned_at=None,
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(365),
    )
    assert effective.exempt is True
    assert effective.retrievability == 1.0


def test_memory_floor_rises_only_above_neutral_importance():
    assert memory_floor(0.3, 0.5) == pytest.approx(0.3)
    assert memory_floor(0.3, 0.0) == pytest.approx(0.3)
    assert memory_floor(0.3, 1.0) == pytest.approx(0.3 + 0.7 * 0.5)


def test_the_factor_is_penalty_only_and_bounded():
    # Defaults: alpha 0.1, floor 0.3 -> worst case 0.93, best case 1.0.
    assert adaptive_factor(0.0, 0.1, 0.3) == pytest.approx(0.93)
    assert adaptive_factor(1.0, 0.1, 0.3) == pytest.approx(1.0)
    assert retention_factor(0.0, 0.3) == pytest.approx(0.3)
    for r in (0.0, 0.2, 0.5, 0.9, 1.0):
        assert 0.93 - 1e-12 <= adaptive_factor(r, 0.1, 0.3) <= 1.0
    # alpha 0 switches decay off entirely.
    assert adaptive_factor(0.0, 0.0, 0.3) == 1.0


def test_the_write_clock_is_tz_aware_utc():
    now = clock.utcnow()
    assert now.tzinfo is not None and now.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- reinforcement


def test_an_immediate_reinforcement_gains_only_the_minimum_spacing_quality():
    settings = _settings(static={"reinforcement_cooldown_hours": 0.0})
    applied = apply_events(_state(), [_event(DAY0 + timedelta(hours=1))], settings)
    assert applied.outcomes[0].outcome == "applied"
    assert applied.outcomes[0].applied_gain == pytest.approx(MIN_SPACING_QUALITY)
    assert applied.state.stability_days == pytest.approx(30.0 * (1 + MIN_SPACING_QUALITY))
    assert applied.state.reinforcement_count == 1


def test_a_spaced_reinforcement_gains_more_than_an_immediate_one():
    settings = _settings()
    immediate = apply_events(_state(), [_event(DAY0 + timedelta(hours=7))], settings)
    spaced = apply_events(_state(), [_event(day(30))], settings)
    assert spaced.outcomes[0].applied_gain == pytest.approx(1 - math.exp(-1))
    assert spaced.state.stability_days > immediate.state.stability_days
    assert spaced.state.last_reinforced_at == day(30)


def test_events_inside_the_cooldown_gain_nothing():
    settings = _settings()  # 6h cooldown by default
    applied = apply_events(_state(), [_event(DAY0 + timedelta(hours=2))], settings)
    assert applied.outcomes[0].outcome == "cooldown"
    assert applied.outcomes[0].applied_gain == 0.0
    assert applied.state.stability_days == 30.0
    assert applied.state.reinforcement_count == 0
    assert applied.state.last_reinforced_at == DAY0


def test_the_cooldown_applies_on_both_sides_of_the_last_reinforcement():
    # A late event two hours BEFORE the last reinforcement is the same burst.
    applied = apply_events(_state(last=day(10)), [_event(day(10) - timedelta(hours=2))], _settings())
    assert applied.outcomes[0].outcome == "cooldown"


def test_a_burst_counts_once_whatever_order_it_is_processed_in():
    settings = _settings()
    burst = [_event(day(30) + timedelta(hours=h)) for h in (0, 1, 2, 3)]
    in_order = apply_events(_state(), burst, settings)
    shuffled = apply_events(_state(), [burst[2], burst[0], burst[3], burst[1]], settings)
    assert in_order.state == shuffled.state
    assert [o.outcome for o in in_order.outcomes] == ["applied", "cooldown", "cooldown", "cooldown"]


def test_returns_diminish_with_every_counted_reinforcement():
    settings = _settings()
    state = _state()
    relative_gains = []
    for k in range(1, 5):
        applied = apply_events(state, [_event(state.last_reinforced_at + timedelta(days=30))], settings)
        # Same elapsed time relative to the same S would give the same q; normalise it out.
        quality = 1 - retrievability(
            state.last_reinforced_at, state.stability_days, state.last_reinforced_at + timedelta(days=30)
        )
        relative_gains.append(applied.outcomes[0].applied_gain / quality)
        state = applied.state
    assert relative_gains == pytest.approx([1 / math.sqrt(1 + k) for k in range(4)])


def test_stability_is_capped_at_the_bank_maximum():
    settings = _settings(max_stability_days=100.0)
    events = [_event(day(30 * k)) for k in range(1, 40)]
    applied = apply_events(_state(), events, settings)
    assert applied.state.stability_days == 100.0
    assert all(o.applied_gain >= 0 for o in applied.outcomes)


def test_spaced_reinforcement_beats_ten_same_day_repetitions():
    for cooldown in (0.0, 6.0):
        settings = _settings(static={"reinforcement_cooldown_hours": cooldown})
        spaced = apply_events(_state(), [_event(day(0)), _event(day(30)), _event(day(120))], settings).state
        same_day = apply_events(_state(), [_event(DAY0 + timedelta(hours=h)) for h in range(10)], settings).state
        assert spaced.stability_days > same_day.stability_days, cooldown
        # And the difference is what recall sees later on.
        assert retrievability(spaced.last_reinforced_at, spaced.stability_days, day(150)) > retrievability(
            same_day.last_reinforced_at, same_day.stability_days, day(150)
        )


def test_a_late_event_counts_minimally_and_never_moves_the_anchor_back():
    settings = _settings(static={"reinforcement_cooldown_hours": 0.0})
    applied = apply_events(_state(last=day(30)), [_event(day(10))], settings)
    assert applied.outcomes[0].applied_gain == pytest.approx(MIN_SPACING_QUALITY)
    assert applied.state.last_reinforced_at == day(30)
    assert applied.state.reinforcement_count == 1


def test_weights_scale_the_gain():
    settings = _settings()
    full = apply_events(_state(), [_event(day(30))], settings).outcomes[0].applied_gain
    weak = apply_events(_state(), [_event(day(30), weight=0.5, independence=0.5)], settings).outcomes[0].applied_gain
    assert weak == pytest.approx(full * 0.25)
    gained = apply_events(_state(), [_event(day(30))], _settings(static={"reinforcement_gain": 2.0}))
    assert gained.outcomes[0].applied_gain == pytest.approx(full * 2)


def test_naive_event_times_are_read_as_utc():
    settings = _settings()
    aware = apply_events(_state(), [_event(day(30))], settings).state
    naive = apply_events(_state(), [_event(day(30).replace(tzinfo=None))], settings).state
    assert naive == aware


def test_an_external_recall_is_an_access_even_inside_the_cooldown():
    settings = _settings()
    t = DAY0 + timedelta(hours=1)
    applied = apply_events(_state(), [_event(t, source="final_external_recall")], settings)
    assert applied.outcomes[0].outcome == "cooldown"
    assert applied.state.access_count == 1
    assert applied.state.last_recalled_at == t
    other = apply_events(_state(), [_event(day(30), source="explicit_confirmation")], settings)
    assert other.state.access_count == 0 and other.state.last_recalled_at is None


def test_initial_events_anchor_on_the_first_and_count_the_rest():
    settings = _settings()
    creating = [_event(day(5)), _event(day(1)), _event(day(40))]
    # A freshly materialized state's anchor need not be the earliest creating mention: seed it
    # later than every event, so only the anchor rule (not a cooldown or late-event path) can
    # move it back to day 1.
    applied = apply_events(_state(last=day(60)), creating, settings, initial=True)
    assert [o.outcome for o in applied.outcomes] == ["initial", "initial", "initial"]
    # Applied in time order: day 1 anchors, day 5 and day 40 reinforce.
    assert [o.event.event_time for o in applied.outcomes] == [day(1), day(5), day(40)]
    assert applied.outcomes[0].applied_gain == 0.0
    # Spacing is measured from the day-1 anchor, not from the seeded day 60 (which would make both
    # events late, at the minimal spacing quality).
    assert applied.outcomes[1].applied_gain == pytest.approx(1 - math.exp(-4 / 30))
    assert applied.outcomes[2].applied_gain > MIN_SPACING_QUALITY
    assert applied.state.reinforcement_count == 2
    assert applied.state.last_reinforced_at == day(40)


def test_an_external_recall_outside_the_cooldown_reinforces_weakly():
    settings = _settings()
    applied = apply_events(_state(), [_event(day(30), source="final_external_recall", weight=0.2)], settings)
    outcome = applied.outcomes[0]
    assert outcome.outcome == "applied"
    assert outcome.applied_gain == pytest.approx(0.2 * (1 - math.exp(-1)))
    assert applied.state.stability_days == pytest.approx(30.0 * (1 + 0.2 * (1 - math.exp(-1))))
    assert applied.state.reinforcement_count == 1
    assert applied.state.access_count == 1
    assert applied.state.last_recalled_at == day(30)
    assert applied.state.last_reinforced_at == day(30)


def test_no_events_is_a_no_op():
    state = _state()
    applied = apply_events(state, [], _settings())
    assert applied.state == state
    assert applied.outcomes == []
