"""The adaptive-memory store writers on PostgreSQL (pg0): reinforcement, retention policy, the read
view, and the invalidate → restore round trip.

What the pure tests (test_adaptive_retention_math.py, test_adaptive_evidence.py) cannot show: that
the ledger's unique key makes a re-sent event a no-op, also under two concurrent producers over
overlapping memories; that a background producer skips a memory being deleted instead of waiting;
that recall and reflect cooldowns leave no ledger row; that nothing here moves
``memory_units.updated_at``; and that curation invalidation parks the learned state on the archive
row (with the edges other claims hold to the memory) and restore puts it back.

Seeding uses raw SQL, so the module is ``memory_backend_incompatible``: memories go straight into
``memory_units`` because the tests need exact ``mentioned_at`` values, and claim / evidence /
relation rows go straight into their tables because their writer (the truth lifecycle) arrives in
a later phase.
"""

import asyncio
import dataclasses
import math
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry, generate_latest

from hindsight_api.config import get_config
from hindsight_api.engine.adaptive import clock
from hindsight_api.engine.adaptive.evidence import reinforcement_event
from hindsight_api.engine.adaptive.models import RecordedEvent, RetentionPolicy, RetentionState
from hindsight_api.engine.adaptive.retention import apply_events, initial_stability
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.db.ops import ReinforcementEventRow, ReinforcementKey, RetentionStateUpdate
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories

pytestmark = pytest.mark.memory_backend_incompatible

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)
#: The adaptive write clock the tests pin (every row's created_at / processed_at / updated_at).
NOW = DAY0 + timedelta(days=400)


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _settings(mode: str = "rank", **overrides) -> AdaptiveSettings:
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": mode}, get_config(), store_supports=True)
    return dataclasses.replace(settings, **overrides)


def _ev(memory_id: str, key: str, at: datetime, source: str = "explicit_confirmation", *, independence: float = 1.0):
    return reinforcement_event(
        memory_id=memory_id,
        source_type=source,  # type: ignore[arg-type]
        source_key=key,
        event_time=at,
        settings=_settings(),
        independence_weight=independence,
    )


def _fresh_state(memory_id: str, anchor: datetime, *, fact_type: str = "world") -> RetentionState:
    """The default row the store creates for a memory with no retention state yet."""
    return RetentionState(
        memory_id=memory_id,
        importance=0.5,
        stability_days=initial_stability(fact_type, 0.5, _settings()),
        last_reinforced_at=anchor,
        reinforcement_count=0,
        access_count=0,
        forgetting_exempt=False,
        retention_state="active",
        last_recalled_at=None,
        below_threshold_since=None,
        archived_at=None,
    )


class _Rollback(Exception):
    pass


@asynccontextmanager
async def _conn(memory):
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        yield conn


async def _bank(memory, request_context) -> str:
    bank_id = f"test-amr-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    return bank_id


async def _seed(
    memory, bank_id: str, n: int = 1, *, mentioned_at: datetime | None = DAY0, fact_type: str = "world"
) -> list[str]:
    ids = [str(uuid.uuid4()) for _ in range(n)]
    async with _conn(memory) as conn:
        for index, memory_id in enumerate(ids):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, mentioned_at) VALUES ($1, $2, $3, $4, $5)",
                uuid.UUID(memory_id),
                bank_id,
                f"Seeded fact number {index} about the payments service ({memory_id[:8]}).",
                fact_type,
                mentioned_at,
            )
    return ids


async def _record(
    memory,
    bank_id: str,
    events,
    *,
    settings: AdaptiveSettings | None = None,
    background: bool = False,
    now: datetime = NOW,
    anchors=None,
    initial: frozenset[str] = frozenset(),
):
    async with _conn(memory) as conn:
        async with conn.transaction():
            return await get_memories().record_reinforcement(
                conn,
                bank_id=bank_id,
                events=events,
                settings=settings or _settings(),
                now=now,
                background=background,
                anchors=anchors,
                initial_memory_ids=initial,
            )


async def _fetch(memory, sql: str, *args):
    async with _conn(memory) as conn:
        return await conn.fetch(sql, *args)


async def _retention(memory, memory_id: str):
    rows = await _fetch(memory, "SELECT * FROM memory_retention_state WHERE memory_id = $1", uuid.UUID(memory_id))
    return rows[0] if rows else None


async def _ledger(memory, memory_id: str):
    return await _fetch(
        memory,
        "SELECT source_type, source_key, event_time, outcome, applied_gain, processed_at "
        "FROM memory_reinforcement_events WHERE memory_id = $1 ORDER BY event_time, source_key",
        uuid.UUID(memory_id),
    )


# --------------------------------------------------------------------------- idempotency


@pytest.mark.asyncio
async def test_the_same_events_twice_apply_once(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)
    events = [_ev(m, "explicit:a", day(30)), _ev(m, "explicit:b", day(90))]

    first = await _record(memory, bank_id, events)

    assert (first.applied, first.deduplicated, first.initialized, first.skipped_missing) == (2, 0, 1, 0)
    expected = apply_events(_fresh_state(m, DAY0), events, _settings())
    assert first.states[m].stability_days == pytest.approx(expected.state.stability_days)
    assert first.states[m].reinforcement_count == 2
    assert [e.outcome for e in first.events] == ["applied", "applied"]
    assert all(e.persisted for e in first.events)
    row = await _retention(memory, m)
    assert row["stability_days"] == pytest.approx(expected.state.stability_days)
    assert row["last_reinforced_at"] == day(90)
    assert row["created_at"] == NOW and row["updated_at"] == NOW
    ledger = await _ledger(memory, m)
    assert [(r["source_key"], r["outcome"]) for r in ledger] == [("explicit:a", "applied"), ("explicit:b", "applied")]
    assert all(r["processed_at"] == NOW for r in ledger)

    second = await _record(memory, bank_id, events, now=NOW + timedelta(days=1))

    assert (second.applied, second.deduplicated, second.initialized) == (0, 2, 0)
    assert [e.outcome for e in second.events] == ["deduplicated", "deduplicated"]
    assert second.states[m] == first.states[m]
    again = await _retention(memory, m)
    # Nothing changed, so nothing was written: not even updated_at.
    assert dict(again) == dict(row)
    assert len(await _ledger(memory, m)) == 2


@pytest.mark.asyncio
async def test_a_key_repeated_in_one_call_counts_once_by_its_earliest_instance(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)

    result = await _record(memory, bank_id, [_ev(m, "explicit:c", day(200)), _ev(m, "explicit:c", day(150))])

    assert [e.outcome for e in result.events] == ["deduplicated", "applied"]
    assert [r["event_time"] for r in await _ledger(memory, m)] == [day(150)]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows_exist", [False, True], ids=["fresh-memories", "existing-rows"])
async def test_concurrent_producers_over_overlapping_memories_count_each_key_once(
    memory, request_context, monkeypatch, rows_exist
):
    """Two transactions reinforcing overlapping memories with a shared key, 20 times: one ledger row per
    distinct key, a reinforcement count that matches the applied rows, and never a deadlock.

    The ledger check is slowed down so both transactions sit between their check and their insert at
    once. On fresh memories the default-row insert's unique key already serializes them; with
    existing rows only the retention-row lock does (without it the second insert conflicts).
    """
    original = PostgreSQLOps.fetch_reinforcement_keys

    async def _slow_check(self, *args, **kwargs):
        found = await original(self, *args, **kwargs)
        await asyncio.sleep(0.02)
        return found

    monkeypatch.setattr(PostgreSQLOps, "fetch_reinforcement_keys", _slow_check)
    bank_id = await _bank(memory, request_context)
    for _ in range(20):
        a, b, c = await _seed(memory, bank_id, 3)
        if rows_exist:
            await _record(memory, bank_id, [_ev(m, "explicit:seed", day(1)) for m in (a, b, c)])
        one = [_ev(a, "explicit:1", day(30)), _ev(b, "explicit:shared", day(30)), _ev(b, "explicit:t1", day(60))]
        two = [
            _ev(c, "explicit:2", day(30)),
            _ev(b, "explicit:shared", day(30)),
            _ev(b, "explicit:t2", day(90)),
            _ev(a, "explicit:1", day(30)),
        ]

        r1, r2 = await asyncio.gather(_record(memory, bank_id, one), _record(memory, bank_id, two))

        assert r1.deduplicated + r2.deduplicated == 2
        assert r1.initialized + r2.initialized == (0 if rows_exist else 3)
        seed = {"explicit:seed"} if rows_exist else set()
        for memory_id, keys in (
            (a, {"explicit:1"}),
            (b, {"explicit:shared", "explicit:t1", "explicit:t2"}),
            (c, {"explicit:2"}),
        ):
            ledger = await _ledger(memory, memory_id)
            assert {r["source_key"] for r in ledger} == keys | seed
            applied = sum(1 for r in ledger if r["outcome"] == "applied")
            assert (await _retention(memory, memory_id))["reinforcement_count"] == applied


# --------------------------------------------------------------------------- locking


@pytest.mark.asyncio
async def test_a_background_producer_skips_a_memory_being_deleted(memory, request_context):
    bank_id = await _bank(memory, request_context)
    doomed, other = await _seed(memory, bank_id, 2)

    with pytest.raises(_Rollback):
        async with _conn(memory) as deleter:
            async with deleter.transaction():
                await deleter.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(doomed))
                result = await asyncio.wait_for(
                    _record(
                        memory,
                        bank_id,
                        [_ev(doomed, "explicit:x", day(30)), _ev(other, "explicit:x", day(30))],
                        background=True,
                    ),
                    timeout=10,
                )
                raise _Rollback

    assert [e.outcome for e in result.events] == ["skipped_missing", "applied"]
    assert (result.skipped_missing, result.applied, result.initialized) == (1, 1, 1)
    assert doomed not in result.states
    # The delete rolled back: the memory is still there, untouched by the producer.
    assert await _retention(memory, doomed) is None
    assert await _ledger(memory, doomed) == []


@pytest.mark.asyncio
async def test_a_request_path_producer_waits_for_a_delete_and_then_skips_the_memory(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (doomed,) = await _seed(memory, bank_id)

    async with _conn(memory) as deleter:
        async with deleter.transaction():
            await deleter.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(doomed))
            task = asyncio.create_task(_record(memory, bank_id, [_ev(doomed, "explicit:x", day(30))]))
            await asyncio.sleep(0.3)
            assert not task.done(), "the request path waits on the parent row instead of skipping it"
    result = await asyncio.wait_for(task, timeout=10)

    assert [e.outcome for e in result.events] == ["skipped_missing"]


@pytest.mark.asyncio
async def test_unknown_or_foreign_memories_are_skipped(memory, request_context):
    bank_id = await _bank(memory, request_context)
    other_bank = await _bank(memory, request_context)
    (foreign,) = await _seed(memory, other_bank)

    result = await _record(
        memory, bank_id, [_ev(foreign, "explicit:x", day(30)), _ev("not-a-uuid", "explicit:x", day(30))]
    )

    assert [e.outcome for e in result.events] == ["skipped_missing", "skipped_missing"]
    assert await _retention(memory, foreign) is None


# --------------------------------------------------------------------------- cooldowns and gates


@pytest.mark.asyncio
async def test_recall_and_reflect_cooldowns_leave_no_ledger_row(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)
    settings = _settings(recall_reinforcement_enabled=True, reinforcement_cooldown_hours=6.0)
    await _record(memory, bank_id, [_ev(m, "explicit:x", day(30))], settings=settings)

    result = await _record(
        memory,
        bank_id,
        [
            _ev(m, "recall:2026-01-31T00:00:00+00:00", day(30) + timedelta(hours=1), "final_external_recall"),
            _ev(m, "reflect:2026-01-31T00:00:00+00:00", day(30) + timedelta(hours=2), "successful_reflect_use"),
            _ev(m, "explicit:y", day(30) + timedelta(hours=3)),
        ],
        settings=settings,
    )

    assert [(e.outcome, e.persisted) for e in result.events] == [
        ("cooldown", False),
        ("cooldown", False),
        ("cooldown", True),
    ]
    ledger = await _ledger(memory, m)
    assert [(r["source_key"], r["outcome"]) for r in ledger] == [("explicit:x", "applied"), ("explicit:y", "cooldown")]
    row = await _retention(memory, m)
    # The recall still counted as an access, but reinforced nothing.
    assert (row["reinforcement_count"], row["access_count"]) == (1, 1)
    assert row["last_recalled_at"] == day(30) + timedelta(hours=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "overrides"),
    [("off", {}), ("observe", {}), ("rank", {"reinforcement_enabled": False})],
    ids=["mode-off", "mode-observe", "bank-switch-off"],
)
async def test_nothing_is_written_when_reinforcement_is_off(memory, request_context, mode, overrides):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)

    result = await _record(memory, bank_id, [_ev(m, "explicit:x", day(30))], settings=_settings(mode, **overrides))

    assert [e.outcome for e in result.events] == ["disabled"]
    assert (result.disabled, result.initialized) == (1, 0)
    assert await _retention(memory, m) is None
    assert await _ledger(memory, m) == []


@pytest.mark.asyncio
async def test_the_static_recall_and_reflect_switches_gate_their_sources_only(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)
    settings = _settings(recall_reinforcement_enabled=False, reflect_reinforcement_enabled=False)

    result = await _record(
        memory,
        bank_id,
        [
            _ev(m, "recall:b", day(30), "final_external_recall"),
            _ev(m, "reflect:b", day(30), "successful_reflect_use"),
            _ev(m, "explicit:x", day(30)),
        ],
        settings=settings,
    )

    assert [e.outcome for e in result.events] == ["disabled", "disabled", "applied"]
    assert [r["source_type"] for r in await _ledger(memory, m)] == ["explicit_confirmation"]


@pytest.mark.asyncio
async def test_reinforcement_and_policies_never_stamp_memory_units_updated_at(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)

    async def _watermark():
        rows = await _fetch(memory, "SELECT updated_at FROM memory_units WHERE id = $1", uuid.UUID(m))
        return rows[0]["updated_at"]

    before = await _watermark()
    await _record(memory, bank_id, [_ev(m, "explicit:x", day(30)), _ev(m, "explicit:y", day(60))])
    async with _conn(memory) as conn:
        await get_memories().set_retention_policy(
            conn, bank_id=bank_id, memory_id=m, importance=0.9, settings=_settings(), now=NOW
        )

    assert await _watermark() == before


# --------------------------------------------------------------------------- initial events


@pytest.mark.asyncio
async def test_a_new_observation_is_anchored_on_its_earliest_creating_evidence(memory, request_context):
    bank_id = await _bank(memory, request_context)
    # The observation row's own mention time is the latest source's; the anchor is the earliest.
    (obs,) = await _seed(memory, bank_id, mentioned_at=day(100), fact_type="observation")
    creating = [
        _ev(obs, "session:s2", day(40), "observation_new_evidence"),
        _ev(obs, "session:s1", day(10), "observation_new_evidence"),
    ]

    result = await _record(memory, bank_id, creating, anchors={obs: day(10)}, initial=frozenset({obs}))

    assert [e.outcome for e in result.events] == ["initial", "initial"]
    gains = {e.source_key: e.applied_gain for e in result.events}
    assert gains["session:s1"] == 0.0 and gains["session:s2"] > 0
    expected = apply_events(_fresh_state(obs, day(10), fact_type="observation"), creating, _settings(), initial=True)
    row = await _retention(memory, obs)
    assert row["last_reinforced_at"] == day(40)
    assert row["reinforcement_count"] == 1
    assert row["stability_days"] == pytest.approx(expected.state.stability_days)

    # The row exists now: a re-sent creation batch is deduplicated, and new evidence in it is applied
    # as ordinary evidence (a late one), never re-anchoring the memory backwards.
    again = await _record(
        memory,
        bank_id,
        [*creating, _ev(obs, "session:s0", day(5), "observation_new_evidence")],
        anchors={obs: day(5)},
        initial=frozenset({obs}),
    )
    assert [e.outcome for e in again.events] == ["deduplicated", "deduplicated", "applied"]
    assert (await _retention(memory, obs))["last_reinforced_at"] == day(40)


@pytest.mark.asyncio
async def test_a_memory_without_any_date_is_anchored_on_its_creation(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id, mentioned_at=None)
    created = (await _fetch(memory, "SELECT created_at FROM memory_units WHERE id = $1", uuid.UUID(m)))[0]

    result = await _record(memory, bank_id, [_ev(m, "explicit:x", created["created_at"] + timedelta(hours=1))])

    assert result.events[0].outcome == "cooldown"
    assert (await _retention(memory, m))["last_reinforced_at"] == created["created_at"]


# --------------------------------------------------------------------------- retention policy


@pytest.mark.asyncio
async def test_importance_rescales_stability_and_exemption_round_trips(memory, request_context):
    bank_id = await _bank(memory, request_context)
    fresh, learned = await _seed(memory, bank_id, 2)
    settings = _settings()
    store = get_memories()

    async with _conn(memory) as conn:
        state = await store.set_retention_policy(
            conn, bank_id=bank_id, memory_id=fresh, importance=0.9, settings=settings, now=NOW
        )
    assert state is not None
    assert state.stability_days == pytest.approx(initial_stability("world", 0.9, settings))

    reinforced = (await _record(memory, bank_id, [_ev(learned, "explicit:x", day(30))])).states[learned]
    async with _conn(memory) as conn:
        state = await store.set_retention_policy(
            conn,
            bank_id=bank_id,
            memory_id=learned,
            importance=1.0,
            forgetting_exempt=True,
            exemption_reason="legal hold",
            settings=settings,
            now=NOW,
        )
    assert state.stability_days == pytest.approx(reinforced.stability_days * 1.5)
    assert state.reinforcement_count == 1
    row = await _retention(memory, learned)
    assert (row["forgetting_exempt"], row["exemption_reason"]) == (True, "legal hold")
    assert row["importance"] == pytest.approx(1.0)

    async with _conn(memory) as conn:
        await store.set_retention_policy(
            conn, bank_id=bank_id, memory_id=learned, forgetting_exempt=False, settings=settings, now=NOW
        )
        assert (
            await store.set_retention_policy(
                conn, bank_id=bank_id, memory_id=str(uuid.uuid4()), importance=0.2, settings=settings, now=NOW
            )
            is None
        )
        with pytest.raises(ValueError):
            await store.set_retention_policy(
                conn, bank_id=bank_id, memory_id=learned, importance=1.5, settings=settings, now=NOW
            )
    row = await _retention(memory, learned)
    assert (row["forgetting_exempt"], row["exemption_reason"]) == (False, None)


@pytest.mark.asyncio
async def test_bulk_policies_merge_per_memory(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)

    async with _conn(memory) as conn:
        states = await get_memories().set_retention_policies(
            conn,
            bank_id=bank_id,
            policies=[
                RetentionPolicy(memory_id=m, importance=0.8),
                RetentionPolicy(memory_id=m, forgetting_exempt=True),
            ],
            settings=_settings(),
            now=NOW,
        )

    assert set(states) == {m}
    assert states[m].importance == pytest.approx(0.8)
    assert states[m].forgetting_exempt is True


# --------------------------------------------------------------------------- read view


@pytest.mark.asyncio
async def test_the_lifecycle_view_reads_lazy_defaults_rows_and_the_newest_events(memory, request_context):
    bank_id = await _bank(memory, request_context)
    bare, busy = await _seed(memory, bank_id, 2)
    settings = _settings()
    store = get_memories()

    async with _conn(memory) as conn:
        view = await store.get_memory_lifecycle(conn, bank_id=bank_id, memory_id=bare, now=day(30), settings=settings)
    assert view is not None
    assert (view.stored, view.archived, view.retention.materialized) == (False, False, False)
    assert view.retention.retrievability == pytest.approx(math.exp(-1))
    assert view.retention.stability_days == pytest.approx(30.0)
    assert view.claim is None and view.recent_events == [] and view.relations_out == []

    await _record(memory, bank_id, [_ev(busy, f"explicit:{i:02d}", day(10 * (i + 1))) for i in range(25)])
    async with _conn(memory) as conn:
        view = await store.get_memory_lifecycle(conn, bank_id=bank_id, memory_id=busy, now=NOW, settings=settings)
        assert (
            await store.get_memory_lifecycle(
                conn, bank_id="some-other-bank", memory_id=busy, now=NOW, settings=settings
            )
            is None
        )
        assert (
            await store.get_memory_lifecycle(
                conn, bank_id=bank_id, memory_id=str(uuid.uuid4()), now=NOW, settings=settings
            )
            is None
        )
    assert view.stored and view.retention.materialized
    assert len(view.recent_events) == 20
    assert view.recent_events[0].source_key == "explicit:24"
    assert [e.event_time for e in view.recent_events] == sorted(
        (e.event_time for e in view.recent_events), reverse=True
    )
    assert view.retention.reinforcement_count == 25
    assert view.model_dump(mode="json")["retention"]["materialized"] is True


# --------------------------------------------------------------------------- invalidate / restore


@pytest.mark.asyncio
async def test_invalidate_then_restore_keeps_stability_count_and_keys(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)
    events = [_ev(m, "explicit:a", day(30)), _ev(m, "explicit:b", day(90))]
    learned = (await _record(memory, bank_id, events)).states[m]

    await memory.update_memory_unit(bank_id, m, state="invalidated", request_context=request_context)

    assert await _retention(memory, m) is None
    archived = await _fetch(memory, "SELECT adaptive_state FROM invalidated_memory_units WHERE id = $1", m)
    assert archived and archived[0]["adaptive_state"] is not None
    async with _conn(memory) as conn:
        view = await get_memories().get_memory_lifecycle(
            conn, bank_id=bank_id, memory_id=m, now=NOW, settings=_settings()
        )
    assert view.archived and view.stored
    assert view.retention.stability_days == pytest.approx(learned.stability_days)
    assert {e.source_key for e in view.recent_events} == {"explicit:a", "explicit:b"}

    await memory.update_memory_unit(bank_id, m, state="valid", request_context=request_context)

    row = await _retention(memory, m)
    assert row["stability_days"] == pytest.approx(learned.stability_days)
    assert row["reinforcement_count"] == learned.reinforcement_count == 2
    assert row["last_reinforced_at"] == learned.last_reinforced_at
    assert [r["source_key"] for r in await _ledger(memory, m)] == ["explicit:a", "explicit:b"]
    resent = await _record(memory, bank_id, events)
    assert resent.deduplicated == 2 and resent.applied == 0


@pytest.mark.asyncio
async def test_an_archived_memory_restores_in_both_copies_of_a_cloned_bank(memory, request_context):
    """The archive carries the snapshot's learned state but not its ledger (events never leave the
    deployment that recorded them, DESIGN.md §7): the clone restores S and the count, the source
    keeps its keys, and restoring one copy never blocks the other on a global primary key."""
    from hindsight_api.engine.transfer import export_bank

    source = await _bank(memory, request_context)
    target = f"test-amr-{uuid.uuid4().hex[:8]}"
    (m,) = await _seed(memory, source)
    events = [_ev(m, "explicit:a", day(30)), _ev(m, "explicit:b", day(90))]
    learned = (await _record(memory, source, events)).states[m]
    await memory.update_memory_unit(source, m, state="invalidated", request_context=request_context)
    async with _conn(memory) as conn:
        archive = await export_bank(conn, source)
    result = await memory.import_bank_async(archive, request_context, target_bank_id=target)
    assert result.invalidated_memories_imported == 1
    # The clone's archive row has a fresh unit id; the engine has no reader for archived ids.
    (clone,) = [
        str(r["id"]) for r in await _fetch(memory, "SELECT id FROM invalidated_memory_units WHERE bank_id = $1", target)
    ]

    await memory.update_memory_unit(target, clone, state="valid", request_context=request_context)
    await memory.update_memory_unit(source, m, state="valid", request_context=request_context)

    for memory_id in (clone, m):
        row = await _retention(memory, memory_id)
        assert row["stability_days"] == pytest.approx(learned.stability_days)
        assert row["reinforcement_count"] == learned.reinforcement_count == 2
    assert [r["source_key"] for r in await _ledger(memory, m)] == ["explicit:a", "explicit:b"]
    assert await _ledger(memory, clone) == [], "the ledger is not exported"
    resent = await _record(memory, source, events)
    assert resent.deduplicated == 2 and resent.applied == 0


@pytest.mark.asyncio
async def test_a_memory_without_adaptive_state_archives_no_snapshot(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (m,) = await _seed(memory, bank_id)

    await memory.update_memory_unit(bank_id, m, state="invalidated", request_context=request_context)

    archived = await _fetch(memory, "SELECT adaptive_state FROM invalidated_memory_units WHERE id = $1", m)
    assert archived[0]["adaptive_state"] is None


async def _seed_supersession(memory, bank_id: str, *, claim_updated_at: datetime) -> SimpleNamespace:
    """Old fact F superseded by fact S: F's claim, its evidence (S), and the relation S → F."""
    old, new = await _seed(memory, bank_id, 2)
    relation_id = uuid.uuid4()
    async with _conn(memory) as conn:
        await conn.execute(
            "UPDATE memory_units SET consolidated_at = $2 WHERE id = $1", uuid.UUID(old), claim_updated_at
        )
        await conn.execute(
            """
            INSERT INTO memory_claim_state
                (memory_id, bank_id, truth_status, retired_via, superseded_by_id, valid_to, created_at, updated_at)
            VALUES ($1, $2, 'superseded', 'supersedes', $3, $4, $4, $4)
            """,
            uuid.UUID(old),
            bank_id,
            uuid.UUID(new),
            claim_updated_at,
        )
        await conn.execute(
            "INSERT INTO memory_claim_evidence (memory_id, evidence_memory_id, bank_id, created_at) "
            "VALUES ($1, $2, $3, $4)",
            uuid.UUID(old),
            uuid.UUID(new),
            bank_id,
            claim_updated_at,
        )
        await conn.execute(
            """
            INSERT INTO memory_lifecycle_relations
                (id, bank_id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot,
                 origin, created_at)
            VALUES ($1, $2, 'supersedes', $3, $4, '{"text": "new"}'::jsonb, '{"text": "old"}'::jsonb,
                    'consolidation', $5)
            """,
            relation_id,
            bank_id,
            uuid.UUID(new),
            uuid.UUID(old),
            claim_updated_at,
        )
    return SimpleNamespace(old=old, new=new, relation_id=relation_id)


async def _store_move(memory, bank_id: str, unit_id: str, *, restore: bool) -> None:
    store = get_memories()
    async with _conn(memory) as conn:
        async with conn.transaction():
            if restore:
                assert await store.restore_memory(conn=conn, fq_table=None, bank_id=bank_id, unit_id=unit_id)
            else:
                assert await store.invalidate_memory(
                    conn=conn, fq_table=None, bank_id=bank_id, unit_id=unit_id, reason="t"
                )


async def _claim(memory, memory_id: str):
    rows = await _fetch(memory, "SELECT * FROM memory_claim_state WHERE memory_id = $1", uuid.UUID(memory_id))
    return rows[0] if rows else None


async def _evidence_of(memory, memory_id: str) -> list[str]:
    rows = await _fetch(
        memory, "SELECT evidence_memory_id FROM memory_claim_evidence WHERE memory_id = $1", uuid.UUID(memory_id)
    )
    return [str(r["evidence_memory_id"]) for r in rows]


async def _relations(memory, bank_id: str):
    return await _fetch(
        memory,
        "SELECT id, source_memory_id, target_memory_id FROM memory_lifecycle_relations WHERE bank_id = $1",
        bank_id,
    )


@pytest.mark.asyncio
async def test_a_restored_superseded_fact_keeps_its_claim_and_stays_consolidated(memory, request_context, monkeypatch):
    monkeypatch.setattr(clock, "utcnow", lambda: day(300))
    bank_id = await _bank(memory, request_context)
    seeded = await _seed_supersession(memory, bank_id, claim_updated_at=day(200))

    await _store_move(memory, bank_id, seeded.old, restore=False)

    assert await _claim(memory, seeded.old) is None
    (relation,) = await _relations(memory, bank_id)
    assert relation["target_memory_id"] is None  # a tombstone, its snapshot kept

    await _store_move(memory, bank_id, seeded.old, restore=True)

    claim = await _claim(memory, seeded.old)
    assert claim["truth_status"] == "superseded"
    assert str(claim["superseded_by_id"]) == seeded.new
    assert await _evidence_of(memory, seeded.old) == [seeded.new]
    (relation,) = await _relations(memory, bank_id)
    assert (str(relation["source_memory_id"]), str(relation["target_memory_id"])) == (seeded.new, seeded.old)
    # restore_memory requeues a restored fact, but a retired one must not read as pending.
    unit = (await _fetch(memory, "SELECT consolidated_at FROM memory_units WHERE id = $1", uuid.UUID(seeded.old)))[0]
    assert unit["consolidated_at"] == day(300)


@pytest.mark.asyncio
async def test_invalidating_a_successor_keeps_the_predecessor_retired_and_restore_reattaches_it(
    memory, request_context, monkeypatch
):
    monkeypatch.setattr(clock, "utcnow", lambda: day(300))
    bank_id = await _bank(memory, request_context)
    seeded = await _seed_supersession(memory, bank_id, claim_updated_at=day(200))

    await _store_move(memory, bank_id, seeded.new, restore=False)

    claim = await _claim(memory, seeded.old)
    # Invalidation never reverts a claim: the predecessor stays superseded, only the edges go.
    assert claim["truth_status"] == "superseded" and claim["superseded_by_id"] is None
    assert await _evidence_of(memory, seeded.old) == []
    (relation,) = await _relations(memory, bank_id)
    assert relation["source_memory_id"] is None

    await _store_move(memory, bank_id, seeded.new, restore=True)

    claim = await _claim(memory, seeded.old)
    assert str(claim["superseded_by_id"]) == seeded.new
    assert claim["updated_at"] == day(200), "undoing the cascade is not a new lifecycle decision"
    assert await _evidence_of(memory, seeded.old) == [seeded.new]
    relations = await _relations(memory, bank_id)
    assert len(relations) == 1, "the tombstone is re-pointed, not duplicated"
    assert (str(relations[0]["source_memory_id"]), str(relations[0]["target_memory_id"])) == (seeded.new, seeded.old)


@pytest.mark.asyncio
async def test_restore_does_not_override_a_newer_lifecycle_decision(memory, request_context, monkeypatch):
    monkeypatch.setattr(clock, "utcnow", lambda: day(300))
    bank_id = await _bank(memory, request_context)
    seeded = await _seed_supersession(memory, bank_id, claim_updated_at=day(200))
    await _store_move(memory, bank_id, seeded.new, restore=False)
    async with _conn(memory) as conn:
        await conn.execute(
            "UPDATE memory_claim_state SET truth_status = 'active', updated_at = $2 WHERE memory_id = $1",
            uuid.UUID(seeded.old),
            day(301),
        )

    await _store_move(memory, bank_id, seeded.new, restore=True)

    claim = await _claim(memory, seeded.old)
    assert (claim["truth_status"], claim["superseded_by_id"]) == ("active", None)
    assert await _evidence_of(memory, seeded.old) == []


@pytest.mark.asyncio
async def test_deleting_a_bank_removes_its_relation_tombstones(memory, request_context):
    bank_id = await _bank(memory, request_context)
    await _seed_supersession(memory, bank_id, claim_updated_at=day(200))
    assert len(await _relations(memory, bank_id)) == 1

    await memory.delete_bank(bank_id, request_context=request_context)

    assert await _relations(memory, bank_id) == []


# --------------------------------------------------------------------------- metrics (no database)


def test_reinforcement_metrics_export_under_their_exact_prometheus_names():
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    from hindsight_api.metrics import MetricsCollector

    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    provider = MeterProvider(metric_readers=[reader])
    config = SimpleNamespace(
        metrics_include_bank_id=False,
        metrics_include_tenant=False,
        recall_diagnostic_phases=True,
        recall_phase_sample_every=1,
    )

    def _recorded(source: str, outcome: str) -> RecordedEvent:
        return RecordedEvent(
            memory_id="m", source_type=source, source_key="k", outcome=outcome, applied_gain=0.0, persisted=True
        )  # type: ignore[arg-type]

    try:
        with (
            patch("hindsight_api.metrics.get_meter", return_value=provider.get_meter("test-adaptive-writes")),
            patch("hindsight_api.config.get_config", return_value=config),
        ):
            collector = MetricsCollector()
        collector.record_adaptive_reinforcement(
            bank_id="b",
            initialized=2,
            events=[
                _recorded("explicit_confirmation", "applied"),
                _recorded("final_external_recall", "cooldown"),
                _recorded("observation_new_evidence", "initial"),
                _recorded("explicit_confirmation", "deduplicated"),
                _recorded("explicit_confirmation", "disabled"),
                _recorded("explicit_confirmation", "skipped_missing"),
            ],
        )
        text = generate_latest(registry).decode()
    finally:
        provider.shutdown()

    lines = text.splitlines()

    def value(name: str, *labels: str) -> str:
        """The sample of ``name`` carrying every one of ``labels`` (exported labels are sorted)."""
        return next(
            line.rsplit(" ", 1)[1]
            for line in lines
            if line.startswith(name + "{") and all(label in line for label in labels)
        )

    assert value("hindsight_adaptive_memory_state_initialized_total") == "2.0"
    for source, outcome in (
        ("explicit_confirmation", "applied"),
        ("final_external_recall", "cooldown"),
        ("observation_new_evidence", "initial"),
    ):
        assert (
            value(
                "hindsight_adaptive_memory_reinforcement_events_total",
                f'outcome="{outcome}"',
                f'source_type="{source}"',
            )
            == "1.0"
        )
    assert (
        value(
            "hindsight_adaptive_memory_reinforcement_events_deduplicated_total", 'source_type="explicit_confirmation"'
        )
        == "1.0"
    )
    assert 'outcome="disabled"' not in text and 'outcome="skipped_missing"' not in text
    assert 'bank_id="b"' not in text


# --------------------------------------------------------------------------- Oracle statements (no database)


class _RecordingOracleConn:
    """Enough of an Oracle connection to run OracleOps' adaptive methods and capture their SQL."""

    backend_type = "oracle"

    def __init__(self, live: set[uuid.UUID], with_state: set[uuid.UUID]):
        self.live = live
        self.with_state = set(with_state)
        self.statements: list[str] = []
        self.inserted_events: list[uuid.UUID] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchrow(self, query: str, *args):
        self.statements.append(query)
        memory_id = args[0]
        if "FROM memory_units" in query:
            if memory_id not in self.live:
                return None
            return {
                "id": memory_id,
                "fact_type": "world",
                "mentioned_at": None,
                "occurred_start": DAY0,
                "event_date": None,
                "created_at": DAY0,
            }
        return {"memory_id": memory_id, "importance": 0.5, "exemption_reason": None}

    async def fetch(self, query: str, *args):
        self.statements.append(query)
        if "FROM memory_retention_state" in query:
            return [{"memory_id": m} for m in args[0] if m in self.with_state]
        if "SELECT id FROM memory_reinforcement_events" in query:
            return [{"id": i} for i in args[0] if i in self.inserted_events]
        return []

    async def executemany(self, query: str, rows):
        self.statements.append(query)
        if "INTO memory_reinforcement_events" in query:
            self.inserted_events.extend(row[0] for row in rows)
        if "INTO memory_retention_state" in query:
            self.with_state.update(row[0] for row in rows)


@pytest.mark.asyncio
async def test_the_oracle_writer_locks_rows_one_by_one_and_speaks_oracle():
    live_a, live_b, gone = sorted(uuid.uuid4() for _ in range(3))
    conn = _RecordingOracleConn(live={live_a, live_b}, with_state={live_a})
    ops = OracleOps()

    lock = await ops.upsert_default_retention_state(
        conn,
        retention_table="memory_retention_state",
        units_table="memory_units",
        bank_id="b",
        memory_ids=[gone, live_b, live_a],
        default_stability={"world": 30.0},
        fallback_stability=30.0,
        anchors={},
        now=NOW,
        skip_locked=True,
    )
    event = ReinforcementEventRow(
        id=uuid.uuid4(),
        memory_id=live_a,
        source_type="explicit_confirmation",
        source_key="explicit:x",
        event_time=day(30),
        weight=1.0,
        independence_weight=1.0,
        applied_gain=0.5,
        outcome="applied",
        processed_at=NOW,
        created_at=NOW,
    )
    inserted = await ops.insert_reinforcement_events(
        conn, table="memory_reinforcement_events", bank_id="b", rows=[event]
    )
    await ops.fetch_reinforcement_keys(
        conn,
        table="memory_reinforcement_events",
        bank_id="b",
        keys=[ReinforcementKey(memory_id=str(live_a), source_type="explicit_confirmation", source_key="explicit:x")],
    )
    await ops.update_retention_state(
        conn,
        table="memory_retention_state",
        bank_id="b",
        rows=[
            RetentionStateUpdate(
                memory_id=live_a,
                importance=0.5,
                stability_days=40.0,
                last_reinforced_at=day(30),
                reinforcement_count=1,
                access_count=0,
                last_recalled_at=None,
                forgetting_exempt=False,
                exemption_reason=None,
            )
        ],
        now=NOW,
    )

    assert lock.present == sorted(str(m) for m in (live_a, live_b))
    assert lock.initialized == frozenset({str(live_b)})  # live_a already had a row
    assert len(lock.rows) == 2
    assert inserted == frozenset({str(event.id)})
    parent_locks = [s for s in conn.statements if "FROM memory_units" in s]
    assert len(parent_locks) == 3 and all(s.rstrip().endswith("FOR UPDATE SKIP LOCKED") for s in parent_locks)
    assert any(
        "IGNORE_ROW_ON_DUPKEY_INDEX(memory_retention_state, pk_memory_retention_state)" in s for s in conn.statements
    )
    assert any("IGNORE_ROW_ON_DUPKEY_INDEX(memory_reinforcement_events, uq_mre_key)" in s for s in conn.statements)
    for statement in conn.statements:
        rewritten = _rewrite_pg_to_oracle(statement).query
        assert "$" not in rewritten and "FOR KEY SHARE" not in rewritten and "ON CONFLICT" not in rewritten, rewritten
