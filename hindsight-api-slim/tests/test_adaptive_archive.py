"""Adaptive-memory archiving and pruning (R25), on pg0 with the ``memory`` fixture.

The sweep is driven through the clock seam (every ``now`` is passed in, or ``adaptive.clock`` is
patched for the engine job), over retention rows seeded with a known stability and anchor so that
retrievability ``R = exp(-days / S)`` is exact: with ``S = 10`` days and the 0.05 threshold, R
crosses at ~29.96 days.

Seeding uses raw SQL (exact texts, anchors and stabilities), so the module is
``memory_backend_incompatible``. Assertions go through the store's methods, the engine job,
``recall_async`` and the lifecycle view.
"""

import asyncio
import dataclasses
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api import config as config_module
from hindsight_api.config import get_config
from hindsight_api.engine.adaptive import clock
from hindsight_api.engine.adaptive.evidence import reinforcement_event
from hindsight_api.engine.adaptive.models import TruthStatus
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.retain import embedding_utils

pytestmark = pytest.mark.memory_backend_incompatible

T0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
STABILITY = 10.0
#: First day R is below the 0.05 threshold for S = 10 (crossing at 29.96 days).
BELOW = 30


def day(n: float) -> datetime:
    return T0 + timedelta(days=n)


def _bank() -> str:
    return f"test-archive-{uuid.uuid4().hex[:8]}"


def _settings(**overrides) -> AdaptiveSettings:
    base = AdaptiveSettings.resolve({"adaptive_memory_mode": "rank"}, get_config(), store_supports=True)
    values = {
        "archive_enabled": True,
        "archive_threshold": 0.05,
        "archive_grace_days": 30.0,
        "archive_protect_importance": 0.8,
        "reinforcement_enabled": True,
    }
    values.update(overrides)
    return dataclasses.replace(base, **values)


def _vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def _seed_units(memory, rc, bank_id: str, texts: list[str], *, fact_type: str = "world") -> list[str]:
    await memory.ensure_bank_profile(bank_id, request_context=rc)
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, texts)
    ids = [str(uuid.uuid4()) for _ in texts]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for unit_id, text, embedding in zip(ids, texts, embeddings):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, mentioned_at, consolidated_at) "
                "VALUES ($1, $2, $3, $4, $5::vector, $6, $6)",
                unit_id,
                bank_id,
                text,
                fact_type,
                _vector(embedding),
                T0,
            )
    return ids


async def _seed_retention(
    memory, bank_id: str, memory_id: str, *, importance: float = 0.5, exempt: bool = False, reason: str | None = None
) -> None:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_retention_state
                (memory_id, bank_id, importance, stability_days, last_reinforced_at, reinforcement_count,
                 forgetting_exempt, exemption_reason, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, 1, $6, $7, $5, $5)
            """,
            uuid.UUID(memory_id),
            bank_id,
            importance,
            STABILITY,
            T0,
            exempt,
            reason,
        )


async def _row(memory, memory_id: str):
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT retention_state, below_threshold_since, archived_at FROM memory_retention_state WHERE memory_id = $1",
            uuid.UUID(memory_id),
        )


async def _sweep(memory, bank_id: str, at: datetime, **overrides):
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        return await get_memories().archive_sweep(conn, bank_id=bank_id, now=at, settings=_settings(**overrides))


async def _retire_with_evidence(memory, bank_id: str, claim_id: str, evidence_id: str, *, status: str | None):
    """A claim citing ``evidence_id``; retired when ``status`` is given, else left active."""
    store = get_memories()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=[claim_id, evidence_id], skip_locked=False)
            await store.add_claim_evidence(conn, bank_id=bank_id, memory_id=claim_id, evidence_ids=[evidence_id])
            if status is not None:
                assert await store.cas_retire(
                    conn,
                    bank_id=bank_id,
                    memory_id=claim_id,
                    to_status=TruthStatus(status),
                    retired_via="explicit",
                    superseded_by_id=None,
                    valid_to=day(1),
                    reason=None,
                    confidence=None,
                )


# --------------------------------------------------------------------------- the sweep


@pytest.mark.asyncio
async def test_the_sweep_marks_waits_out_the_grace_period_then_archives(memory, request_context):
    bank_id = _bank()
    [m] = await _seed_units(memory, request_context, bank_id, ["The billing service runs Java 21."])
    await _seed_retention(memory, bank_id, m)

    early = await _sweep(memory, bank_id, day(20))
    assert (early.scanned, early.marked, early.archived) == (1, 0, 0), "R = 0.135 is above the threshold"
    assert (await _row(memory, m))["below_threshold_since"] is None

    marked = await _sweep(memory, bank_id, day(BELOW + 10))
    assert (marked.marked, marked.archived) == (1, 0)
    assert (await _row(memory, m))["below_threshold_since"] == day(BELOW + 10), "the sweep records when it saw it"

    waiting = await _sweep(memory, bank_id, day(BELOW + 39))
    assert (waiting.marked, waiting.archived) == (0, 0), "29 days below is inside the 30-day grace"

    archived = await _sweep(memory, bank_id, day(BELOW + 40))
    assert archived.archived == 1
    row = await _row(memory, m)
    assert (row["retention_state"], row["archived_at"]) == ("archived", day(BELOW + 40))

    again = await _sweep(memory, bank_id, day(BELOW + 41))
    assert again.scanned == 0, "archived rows are not swept again"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_reinforced_memory_loses_its_mark(memory, request_context):
    bank_id = _bank()
    [m] = await _seed_units(memory, request_context, bank_id, ["The billing service runs Java 21."])
    await _seed_retention(memory, bank_id, m)
    assert (await _sweep(memory, bank_id, day(BELOW + 5))).marked == 1

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        result = await get_memories().record_reinforcement(
            conn,
            bank_id=bank_id,
            events=[
                reinforcement_event(
                    memory_id=m,
                    source_type="explicit_confirmation",
                    source_key="explicit:confirm-1",
                    event_time=day(BELOW + 6),
                    settings=_settings(),
                )
            ],
            settings=_settings(),
            now=day(BELOW + 6),
            background=False,
        )
    assert result.applied == 1
    # Cleared by the reinforcement itself, not by a later sweep: on Oracle (no maintenance loop) or
    # with the loop interval at 0, the next sweep may only run once R is below the threshold again,
    # and the old mark would then count the time before this use toward the grace period.
    assert result.states[m].below_threshold_since is None
    assert (await _row(memory, m))["below_threshold_since"] is None

    after = await _sweep(memory, bank_id, day(BELOW + 7))
    assert (after.cleared, after.archived) == (0, 0)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_protected_memories_are_never_archived(memory, request_context):
    """Importance at or above the protection level, an exemption (a hold included), and evidence of a
    claim that is not active all keep a memory out of the archive; evidence of an active claim
    does not."""
    bank_id = _bank()
    important, held, evidence_retired, evidence_active, claim_retired, claim_active, plain = await _seed_units(
        memory, request_context, bank_id, [f"fact {i}" for i in range(7)]
    )
    await _seed_retention(memory, bank_id, important, importance=0.9)
    await _seed_retention(memory, bank_id, held, exempt=True, reason="hold")
    for m in (evidence_retired, evidence_active, plain):
        await _seed_retention(memory, bank_id, m)
    await _retire_with_evidence(memory, bank_id, claim_retired, evidence_retired, status="superseded")
    await _retire_with_evidence(memory, bank_id, claim_active, evidence_active, status=None)

    first = await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)
    assert first.archived == 2
    assert {
        m: (await _row(memory, m))["retention_state"] for m in (important, held, evidence_retired)
    } == dict.fromkeys((important, held, evidence_retired), "active")
    assert (await _row(memory, evidence_active))["retention_state"] == "archived"
    assert (await _row(memory, plain))["retention_state"] == "archived"
    assert (await _row(memory, held))["below_threshold_since"] is None, "an exempt memory is always retrievable"
    assert first.protected == 2, "the important memory and the retired claim's evidence"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_lazily_defaulted_memories_and_other_banks_are_never_swept(memory, request_context):
    bank_id, other = _bank(), _bank()
    [lazy, stored] = await _seed_units(memory, request_context, bank_id, ["no row", "has a row"])
    [foreign] = await _seed_units(memory, request_context, other, ["another bank"])
    await _seed_retention(memory, bank_id, stored)
    await _seed_retention(memory, other, foreign)

    result = await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)
    assert (result.scanned, result.archived) == (1, 1)
    assert await _row(memory, lazy) is None, "a memory on its lazy defaults has no row to archive"
    assert (await _row(memory, foreign))["retention_state"] == "active"
    await memory.delete_bank(bank_id, request_context=request_context)
    await memory.delete_bank(other, request_context=request_context)


@pytest.mark.asyncio
async def test_the_sweep_pages_through_every_row(memory, request_context):
    bank_id = _bank()
    ids = await _seed_units(memory, request_context, bank_id, [f"fact {i}" for i in range(5)])
    for m in ids:
        await _seed_retention(memory, bank_id, m)
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        result = await get_memories().archive_sweep(
            conn, bank_id=bank_id, now=day(BELOW + 1), settings=_settings(archive_grace_days=0.0), batch=2
        )
    assert (result.scanned, result.archived) == (5, 5)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_row_another_transaction_holds_is_skipped_not_waited_on(memory, request_context):
    bank_id = _bank()
    held, free = await _seed_units(memory, request_context, bank_id, ["held", "free"])
    for m in (held, free):
        await _seed_retention(memory, bank_id, m)
    pool = await memory._get_pool()
    async with pool.acquire() as holder:
        async with holder.transaction():
            await holder.execute(
                "SELECT 1 FROM memory_retention_state WHERE memory_id = $1 FOR UPDATE", uuid.UUID(held)
            )
            result = await asyncio.wait_for(_sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0), timeout=30)
    assert (result.archived, result.skipped_locked) == (1, 1)
    assert (await _row(memory, held))["retention_state"] == "active"
    await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- unarchive


@pytest.mark.asyncio
async def test_an_applied_reinforcement_unarchives_and_a_cooldown_does_not(memory, request_context):
    bank_id = _bank()
    [m] = await _seed_units(memory, request_context, bank_id, ["The billing service runs Java 21."])
    await _seed_retention(memory, bank_id, m)
    assert (await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)).archived == 1
    store = get_memories()
    settings = _settings()

    async def _confirm(key: str, at: datetime):
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            return await store.record_reinforcement(
                conn,
                bank_id=bank_id,
                events=[
                    reinforcement_event(
                        memory_id=m,
                        source_type="explicit_confirmation",
                        source_key=key,
                        event_time=at,
                        settings=settings,
                    )
                ],
                settings=settings,
                now=day(BELOW + 2),
                background=False,
            )

    # Within the cooldown of the last reinforcement (T0): no gain, so no new evidence of use.
    cooled = await _confirm("explicit:early", T0 + timedelta(hours=1))
    assert cooled.events[0].outcome == "cooldown"
    assert (await _row(memory, m))["retention_state"] == "archived"

    applied = await _confirm("explicit:late", day(BELOW + 2))
    assert applied.events[0].outcome == "applied"
    assert applied.states[m].retention_state == "active"
    row = await _row(memory, m)
    assert (row["retention_state"], row["archived_at"], row["below_threshold_since"]) == ("active", None, None)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.fixture
def archive_config(monkeypatch):
    """Turn the static archive (and optionally prune) switches on for the engine paths."""

    def _set(**fields):
        raw = config_module._get_raw_config()
        values = {"adaptive_memory_archive_enabled": True, "adaptive_memory_archive_grace_days": 0.0}
        values.update(fields)
        monkeypatch.setattr(config_module, "_config_cache", dataclasses.replace(raw, **values))

    return _set


@pytest.mark.asyncio
async def test_patch_retention_state_active_unarchives_and_hold_protects(memory, request_context, archive_config):
    archive_config()
    bank_id = _bank()
    [m] = await _seed_units(memory, request_context, bank_id, ["The billing service runs Java 21."])
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})
    await _seed_retention(memory, bank_id, m)
    assert (await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)).archived == 1

    view = await memory.update_memory_lifecycle(bank_id, m, retention_state="active", request_context=request_context)
    assert view is not None and view.retention.retention_state == "active"

    # A hold (like any protection) placed on an archived memory brings it back: the sweep would not
    # archive it again, and recall must not keep suppressing it.
    assert (await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)).archived == 1
    held = await memory.update_memory_lifecycle(bank_id, m, hold=True, request_context=request_context)
    assert (held.retention.forgetting_exempt, held.retention.exemption_reason) == (True, "hold")
    assert held.retention.retention_state == "active"
    assert (await _row(memory, m))["archived_at"] is None
    assert (await _sweep(memory, bank_id, day(BELOW + 100), archive_grace_days=0.0)).archived == 0

    with pytest.raises(ValueError):
        await memory.update_memory_lifecycle(
            bank_id, m, hold=True, forgetting_exempt=True, request_context=request_context
        )
    released = await memory.update_memory_lifecycle(bank_id, m, hold=False, request_context=request_context)
    assert (released.retention.forgetting_exempt, released.retention.exemption_reason) == (False, None)
    await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- recall


@pytest.mark.asyncio
async def test_recall_drops_an_archived_memory_unless_the_cue_is_strong(memory, request_context, archive_config):
    archive_config()
    bank_id = _bank()
    archived_text = "The vault recovery passphrase is kept in the blue binder on the third shelf."
    [archived, other] = await _seed_units(
        memory, request_context, bank_id, [archived_text, "The cafeteria serves lasagna on Wednesdays."]
    )
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})
    await _seed_retention(memory, bank_id, archived)
    assert (await _sweep(memory, bank_id, day(BELOW + 1), archive_grace_days=0.0)).archived == 1

    async def _recall(query: str):
        return await memory.recall_async(
            bank_id=bank_id,
            query=query,
            question_date=day(BELOW + 2),
            request_context=request_context,
            fact_type=["world"],
            budget=Budget.MID,
            max_tokens=4096,
            enable_trace=True,
        )

    strong = await _recall("Where is the vault recovery passphrase kept?")
    assert archived in [str(r.id) for r in strong.results], "a strong cue still brings an archived memory back"
    assert "archive_suppressed" not in strong.trace["adaptive"]

    weak = await _recall("What does the cafeteria serve on Wednesdays?")
    assert archived not in [str(r.id) for r in weak.results]
    assert other in [str(r.id) for r in weak.results]
    assert weak.trace["adaptive"]["archive_suppressed"] == [archived]

    # An exempt row is never suppressed, even one archived before the exemption was set.
    await store_hold(memory, bank_id, archived)
    held = await _recall("What does the cafeteria serve on Wednesdays?")
    assert "archive_suppressed" not in held.trace["adaptive"]

    # Archiving off: the archived row is an ordinary candidate again.
    archive_config(adaptive_memory_archive_enabled=False)
    plain = await _recall("What does the cafeteria serve on Wednesdays?")
    assert "archive_suppressed" not in plain.trace["adaptive"]
    await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- the engine job + prune


@pytest.mark.asyncio
async def test_the_job_does_nothing_unless_archiving_is_on_and_the_mode_ranks(memory, request_context, archive_config):
    bank_id = _bank()
    [m] = await _seed_units(memory, request_context, bank_id, ["fact"])
    await _seed_retention(memory, bank_id, m)

    archive_config(adaptive_memory_archive_enabled=False)
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})
    assert not (await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)).ran
    assert (await memory.submit_async_adaptive_memory_maintenance(bank_id, request_context=request_context))["no_work"]

    archive_config()
    for mode in ("off", "observe"):
        await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": mode})
        assert not (await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)).ran
    assert (await _row(memory, m))["retention_state"] == "active"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_prune_is_off_by_default_and_deletes_only_eligible_archived_memories(
    memory, request_context, archive_config, monkeypatch
):
    bank_id = _bank()
    texts = ["plain", "claim evidence", "held", "source of a current observation", "successor", "retired claim"]
    plain, evidence, held, source, successor, retired = await _seed_units(memory, request_context, bank_id, texts)
    [observation] = await _seed_units(memory, request_context, bank_id, ["an observation"], fact_type="observation")
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_units SET source_memory_ids = $2, proof_count = 1 WHERE id = $1",
            uuid.UUID(observation),
            [uuid.UUID(source)],
        )
    for m in (plain, evidence, held, source, successor):
        await _seed_retention(memory, bank_id, m)
    await _retire_with_evidence(memory, bank_id, retired, evidence, status=None)
    store = get_memories()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=[retired, successor], skip_locked=False)
            assert await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=retired,
                to_status=TruthStatus.SUPERSEDED,
                retired_via="explicit",
                superseded_by_id=successor,
                valid_to=day(1),
                reason=None,
                confidence=None,
            )
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})

    # Archive everything once (prune off: the default).
    archive_config()
    monkeypatch.setattr(clock, "utcnow", lambda: day(BELOW + 1))
    first = await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)
    assert (first.ran, first.pruned) == (True, 0)
    # The retired claim's evidence is protected from archiving; its claim is not active.
    assert first.archive.archived == 4
    # An exemption written straight to the store leaves the row archived; it still keeps it from
    # being pruned. (The PATCH path also unarchives it.)
    await store_hold(memory, bank_id, held)

    archive_config(adaptive_memory_prune_enabled=True, adaptive_memory_prune_after_days=10.0)
    monkeypatch.setattr(clock, "utcnow", lambda: day(BELOW + 5))
    too_soon = await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)
    assert too_soon.pruned == 0, "archived 4 days ago, prune_after_days is 10"

    monkeypatch.setattr(clock, "utcnow", lambda: day(BELOW + 12))
    pruned = await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)
    assert pruned.pruned == 1
    async with pool.acquire() as conn:
        live = {str(r["id"]) for r in await conn.fetch("SELECT id FROM memory_units WHERE bank_id = $1", bank_id)}
    assert plain not in live
    assert {evidence, held, source, successor, observation, retired} <= live
    await memory.delete_bank(bank_id, request_context=request_context)


async def store_hold(memory, bank_id: str, memory_id: str) -> None:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await get_memories().set_retention_policy(
            conn,
            bank_id=bank_id,
            memory_id=memory_id,
            forgetting_exempt=True,
            exemption_reason="hold",
            settings=_settings(),
            now=day(BELOW + 2),
        )


async def _archive_at(memory, bank_id: str, archived: dict[str, datetime]) -> None:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for memory_id, at in archived.items():
            await conn.execute(
                "UPDATE memory_retention_state SET retention_state = 'archived', archived_at = $3 "
                "WHERE bank_id = $1 AND memory_id = $2",
                bank_id,
                uuid.UUID(memory_id),
                at,
            )


@pytest.mark.asyncio
async def test_protected_rows_at_the_head_of_the_archive_do_not_stall_prune(
    memory, request_context, archive_config, monkeypatch
):
    """Raw facts behind a current observation are archived first and are never deletable. A page
    made only of them must not end the scan: the cursor moves on to the deletable rows behind."""
    from hindsight_api.engine import memory_engine as engine_module

    bank_id = _bank()
    sources = await _seed_units(memory, request_context, bank_id, ["source one", "source two", "source three"])
    [plain] = await _seed_units(memory, request_context, bank_id, ["plain"])
    [observation] = await _seed_units(memory, request_context, bank_id, ["an observation"], fact_type="observation")
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_units SET source_memory_ids = $2, proof_count = 3 WHERE id = $1",
            uuid.UUID(observation),
            [uuid.UUID(m) for m in sources],
        )
    for m in (*sources, plain):
        await _seed_retention(memory, bank_id, m)
    await _archive_at(memory, bank_id, {**{m: day(1) for m in sources}, plain: day(2)})
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})
    archive_config(adaptive_memory_prune_enabled=True, adaptive_memory_prune_after_days=1.0)
    monkeypatch.setattr(engine_module, "_PRUNE_PAGE_SIZE", 2)
    monkeypatch.setattr(clock, "utcnow", lambda: day(BELOW + 1))

    result = await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)

    assert result.pruned == 1
    async with pool.acquire() as conn:
        live = {str(r["id"]) for r in await conn.fetch("SELECT id FROM memory_units WHERE bank_id = $1", bank_id)}
    assert plain not in live
    assert {*sources, observation} <= live
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_prune_rechecks_a_page_under_row_locks_before_deleting(
    memory, request_context, archive_config, monkeypatch
):
    """A memory that got a hold, or was unarchived, after its page was read is not deleted."""
    bank_id = _bank()
    held, revived, plain = await _seed_units(memory, request_context, bank_id, ["held", "revived", "plain"])
    for m in (held, revived, plain):
        await _seed_retention(memory, bank_id, m)
    await _archive_at(memory, bank_id, {held: day(1), revived: day(1), plain: day(1)})
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "rank"})
    archive_config(adaptive_memory_prune_enabled=True, adaptive_memory_prune_after_days=1.0)
    monkeypatch.setattr(clock, "utcnow", lambda: day(BELOW + 1))
    store = get_memories()
    read_page = store.prune_candidates

    async def page_then_race(conn, **kwargs):
        page = await read_page(conn, **kwargs)
        assert set(page.memory_ids) == {held, revived, plain}
        # Lands between the page read and the delete transaction.
        await store_hold(memory, bank_id, held)
        async with (await memory._get_pool()).acquire() as other:
            await store.unarchive_memories(other, bank_id=bank_id, memory_ids=[revived], now=day(BELOW + 1))
        return page

    monkeypatch.setattr(store, "prune_candidates", page_then_race)
    result = await memory.run_adaptive_memory_maintenance(bank_id, request_context=request_context)

    assert result.pruned == 1
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        live = {str(r["id"]) for r in await conn.fetch("SELECT id FROM memory_units WHERE bank_id = $1", bank_id)}
    assert plain not in live
    assert {held, revived} <= live
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_consolidation_submits_the_job_and_the_loop_job_is_off_in_tests(memory, request_context, monkeypatch):
    from unittest.mock import AsyncMock

    from hindsight_api.engine import consolidation as consolidation_module
    from hindsight_api.engine.maintenance import MaintenanceLoop

    assert get_config().adaptive_memory_maintenance_interval_seconds == 0, "conftest switches the loop job off"
    assert not MaintenanceLoop._any_job_enabled()

    monkeypatch.setattr(consolidation_module, "run_consolidation_job", AsyncMock(return_value={}))
    submit = AsyncMock(return_value={"operation_id": None, "no_work": True})
    monkeypatch.setattr(memory, "submit_async_adaptive_memory_maintenance", submit)
    await memory._handle_consolidation({"bank_id": "some-bank"})
    assert submit.await_args.kwargs["bank_id"] == "some-bank"


def test_archive_metrics_export_under_their_exact_prometheus_names():
    from types import SimpleNamespace
    from unittest.mock import patch

    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from prometheus_client import CollectorRegistry, generate_latest

    from hindsight_api.metrics import MetricsCollector

    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    provider = MeterProvider(metric_readers=[reader])
    cfg = SimpleNamespace(
        metrics_include_bank_id=False,
        metrics_include_tenant=False,
        recall_diagnostic_phases=True,
        recall_phase_sample_every=1,
    )
    try:
        with (
            patch("hindsight_api.metrics.get_meter", return_value=provider.get_meter("test-archive")),
            patch("hindsight_api.config.get_config", return_value=cfg),
        ):
            collector = MetricsCollector()
        collector.record_adaptive_archive_suppressed(bank_id="b", count=2)
        collector.record_adaptive_archive_suppressed(bank_id="b", count=0)
        collector.record_adaptive_archive_maintenance(bank_id="b", archived=3, pruned=1)
        text = generate_latest(registry).decode()
    finally:
        provider.shutdown()

    def value(name: str) -> str:
        line = next(line for line in text.splitlines() if line.split("{")[0].split(" ")[0] == name)
        return line.rsplit(" ", 1)[1]

    assert value("hindsight_adaptive_memory_archive_suppressed_total") == "2.0"
    assert value("hindsight_adaptive_memory_archived_total") == "3.0"
    assert value("hindsight_adaptive_memory_pruned_total") == "1.0"


# --------------------------------------------------------------------------- Oracle statements (no database)


class _RecordingOracleConn:
    """Enough of an Oracle connection to run the maintenance statements and capture their SQL."""

    backend_type = "oracle"

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.statements: list[str] = []
        self._paged = False

    def transaction(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _tx():
            yield self

        return _tx()

    async def fetch(self, query: str, *args):
        self.statements.append(query)
        if "FROM memory_retention_state" in query and "ORDER BY last_reinforced_at" in query:
            if self._paged:
                return []
            self._paged = True
            return self.rows
        if "FOR UPDATE SKIP LOCKED" in query:
            return self.rows
        if "FROM memory_retention_state r" in query and "archived" in query:
            return [{"memory_id": row["memory_id"], "forgetting_exempt": 0, "archived_at": day(1)} for row in self.rows]
        return []

    async def execute(self, query: str, *args):
        self.statements.append(query)
        return "UPDATE 0"


def _oracle_row(memory_id: uuid.UUID) -> dict:
    return {
        "memory_id": memory_id,
        "importance": 0.5,
        "stability_days": STABILITY,
        "last_reinforced_at": T0,
        "reinforcement_count": 1,
        "access_count": 0,
        "forgetting_exempt": 0,  # NUMBER(1) on Oracle
        "retention_state": "active",
        "last_recalled_at": None,
        "below_threshold_since": day(1),
        "archived_at": None,
    }


@pytest.mark.asyncio
async def test_the_maintenance_statements_speak_oracle_after_the_rewriter():
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle
    from hindsight_api.engine.memories.pg import adaptive_maintenance

    def fq(name: str) -> str:
        return name

    ids = [uuid.uuid4(), uuid.uuid4()]
    conn = _RecordingOracleConn([_oracle_row(i) for i in ids])
    settings = _settings(archive_grace_days=0.0, prune_after_days=0.0)

    result = await adaptive_maintenance.archive_sweep(
        conn, fq_table=fq, bank_id="b", now=day(BELOW + 1), settings=settings
    )
    assert (result.scanned, result.archived) == (2, 2)
    await adaptive_maintenance.grounding_status(conn, fq_table=fq, bank_id="b", ids=[str(i) for i in ids])
    page = await adaptive_maintenance.prune_candidates(
        conn, fq_table=fq, bank_id="b", now=day(BELOW + 1), settings=settings, limit=2
    )
    assert page.next_after is not None, "a full page continues"
    await adaptive_maintenance.prune_candidates(
        conn, fq_table=fq, bank_id="b", now=day(BELOW + 1), settings=settings, after=page.next_after
    )
    assert await adaptive_maintenance.lock_prunable(
        conn, fq_table=fq, bank_id="b", memory_ids=ids, now=day(BELOW + 1), settings=settings
    ) == sorted(str(i) for i in ids)
    await adaptive_maintenance.banks_with_retention_state(conn, fq_table=fq)
    conn.rows = [{"memory_id": i} for i in ids]
    await adaptive_maintenance.unarchive_memories(conn, fq_table=fq, bank_id="b", memory_ids=ids, now=day(1))

    rewritten = [_rewrite_pg_to_oracle(s).query for s in conn.statements]
    for query in rewritten:
        assert "$" not in query and "::" not in query and "LIMIT" not in query, query
    assert any("FETCH FIRST 500 ROWS ONLY" in q and "ORDER BY last_reinforced_at, memory_id" in q for q in rewritten)
    assert any("FOR UPDATE SKIP LOCKED" in q and "IN (/*EXPAND:2*/)" in q for q in rewritten)
    assert any("LEFT JOIN memory_claim_state c ON c.memory_id = mu.id" in q for q in rewritten)
    assert not any("FALSE" in q or "TRUE" in q for q in rewritten), "no boolean literal: BOOLEAN vs NUMBER(1)"
    assert any(
        "r.archived_at > :3 OR (r.archived_at = :3 AND r.memory_id > :4)" in q and "FETCH FIRST 500 ROWS ONLY" in q
        for q in rewritten
    ), "the prune keyset"
    assert any(
        "r.retention_state = 'archived'" in q and "FOR UPDATE SKIP LOCKED" in q and "IN (/*EXPAND:3*/)" in q
        for q in rewritten
    ), "the prune re-check under row locks"
    # The prune's observation check goes through the junction table on Oracle.
    assert any("observation_sources" in q for q in rewritten)
