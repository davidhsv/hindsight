"""The claim-lifecycle store on PostgreSQL (pg0): locking, CAS, evidence, relations, inheritance,
the requeue invariant, delete repair, re-ingest carry-forward, successor re-linking and transfer.

What the consolidation and recall tests cannot show on their own: that ``lock_claims`` + CAS lets
exactly one of two racing writers retire a claim, without deadlock; that every way of requeuing a
fact leaves a superseded or refuted one consolidated; that every physical delete of raw facts
reverts exactly the claims that only those facts justified (and that invalidation does not); that
an unchanged re-ingest keeps a supersession whose evidence lived on the re-extracted facts; and that
a transfer carries the state across fresh ids.

Seeding mixes the engine (retain with the mock LLM, so documents, chunks and the delete / re-ingest
paths are the real ones) with raw SQL for observations, whose builder (consolidation) is not under
test here. Claims are written through the store interface the consolidation and PATCH writers use.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.adaptive import clock
from hindsight_api.engine.adaptive.claims import REVERTED_EVIDENCE_DELETED, reingest_text_key
from hindsight_api.engine.adaptive.models import TruthStatus
from hindsight_api.engine.db.ops import LifecycleRelationRow
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.retain import embedding_utils

pytestmark = pytest.mark.memory_backend_incompatible

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)
NOW = DAY0 + timedelta(days=400)
SUPERSEDED = TruthStatus.SUPERSEDED
REFUTED = TruthStatus.REFUTED
ACTIVE = TruthStatus.ACTIVE


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(clock, "utcnow", lambda: NOW)


@pytest.fixture(autouse=True)
def _observations_off(monkeypatch):
    """Retain without the inline consolidation it would otherwise run (the mock would build an
    observation per fact, which changes which sources other observations still cite)."""
    monkeypatch.setattr(_get_raw_config(), "enable_observations", False)


@asynccontextmanager
async def _conn(memory):
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        yield conn


@asynccontextmanager
async def _tx(memory):
    async with _conn(memory) as conn:
        async with conn.transaction():
            yield conn


async def _fetch(memory, sql: str, *args):
    async with _conn(memory) as conn:
        return await conn.fetch(sql, *args)


async def _bank(memory, request_context) -> str:
    bank_id = f"test-acs-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    return bank_id


async def _seed(
    memory,
    bank_id: str,
    n: int = 1,
    *,
    fact_type: str = "world",
    tags: list[str] | None = None,
    occurred_start: datetime | None = None,
    consolidated: bool = True,
    text: str | None = None,
) -> list[str]:
    ids = [str(uuid.uuid4()) for _ in range(n)]
    async with _conn(memory) as conn:
        for index, memory_id in enumerate(ids):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, occurred_start, mentioned_at, "
                "consolidated_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                uuid.UUID(memory_id),
                bank_id,
                text or f"Seeded fact {index} about the payments service ({memory_id[:8]}).",
                fact_type,
                tags or [],
                occurred_start,
                DAY0,
                day(1) if consolidated else None,
            )
    return ids


async def _observation(memory, bank_id: str, sources: list[str], *, tags: list[str] | None = None, text=None) -> str:
    (obs_id,) = await _seed(memory, bank_id, fact_type="observation", tags=tags, consolidated=False, text=text)
    async with _conn(memory) as conn:
        await conn.execute(
            "UPDATE memory_units SET source_memory_ids = $2 WHERE id = $1",
            uuid.UUID(obs_id),
            [uuid.UUID(s) for s in sources],
        )
    return obs_id


async def _claim(memory, memory_id: str):
    rows = await _fetch(memory, "SELECT * FROM memory_claim_state WHERE memory_id = $1", uuid.UUID(memory_id))
    return rows[0] if rows else None


async def _status(memory, memory_id: str) -> str | None:
    claim = await _claim(memory, memory_id)
    return None if claim is None else claim["truth_status"]


async def _evidence_of(memory, memory_id: str) -> list[str]:
    rows = await _fetch(
        memory,
        "SELECT evidence_memory_id FROM memory_claim_evidence WHERE memory_id = $1 ORDER BY evidence_memory_id",
        uuid.UUID(memory_id),
    )
    return [str(r["evidence_memory_id"]) for r in rows]


async def _consolidated_at(memory, memory_id: str):
    rows = await _fetch(memory, "SELECT consolidated_at FROM memory_units WHERE id = $1", uuid.UUID(memory_id))
    return rows[0]["consolidated_at"] if rows else "missing"


async def _retire(
    memory,
    bank_id: str,
    *,
    target: str,
    successor: str | None,
    evidence: list[str],
    status: TruthStatus = SUPERSEDED,
    retired_via: str = "supersedes",
    inherit: bool = False,
    at: datetime = day(10),
) -> list[str]:
    """Retire ``target`` the way the consolidation writer does: one lock pass, CAS, relation,
    evidence, then raw inheritance. Returns the inherited ids."""
    store = get_memories()
    async with _tx(memory) as conn:
        candidates: list[str] = []
        if inherit:
            candidates = (await store.observation_sources(conn, bank_id=bank_id, observation_ids=[target])).get(
                target, []
            )
        ids = [target, *evidence, *candidates, *([successor] if successor else [])]
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=ids, skip_locked=False)
        if successor:
            await store.set_claim(conn, bank_id=bank_id, memory_id=successor, truth_status=ACTIVE, valid_from=at)
        assert await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=target,
            to_status=status,
            retired_via=retired_via,
            superseded_by_id=successor,
            valid_to=at,
            reason="test",
            confidence=None,
        )
        if successor:
            await store.insert_relation(
                conn,
                bank_id=bank_id,
                relation_type="supersedes" if status is SUPERSEDED else "refutes",
                source_id=successor,
                target_id=target,
                origin="consolidation",
                reason=None,
                confidence=None,
                evidence={"fact_ids": evidence},
            )
        await store.add_claim_evidence(conn, bank_id=bank_id, memory_id=target, evidence_ids=evidence)
        if not inherit:
            return []
        return await store.inherit_from_observation(
            conn,
            bank_id=bank_id,
            observation_id=target,
            to_status=status,
            successor_id=successor,
            valid_to=at,
            evidence_ids=evidence,
            candidate_source_ids=None,
        )


# =========================================================================== locking and CAS


@pytest.mark.asyncio
async def test_lock_claims_creates_default_rows_for_live_memories_of_the_bank_only(memory, request_context):
    bank_id = await _bank(memory, request_context)
    other = await _bank(memory, request_context)
    mine = await _seed(memory, bank_id, 2)
    (foreign,) = await _seed(memory, other)

    async with _tx(memory) as conn:
        locked = await get_memories().lock_claims(
            conn,
            bank_id=bank_id,
            memory_ids=[mine[1], foreign, "not-a-uuid", str(uuid.uuid4()), mine[0]],
            skip_locked=False,
        )

    assert locked == set(mine)
    for memory_id in mine:
        claim = await _claim(memory, memory_id)
        assert claim["truth_status"] == "active" and claim["created_at"] == NOW and claim["valid_from"] is None
    assert await _claim(memory, foreign) is None


@pytest.mark.asyncio
async def test_cas_retire_wins_once_and_a_restore_reopens_it(memory, request_context):
    bank_id = await _bank(memory, request_context)
    target, successor = await _seed(memory, bank_id, 2)
    store = get_memories()

    async with _tx(memory) as conn:
        # Absent row: lock_claims creates it, so the first CAS wins.
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=[target, successor], skip_locked=False)
        first = await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=target,
            to_status=SUPERSEDED,
            retired_via="supersedes",
            superseded_by_id=successor,
            valid_to=day(5),
            reason="newer",
            confidence=0.9,
        )
        second = await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=target,
            to_status=REFUTED,
            retired_via="refutes",
            superseded_by_id=None,
            valid_to=day(6),
            reason="again",
            confidence=None,
        )
    assert (first, second) == (True, False)
    claim = await _claim(memory, target)
    assert (claim["truth_status"], claim["retired_via"], str(claim["superseded_by_id"])) == (
        "superseded",
        "supersedes",
        successor,
    )
    assert claim["valid_to"] == day(5) and claim["reason"] == "newer" and claim["confidence"] == pytest.approx(0.9)

    async with _tx(memory) as conn:
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=[target], skip_locked=False)
        await store.set_claim(conn, bank_id=bank_id, memory_id=target, truth_status=ACTIVE, reason="restored")
        # Present row, active again: the CAS wins; a pointer to a memory that is not live is NULL.
        assert await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=target,
            to_status=REFUTED,
            retired_via="explicit",
            superseded_by_id=str(uuid.uuid4()),
            valid_to=None,
            reason="wrong",
            confidence=None,
        )
        with pytest.raises(ValueError):
            await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=target,
                to_status=ACTIVE,
                retired_via="explicit",
                superseded_by_id=None,
                valid_to=None,
                reason=None,
                confidence=None,
            )
    claim = await _claim(memory, target)
    assert (claim["truth_status"], claim["superseded_by_id"], claim["valid_to"]) == ("refuted", None, None)


@pytest.mark.asyncio
async def test_a_skip_locked_pass_skips_a_memory_being_deleted(memory, request_context):
    bank_id = await _bank(memory, request_context)
    held, free = await _seed(memory, bank_id, 2)
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as deleter:
        async with deleter.transaction():
            await deleter.fetch("SELECT id FROM memory_units WHERE id = $1 FOR UPDATE", uuid.UUID(held))
            async with _tx(memory) as conn:
                locked = await asyncio.wait_for(
                    get_memories().lock_claims(conn, bank_id=bank_id, memory_ids=[held, free], skip_locked=True),
                    timeout=10,
                )
    assert locked == {free}


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(20))
async def test_consolidation_and_patch_race_for_one_claim_and_exactly_one_wins(memory, request_context, iteration):
    """Both writers lock everything they touch in one sorted pass first, so whichever input order
    they gather ids in, they serialize on the claim rows: one CAS wins, the other loses, no deadlock."""
    bank_id = await _bank(memory, request_context)
    target, successor, cited, other = await _seed(memory, bank_id, 4)
    store = get_memories()
    gate = asyncio.Event()

    async def consolidation() -> bool:
        async with _tx(memory) as conn:
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=[successor, cited, target], skip_locked=False)
            gate.set()
            await asyncio.sleep(0.01)
            return await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=target,
                to_status=SUPERSEDED,
                retired_via="supersedes",
                superseded_by_id=successor,
                valid_to=day(9),
                reason="consolidation",
                confidence=None,
            )

    async def patch() -> bool:
        if iteration % 2:
            await gate.wait()
        async with _tx(memory) as conn:
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=[target, other], skip_locked=False)
            return await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=target,
                to_status=REFUTED,
                retired_via="explicit",
                superseded_by_id=None,
                valid_to=day(9),
                reason="patch",
                confidence=None,
            )

    results = await asyncio.wait_for(asyncio.gather(consolidation(), patch()), timeout=30)
    assert sorted(results) == [False, True]
    winner = "superseded" if results[0] else "refuted"
    assert await _status(memory, target) == winner


# =========================================================================== evidence and relations


@pytest.mark.asyncio
async def test_claim_evidence_is_recorded_once_and_only_for_live_facts(memory, request_context):
    bank_id = await _bank(memory, request_context)
    holder, e1, e2 = await _seed(memory, bank_id, 3)
    store = get_memories()
    async with _tx(memory) as conn:
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=[holder], skip_locked=False)
        first = await store.add_claim_evidence(
            conn, bank_id=bank_id, memory_id=holder, evidence_ids=[e1, e2, str(uuid.uuid4()), holder]
        )
        again = await store.add_claim_evidence(conn, bank_id=bank_id, memory_id=holder, evidence_ids=[e1])
    assert (first, again) == (2, 0)
    assert await _evidence_of(memory, holder) == sorted([e1, e2])


@pytest.mark.asyncio
async def test_a_relation_snapshots_both_endpoints_and_is_written_once(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (fact,) = await _seed(memory, bank_id, tags=["team:payments"], text="We run Java 21 in production.")
    obs = await _observation(memory, bank_id, [fact], tags=["team:payments"], text="The service runs Java 21.")
    store = get_memories()
    async with _tx(memory) as conn:
        kwargs = dict(bank_id=bank_id, relation_type="refines", origin="explicit", reason="r", confidence=0.5)
        rid = await store.insert_relation(conn, source_id=obs, target_id=fact, evidence={"fact_ids": [fact]}, **kwargs)
        duplicate = await store.insert_relation(conn, source_id=obs, target_id=fact, evidence={}, **kwargs)
        tombstone = await store.insert_relation(conn, source_id=obs, target_id=str(uuid.uuid4()), evidence={}, **kwargs)
        nothing = await store.insert_relation(conn, source_id=None, target_id=str(uuid.uuid4()), evidence={}, **kwargs)
    assert rid is not None and duplicate is None and tombstone is not None and nothing is None
    (row,) = await _fetch(memory, "SELECT * FROM memory_lifecycle_relations WHERE id = $1", uuid.UUID(rid))
    source_snapshot = (
        json.loads(row["source_snapshot"]) if isinstance(row["source_snapshot"], str) else row["source_snapshot"]
    )
    target_snapshot = (
        json.loads(row["target_snapshot"]) if isinstance(row["target_snapshot"], str) else row["target_snapshot"]
    )
    assert source_snapshot == {
        "text": "The service runs Java 21.",
        "fact_type": "observation",
        "tags": ["team:payments"],
        "source_memory_ids": [fact],
    }
    assert target_snapshot["text"] == "We run Java 21 in production." and target_snapshot["source_memory_ids"] == []
    (row,) = await _fetch(memory, "SELECT * FROM memory_lifecycle_relations WHERE id = $1", uuid.UUID(tombstone))
    assert row["target_memory_id"] is None


# =========================================================================== raw inheritance


@pytest.mark.asyncio
async def test_inheritance_retires_only_the_sources_no_live_observation_stands_on(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (wider,) = await _seed(memory, bank_id, tags=["a", "b"], consolidated=False)  # a superset of the scope
    (experience,) = await _seed(memory, bank_id, fact_type="experience", tags=["a"])
    (dated,) = await _seed(memory, bank_id, tags=["a"], occurred_start=day(3))
    (narrower,) = await _seed(memory, bank_id, tags=[])  # not in the observation's scope
    (shared_live,) = await _seed(memory, bank_id, tags=["a"])
    (shared_retired,) = await _seed(memory, bank_id, tags=["a"])
    (cited,) = await _seed(memory, bank_id, tags=["a"])
    sources = [wider, experience, dated, narrower, shared_live, shared_retired]
    target = await _observation(memory, bank_id, sources, tags=["a"])
    successor = await _observation(memory, bank_id, [cited], tags=["a"])
    await _observation(memory, bank_id, [shared_live], tags=["a"])  # still active: keeps its source current
    retired_other = await _observation(memory, bank_id, [shared_retired], tags=["a"])
    await _retire(memory, bank_id, target=retired_other, successor=None, evidence=[cited])

    inherited = await _retire(memory, bank_id, target=target, successor=successor, evidence=[cited], inherit=True)

    assert sorted(inherited) == sorted([wider, shared_retired])
    for fact in (wider, shared_retired):
        claim = await _claim(memory, fact)
        assert claim["truth_status"] == "superseded" and claim["retired_via"] == "inherited"
        assert str(claim["inherited_from_id"]) == target and str(claim["superseded_by_id"]) == successor
        assert claim["valid_to"] == day(10)
        assert await _evidence_of(memory, fact) == [cited]
    assert await _consolidated_at(memory, wider) == NOW, "a retired raw fact is never left pending"
    for fact in (experience, dated, narrower, shared_live):
        assert await _status(memory, fact) == "active"


@pytest.mark.asyncio
async def test_an_untagged_shared_scope_observation_passes_its_status_to_tagged_sources(memory, request_context):
    """Coding-agents banks use ``shared`` scopes: observations untagged, facts with provenance tags.
    Every tag set is a superset of the empty one, so the stale raw fact inherits."""
    bank_id = await _bank(memory, request_context)
    (stale,) = await _seed(memory, bank_id, tags=["source:chat", "harness:claude-code"])
    (fresh,) = await _seed(memory, bank_id, tags=["source:chat", "harness:claude-code"])
    target = await _observation(memory, bank_id, [stale], tags=[])
    successor = await _observation(memory, bank_id, [fresh], tags=[])

    inherited = await _retire(memory, bank_id, target=target, successor=successor, evidence=[fresh], inherit=True)

    assert inherited == [stale]


@pytest.mark.asyncio
async def test_explicit_candidates_limit_inheritance_to_the_named_sources(memory, request_context):
    bank_id = await _bank(memory, request_context)
    stale, still_true, cited = await _seed(memory, bank_id, 3)
    (outsider,) = await _seed(memory, bank_id)
    target = await _observation(memory, bank_id, [stale, still_true])
    store = get_memories()
    async with _tx(memory) as conn:
        await store.lock_claims(
            conn, bank_id=bank_id, memory_ids=[target, stale, still_true, cited, outsider], skip_locked=False
        )
        await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=target,
            to_status=REFUTED,
            retired_via="refutes",
            superseded_by_id=None,
            valid_to=day(4),
            reason=None,
            confidence=None,
        )
        inherited = await store.inherit_from_observation(
            conn,
            bank_id=bank_id,
            observation_id=target,
            to_status=REFUTED,
            successor_id=None,
            valid_to=day(4),
            evidence_ids=[cited],
            candidate_source_ids=[stale, outsider],
        )
    assert inherited == [stale]
    assert await _status(memory, stale) == "refuted"
    assert await _status(memory, still_true) == "active"
    assert await _status(memory, outsider) == "active", "not a source of the observation"


@pytest.mark.asyncio
async def test_a_split_inheritance_ignores_the_facets_but_not_other_live_observations(memory, request_context):
    bank_id = await _bank(memory, request_context)
    general, shared, cited = await _seed(memory, bank_id, 3)
    parent = await _observation(memory, bank_id, [general, shared])
    facet = await _observation(memory, bank_id, [general, cited])
    other = await _observation(memory, bank_id, [shared])
    store = get_memories()
    async with _tx(memory) as conn:
        await store.lock_claims(
            conn, bank_id=bank_id, memory_ids=[parent, facet, other, general, shared, cited], skip_locked=False
        )
        await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=parent,
            to_status=SUPERSEDED,
            retired_via="splits",
            superseded_by_id=None,
            valid_to=day(4),
            reason=None,
            confidence=None,
        )
        inherited = await store.inherit_from_observation(
            conn,
            bank_id=bank_id,
            observation_id=parent,
            to_status=SUPERSEDED,
            successor_id=None,
            valid_to=day(4),
            evidence_ids=[cited],
            candidate_source_ids=[general, shared],
            retired_via="splits",
            ignored_citer_ids=[facet],
        )
    assert inherited == [general], "the facet citing it does not keep it current"
    claim = await _claim(memory, general)
    assert (claim["truth_status"], claim["retired_via"], claim["superseded_by_id"]) == ("superseded", "splits", None)
    assert (str(claim["inherited_from_id"]), claim["valid_to"]) == (parent, day(4))
    assert await _evidence_of(memory, general) == [cited]
    assert await _status(memory, shared) == "active", "another live observation still stands on it"


# =========================================================================== the requeue invariant


async def _retired_facts(memory, bank_id: str) -> SimpleNamespace:
    """An active fact, a superseded one and a refuted one, all consolidated."""
    active, superseded, refuted = await _seed(memory, bank_id, 3)
    (cited,) = await _seed(memory, bank_id)
    await _retire(memory, bank_id, target=superseded, successor=None, evidence=[cited])
    await _retire(
        memory, bank_id, target=refuted, successor=None, evidence=[cited], status=REFUTED, retired_via="refutes"
    )
    return SimpleNamespace(active=active, superseded=superseded, refuted=refuted)


@pytest.mark.asyncio
async def test_requeue_units_never_requeues_a_superseded_or_refuted_fact(memory, request_context):
    bank_id = await _bank(memory, request_context)
    seeded = await _retired_facts(memory, bank_id)
    active, superseded, refuted = seeded.active, seeded.superseded, seeded.refuted
    obs = await _observation(memory, bank_id, [active])
    async with _conn(memory) as conn:
        await conn.execute(
            "UPDATE memory_units SET consolidation_failed_at = $2 WHERE id = ANY($1::uuid[])",
            [uuid.UUID(active), uuid.UUID(superseded)],
            day(2),
        )

    async with _tx(memory) as conn:
        requeued = await get_memories().requeue_units(
            conn, bank_id=bank_id, unit_ids=[active, superseded, refuted, obs], reason="test", clear_failed=True
        )

    assert requeued == 1
    assert await _consolidated_at(memory, active) is None
    rows = await _fetch(memory, "SELECT consolidation_failed_at FROM memory_units WHERE id = $1", uuid.UUID(active))
    assert rows[0]["consolidation_failed_at"] is None
    for fact in (superseded, refuted):
        assert await _consolidated_at(memory, fact) == day(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ["sweep", "clear_observations", "retry_failed", "clear_for_memory", "edit", "mark_consolidated", "retag"],
)
async def test_every_requeue_path_leaves_a_retired_fact_consolidated(memory, request_context, path):
    bank_id = await _bank(memory, request_context)
    seeded = await _retired_facts(memory, bank_id)
    active, superseded, refuted = seeded.active, seeded.superseded, seeded.refuted
    (doomed,) = await _seed(memory, bank_id)
    store = get_memories()
    if path == "sweep":
        await _observation(memory, bank_id, [doomed, active, superseded, refuted])
        async with _tx(memory) as conn:
            await store.delete_stale_observations(
                conn=conn, ops=(await memory._get_backend()).ops, fq_table=None, bank_id=bank_id, fact_ids=[doomed]
            )
    elif path == "clear_observations":
        async with _tx(memory) as conn:
            await store.clear_observations_and_requeue(conn=conn, fq_table=None, bank_id=bank_id)
    elif path == "retry_failed":
        async with _conn(memory) as conn:
            await conn.execute(
                "UPDATE memory_units SET consolidated_at = NULL, consolidation_failed_at = $2 "
                "WHERE id = ANY($1::uuid[])",
                [uuid.UUID(m) for m in (active, superseded, refuted)],
                day(2),
            )
        async with _tx(memory) as conn:
            assert await store.requeue_failed_consolidation(conn=conn, fq_table=None, bank_id=bank_id) == 1
        # The retired ones keep their failure marker: they are out of the queue either way.
        for fact in (superseded, refuted):
            rows = await _fetch(
                memory, "SELECT consolidation_failed_at FROM memory_units WHERE id = $1", uuid.UUID(fact)
            )
            assert rows[0]["consolidation_failed_at"] == day(2)
        return
    elif path == "clear_for_memory":
        async with _tx(memory) as conn:
            for fact in (active, superseded, refuted):
                await store.requeue_source_memory(conn=conn, fq_table=None, bank_id=bank_id, unit_id=uuid.UUID(fact))
    elif path == "edit":
        for fact in (active, superseded, refuted):
            await memory.update_memory_unit(
                bank_id, fact, text="An edited statement about the payments service.", request_context=request_context
            )
    elif path == "mark_consolidated":
        async with _tx(memory) as conn:
            await store.mark_consolidated(
                conn=conn, fq_table=None, bank_id=bank_id, unit_ids=[active, superseded, refuted], when=None
            )
    elif path == "retag":
        async with _conn(memory) as conn:
            await conn.execute(
                "INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at) "
                "VALUES ('doc-retag', $1, 'x', 'h', $2, $2)",
                bank_id,
                day(1),
            )
            await conn.execute(
                "UPDATE memory_units SET document_id = 'doc-retag' WHERE id = ANY($1::uuid[])",
                [uuid.UUID(m) for m in (active, superseded, refuted)],
            )
        await memory.update_document("doc-retag", bank_id, tags=["moved"], request_context=request_context)
    assert await _consolidated_at(memory, active) is None, f"{path} requeues the active fact"
    for fact in (superseded, refuted):
        assert await _consolidated_at(memory, fact) is not None, f"{path} requeued a retired fact"


# =========================================================================== delete repair


async def _doc_facts(memory, bank_id: str, document_id: str) -> list[str]:
    rows = await _fetch(
        memory,
        "SELECT id FROM memory_units WHERE bank_id = $1 AND document_id = $2 ORDER BY created_at, id",
        bank_id,
        document_id,
    )
    return [str(r["id"]) for r in rows]


async def _retain(memory, request_context, bank_id: str, document_id: str, content: str, *, force: bool = False):
    """Retain through the engine (mock LLM: one fact per sentence of the prompt, so a few prompt
    lines come back as facts too); the document's fact ids after it."""
    item = {"content": content, "document_id": document_id, "event_date": day(1)}
    if force:
        item["force_reextract"] = True
    await memory.retain_batch_async(bank_id=bank_id, contents=[item], request_context=request_context)
    return await _doc_facts(memory, bank_id, document_id)


async def _fact_with(memory, bank_id: str, needle: str) -> str:
    """The one live raw fact of the bank whose text contains ``needle``."""
    rows = await _fetch(
        memory,
        "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type <> 'observation' AND text LIKE $2",
        bank_id,
        f"%{needle}%",
    )
    assert len(rows) == 1, (needle, rows)
    return str(rows[0]["id"])


def _assert_reverted(claim) -> None:
    assert claim["truth_status"] == "active"
    assert claim["valid_to"] is None and claim["retired_via"] is None and claim["superseded_by_id"] is None
    assert claim["reason"] == REVERTED_EVIDENCE_DELETED


@pytest.mark.asyncio
async def test_deleting_a_document_reverts_only_the_claims_it_alone_justified(memory, request_context):
    bank_id = await _bank(memory, request_context)
    d2 = await _retain(memory, request_context, bank_id, "doc-new", "The service moved to Java 25 in March.")
    d3 = await _retain(memory, request_context, bank_id, "doc-other", "Kafka carries the payment events now.")
    (stale,) = await _seed(memory, bank_id)
    old_obs = await _observation(memory, bank_id, [stale])
    new_obs = await _observation(memory, bank_id, d2)
    (mixed,) = await _seed(memory, bank_id)
    inherited = await _retire(memory, bank_id, target=old_obs, successor=new_obs, evidence=d2, inherit=True)
    await _retire(memory, bank_id, target=mixed, successor=None, evidence=[*d2, *d3])
    assert inherited == [stale]

    await memory.delete_document("doc-new", bank_id, request_context=request_context)

    _assert_reverted(await _claim(memory, old_obs))
    _assert_reverted(await _claim(memory, stale))
    assert await _consolidated_at(memory, stale) is None, "a reverted raw fact goes back to consolidation"
    assert await _status(memory, mixed) == "superseded", "its other evidence still stands"
    (relation,) = await _fetch(memory, "SELECT * FROM memory_lifecycle_relations WHERE bank_id = $1", bank_id)
    assert relation["source_memory_id"] is None and str(relation["target_memory_id"]) == old_obs, "a tombstone"
    snapshot = relation["source_snapshot"]
    snapshot = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    assert snapshot["fact_type"] == "observation"

    await memory.delete_document("doc-other", bank_id, request_context=request_context)
    _assert_reverted(await _claim(memory, mixed))


@pytest.mark.asyncio
async def test_single_and_bulk_deletes_revert_once_all_evidence_is_gone(memory, request_context):
    bank_id = await _bank(memory, request_context)
    e1, e2, e3, target = await _seed(memory, bank_id, 4)
    await _retire(memory, bank_id, target=target, successor=None, evidence=[e1, e2, e3])

    await memory.delete_memory_unit(e1, bank_id=bank_id, request_context=request_context)
    assert await _status(memory, target) == "superseded"
    assert await _evidence_of(memory, target) == sorted([e2, e3])

    await memory.delete_memory_units([e2, e3], bank_id=bank_id, request_context=request_context)
    _assert_reverted(await _claim(memory, target))


@pytest.mark.asyncio
async def test_a_successor_without_evidence_reverts_its_predecessor_when_deleted(memory, request_context):
    """An explicit supersession names a successor and no evidence: deleting the successor is what
    loses it. A delete without a bank (the unscoped API) resolves the bank itself."""
    bank_id = await _bank(memory, request_context)
    target, successor = await _seed(memory, bank_id, 2)
    await _retire(memory, bank_id, target=target, successor=successor, evidence=[], retired_via="explicit")

    await memory.delete_memory_unit(successor, request_context=request_context)

    _assert_reverted(await _claim(memory, target))


@pytest.mark.asyncio
async def test_deleting_a_bank_s_memories_by_type_reverts_their_dependents(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (stale,) = await _seed(memory, bank_id, fact_type="experience")
    evidence = await _seed(memory, bank_id, 2)
    target = await _observation(memory, bank_id, [stale])
    await _retire(memory, bank_id, target=target, successor=None, evidence=evidence)
    explicit_target = (await _seed(memory, bank_id, fact_type="experience"))[0]
    successor_obs = await _observation(memory, bank_id, [stale])
    await _retire(memory, bank_id, target=explicit_target, successor=successor_obs, evidence=[], retired_via="explicit")

    await memory.delete_bank(bank_id, fact_type="world", request_context=request_context)
    _assert_reverted(await _claim(memory, target))
    assert await _status(memory, explicit_target) == "superseded"

    await memory.delete_bank(bank_id, fact_type="observation", request_context=request_context)
    _assert_reverted(await _claim(memory, explicit_target))


@pytest.mark.asyncio
async def test_invalidating_the_evidence_does_not_revert_and_restoring_it_reattaches(memory, request_context):
    bank_id = await _bank(memory, request_context)
    evidence, target = await _seed(memory, bank_id, 2)
    await _retire(memory, bank_id, target=target, successor=None, evidence=[evidence])

    await memory.update_memory_unit(bank_id, evidence, state="invalidated", request_context=request_context)
    assert await _status(memory, target) == "superseded", "invalidation is temporary: no revert"
    assert await _evidence_of(memory, target) == []

    await memory.update_memory_unit(bank_id, evidence, state="valid", request_context=request_context)
    assert await _status(memory, target) == "superseded"
    assert await _evidence_of(memory, target) == [evidence]


# =========================================================================== retired observations are history


async def _sources(memory, observation_id: str) -> list[str] | None:
    """An observation's source ids, or None when the row is gone."""
    rows = await _fetch(memory, "SELECT source_memory_ids FROM memory_units WHERE id = $1", uuid.UUID(observation_id))
    return [str(s) for s in (rows[0]["source_memory_ids"] or [])] if rows else None


@pytest.mark.asyncio
async def test_a_retired_observation_outlives_an_edit_an_invalidation_and_a_truth_change_of_its_sources(
    memory, request_context
):
    """The stale-observation sweep rebuilds the live observations on a changed fact; a superseded
    one is history, so its row, text and source list stay through every change that leaves the
    fact in the bank (an invalidated fact comes back under the same id, so nothing is re-pointed)."""
    bank_id = await _bank(memory, request_context)
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "lifecycle"})
    stale, other, curated = await _seed(memory, bank_id, 3)
    (evidence,) = await _seed(memory, bank_id)
    live = await _observation(memory, bank_id, [stale, other], text="The payments stack summary.")
    retired = await _observation(memory, bank_id, [stale, curated], text="The payments service runs Java 21.")
    successor = await _observation(memory, bank_id, [evidence], text="The payments service runs Java 25.")
    await _retire(memory, bank_id, target=retired, successor=successor, evidence=[evidence])

    await memory.update_memory_unit(
        bank_id, stale, text="The payments service ran Java 21 until March.", request_context=request_context
    )
    assert await _sources(memory, live) is None, "the live observation on the edited fact is rebuilt"
    assert await _sources(memory, retired) == [stale, curated]

    await memory.update_memory_unit(bank_id, curated, state="invalidated", request_context=request_context)
    assert await _sources(memory, retired) == [stale, curated]
    await memory.update_memory_unit(bank_id, curated, state="valid", request_context=request_context)
    assert await _sources(memory, retired) == [stale, curated]

    await memory.update_memory_lifecycle(
        bank_id, curated, truth_status="refuted", reason="never true", request_context=request_context
    )
    assert await _status(memory, curated) == "refuted"

    (row,) = await _fetch(memory, "SELECT text FROM memory_units WHERE id = $1", uuid.UUID(retired))
    assert row["text"] == "The payments service runs Java 21."
    assert await _sources(memory, retired) == [stale, curated]
    claim = await _claim(memory, retired)
    assert claim["truth_status"] == "superseded" and str(claim["superseded_by_id"]) == successor


@pytest.mark.asyncio
async def test_deleting_a_retired_observation_s_sources_keeps_it_until_it_would_stand_on_nothing(
    memory, request_context
):
    """A physical delete drops the deleted facts from a retired observation and keeps the row: its
    claim stands on its evidence, not on its sources. When that evidence goes too, it would revert
    to a current belief citing nothing, so it is deleted instead."""
    bank_id = await _bank(memory, request_context)
    inventory = await _retain(memory, request_context, bank_id, "inventory", "The billing service runs on Java 21.")
    upgrade = await _retain(memory, request_context, bank_id, "upgrade", "The billing service moved to Java 25.")
    retired = await _observation(memory, bank_id, inventory, text="Billing runs Java 21.")
    successor = await _observation(memory, bank_id, upgrade, text="Billing runs Java 25.")
    await _retire(memory, bank_id, target=retired, successor=successor, evidence=upgrade, inherit=True)

    await memory.delete_document("inventory", bank_id, request_context=request_context)

    assert await _sources(memory, retired) == [], "the deleted facts leave its sources, the row stays"
    claim = await _claim(memory, retired)
    assert claim["truth_status"] == "superseded" and str(claim["superseded_by_id"]) == successor

    await memory.delete_document("upgrade", bank_id, request_context=request_context)

    assert await _sources(memory, successor) is None, "the live successor is swept with its facts"
    assert await _sources(memory, retired) is None, "reverting would make it current on nothing: deleted"
    assert await _claim(memory, retired) is None


@pytest.mark.asyncio
async def test_a_retired_observation_on_an_invalidated_fact_is_deleted_not_revived_when_its_evidence_goes(
    memory, request_context
):
    bank_id = await _bank(memory, request_context)
    source, evidence = await _seed(memory, bank_id, 2)
    retired = await _observation(memory, bank_id, [source])
    await _retire(memory, bank_id, target=retired, successor=None, evidence=[evidence])
    await memory.update_memory_unit(bank_id, source, state="invalidated", request_context=request_context)

    await memory.delete_memory_unit(evidence, bank_id=bank_id, request_context=request_context)

    assert await _sources(memory, retired) is None, "its only source is curated away: it may not become current"


# =========================================================================== re-ingest carry-forward


STATEMENT = "The payments service runs on Java 25 since the March upgrade."
SECOND = "The deploys go through the blue-green pipeline every Tuesday."


async def _supersession_on(memory, bank_id: str, evidence: list[str]) -> SimpleNamespace:
    """A predecessor observation retired on ``evidence``, with an inherited stale raw fact."""
    (stale,) = await _seed(memory, bank_id)
    old_obs = await _observation(memory, bank_id, [stale])
    new_obs = await _observation(memory, bank_id, evidence)
    await _retire(memory, bank_id, target=old_obs, successor=new_obs, evidence=evidence, inherit=True)
    return SimpleNamespace(old_obs=old_obs, stale=stale)


@pytest.mark.asyncio
async def test_an_unchanged_full_replace_reingest_keeps_the_supersession(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v1 = await _retain(memory, request_context, bank_id, "pom", f"{STATEMENT} {SECOND}")
    statement = await _fact_with(memory, bank_id, "Java 25")
    second = await _fact_with(memory, bank_id, "blue-green")
    supersession = await _supersession_on(memory, bank_id, [statement])
    old_obs, stale = supersession.old_obs, supersession.stale
    # The re-ingested fact's own state travels too: a claim (retired), retention, a relation end.
    (other,) = await _seed(memory, bank_id)
    await _retire(memory, bank_id, target=second, successor=other, evidence=[other], retired_via="explicit")
    async with _conn(memory) as conn:
        await conn.execute(
            "INSERT INTO memory_retention_state (memory_id, bank_id, importance, stability_days, last_reinforced_at, "
            "reinforcement_count, created_at, updated_at) VALUES ($1, $2, 0.8, 77.5, $3, 3, $3, $3)",
            uuid.UUID(statement),
            bank_id,
            day(20),
        )

    v2 = await _retain(memory, request_context, bank_id, "pom", f"{STATEMENT} {SECOND}", force=True)

    assert set(v2).isdisjoint(v1), "a full replace re-extracts under new ids"
    new_statement = await _fact_with(memory, bank_id, "Java 25")
    new_second = await _fact_with(memory, bank_id, "blue-green")
    assert await _status(memory, old_obs) == "superseded"
    assert await _evidence_of(memory, old_obs) == [new_statement], "the evidence follows the re-extracted fact"
    assert await _status(memory, stale) == "superseded"
    assert await _evidence_of(memory, stale) == [new_statement]
    carried = await _claim(memory, new_second)
    assert carried["truth_status"] == "superseded" and str(carried["superseded_by_id"]) == other
    assert await _consolidated_at(memory, new_second) == NOW, "a carried retired claim is never pending"
    (retention,) = await _fetch(
        memory, "SELECT * FROM memory_retention_state WHERE memory_id = $1", uuid.UUID(new_statement)
    )
    assert (retention["stability_days"], retention["reinforcement_count"]) == (77.5, 3)
    (relation,) = await _fetch(
        memory, "SELECT target_memory_id FROM memory_lifecycle_relations WHERE source_memory_id = $1", uuid.UUID(other)
    )
    assert str(relation["target_memory_id"]) == new_second, "the tombstone is re-attached to the re-extracted fact"

    # A re-ingest that no longer says it drops the evidence: the predecessor is current again.
    await _retain(memory, request_context, bank_id, "pom", "The payments team moved to a new office.", force=True)
    _assert_reverted(await _claim(memory, old_obs))
    _assert_reverted(await _claim(memory, stale))


@pytest.mark.asyncio
async def test_a_delta_reingest_carries_the_unchanged_sentences_forward(memory, request_context):
    bank_id = await _bank(memory, request_context)
    await _retain(memory, request_context, bank_id, "readme", STATEMENT)
    statement = await _fact_with(memory, bank_id, "Java 25")
    supersession = await _supersession_on(memory, bank_id, [statement])
    old_obs, stale = supersession.old_obs, supersession.stale

    # One paragraph, so the one chunk changes and its facts are re-extracted (delta path).
    v2 = await _retain(memory, request_context, bank_id, "readme", f"{STATEMENT} {SECOND}")

    assert statement not in v2, "the changed chunk was re-extracted"
    new_statement = await _fact_with(memory, bank_id, "Java 25")
    assert await _status(memory, old_obs) == "superseded"
    assert await _evidence_of(memory, old_obs) == [new_statement]

    await _retain(memory, request_context, bank_id, "readme", SECOND)
    _assert_reverted(await _claim(memory, old_obs))
    _assert_reverted(await _claim(memory, stale))


@pytest.mark.asyncio
async def test_a_multi_batch_replace_carries_state_onto_a_fact_a_later_batch_writes(
    memory, request_context, monkeypatch
):
    """A streaming retain writes one chunk batch per transaction: the stash taken by the first batch
    is carried onto facts the last batch inserts, and nothing reverts in between."""
    config = memory._config_resolver._global_config
    monkeypatch.setattr(config, "retain_chunk_size", 260)
    monkeypatch.setattr(config, "retain_chunk_batch_size", 1)
    paragraphs = [
        f"Paragraph {name} explains that the {name} service moved to the new cluster during the spring migration "
        f"window. The {name} owners confirmed the rollout went through the regular change review process."
        for name in ("alpha", "bravo", "charlie")
    ]
    content = "\n\n".join(paragraphs)
    await _retain(memory, request_context, bank_id := await _bank(memory, request_context), "big", content)
    chunks = await _fetch(memory, "SELECT chunk_id FROM chunks WHERE bank_id = $1", bank_id)
    assert len(chunks) == 3, "one chunk per paragraph, one batch per chunk"
    evidence = await _fact_with(memory, bank_id, "charlie service moved")
    supersession = await _supersession_on(memory, bank_id, [evidence])
    old_obs, stale = supersession.old_obs, supersession.stale

    await _retain(memory, request_context, bank_id, "big", content, force=True)

    new_evidence = await _fact_with(memory, bank_id, "charlie service moved")
    assert new_evidence != evidence
    assert await _status(memory, old_obs) == "superseded"
    assert await _evidence_of(memory, old_obs) == [new_evidence]
    assert await _status(memory, stale) == "superseded"


@pytest.mark.asyncio
async def test_a_reingest_of_a_retired_observation_s_own_document_re_points_its_sources(memory, request_context):
    """The predecessor side of a re-ingest: the superseded observation built on the re-extracted
    document keeps its row and cites the re-extracted facts; a revision that drops the fact leaves
    it citing nothing, still retired (its evidence lives elsewhere)."""
    bank_id = await _bank(memory, request_context)
    await _retain(memory, request_context, bank_id, "pom", f"{STATEMENT} {SECOND}")
    old_fact = await _fact_with(memory, bank_id, "Java 25")
    (evidence,) = await _seed(memory, bank_id)
    retired = await _observation(memory, bank_id, [old_fact], text="Payments run Java 25 (old pom).")
    successor = await _observation(memory, bank_id, [evidence], text="Payments run Java 26.")
    await _retire(memory, bank_id, target=retired, successor=successor, evidence=[evidence], inherit=True)

    await _retain(memory, request_context, bank_id, "pom", f"{STATEMENT} {SECOND}", force=True)

    new_fact = await _fact_with(memory, bank_id, "Java 25")
    assert new_fact != old_fact
    assert await _sources(memory, retired) == [new_fact]
    assert await _status(memory, retired) == "superseded"
    assert await _status(memory, new_fact) == "superseded", "the inherited claim is carried too"

    # Delta path: one chunk, so a changed paragraph re-extracts its facts in place.
    await _retain(memory, request_context, bank_id, "pom", f"{STATEMENT} {SECOND} Builds run on Gradle.")
    delta_fact = await _fact_with(memory, bank_id, "Java 25")
    assert delta_fact != new_fact
    assert await _sources(memory, retired) == [delta_fact]

    await _retain(memory, request_context, bank_id, "pom", SECOND, force=True)
    assert await _sources(memory, retired) == []
    assert await _status(memory, retired) == "superseded"


@pytest.mark.asyncio
async def test_a_multi_batch_replace_re_keys_in_document_edges_whichever_batch_matches_first(
    memory, request_context, monkeypatch
):
    """A claim carried by the first batch whose evidence (and successor) is a fact of the same
    document that a later batch re-extracts keeps that edge, and so does the reverse."""
    config = memory._config_resolver._global_config
    monkeypatch.setattr(config, "retain_chunk_size", 260)
    monkeypatch.setattr(config, "retain_chunk_batch_size", 1)
    paragraphs = [
        f"Paragraph {name} explains that the {name} service moved to the new cluster during the spring migration "
        f"window. The {name} owners confirmed the rollout went through the regular change review process."
        for name in ("alpha", "bravo", "charlie")
    ]
    content = "\n\n".join(paragraphs)
    await _retain(memory, request_context, bank_id := await _bank(memory, request_context), "big", content)
    alpha = await _fact_with(memory, bank_id, "alpha service moved")
    charlie = await _fact_with(memory, bank_id, "charlie service moved")
    alpha_owners = await _fact_with(memory, bank_id, "alpha owners")
    charlie_owners = await _fact_with(memory, bank_id, "charlie owners")
    await _retire(memory, bank_id, target=alpha, successor=charlie, evidence=[charlie], retired_via="explicit")
    await _retire(
        memory, bank_id, target=charlie_owners, successor=alpha_owners, evidence=[alpha_owners], retired_via="explicit"
    )

    await _retain(memory, request_context, bank_id, "big", content, force=True)

    for holder, target in (("alpha service moved", "charlie service moved"), ("charlie owners", "alpha owners")):
        new_holder = await _fact_with(memory, bank_id, holder)
        new_target = await _fact_with(memory, bank_id, target)
        claim = await _claim(memory, new_holder)
        assert claim["truth_status"] == "superseded", holder
        assert str(claim["superseded_by_id"]) == new_target, holder
        assert await _evidence_of(memory, new_holder) == [new_target], holder


def test_the_carry_forward_key_ignores_spacing_and_case():
    assert reingest_text_key("Java  25 runs\nhere.") == reingest_text_key("java 25 RUNS here.")
    assert reingest_text_key("Java 25") != reingest_text_key("Java 21")


# =========================================================================== successor re-link


@pytest.mark.asyncio
async def test_a_rebuilt_successor_is_relinked_and_takes_the_predecessors_end_as_its_start(memory, request_context):
    bank_id = await _bank(memory, request_context)
    e1, e2, e3 = await _seed(memory, bank_id, 3)
    (stale,) = await _seed(memory, bank_id)
    target = await _observation(memory, bank_id, [stale])
    successor = await _observation(memory, bank_id, [e1, e2])
    await _retire(memory, bank_id, target=target, successor=successor, evidence=[e1, e2], at=day(12))
    (explicit,) = await _seed(memory, bank_id)
    await _retire(memory, bank_id, target=explicit, successor=None, evidence=[e1], retired_via="explicit")
    # The edit / delta sweep rebuilds the successor: the old row goes (its pointer SET NULL).
    async with _conn(memory) as conn:
        await conn.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(successor))
    assert (await _claim(memory, target))["superseded_by_id"] is None
    partial = await _observation(memory, bank_id, [e1])
    rebuilt = await _observation(memory, bank_id, [e1, e2, e3])

    async with _tx(memory) as conn:
        relinked = await get_memories().relink_rebuilt_successors(
            conn, bank_id=bank_id, new_observation_ids=[partial, rebuilt]
        )

    assert relinked == 1
    assert str((await _claim(memory, target))["superseded_by_id"]) == rebuilt
    assert (await _claim(memory, rebuilt))["valid_from"] == day(12)
    assert (await _claim(memory, explicit))["superseded_by_id"] is None, "an explicit retirement is not re-linked"


@pytest.mark.asyncio
async def test_a_relink_inside_the_caller_s_lock_pass_locks_only_the_new_observations(
    memory, request_context, monkeypatch
):
    """Consolidation puts ``relinkable_claims`` in its one lock pass and hands the locked set over:
    the re-link then locks no claim another writer can hold, and leaves a candidate outside the set
    for the next rebuild."""
    from hindsight_api.engine.memories.pg import adaptive_claims

    bank_id = await _bank(memory, request_context)
    e1, e2 = await _seed(memory, bank_id, 2)
    (stale,) = await _seed(memory, bank_id)
    target = await _observation(memory, bank_id, [stale])
    successor = await _observation(memory, bank_id, [e1, e2])
    await _retire(memory, bank_id, target=target, successor=successor, evidence=[e1, e2], at=day(12))
    async with _conn(memory) as conn:
        await conn.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(successor))
    rebuilt = await _observation(memory, bank_id, [e1, e2])
    store = get_memories()
    calls: list[list[str]] = []
    real = adaptive_claims.lock_claims

    async def spy(conn, **kwargs):
        calls.append(sorted(str(i) for i in kwargs["memory_ids"]))
        return await real(conn, **kwargs)

    async with _tx(memory) as conn:
        assert await store.relinkable_claims(conn, bank_id=bank_id, evidence_ids=[e1, e2]) == {target}
        monkeypatch.setattr(adaptive_claims, "lock_claims", spy)
        assert (
            await store.relink_rebuilt_successors(
                conn, bank_id=bank_id, new_observation_ids=[rebuilt], locked_claim_ids=set()
            )
            == 0
        ), "a candidate outside the caller's pass is left alone"
        monkeypatch.setattr(adaptive_claims, "lock_claims", real)
        locked = await store.lock_claims(conn, bank_id=bank_id, memory_ids=[target], skip_locked=True)
        monkeypatch.setattr(adaptive_claims, "lock_claims", spy)
        relinked = await store.relink_rebuilt_successors(
            conn, bank_id=bank_id, new_observation_ids=[rebuilt], locked_claim_ids=locked
        )

    assert relinked == 1
    assert calls == [[rebuilt]], "only the new observation's own claim row is locked here"
    assert str((await _claim(memory, target))["superseded_by_id"]) == rebuilt
    assert (await _claim(memory, rebuilt))["valid_from"] == day(12)


# =========================================================================== scope cap, dedup probe, shadow


@pytest.mark.asyncio
async def test_the_scope_cap_counts_only_live_observations(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (fact,) = await _seed(memory, bank_id)
    live = await _observation(memory, bank_id, [fact], tags=["a"])
    await _observation(memory, bank_id, [fact], tags=["a", "b"])
    retired = await _observation(memory, bank_id, [fact], tags=["a"])
    await _observation(memory, bank_id, [fact], tags=["c"])
    await _retire(memory, bank_id, target=retired, successor=live, evidence=[fact])
    store = get_memories()
    async with _conn(memory) as conn:
        assert await store.count_active_observations(conn, bank_id=bank_id, tags=["a"], tags_match="all_strict") == 2
        assert await store.count_active_observations(conn, bank_id=bank_id, tags=["a"], tags_match="exact") == 1
        assert await store.count_active_observations(conn, bank_id=bank_id, tags=None, tags_match="any") == 3
        assert await store.count_observations_with_tags(conn=conn, fq_table=None, bank_id=bank_id, tags=["a"]) == 3


@pytest.mark.asyncio
async def test_the_dedup_probe_can_exclude_retired_twins_before_its_limit(memory, request_context):
    bank_id = await _bank(memory, request_context)
    (fact,) = await _seed(memory, bank_id)
    text = "The payments service runs on Java 25."
    live_text = "The payments service now runs on Java 25 in production."
    twins = [await _observation(memory, bank_id, [fact], text=text) for _ in range(3)]
    live = await _observation(memory, bank_id, [fact], text=live_text)
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text, live_text])
    async with _conn(memory) as conn:
        for memory_id, embedding in [*((t, embeddings[0]) for t in twins), (live, embeddings[1])]:
            await conn.execute(
                "UPDATE memory_units SET embedding = $2::vector WHERE id = $1", uuid.UUID(memory_id), str(embedding)
            )
    for twin in twins:
        await _retire(memory, bank_id, target=twin, successor=live, evidence=[fact])
    backend = await memory._get_backend()

    async def probe(exclude: bool) -> list[str]:
        arms = await get_memories().recall_unified(
            conn=backend,
            bank_id=bank_id,
            fact_types=["observation"],
            query_embedding=str(embeddings[0]),
            query_text=text,
            limit=3,
            enable_graph=False,
            exclude_non_active=exclude,
        )
        return [str(r.id) for r in arms["observation"].semantic]

    assert live not in await probe(False), "three retired near-twins fill the limit"
    assert await probe(True) == [live]


@pytest.mark.asyncio
async def test_an_observe_mode_shadow_record_is_replaced_by_the_latest_decision(memory, request_context):
    bank_id = await _bank(memory, request_context)
    target, e1, e2 = await _seed(memory, bank_id, 3)
    store = get_memories()
    async with _tx(memory) as conn:
        assert await store.record_shadow_retirement(
            conn,
            bank_id=bank_id,
            memory_id=target,
            would_status=SUPERSEDED,
            deciding_snapshot={"a": 1},
            evidence_ids=[e1],
        )
        assert await store.record_shadow_retirement(
            conn, bank_id=bank_id, memory_id=target, would_status=REFUTED, deciding_snapshot={"b": 2}, evidence_ids=[e2]
        )
        assert not await store.record_shadow_retirement(
            conn,
            bank_id=bank_id,
            memory_id=str(uuid.uuid4()),
            would_status=REFUTED,
            deciding_snapshot={},
            evidence_ids=[],
        )
        shadows = await store.shadow_for(conn, bank_id=bank_id, memory_ids=[target, e1])
    assert list(shadows) == [target]
    assert shadows[target].would_status == "refuted" and shadows[target].deciding_snapshot == {"b": 2}
    assert shadows[target].evidence_ids == [e2]
    assert await _claim(memory, target) is None, "observe mode never touches the claim"


# =========================================================================== transfer


async def _lifecycle_bank(memory, request_context) -> SimpleNamespace:
    """A bank with every kind of adaptive state on memories that a transfer carries."""
    bank_id = await _bank(memory, request_context)
    await _retain(memory, request_context, bank_id, "old", "The payments service runs on Java 21.")
    await _retain(memory, request_context, bank_id, "new", STATEMENT)
    old = [await _fact_with(memory, bank_id, "Java 21")]
    new = [await _fact_with(memory, bank_id, "Java 25")]
    old_obs = await _observation(memory, bank_id, old, text="Payments run Java 21.")
    new_obs = await _observation(memory, bank_id, new, text="Payments run Java 25.")
    await _retire(memory, bank_id, target=old_obs, successor=new_obs, evidence=new, inherit=True, at=day(15))
    async with _conn(memory) as conn:
        await conn.execute(
            "INSERT INTO memory_retention_state (memory_id, bank_id, importance, stability_days, last_reinforced_at, "
            "reinforcement_count, access_count, forgetting_exempt, exemption_reason, created_at, updated_at) "
            "VALUES ($1, $2, 0.9, 64.0, $3, 2, 5, TRUE, 'policy', $3, $3)",
            uuid.UUID(new[0]),
            bank_id,
            day(30),
        )
    return SimpleNamespace(bank_id=bank_id, ids={"old": old[0], "new": new[0], "old_obs": old_obs, "new_obs": new_obs})


async def _by_text(memory, bank_id: str) -> dict[str, str]:
    rows = await _fetch(memory, "SELECT id, text FROM memory_units WHERE bank_id = $1", bank_id)
    return {r["text"]: str(r["id"]) for r in rows}


async def _assert_carried(memory, source: str, target: str, ids: dict[str, str]) -> None:
    src = {i: t for t, i in (await _by_text(memory, source)).items()}
    dst = await _by_text(memory, target)
    m = {key: dst[src[source_id]] for key, source_id in ids.items()}
    old_obs = await _claim(memory, m["old_obs"])
    assert old_obs["truth_status"] == "superseded" and str(old_obs["superseded_by_id"]) == m["new_obs"]
    assert old_obs["valid_to"] == day(15)
    assert await _evidence_of(memory, m["old_obs"]) == [m["new"]]
    stale = await _claim(memory, m["old"])
    assert stale["truth_status"] == "superseded" and str(stale["inherited_from_id"]) == m["old_obs"]
    assert await _consolidated_at(memory, m["old"]) is not None, "an imported retired fact is not pending"
    assert (await _claim(memory, m["new_obs"]))["valid_from"] == day(15)
    (retention,) = await _fetch(
        memory, "SELECT * FROM memory_retention_state WHERE memory_id = $1", uuid.UUID(m["new"])
    )
    assert (retention["stability_days"], retention["reinforcement_count"], retention["access_count"]) == (64.0, 2, 5)
    assert retention["forgetting_exempt"] and retention["exemption_reason"] == "policy"
    (relation,) = await _fetch(memory, "SELECT * FROM memory_lifecycle_relations WHERE bank_id = $1", target)
    assert (str(relation["source_memory_id"]), str(relation["target_memory_id"])) == (m["new_obs"], m["old_obs"])


@pytest.mark.asyncio
async def test_a_bank_transfer_carries_claims_retention_relations_and_evidence(memory, request_context):
    from hindsight_api.engine.transfer import export_bank

    seeded = await _lifecycle_bank(memory, request_context)
    source, ids = seeded.bank_id, seeded.ids
    target = f"test-acs-{uuid.uuid4().hex[:8]}"
    async with _conn(memory) as conn:
        archive = await export_bank(conn, source)
    await memory.import_bank_async(archive, request_context, target_bank_id=target)
    await _assert_carried(memory, source, target, ids)


@pytest.mark.asyncio
async def test_a_clone_carries_the_same_state(memory, request_context):
    from hindsight_api.engine.transfer import build_bank_archive, load_bank_export

    seeded = await _lifecycle_bank(memory, request_context)
    source, ids = seeded.bank_id, seeded.ids
    target = f"test-acs-{uuid.uuid4().hex[:8]}"
    async with _tx(memory) as conn:
        payload = await load_bank_export(conn, source, memories=get_memories())
    await memory.import_bank_async(await build_bank_archive(payload), request_context, target_bank_id=target)
    await _assert_carried(memory, source, target, ids)


@pytest.mark.asyncio
async def test_a_document_import_into_another_bank_carries_its_facts_state(memory, request_context):
    source = await _bank(memory, request_context)
    target = await _bank(memory, request_context)
    await _retain(memory, request_context, source, "doc", f"{STATEMENT} {SECOND}")
    a = await _fact_with(memory, source, "Java 25")
    b = await _fact_with(memory, source, "blue-green")
    await _retire(memory, source, target=a, successor=b, evidence=[b], retired_via="explicit")

    archive = await memory.export_documents_async(source, request_context, document_ids=["doc"])
    await memory.import_documents_async(target, archive, request_context)

    a2 = await _fact_with(memory, target, "Java 25")
    b2 = await _fact_with(memory, target, "blue-green")
    claim = await _claim(memory, a2)
    assert claim["truth_status"] == "superseded" and str(claim["superseded_by_id"]) == b2
    assert await _evidence_of(memory, a2) == [b2]
    (relation,) = await _fetch(memory, "SELECT * FROM memory_lifecycle_relations WHERE bank_id = $1", target)
    assert (str(relation["source_memory_id"]), str(relation["target_memory_id"])) == (b2, a2)


@pytest.mark.asyncio
async def test_an_imported_retired_fact_commits_with_its_claim_and_its_stamp(memory, request_context, monkeypatch):
    """Each replay batch writes its facts' claims in the transaction that inserts them: with the
    closing pass (which re-keys the pointers) never run, a superseded fact from an archive exported
    without observations is already retired and stamped consolidated, so no recall shows it as
    current and no consolidation consumes it in between."""
    from hindsight_api.engine.transfer import importer

    source = await _bank(memory, request_context)
    target = await _bank(memory, request_context)
    await _retain(memory, request_context, source, "doc", f"{STATEMENT} {SECOND}")
    a = await _fact_with(memory, source, "Java 25")
    b = await _fact_with(memory, source, "blue-green")
    await _retire(memory, source, target=a, successor=b, evidence=[b], retired_via="explicit")
    archive = await memory.export_documents_async(source, request_context, document_ids=["doc"])

    async def no_closing_pass(*_args, **_kwargs):
        return importer._LifecycleRestore()

    monkeypatch.setattr(importer, "_restore_lifecycle", no_closing_pass)
    await memory.import_documents_async(target, archive, request_context)

    a2 = await _fact_with(memory, target, "Java 25")
    claim = await _claim(memory, a2)
    assert claim["truth_status"] == "superseded" and claim["superseded_by_id"] is None, "pointer left to the pass"
    assert await _consolidated_at(memory, a2) is not None


@pytest.mark.asyncio
async def test_an_archive_of_a_bank_without_adaptive_state_carries_none_of_it(memory, request_context):
    from hindsight_api.engine.transfer import export_bank

    bank_id = await _bank(memory, request_context)
    facts = await _retain(memory, request_context, bank_id, "doc", f"{STATEMENT} {SECOND}")
    await _observation(memory, bank_id, facts)
    async with _conn(memory) as conn:
        archive = await export_bank(conn, bank_id)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        assert "lifecycle_relations.json" not in names and "claim_evidence.json" not in names
        for name in names:
            if name.startswith("documents/") or name == "observations.json":
                body = zf.read(name).decode()
                assert '"retention"' not in body and '"claim"' not in body, name


@pytest.mark.asyncio
async def test_an_archived_memory_s_snapshot_is_rekeyed_onto_the_target_bank(memory, request_context):
    from hindsight_api.engine.transfer import export_bank

    source = await _bank(memory, request_context)
    await _retain(memory, request_context, source, "doc", f"{STATEMENT} {SECOND}")
    evidence = await _fact_with(memory, source, "Java 25")
    holder = await _fact_with(memory, source, "blue-green")
    await _retire(memory, source, target=holder, successor=None, evidence=[evidence], retired_via="explicit")
    async with _conn(memory) as conn:
        await conn.execute(
            "INSERT INTO memory_reinforcement_events (id, bank_id, memory_id, source_type, source_key, event_time, "
            "weight, outcome, created_at) VALUES ($1, $2, $3, 'explicit_confirmation', 'explicit:k', $4, 1.0, "
            "'applied', $4)",
            uuid.uuid4(),
            source,
            uuid.UUID(evidence),
            day(3),
        )
    await memory.update_memory_unit(source, evidence, state="invalidated", request_context=request_context)
    async with _conn(memory) as conn:
        archive = await export_bank(conn, source)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        (row,) = json.loads(zf.read("data/invalidated_memory_units.json"))
    state = row["adaptive_state"] if isinstance(row["adaptive_state"], dict) else json.loads(row["adaptive_state"])
    assert "events" not in state and row["source_id"] == evidence, "the ledger stays home; the id travels"

    target = f"test-acs-{uuid.uuid4().hex[:8]}"
    await memory.import_bank_async(archive, request_context, target_bank_id=target)
    holder2 = await _fact_with(memory, target, "blue-green")
    (archived,) = await _fetch(
        memory, "SELECT id, adaptive_state FROM invalidated_memory_units WHERE bank_id = $1", target
    )
    snapshot = archived["adaptive_state"]
    snapshot = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    assert snapshot["memory_id"] == str(archived["id"])
    assert [e["memory_id"] for e in snapshot["evidence_in"]] == [holder2]

    await memory.update_memory_unit(target, str(archived["id"]), state="valid", request_context=request_context)
    assert await _evidence_of(memory, holder2) == [str(archived["id"])]


# =========================================================================== Oracle statements (no database)


class _RecordingOracleConn:
    """Enough of an Oracle connection to run OracleOps' claim methods and capture their SQL."""

    backend_type = "oracle"

    def __init__(self, live: set[uuid.UUID]):
        self.live = live
        self.claims: set[uuid.UUID] = set()
        self.relations: set[uuid.UUID] = set()
        self.statements: list[str] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchrow(self, query: str, *args):
        self.statements.append(query)
        key = args[0]
        if "FROM memory_units" in query:
            return {"id": key} if key in self.live else None
        if "FROM memory_claim_state" in query:
            return {"memory_id": key} if key in self.claims else None
        if "FROM memory_lifecycle_relations" in query:
            return {"id": key} if key in self.relations else None
        return None

    async def fetch(self, query: str, *args):
        self.statements.append(query)
        if "FROM memory_claim_state" in query:
            return [{"memory_id": m} for m in args[0] if m in self.claims]
        return []

    async def execute(self, query: str, *args):
        self.statements.append(query)
        if "INTO memory_lifecycle_relations" in query:
            self.relations.add(args[0])
        return "UPDATE 0"

    async def executemany(self, query: str, rows):
        self.statements.append(query)
        if "INTO memory_claim_state" in query:
            self.claims.update(row[0] for row in rows)


@pytest.mark.asyncio
async def test_the_oracle_claim_writers_lock_rows_one_by_one_and_speak_oracle():
    a, b, gone = sorted(uuid.uuid4() for _ in range(3))
    conn = _RecordingOracleConn(live={a, b})
    ops = OracleOps()

    locked = await ops.upsert_default_claim_state(
        conn,
        claims_table="memory_claim_state",
        units_table="memory_units",
        bank_id="b",
        memory_ids=[gone, b, a],
        now=NOW,
        skip_locked=True,
    )
    await ops.insert_claim_evidence(
        conn, evidence_table="memory_claim_evidence", units_table="memory_units", bank_id="b", pairs=[(a, b)], now=NOW
    )
    inserted = await ops.insert_lifecycle_relation(
        conn,
        table="memory_lifecycle_relations",
        row=LifecycleRelationRow(
            id=uuid.uuid4(),
            bank_id="b",
            relation_type="supersedes",
            source_memory_id=a,
            target_memory_id=b,
            source_snapshot="{}",
            target_snapshot="{}",
            origin="consolidation",
            reason=None,
            confidence=None,
            evidence="{}",
            created_at=NOW,
        ),
    )
    await ops.lock_relations_for_delete(conn, table="memory_lifecycle_relations", bank_id="b", unit_ids=[a])
    await ops.upsert_claim_shadow(
        conn,
        table="memory_claim_shadow",
        units_table="memory_units",
        bank_id="b",
        memory_id=a,
        would_status="superseded",
        deciding_snapshot="{}",
        evidence_ids="[]",
        now=NOW,
    )

    assert locked == sorted(str(m) for m in (a, b))
    assert inserted
    parent_locks = [s for s in conn.statements if "FROM memory_units" in s]
    assert parent_locks and all(" FOR UPDATE" in s for s in parent_locks)
    assert any("IGNORE_ROW_ON_DUPKEY_INDEX(memory_claim_state, pk_memory_claim_state)" in s for s in conn.statements)
    assert any(
        "IGNORE_ROW_ON_DUPKEY_INDEX(memory_claim_evidence, pk_memory_claim_evidence)" in s for s in conn.statements
    )
    assert any("IGNORE_ROW_ON_DUPKEY_INDEX(memory_lifecycle_relations, uq_mlr_edge)" in s for s in conn.statements)
    assert any("IGNORE_ROW_ON_DUPKEY_INDEX(memory_claim_shadow, pk_memory_claim_shadow)" in s for s in conn.statements)
    for statement in conn.statements:
        rewritten = _rewrite_pg_to_oracle(statement).query
        assert "$" not in rewritten and "FOR KEY SHARE" not in rewritten and "ON CONFLICT" not in rewritten, rewritten


def test_the_shared_dialect_claim_statements_rewrite_to_oracle():
    """The store's own statements (pg/adaptive_claims.py) run on Oracle through the rewriter."""
    from hindsight_api.engine.memories.pg import adaptive_claims

    source = open(adaptive_claims.__file__).read()
    assert "LEAST(" not in source and "unnest(" not in source and "::jsonb" not in source
    for sql in (
        "UPDATE memory_units mu SET consolidated_at = NULL WHERE mu.bank_id = $1 AND mu.id = ANY($2::uuid[]) "
        "AND NOT EXISTS (SELECT 1 FROM memory_claim_state c WHERE c.memory_id = mu.id AND c.truth_status IN "
        "('superseded', 'refuted'))",
        "SELECT memory_id FROM memory_claim_evidence WHERE bank_id = $1 AND evidence_memory_id = ANY($2::uuid[]) "
        "UNION SELECT memory_id FROM memory_claim_state WHERE bank_id = $1 AND superseded_by_id = ANY($2::uuid[])",
        "SELECT 1 FROM memory_claim_state WHERE bank_id = $1 AND truth_status IN ('superseded', 'refuted') LIMIT 1",
        "DELETE FROM observation_sources WHERE observation_id = $1 AND source_id = ANY($2::uuid[])",
        "DELETE FROM memory_units WHERE bank_id = $1 AND id = ANY($2::uuid[]) AND fact_type = 'observation'",
    ):
        rewritten = _rewrite_pg_to_oracle(sql).query
        assert "$" not in rewritten and "::" not in rewritten, rewritten
