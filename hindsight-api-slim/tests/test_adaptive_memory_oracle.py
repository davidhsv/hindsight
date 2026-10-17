"""Oracle parity of curation invalidate/restore (adaptive snapshot included), the causal-link snapshot
and the ``merged_into`` fold.

Most of this file runs without a database: a scripted Oracle connection answers the store's
statements and records them, so the Oracle branches execute end to end and every statement is
pushed through the real PG-to-Oracle rewriter. The ``oracle``-marked tests at the bottom need a live
Oracle 23ai (``ORACLE_TEST_DSN``; ``pytest -m oracle -n0``).
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from hindsight_api.engine.causal_links import CANONICAL_CAUSAL_LINK_TYPE
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.oracle import OracleConnection, _convert_args, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.pg import adaptive_archive, links, writes
from hindsight_api.engine.memories.pg.adaptive_archive import AdaptiveSnapshot

BANK = "bank-ora"
CAUSAL = CANONICAL_CAUSAL_LINK_TYPE
#: The driver hands TIMESTAMP WITH TIME ZONE back without a zone; the writers bind UTC.
NAIVE = datetime(2026, 9, 1, 12, 0, 0)


def _bare(table: str) -> str:
    return table


def _assert_speaks_oracle(statements: list[str]) -> None:
    """Every recorded statement survives the rewriter as Oracle SQL: no PG binds, casts or clauses."""
    for statement in statements:
        rewritten = _rewrite_pg_to_oracle(statement).query
        assert not re.search(r"\$\d", rewritten), rewritten
        assert "::" not in rewritten, rewritten
        for pg_only in (
            "FOR KEY SHARE",
            "ON CONFLICT",
            "jsonb_to_recordset",
            "unnest(",
            "array_agg",
            "pg_attribute",
            "regclass",
            "SELECT EXISTS",
            "to_tsvector",
        ):
            assert pg_only not in rewritten, (pg_only, rewritten)
        if re.match(r"\s*INSERT\b", rewritten) and re.search(r"\bSELECT\b", rewritten):
            assert "RETURNING" not in rewritten, rewritten  # no RETURNING for INSERT ... SELECT on Oracle


class _ScriptedOracle:
    """An Oracle connection whose answers come from ``answer(method, query, args)``.

    ``answer`` returns the value for that call, or ``None`` for "no row" / the default status.
    """

    backend_type = "oracle"

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[tuple[str, str, tuple]] = []

    @property
    def statements(self) -> list[str]:
        return [q for _m, q, _a in self.calls]

    def sent(self, method: str, needle: str) -> list[tuple]:
        return [a for m, q, a in self.calls if m == method and needle in q]

    def parse_json(self, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def _call(self, method: str, query: str, args: tuple) -> Any:
        self.calls.append((method, query, args))
        return self.answer(method, query, args)

    async def fetch(self, query, *args):
        return await self._call("fetch", query, args) or []

    async def fetchrow(self, query, *args):
        return await self._call("fetchrow", query, args)

    async def fetchval(self, query, *args):
        return await self._call("fetchval", query, args)

    async def execute(self, query, *args):
        return await self._call("execute", query, args) or "UPDATE 1"

    async def executemany(self, query, rows):
        self.calls.append(("executemany", query, tuple(rows)))


@pytest.fixture
def oracle_backend(monkeypatch):
    """The configured backend reads as Oracle (bare table names, Oracle-only branches)."""
    monkeypatch.setattr("hindsight_api.engine.schema._is_oracle", lambda: True)


# --------------------------------------------------------------------------- column list


@pytest.mark.asyncio
async def test_the_archive_column_list_comes_from_all_tab_columns_quoted_upper_case(oracle_backend):
    def answer(method, query, args):
        assert "all_tab_columns" in query and args == ("memory_units",)
        return [{"column_name": c} for c in ("ID", "BANK_ID", "TEXT", "EMBEDDING", "SEARCH_VECTOR", "TAGS")]

    conn = _ScriptedOracle(answer)
    assert await writes._archive_columns(conn, _bare) == '"ID", "BANK_ID", "TEXT", "TAGS"'
    assert await OracleOps().table_columns(conn, '"tenant_a".memory_units') == [
        '"ID"',
        '"BANK_ID"',
        '"TEXT"',
        '"EMBEDDING"',
        '"SEARCH_VECTOR"',
        '"TAGS"',
    ]
    _assert_speaks_oracle(conn.statements)


def test_oracle_rebuilds_no_search_vector_and_postgres_keeps_its_expression():
    from hindsight_api.config import get_config

    config = get_config()
    assert OracleOps().search_vector_expr(config) is None
    assert OracleOps().search_vector_expr(config, text_col="$3", context_col="$4") is None
    from hindsight_api.engine.db.ops_postgresql import pg_search_vector_expr

    assert PostgreSQLOps().search_vector_expr(config, text_col="$3", context_col="$4") == pg_search_vector_expr(
        config, text_col="$3", context_col="$4"
    )


# --------------------------------------------------------------------------- causal snapshot


@pytest.mark.asyncio
async def test_the_causal_snapshot_reads_the_archive_through_json_table_and_merges_both_arms(oracle_backend):
    unit, peer, archived_peer = (uuid.uuid4() for _ in range(3))

    def answer(method, query, args):
        if "FROM memory_links" in query:
            return [{"from_unit_id": unit, "to_unit_id": peer, "link_type": CAUSAL, "weight": 0.5}]
        if "JSON_TABLE" in query:
            return [
                # The same edge parked on an archived peer too: one descriptor, not two.
                {"from_unit_id": str(unit).upper(), "to_unit_id": str(peer), "link_type": CAUSAL, "weight": 0.5},
                {"from_unit_id": str(archived_peer), "to_unit_id": str(unit), "link_type": CAUSAL, "weight": None},
            ]
        raise AssertionError(query)

    conn = _ScriptedOracle(answer)
    descriptors = await links.snapshot_causal_links(conn, BANK, str(unit))

    assert [(d.from_unit_id, d.to_unit_id, d.weight) for d in descriptors] == [
        (str(unit), str(peer), 0.5),
        (str(archived_peer), str(unit), 1.0),  # no weight in the JSON: the default
    ]
    archived_sql = conn.statements[1]
    assert "jsonb_to_recordset" not in archived_sql and "HEXTORAW(REPLACE(d.from_unit_id, '-', ''))" in archived_sql
    _assert_speaks_oracle(conn.statements)
    # The unit id binds as RAW(16) and the type list expands into an IN list.
    query, _ignore, _ret = _rewrite_pg_to_oracle(archived_sql)
    params = {str(i + 1): v for i, v in enumerate(_convert_args((str(unit), BANK, [CAUSAL, "causes"])))}
    expanded, bound = OracleConnection._expand_any_lists(query, params)
    assert isinstance(params["1"], bytes) and len(params["1"]) == 16
    assert "/*EXPAND" not in expanded and sorted(v for v in bound.values() if isinstance(v, str)) == sorted(
        [BANK, CAUSAL, "causes"]
    )


@pytest.mark.asyncio
async def test_rematerialize_keeps_only_edges_whose_endpoints_are_live_on_oracle(oracle_backend):
    live_a, live_b, archived = (uuid.uuid4() for _ in range(3))

    def answer(method, query, args):
        if method == "fetch" and "FROM memory_units" in query:
            return [{"id": live_a}, {"id": live_b}]
        raise AssertionError(query)

    conn = _ScriptedOracle(answer)
    submitted = await links.rematerialize_causal_links(
        conn,
        BANK,
        [
            {"from_unit_id": str(live_a), "to_unit_id": str(live_b), "link_type": CAUSAL, "weight": 1.0},
            {"from_unit_id": str(archived), "to_unit_id": str(live_a), "link_type": CAUSAL, "weight": 1.0},
            {"from_unit_id": "not-a-uuid", "to_unit_id": str(live_a), "link_type": CAUSAL, "weight": 1.0},
        ],
        ops=OracleOps(),
    )

    assert submitted == 1
    (inserted,) = conn.sent("executemany", "INSERT INTO memory_links")
    assert [(row[0], row[1]) for row in inserted] == [(str(live_a), str(live_b))]
    (probe,) = conn.sent("fetch", "FROM memory_units")
    assert sorted(probe[1]) == sorted(str(u) for u in (live_a, live_b, archived))
    _assert_speaks_oracle(conn.statements)


# --------------------------------------------------------------------------- merged_into fold


def _fold_answer(text: str, stored: list[str] | None):
    def answer(method, query, args):
        if method == "fetchrow" and "FOR UPDATE" in query:
            return None if stored is None else {"text": text, "source_memory_ids": stored}
        if method == "execute" and query.lstrip().startswith("UPDATE"):
            return "UPDATE 1"
        raise AssertionError(query)

    return answer


@pytest.mark.asyncio
async def test_the_oracle_fold_rewrites_the_json_list_recounts_proof_and_posts_the_junction(oracle_backend):
    from hindsight_api.engine.consolidation.consolidator import _TemporalBounds
    from hindsight_api.engine.memories.pg.consolidation import fold_sources_into_observation

    observation, kept, new = (uuid.uuid4() for _ in range(3))
    conn = _ScriptedOracle(_fold_answer("old text", [str(kept)]))
    folded = await fold_sources_into_observation(
        conn=conn,
        fq_table=_bare,
        observation_id=str(observation),
        expected_text="old text",
        merged_text="merged text",
        source_memory_ids=[new, kept, new],
        bounds=_TemporalBounds(event_date=NAIVE, occurred_start=None, occurred_end=None, mentioned_at=NAIVE),
    )

    assert folded
    (update,) = conn.sent("execute", "UPDATE memory_units")
    assert update[0] == "merged text"
    assert update[1] == [kept, new] and update[2] == 2  # distinct union, proof_count its length
    assert update[3] == NAIVE and update[4] is None and update[7] == observation
    (junction,) = conn.sent("executemany", "observation_sources")
    assert list(junction) == [(observation, new), (observation, kept)]
    update_sql = next(q for q in conn.statements if q.lstrip().startswith("UPDATE"))
    assert "COALESCE(LEAST(event_date, COALESCE($4, event_date)), $4)" in update_sql
    assert "search_vector" not in update_sql
    _assert_speaks_oracle(conn.statements)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_text, stored", [("rewritten meanwhile", ["x"]), ("old text", None)])
async def test_the_oracle_fold_misses_when_the_survivor_changed_or_vanished(oracle_backend, stored_text, stored):
    from hindsight_api.engine.consolidation.consolidator import _TemporalBounds
    from hindsight_api.engine.memories.pg.consolidation import fold_sources_into_observation

    conn = _ScriptedOracle(_fold_answer(stored_text, stored))
    assert not await fold_sources_into_observation(
        conn=conn,
        fq_table=_bare,
        observation_id=str(uuid.uuid4()),
        expected_text="old text",
        merged_text="merged",
        source_memory_ids=[uuid.uuid4()],
        bounds=_TemporalBounds(),
    )
    assert [m for m, _q, _a in conn.calls] == ["fetchrow"]  # no write, no junction rows


# --------------------------------------------------------------------------- invalidate -> restore


class _ArchiveScenario:
    """One memory ``m`` with adaptive state about a live peer ``p`` and a deleted one ``g``."""

    def __init__(self):
        self.m, self.p, self.g, self.entity = sorted(uuid.uuid4() for _ in range(4))
        self.rel_out, self.rel_in = sorted(uuid.uuid4() for _ in range(2))
        self.archive: dict[str, Any] = {}

    def relation(self, relation_id, source, target) -> dict[str, Any]:
        return {
            "id": relation_id,
            "relation_type": "supersedes",
            "source_memory_id": source,
            "target_memory_id": target,
            "source_snapshot": json.dumps({"text": "Java 25"}),
            "target_snapshot": json.dumps({"text": "Java 21"}),
            "origin": "consolidation",
            "reason": "newer release",
            "confidence": 0.9,
            "evidence": json.dumps({"facts": []}),
            "created_at": NAIVE,
        }

    def invalidate_answer(self, method, query, args):
        m, p, g = self.m, self.p, self.g
        if "all_tab_columns" in query:
            return [{"column_name": c} for c in ("ID", "BANK_ID", "TEXT", "EMBEDDING", "SEARCH_VECTOR")]
        if "FROM unit_entities" in query:
            return [{"entity_id": self.entity}]
        if "FROM memory_links" in query:
            return [{"from_unit_id": m, "to_unit_id": p, "link_type": CAUSAL, "weight": 1.0}]
        if "JSON_TABLE" in query:
            return []
        if method == "execute" and "INSERT INTO invalidated_memory_units" in query:
            self.archive["causal_links"] = args[4]
            return "INSERT 0 1"
        if "UNION ALL" in query:
            return 1
        if "FROM memory_retention_state" in query:
            return {
                "importance": 0.8,
                "stability_days": 12.5,
                "last_reinforced_at": NAIVE,
                "last_recalled_at": None,
                "reinforcement_count": 3,
                "access_count": 1,
                "forgetting_exempt": 1,  # NUMBER(1)
                "exemption_reason": "hold",
                "retention_state": "active",
                "below_threshold_since": None,
                "archived_at": None,
                "model_version": 1,
                "created_at": NAIVE,
                "updated_at": NAIVE,
            }
        if method == "fetchrow" and "FROM memory_claim_state" in query:
            return {
                "truth_status": "superseded",
                "confidence": None,
                "valid_from": None,
                "valid_to": NAIVE,
                "retired_via": "consolidation",
                "superseded_by_id": p,
                "inherited_from_id": None,
                "scope_fingerprint": "[]",
                "reason": "replaced",
                "created_at": NAIVE,
                "updated_at": NAIVE,
            }
        if "FROM memory_reinforcement_events" in query:
            return [
                {
                    "source_type": "explicit_feedback",
                    "source_key": "k1",
                    "event_time": NAIVE,
                    "weight": 1.0,
                    "independence_weight": 1.0,
                    "applied_gain": 0.2,
                    "outcome": "applied",
                    "processed_at": NAIVE,
                    "created_at": NAIVE,
                }
            ]
        if "FROM memory_claim_evidence WHERE memory_id" in query:
            return [
                {"memory_id": m, "evidence_memory_id": p, "created_at": NAIVE},
                {"memory_id": m, "evidence_memory_id": g, "created_at": NAIVE},
            ]
        if "FROM memory_claim_evidence WHERE evidence_memory_id" in query:
            return [{"memory_id": p, "evidence_memory_id": m, "created_at": NAIVE}]
        if "WHERE superseded_by_id" in query or "WHERE inherited_from_id" in query:
            return []
        if "FROM memory_lifecycle_relations WHERE source_memory_id" in query:
            return [self.relation(self.rel_out, m, p)]
        if "FROM memory_lifecycle_relations WHERE target_memory_id" in query:
            return [self.relation(self.rel_in, g, m)]
        if method == "execute" and "SET adaptive_state" in query:
            self.archive["adaptive_state"] = args[2]
            return "UPDATE 1"
        if method == "execute":
            return None
        raise AssertionError((method, query))

    def restore_answer(self, method, query, args):
        m, p = self.m, self.p
        if "all_tab_columns" in query:
            return [{"column_name": c} for c in ("ID", "BANK_ID", "TEXT", "EMBEDDING", "SEARCH_VECTOR")]
        if method == "fetchrow" and "FROM invalidated_memory_units" in query:
            return {
                "id": m,
                "text": "Uses Java 21",
                "fact_type": "world",
                "context": None,
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": NAIVE,
                "document_id": None,
                "chunk_id": None,
                "tags": [],
                "metadata": {},
                "proof_count": 1,
                "event_date": NAIVE,
                "created_at": NAIVE,
                "consolidated_at": None,
                # A CLOB of JSON text on Oracle, not a uuid[].
                "entity_ids": json.dumps([str(self.entity)]),
            }
        if method == "fetch" and "FROM entities" in query:
            return [{"id": self.entity}]
        if method == "fetchval" and "SELECT causal_links" in query:
            return self.archive["causal_links"]
        if method == "fetch" and "FROM memory_units WHERE bank_id = $1 AND id = ANY" in query:
            return [{"id": m}, {"id": p}]
        if method == "fetchval" and "SELECT adaptive_state" in query:
            return self.archive["adaptive_state"]
        if method == "fetchrow" and "FROM memory_units WHERE id = $1" in query:
            return {"id": args[0]} if args[0] == p else None
        if method == "fetch" and "FROM memory_reinforcement_events WHERE id" in query:
            return [{"id": i} for i in args[0]]
        if method == "fetchrow" and "FROM memory_claim_state" in query:
            return {
                "memory_id": args[0],
                "superseded_by_id": None,
                "inherited_from_id": None,
                "updated_at": NAIVE - timedelta(days=1),
            }
        if method == "fetchrow" and "FROM memory_lifecycle_relations" in query:
            # rel_out lost its source to the SET NULL cascade; rel_in is gone altogether.
            return {"source_memory_id": None, "target_memory_id": p} if args[0] == self.rel_out else None
        if method == "fetchval" and "INSERT INTO memory_lifecycle_relations" in query:
            return args[0]
        if method == "fetchval" and "SELECT 1 FROM memory_lifecycle_relations" in query:
            return None
        if method == "execute":
            return None
        raise AssertionError((method, query))


@pytest.mark.asyncio
async def test_invalidate_parks_the_adaptive_state_on_oracle(oracle_backend):
    s = _ArchiveScenario()
    conn = _ScriptedOracle(s.invalidate_answer)

    assert await writes.invalidate_memory(conn=conn, fq_table=_bare, bank_id=BANK, unit_id=str(s.m), reason="wrong")

    (insert,) = conn.sent("execute", "INSERT INTO invalidated_memory_units")
    insert_sql = next(q for q in conn.statements if "INSERT INTO invalidated_memory_units" in q)
    assert '"ID", "BANK_ID", "TEXT", invalidation_reason' in insert_sql and "RETURNING" not in insert_sql
    assert insert[2] == [s.entity]
    assert json.loads(s.archive["causal_links"])[0]["to_unit_id"] == str(s.p)
    snapshot = AdaptiveSnapshot.model_validate_json(s.archive["adaptive_state"])
    assert snapshot.memory_id == str(s.m)
    assert snapshot.retention is not None and snapshot.retention.forgetting_exempt is True
    assert snapshot.retention.last_reinforced_at == NAIVE.replace(tzinfo=UTC)  # zone-less read as UTC
    assert snapshot.claim is not None and snapshot.claim.superseded_by_id == str(s.p)
    assert [e.evidence_memory_id for e in snapshot.evidence_out] == [str(s.p), str(s.g)]
    assert [e.memory_id for e in snapshot.evidence_in] == [str(s.p)]
    assert sorted(r.id for r in snapshot.relations) == sorted(str(r) for r in (s.rel_out, s.rel_in))
    assert snapshot.relations[0].source_snapshot == {"text": "Java 25"}
    # The parent lock comes before the snapshot reads, the DELETE last.
    lock = next(i for i, q in enumerate(conn.statements) if "FOR UPDATE" in q)
    probe = next(i for i, q in enumerate(conn.statements) if "UNION ALL" in q)
    assert lock < probe and conn.statements[-1].startswith("DELETE FROM memory_units")
    _assert_speaks_oracle(conn.statements)


@pytest.mark.asyncio
async def test_restore_rematerializes_the_parked_state_on_oracle(oracle_backend):
    s = _ArchiveScenario()
    await writes.invalidate_memory(
        conn=_ScriptedOracle(s.invalidate_answer), fq_table=_bare, bank_id=BANK, unit_id=str(s.m), reason=None
    )
    conn = _ScriptedOracle(s.restore_answer)

    restored = await writes.restore_memory(conn=conn, fq_table=_bare, bank_id=BANK, unit_id=str(s.m))

    assert restored is not None and restored.entity_ids == [str(s.entity)]
    assert not any("search_vector" in q for q in conn.statements)  # Oracle Text syncs on commit
    (postings,) = conn.sent("executemany", "INTO unit_entities")
    assert [row[1] for row in postings] == [str(s.entity)]
    # The causal edge to the live peer comes back.
    (edges,) = conn.sent("executemany", "INSERT INTO memory_links")
    assert [(row[0], row[1]) for row in edges] == [(str(s.m), str(s.p))]
    # Peers are locked one row at a time FOR UPDATE, in id order; the deleted one is dropped.
    peer_locks = [a[0] for a in conn.sent("fetchrow", "FROM memory_units WHERE id = $1")]
    assert peer_locks == sorted([s.p, s.g])
    (retention,) = conn.sent("execute", "INTO memory_retention_state")
    assert retention[8] == 1 and retention[9] == "hold"  # NUMBER(1), not a Python bool
    (claim,) = conn.sent("execute", "INTO memory_claim_state")
    assert claim[2] == "superseded" and claim[7] == s.p
    (evidence,) = conn.sent("executemany", "INTO memory_claim_evidence")
    assert sorted((row[0], row[1]) for row in evidence) == sorted([(s.m, s.p), (s.p, s.m)])
    evidence_sql = next(q for q in conn.statements if "INTO memory_claim_evidence" in q)
    assert "IGNORE_ROW_ON_DUPKEY_INDEX(memory_claim_evidence, pk_memory_claim_evidence)" in evidence_sql
    # rel_out is re-pointed at the restored memory; rel_in's other endpoint is gone, so it stays dropped.
    (repoint,) = conn.sent("execute", "UPDATE memory_lifecycle_relations")
    assert repoint == (s.rel_out, s.m, s.p)
    assert not conn.sent("fetchval", "INSERT INTO memory_lifecycle_relations")
    # A superseded claim keeps the restored raw fact out of the consolidation queue.
    assert conn.sent("execute", "SET consolidated_at = $3")
    assert conn.statements[-1].startswith("DELETE FROM invalidated_memory_units")
    _assert_speaks_oracle(conn.statements)


@pytest.mark.asyncio
async def test_the_archived_lifecycle_view_reads_the_oracle_snapshot(oracle_backend):
    from hindsight_api.config import get_config
    from hindsight_api.engine.adaptive.settings import AdaptiveSettings

    s = _ArchiveScenario()
    await writes.invalidate_memory(
        conn=_ScriptedOracle(s.invalidate_answer), fq_table=_bare, bank_id=BANK, unit_id=str(s.m), reason=None
    )
    row = {"fact_type": "world", "mentioned_at": NAIVE, "occurred_start": None, "event_date": NAIVE}
    conn = _ScriptedOracle(lambda method, query, args: {**row, "adaptive_state": s.archive["adaptive_state"]})

    view = await adaptive_archive.archived_lifecycle(
        conn,
        fq_table=_bare,
        bank_id=BANK,
        memory_uuid=s.m,
        now=datetime.now(UTC),
        settings=AdaptiveSettings.resolve({"adaptive_memory_mode": "lifecycle"}, get_config(), store_supports=True),
    )

    assert view is not None and view.archived and view.stored
    assert view.claim.superseded_by_id == str(s.p)
    assert [r.id for r in view.relations_out] == [str(s.rel_out)] and [r.id for r in view.relations_in] == [
        str(s.rel_in)
    ]


@pytest.mark.asyncio
async def test_a_memory_without_adaptive_state_parks_nothing_on_oracle(oracle_backend):
    def answer(method, query, args):
        if "UNION ALL" in query:
            return None
        if method == "execute":
            return None
        raise AssertionError(query)

    conn = _ScriptedOracle(answer)
    assert not await adaptive_archive.park_adaptive_state(conn, fq_table=_bare, bank_id=BANK, unit_id=str(uuid.uuid4()))
    assert not any("adaptive_state" in q for q in conn.statements)
    _assert_speaks_oracle(conn.statements)


@pytest.mark.asyncio
async def test_the_curation_edit_and_re_embed_write_plain_binds_on_oracle(oracle_backend):
    """``$n::vector`` loses its cast (Oracle converts the vector text on write, as retain relies on) and
    an edit writes no ``search_vector``, whatever text-search backend the config names."""
    conn = _ScriptedOracle(lambda method, query, args: None)
    unit = str(uuid.uuid4())

    await writes.set_memory_embedding(conn=conn, fq_table=_bare, bank_id=BANK, unit_id=unit, embedding="[0.1, 0.2]")
    await writes.apply_edit(
        conn=conn,
        fq_table=_bare,
        bank_id=BANK,
        unit_id=unit,
        text="edited",
        context=None,
        fact_type="world",
        occurred_start=None,
        occurred_end=None,
        event_date=NAIVE,
        mentioned_at=None,
        entity_ids=None,
        embedding="[0.3, 0.4]",
    )

    edit_sql = next(q for q in conn.statements if "SET text = $3" in q)
    assert "search_vector" not in edit_sql and "embedding = $9::vector" in edit_sql
    assert "embedding = :9" in _rewrite_pg_to_oracle(edit_sql).query
    _assert_speaks_oracle(conn.statements)


@pytest.mark.asyncio
async def test_the_state_load_and_lifecycle_view_take_oracle_rows(oracle_backend):
    """Zone-less timestamps, NUMBER(1) booleans and CLOB JSON as the Oracle driver returns them."""
    from hindsight_api.config import get_config
    from hindsight_api.engine.adaptive.settings import AdaptiveSettings
    from hindsight_api.engine.memories.pg.adaptive import get_memory_lifecycle, load_adaptive_state

    memory, successor = uuid.uuid4(), uuid.uuid4()
    retention = {
        "memory_id": memory,
        "importance": 0.5,
        "stability_days": 3.0,
        "last_reinforced_at": NAIVE,
        "reinforcement_count": 1,
        "access_count": 0,
        "forgetting_exempt": 0,
        "exemption_reason": None,
        "retention_state": "active",
        "last_recalled_at": None,
        "below_threshold_since": None,
        "archived_at": None,
        "updated_at": NAIVE,
    }
    claim = {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": None,
        "valid_to": NAIVE,
        "retired_via": "consolidation",
        "superseded_by_id": successor,
        "inherited_from_id": None,
        "scope_fingerprint": None,
        "reason": None,
        "updated_at": NAIVE,
    }

    def answer(method, query, args):
        if "FULL OUTER JOIN" in query:
            return [
                {
                    **{k: v for k, v in retention.items() if k != "memory_id"},
                    **claim,
                    "retention_memory_id": memory,
                    "claim_memory_id": memory,
                }
            ]
        if "FROM memory_units" in query:
            return {"fact_type": "world", "mentioned_at": NAIVE, "occurred_start": None, "event_date": NAIVE}
        if "FROM memory_retention_state" in query:
            return retention
        if "FROM memory_claim_state" in query:
            return claim
        if "FROM memory_claim_evidence" in query:
            return [{"evidence_memory_id": successor}]
        if "FROM memory_lifecycle_relations" in query:
            return []
        if "FROM memory_reinforcement_events" in query:
            return [
                {
                    "source_type": "explicit_feedback",
                    "source_key": "k",
                    "event_time": NAIVE,
                    "weight": 1.0,
                    "independence_weight": 1.0,
                    "applied_gain": 0.1,
                    "outcome": "applied",
                    "processed_at": NAIVE,
                }
            ]
        if "FROM memory_claim_shadow" in query:
            return None
        raise AssertionError(query)

    conn = _ScriptedOracle(answer)
    states = await load_adaptive_state(conn, fq_table=_bare, bank_id=BANK, memory_ids=[memory])
    assert states[str(memory)].claim.superseded_by_id == str(successor)
    assert states[str(memory)].retention.forgetting_exempt is False

    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": "lifecycle"}, get_config(), store_supports=True)
    view = await get_memory_lifecycle(
        conn, fq_table=_bare, bank_id=BANK, memory_id=str(memory), now=NAIVE + timedelta(days=2), settings=settings
    )
    assert view is not None and view.claim.truth_status == "superseded"
    assert view.claim.evidence_memory_ids == [str(successor)] and view.recent_events
    _assert_speaks_oracle(conn.statements)


# --------------------------------------------------------------------------- live Oracle 23ai


async def _seed_unit(conn, bank_id: str, text: str, fact_type: str = "world") -> uuid.UUID:
    unit_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, event_date, fact_type) VALUES ($1, $2, $3, $4, $5)",
        unit_id,
        bank_id,
        text,
        datetime.now(UTC),
        fact_type,
    )
    return unit_id


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_oracle_invalidate_restore_round_trip_keeps_adaptive_state_and_causal_edges(oracle_memory):
    from hindsight_api.engine.memories.pg.adaptive_claims import add_claim_evidence, cas_retire, lock_claims

    backend = await oracle_memory._get_backend()
    bank_id = f"ora-curation-{uuid.uuid4().hex[:8]}"
    async with backend.transaction() as conn:
        old = await _seed_unit(conn, bank_id, "The service runs on Java 21")
        new = await _seed_unit(conn, bank_id, "The service runs on Java 25")
        await conn.execute(
            "INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id) VALUES ($1, $2, $3, 1.0, $4)",
            old,
            new,
            CAUSAL,
            bank_id,
        )
        await lock_claims(conn, fq_table=_bare, bank_id=bank_id, memory_ids=[old, new], skip_locked=False)
        assert await cas_retire(
            conn,
            fq_table=_bare,
            bank_id=bank_id,
            memory_id=str(old),
            to_status="superseded",
            retired_via="consolidation",
            superseded_by_id=str(new),
            valid_to=None,
            reason=None,
            confidence=None,
        )
        assert await add_claim_evidence(
            conn, fq_table=_bare, bank_id=bank_id, memory_id=str(old), evidence_ids=[str(new)]
        )
    try:
        async with backend.transaction() as conn:
            assert await writes.invalidate_memory(
                conn=conn, fq_table=_bare, bank_id=bank_id, unit_id=str(old), reason="test"
            )
        async with backend.acquire() as conn:
            raw = await conn.fetchval("SELECT adaptive_state FROM invalidated_memory_units WHERE id = $1", old)
            assert AdaptiveSnapshot.model_validate(conn.parse_json(raw)).claim.superseded_by_id == str(new)
            assert await conn.fetchval("SELECT COUNT(*) FROM memory_claim_state WHERE memory_id = $1", old) == 0
        async with backend.transaction() as conn:
            assert await writes.restore_memory(conn=conn, fq_table=_bare, bank_id=bank_id, unit_id=str(old))
        async with backend.acquire() as conn:
            claim = await conn.fetchrow(
                "SELECT truth_status, superseded_by_id FROM memory_claim_state WHERE memory_id = $1", old
            )
            assert claim["truth_status"] == "superseded" and str(claim["superseded_by_id"]) == str(new)
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM memory_claim_evidence WHERE memory_id = $1 AND evidence_memory_id = $2",
                    old,
                    new,
                )
                == 1
            )
            assert (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM memory_links WHERE from_unit_id = $1 AND to_unit_id = $2", old, new
                )
                == 1
            )
            assert await conn.fetchval("SELECT COUNT(*) FROM invalidated_memory_units WHERE id = $1", old) == 0
    finally:
        async with backend.transaction() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM invalidated_memory_units WHERE bank_id = $1", bank_id)


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_oracle_merged_into_fold_moves_sources_into_the_survivor(oracle_memory):
    from hindsight_api.engine.consolidation.consolidator import _TemporalBounds
    from hindsight_api.engine.memories.pg.adaptive_claims import observation_sources
    from hindsight_api.engine.memories.pg.consolidation import fold_sources_into_observation

    backend = await oracle_memory._get_backend()
    bank_id = f"ora-fold-{uuid.uuid4().hex[:8]}"
    async with backend.transaction() as conn:
        kept = await _seed_unit(conn, bank_id, "fact one")
        moved = await _seed_unit(conn, bank_id, "fact two")
        survivor = uuid.uuid4()
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, event_date, fact_type, source_memory_ids, proof_count) "
            "VALUES ($1, $2, $3, $4, 'observation', $5, 1)",
            survivor,
            bank_id,
            "the survivor",
            datetime.now(UTC),
            [kept],
        )
        await conn.execute(
            "INSERT INTO observation_sources (observation_id, source_id) VALUES ($1, $2)", survivor, kept
        )
    try:
        async with backend.transaction() as conn:
            assert await fold_sources_into_observation(
                conn=conn,
                fq_table=_bare,
                observation_id=str(survivor),
                expected_text="the survivor",
                merged_text="the survivor",
                source_memory_ids=[moved],
                bounds=_TemporalBounds(),
            )
            assert not await fold_sources_into_observation(
                conn=conn,
                fq_table=_bare,
                observation_id=str(survivor),
                expected_text="stale text",
                merged_text="x",
                source_memory_ids=[moved],
                bounds=_TemporalBounds(),
            )
        async with backend.acquire() as conn:
            sources = await observation_sources(conn, fq_table=_bare, bank_id=bank_id, observation_ids=[survivor])
            assert sorted(sources[str(survivor)]) == sorted([str(kept), str(moved)])
            assert await conn.fetchval("SELECT proof_count FROM memory_units WHERE id = $1", survivor) == 2
    finally:
        async with backend.transaction() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
