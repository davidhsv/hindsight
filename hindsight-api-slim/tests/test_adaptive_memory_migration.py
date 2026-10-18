"""Migration ``a7d3e91c5f42``: the adaptive-memory side tables.

What the schema itself must guarantee, independent of the code that will write it:

* rows that belong to one memory die with it (CASCADE), while the references to
  *other* memories (a claim's successor, a relation's endpoints) are nulled, so a
  claim or an audit relation outlives the memory it pointed at;
* several relation tombstones of the same shape can coexist — otherwise the
  second endpoint delete that produces one would fail the parent DELETE;
* the CASCADE FKs are checked at COMMIT on PostgreSQL (the ``9f8e7d6c5b4a``
  deadlock fix), the SET NULL ones at the statement;
* CHECK constraints reject values outside the closed vocabularies;
* the curation archive gains ``adaptive_state`` while ``memory_units`` is untouched;
* the migration downgrades cleanly, re-applies over itself, and lands in the
  tenant schema it is run for.

These are DDL properties, so the tests read and write the tables with raw SQL on a
dedicated pg0 instance (like test_migration_drop_access_count): what is under test is
what the database enforces whichever code writes the rows. The Oracle half runs under ``-m oracle``
against ``ORACLE_TEST_DSN`` (``scripts/dev/start-oracle.sh``).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")
_REVISION = "a7d3e91c5f42"
_PRE_REVISION = "d4f7a2c9e1b5"

_TABLES = (
    "memory_retention_state",
    "memory_claim_state",
    "memory_claim_evidence",
    "memory_lifecycle_relations",
    "memory_reinforcement_events",
    "memory_claim_shadow",
)
_PG_INDEXES = {
    "idx_am_rs_bank_reinforced",
    "idx_am_rs_bank_state",
    "idx_am_cs_bank_status",
    "idx_am_cs_superseded_by",
    "idx_am_cs_inherited_from",
    "idx_am_ce_evidence",
    "uq_mlr_edge",
    "idx_am_rel_target",
    "idx_am_rel_bank_created",
    "uq_mre_key",
    "idx_am_ev_bank_created",
    "idx_am_ev_unprocessed",
}
# (table, constraint) -> (ON DELETE action, deferred on PostgreSQL).
_FKS = {
    ("memory_retention_state", "fk_mrs_memory"): ("CASCADE", True),
    ("memory_claim_state", "fk_mcs_memory"): ("CASCADE", True),
    ("memory_claim_state", "fk_mcs_superseded_by"): ("SET NULL", False),
    ("memory_claim_state", "fk_mcs_inherited_from"): ("SET NULL", False),
    ("memory_claim_evidence", "fk_mce_memory"): ("CASCADE", True),
    ("memory_claim_evidence", "fk_mce_evidence"): ("CASCADE", True),
    ("memory_lifecycle_relations", "fk_mlr_source"): ("SET NULL", False),
    ("memory_lifecycle_relations", "fk_mlr_target"): ("SET NULL", False),
    ("memory_reinforcement_events", "fk_mre_memory"): ("CASCADE", True),
    ("memory_claim_shadow", "fk_mcsh_memory"): ("CASCADE", True),
}
_PG_DELETE_ACTIONS = {"c": "CASCADE", "n": "SET NULL"}

_NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

# PostgreSQL SQLSTATEs.
_CHECK_VIOLATION = "23514"
_FK_VIOLATION = "23503"
_UNIQUE_VIOLATION = "23505"
# Oracle error numbers.
_ORA_UNIQUE = 1
_ORA_CHECK = 2290
_ORA_FK_PARENT_MISSING = 2291


def _alembic_cfg(db_url: str, *, target_schema: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    if target_schema:
        cfg.set_main_option("target_schema", target_schema)
    return cfg


# --------------------------------------------------------------------------- rows
#
# One valid row per table, built from the memory ids it references, so each CHECK
# case below differs from an accepted row in exactly one column. Values are plain
# Python; each dialect section adapts uuids, booleans and JSON to its driver.


def _valid_rows(subject: uuid.UUID, other: uuid.UUID, bank_id: str) -> dict[str, dict[str, Any]]:
    return {
        "memory_retention_state": {
            "memory_id": subject,
            "bank_id": bank_id,
            "importance": 0.5,
            "stability_days": 30.0,
            "last_reinforced_at": _NOW,
            "retention_state": "active",
            "created_at": _NOW,
            "updated_at": _NOW,
        },
        "memory_claim_state": {
            "memory_id": subject,
            "bank_id": bank_id,
            "truth_status": "superseded",
            "confidence": 0.9,
            "retired_via": "supersedes",
            "superseded_by_id": other,
            "inherited_from_id": other,
            "created_at": _NOW,
            "updated_at": _NOW,
        },
        "memory_claim_evidence": {
            "memory_id": subject,
            "evidence_memory_id": other,
            "bank_id": bank_id,
            "created_at": _NOW,
        },
        "memory_lifecycle_relations": {
            "id": uuid.uuid4(),
            "bank_id": bank_id,
            "relation_type": "supersedes",
            "source_memory_id": other,
            "target_memory_id": subject,
            "origin": "consolidation",
            "created_at": _NOW,
        },
        "memory_reinforcement_events": {
            "id": uuid.uuid4(),
            "bank_id": bank_id,
            "memory_id": subject,
            "source_type": "explicit_confirmation",
            "source_key": "explicit:req-1",
            "event_time": _NOW,
            "weight": 1.0,
            "outcome": "applied",
            "created_at": _NOW,
        },
        "memory_claim_shadow": {
            "memory_id": subject,
            "bank_id": bank_id,
            "would_status": "refuted",
            "created_at": _NOW,
        },
    }


# (table, column, value outside its CHECK).
_BAD_VALUES = [
    ("memory_retention_state", "importance", 1.5),
    ("memory_retention_state", "importance", -0.1),
    ("memory_retention_state", "stability_days", 0.0),
    ("memory_retention_state", "retention_state", "deleted"),
    ("memory_claim_state", "truth_status", "stale"),
    ("memory_claim_state", "confidence", 1.2),
    ("memory_claim_state", "retired_via", "deleted"),
    ("memory_lifecycle_relations", "relation_type", "replaces"),
    ("memory_lifecycle_relations", "origin", "llm"),
    ("memory_reinforcement_events", "source_type", "recall"),
    ("memory_reinforcement_events", "outcome", "skipped"),
    ("memory_claim_shadow", "would_status", "active"),
]


def _insert_sql(table: str, row: dict[str, Any]) -> str:
    # The JSON columns are left to their defaults, so every value binds as is.
    cols = list(row)
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(f':{c}' for c in cols)})"


# ===================================================================== PostgreSQL

pg_tests = pytest.mark.xdist_group("adaptive-memory-migration-pg0")


def _reset_public_schema(db_url: str) -> None:
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def pg_url() -> str:
    """A dedicated pg0 database migrated to head from an empty schema."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 pick a free port; a fixed one is not xdist-safe.
    pg0 = EmbeddedPostgres(name="hindsight-adaptive-memory-migration-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()
    _reset_public_schema(url)
    command.upgrade(_alembic_cfg(url), "heads")
    return url


@pytest.fixture
def pg_engine(pg_url: str) -> Iterator[Engine]:
    engine = create_engine(pg_url)
    try:
        yield engine
    finally:
        engine.dispose()


def _pg_unit(conn: Connection, bank_id: str) -> uuid.UUID:
    unit_id = uuid.uuid4()
    conn.execute(
        text("INSERT INTO memory_units (id, bank_id, text, fact_type) VALUES (:id, :b, :t, 'world')"),
        {"id": unit_id, "b": bank_id, "t": f"fact {unit_id}"},
    )
    return unit_id


def _pg_bank(conn: Connection) -> str:
    bank_id = f"am-{uuid.uuid4().hex[:8]}"
    conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": bank_id})
    return bank_id


def _pg_insert(conn: Connection, table: str, row: dict[str, Any]) -> None:
    conn.execute(text(_insert_sql(table, row)), row)


def _pg_count(conn: Connection, sql: str, **params: Any) -> int:
    return int(conn.execute(text(sql), params).scalar_one())


def _sqlstate(exc: DBAPIError) -> str | None:
    return getattr(exc.orig, "pgcode", None)


def _columns(conn: Connection, table: str, schema: str = "public") -> dict[str, tuple[str, str]]:
    rows = conn.execute(
        text(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t"
        ),
        {"s": schema, "t": table},
    )
    return {r[0]: (r[1], r[2]) for r in rows}


def _pg_foreign_keys(conn: Connection, schema: str) -> dict[tuple[str, str], tuple[str, bool, str]]:
    """(table, constraint) -> (delete action, initially deferred, referenced table) for the new tables."""
    rows = conn.execute(
        text(
            """
            SELECT cl.relname, c.conname, c.confdeltype, c.condeferred, c.confrelid::regclass::text
            FROM pg_constraint c
            JOIN pg_class cl ON cl.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = cl.relnamespace
            WHERE c.contype = 'f' AND n.nspname = :s AND cl.relname = ANY(:tables)
            """
        ),
        {"s": schema, "tables": list(_TABLES)},
    )
    return {(r[0], r[1]): (_PG_DELETE_ACTIONS.get(r[2], r[2]), r[3], r[4]) for r in rows}


@pg_tests
def test_upgrade_creates_the_tables_and_leaves_memory_units_alone(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
        }
        assert set(_TABLES) <= tables

        indexes = {
            r[0]
            for r in conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = ANY(:t)"), {"t": list(_TABLES)}
            )
        }
        assert _PG_INDEXES <= indexes, f"missing indexes: {sorted(_PG_INDEXES - indexes)}"

        fks = _pg_foreign_keys(conn, "public")
        assert {key: (action, deferred) for key, (action, deferred, _ref) in fks.items()} == _FKS
        assert {ref for _action, _deferred, ref in fks.values()} == {"memory_units"}

        assert _columns(conn, "invalidated_memory_units")["adaptive_state"] == ("jsonb", "YES")
        # Archive only: the live table gets no column, so the curation
        # INSERT…SELECT (column list read from memory_units) is unchanged.
        assert "adaptive_state" not in _columns(conn, "memory_units")


@pg_tests
def test_valid_rows_are_accepted(pg_engine: Engine) -> None:
    """The baseline every CHECK case below departs from in one column."""
    with pg_engine.connect() as conn:
        bank_id = _pg_bank(conn)
        subject, other = _pg_unit(conn, bank_id), _pg_unit(conn, bank_id)
        for table, row in _valid_rows(subject, other, bank_id).items():
            _pg_insert(conn, table, row)
        # Server-side defaults the writers rely on.
        defaults = conn.execute(
            text(
                "SELECT reinforcement_count, access_count, forgetting_exempt, model_version "
                "FROM memory_retention_state WHERE memory_id = :m"
            ),
            {"m": subject},
        ).one()
        assert tuple(defaults) == (0, 0, False, 1)
        snapshots = conn.execute(
            text(
                "SELECT source_snapshot, target_snapshot, evidence FROM memory_lifecycle_relations WHERE bank_id = :b"
            ),
            {"b": bank_id},
        ).one()
        assert tuple(snapshots) == ({}, {}, {})
        assert (
            conn.execute(
                text("SELECT evidence_ids FROM memory_claim_shadow WHERE memory_id = :m"), {"m": subject}
            ).scalar_one()
            == []
        )
        conn.rollback()


@pg_tests
@pytest.mark.parametrize(("table", "column", "value"), _BAD_VALUES, ids=lambda v: str(v))
def test_check_constraints_reject_out_of_vocabulary_values(
    pg_engine: Engine, table: str, column: str, value: Any
) -> None:
    with pg_engine.connect() as conn:
        bank_id = _pg_bank(conn)
        subject, other = _pg_unit(conn, bank_id), _pg_unit(conn, bank_id)
        row = {**_valid_rows(subject, other, bank_id)[table], column: value}
        with pytest.raises(DBAPIError) as exc_info:
            _pg_insert(conn, table, row)
        assert _sqlstate(exc_info.value) == _CHECK_VIOLATION
        conn.rollback()


@pg_tests
def test_deleting_a_memory_cascades_its_rows_and_nulls_references_to_it(pg_engine: Engine) -> None:
    with pg_engine.begin() as conn:
        bank_id = _pg_bank(conn)
        subject, successor, evidence = (_pg_unit(conn, bank_id) for _ in range(3))
        rows = _valid_rows(subject, successor, bank_id)
        for table in (
            "memory_retention_state",
            "memory_claim_state",
            "memory_reinforcement_events",
            "memory_claim_shadow",
        ):
            _pg_insert(conn, table, rows[table])
        _pg_insert(conn, "memory_claim_evidence", {**rows["memory_claim_evidence"], "evidence_memory_id": evidence})
        relation_id = rows["memory_lifecycle_relations"]["id"]
        _pg_insert(conn, "memory_lifecycle_relations", rows["memory_lifecycle_relations"])

    with pg_engine.begin() as conn:
        # The successor goes: the claim and the relation stay, pointing at nothing.
        conn.execute(text("DELETE FROM memory_units WHERE id = :id"), {"id": successor})
        claim = conn.execute(
            text(
                "SELECT truth_status, superseded_by_id, inherited_from_id FROM memory_claim_state WHERE memory_id = :m"
            ),
            {"m": subject},
        ).one()
        assert tuple(claim) == ("superseded", None, None)
        relation = conn.execute(
            text("SELECT source_memory_id, target_memory_id FROM memory_lifecycle_relations WHERE id = :id"),
            {"id": relation_id},
        ).one()
        assert tuple(relation) == (None, subject)

        # The evidence fact goes: its evidence row goes with it, the claim stays.
        conn.execute(text("DELETE FROM memory_units WHERE id = :id"), {"id": evidence})
        assert _pg_count(conn, "SELECT count(*) FROM memory_claim_evidence WHERE memory_id = :m", m=subject) == 0
        assert _pg_count(conn, "SELECT count(*) FROM memory_claim_state WHERE memory_id = :m", m=subject) == 1

        # The memory itself goes: every row that is about it goes too, and the
        # relation survives as a tombstone with both endpoints NULL.
        conn.execute(text("DELETE FROM memory_units WHERE id = :id"), {"id": subject})
        for table in (
            "memory_retention_state",
            "memory_claim_state",
            "memory_reinforcement_events",
            "memory_claim_shadow",
        ):
            assert _pg_count(conn, f"SELECT count(*) FROM {table} WHERE bank_id = :b", b=bank_id) == 0, table
        relation = conn.execute(
            text("SELECT source_memory_id, target_memory_id FROM memory_lifecycle_relations WHERE id = :id"),
            {"id": relation_id},
        ).one()
        assert tuple(relation) == (None, None)


@pg_tests
def test_relation_tombstones_of_the_same_shape_coexist(pg_engine: Engine) -> None:
    """Two predecessors superseded by the same memory, both deleted, leave two
    ``(NULL, target, 'supersedes')`` rows. A live duplicate edge is still refused."""
    with pg_engine.begin() as conn:
        bank_id = _pg_bank(conn)
        target, first, second = (_pg_unit(conn, bank_id) for _ in range(3))
        base = _valid_rows(target, target, bank_id)["memory_lifecycle_relations"]
        for source in (first, second):
            _pg_insert(conn, "memory_lifecycle_relations", {**base, "id": uuid.uuid4(), "source_memory_id": source})
        conn.execute(text("DELETE FROM memory_units WHERE id = ANY(:ids)"), {"ids": [first, second]})
        assert (
            _pg_count(
                conn,
                "SELECT count(*) FROM memory_lifecycle_relations "
                "WHERE bank_id = :b AND source_memory_id IS NULL AND target_memory_id = :t",
                b=bank_id,
                t=target,
            )
            == 2
        )

    with pg_engine.connect() as conn:
        live = _pg_unit(conn, bank_id)
        _pg_insert(conn, "memory_lifecycle_relations", {**base, "id": uuid.uuid4(), "source_memory_id": live})
        with pytest.raises(DBAPIError) as exc_info:
            _pg_insert(conn, "memory_lifecycle_relations", {**base, "id": uuid.uuid4(), "source_memory_id": live})
        assert _sqlstate(exc_info.value) == _UNIQUE_VIOLATION
        conn.rollback()


@pg_tests
def test_reinforcement_events_are_unique_per_memory_source_and_key(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        bank_id = _pg_bank(conn)
        subject = _pg_unit(conn, bank_id)
        event = _valid_rows(subject, subject, bank_id)["memory_reinforcement_events"]
        _pg_insert(conn, "memory_reinforcement_events", event)
        # Same key under another source type is a different event.
        _pg_insert(
            conn,
            "memory_reinforcement_events",
            {**event, "id": uuid.uuid4(), "source_type": "successful_reflect_use"},
        )
        with pytest.raises(DBAPIError) as exc_info:
            _pg_insert(conn, "memory_reinforcement_events", {**event, "id": uuid.uuid4()})
        assert _sqlstate(exc_info.value) == _UNIQUE_VIOLATION
        conn.rollback()


@pg_tests
def test_cascade_fks_are_checked_at_commit_and_set_null_fks_at_the_statement(pg_engine: Engine) -> None:
    missing = uuid.uuid4()
    with pg_engine.connect() as conn:
        bank_id = _pg_bank(conn)
        conn.commit()
        row = _valid_rows(missing, missing, bank_id)["memory_retention_state"]
        # Deferred: the INSERT itself takes no lock on (and does not check) the parent.
        _pg_insert(conn, "memory_retention_state", row)
        with pytest.raises(DBAPIError) as exc_info:
            conn.commit()
        assert _sqlstate(exc_info.value) == _FK_VIOLATION

    with pg_engine.connect() as conn:
        subject = _pg_unit(conn, bank_id)
        claim = {**_valid_rows(subject, missing, bank_id)["memory_claim_state"], "inherited_from_id": None}
        with pytest.raises(DBAPIError) as exc_info:
            _pg_insert(conn, "memory_claim_state", claim)
        assert _sqlstate(exc_info.value) == _FK_VIOLATION
        conn.rollback()


@pg_tests
def test_downgrade_removes_everything_and_the_upgrade_reapplies_over_itself(pg_url: str, pg_engine: Engine) -> None:
    cfg = _alembic_cfg(pg_url)
    try:
        with pg_engine.connect() as conn:
            live_before = _columns(conn, "memory_units")
            archive_before = set(_columns(conn, "invalidated_memory_units"))

        command.downgrade(cfg, _PRE_REVISION)
        with pg_engine.connect() as conn:
            for table in _TABLES:
                assert not _columns(conn, table), f"downgrade left {table} behind"
            assert set(_columns(conn, "invalidated_memory_units")) == archive_before - {"adaptive_state"}
            assert _columns(conn, "memory_units") == live_before

        command.upgrade(cfg, _REVISION)
        # A rerun over objects that already exist (an interrupted earlier run, or a
        # tenant provisioned twice) must be a no-op, not an error.
        command.stamp(cfg, _PRE_REVISION)
        command.upgrade(cfg, _REVISION)
        with pg_engine.connect() as conn:
            for table in _TABLES:
                assert _columns(conn, table), f"re-upgrade did not recreate {table}"
            assert set(_columns(conn, "invalidated_memory_units")) == archive_before
            assert _columns(conn, "memory_units") == live_before
    finally:
        # Leave the module's database at head for the other tests.
        command.upgrade(cfg, "heads")


@pg_tests
def test_a_tenant_schema_gets_its_own_tables_referencing_its_own_memory_units(pg_url: str, pg_engine: Engine) -> None:
    schema = f"am_tenant_{uuid.uuid4().hex[:6]}"
    try:
        command.upgrade(_alembic_cfg(pg_url, target_schema=schema), "heads")
        with pg_engine.connect() as conn:
            for table in _TABLES:
                assert _columns(conn, table, schema), f"{table} missing from {schema}"
            assert "adaptive_state" in _columns(conn, "invalidated_memory_units", schema)
            fks = _pg_foreign_keys(conn, schema)
            assert set(fks) == set(_FKS)
            assert {ref for _action, _deferred, ref in fks.values()} == {f"{schema}.memory_units"}
    finally:
        engine = create_engine(pg_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            engine.dispose()


# ========================================================================= Oracle


def _ora_value(value: Any) -> Any:
    # RAW(16) ids bind as bytes, NUMBER(1) booleans as 0/1, timestamps as naive UTC.
    if isinstance(value, uuid.UUID):
        return value.bytes
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _ora_insert(cursor: Any, table: str, row: dict[str, Any]) -> None:
    cursor.execute(_insert_sql(table, row), {k: _ora_value(v) for k, v in row.items()})


def _ora_unit(cursor: Any, bank_id: str) -> uuid.UUID:
    unit_id = uuid.uuid4()
    cursor.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date) VALUES (:id, :b, :t, 'world', :d)",
        {"id": unit_id.bytes, "b": bank_id, "t": f"fact {unit_id}", "d": _ora_value(_NOW)},
    )
    return unit_id


def _ora_count(cursor: Any, sql: str, **params: Any) -> int:
    cursor.execute(sql, {k: _ora_value(v) for k, v in params.items()})
    return int(cursor.fetchone()[0])


def _ora_code(exc: Exception) -> int | None:
    return getattr(exc.args[0], "code", None) if exc.args else None


@pytest.fixture
def ora_cursor(oracle_db_url: str) -> Iterator[Any]:
    """A cursor on the session's migrated Oracle schema; everything it writes is rolled back."""
    import oracledb

    parts = urlsplit(oracle_db_url)
    conn = oracledb.connect(
        user=parts.username,
        password=parts.password,
        dsn=f"{parts.hostname}:{parts.port or 1521}/{parts.path.lstrip('/')}",
    )
    try:
        cursor = conn.cursor()
        yield cursor
        cursor.close()
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.oracle
def test_oracle_upgrade_creates_the_tables_indexes_and_foreign_keys(ora_cursor: Any) -> None:
    ora_cursor.execute("SELECT LOWER(table_name) FROM user_tables")
    assert set(_TABLES) <= {r[0] for r in ora_cursor.fetchall()}

    ora_cursor.execute("SELECT LOWER(index_name) FROM user_indexes")
    indexes = {r[0] for r in ora_cursor.fetchall()}
    # No partial indexes on Oracle: the unprocessed-events one is PG-only, and the
    # function-based uq_mlr_edge cannot serve a source probe, so source has its own.
    expected = (_PG_INDEXES - {"idx_am_ev_unprocessed"}) | {"idx_am_rel_source"}
    assert expected <= indexes, f"missing indexes: {sorted(expected - indexes)}"

    ora_cursor.execute(
        "SELECT LOWER(c.table_name), LOWER(c.constraint_name), c.delete_rule, LOWER(p.table_name) "
        "FROM user_constraints c JOIN user_constraints p ON p.constraint_name = c.r_constraint_name "
        "WHERE c.constraint_type = 'R' AND LOWER(c.table_name) IN (" + ", ".join(f"'{t}'" for t in _TABLES) + ")"
    )
    fks = {(r[0], r[1]): (r[2], r[3]) for r in ora_cursor.fetchall()}
    assert fks == {key: (action, "memory_units") for key, (action, _deferred) in _FKS.items()}

    ora_cursor.execute(
        "SELECT LOWER(column_name) FROM user_tab_columns WHERE table_name IN ('MEMORY_UNITS', 'INVALIDATED_MEMORY_UNITS') "
        "AND column_name = 'ADAPTIVE_STATE'"
    )
    assert [r[0] for r in ora_cursor.fetchall()] == ["adaptive_state"]


@pytest.mark.oracle
def test_oracle_rows_cascade_references_null_and_tombstones_coexist(ora_cursor: Any) -> None:
    bank_id = f"am-{uuid.uuid4().hex[:8]}"
    subject, successor, second = (_ora_unit(ora_cursor, bank_id) for _ in range(3))
    for table, row in _valid_rows(subject, successor, bank_id).items():
        _ora_insert(ora_cursor, table, row)
    # A second predecessor of the same target, so deleting both sources leaves
    # two (NULL, subject, 'supersedes') tombstones.
    relation = _valid_rows(subject, second, bank_id)["memory_lifecycle_relations"]
    _ora_insert(ora_cursor, "memory_lifecycle_relations", relation)

    ora_cursor.execute("DELETE FROM memory_units WHERE id IN (:a, :b)", {"a": successor.bytes, "b": second.bytes})
    ora_cursor.execute(
        "SELECT superseded_by_id, inherited_from_id FROM memory_claim_state WHERE memory_id = :m", {"m": subject.bytes}
    )
    assert ora_cursor.fetchone() == (None, None)
    assert _ora_count(ora_cursor, "SELECT COUNT(*) FROM memory_claim_evidence WHERE memory_id = :m", m=subject) == 0
    assert (
        _ora_count(
            ora_cursor,
            "SELECT COUNT(*) FROM memory_lifecycle_relations WHERE source_memory_id IS NULL AND target_memory_id = :t",
            t=subject,
        )
        == 2
    )

    ora_cursor.execute("DELETE FROM memory_units WHERE id = :m", {"m": subject.bytes})
    for table in ("memory_retention_state", "memory_claim_state", "memory_reinforcement_events", "memory_claim_shadow"):
        assert _ora_count(ora_cursor, f"SELECT COUNT(*) FROM {table} WHERE bank_id = :b", b=bank_id) == 0, table
    assert (
        _ora_count(
            ora_cursor,
            "SELECT COUNT(*) FROM memory_lifecycle_relations "
            "WHERE bank_id = :b AND source_memory_id IS NULL AND target_memory_id IS NULL",
            b=bank_id,
        )
        == 2
    )


@pytest.mark.oracle
def test_oracle_constraints_reject_bad_rows(ora_cursor: Any) -> None:
    import oracledb

    bank_id = f"am-{uuid.uuid4().hex[:8]}"
    subject, other = _ora_unit(ora_cursor, bank_id), _ora_unit(ora_cursor, bank_id)
    rows = _valid_rows(subject, other, bank_id)
    for table, column, value in _BAD_VALUES:
        with pytest.raises(oracledb.DatabaseError) as exc_info:
            _ora_insert(ora_cursor, table, {**rows[table], column: value})
        assert _ora_code(exc_info.value) == _ORA_CHECK, (table, column, value)

    _ora_insert(ora_cursor, "memory_lifecycle_relations", rows["memory_lifecycle_relations"])
    with pytest.raises(oracledb.DatabaseError) as exc_info:
        _ora_insert(
            ora_cursor, "memory_lifecycle_relations", {**rows["memory_lifecycle_relations"], "id": uuid.uuid4()}
        )
    assert _ora_code(exc_info.value) == _ORA_UNIQUE

    _ora_insert(ora_cursor, "memory_reinforcement_events", rows["memory_reinforcement_events"])
    with pytest.raises(oracledb.DatabaseError) as exc_info:
        _ora_insert(
            ora_cursor, "memory_reinforcement_events", {**rows["memory_reinforcement_events"], "id": uuid.uuid4()}
        )
    assert _ora_code(exc_info.value) == _ORA_UNIQUE

    with pytest.raises(oracledb.DatabaseError) as exc_info:
        _ora_insert(ora_cursor, "memory_retention_state", {**rows["memory_retention_state"], "memory_id": uuid.uuid4()})
    assert _ora_code(exc_info.value) == _ORA_FK_PARENT_MISSING


@pytest.mark.oracle
def test_oracle_downgrade_and_reupgrade(oracle_db_url: str, ora_cursor: Any) -> None:
    from hindsight_api.migrations import run_migrations

    cfg = _alembic_cfg(oracle_db_url)

    def tables() -> set[str]:
        ora_cursor.execute("SELECT LOWER(table_name) FROM user_tables")
        return {r[0] for r in ora_cursor.fetchall()}

    def archive_has_column() -> bool:
        ora_cursor.execute(
            "SELECT COUNT(*) FROM user_tab_columns "
            "WHERE table_name = 'INVALIDATED_MEMORY_UNITS' AND column_name = 'ADAPTIVE_STATE'"
        )
        return ora_cursor.fetchone()[0] == 1

    try:
        command.downgrade(cfg, _PRE_REVISION)
        assert not set(_TABLES) & tables()
        assert not archive_has_column()

        command.upgrade(cfg, _REVISION)
        command.stamp(cfg, _PRE_REVISION)
        command.upgrade(cfg, _REVISION)
        assert set(_TABLES) <= tables()
        assert archive_has_column()
    finally:
        run_migrations(oracle_db_url)
