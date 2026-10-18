"""Add the adaptive-memory side tables (retention, claim state, lifecycle audit)

Adaptive memory keeps three independent kinds of per-memory state, none of which
lives on ``memory_units`` itself (that table is unchanged):

- ``memory_retention_state`` — how readily a memory comes back without a strong
  cue: stability (days), last reinforcement, importance, exemption, and the
  archive bookkeeping. A raw fact with no row reads lazy defaults, so rows are
  only materialized when something has been learned about the memory.
- ``memory_claim_state`` — whether a memory is still believed: truth status,
  validity interval, and how it was retired. ``memory_claim_evidence`` lists the
  raw facts (stable ids, unlike observations, which consolidation rebuilds under
  new ids) that justify a non-active claim, so the claim can survive the rebuild
  of the observation that decided it and revert once all its evidence is gone.
- ``memory_lifecycle_relations`` — the audit trail of supersede / refute / split /
  merge / conflict decisions. Endpoints are ``ON DELETE SET NULL`` and each row
  carries JSON snapshots of both sides, so provenance survives endpoint churn as
  a tombstone instead of dangling or vanishing.
- ``memory_reinforcement_events`` — the idempotency ledger for reinforcement:
  ``UNIQUE (memory_id, source_type, source_key)`` is what stops the same evidence
  from reinforcing a memory twice.
- ``memory_claim_shadow`` — observe-mode "would retire" records; diagnostic only,
  never read for filtering.
- ``invalidated_memory_units.adaptive_state`` — the archive-side snapshot of the
  above, so curation (which MOVES a row and lets the FK cascade take its side
  rows) can restore a memory's learned state on revert. Archive only:
  ``memory_units`` gets no column, so the lockstep INSERT…SELECT between the two
  tables is unaffected.

Rows that belong to one memory die with it (``ON DELETE CASCADE``). On
PostgreSQL those FKs are ``DEFERRABLE INITIALLY DEFERRED`` for the reason
``9f8e7d6c5b4a`` gives for ``memory_links``: an immediate FK check takes
``FOR KEY SHARE`` on the parent at INSERT time and deadlocks against a cascading
delete; deferring moves the check to COMMIT. The ``SET NULL`` references (successor,
inheritance source, relation endpoints) are immediate, so a write naming a
memory that is already gone fails at its statement rather than at COMMIT.

No backfill: every table starts empty and nothing is written while the bank's
adaptive-memory mode is ``off``. The tables are new, so plain
``CREATE INDEX IF NOT EXISTS`` is enough (no CONCURRENTLY).

Oracle differences: no deferrable FKs (as in ``9f8e7d6c5b4a``), no partial
indexes (plain ones instead; the unprocessed-events index is PostgreSQL-only),
and the relations uniqueness is a function-based unique index over rows whose
two endpoints are both set. A plain composite UNIQUE would treat
``(NULL, X, 'supersedes')`` tombstones as duplicates on Oracle, so the second
endpoint delete that produced one would fail the parent DELETE with ORA-00001;
PostgreSQL's NULLS DISTINCT default already ignores such rows, and the index
reproduces that.

Revision ID: a7d3e91c5f42
Revises: d4f7a2c9e1b5
Create Date: 2026-10-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a7d3e91c5f42"
down_revision: str | Sequence[str] | None = "d4f7a2c9e1b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Creation order; downgrades drop in reverse. Every table references memory_units
# only, never each other, so any order would be FK-safe.
_TABLES = (
    "memory_retention_state",
    "memory_claim_state",
    "memory_claim_evidence",
    "memory_lifecycle_relations",
    "memory_reinforcement_events",
    "memory_claim_shadow",
)

_TRUTH_STATUSES = "'active', 'uncertain', 'superseded', 'refuted'"
_RETIRED_VIA = "'supersedes', 'refutes', 'merges', 'splits', 'explicit', 'inherited'"
_RELATION_TYPES = "'supersedes', 'refutes', 'refines', 'splits', 'merges', 'conflicts'"
_RELATION_ORIGINS = "'consolidation', 'explicit', 'inherited'"
_SOURCE_TYPES = (
    "'explicit_confirmation', 'independent_retain_confirmation', 'successful_reflect_use', "
    "'observation_new_evidence', 'final_external_recall'"
)
_OUTCOMES = "'applied', 'cooldown', 'initial'"
_WOULD_STATUSES = "'superseded', 'refuted'"

# The relations uniqueness on Oracle: each key part is NULL unless both endpoints
# are set, and Oracle does not index an all-NULL key (see the module docstring).
_BOTH_ENDPOINTS = "source_memory_id IS NOT NULL AND target_memory_id IS NOT NULL"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    mu = f"{schema}memory_units"
    cascade = f"REFERENCES {mu} (id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED"
    set_null = f"REFERENCES {mu} (id) ON DELETE SET NULL"

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_retention_state (
            memory_id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            stability_days DOUBLE PRECISION NOT NULL,
            last_reinforced_at TIMESTAMPTZ NOT NULL,
            last_recalled_at TIMESTAMPTZ,
            reinforcement_count INTEGER NOT NULL DEFAULT 0,
            access_count INTEGER NOT NULL DEFAULT 0,
            forgetting_exempt BOOLEAN NOT NULL DEFAULT FALSE,
            exemption_reason TEXT,
            retention_state TEXT NOT NULL DEFAULT 'active',
            below_threshold_since TIMESTAMPTZ,
            archived_at TIMESTAMPTZ,
            model_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_retention_state PRIMARY KEY (memory_id),
            CONSTRAINT fk_mrs_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mrs_importance CHECK (importance >= 0 AND importance <= 1),
            CONSTRAINT chk_mrs_stability CHECK (stability_days > 0),
            CONSTRAINT chk_mrs_state CHECK (retention_state IN ('active', 'archived'))
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_rs_bank_reinforced "
        f"ON {schema}memory_retention_state (bank_id, last_reinforced_at)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_rs_bank_state "
        f"ON {schema}memory_retention_state (bank_id, retention_state, below_threshold_since)"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_claim_state (
            memory_id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            truth_status TEXT NOT NULL,
            confidence REAL,
            valid_from TIMESTAMPTZ,
            valid_to TIMESTAMPTZ,
            retired_via TEXT,
            superseded_by_id UUID,
            inherited_from_id UUID,
            scope_fingerprint TEXT,
            reason TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_claim_state PRIMARY KEY (memory_id),
            CONSTRAINT fk_mcs_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT fk_mcs_superseded_by FOREIGN KEY (superseded_by_id) {set_null},
            CONSTRAINT fk_mcs_inherited_from FOREIGN KEY (inherited_from_id) {set_null},
            CONSTRAINT chk_mcs_truth_status CHECK (truth_status IN ({_TRUTH_STATUSES})),
            CONSTRAINT chk_mcs_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
            CONSTRAINT chk_mcs_retired_via CHECK (retired_via IS NULL OR retired_via IN ({_RETIRED_VIA}))
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_cs_bank_status ON {schema}memory_claim_state (bank_id, truth_status)"
    )
    # Partial: almost every claim row has neither reference, and these indexes
    # exist for the FK's SET NULL lookup and the successor re-link, which both
    # probe by a non-NULL id.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_cs_superseded_by ON {schema}memory_claim_state (superseded_by_id) "
        "WHERE superseded_by_id IS NOT NULL"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_cs_inherited_from ON {schema}memory_claim_state (inherited_from_id) "
        "WHERE inherited_from_id IS NOT NULL"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_claim_evidence (
            memory_id UUID NOT NULL,
            evidence_memory_id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_claim_evidence PRIMARY KEY (memory_id, evidence_memory_id),
            CONSTRAINT fk_mce_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT fk_mce_evidence FOREIGN KEY (evidence_memory_id) {cascade}
        )
        """
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_am_ce_evidence ON {schema}memory_claim_evidence (evidence_memory_id)")

    # NULLS DISTINCT (the PG default) is load-bearing on uq_mlr_edge: once an
    # endpoint is deleted the row becomes a (NULL, X, type) tombstone, and several
    # tombstones of the same shape must coexist. The unique index also serves
    # lookups by source_memory_id (its leading column), so that one needs no
    # index of its own here.
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_lifecycle_relations (
            id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            source_memory_id UUID,
            target_memory_id UUID,
            source_snapshot JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            target_snapshot JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            origin TEXT NOT NULL,
            reason TEXT,
            confidence REAL,
            evidence JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_lifecycle_relations PRIMARY KEY (id),
            CONSTRAINT fk_mlr_source FOREIGN KEY (source_memory_id) {set_null},
            CONSTRAINT fk_mlr_target FOREIGN KEY (target_memory_id) {set_null},
            CONSTRAINT chk_mlr_relation_type CHECK (relation_type IN ({_RELATION_TYPES})),
            CONSTRAINT chk_mlr_origin CHECK (origin IN ({_RELATION_ORIGINS})),
            CONSTRAINT uq_mlr_edge UNIQUE (source_memory_id, target_memory_id, relation_type)
        )
        """
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_am_rel_target ON {schema}memory_lifecycle_relations (target_memory_id)")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_rel_bank_created ON {schema}memory_lifecycle_relations (bank_id, created_at)"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_reinforcement_events (
            id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            memory_id UUID NOT NULL,
            source_type TEXT NOT NULL,
            source_key VARCHAR(512) NOT NULL,
            event_time TIMESTAMPTZ NOT NULL,
            weight REAL NOT NULL,
            independence_weight REAL NOT NULL DEFAULT 1.0,
            applied_gain REAL NOT NULL DEFAULT 0,
            outcome TEXT NOT NULL,
            processed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_reinforcement_events PRIMARY KEY (id),
            CONSTRAINT fk_mre_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mre_source_type CHECK (source_type IN ({_SOURCE_TYPES})),
            CONSTRAINT chk_mre_outcome CHECK (outcome IN ({_OUTCOMES})),
            CONSTRAINT uq_mre_key UNIQUE (memory_id, source_type, source_key)
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_ev_bank_created ON {schema}memory_reinforcement_events (bank_id, created_at)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_am_ev_unprocessed ON {schema}memory_reinforcement_events (bank_id) "
        "WHERE processed_at IS NULL"
    )

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_claim_shadow (
            memory_id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            would_status TEXT NOT NULL,
            deciding_snapshot JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT pk_memory_claim_shadow PRIMARY KEY (memory_id),
            CONSTRAINT fk_mcsh_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mcsh_would_status CHECK (would_status IN ({_WOULD_STATUSES}))
        )
        """
    )

    # Nullable with no default: metadata-only on any table size, and NULL means
    # "archived with no adaptive state", which is every row archived before now.
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units ADD COLUMN IF NOT EXISTS adaptive_state JSONB")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP COLUMN IF EXISTS adaptive_state")
    # Each table's indexes and constraints go with it.
    for table in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {schema}{table}")


def _oracle_execute_ignoring(sql: str, *codes: int) -> None:
    """Run one DDL statement, swallowing the given ORA- codes (as positive numbers).

    DDL autocommits on Oracle, so a partial earlier run leaves some objects
    behind; tolerating "already exists" / "does not exist" lets a rerun finish
    (mirrors ``c4f7a91b2d38`` and ``c7d1e9a4b3f2``).
    """
    tolerated = " OR ".join(f"SQLCODE = -{code}" for code in codes)
    block = (
        f"BEGIN EXECUTE IMMEDIATE :stmt; EXCEPTION WHEN OTHERS THEN IF {tolerated} THEN NULL; ELSE RAISE; END IF; END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


# ORA-00955: name is already used by an existing object.
_EXISTS = 955
# ORA-01430: column being added already exists in table.
_COLUMN_EXISTS = 1430
# ORA-00942: table or view does not exist.
_NO_TABLE = 942
# ORA-00904: invalid identifier (the column is already gone).
_NO_COLUMN = 904


def _oracle_upgrade() -> None:
    # Types follow the baseline (o1a2b3c4d5e6): RAW(16) ids, VARCHAR2(256) bank_id,
    # NUMBER(1) booleans, BINARY_DOUBLE reals, CLOB + IS JSON for JSON, CLOB for
    # free text. Enum-like and compared columns are VARCHAR2, which Oracle can
    # index and compare (a CLOB can be neither).
    cascade = "REFERENCES memory_units (id) ON DELETE CASCADE"
    set_null = "REFERENCES memory_units (id) ON DELETE SET NULL"

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_retention_state (
            memory_id             RAW(16)        NOT NULL,
            bank_id               VARCHAR2(256)  NOT NULL,
            importance            BINARY_DOUBLE  DEFAULT 0.5 NOT NULL,
            stability_days        BINARY_DOUBLE  NOT NULL,
            last_reinforced_at    TIMESTAMP WITH TIME ZONE NOT NULL,
            last_recalled_at      TIMESTAMP WITH TIME ZONE,
            reinforcement_count   NUMBER(10)     DEFAULT 0 NOT NULL,
            access_count          NUMBER(10)     DEFAULT 0 NOT NULL,
            forgetting_exempt     NUMBER(1)      DEFAULT 0 NOT NULL,
            exemption_reason      CLOB,
            retention_state       VARCHAR2(16)   DEFAULT 'active' NOT NULL,
            below_threshold_since TIMESTAMP WITH TIME ZONE,
            archived_at           TIMESTAMP WITH TIME ZONE,
            model_version         NUMBER(10)     DEFAULT 1 NOT NULL,
            created_at            TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at            TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_retention_state PRIMARY KEY (memory_id),
            CONSTRAINT fk_mrs_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mrs_importance CHECK (importance >= 0 AND importance <= 1),
            CONSTRAINT chk_mrs_stability CHECK (stability_days > 0),
            CONSTRAINT chk_mrs_state CHECK (retention_state IN ('active', 'archived'))
        )
        """,
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_rs_bank_reinforced ON memory_retention_state (bank_id, last_reinforced_at)",
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_rs_bank_state ON memory_retention_state (bank_id, retention_state, below_threshold_since)",
        _EXISTS,
    )

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_claim_state (
            memory_id         RAW(16)        NOT NULL,
            bank_id           VARCHAR2(256)  NOT NULL,
            truth_status      VARCHAR2(16)   NOT NULL,
            confidence        BINARY_DOUBLE,
            valid_from        TIMESTAMP WITH TIME ZONE,
            valid_to          TIMESTAMP WITH TIME ZONE,
            retired_via       VARCHAR2(16),
            superseded_by_id  RAW(16),
            inherited_from_id RAW(16),
            scope_fingerprint VARCHAR2(1024),
            reason            CLOB,
            created_at        TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at        TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_claim_state PRIMARY KEY (memory_id),
            CONSTRAINT fk_mcs_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT fk_mcs_superseded_by FOREIGN KEY (superseded_by_id) {set_null},
            CONSTRAINT fk_mcs_inherited_from FOREIGN KEY (inherited_from_id) {set_null},
            CONSTRAINT chk_mcs_truth_status CHECK (truth_status IN ({_TRUTH_STATUSES})),
            CONSTRAINT chk_mcs_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
            CONSTRAINT chk_mcs_retired_via CHECK (retired_via IS NULL OR retired_via IN ({_RETIRED_VIA}))
        )
        """,
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_cs_bank_status ON memory_claim_state (bank_id, truth_status)",
        _EXISTS,
    )
    # Plain rather than partial (Oracle has none). An unindexed FK column would
    # also make every memory_units delete lock this whole table.
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_cs_superseded_by ON memory_claim_state (superseded_by_id)",
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_cs_inherited_from ON memory_claim_state (inherited_from_id)",
        _EXISTS,
    )

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_claim_evidence (
            memory_id          RAW(16)        NOT NULL,
            evidence_memory_id RAW(16)        NOT NULL,
            bank_id            VARCHAR2(256)  NOT NULL,
            created_at         TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_claim_evidence PRIMARY KEY (memory_id, evidence_memory_id),
            CONSTRAINT fk_mce_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT fk_mce_evidence FOREIGN KEY (evidence_memory_id) {cascade}
        )
        """,
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_ce_evidence ON memory_claim_evidence (evidence_memory_id)",
        _EXISTS,
    )

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_lifecycle_relations (
            id               RAW(16)        NOT NULL,
            bank_id          VARCHAR2(256)  NOT NULL,
            relation_type    VARCHAR2(16)   NOT NULL,
            source_memory_id RAW(16),
            target_memory_id RAW(16),
            source_snapshot  CLOB           DEFAULT '{{}}' NOT NULL
                                            CONSTRAINT mlr_source_snapshot_json CHECK (source_snapshot IS JSON),
            target_snapshot  CLOB           DEFAULT '{{}}' NOT NULL
                                            CONSTRAINT mlr_target_snapshot_json CHECK (target_snapshot IS JSON),
            origin           VARCHAR2(16)   NOT NULL,
            reason           CLOB,
            confidence       BINARY_DOUBLE,
            evidence         CLOB           DEFAULT '{{}}' NOT NULL
                                            CONSTRAINT mlr_evidence_json CHECK (evidence IS JSON),
            created_at       TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_lifecycle_relations PRIMARY KEY (id),
            CONSTRAINT fk_mlr_source FOREIGN KEY (source_memory_id) {set_null},
            CONSTRAINT fk_mlr_target FOREIGN KEY (target_memory_id) {set_null},
            CONSTRAINT chk_mlr_relation_type CHECK (relation_type IN ({_RELATION_TYPES})),
            CONSTRAINT chk_mlr_origin CHECK (origin IN ({_RELATION_ORIGINS}))
        )
        """,
        _EXISTS,
    )
    _oracle_execute_ignoring(
        f"""
        CREATE UNIQUE INDEX uq_mlr_edge ON memory_lifecycle_relations (
            CASE WHEN {_BOTH_ENDPOINTS} THEN source_memory_id END,
            CASE WHEN {_BOTH_ENDPOINTS} THEN target_memory_id END,
            CASE WHEN {_BOTH_ENDPOINTS} THEN relation_type END
        )
        """,
        _EXISTS,
    )
    # The function-based unique index cannot serve a plain source_memory_id probe
    # (the FK's SET NULL lookup), so source gets its own index here, unlike PG.
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_rel_source ON memory_lifecycle_relations (source_memory_id)",
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_rel_target ON memory_lifecycle_relations (target_memory_id)",
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_rel_bank_created ON memory_lifecycle_relations (bank_id, created_at)",
        _EXISTS,
    )

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_reinforcement_events (
            id                  RAW(16)        NOT NULL,
            bank_id             VARCHAR2(256)  NOT NULL,
            memory_id           RAW(16)        NOT NULL,
            source_type         VARCHAR2(64)   NOT NULL,
            source_key          VARCHAR2(512)  NOT NULL,
            event_time          TIMESTAMP WITH TIME ZONE NOT NULL,
            weight              BINARY_DOUBLE  NOT NULL,
            independence_weight BINARY_DOUBLE  DEFAULT 1.0 NOT NULL,
            applied_gain        BINARY_DOUBLE  DEFAULT 0 NOT NULL,
            outcome             VARCHAR2(16)   NOT NULL,
            processed_at        TIMESTAMP WITH TIME ZONE,
            created_at          TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_reinforcement_events PRIMARY KEY (id),
            CONSTRAINT fk_mre_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mre_source_type CHECK (source_type IN ({_SOURCE_TYPES})),
            CONSTRAINT chk_mre_outcome CHECK (outcome IN ({_OUTCOMES})),
            CONSTRAINT uq_mre_key UNIQUE (memory_id, source_type, source_key)
        )
        """,
        _EXISTS,
    )
    _oracle_execute_ignoring(
        "CREATE INDEX idx_am_ev_bank_created ON memory_reinforcement_events (bank_id, created_at)",
        _EXISTS,
    )

    _oracle_execute_ignoring(
        f"""
        CREATE TABLE memory_claim_shadow (
            memory_id         RAW(16)        NOT NULL,
            bank_id           VARCHAR2(256)  NOT NULL,
            would_status      VARCHAR2(16)   NOT NULL,
            deciding_snapshot CLOB           DEFAULT '{{}}' NOT NULL
                                             CONSTRAINT mcsh_snapshot_json CHECK (deciding_snapshot IS JSON),
            evidence_ids      CLOB           DEFAULT '[]' NOT NULL
                                             CONSTRAINT mcsh_evidence_ids_json CHECK (evidence_ids IS JSON),
            created_at        TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT pk_memory_claim_shadow PRIMARY KEY (memory_id),
            CONSTRAINT fk_mcsh_memory FOREIGN KEY (memory_id) {cascade},
            CONSTRAINT chk_mcsh_would_status CHECK (would_status IN ({_WOULD_STATUSES}))
        )
        """,
        _EXISTS,
    )

    _oracle_execute_ignoring(
        "ALTER TABLE invalidated_memory_units ADD (adaptive_state CLOB "
        "CONSTRAINT imu_adaptive_state_json CHECK (adaptive_state IS JSON OR adaptive_state IS NULL))",
        _COLUMN_EXISTS,
    )


def _oracle_downgrade() -> None:
    _oracle_execute_ignoring("ALTER TABLE invalidated_memory_units DROP COLUMN adaptive_state", _NO_COLUMN)
    for table in reversed(_TABLES):
        _oracle_execute_ignoring(f"DROP TABLE {table} CASCADE CONSTRAINTS", _NO_TABLE)


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
