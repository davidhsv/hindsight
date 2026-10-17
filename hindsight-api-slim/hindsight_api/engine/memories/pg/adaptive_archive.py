"""Adaptive state across a curation invalidate → restore round trip (DESIGN.md §7, R26).

Invalidation MOVES a memory: its row is copied to ``invalidated_memory_units`` and deleted from
``memory_units``, and the FK cascade takes everything adaptive memory keeps about it — the
retention row, the reinforcement ledger, the claim row and the claim-evidence rows on both sides —
while the ``SET NULL`` references (other claims' ``superseded_by_id`` / ``inherited_from_id``, the
lifecycle relations' endpoints) are cleared. Curation invalidation is meant to be temporary removal
that preserves learned state, so :func:`park_adaptive_state` snapshots all of it onto the archive
row's ``adaptive_state`` before the delete, and :func:`unpark_adaptive_state` puts it back on
restore. A memory with no adaptive state gets no snapshot: ``adaptive_state`` stays NULL, which is
what every archive row looked like before this feature (off-mode exports are unchanged).

Decisions, each a deliberate trade:

- **Invalidation never reverts a claim.** A predecessor whose only evidence is the invalidated fact
  keeps its status (with no evidence rows) while the fact is archived; restore re-attaches the
  evidence. Reverting on invalidation would make a temporary curation permanent for the
  predecessor, because the revert is a lifecycle action restore cannot tell from a user's.
- **Inbound edges are re-attached only to claims nothing touched since.** Evidence rows and
  ``superseded_by_id`` / ``inherited_from_id`` pointers of OTHER memories are restored only when
  that memory's claim row is unchanged since the invalidation (``updated_at <= invalidated_at``);
  a newer lifecycle decision about it wins over the snapshot.
- **Relation tombstones are re-pointed, not duplicated.** The relation rows themselves survive the
  delete (SET NULL); restore writes the memory back into the endpoint it held, by relation id, and
  only re-inserts a row that has disappeared altogether (when its other endpoint is live).
- **Recall and reflect ledger keys are kept only inside the cooldown horizon.** Their keys name a
  cooldown window (``recall:<bucket>``), so an older one can never be produced again and keeping it
  would only grow the snapshot. Explicit, retain and observation keys are kept whatever their age:
  re-sending one must stay a no-op after a restore.
- **Peers must be live in the same bank.** Every id a snapshot names is re-checked against the bank
  under ``FOR KEY SHARE`` (so it cannot be deleted before COMMIT); ids that are gone, archived or
  belong to another bank (an archive imported from elsewhere carries source-bank ids) are dropped
  and counted, like ``rematerialize_causal_links`` does for causal edges.

PostgreSQL only for now (both functions are no-ops on Oracle, as before this feature): Oracle
parity for invalidate/restore is planned with the rest of the Oracle archive work (P4).

Known gap: if a claim and its evidence fact are BOTH invalidated, evidence first, and restored in
the same order, the edge is lost — the claim's snapshot was taken after the cascade had already
removed it, and the evidence fact's restore found the claim still archived. The causal-link
snapshot avoids this by merging descriptors parked on archived peers; claims are first written in
P3, which owns that extension.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...adaptive import clock
from ...adaptive.models import FINAL_EXTERNAL_RECALL, RetentionState, TruthStatus
from ...adaptive.settings import AdaptiveSettings
from ...adaptive.views import (
    RECENT_EVENTS_LIMIT,
    ClaimView,
    MemoryLifecycle,
    ReinforcementEventView,
    RelationView,
)
from ...db.ops import ReinforcementEventRow

logger = logging.getLogger(__name__)

#: Sources whose ledger keys name a cooldown window (see the module docstring).
_WINDOW_KEYED_SOURCES = [FINAL_EXTERNAL_RECALL, "successful_reflect_use"]

#: Claim statuses that mean "retired": such a raw fact must never read as pending consolidation.
_RETIRED = frozenset({TruthStatus.SUPERSEDED.value, TruthStatus.REFUTED.value})


class _Retention(BaseModel):
    importance: float
    stability_days: float
    last_reinforced_at: datetime
    last_recalled_at: datetime | None = None
    reinforcement_count: int = 0
    access_count: int = 0
    forgetting_exempt: bool = False
    exemption_reason: str | None = None
    retention_state: str = "active"
    below_threshold_since: datetime | None = None
    archived_at: datetime | None = None
    model_version: int = 1
    created_at: datetime
    updated_at: datetime


class _Claim(BaseModel):
    truth_status: str
    confidence: float | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retired_via: str | None = None
    superseded_by_id: str | None = None
    inherited_from_id: str | None = None
    scope_fingerprint: str | None = None
    reason: str | None = None
    created_at: datetime
    updated_at: datetime


class _Evidence(BaseModel):
    memory_id: str
    evidence_memory_id: str
    created_at: datetime


class _Event(BaseModel):
    """One ledger row. Its row id is deliberately not kept: restore mints a fresh one, because an
    exported archive carries this snapshot verbatim, and a bank cloned within the same database
    would otherwise restore the same ids into both copies (``memory_reinforcement_events.id`` is a
    global primary key, and nothing references it; dedup goes by ``(memory_id, source_type,
    source_key)``). Snapshots written with an ``id`` still parse: extra keys are ignored."""

    source_type: str
    source_key: str
    event_time: datetime
    weight: float
    independence_weight: float
    applied_gain: float
    outcome: str
    processed_at: datetime | None = None
    created_at: datetime


class _Relation(BaseModel):
    id: str
    relation_type: str
    source_memory_id: str | None = None
    target_memory_id: str | None = None
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    target_snapshot: dict[str, Any] = Field(default_factory=dict)
    origin: str
    reason: str | None = None
    confidence: float | None = None
    evidence: Any = Field(default_factory=dict)
    created_at: datetime


class AdaptiveSnapshot(BaseModel):
    """``invalidated_memory_units.adaptive_state``: a memory's adaptive rows as plain data."""

    version: Literal[1] = 1
    memory_id: str
    """The memory's id when it was invalidated. Restore writes under the archive row's id, which
    differs for an archive imported from another bank; this one only identifies the memory's own
    side of the edges below."""
    invalidated_at: datetime
    """The adaptive write clock at invalidation; inbound edges are re-attached only to claims whose
    ``updated_at`` is not later."""
    retention: _Retention | None = None
    claim: _Claim | None = None
    events: list[_Event] = Field(default_factory=list)
    evidence_out: list[_Evidence] = Field(default_factory=list)
    """Evidence rows of this memory's own claim (``memory_id`` = this memory)."""
    evidence_in: list[_Evidence] = Field(default_factory=list)
    """Evidence rows of OTHER claims that cite this memory (``evidence_memory_id`` = this memory)."""
    superseded_by_refs: list[str] = Field(default_factory=list)
    """Memories whose claim names this one as its successor."""
    inherited_from_refs: list[str] = Field(default_factory=list)
    """Memories whose claim was inherited from this one."""
    relations: list[_Relation] = Field(default_factory=list)


@dataclass(frozen=True)
class RematerializedState:
    """What :func:`unpark_adaptive_state` put back."""

    restored: bool
    """A snapshot was present (and rematerialized)."""
    dropped: int
    """Snapshot edges not restored: their peer is gone, or a newer lifecycle action owns it."""
    retired: bool
    """The restored claim is superseded or refuted, so ``consolidated_at`` was re-stamped."""


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def _postgres(conn) -> bool:
    return getattr(conn, "backend_type", "postgresql") == "postgresql"


def _parse(conn, raw: Any) -> AdaptiveSnapshot | None:
    """The stored snapshot, or None when the row has none. A malformed one raises: only this module
    writes the column, so it is a bug or a hand edit, and silently dropping it would lose the state."""
    data = conn.parse_json(raw)
    if data is None:
        return None
    return AdaptiveSnapshot.model_validate(data)


# --------------------------------------------------------------------------- invalidate


async def snapshot_adaptive_state(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_id: uuid.UUID, now: datetime, cooldown_hours: float
) -> AdaptiveSnapshot | None:
    """Every adaptive row about ``memory_id``, or None when there is none (the off-mode case).

    The first statement is one indexed existence probe per table and edge, so a memory without
    adaptive state costs exactly one round trip.
    """
    retention_t = fq_table("memory_retention_state")
    claim_t = fq_table("memory_claim_state")
    evidence_t = fq_table("memory_claim_evidence")
    events_t = fq_table("memory_reinforcement_events")
    relations_t = fq_table("memory_lifecycle_relations")
    has_state = await conn.fetchval(
        f"""
        SELECT EXISTS (SELECT 1 FROM {retention_t} WHERE memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {claim_t} WHERE memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {events_t} WHERE memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {evidence_t} WHERE memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {evidence_t} WHERE evidence_memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {claim_t} WHERE superseded_by_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {claim_t} WHERE inherited_from_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {relations_t} WHERE source_memory_id = $1 AND bank_id = $2)
            OR EXISTS (SELECT 1 FROM {relations_t} WHERE target_memory_id = $1 AND bank_id = $2)
        """,
        memory_id,
        bank_id,
    )
    if not has_state:
        return None

    retention = await conn.fetchrow(
        f"SELECT importance, stability_days, last_reinforced_at, last_recalled_at, reinforcement_count, "
        f"access_count, forgetting_exempt, exemption_reason, retention_state, below_threshold_since, archived_at, "
        f"model_version, created_at, updated_at FROM {retention_t} WHERE memory_id = $1 AND bank_id = $2",
        memory_id,
        bank_id,
    )
    claim = await conn.fetchrow(
        f"SELECT truth_status, confidence, valid_from, valid_to, retired_via, superseded_by_id, inherited_from_id, "
        f"scope_fingerprint, reason, created_at, updated_at FROM {claim_t} WHERE memory_id = $1 AND bank_id = $2",
        memory_id,
        bank_id,
    )
    events = await conn.fetch(
        f"""
        SELECT source_type, source_key, event_time, weight, independence_weight, applied_gain, outcome,
               processed_at, created_at
        FROM {events_t}
        WHERE memory_id = $1 AND bank_id = $2 AND (NOT (source_type = ANY($3::text[])) OR event_time >= $4)
        ORDER BY event_time, id
        """,
        memory_id,
        bank_id,
        _WINDOW_KEYED_SOURCES,
        now - timedelta(hours=max(0.0, cooldown_hours)),
    )
    evidence_out = await conn.fetch(
        f"SELECT memory_id, evidence_memory_id, created_at FROM {evidence_t} "
        f"WHERE memory_id = $1 AND bank_id = $2 ORDER BY evidence_memory_id",
        memory_id,
        bank_id,
    )
    evidence_in = await conn.fetch(
        f"SELECT memory_id, evidence_memory_id, created_at FROM {evidence_t} "
        f"WHERE evidence_memory_id = $1 AND bank_id = $2 ORDER BY memory_id",
        memory_id,
        bank_id,
    )
    superseded_refs = await conn.fetch(
        f"SELECT memory_id FROM {claim_t} WHERE superseded_by_id = $1 AND bank_id = $2 ORDER BY memory_id",
        memory_id,
        bank_id,
    )
    inherited_refs = await conn.fetch(
        f"SELECT memory_id FROM {claim_t} WHERE inherited_from_id = $1 AND bank_id = $2 ORDER BY memory_id",
        memory_id,
        bank_id,
    )
    # Two single-column arms, so each endpoint index drives its own (as delete_unit_links does).
    relations = await conn.fetch(
        f"""
        SELECT id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot, origin,
               reason, confidence, evidence, created_at
        FROM {relations_t} WHERE source_memory_id = $1 AND bank_id = $2
        UNION
        SELECT id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot, origin,
               reason, confidence, evidence, created_at
        FROM {relations_t} WHERE target_memory_id = $1 AND bank_id = $2
        ORDER BY created_at, id
        """,
        memory_id,
        bank_id,
    )

    def _evidence(row: Any) -> _Evidence:
        return _Evidence(
            memory_id=str(row["memory_id"]),
            evidence_memory_id=str(row["evidence_memory_id"]),
            created_at=row["created_at"],
        )

    return AdaptiveSnapshot(
        memory_id=str(memory_id),
        invalidated_at=now,
        retention=_Retention.model_validate(dict(retention)) if retention is not None else None,
        claim=(
            _Claim.model_validate(
                {
                    **dict(claim),
                    "superseded_by_id": _str(claim["superseded_by_id"]),
                    "inherited_from_id": _str(claim["inherited_from_id"]),
                }
            )
            if claim is not None
            else None
        ),
        events=[_Event.model_validate(dict(row)) for row in events],
        evidence_out=[_evidence(row) for row in evidence_out],
        evidence_in=[_evidence(row) for row in evidence_in],
        superseded_by_refs=[str(row["memory_id"]) for row in superseded_refs],
        inherited_from_refs=[str(row["memory_id"]) for row in inherited_refs],
        relations=[
            _Relation(
                id=str(row["id"]),
                relation_type=row["relation_type"],
                source_memory_id=_str(row["source_memory_id"]),
                target_memory_id=_str(row["target_memory_id"]),
                source_snapshot=conn.parse_json(row["source_snapshot"]) or {},
                target_snapshot=conn.parse_json(row["target_snapshot"]) or {},
                origin=row["origin"],
                reason=row["reason"],
                confidence=row["confidence"],
                evidence=conn.parse_json(row["evidence"]),
                created_at=row["created_at"],
            )
            for row in relations
        ],
    )


async def park_adaptive_state(conn, *, fq_table: Callable[[str], str], bank_id: str, unit_id: str) -> bool:
    """Snapshot ``unit_id``'s adaptive state onto its (already inserted) archive row; True if there was any.

    Called by ``invalidate_memory`` after the archive INSERT and the link delete, right before the
    ``memory_units`` DELETE. It first locks the parent row ``FOR UPDATE`` — the lock the DELETE
    takes anyway, at the same point in the lock order, so the move takes no new lock — which means
    no adaptive writer (they all hold ``FOR KEY SHARE`` on the parent while they write) can commit
    between this snapshot and the delete, and the cascade drops nothing the snapshot missed.
    """
    if not _postgres(conn):
        return False
    from ....config import get_config

    memory_id = uuid.UUID(str(unit_id))
    await conn.execute(
        f"SELECT 1 FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 FOR UPDATE", memory_id, bank_id
    )
    snapshot = await snapshot_adaptive_state(
        conn,
        fq_table=fq_table,
        bank_id=bank_id,
        memory_id=memory_id,
        now=clock.utcnow(),
        cooldown_hours=float(get_config().adaptive_memory_reinforcement_cooldown_hours),
    )
    if snapshot is None:
        return False
    await conn.execute(
        f"UPDATE {fq_table('invalidated_memory_units')} SET adaptive_state = $3::jsonb WHERE id = $1 AND bank_id = $2",
        str(memory_id),
        bank_id,
        snapshot.model_dump_json(),
    )
    return True


# --------------------------------------------------------------------------- restore


async def unpark_adaptive_state(
    conn, *, ops, fq_table: Callable[[str], str], bank_id: str, unit_id: str
) -> RematerializedState:
    """Rematerialize the archive row's snapshot for ``unit_id``, which ``restore_memory`` has just re-inserted.

    Runs before the archive row is deleted. Peers are locked ``FOR KEY SHARE`` in id order, then the
    claim rows of other memories this one re-attaches to ``FOR UPDATE`` in id order: the parents
    first and the side rows second, the order every adaptive writer uses.
    """
    nothing = RematerializedState(restored=False, dropped=0, retired=False)
    if not _postgres(conn):
        return nothing
    memory_id = uuid.UUID(str(unit_id))
    raw = await conn.fetchval(
        f"SELECT adaptive_state FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
        str(memory_id),
        bank_id,
    )
    snapshot = _parse(conn, raw)
    if snapshot is None:
        return nothing
    self_id = str(memory_id)
    # The id the snapshot's edges call this memory. Equal to self_id unless the archive was imported
    # (the importer mints fresh ids); then every peer below is a source-bank id, and dropped.
    old_id = snapshot.memory_id
    now = clock.utcnow()
    mu = fq_table("memory_units")
    claim_t = fq_table("memory_claim_state")
    evidence_t = fq_table("memory_claim_evidence")
    relations_t = fq_table("memory_lifecycle_relations")

    peers: set[str] = set()
    peers.update(e.evidence_memory_id for e in snapshot.evidence_out)
    peers.update(e.memory_id for e in snapshot.evidence_in)
    peers.update(snapshot.superseded_by_refs)
    peers.update(snapshot.inherited_from_refs)
    if snapshot.claim is not None:
        peers.update(p for p in (snapshot.claim.superseded_by_id, snapshot.claim.inherited_from_id) if p)
    for relation in snapshot.relations:
        peers.update(p for p in (relation.source_memory_id, relation.target_memory_id) if p)
    peers.discard(old_id)
    live: set[str] = set()
    if peers:
        rows = await conn.fetch(
            f"SELECT id FROM {mu} WHERE bank_id = $1 AND id = ANY($2::uuid[]) ORDER BY id FOR KEY SHARE",
            bank_id,
            sorted(uuid.UUID(p) for p in peers),
        )
        live = {str(r["id"]) for r in rows}
    dropped = 0

    if snapshot.retention is not None:
        r = snapshot.retention
        await conn.execute(
            f"""
            INSERT INTO {fq_table("memory_retention_state")}
                (memory_id, bank_id, importance, stability_days, last_reinforced_at, last_recalled_at,
                 reinforcement_count, access_count, forgetting_exempt, exemption_reason, retention_state,
                 below_threshold_since, archived_at, model_version, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
            memory_id,
            bank_id,
            r.importance,
            r.stability_days,
            r.last_reinforced_at,
            r.last_recalled_at,
            r.reinforcement_count,
            r.access_count,
            r.forgetting_exempt,
            r.exemption_reason,
            r.retention_state,
            r.below_threshold_since,
            r.archived_at,
            r.model_version,
            r.created_at,
            r.updated_at,
        )
    if snapshot.events:
        rows_to_insert = [
            ReinforcementEventRow(
                id=uuid.uuid4(),
                memory_id=memory_id,
                source_type=e.source_type,
                source_key=e.source_key,
                event_time=e.event_time,
                weight=e.weight,
                independence_weight=e.independence_weight,
                applied_gain=e.applied_gain,
                outcome=e.outcome,
                processed_at=e.processed_at or e.created_at,
                created_at=e.created_at,
            )
            for e in snapshot.events
        ]
        await ops.insert_reinforcement_events(
            conn, table=fq_table("memory_reinforcement_events"), bank_id=bank_id, rows=rows_to_insert
        )

    retired = False
    if snapshot.claim is not None:
        c = snapshot.claim
        superseded_by = c.superseded_by_id if c.superseded_by_id in live else None
        inherited_from = c.inherited_from_id if c.inherited_from_id in live else None
        for wanted, kept in ((c.superseded_by_id, superseded_by), (c.inherited_from_id, inherited_from)):
            if wanted is not None and kept is None:
                dropped += 1
        await conn.execute(
            f"""
            INSERT INTO {claim_t}
                (memory_id, bank_id, truth_status, confidence, valid_from, valid_to, retired_via, superseded_by_id,
                 inherited_from_id, scope_fingerprint, reason, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
            memory_id,
            bank_id,
            c.truth_status,
            c.confidence,
            c.valid_from,
            c.valid_to,
            c.retired_via,
            uuid.UUID(superseded_by) if superseded_by else None,
            uuid.UUID(inherited_from) if inherited_from else None,
            c.scope_fingerprint,
            c.reason,
            c.created_at,
            c.updated_at,
        )
        retired = c.truth_status in _RETIRED

    # (claim memory, evidence memory, created_at) rows to write, re-keyed onto the restored id.
    evidence_rows = [
        _Evidence(memory_id=self_id, evidence_memory_id=e.evidence_memory_id, created_at=e.created_at)
        for e in snapshot.evidence_out
        if e.evidence_memory_id in live
    ]
    dropped += len(snapshot.evidence_out) - len(evidence_rows)

    # Inbound edges: only onto claims no lifecycle action has touched since the invalidation.
    inbound = (
        {e.memory_id for e in snapshot.evidence_in}
        | set(snapshot.superseded_by_refs)
        | set(snapshot.inherited_from_refs)
    ) & live
    eligible: dict[str, Any] = {}
    if inbound:
        for row in await conn.fetch(
            f"SELECT memory_id, superseded_by_id, inherited_from_id, updated_at FROM {claim_t} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) ORDER BY memory_id FOR UPDATE",
            bank_id,
            sorted(uuid.UUID(m) for m in inbound),
        ):
            if row["updated_at"] <= snapshot.invalidated_at:
                eligible[str(row["memory_id"])] = row
    for e in snapshot.evidence_in:
        if e.memory_id in eligible:
            evidence_rows.append(_Evidence(memory_id=e.memory_id, evidence_memory_id=self_id, created_at=e.created_at))
        else:
            dropped += 1
    if evidence_rows:
        await conn.execute(
            f"""
            INSERT INTO {evidence_t} (memory_id, evidence_memory_id, bank_id, created_at)
            SELECT m, e, $1, c FROM unnest($2::uuid[], $3::uuid[], $4::timestamptz[]) AS u(m, e, c)
            ORDER BY m, e
            ON CONFLICT (memory_id, evidence_memory_id) DO NOTHING
            """,
            bank_id,
            [uuid.UUID(e.memory_id) for e in evidence_rows],
            [uuid.UUID(e.evidence_memory_id) for e in evidence_rows],
            [e.created_at for e in evidence_rows],
        )
    # Pointers are written back only where the FK's SET NULL left them empty, and without stamping
    # updated_at: this undoes the cascade, it is not a new decision about that claim.
    for column, refs in (
        ("superseded_by_id", snapshot.superseded_by_refs),
        ("inherited_from_id", snapshot.inherited_from_refs),
    ):
        targets = sorted(m for m in refs if m in eligible and eligible[m][column] is None)
        dropped += len(refs) - len(targets)
        if targets:
            await conn.execute(
                f"UPDATE {claim_t} SET {column} = $1 WHERE bank_id = $2 AND memory_id = ANY($3::uuid[]) "
                f"AND {column} IS NULL",
                memory_id,
                bank_id,
                [uuid.UUID(m) for m in targets],
            )

    dropped += await _restore_relations(conn, relations_t, bank_id, self_id, old_id, snapshot, live)

    if retired:
        # Requeue invariant (DESIGN.md §7): restore_memory reset consolidated_at to NULL, but a
        # retired raw fact must never read as pending, or consolidation would re-ask about it and
        # could rebuild the retired belief as a new active observation.
        await conn.execute(
            f"UPDATE {mu} SET consolidated_at = $3 WHERE id = $1 AND bank_id = $2", memory_id, bank_id, now
        )
    if dropped:
        logger.info(
            "Restored adaptive state of memory %s in bank %s; %d edge(s) dropped (peer gone or re-decided)",
            self_id,
            bank_id,
            dropped,
        )
    return RematerializedState(restored=True, dropped=dropped, retired=retired)


async def _restore_relations(
    conn, relations_t: str, bank_id: str, self_id: str, old_id: str, snapshot: AdaptiveSnapshot, live: set[str]
) -> int:
    """Re-point each relation tombstone at the restored memory; returns how many were dropped.

    ``old_id`` is the memory's id in the snapshot's descriptors, ``self_id`` the id it is restored
    under (they differ only for an imported archive, whose relation rows never exist here).
    """
    dropped = 0
    for relation in sorted(snapshot.relations, key=lambda r: r.id):
        ours_source = relation.source_memory_id == old_id
        ours_target = relation.target_memory_id == old_id
        relation_id = uuid.UUID(relation.id)
        current = await conn.fetchrow(
            f"SELECT source_memory_id, target_memory_id FROM {relations_t} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
            relation_id,
            bank_id,
        )
        if current is None:
            peer = relation.target_memory_id if ours_source else relation.source_memory_id
            if peer is None or peer not in live:
                dropped += 1
                continue
            inserted = await conn.fetchval(
                f"""
                INSERT INTO {relations_t}
                    (id, bank_id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot,
                     origin, reason, confidence, evidence, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10, $11::jsonb, $12)
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                relation_id,
                bank_id,
                relation.relation_type,
                uuid.UUID(self_id) if ours_source else uuid.UUID(peer),
                uuid.UUID(self_id) if ours_target else uuid.UUID(peer),
                _json(relation.source_snapshot),
                _json(relation.target_snapshot),
                relation.origin,
                relation.reason,
                relation.confidence,
                _json(relation.evidence if relation.evidence is not None else {}),
                relation.created_at,
            )
            dropped += inserted is None
            continue
        source = _str(current["source_memory_id"])
        target = _str(current["target_memory_id"])
        if ours_source and source is None:
            source = self_id
        if ours_target and target is None:
            target = self_id
        if (source, target) == (_str(current["source_memory_id"]), _str(current["target_memory_id"])):
            continue
        if source is not None and target is not None:
            # uq_mlr_edge: a live edge of the same shape decided since wins; this one stays a tombstone.
            clash = await conn.fetchval(
                f"SELECT 1 FROM {relations_t} WHERE source_memory_id = $1 AND target_memory_id = $2 "
                f"AND relation_type = $3 AND id <> $4",
                uuid.UUID(source),
                uuid.UUID(target),
                relation.relation_type,
                relation_id,
            )
            if clash:
                dropped += 1
                continue
        await conn.execute(
            f"UPDATE {relations_t} SET source_memory_id = $2, target_memory_id = $3 WHERE id = $1",
            relation_id,
            uuid.UUID(source) if source else None,
            uuid.UUID(target) if target else None,
        )
    return dropped


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


# --------------------------------------------------------------------------- read view


async def archived_lifecycle(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_uuid: uuid.UUID,
    now: datetime,
    settings: AdaptiveSettings,
) -> MemoryLifecycle | None:
    """The lifecycle view of an invalidated memory, from its archive snapshot; None when it is not archived."""
    from .adaptive import retention_view

    row = await conn.fetchrow(
        f"SELECT fact_type, mentioned_at, occurred_start, event_date, adaptive_state "
        f"FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
        str(memory_uuid),
        bank_id,
    )
    if row is None:
        return None
    snapshot = _parse(conn, row["adaptive_state"])
    self_id = str(memory_uuid)
    retention: RetentionState | None = None
    exemption_reason: str | None = None
    updated_at: datetime | None = None
    if snapshot is not None and snapshot.retention is not None:
        r = snapshot.retention
        retention = RetentionState(
            memory_id=self_id,
            importance=r.importance,
            stability_days=r.stability_days,
            last_reinforced_at=r.last_reinforced_at,
            reinforcement_count=r.reinforcement_count,
            access_count=r.access_count,
            forgetting_exempt=r.forgetting_exempt,
            retention_state=r.retention_state,
            last_recalled_at=r.last_recalled_at,
            below_threshold_since=r.below_threshold_since,
            archived_at=r.archived_at,
        )
        exemption_reason = r.exemption_reason
        updated_at = r.updated_at
    claim: ClaimView | None = None
    if snapshot is not None and snapshot.claim is not None:
        c = snapshot.claim
        claim = ClaimView(
            truth_status=c.truth_status,
            confidence=c.confidence,
            valid_from=c.valid_from,
            valid_to=c.valid_to,
            retired_via=c.retired_via,
            superseded_by_id=c.superseded_by_id,
            inherited_from_id=c.inherited_from_id,
            reason=c.reason,
            evidence_memory_ids=sorted(e.evidence_memory_id for e in snapshot.evidence_out),
            updated_at=c.updated_at,
        )
    relations = snapshot.relations if snapshot is not None else []
    events = sorted(snapshot.events, key=lambda e: e.event_time, reverse=True) if snapshot is not None else []

    def _view(r: _Relation) -> RelationView:
        return RelationView(
            id=r.id,
            relation_type=r.relation_type,
            source_memory_id=r.source_memory_id,
            target_memory_id=r.target_memory_id,
            source_snapshot=r.source_snapshot,
            target_snapshot=r.target_snapshot,
            origin=r.origin,
            reason=r.reason,
            confidence=r.confidence,
            created_at=r.created_at,
        )

    return MemoryLifecycle(
        memory_id=self_id,
        archived=True,
        stored=snapshot is not None,
        retention=retention_view(
            retention,
            exemption_reason=exemption_reason,
            updated_at=updated_at,
            fact_type=row["fact_type"],
            mentioned_at=row["mentioned_at"],
            occurred_start=row["occurred_start"],
            event_date=row["event_date"],
            settings=settings,
            now=now,
        ),
        claim=claim,
        relations_out=[
            _view(r) for r in relations if snapshot is not None and r.source_memory_id == snapshot.memory_id
        ],
        relations_in=[_view(r) for r in relations if snapshot is not None and r.target_memory_id == snapshot.memory_id],
        recent_events=[
            ReinforcementEventView(
                source_type=e.source_type,
                source_key=e.source_key,
                event_time=e.event_time,
                weight=e.weight,
                independence_weight=e.independence_weight,
                applied_gain=e.applied_gain,
                outcome=e.outcome,
                processed_at=e.processed_at,
            )
            for e in events[:RECENT_EVENTS_LIMIT]
        ],
    )
