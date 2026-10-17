"""Adaptive-memory state across a bank or document transfer.

The replay gives every memory a fresh id, so the rows keyed by memory id cannot travel verbatim.
Export puts each memory's retention and claim on its ``TransferFact`` / ``TransferObservation``
(``attach_adaptive_state``) and the edges between memories in two archive sections that cite source
ids (``dump_lifecycle_relations``, ``dump_claim_evidence``); import writes them back once every
memory exists and the source → target id map is complete (``restore_*``). The reinforcement ledger
does not travel (the archive carries the state it produced; see ``transfer/export.py``), and the
curation archive's snapshots are re-keyed by ``remap_archived_state``.

Everything is optional on the wire: a bank with no adaptive state exports exactly what it did
before, and imports nothing here.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...adaptive import clock
from ...adaptive.models import TruthStatus
from ...db.ops import LifecycleRelationRow, RetentionStateUpdate
from ...transfer.schema import (
    ClaimEvidenceRestore,
    ImportedMemoryState,
    LifecycleRelationsRestore,
    TransferClaim,
    TransferClaimEvidence,
    TransferLifecycleRelation,
    TransferRetention,
)
from .adaptive_archive import AdaptiveSnapshot
from .adaptive_claims import _chunks, _insert_evidence_pairs, _ops, _stamp_consolidated, _uuids, lock_claims, set_claim

logger = logging.getLogger(__name__)


def _as_dict(conn, value: Any) -> dict[str, Any]:
    parsed = conn.parse_json(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, dict) else {}


# ------------------------------------------------------------------ export


async def attach_adaptive_state(
    conn, *, fq_table: Callable[[str], str], bank_id: str, items: Mapping[str, Any]
) -> None:
    """Set ``retention`` / ``claim`` on each exported fact or observation that has state.

    ``items`` maps a memory's id to its ``TransferFact`` / ``TransferObservation``. Two indexed
    reads per chunk of ids; a memory with no rows keeps both fields None (absent from the archive).
    """
    ids = _uuids(items)
    for chunk in _chunks(conn, ids):
        for row in await conn.fetch(
            f"SELECT memory_id, importance, stability_days, last_reinforced_at, reinforcement_count, access_count, "
            f"last_recalled_at, forgetting_exempt, exemption_reason FROM {fq_table('memory_retention_state')} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            item = items.get(str(row["memory_id"]))
            if item is not None:
                item.retention = TransferRetention(
                    importance=float(row["importance"]),
                    stability_days=float(row["stability_days"]),
                    last_reinforced_at=row["last_reinforced_at"],
                    reinforcement_count=int(row["reinforcement_count"]),
                    access_count=int(row["access_count"]),
                    last_recalled_at=row["last_recalled_at"],
                    forgetting_exempt=bool(row["forgetting_exempt"]),
                    exemption_reason=row["exemption_reason"],
                )
        for row in await conn.fetch(
            f"SELECT memory_id, truth_status, confidence, valid_from, valid_to, retired_via, superseded_by_id, "
            f"inherited_from_id, scope_fingerprint, reason, updated_at FROM {fq_table('memory_claim_state')} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            item = items.get(str(row["memory_id"]))
            if item is not None:
                confidence = row["confidence"]
                item.claim = TransferClaim(
                    truth_status=row["truth_status"],
                    confidence=None if confidence is None else float(confidence),
                    valid_from=row["valid_from"],
                    valid_to=row["valid_to"],
                    retired_via=row["retired_via"],
                    superseded_by_id=None if row["superseded_by_id"] is None else str(row["superseded_by_id"]),
                    inherited_from_id=None if row["inherited_from_id"] is None else str(row["inherited_from_id"]),
                    scope_fingerprint=row["scope_fingerprint"],
                    reason=row["reason"],
                    updated_at=row["updated_at"],
                )


async def dump_lifecycle_relations(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Collection[str] | None
) -> list[TransferLifecycleRelation]:
    """The bank's lifecycle relations for an archive, oldest first.

    ``memory_ids`` is the exported set (None: the whole bank, tombstones included). With a set, a
    relation is carried when one endpoint is in it; the other endpoint, if not exported, travels as
    a tombstone (its snapshot says what it was).
    """
    exported = None if memory_ids is None else {str(m) for m in memory_ids}
    rows = await conn.fetch(
        f"SELECT id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot, origin, "
        f"reason, confidence, evidence, created_at FROM {fq_table('memory_lifecycle_relations')} "
        f"WHERE bank_id = $1 ORDER BY created_at, id",
        bank_id,
    )
    out: list[TransferLifecycleRelation] = []
    for row in rows:
        source = None if row["source_memory_id"] is None else str(row["source_memory_id"])
        target = None if row["target_memory_id"] is None else str(row["target_memory_id"])
        if exported is not None:
            source = source if source in exported else None
            target = target if target in exported else None
            if source is None and target is None:
                continue
        confidence = row["confidence"]
        out.append(
            TransferLifecycleRelation(
                source_id=str(row["id"]),
                relation_type=row["relation_type"],
                source_memory_id=source,
                target_memory_id=target,
                source_snapshot=_as_dict(conn, row["source_snapshot"]),
                target_snapshot=_as_dict(conn, row["target_snapshot"]),
                origin=row["origin"],
                reason=row["reason"],
                confidence=None if confidence is None else float(confidence),
                evidence=_as_dict(conn, row["evidence"]),
                created_at=row["created_at"],
            )
        )
    return out


async def dump_claim_evidence(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Collection[str] | None
) -> list[TransferClaimEvidence]:
    """The bank's claim-evidence rows whose two memories are both exported (None: the whole bank)."""
    exported = None if memory_ids is None else {str(m) for m in memory_ids}
    rows = await conn.fetch(
        f"SELECT memory_id, evidence_memory_id, created_at FROM {fq_table('memory_claim_evidence')} "
        f"WHERE bank_id = $1 ORDER BY memory_id, evidence_memory_id",
        bank_id,
    )
    return [
        TransferClaimEvidence(
            memory_id=str(row["memory_id"]),
            evidence_memory_id=str(row["evidence_memory_id"]),
            created_at=row["created_at"],
        )
        for row in rows
        if exported is None or (str(row["memory_id"]) in exported and str(row["evidence_memory_id"]) in exported)
    ]


def strip_archived_events(conn, value: Any) -> Any:
    """An archived memory's ``adaptive_state`` without its reinforcement ledger, for export.

    Events never leave the deployment that recorded them (their keys name its requests and
    sessions); the snapshot keeps the retention they produced. The value keeps its encoding (JSON
    text stays text, a decoded object stays an object), so the importer reads it as before.
    """
    state = _as_dict(conn, value)
    if not state:
        return value
    stripped = AdaptiveSnapshot.model_validate(state).model_dump(mode="json", exclude={"events"})
    return json.dumps(stripped) if isinstance(value, str) else stripped


# ------------------------------------------------------------------ import


async def restore_memory_state(
    conn, *, fq_table: Callable[[str], str], bank_id: str, rows: Sequence[ImportedMemoryState]
) -> int:
    """Write imported claims and retention onto their new memories; how many memories got state.

    Claims keep the source's ``updated_at`` (when the decision was made). A raw fact whose claim
    is superseded or refuted is stamped consolidated, so it never enters the target's queue
    (the requeue invariant).
    """
    rows = [r for r in rows if r.claim is not None or r.retention is not None]
    if not rows:
        return 0
    now = clock.utcnow()
    with_claim = {r.unit_id: r.claim for r in rows if r.claim is not None}
    locked = (
        await lock_claims(
            conn, fq_table=fq_table, bank_id=bank_id, memory_ids=list(with_claim), skip_locked=False, now=now
        )
        if with_claim
        else set()
    )
    retired: list[str] = []
    for unit_id, claim in sorted(with_claim.items()):
        if claim is None or unit_id not in locked:
            continue
        status = TruthStatus(claim.truth_status)
        await set_claim(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            memory_id=unit_id,
            truth_status=status,
            valid_from=claim.valid_from,
            valid_to=claim.valid_to,
            confidence=claim.confidence,
            scope_fingerprint=claim.scope_fingerprint,
            retired_via=claim.retired_via,
            superseded_by_id=claim.superseded_by_id,
            inherited_from_id=claim.inherited_from_id,
            reason=claim.reason,
            now=claim.updated_at or now,
        )
        if status in (TruthStatus.SUPERSEDED, TruthStatus.REFUTED):
            retired.append(unit_id)
    if retired:
        await _stamp_consolidated(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=retired, now=now)

    with_retention = {r.unit_id: r.retention for r in rows if r.retention is not None}
    if with_retention:
        ops = _ops(conn)
        lock = await ops.upsert_default_retention_state(
            conn,
            retention_table=fq_table("memory_retention_state"),
            units_table=fq_table("memory_units"),
            bank_id=bank_id,
            memory_ids=_uuids(with_retention),
            default_stability={},
            fallback_stability=1.0,
            anchors={},
            now=now,
            skip_locked=False,
        )
        updates = [
            RetentionStateUpdate(
                memory_id=uuid.UUID(key),
                importance=retention.importance,
                stability_days=retention.stability_days,
                last_reinforced_at=retention.last_reinforced_at,
                reinforcement_count=retention.reinforcement_count,
                access_count=retention.access_count,
                last_recalled_at=retention.last_recalled_at,
                forgetting_exempt=retention.forgetting_exempt,
                exemption_reason=retention.exemption_reason,
            )
            for key, retention in sorted(with_retention.items())
            if retention is not None and key in set(lock.present)
        ]
        await ops.update_retention_state(
            conn, table=fq_table("memory_retention_state"), bank_id=bank_id, rows=updates, now=now
        )
    return len({*locked, *with_retention})


async def restore_lifecycle_relations(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    rows: Sequence[TransferLifecycleRelation],
    unit_id_map: Mapping[str, str],
) -> LifecycleRelationsRestore:
    """Insert imported relations with fresh ids and target endpoints; an endpoint that did not come
    back is written as a tombstone (and counted), its snapshot kept."""
    if not rows:
        return LifecycleRelationsRestore()
    ops = _ops(conn)
    table = fq_table("memory_lifecycle_relations")
    id_map: dict[str, str] = {}
    dropped = 0
    for relation in rows:
        endpoints: list[uuid.UUID | None] = []
        for source_id in (relation.source_memory_id, relation.target_memory_id):
            mapped = unit_id_map.get(source_id) if source_id is not None else None
            if source_id is not None and mapped is None:
                dropped += 1
            endpoints.append(uuid.UUID(mapped) if mapped is not None else None)
        row = LifecycleRelationRow(
            id=uuid.uuid4(),
            bank_id=bank_id,
            relation_type=relation.relation_type,
            source_memory_id=endpoints[0],
            target_memory_id=endpoints[1],
            source_snapshot=json.dumps(relation.source_snapshot),
            target_snapshot=json.dumps(relation.target_snapshot),
            origin=relation.origin,
            reason=relation.reason,
            confidence=relation.confidence,
            evidence=json.dumps(relation.evidence, default=str),
            created_at=relation.created_at or clock.utcnow(),
        )
        if await ops.insert_lifecycle_relation(conn, table=table, row=row):
            id_map[relation.source_id] = str(row.id)
    if dropped:
        logger.info(
            "[transfer] %d lifecycle relation endpoint(s) of bank %s did not resolve (tombstoned)", dropped, bank_id
        )
    return LifecycleRelationsRestore(id_map=id_map, dropped_endpoints=dropped)


async def restore_claim_evidence(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    rows: Sequence[TransferClaimEvidence],
    unit_id_map: Mapping[str, str],
) -> ClaimEvidenceRestore:
    """Insert imported evidence rows whose two memories both came back; the rest are counted."""
    pairs: list[tuple[uuid.UUID, uuid.UUID]] = []
    dropped = 0
    for row in rows:
        holder = unit_id_map.get(row.memory_id)
        evidence = unit_id_map.get(row.evidence_memory_id)
        if holder is None or evidence is None:
            dropped += 1
            continue
        pairs.append((uuid.UUID(holder), uuid.UUID(evidence)))
    inserted = await _insert_evidence_pairs(conn, fq_table=fq_table, bank_id=bank_id, pairs=pairs, now=None)
    if dropped:
        logger.info(
            "[transfer] %d claim-evidence row(s) of bank %s named a memory that was not imported", dropped, bank_id
        )
    return ClaimEvidenceRestore(inserted=inserted, dropped=dropped)


@dataclass(frozen=True)
class RemappedSnapshot:
    """An archived memory's snapshot re-keyed onto the target, and how many references were dropped."""

    state: dict[str, Any]
    dropped: int


def remap_archived_state(
    state: Mapping[str, Any], *, new_id: str, id_map: Mapping[str, str], relation_id_map: Mapping[str, str]
) -> RemappedSnapshot:
    """An archived memory's ``adaptive_state`` re-keyed onto the target bank.

    The memory itself becomes ``new_id``; every other id goes through ``id_map`` (live and archived
    memories of the import); an id that does not resolve is dropped (an edge) or cleared (a pointer)
    and counted. Relation descriptors take the id their imported row got, or a fresh one (restore
    re-inserts such a row only if its peer is live). Ledger events are dropped (they never travel).
    """
    snapshot = AdaptiveSnapshot.model_validate(state)
    old_id = snapshot.memory_id
    dropped = 0

    def _peer(value: str | None) -> str | None:
        nonlocal dropped
        if value is None:
            return None
        if value == old_id:
            return new_id
        mapped = id_map.get(value)
        if mapped is None:
            dropped += 1
        return mapped

    claim = snapshot.claim
    if claim is not None:
        claim = claim.model_copy(
            update={
                "superseded_by_id": _peer(claim.superseded_by_id),
                "inherited_from_id": _peer(claim.inherited_from_id),
            }
        )
    evidence_out = [
        e.model_copy(update={"memory_id": new_id, "evidence_memory_id": peer})
        for e in snapshot.evidence_out
        if (peer := _peer(e.evidence_memory_id)) is not None
    ]
    evidence_in = [
        e.model_copy(update={"memory_id": peer, "evidence_memory_id": new_id})
        for e in snapshot.evidence_in
        if (peer := _peer(e.memory_id)) is not None
    ]
    relations = [
        r.model_copy(
            update={
                "id": relation_id_map.get(r.id) or str(uuid.uuid4()),
                "source_memory_id": _peer(r.source_memory_id),
                "target_memory_id": _peer(r.target_memory_id),
            }
        )
        for r in snapshot.relations
    ]
    remapped = snapshot.model_copy(
        update={
            "memory_id": new_id,
            "claim": claim,
            "events": [],
            "evidence_out": evidence_out,
            "evidence_in": evidence_in,
            "superseded_by_refs": [p for p in (_peer(v) for v in snapshot.superseded_by_refs) if p is not None],
            "inherited_from_refs": [p for p in (_peer(v) for v in snapshot.inherited_from_refs) if p is not None],
            "relations": relations,
        }
    )
    return RemappedSnapshot(state=remapped.model_dump(mode="json", exclude={"events"}), dropped=dropped)


__all__ = [
    "RemappedSnapshot",
    "attach_adaptive_state",
    "dump_claim_evidence",
    "dump_lifecycle_relations",
    "remap_archived_state",
    "restore_claim_evidence",
    "restore_lifecycle_relations",
    "restore_memory_state",
    "strip_archived_events",
]
