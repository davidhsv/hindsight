"""Claim lifecycle on the SQL store: truth status, evidence, relations, requeue, delete repair, re-ingest.

The writers behind ``MemoriesExtension.lock_claims`` / ``cas_retire`` / ``set_claim`` /
``add_claim_evidence`` / ``insert_relation`` / ``inherit_from_observation`` and the invariants the
rest of the store keeps for them:

- **Locking.** ``lock_claims`` is the only way to take claim-row locks: one sorted pass that upserts
  default ``active`` rows (parents ``FOR KEY SHARE``, the ``bulk_insert_links`` shape) and then
  locks the claim rows ``FOR UPDATE`` in id order. Every other writer here assumes its rows were
  locked by that pass in the same transaction.
- **Requeue invariant.** A superseded or refuted raw fact never has ``consolidated_at IS NULL``:
  ``requeue_units`` / ``requeue_bank`` are the only statements that clear it, and they skip such
  facts. ``tests/test_requeue_choke_point.py`` fails on any other.
- **Delete repair.** Every primitive that physically deletes raw facts calls
  ``repair_lifecycle_for_deleted_facts`` first: a claim whose evidence is ALL among the doomed facts
  reverts to active (its raw fact is requeued). Invalidation, edits and observation rebuilds do not
  call it. ``relock_relations_before_delete`` orders the relation rows the delete's SET NULL cascade
  will touch.
- **Retired observations are history.** The stale-observation sweep never deletes a superseded or
  refuted observation; the delete repair drops deleted facts from its sources, a re-ingest re-points
  them at the re-extracted facts, and it is deleted only when its claim would revert to active while
  it cites no live memory (a current belief standing on nothing).
- **Re-ingest.** ``stash_reingest_lifecycle`` (before a document's old facts are deleted) and
  ``carry_forward_reingest`` (after the re-extracted facts are inserted) move the state of a fact
  whose text comes back unchanged onto its new id, and re-point the evidence and pointers other
  claims held on it, so an unchanged re-ingest neither resurrects nor drops a supersession.

Statements are in the shared SQL dialect unless they differ per backend, in which case they are a
``DataAccessOps`` method. Id lists are chunked at the Oracle IN-list limit on Oracle only. Every
timestamp written is the caller's ``now`` (``adaptive.clock``), never the database's.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ...adaptive import clock
from ...adaptive.claims import (
    RETIRED_STATUSES_SQL,
    REVERTED_EVIDENCE_DELETED,
    CarryForwardResult,
    InboundReference,
    ReingestStash,
    StashedClaim,
    StashedEvent,
    StashedRetention,
    StashedUnit,
    reingest_text_key,
)
from ...adaptive.models import ClaimState, TruthStatus
from ...adaptive.views import ClaimShadowView
from ...db.ops import LifecycleRelationRow, ReinforcementEventRow, RetentionStateUpdate
from ...db.ops_oracle import ORACLE_IN_LIST_LIMIT
from ...search.tags import TagsMatch, build_tags_where_clause_simple
from .adaptive import _claim_from_row

logger = logging.getLogger(__name__)

_RAW_FACT_TYPES_SQL = "'experience', 'world'"
_RETIREABLE_SQL = "'active', 'uncertain'"
#: ``retired_via`` values whose successor an observation rebuild re-links.
_RELINKABLE_VIA_SQL = "'supersedes', 'refutes', 'inherited'"
_CLAIM_COLUMNS = (
    "memory_id AS claim_memory_id, truth_status, confidence, valid_from, valid_to, retired_via, "
    "superseded_by_id, inherited_from_id, scope_fingerprint, reason"
)


# --------------------------------------------------------------------------- helpers


def _now(now: datetime | None) -> datetime:
    value = now if now is not None else clock.utcnow()
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _uuid(value: Any) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _uuids(values: Iterable[Any]) -> list[uuid.UUID]:
    """Distinct parseable ids, sorted (the lock order every writer shares)."""
    return sorted({u for u in (_uuid(v) for v in values) if u is not None})


def _chunks(conn, ids: Sequence[Any]) -> list[list[Any]]:
    """``ids`` in IN-list sized pieces on Oracle (ORA-01795); one piece on PostgreSQL."""
    items = list(ids)
    if not items:
        return []
    if getattr(conn, "backend_type", None) == "oracle":
        return [items[i : i + ORACLE_IN_LIST_LIMIT] for i in range(0, len(items), ORACLE_IN_LIST_LIMIT)]
    return [items]


def _rowcount(status: Any) -> int:
    try:
        return int(str(status).split()[-1])
    except (ValueError, IndexError):
        return 0


def _str_list(conn, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = conn.parse_json(value)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    return [str(v) for v in value]


def _ops(conn):
    from .graph import _ops_for

    return _ops_for(conn)


def _validate_status(status: TruthStatus | str) -> TruthStatus:
    return status if isinstance(status, TruthStatus) else TruthStatus(status)


# --------------------------------------------------------------------------- locking and the claim row


async def lock_claims(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_ids: Sequence[Any],
    skip_locked: bool,
    now: datetime | None = None,
) -> set[str]:
    """Upsert default active claim rows for the live ids and lock every claim row, sorted; the ids locked."""
    ids = _uuids(memory_ids)
    if not ids:
        return set()
    locked = await _ops(conn).upsert_default_claim_state(
        conn,
        claims_table=fq_table("memory_claim_state"),
        units_table=fq_table("memory_units"),
        bank_id=bank_id,
        memory_ids=ids,
        now=_now(now),
        skip_locked=skip_locked,
    )
    return set(locked)


def _pointer_sql(units: str, param: int) -> str:
    """A pointer column value: the id when it is a live memory of the bank, else NULL (the FK's rule)."""
    return f"(SELECT u.id FROM {units} u WHERE u.id = ${param} AND u.bank_id = $1)"


async def cas_retire(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_id: str,
    to_status: TruthStatus | str,
    retired_via: str,
    superseded_by_id: str | None,
    valid_to: datetime | None,
    reason: str | None,
    confidence: float | None,
    inherited_from_id: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Move a locked claim from active|uncertain to ``to_status``; True when this call won."""
    status = _validate_status(to_status)
    if status is TruthStatus.ACTIVE:
        raise ValueError("cas_retire cannot retire to 'active'; use set_claim for a restore")
    target = _uuid(memory_id)
    if target is None:
        return False
    successor = _uuid(superseded_by_id)
    inherited = _uuid(inherited_from_id)
    if successor == target:
        successor = None
    if inherited == target:
        inherited = None
    units = fq_table("memory_units")
    result = await conn.execute(
        f"""
        UPDATE {fq_table("memory_claim_state")}
        SET truth_status = $3, retired_via = $4,
            superseded_by_id = {_pointer_sql(units, 5)},
            inherited_from_id = {_pointer_sql(units, 6)},
            valid_to = $7, reason = $8, confidence = $9, updated_at = $10
        WHERE bank_id = $1 AND memory_id = $2 AND truth_status IN ({_RETIREABLE_SQL})
        """,
        bank_id,
        target,
        status.value,
        retired_via,
        successor,
        inherited,
        valid_to,
        reason,
        confidence,
        _now(now),
    )
    return _rowcount(result) == 1


async def set_claim(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_id: str,
    truth_status: TruthStatus | str,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    confidence: float | None = None,
    scope_fingerprint: str | None = None,
    retired_via: str | None = None,
    superseded_by_id: str | None = None,
    inherited_from_id: str | None = None,
    reason: str | None = None,
    now: datetime | None = None,
) -> None:
    """Write every column of a locked claim row exactly as given (None writes NULL)."""
    status = _validate_status(truth_status)
    target = _uuid(memory_id)
    if target is None:
        return
    successor = _uuid(superseded_by_id)
    inherited = _uuid(inherited_from_id)
    units = fq_table("memory_units")
    result = await conn.execute(
        f"""
        UPDATE {fq_table("memory_claim_state")}
        SET truth_status = $3, valid_from = $4, valid_to = $5, confidence = $6, scope_fingerprint = $7,
            retired_via = $8,
            superseded_by_id = {_pointer_sql(units, 9)},
            inherited_from_id = {_pointer_sql(units, 10)},
            reason = $11, updated_at = $12
        WHERE bank_id = $1 AND memory_id = $2
        """,
        bank_id,
        target,
        status.value,
        valid_from,
        valid_to,
        confidence,
        scope_fingerprint,
        retired_via,
        None if successor == target else successor,
        None if inherited == target else inherited,
        reason,
        _now(now),
    )
    if _rowcount(result) == 0:
        # Not an error a caller can act on at run time, but a broken lock protocol (no lock_claims
        # pass for this id, or it is not live): say so instead of silently dropping the write.
        logger.warning("[LIFECYCLE] set_claim found no claim row for %s in bank %s (not locked?)", memory_id, bank_id)


async def add_claim_evidence(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_id: str,
    evidence_ids: Sequence[str],
    now: datetime | None = None,
) -> int:
    """Record the facts a claim rests on (only live ones; duplicates ignored); rows inserted."""
    holder = _uuid(memory_id)
    if holder is None:
        return 0
    return await _insert_evidence_pairs(
        conn, fq_table=fq_table, bank_id=bank_id, pairs=[(holder, e) for e in _uuids(evidence_ids)], now=now
    )


async def _insert_evidence_pairs(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    pairs: Sequence[tuple[uuid.UUID, uuid.UUID]],
    now: datetime | None,
) -> int:
    pairs = [p for p in pairs if p[0] != p[1]]
    if not pairs:
        return 0
    return await _ops(conn).insert_claim_evidence(
        conn,
        evidence_table=fq_table("memory_claim_evidence"),
        units_table=fq_table("memory_units"),
        bank_id=bank_id,
        pairs=pairs,
        now=_now(now),
    )


async def claims_for(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> dict[str, ClaimState]:
    """The claim rows of ``memory_ids`` (lock-free); an id without a row is absent, i.e. active."""
    out: dict[str, ClaimState] = {}
    for chunk in _chunks(conn, _uuids(memory_ids)):
        for row in await conn.fetch(
            f"SELECT {_CLAIM_COLUMNS} FROM {fq_table('memory_claim_state')} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            claim = _claim_from_row(row)
            out[claim.memory_id] = claim
    return out


async def _evidence_of(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> dict[str, set[str]]:
    """``{claim memory id: {evidence ids}}`` for ``memory_ids`` (ids without evidence are absent)."""
    out: dict[str, set[str]] = {}
    for chunk in _chunks(conn, _uuids(memory_ids)):
        for row in await conn.fetch(
            f"SELECT memory_id, evidence_memory_id FROM {fq_table('memory_claim_evidence')} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            out.setdefault(str(row["memory_id"]), set()).add(str(row["evidence_memory_id"]))
    return out


async def claim_evidence(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> dict[str, list[str]]:
    """``{claim memory id: [evidence ids, sorted]}`` for ``memory_ids``; ids without evidence are absent."""
    return {
        memory_id: sorted(evidence)
        for memory_id, evidence in (
            await _evidence_of(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=memory_ids)
        ).items()
    }


# --------------------------------------------------------------------------- relations


@dataclass
class _EndpointSnapshot:
    """What a relation records about a live endpoint, so the edge stays readable once it is deleted."""

    text: str
    fact_type: str
    tags: list[str]
    source_memory_ids: list[str] = field(default_factory=list)


async def _snapshots(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> dict[str, _EndpointSnapshot]:
    """The audit snapshot of each live memory: text, fact_type, tags and (observations) source ids."""
    ids = _uuids(memory_ids)
    if not ids:
        return {}
    out: dict[str, _EndpointSnapshot] = {}
    observations: list[str] = []
    for chunk in _chunks(conn, ids):
        for row in await conn.fetch(
            f"SELECT id, text, fact_type, tags FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            key = str(row["id"])
            out[key] = _EndpointSnapshot(
                text=row["text"], fact_type=row["fact_type"], tags=_str_list(conn, row["tags"])
            )
            if row["fact_type"] == "observation":
                observations.append(key)
    if observations:
        sources = await observation_sources(conn, fq_table=fq_table, bank_id=bank_id, observation_ids=observations)
        for key, source_ids in sources.items():
            out[key].source_memory_ids = source_ids
    return out


async def insert_relation(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    relation_type: str,
    source_id: str | None,
    target_id: str | None,
    origin: str,
    reason: str | None,
    confidence: float | None,
    evidence: Mapping[str, Any],
    now: datetime | None = None,
) -> str | None:
    """Write one relation with both endpoint snapshots; its id, or None when the edge already exists.

    An endpoint that is not a live memory of the bank is written as a tombstone (NULL, empty
    snapshot); with neither endpoint live there is nothing to relate and nothing is written.
    """
    source = _uuid(source_id)
    target = _uuid(target_id)
    snaps = await _snapshots(
        conn, fq_table=fq_table, bank_id=bank_id, memory_ids=[i for i in (source, target) if i is not None]
    )
    source = source if source is not None and str(source) in snaps else None
    target = target if target is not None and str(target) in snaps else None
    if source is None and target is None:
        return None
    row = LifecycleRelationRow(
        id=uuid.uuid4(),
        bank_id=bank_id,
        relation_type=relation_type,
        source_memory_id=source,
        target_memory_id=target,
        source_snapshot=json.dumps(asdict(snaps[str(source)]) if source else {}),
        target_snapshot=json.dumps(asdict(snaps[str(target)]) if target else {}),
        origin=origin,
        reason=reason,
        confidence=confidence,
        evidence=json.dumps(dict(evidence), default=str),
        created_at=_now(now),
    )
    inserted = await _ops(conn).insert_lifecycle_relation(conn, table=fq_table("memory_lifecycle_relations"), row=row)
    return str(row.id) if inserted else None


async def relock_relations_before_delete(
    conn, *, fq_table: Callable[[str], str], bank_id: str | None, unit_ids: Sequence[Any]
) -> None:
    """Lock the relation rows touching ``unit_ids`` in one total order before their parents go."""
    ids = _uuids(unit_ids)
    if not ids or not bank_id:
        return
    await _ops(conn).lock_relations_for_delete(
        conn, table=fq_table("memory_lifecycle_relations"), bank_id=bank_id, unit_ids=ids
    )


# --------------------------------------------------------------------------- observations and inheritance


async def observation_sources(
    conn, *, fq_table: Callable[[str], str], bank_id: str, observation_ids: Sequence[Any]
) -> dict[str, list[str]]:
    """``{observation id: [source ids]}`` for the observations of ``observation_ids`` that exist."""
    ids = _uuids(observation_ids)
    if not ids:
        return {}
    ops = _ops(conn)
    out: dict[str, list[str]] = {}
    for chunk in _chunks(conn, ids):
        if ops.uses_observation_sources_table:
            for row in await conn.fetch(
                f"SELECT id FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND id = ANY($2::uuid[]) AND fact_type = 'observation'",
                bank_id,
                chunk,
            ):
                out[str(row["id"])] = []
            for row in await conn.fetch(
                f"SELECT observation_id, source_id FROM {fq_table('observation_sources')} "
                f"WHERE observation_id = ANY($1::uuid[]) ORDER BY observation_id, source_id",
                chunk,
            ):
                if str(row["observation_id"]) in out:
                    out[str(row["observation_id"])].append(str(row["source_id"]))
        else:
            for row in await conn.fetch(
                f"SELECT id, source_memory_ids FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND id = ANY($2::uuid[]) AND fact_type = 'observation'",
                bank_id,
                chunk,
            ):
                out[str(row["id"])] = [str(s) for s in (row["source_memory_ids"] or [])]
    return out


async def retired_claim_ids(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> set[str]:
    """The ids among ``memory_ids`` whose claim is superseded or refuted (lock-free)."""
    out: set[str] = set()
    for chunk in _chunks(conn, _uuids(memory_ids)):
        for row in await conn.fetch(
            f"SELECT memory_id FROM {fq_table('memory_claim_state')} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND truth_status IN ({RETIRED_STATUSES_SQL})",
            bank_id,
            chunk,
        ):
            out.add(str(row["memory_id"]))
    return out


async def _bank_has_retired_claims(conn, *, fq_table: Callable[[str], str], bank_id: str) -> bool:
    """Whether any memory of the bank is superseded or refuted: one probe of ``(bank_id, truth_status)``,
    so a bank that never retired anything (every off-mode bank) skips the source-overlap search."""
    return bool(
        await conn.fetchval(
            f"SELECT 1 FROM {fq_table('memory_claim_state')} "
            f"WHERE bank_id = $1 AND truth_status IN ({RETIRED_STATUSES_SQL}) LIMIT 1",
            bank_id,
        )
    )


async def _retired_observations_citing(
    conn, *, fq_table: Callable[[str], str], bank_id: str, unit_ids: Sequence[Any]
) -> dict[str, list[str]]:
    """``{observation id: its sources}`` for the superseded or refuted observations citing any of ``unit_ids``."""
    from .writes import observations_for_sources

    if not unit_ids or not await _bank_has_retired_claims(conn, fq_table=fq_table, bank_id=bank_id):
        return {}
    citing: set[str] = set()
    for chunk in _chunks(conn, _uuids(unit_ids)):
        citing |= {
            obs.unit_id
            for obs in await observations_for_sources(
                conn=conn, ops=_ops(conn), fq_table=fq_table, bank_id=bank_id, unit_ids=chunk
            )
        }
    retired = await retired_claim_ids(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(citing))
    if not retired:
        return {}
    return await observation_sources(conn, fq_table=fq_table, bank_id=bank_id, observation_ids=sorted(retired))


async def _set_observation_sources(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    sources: Sequence[str],
    removed: Sequence[str],
) -> None:
    """Rewrite an observation's source list (and Oracle's junction rows); nothing else on the row moves.

    Bookkeeping on a retired observation's provenance, so ``updated_at`` stays put, as in
    ``attach_sources_to_observation``.
    """
    await conn.execute(
        f"UPDATE {fq_table('memory_units')} SET source_memory_ids = $1, proof_count = $2 "
        f"WHERE id = $3 AND bank_id = $4 AND fact_type = 'observation'",
        [uuid.UUID(s) for s in sources],
        len(sources),
        uuid.UUID(observation_id),
        bank_id,
    )
    if removed and _ops(conn).uses_observation_sources_table:
        for chunk in _chunks(conn, _uuids(removed)):
            await conn.execute(
                f"DELETE FROM {fq_table('observation_sources')} WHERE observation_id = $1 AND source_id = ANY($2::uuid[])",
                uuid.UUID(observation_id),
                chunk,
            )


async def _detach_from_retired_observations(
    conn, *, fq_table: Callable[[str], str], bank_id: str, unit_ids: Sequence[Any]
) -> dict[str, tuple[str, ...]]:
    """Drop ``unit_ids`` from the sources of the superseded/refuted observations citing them.

    A retired observation is history, not a summary of its sources: losing one does not make its
    text stale, so it keeps its row (the stale-observation sweep leaves it alone) and only the ids
    that are about to stop existing leave its source list. Returns ``{observation id: the ids it
    lost}``. The observations themselves are never among ``unit_ids``' targets.
    """
    gone = {str(u) for u in _uuids(unit_ids)}
    detached: dict[str, tuple[str, ...]] = {}
    for observation_id, sources in sorted(
        (await _retired_observations_citing(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sorted(gone))).items()
    ):
        if observation_id in gone:
            continue
        removed = tuple(s for s in sources if s in gone)
        if not removed:
            continue
        await _set_observation_sources(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            observation_id=observation_id,
            sources=[s for s in sources if s not in gone],
            removed=removed,
        )
        detached[observation_id] = removed
    return detached


async def _observations_left_unsourced(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any], doomed: set[str]
) -> set[str]:
    """The observations among ``memory_ids`` that cite no live memory outside ``doomed``.

    Such an observation must not become current again: it would stand on nothing (its sources were
    deleted or are archived by curation). The callers delete it instead of reverting it.
    """
    sources = await observation_sources(conn, fq_table=fq_table, bank_id=bank_id, observation_ids=memory_ids)
    if not sources:
        return set()
    remaining = {s for cited in sources.values() for s in cited if s not in doomed}
    live: set[str] = set()
    for chunk in _chunks(conn, _uuids(remaining)):
        live |= {
            str(row["id"])
            for row in await conn.fetch(
                f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = ANY($2::uuid[])",
                bank_id,
                chunk,
            )
        }
    return {observation_id for observation_id, cited in sources.items() if not any(s in live for s in cited)}


async def _delete_unsourced_observations(
    conn, *, fq_table: Callable[[str], str], bank_id: str, observation_ids: Sequence[Any]
) -> int:
    """Delete observations that stand on nothing, with their history (as the stale-observation sweep does).

    The caller has locked their relation rows (``relock_relations_before_delete``) and their claims.
    """
    deleted = 0
    for chunk in _chunks(conn, _uuids(observation_ids)):
        await conn.execute(
            f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1 AND observation_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        )
        deleted += _rowcount(
            await conn.execute(
                f"DELETE FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND id = ANY($2::uuid[]) AND fact_type = 'observation'",
                bank_id,
                chunk,
            )
        )
    if deleted:
        logger.info("[LIFECYCLE] deleted %d retired observation(s) left with no source in bank %s", deleted, bank_id)
    return deleted


async def _observations_citing(
    conn, *, fq_table: Callable[[str], str], bank_id: str, unit_ids: Sequence[Any]
) -> dict[str, set[str]]:
    """``{source id: {observation ids citing it}}`` (GIN ``&&`` on PG, the junction on Oracle)."""
    from .writes import observations_for_sources

    wanted = {str(u) for u in _uuids(unit_ids)}
    out: dict[str, set[str]] = {}
    for chunk in _chunks(conn, sorted(wanted)):
        for obs in await observations_for_sources(
            conn=conn, ops=_ops(conn), fq_table=fq_table, bank_id=bank_id, unit_ids=chunk
        ):
            for source in obs.source_memory_ids:
                if source in wanted:
                    out.setdefault(source, set()).add(obs.unit_id)
    return out


async def _stamp_consolidated(
    conn, *, fq_table: Callable[[str], str], bank_id: str, unit_ids: Sequence[Any], now
) -> None:
    """Stamp ``consolidated_at`` on raw facts that have none (the requeue invariant's other half)."""
    for chunk in _chunks(conn, _uuids(unit_ids)):
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET consolidated_at = $3, consolidation_failed_at = NULL "
            f"WHERE bank_id = $1 AND id = ANY($2::uuid[]) AND consolidated_at IS NULL "
            f"AND fact_type IN ({_RAW_FACT_TYPES_SQL})",
            bank_id,
            chunk,
            _now(now),
        )


async def inherit_from_observation(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    to_status: TruthStatus | str,
    successor_id: str | None,
    valid_to: datetime,
    evidence_ids: Sequence[str],
    candidate_source_ids: Sequence[str] | None,
    reason: str | None = None,
    retired_via: str = "inherited",
    ignored_citer_ids: Sequence[str] = (),
    now: datetime | None = None,
) -> list[str]:
    """Retire the raw sources of a retired observation that no live observation still stands on.

    See the store interface for the exclusions; candidates must already be locked by the caller's
    ``lock_claims`` pass (an unlocked candidate has no claim row and is skipped).
    """
    status = _validate_status(to_status)
    observation = _uuid(observation_id)
    if observation is None or status is TruthStatus.ACTIVE:
        return []
    obs_key = str(observation)
    sources = (await observation_sources(conn, fq_table=fq_table, bank_id=bank_id, observation_ids=[observation])).get(
        obs_key
    )
    if not sources:
        return []
    source_set = set(sources)
    if candidate_source_ids is None:
        candidates = sorted(source_set)
    else:
        candidates = sorted({str(c) for c in _uuids(candidate_source_ids)} & source_set)
    if not candidates:
        return []

    obs_row = await conn.fetchrow(
        f"SELECT tags FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = $2", bank_id, observation
    )
    obs_tags = set(_str_list(conn, obs_row["tags"])) if obs_row is not None else set()
    eligible: list[str] = []
    for chunk in _chunks(conn, [uuid.UUID(c) for c in candidates]):
        for row in await conn.fetch(
            f"SELECT id, fact_type, occurred_start, tags FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            # Dated events and experiences stay true (they happened); a source in a narrower scope
            # than the observation is not this observation's to retire (the pool's all_strict rule).
            if row["fact_type"] != "world" or row["occurred_start"] is not None:
                continue
            if not obs_tags.issubset(set(_str_list(conn, row["tags"]))):
                continue
            eligible.append(str(row["id"]))
    if not eligible:
        return []

    citing = await _observations_citing(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=eligible)
    ignored = {obs_key, *(str(i) for i in _uuids(ignored_citer_ids))}
    others = sorted({o for obs_ids in citing.values() for o in obs_ids if o not in ignored})
    other_claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=others)

    def _live_elsewhere(fact_id: str) -> bool:
        for other in citing.get(fact_id, set()):
            if other in ignored:
                continue
            claim = other_claims.get(other)
            if claim is None or claim.truth_status in (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN):
                return True
        return False

    eligible = [f for f in eligible if not _live_elsewhere(f)]
    if not eligible:
        return []

    current = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=eligible)
    # Rows locked by the caller's lock_claims pass cannot change under us, so the CAS outcome is
    # known from this read: active|uncertain rows win, retired ones and rows never locked do not.
    winners = sorted(
        f for f in eligible if f in current and current[f].truth_status in (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN)
    )
    if not winners:
        return []
    when = _now(now)
    units = fq_table("memory_units")
    successor = _uuid(successor_id)
    for chunk in _chunks(conn, [uuid.UUID(w) for w in winners]):
        await conn.execute(
            f"""
            UPDATE {fq_table("memory_claim_state")}
            SET truth_status = $3, retired_via = $9,
                superseded_by_id = {_pointer_sql(units, 4)},
                inherited_from_id = {_pointer_sql(units, 5)},
                valid_to = $6, reason = $7, updated_at = $8
            WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND truth_status IN ({_RETIREABLE_SQL})
            """,
            bank_id,
            chunk,
            status.value,
            successor,
            observation,
            valid_to,
            reason if reason is not None else f"inherited:{status.value}",
            when,
            retired_via,
        )
    evidence = _uuids(evidence_ids)
    if evidence:
        await _insert_evidence_pairs(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            pairs=[(uuid.UUID(w), e) for w in winners for e in evidence],
            now=when,
        )
    await _stamp_consolidated(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=winners, now=when)
    return winners


async def relinkable_claims(
    conn, *, fq_table: Callable[[str], str], bank_id: str, evidence_ids: Sequence[Any]
) -> set[str]:
    """Retired claims with no successor, retired via supersedes/refutes/inherited, that cite any of
    ``evidence_ids`` as evidence: the ones ``relink_rebuilt_successors`` may re-link once an
    observation citing those facts exists (lock-free)."""
    candidates: set[str] = set()
    for chunk in _chunks(conn, _uuids(evidence_ids)):
        for row in await conn.fetch(
            f"""
            SELECT DISTINCT e.memory_id
            FROM {fq_table("memory_claim_evidence")} e
            JOIN {fq_table("memory_claim_state")} c ON c.memory_id = e.memory_id
            WHERE e.bank_id = $1 AND e.evidence_memory_id = ANY($2::uuid[])
              AND c.superseded_by_id IS NULL
              AND c.retired_via IN ({_RELINKABLE_VIA_SQL})
              AND c.truth_status IN ({RETIRED_STATUSES_SQL})
            """,
            bank_id,
            chunk,
        ):
            candidates.add(str(row["memory_id"]))
    return candidates


async def relink_rebuilt_successors(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    new_observation_ids: Sequence[Any],
    now: datetime | None = None,
    locked_claim_ids: Collection[str] | None = None,
) -> int:
    """Point retired claims whose successor was rebuilt at the new observation that carries their evidence.

    ``locked_claim_ids``: the claims the caller's own ``lock_claims`` pass already holds (it put the
    ``relinkable_claims`` of the new observations' sources in it). Only those are re-linked, and the
    only claim rows locked here are the new observations' own, which no other transaction can see
    yet, so the caller's single sorted pass stays the only one over contended rows; a candidate
    outside it is left for the next rebuild. None: this call takes its own pass over every candidate.
    """
    new_sources = await observation_sources(
        conn, fq_table=fq_table, bank_id=bank_id, observation_ids=new_observation_ids
    )
    if not new_sources:
        return 0
    all_sources = sorted({s for sources in new_sources.values() for s in sources})
    if not all_sources:
        return 0
    candidates = await relinkable_claims(conn, fq_table=fq_table, bank_id=bank_id, evidence_ids=all_sources)
    candidates -= set(new_sources)
    if locked_claim_ids is not None:
        candidates &= set(locked_claim_ids)
    if not candidates:
        return 0
    evidence = await _evidence_of(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(candidates))
    chosen: dict[str, str] = {}
    for claim_id in sorted(candidates):
        facts = evidence.get(claim_id, set())
        if not facts:
            continue
        covering = [(len(src), obs) for obs, src in new_sources.items() if facts.issubset(set(src))]
        if covering:
            chosen[claim_id] = min(covering)[1]
    if not chosen:
        return 0
    when = _now(now)
    if locked_claim_ids is None:
        locked = await lock_claims(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            memory_ids=[*chosen, *set(chosen.values())],
            skip_locked=False,
            now=when,
        )
    else:
        fresh = sorted(set(chosen.values()) - set(locked_claim_ids))
        locked = set(locked_claim_ids) | (
            await lock_claims(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=fresh, skip_locked=False, now=when)
            if fresh
            else set()
        )
    claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=[*chosen, *set(chosen.values())])
    claims_table = fq_table("memory_claim_state")
    relinked = 0
    successor_valid_from: dict[str, datetime] = {}
    for claim_id, obs_id in sorted(chosen.items()):
        claim = claims.get(claim_id)
        successor = claims.get(obs_id)
        if claim_id not in locked or obs_id not in locked or claim is None:
            continue
        if claim.superseded_by_id is not None or claim.truth_status not in (
            TruthStatus.SUPERSEDED,
            TruthStatus.REFUTED,
        ):
            continue
        if successor is not None and successor.truth_status not in (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN):
            continue
        result = await conn.execute(
            f"UPDATE {claims_table} SET superseded_by_id = $3 WHERE bank_id = $1 AND memory_id = $2 "
            f"AND superseded_by_id IS NULL",
            bank_id,
            uuid.UUID(claim_id),
            uuid.UUID(obs_id),
        )
        if _rowcount(result):
            relinked += 1
            if claim.valid_to is not None and claim.retired_via != "inherited":
                prior = successor_valid_from.get(obs_id)
                successor_valid_from[obs_id] = claim.valid_to if prior is None else max(prior, claim.valid_to)
    for obs_id, valid_from in sorted(successor_valid_from.items()):
        await conn.execute(
            f"UPDATE {claims_table} SET valid_from = $3, updated_at = $4 "
            f"WHERE bank_id = $1 AND memory_id = $2 AND valid_from IS NULL",
            bank_id,
            uuid.UUID(obs_id),
            valid_from,
            when,
        )
    return relinked


async def count_active_observations(
    conn, *, fq_table: Callable[[str], str], bank_id: str, tags: list[str] | None, tags_match: TagsMatch
) -> int:
    """Observations in the tag scope whose claim is not superseded|refuted (the lifecycle scope cap)."""
    params: list[Any] = [bank_id]
    tags_clause = build_tags_where_clause_simple(tags, 2, table_alias="mu.", match=tags_match)
    if tags:
        params.append(tags)
    value = await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {fq_table("memory_units")} mu
        WHERE mu.bank_id = $1 AND mu.fact_type = 'observation' {tags_clause}
          AND NOT EXISTS (
              SELECT 1 FROM {fq_table("memory_claim_state")} c
              WHERE c.memory_id = mu.id AND c.truth_status IN ({RETIRED_STATUSES_SQL})
          )
        """,
        *params,
    )
    return int(value or 0)


def retired_claim_filter_sql(fq_table: Callable[[str], str], unit_column: str) -> str:
    """``AND NOT EXISTS (superseded|refuted claim)`` for a recall arm (``exclude_non_active``)."""
    return (
        f" AND NOT EXISTS (SELECT 1 FROM {fq_table('memory_claim_state')} xc "
        f"WHERE xc.memory_id = {unit_column} AND xc.truth_status IN ({RETIRED_STATUSES_SQL}))"
    )


# --------------------------------------------------------------------------- observe-mode shadow


async def record_shadow_retirement(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_id: str,
    would_status: TruthStatus | str,
    deciding_snapshot: Mapping[str, Any],
    evidence_ids: Sequence[str],
    now: datetime | None = None,
) -> bool:
    """Record what observe mode would have retired ``memory_id`` as (latest decision wins)."""
    status = _validate_status(would_status)
    if status not in (TruthStatus.SUPERSEDED, TruthStatus.REFUTED):
        raise ValueError(f"would_status must be superseded or refuted, got {status.value!r}")
    target = _uuid(memory_id)
    if target is None:
        return False
    return await _ops(conn).upsert_claim_shadow(
        conn,
        table=fq_table("memory_claim_shadow"),
        units_table=fq_table("memory_units"),
        bank_id=bank_id,
        memory_id=target,
        would_status=status.value,
        deciding_snapshot=json.dumps(dict(deciding_snapshot), default=str),
        evidence_ids=json.dumps([str(e) for e in _uuids(evidence_ids)]),
        now=_now(now),
    )


async def shadow_for(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any]
) -> dict[str, ClaimShadowView]:
    """The observe-mode shadow records of ``memory_ids``; ids without one are absent."""
    out: dict[str, ClaimShadowView] = {}
    for chunk in _chunks(conn, _uuids(memory_ids)):
        for row in await conn.fetch(
            f"SELECT memory_id, would_status, deciding_snapshot, evidence_ids, created_at "
            f"FROM {fq_table('memory_claim_shadow')} WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            snapshot = conn.parse_json(row["deciding_snapshot"])
            out[str(row["memory_id"])] = ClaimShadowView(
                would_status=row["would_status"],
                deciding_snapshot=snapshot if isinstance(snapshot, dict) else {},
                evidence_ids=_str_list(conn, row["evidence_ids"]),
                created_at=row["created_at"],
            )
    return out


# --------------------------------------------------------------------------- requeue (the choke point)


def _requeue_set(clear_failed: bool) -> str:
    return "consolidated_at = NULL" + (", consolidation_failed_at = NULL" if clear_failed else "")


#: Never requeue a superseded or refuted fact: consolidating it again would let the
#: model rebuild a current observation from a retired belief.
def _not_retired(fq_table: Callable[[str], str]) -> str:
    return (
        f"NOT EXISTS (SELECT 1 FROM {fq_table('memory_claim_state')} c "
        f"WHERE c.memory_id = mu.id AND c.truth_status IN ({RETIRED_STATUSES_SQL}))"
    )


async def requeue_units(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: Sequence[Any],
    reason: str,
    clear_failed: bool = False,
) -> int:
    """Return raw facts to the consolidation queue, skipping retired ones; how many were requeued.

    Bookkeeping, not an edit: ``updated_at`` is left alone (``META_UPDATED_AT`` in ``..base``).
    """
    requeued = 0
    for chunk in _chunks(conn, _uuids(unit_ids)):
        result = await conn.execute(
            f"""
            UPDATE {fq_table("memory_units")} mu
            SET {_requeue_set(clear_failed)}
            WHERE mu.bank_id = $1 AND mu.id = ANY($2::uuid[])
              AND mu.fact_type IN ({_RAW_FACT_TYPES_SQL})
              AND {_not_retired(fq_table)}
            """,
            bank_id,
            chunk,
        )
        requeued += _rowcount(result)
    logger.debug("[LIFECYCLE] requeued %d unit(s) in bank %s (%s)", requeued, bank_id, reason)
    return requeued


async def requeue_bank(
    conn, *, fq_table: Callable[[str], str], bank_id: str, reason: str, only_failed: bool = False
) -> int:
    """``requeue_units`` over a whole bank: every raw fact, or (``only_failed``) the failed ones.

    ``only_failed`` also clears the failure marker, which is what a retry of failed consolidation is.
    """
    failed_clause = " AND mu.consolidation_failed_at IS NOT NULL" if only_failed else ""
    result = await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")} mu
        SET {_requeue_set(only_failed)}
        WHERE mu.bank_id = $1 AND mu.fact_type IN ({_RAW_FACT_TYPES_SQL}){failed_clause}
          AND {_not_retired(fq_table)}
        """,
        bank_id,
    )
    requeued = _rowcount(result)
    logger.debug("[LIFECYCLE] requeued %d unit(s) in bank %s (%s)", requeued, bank_id, reason)
    return requeued


# --------------------------------------------------------------------------- delete repair


async def _revert_claims(
    conn, *, fq_table: Callable[[str], str], bank_id: str, claim_ids: Sequence[str], reason_tag: str, now: datetime
) -> int:
    """Return locked claims to active (validity end, retirement and pointers cleared), requeue raw ones."""
    ids = _uuids(claim_ids)
    if not ids:
        return 0
    reverted = 0
    for chunk in _chunks(conn, ids):
        result = await conn.execute(
            f"""
            UPDATE {fq_table("memory_claim_state")}
            SET truth_status = 'active', valid_to = NULL, retired_via = NULL, superseded_by_id = NULL,
                inherited_from_id = NULL, reason = $3, updated_at = $4
            WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND truth_status <> 'active'
            """,
            bank_id,
            chunk,
            REVERTED_EVIDENCE_DELETED,
            now,
        )
        reverted += _rowcount(result)
    await requeue_units(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=ids, reason="lifecycle_revert")
    from ....metrics import get_metrics_collector

    get_metrics_collector().record_adaptive_claims_reverted(bank_id=bank_id, reason=reason_tag, count=reverted)
    if reverted:
        logger.info("[LIFECYCLE] reverted %d claim(s) in bank %s (%s)", reverted, bank_id, reason_tag)
    return reverted


async def _claims_losing_evidence(
    conn, *, fq_table: Callable[[str], str], bank_id: str, doomed: Sequence[uuid.UUID]
) -> set[str]:
    """Claims that cite a doomed fact as evidence or name one as their successor."""
    found: set[str] = set()
    for chunk in _chunks(conn, list(doomed)):
        for row in await conn.fetch(
            f"""
            SELECT memory_id FROM {fq_table("memory_claim_evidence")}
            WHERE bank_id = $1 AND evidence_memory_id = ANY($2::uuid[])
            UNION
            SELECT memory_id FROM {fq_table("memory_claim_state")}
            WHERE bank_id = $1 AND superseded_by_id = ANY($2::uuid[])
            """,
            bank_id,
            chunk,
        ):
            found.add(str(row["memory_id"]))
    return found


async def repair_lifecycle_for_deleted_facts(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str | None,
    doomed_ids: Sequence[Any],
    now: datetime | None = None,
) -> int:
    """Revert the claims that only the doomed facts justified; called before the physical delete.

    A non-active claim reverts to active when every evidence fact it cites is doomed (at least one
    is), or when its successor is doomed and none of its evidence survives. The doomed memories' own
    claims go with them. Relation rows are never touched: they stay as SET NULL tombstones.

    Retired observations outlive their sources: the superseded/refuted observations citing a doomed
    fact only lose it from their source list (their row, text and history stay; their claim stands
    on its evidence, not on its sources). One whose claim would revert while it cites no live memory
    any more is deleted instead, since it would become a current belief standing on nothing.
    """
    doomed = _uuids(doomed_ids)
    if not doomed or not bank_id:
        return 0
    doomed_keys = {str(d) for d in doomed}
    await _detach_from_retired_observations(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=doomed)
    candidates = await _claims_losing_evidence(conn, fq_table=fq_table, bank_id=bank_id, doomed=doomed)
    candidates -= doomed_keys
    if not candidates:
        return 0
    # Decided before the claim pass so their relation rows are locked in the same order as the
    # doomed facts' (relations, then claims); deleted only if their claim does revert.
    unsourced = await _observations_left_unsourced(
        conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(candidates), doomed=doomed_keys
    )
    if unsourced:
        await relock_relations_before_delete(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sorted(unsourced))
    when = _now(now)
    locked = await lock_claims(
        conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(candidates), skip_locked=False, now=when
    )
    claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(locked))
    evidence = await _evidence_of(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(locked))
    revert: list[str] = []
    for claim_id in sorted(locked):
        claim = claims.get(claim_id)
        if claim is None or claim.truth_status is TruthStatus.ACTIVE:
            continue
        facts = evidence.get(claim_id, set())
        if facts:
            if facts <= doomed_keys:
                revert.append(claim_id)
        elif claim.superseded_by_id in doomed_keys:
            revert.append(claim_id)
    await _delete_unsourced_observations(
        conn, fq_table=fq_table, bank_id=bank_id, observation_ids=[c for c in revert if c in unsourced]
    )
    return await _revert_claims(
        conn,
        fq_table=fq_table,
        bank_id=bank_id,
        claim_ids=[c for c in revert if c not in unsourced],
        reason_tag="evidence_deleted",
        now=when,
    )


# --------------------------------------------------------------------------- re-ingest carry-forward


async def stash_reingest_lifecycle(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    old_unit_ids: Sequence[Any],
    now: datetime | None = None,
) -> ReingestStash | None:
    """Snapshot the lifecycle state on a document's outgoing facts and detach what other claims hold on them.

    Call inside the re-ingest's transaction, before the old facts are deleted. The superseded and
    refuted observations citing an outgoing fact lose it from their sources here (the carry-forward
    adds the re-extracted fact back). Returns None (and costs one probe, plus one index probe for a
    retired claim in the bank) when no outgoing fact carries any state and nothing points at one.
    """
    old = _uuids(old_unit_ids)
    if not old:
        return None
    claims_t = fq_table("memory_claim_state")
    evidence_t = fq_table("memory_claim_evidence")
    retention_t = fq_table("memory_retention_state")
    events_t = fq_table("memory_reinforcement_events")
    relations_t = fq_table("memory_lifecycle_relations")

    # One probe, each arm on its own index: in a bank that never wrote adaptive state this is the
    # whole cost of carry-forward.
    probe: set[str] = set()
    for chunk in _chunks(conn, old):
        for row in await conn.fetch(
            f"""
            SELECT memory_id AS hit_id FROM {claims_t} WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])
            UNION SELECT memory_id FROM {retention_t} WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])
            UNION SELECT memory_id FROM {events_t} WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])
            UNION SELECT evidence_memory_id FROM {evidence_t} WHERE bank_id = $1 AND evidence_memory_id = ANY($2::uuid[])
            UNION SELECT superseded_by_id FROM {claims_t} WHERE bank_id = $1 AND superseded_by_id = ANY($2::uuid[])
            UNION SELECT inherited_from_id FROM {claims_t} WHERE bank_id = $1 AND inherited_from_id = ANY($2::uuid[])
            UNION SELECT source_memory_id FROM {relations_t} WHERE bank_id = $1 AND source_memory_id = ANY($2::uuid[])
            UNION SELECT target_memory_id FROM {relations_t} WHERE bank_id = $1 AND target_memory_id = ANY($2::uuid[])
            """,
            bank_id,
            chunk,
        ):
            probe.add(str(row["hit_id"]))
    # The retired observations citing the outgoing facts, whose sources the delete would otherwise
    # leave dangling (looked up only in a bank that holds a retired claim at all: one index probe).
    retired_sources = await _detach_from_retired_observations(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=old)
    if not probe and not retired_sources:
        return None

    old_keys = {str(o) for o in old}
    own_claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=old)
    own_evidence = await _evidence_of(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=old)
    retention: dict[str, StashedRetention] = {}
    events: dict[str, list[StashedEvent]] = {}
    inbound: list[InboundReference] = []
    for chunk in _chunks(conn, old):
        for row in await conn.fetch(
            f"SELECT memory_id, importance, stability_days, last_reinforced_at, reinforcement_count, access_count, "
            f"last_recalled_at, forgetting_exempt, exemption_reason FROM {retention_t} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            retention[str(row["memory_id"])] = StashedRetention(
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
            f"SELECT memory_id, source_type, source_key, event_time, weight, independence_weight, applied_gain, "
            f"outcome, processed_at, created_at FROM {events_t} WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            events.setdefault(str(row["memory_id"]), []).append(
                StashedEvent(
                    source_type=row["source_type"],
                    source_key=row["source_key"],
                    event_time=row["event_time"],
                    weight=float(row["weight"]),
                    independence_weight=float(row["independence_weight"]),
                    applied_gain=float(row["applied_gain"]),
                    outcome=row["outcome"],
                    processed_at=row["processed_at"],
                    created_at=row["created_at"],
                )
            )
        for row in await conn.fetch(
            f"SELECT memory_id, evidence_memory_id FROM {evidence_t} "
            f"WHERE bank_id = $1 AND evidence_memory_id = ANY($2::uuid[])",
            bank_id,
            chunk,
        ):
            if str(row["memory_id"]) not in old_keys:
                inbound.append(InboundReference("evidence", str(row["memory_id"]), str(row["evidence_memory_id"])))
        for column, kind in (("superseded_by_id", "superseded_by"), ("inherited_from_id", "inherited_from")):
            for row in await conn.fetch(
                f"SELECT memory_id, {column} FROM {claims_t} WHERE bank_id = $1 AND {column} = ANY($2::uuid[])",
                bank_id,
                chunk,
            ):
                if str(row["memory_id"]) not in old_keys:
                    inbound.append(InboundReference(kind, str(row["memory_id"]), str(row[column])))  # type: ignore[arg-type]
        for column, kind in (("source_memory_id", "relation_source"), ("target_memory_id", "relation_target")):
            for row in await conn.fetch(
                f"SELECT id, {column} FROM {relations_t} WHERE bank_id = $1 AND {column} = ANY($2::uuid[])",
                bank_id,
                chunk,
            ):
                inbound.append(InboundReference(kind, str(row["id"]), str(row[column])))  # type: ignore[arg-type]

    # In-document edges (a fact's claim citing, or pointing at, another fact of the same document)
    # are kept on the holder's own claim; their targets must be matchable too.
    in_document = {e for cited in own_evidence.values() for e in cited} | {
        pointer for c in own_claims.values() for pointer in (c.superseded_by_id, c.inherited_from_id) if pointer
    }
    involved = (
        set(own_claims)
        | set(retention)
        | set(events)
        | {ref.old_unit_id for ref in inbound}
        | {s for cited in retired_sources.values() for s in cited}
        | in_document
    ) & old_keys
    if not involved:
        return None
    units: dict[str, StashedUnit] = {}
    for chunk in _chunks(conn, sorted(uuid.UUID(i) for i in involved)):
        for row in await conn.fetch(
            f"SELECT id, text, fact_type FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = ANY($2::uuid[]) "
            f"ORDER BY created_at, id",
            bank_id,
            chunk,
        ):
            key = str(row["id"])
            claim = own_claims.get(key)
            units[key] = StashedUnit(
                unit_id=key,
                fact_type=row["fact_type"],
                text_key=reingest_text_key(row["text"]),
                claim=(
                    StashedClaim(
                        truth_status=claim.truth_status,
                        confidence=claim.confidence,
                        valid_from=claim.valid_from,
                        valid_to=claim.valid_to,
                        retired_via=claim.retired_via,
                        superseded_by_id=claim.superseded_by_id,
                        inherited_from_id=claim.inherited_from_id,
                        scope_fingerprint=claim.scope_fingerprint,
                        reason=claim.reason,
                        evidence_ids=tuple(sorted(own_evidence.get(key, set()))),
                    )
                    if claim is not None
                    else None
                ),
                retention=retention.get(key),
                events=tuple(events.get(key, ())),
            )

    # Detach what other claims hold on the outgoing facts, so the delete's own repair sees none of
    # it: whether such a claim keeps its evidence is decided once the re-extracted facts exist.
    holders = sorted({ref.holder_id for ref in inbound if ref.kind in ("evidence", "superseded_by", "inherited_from")})
    if holders:
        when = _now(now)
        await lock_claims(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=holders, skip_locked=False, now=when)
        holder_ids = [uuid.UUID(h) for h in holders]
        for chunk in _chunks(conn, old):
            for holder_chunk in _chunks(conn, holder_ids):
                await conn.execute(
                    f"DELETE FROM {evidence_t} WHERE bank_id = $1 AND evidence_memory_id = ANY($2::uuid[]) "
                    f"AND memory_id = ANY($3::uuid[])",
                    bank_id,
                    chunk,
                    holder_chunk,
                )
                for column in ("superseded_by_id", "inherited_from_id"):
                    await conn.execute(
                        f"UPDATE {claims_t} SET {column} = NULL WHERE bank_id = $1 AND {column} = ANY($2::uuid[]) "
                        f"AND memory_id = ANY($3::uuid[])",
                        bank_id,
                        chunk,
                        holder_chunk,
                    )
    return ReingestStash(
        bank_id=bank_id, document_id=document_id, units=units, inbound=inbound, retired_sources=retired_sources
    )


def _merged_retention(
    carried: StashedRetention, existing: Any | None, settings: Any | None
) -> RetentionStateUpdate | None:
    """The carried learned state, under an importance/exemption this same retain set explicitly."""
    from ...adaptive.retention import rescale_for_importance

    if existing is None:
        return None
    importance = carried.importance
    stability = carried.stability_days
    exempt = carried.forgetting_exempt
    exemption_reason = carried.exemption_reason
    existing_importance = float(existing["importance"])
    if existing_importance != carried.importance and settings is not None:
        importance = existing_importance
        stability = rescale_for_importance(carried.stability_days, carried.importance, importance, settings)
    if bool(existing["forgetting_exempt"]):
        exempt = True
        exemption_reason = existing["exemption_reason"] or exemption_reason
    return RetentionStateUpdate(
        memory_id=uuid.UUID(str(existing["memory_id"])),
        importance=importance,
        stability_days=stability,
        last_reinforced_at=carried.last_reinforced_at,
        reinforcement_count=carried.reinforcement_count,
        access_count=carried.access_count,
        last_recalled_at=carried.last_recalled_at,
        forgetting_exempt=exempt,
        exemption_reason=exemption_reason,
    )


async def carry_forward_reingest(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    stash: ReingestStash | None,
    new_unit_ids: Sequence[Any],
    final: bool,
    now: datetime | None = None,
    settings: Any | None = None,
) -> CarryForwardResult:
    """Move stashed state onto re-extracted facts with the same (document, text, fact type).

    Call after the new facts are inserted, in the same transaction. A streaming retain inserts a
    document over several transactions and calls this once per batch; ``final`` (the last call)
    reverts every claim whose evidence was only on facts no batch re-extracted, and counts them.

    A carried claim's in-document edges (evidence or a pointer on another fact of the same
    document) are re-keyed whichever batch re-extracts the other end: the call that matches it adds
    the edge to a holder carried earlier. The retired observations the stash detached get each
    re-extracted fact back as a source; one left citing nothing keeps its row (it is history), and
    one whose claim the final call would revert while it stands on nothing is deleted instead.
    """
    if stash is None or stash.empty:
        return CarryForwardResult()
    ops = _ops(conn)
    when = _now(now)
    units_t = fq_table("memory_units")

    # A batch transaction that rolled back took its re-extracted facts with it: forget those pairs.
    if stash.matched:
        live: set[str] = set()
        for chunk in _chunks(conn, _uuids(stash.matched.values())):
            for row in await conn.fetch(
                f"SELECT id FROM {units_t} WHERE bank_id = $1 AND id = ANY($2::uuid[])", bank_id, chunk
            ):
                live.add(str(row["id"]))
        stash.matched = {o: n for o, n in stash.matched.items() if n in live}

    taken = set(stash.matched.values())
    by_key: dict[tuple[str, str], list[str]] = {}
    for chunk in _chunks(conn, _uuids(new_unit_ids)):
        for row in await conn.fetch(
            f"SELECT id, text, fact_type FROM {units_t} WHERE bank_id = $1 AND document_id = $2 "
            f"AND id = ANY($3::uuid[]) ORDER BY created_at, id",
            bank_id,
            stash.document_id,
            chunk,
        ):
            if str(row["id"]) not in taken:
                by_key.setdefault((reingest_text_key(row["text"]), row["fact_type"]), []).append(str(row["id"]))
    pairs: dict[str, str] = {}
    for old_id, unit in stash.units.items():
        if old_id in stash.matched:
            continue
        candidates = by_key.get((unit.text_key, unit.fact_type))
        if candidates:
            pairs[old_id] = candidates.pop(0)

    full_map = {**stash.matched, **pairs}

    def _remap(value: str | None) -> str | None:
        return full_map.get(value, value) if value is not None else None

    # Claims carried by an earlier call whose evidence or pointers name a fact matched only now: the
    # edge the earlier call could not re-key (its target did not exist yet) is added here.
    late_edges: list[tuple[str, str, str]] = []
    for old_holder, new_holder in stash.matched.items():
        held = stash.units[old_holder].claim
        if held is None:
            continue
        late_edges += [(new_holder, "evidence", pairs[e]) for e in held.evidence_ids if e in pairs]
        for kind, target in (("superseded_by", held.superseded_by_id), ("inherited_from", held.inherited_from_id)):
            if target is not None and target in pairs:
                late_edges.append((new_holder, kind, pairs[target]))
    late_edges = [edge for edge in late_edges if edge[0] != edge[2]]

    # The retired observations that cited an outgoing fact get its re-extracted twin back as a source.
    reattach = {
        observation_id: [uuid.UUID(pairs[c]) for c in cited if c in pairs]
        for observation_id, cited in sorted(stash.retired_sources.items())
    }
    if any(reattach.values()):
        from .consolidation import attach_sources_to_observation

        for observation_id, added in reattach.items():
            if added:
                await attach_sources_to_observation(
                    conn=conn,
                    ops=ops,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    observation_id=observation_id,
                    source_memory_ids=added,
                )

    # Every claim this call writes, locked in one sorted pass before any write (lock_claims rule).
    carried_claims = {new: stash.units[old] for old, new in pairs.items() if stash.units[old].claim is not None}
    repointed = [ref for ref in stash.inbound if ref.old_unit_id in pairs]
    unmatched_old = {o for o in stash.units if o not in full_map}
    revert_candidates: set[str] = set()
    unsourced: set[str] = set()
    if final:
        revert_candidates |= {
            ref.holder_id
            for ref in stash.inbound
            if ref.kind in ("evidence", "superseded_by") and ref.old_unit_id in unmatched_old
        }
        revert_candidates |= {
            new
            for old, new in full_map.items()
            if stash.units[old].claim is not None
            and any(e in unmatched_old for e in stash.units[old].claim.evidence_ids)  # type: ignore[union-attr]
        }
        # A retired observation that would revert standing on nothing is deleted instead; its
        # relation rows are locked before the claims, in the delete repair's order.
        unsourced = await _observations_left_unsourced(
            conn, fq_table=fq_table, bank_id=bank_id, memory_ids=sorted(revert_candidates), doomed=set()
        )
        if unsourced:
            await relock_relations_before_delete(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sorted(unsourced))
    lock_ids = sorted(
        set(carried_claims)
        | {ref.holder_id for ref in repointed if ref.kind in ("evidence", "superseded_by", "inherited_from")}
        | {holder for holder, _kind, _target in late_edges}
        | revert_candidates
    )
    locked = (
        await lock_claims(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=lock_ids, skip_locked=False, now=when)
        if lock_ids
        else set()
    )

    # The facts' own claims, evidence (re-keyed through the map) and consolidation stamp.
    retired_new: list[str] = []
    for new_id, unit in sorted(carried_claims.items()):
        claim = unit.claim
        if claim is None or new_id not in locked:
            continue
        await set_claim(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            memory_id=new_id,
            truth_status=claim.truth_status,
            valid_from=claim.valid_from,
            valid_to=claim.valid_to,
            confidence=claim.confidence,
            scope_fingerprint=claim.scope_fingerprint,
            retired_via=claim.retired_via,
            superseded_by_id=_remap(claim.superseded_by_id),
            inherited_from_id=_remap(claim.inherited_from_id),
            reason=claim.reason,
            now=when,
        )
        if claim.evidence_ids:
            await _insert_evidence_pairs(
                conn,
                fq_table=fq_table,
                bank_id=bank_id,
                pairs=[(uuid.UUID(new_id), e) for e in _uuids(_remap(e) for e in claim.evidence_ids)],
                now=when,
            )
        if claim.truth_status in (TruthStatus.SUPERSEDED, TruthStatus.REFUTED):
            retired_new.append(new_id)
    if retired_new:
        await _stamp_consolidated(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=retired_new, now=when)

    # Retention and the reinforcement ledger: what the fact learned, and the keys that dedup it.
    carried_retention = {new: stash.units[old].retention for old, new in pairs.items() if stash.units[old].retention}
    if carried_retention:
        lock = await ops.upsert_default_retention_state(
            conn,
            retention_table=fq_table("memory_retention_state"),
            units_table=units_t,
            bank_id=bank_id,
            memory_ids=_uuids(carried_retention),
            default_stability={},
            fallback_stability=1.0,
            anchors={},
            now=when,
            skip_locked=False,
        )
        updates: list[RetentionStateUpdate] = []
        for row in lock.rows:
            key = str(row["memory_id"])
            carried = carried_retention.get(key)
            if carried is None:
                continue
            if key in lock.initialized:
                updates.append(
                    RetentionStateUpdate(
                        memory_id=uuid.UUID(key),
                        importance=carried.importance,
                        stability_days=carried.stability_days,
                        last_reinforced_at=carried.last_reinforced_at,
                        reinforcement_count=carried.reinforcement_count,
                        access_count=carried.access_count,
                        last_recalled_at=carried.last_recalled_at,
                        forgetting_exempt=carried.forgetting_exempt,
                        exemption_reason=carried.exemption_reason,
                    )
                )
            else:
                merged = _merged_retention(carried, row, settings)
                if merged is not None:
                    updates.append(merged)
        await ops.update_retention_state(
            conn, table=fq_table("memory_retention_state"), bank_id=bank_id, rows=updates, now=when
        )
    ledger = [
        ReinforcementEventRow(
            id=uuid.uuid4(),
            memory_id=uuid.UUID(new),
            source_type=event.source_type,
            source_key=event.source_key,
            event_time=event.event_time,
            weight=event.weight,
            independence_weight=event.independence_weight,
            applied_gain=event.applied_gain,
            outcome=event.outcome,
            processed_at=event.processed_at or when,
            created_at=event.created_at,
        )
        for old, new in pairs.items()
        for event in stash.units[old].events
    ]
    if ledger:
        await ops.insert_reinforcement_events(
            conn, table=fq_table("memory_reinforcement_events"), bank_id=bank_id, rows=ledger
        )

    # What other claims and relations held on the outgoing facts, re-pointed at the new ids.
    evidence_pairs = [
        (uuid.UUID(ref.holder_id), uuid.UUID(pairs[ref.old_unit_id]))
        for ref in repointed
        if ref.kind == "evidence" and ref.holder_id in locked
    ]
    if evidence_pairs:
        await _insert_evidence_pairs(conn, fq_table=fq_table, bank_id=bank_id, pairs=evidence_pairs, now=when)
    claims_t = fq_table("memory_claim_state")
    relations_t = fq_table("memory_lifecycle_relations")
    for ref in repointed:
        new_id = uuid.UUID(pairs[ref.old_unit_id])
        if ref.kind in ("superseded_by", "inherited_from"):
            if ref.holder_id not in locked:
                continue
            column = f"{ref.kind}_id"
            # A re-key, not a decision: updated_at stays the time the claim was decided.
            await conn.execute(
                f"UPDATE {claims_t} SET {column} = $3 WHERE bank_id = $1 AND memory_id = $2 AND {column} IS NULL",
                bank_id,
                uuid.UUID(ref.holder_id),
                new_id,
            )
        elif ref.kind in ("relation_source", "relation_target"):
            column, other = (
                ("source_memory_id", "target_memory_id")
                if ref.kind == "relation_source"
                else ("target_memory_id", "source_memory_id")
            )
            # Re-attach the tombstone the delete left, unless an edge of the same shape exists now.
            await conn.execute(
                f"""
                UPDATE {relations_t} r SET {column} = $3
                WHERE r.bank_id = $1 AND r.id = $2 AND r.{column} IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM {relations_t} x
                      WHERE x.{column} = $3 AND x.{other} = r.{other} AND x.relation_type = r.relation_type
                  )
                """,
                bank_id,
                uuid.UUID(ref.holder_id),
                new_id,
            )
    late_evidence = [
        (uuid.UUID(holder), uuid.UUID(target))
        for holder, kind, target in late_edges
        if kind == "evidence" and holder in locked
    ]
    if late_evidence:
        await _insert_evidence_pairs(conn, fq_table=fq_table, bank_id=bank_id, pairs=late_evidence, now=when)
    for holder, kind, target in late_edges:
        if kind == "evidence" or holder not in locked:
            continue
        column = f"{kind}_id"
        await conn.execute(
            f"UPDATE {claims_t} SET {column} = $3 WHERE bank_id = $1 AND memory_id = $2 AND {column} IS NULL",
            bank_id,
            uuid.UUID(holder),
            uuid.UUID(target),
        )
    stash.matched.update(pairs)

    reverted = 0
    if final:
        to_check = sorted(c for c in revert_candidates if c in locked)
        claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=to_check)
        evidence = await _evidence_of(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=to_check)
        revert = [
            c
            for c in to_check
            if c in claims and claims[c].truth_status is not TruthStatus.ACTIVE and not evidence.get(c)
        ]
        await _delete_unsourced_observations(
            conn, fq_table=fq_table, bank_id=bank_id, observation_ids=[c for c in revert if c in unsourced]
        )
        reverted = await _revert_claims(
            conn,
            fq_table=fq_table,
            bank_id=bank_id,
            claim_ids=[c for c in revert if c not in unsourced],
            reason_tag="reingest_unmatched",
            now=when,
        )
    unmatched = len([o for o in stash.units if o not in stash.matched])
    from ....metrics import get_metrics_collector

    get_metrics_collector().record_adaptive_carry_forward(
        bank_id=bank_id, matched=len(pairs), unmatched=unmatched if final else 0
    )
    if pairs or final:
        logger.info(
            "[LIFECYCLE] re-ingest of document %s in bank %s: carried %d fact(s) forward, %d unmatched%s",
            stash.document_id,
            bank_id,
            len(pairs),
            unmatched,
            f", {reverted} claim(s) reverted" if final else "",
        )
    return CarryForwardResult(matched=len(pairs), unmatched=unmatched, reverted=reverted)


__all__ = [
    "add_claim_evidence",
    "carry_forward_reingest",
    "cas_retire",
    "claims_for",
    "count_active_observations",
    "inherit_from_observation",
    "insert_relation",
    "lock_claims",
    "observation_sources",
    "record_shadow_retirement",
    "relink_rebuilt_successors",
    "relinkable_claims",
    "relock_relations_before_delete",
    "repair_lifecycle_for_deleted_facts",
    "requeue_bank",
    "requeue_units",
    "retired_claim_filter_sql",
    "retired_claim_ids",
    "set_claim",
    "shadow_for",
    "stash_reingest_lifecycle",
]
