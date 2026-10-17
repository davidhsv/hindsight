"""Adaptive-memory state on the SQL store: ``memory_retention_state`` and ``memory_claim_state``.

The reader recall uses: one query per recall over the post-trim candidate ids (at most
``reranker_max_candidates``, 300 by default), a ``FULL OUTER JOIN`` of the two tables' rows for
those ids. Measured on a 50k-memory bank with 25k retention and 1k claim rows, probing 300 ids
(P1-NOTES.md): each side gets the plan it would get as a query of its own (an index scan on the
retention primary key; the claim side whatever its size warrants), and over a local socket the
join's 0.53 ms median matches the 0.55 ms of two separate queries. It is one round trip instead of
two, which is the cost that dominates against a networked database or read replica.

Written in the shared SQL dialect (``= ANY($n::uuid[])``), which the Oracle connection rewrites to
an IN list; ids are chunked at the Oracle IN-list limit so a large pool cannot hit ORA-01795. The
chunking costs Postgres nothing below 1000 ids, which is every recall. Every selected column keeps
an ``*_id`` name where it holds a uuid, which is how the Oracle driver knows to decode RAW(16).

The writers below (reinforcement, explicit retention policy) and the read view follow the lock
protocol in ``DataAccessOps.upsert_default_retention_state``; the dialect-specific statements live
there, the decisions here. Every timestamp they write is the caller's ``now`` (the adaptive write
clock), never the database's.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from ...adaptive.evidence import bound_key
from ...adaptive.models import (
    FINAL_EXTERNAL_RECALL,
    AdaptiveState,
    ClaimState,
    RecordedEvent,
    ReinforcementEvent,
    ReinforcementResult,
    RetentionPolicy,
    RetentionState,
    TruthStatus,
)
from ...adaptive.retention import (
    DEFAULT_IMPORTANCE,
    apply_events,
    effective_retention,
    initial_stability,
    rescale_for_importance,
)
from ...adaptive.settings import AdaptiveSettings
from ...adaptive.views import (
    RECENT_EVENTS_LIMIT,
    ClaimShadowView,
    ClaimView,
    MemoryLifecycle,
    ReinforcementEventView,
    RelationView,
    RetentionView,
)
from ...db.ops import ReinforcementEventRow, ReinforcementKey, RetentionStateUpdate
from ...db.ops_oracle import ORACLE_IN_LIST_LIMIT

logger = logging.getLogger(__name__)

#: One row per id that has either row: ``retention_memory_id`` / ``claim_memory_id`` say which.
_LOAD_SQL = """
SELECT r.memory_id AS retention_memory_id, r.importance, r.stability_days, r.last_reinforced_at,
       r.reinforcement_count, r.access_count, r.forgetting_exempt, r.retention_state, r.last_recalled_at,
       r.below_threshold_since, r.archived_at,
       c.memory_id AS claim_memory_id, c.truth_status, c.confidence, c.valid_from, c.valid_to, c.retired_via,
       c.superseded_by_id, c.inherited_from_id, c.scope_fingerprint, c.reason
FROM (
    SELECT memory_id, importance, stability_days, last_reinforced_at, reinforcement_count, access_count,
           forgetting_exempt, retention_state, last_recalled_at, below_threshold_since, archived_at
    FROM {retention}
    WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])
) r
FULL OUTER JOIN (
    SELECT memory_id, truth_status, confidence, valid_from, valid_to, retired_via, superseded_by_id,
           inherited_from_id, scope_fingerprint, reason
    FROM {claims}
    WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])
) c ON c.memory_id = r.memory_id
"""


def _ids(memory_ids: Sequence[Any]) -> list[uuid.UUID]:
    """Distinct ids as UUIDs, in first-seen order; unparseable ones are dropped (read as absent)."""
    out: dict[uuid.UUID, None] = {}
    for raw in memory_ids:
        try:
            out[raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))] = None
        except (ValueError, AttributeError, TypeError):
            continue
    return list(out)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _retention_from_row(row: Any) -> RetentionState:
    return RetentionState(
        memory_id=str(row["retention_memory_id"]),
        importance=float(row["importance"]),
        stability_days=float(row["stability_days"]),
        last_reinforced_at=row["last_reinforced_at"],
        reinforcement_count=int(row["reinforcement_count"]),
        access_count=int(row["access_count"]),
        # BOOLEAN on Postgres, NUMBER(1) on Oracle.
        forgetting_exempt=bool(row["forgetting_exempt"]),
        retention_state=str(row["retention_state"]),
        last_recalled_at=row["last_recalled_at"],
        below_threshold_since=row["below_threshold_since"],
        archived_at=row["archived_at"],
    )


def _claim_from_row(row: Any) -> ClaimState:
    confidence = row["confidence"]
    return ClaimState(
        memory_id=str(row["claim_memory_id"]),
        truth_status=TruthStatus(row["truth_status"]),
        confidence=None if confidence is None else float(confidence),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        retired_via=row["retired_via"],
        superseded_by_id=_opt_str(row["superseded_by_id"]),
        inherited_from_id=_opt_str(row["inherited_from_id"]),
        scope_fingerprint=row["scope_fingerprint"],
        reason=row["reason"],
    )


async def load_adaptive_state(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_ids: Sequence[Any],
) -> dict[str, AdaptiveState]:
    """The retention and claim rows of ``memory_ids`` in ``bank_id``; memories with neither are absent."""
    ids = _ids(memory_ids)
    if not ids:
        return {}
    sql = _LOAD_SQL.format(retention=fq_table("memory_retention_state"), claims=fq_table("memory_claim_state"))
    states: dict[str, AdaptiveState] = {}
    for start in range(0, len(ids), ORACLE_IN_LIST_LIMIT):
        for row in await conn.fetch(sql, bank_id, ids[start : start + ORACLE_IN_LIST_LIMIT]):
            retention = _retention_from_row(row) if row["retention_memory_id"] is not None else None
            claim = _claim_from_row(row) if row["claim_memory_id"] is not None else None
            memory_id = retention.memory_id if retention is not None else claim.memory_id if claim is not None else None
            if memory_id is not None:
                states[memory_id] = AdaptiveState(retention=retention, claim=claim)
    return states


#: ``->>`` on the bare column: the Oracle rewriter turns ``col->>'key'`` into ``JSON_VALUE(col,
#: '$.key')`` (a table alias in front would not survive it), and both return ``'true'`` for a JSON
#: true. Hits the (bank_id, id) primary key.
_MINTED_DOCUMENTS_SQL = """
SELECT id FROM {documents}
WHERE bank_id = $1 AND id = ANY($2::text[]) AND retain_params->>'document_id_minted' = 'true'
"""


async def minted_document_ids(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_ids: Sequence[Any],
) -> frozenset[str]:
    """The ids among ``document_ids`` whose document records a server-minted id."""
    ids = list(dict.fromkeys(str(d) for d in document_ids if d))
    if not ids:
        return frozenset()
    sql = _MINTED_DOCUMENTS_SQL.format(documents=fq_table("documents"))
    minted: set[str] = set()
    for start in range(0, len(ids), ORACLE_IN_LIST_LIMIT):
        for row in await conn.fetch(sql, bank_id, ids[start : start + ORACLE_IN_LIST_LIMIT]):
            minted.add(str(row["id"]))
    return frozenset(minted)


# --------------------------------------------------------------------------- shared helpers


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _state_from_row(row: Any) -> RetentionState:
    """A locked ``memory_retention_state`` row (``RETENTION_STATE_COLUMNS``) as a :class:`RetentionState`."""
    return RetentionState(
        memory_id=str(row["memory_id"]),
        importance=float(row["importance"]),
        stability_days=float(row["stability_days"]),
        last_reinforced_at=_as_utc(row["last_reinforced_at"]),
        reinforcement_count=int(row["reinforcement_count"]),
        access_count=int(row["access_count"]),
        forgetting_exempt=bool(row["forgetting_exempt"]),
        retention_state=str(row["retention_state"]),
        last_recalled_at=_as_utc(row["last_recalled_at"]) if row["last_recalled_at"] is not None else None,
        below_threshold_since=row["below_threshold_since"],
        archived_at=row["archived_at"],
    )


def _default_stability(settings: AdaptiveSettings) -> dict[str, float]:
    """``initial_stability`` at neutral importance for every fact type the multiplier map names."""
    return {
        fact_type: initial_stability(fact_type, DEFAULT_IMPORTANCE, settings) for fact_type in settings.type_stability
    }


def _fallback_stability(settings: AdaptiveSettings) -> float:
    # A fact type the map does not name gets initial_stability's own default multiplier (1.0). The
    # map's keys are validated fact types, so the empty string never names one.
    return initial_stability("", DEFAULT_IMPORTANCE, settings)


def _update_of(state: RetentionState, exemption_reason: str | None) -> RetentionStateUpdate:
    return RetentionStateUpdate(
        memory_id=uuid.UUID(state.memory_id),
        importance=state.importance,
        stability_days=state.stability_days,
        last_reinforced_at=state.last_reinforced_at,
        reinforcement_count=state.reinforcement_count,
        access_count=state.access_count,
        last_recalled_at=state.last_recalled_at,
        forgetting_exempt=state.forgetting_exempt,
        exemption_reason=exemption_reason,
    )


# --------------------------------------------------------------------------- reinforcement

#: Sources whose cooldown events are never written to the ledger. Their keys name a cooldown window
#: (``recall:<bucket>`` / ``reflect:<bucket>``), so the row would only stop a second use inside the
#: window, which the cooldown absorbs anyway; keeping them would grow the ledger by a row per recall.
_UNPERSISTED_COOLDOWN_SOURCES = frozenset({FINAL_EXTERNAL_RECALL, "successful_reflect_use"})


class ReinforcementLedgerConflict(RuntimeError):
    """A ledger insert met a key that was absent when this writer checked, under its row lock.

    Cannot happen while every ledger writer follows the lock protocol (the check and the insert run
    under the memory's retention-row lock), so it is raised rather than absorbed: the retention row
    was already computed with that event's gain, and writing it would count the event twice.
    """


@dataclass
class _Pending:
    index: int
    key: ReinforcementKey
    event: ReinforcementEvent


def _disabled(event: ReinforcementEvent, settings: AdaptiveSettings) -> bool:
    """Whether reinforcement is off for this event: the mode, the bank switch, or a static source switch."""
    if not (settings.reinforcement_enabled and settings.effective_mode.applies_factor):
        return True
    if event.source_type == FINAL_EXTERNAL_RECALL:
        return not settings.recall_reinforcement_enabled
    if event.source_type == "successful_reflect_use":
        return not settings.reflect_reinforcement_enabled
    return False


async def record_reinforcement(
    conn,
    *,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    events: Sequence[ReinforcementEvent],
    settings: AdaptiveSettings,
    now: datetime,
    background: bool,
    anchors: Mapping[str, datetime] | None = None,
    initial_memory_ids: frozenset[str] = frozenset(),
) -> ReinforcementResult:
    """Record reinforcement events and apply them to their memories' retention.

    Runs in a transaction of its own: a savepoint inside the caller's transaction, which is where
    it belongs (consolidation's apply, a producer's short background transaction), or a real one
    for an autocommitting caller, since the row locks must last from the ledger check to the writes.

    1. Events reinforcement is off for are ``disabled`` (nothing at all is written when the mode is
       not rank/lifecycle or the bank's switch is off); ids that do not parse are ``skipped_missing``;
       a key repeated within the call counts once, by its earliest instance.
    2. The parents and retention rows are locked (default rows created where missing); a memory that
       is not live in the bank, or — ``background`` — is being deleted, is ``skipped_missing``.
    3. Keys already in the ledger are ``deduplicated``: checked under the row lock, so the check sees
       every committed event and no concurrent writer of the same memory can be between it and the
       insert.
    4. The rest are applied per memory (:func:`apply_events`; ``initial`` semantics only for a memory in
       ``initial_memory_ids`` whose row this call created), then written: applied and initial events,
       and cooldowns except recall/reflect ones, plus one update of the changed rows.

    ``memory_units`` is never written, so ``updated_at`` (the write watermark) never moves.
    """
    now = _as_utc(now)
    recorded: list[RecordedEvent | None] = [None] * len(events)

    def _mark(index: int, event: ReinforcementEvent, outcome: Any, *, gain: float = 0.0, persisted: bool = False):
        recorded[index] = RecordedEvent(
            memory_id=str(event.memory_id),
            source_type=event.source_type,
            source_key=event.source_key,
            outcome=outcome,
            applied_gain=gain,
            persisted=persisted,
        )

    by_key: dict[ReinforcementKey, list[_Pending]] = {}
    for index, raw in enumerate(events):
        event = replace(raw, source_key=bound_key(raw.source_key))
        if _disabled(event, settings):
            _mark(index, event, "disabled")
            continue
        memory_uuid = _uuid(event.memory_id)
        if memory_uuid is None:
            _mark(index, event, "skipped_missing")
            continue
        event = replace(event, memory_id=str(memory_uuid), event_time=_as_utc(event.event_time))
        key = ReinforcementKey(memory_id=str(memory_uuid), source_type=event.source_type, source_key=event.source_key)
        by_key.setdefault(key, []).append(_Pending(index=index, key=key, event=event))

    pending: list[_Pending] = []
    for group in by_key.values():
        group.sort(key=lambda p: (p.event.event_time, p.index))
        pending.append(group[0])
        for duplicate in group[1:]:
            _mark(duplicate.index, duplicate.event, "deduplicated")

    initialized: frozenset[str] = frozenset()
    states: dict[str, RetentionState] = {}
    if pending:
        # Its own transaction (a savepoint inside the caller's): the row locks below must be held
        # from the ledger check to the writes, which an autocommitting caller would not give them.
        async with conn.transaction():
            anchor_map: dict[uuid.UUID, datetime] = {}
            for raw_id, anchor in (anchors or {}).items():
                anchor_uuid = _uuid(raw_id)
                if anchor_uuid is not None:
                    anchor_map[anchor_uuid] = _as_utc(anchor)
            lock = await ops.upsert_default_retention_state(
                conn,
                retention_table=fq_table("memory_retention_state"),
                units_table=fq_table("memory_units"),
                bank_id=bank_id,
                memory_ids=sorted({uuid.UUID(p.key.memory_id) for p in pending}),
                default_stability=_default_stability(settings),
                fallback_stability=_fallback_stability(settings),
                anchors=anchor_map,
                now=now,
                skip_locked=background,
            )
            initialized = lock.initialized
            base = {str(row["memory_id"]): _state_from_row(row) for row in lock.rows}
            reasons = {str(row["memory_id"]): row["exemption_reason"] for row in lock.rows}
            states = dict(base)

            live: list[_Pending] = []
            for p in pending:
                if p.key.memory_id in base:
                    live.append(p)
                else:
                    _mark(p.index, p.event, "skipped_missing")
            events_table = fq_table("memory_reinforcement_events")
            existing = await ops.fetch_reinforcement_keys(
                conn, table=events_table, bank_id=bank_id, keys=[p.key for p in live]
            )
            fresh: dict[str, list[_Pending]] = {}
            for p in live:
                if p.key in existing:
                    _mark(p.index, p.event, "deduplicated")
                else:
                    fresh.setdefault(p.key.memory_id, []).append(p)

            initial_ids = {str(u) for u in (_uuid(m) for m in initial_memory_ids) if u is not None}
            rows: list[ReinforcementEventRow] = []
            for memory_id, group in fresh.items():
                by_identity = {(p.event.source_type, p.event.source_key): p for p in group}
                applied = apply_events(
                    base[memory_id],
                    [p.event for p in group],
                    settings,
                    # A re-sent creation batch for a memory that already has a row is ordinary evidence:
                    # initial semantics would re-anchor it and move last_reinforced_at backwards.
                    initial=memory_id in initial_ids and memory_id in initialized,
                )
                states[memory_id] = applied.state
                for outcome in applied.outcomes:
                    p = by_identity[(outcome.event.source_type, outcome.event.source_key)]
                    persist = outcome.outcome != "cooldown" or p.event.source_type not in _UNPERSISTED_COOLDOWN_SOURCES
                    _mark(p.index, p.event, outcome.outcome, gain=outcome.applied_gain, persisted=persist)
                    if persist:
                        rows.append(
                            ReinforcementEventRow(
                                id=uuid.uuid4(),
                                memory_id=uuid.UUID(memory_id),
                                source_type=p.event.source_type,
                                source_key=p.event.source_key,
                                event_time=p.event.event_time,
                                weight=p.event.weight,
                                independence_weight=p.event.independence_weight,
                                applied_gain=outcome.applied_gain,
                                outcome=outcome.outcome,
                                processed_at=now,
                                created_at=now,
                            )
                        )
            inserted = await ops.insert_reinforcement_events(conn, table=events_table, bank_id=bank_id, rows=rows)
            if len(inserted) != len(rows):
                missing = sorted(
                    f"{r.memory_id}/{r.source_type}/{r.source_key}" for r in rows if str(r.id) not in inserted
                )
                raise ReinforcementLedgerConflict(
                    f"reinforcement keys appeared under the retention-row lock: {missing}"
                )
            await ops.update_retention_state(
                conn,
                table=fq_table("memory_retention_state"),
                bank_id=bank_id,
                rows=[_update_of(state, reasons[mid]) for mid, state in states.items() if state != base[mid]],
                now=now,
            )

    done = tuple(r for r in recorded if r is not None)
    result = ReinforcementResult(
        applied=sum(1 for r in done if r.outcome == "applied"),
        cooldown=sum(1 for r in done if r.outcome == "cooldown"),
        initial=sum(1 for r in done if r.outcome == "initial"),
        deduplicated=sum(1 for r in done if r.outcome == "deduplicated"),
        initialized=len(initialized),
        skipped_missing=sum(1 for r in done if r.outcome == "skipped_missing"),
        disabled=sum(1 for r in done if r.outcome == "disabled"),
        events=done,
        states=states,
    )
    from ....metrics import get_metrics_collector

    # Recorded before the caller commits: a producer whose transaction later rolls back over-counts.
    # The producers run this last in their unit of work (a savepoint, or a transaction of its own),
    # so that is a failure path, and the counters are diagnostic, not an audit.
    get_metrics_collector().record_adaptive_reinforcement(
        bank_id=bank_id, initialized=result.initialized, events=result.events
    )
    return result


# --------------------------------------------------------------------------- explicit retention policy


def _check_policy(policy: RetentionPolicy) -> None:
    importance = policy.importance
    if importance is not None and (
        isinstance(importance, bool) or not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0)
    ):
        raise ValueError(f"importance must be a number in [0, 1], got {importance!r}")


async def set_retention_policies(
    conn,
    *,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    policies: Sequence[RetentionPolicy],
    settings: AdaptiveSettings,
    now: datetime,
) -> dict[str, RetentionState]:
    """Apply explicit retention settings, creating default rows where missing; returns the new states.

    Several policies for one memory merge in order (a later non-None field wins). Memories that are
    not live in the bank are absent from the result. The parents are locked without SKIP LOCKED:
    this is a request-path (or retain-transaction) write that must not silently drop a setting.
    """
    now = _as_utc(now)
    merged: dict[uuid.UUID, RetentionPolicy] = {}
    for policy in policies:
        _check_policy(policy)
        memory_uuid = _uuid(policy.memory_id)
        if memory_uuid is None:
            continue
        prior = merged.get(memory_uuid)
        merged[memory_uuid] = (
            replace(policy, memory_id=str(memory_uuid))
            if prior is None
            else RetentionPolicy(
                memory_id=str(memory_uuid),
                importance=policy.importance if policy.importance is not None else prior.importance,
                forgetting_exempt=(
                    policy.forgetting_exempt if policy.forgetting_exempt is not None else prior.forgetting_exempt
                ),
                exemption_reason=(
                    policy.exemption_reason if policy.exemption_reason is not None else prior.exemption_reason
                ),
            )
        )
    if not merged:
        return {}
    async with conn.transaction():
        lock = await ops.upsert_default_retention_state(
            conn,
            retention_table=fq_table("memory_retention_state"),
            units_table=fq_table("memory_units"),
            bank_id=bank_id,
            memory_ids=sorted(merged),
            default_stability=_default_stability(settings),
            fallback_stability=_fallback_stability(settings),
            anchors={},
            now=now,
            skip_locked=False,
        )
        states: dict[str, RetentionState] = {}
        updates: list[RetentionStateUpdate] = []
        for row in lock.rows:
            state = _state_from_row(row)
            reason = row["exemption_reason"]
            policy = merged[uuid.UUID(state.memory_id)]
            new_state = state
            new_reason = reason
            if policy.importance is not None:
                new_state = replace(
                    new_state,
                    importance=float(policy.importance),
                    stability_days=rescale_for_importance(
                        state.stability_days, state.importance, float(policy.importance), settings
                    ),
                )
            if policy.forgetting_exempt is not None:
                new_state = replace(new_state, forgetting_exempt=policy.forgetting_exempt)
            if policy.forgetting_exempt is False:
                new_reason = None
            elif policy.exemption_reason is not None:
                new_reason = policy.exemption_reason
            states[state.memory_id] = new_state
            if new_state != state or new_reason != reason:
                updates.append(_update_of(new_state, new_reason))
        await ops.update_retention_state(
            conn, table=fq_table("memory_retention_state"), bank_id=bank_id, rows=updates, now=now
        )
    return states


# --------------------------------------------------------------------------- read view


def _json_object(conn, value: Any) -> dict[str, Any]:
    parsed = conn.parse_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _json_list(conn, value: Any) -> list[str]:
    parsed = conn.parse_json(value)
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _relation_view(conn, row: Any) -> RelationView:
    confidence = row["confidence"]
    return RelationView(
        id=str(row["id"]),
        relation_type=row["relation_type"],
        source_memory_id=_opt_str(row["source_memory_id"]),
        target_memory_id=_opt_str(row["target_memory_id"]),
        source_snapshot=_json_object(conn, row["source_snapshot"]),
        target_snapshot=_json_object(conn, row["target_snapshot"]),
        origin=row["origin"],
        reason=row["reason"],
        confidence=None if confidence is None else float(confidence),
        created_at=row["created_at"],
    )


def retention_view(
    retention: RetentionState | None,
    *,
    exemption_reason: str | None,
    updated_at: datetime | None,
    fact_type: str,
    mentioned_at: datetime | None,
    occurred_start: datetime | None,
    event_date: datetime | None,
    settings: AdaptiveSettings,
    now: datetime,
) -> RetentionView:
    """The retention half of the view: the stored row, or the lazy defaults, with R at ``now``."""
    effective = effective_retention(
        AdaptiveState(retention=retention, claim=None) if retention is not None else None,
        fact_type=fact_type,
        mentioned_at=mentioned_at,
        occurred_start=occurred_start,
        event_date=event_date,
        settings=settings,
        now=now,
    )
    return RetentionView(
        materialized=effective.materialized,
        retrievability=effective.retrievability,
        stability_days=effective.stability_days,
        importance=effective.importance,
        last_reinforced_at=effective.last_reinforced_at,
        reinforcement_count=effective.reinforcement_count,
        access_count=retention.access_count if retention is not None else 0,
        last_recalled_at=retention.last_recalled_at if retention is not None else None,
        forgetting_exempt=effective.exempt,
        exemption_reason=exemption_reason,
        retention_state=retention.retention_state if retention is not None else "active",
        below_threshold_since=retention.below_threshold_since if retention is not None else None,
        archived_at=retention.archived_at if retention is not None else None,
        updated_at=updated_at,
    )


async def get_memory_lifecycle(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_id: str,
    now: datetime,
    settings: AdaptiveSettings,
) -> MemoryLifecycle | None:
    """The adaptive view of one memory (live, else archived), or None when the bank has no such memory.

    Plain single-row reads in the shared dialect. An archived (curation-invalidated) memory is read
    from the snapshot taken when it was invalidated (``invalidated_memory_units.adaptive_state``).
    """
    from .adaptive_archive import archived_lifecycle

    memory_uuid = _uuid(memory_id)
    if memory_uuid is None:
        return None
    now = _as_utc(now)
    unit = await conn.fetchrow(
        f"SELECT fact_type, mentioned_at, occurred_start, event_date FROM {fq_table('memory_units')} "
        f"WHERE id = $1 AND bank_id = $2",
        memory_uuid,
        bank_id,
    )
    if unit is None:
        return await archived_lifecycle(
            conn, fq_table=fq_table, bank_id=bank_id, memory_uuid=memory_uuid, now=now, settings=settings
        )

    retention_row = await conn.fetchrow(
        f"SELECT memory_id, importance, stability_days, last_reinforced_at, reinforcement_count, access_count, "
        f"forgetting_exempt, exemption_reason, retention_state, last_recalled_at, below_threshold_since, "
        f"archived_at, updated_at FROM {fq_table('memory_retention_state')} WHERE memory_id = $1 AND bank_id = $2",
        memory_uuid,
        bank_id,
    )
    claim_row = await conn.fetchrow(
        f"SELECT truth_status, confidence, valid_from, valid_to, retired_via, superseded_by_id, "
        f"inherited_from_id, reason, updated_at FROM {fq_table('memory_claim_state')} "
        f"WHERE memory_id = $1 AND bank_id = $2",
        memory_uuid,
        bank_id,
    )
    claim: ClaimView | None = None
    if claim_row is not None:
        evidence = await conn.fetch(
            f"SELECT evidence_memory_id FROM {fq_table('memory_claim_evidence')} "
            f"WHERE memory_id = $1 AND bank_id = $2 ORDER BY evidence_memory_id",
            memory_uuid,
            bank_id,
        )
        confidence = claim_row["confidence"]
        claim = ClaimView(
            truth_status=claim_row["truth_status"],
            confidence=None if confidence is None else float(confidence),
            valid_from=claim_row["valid_from"],
            valid_to=claim_row["valid_to"],
            retired_via=claim_row["retired_via"],
            superseded_by_id=_opt_str(claim_row["superseded_by_id"]),
            inherited_from_id=_opt_str(claim_row["inherited_from_id"]),
            reason=claim_row["reason"],
            evidence_memory_ids=[str(r["evidence_memory_id"]) for r in evidence],
            updated_at=claim_row["updated_at"],
        )
    relation_cols = (
        "id, relation_type, source_memory_id, target_memory_id, source_snapshot, target_snapshot, origin, "
        "reason, confidence, created_at"
    )
    relations_table = fq_table("memory_lifecycle_relations")
    relations_out = await conn.fetch(
        f"SELECT {relation_cols} FROM {relations_table} WHERE source_memory_id = $1 AND bank_id = $2 "
        f"ORDER BY created_at, id",
        memory_uuid,
        bank_id,
    )
    relations_in = await conn.fetch(
        f"SELECT {relation_cols} FROM {relations_table} WHERE target_memory_id = $1 AND bank_id = $2 "
        f"ORDER BY created_at, id",
        memory_uuid,
        bank_id,
    )
    event_rows = await conn.fetch(
        f"SELECT source_type, source_key, event_time, weight, independence_weight, applied_gain, outcome, "
        f"processed_at FROM {fq_table('memory_reinforcement_events')} WHERE memory_id = $1 AND bank_id = $2 "
        f"ORDER BY event_time DESC, created_at DESC LIMIT {RECENT_EVENTS_LIMIT}",
        memory_uuid,
        bank_id,
    )
    shadow_row = await conn.fetchrow(
        f"SELECT would_status, deciding_snapshot, evidence_ids, created_at FROM {fq_table('memory_claim_shadow')} "
        f"WHERE memory_id = $1 AND bank_id = $2",
        memory_uuid,
        bank_id,
    )

    retention = _state_from_row(retention_row) if retention_row is not None else None
    return MemoryLifecycle(
        memory_id=str(memory_uuid),
        archived=False,
        stored=(
            retention_row is not None
            or claim_row is not None
            or shadow_row is not None
            or bool(relations_out)
            or bool(relations_in)
            or bool(event_rows)
        ),
        retention=retention_view(
            retention,
            exemption_reason=retention_row["exemption_reason"] if retention_row is not None else None,
            updated_at=retention_row["updated_at"] if retention_row is not None else None,
            fact_type=unit["fact_type"],
            mentioned_at=unit["mentioned_at"],
            occurred_start=unit["occurred_start"],
            event_date=unit["event_date"],
            settings=settings,
            now=now,
        ),
        claim=claim,
        relations_out=[_relation_view(conn, r) for r in relations_out],
        relations_in=[_relation_view(conn, r) for r in relations_in],
        recent_events=[
            ReinforcementEventView(
                source_type=r["source_type"],
                source_key=r["source_key"],
                event_time=r["event_time"],
                weight=float(r["weight"]),
                independence_weight=float(r["independence_weight"]),
                applied_gain=float(r["applied_gain"]),
                outcome=r["outcome"],
                processed_at=r["processed_at"],
            )
            for r in event_rows
        ],
        shadow=(
            ClaimShadowView(
                would_status=shadow_row["would_status"],
                deciding_snapshot=_json_object(conn, shadow_row["deciding_snapshot"]),
                evidence_ids=_json_list(conn, shadow_row["evidence_ids"]),
                created_at=shadow_row["created_at"],
            )
            if shadow_row is not None
            else None
        ),
    )
