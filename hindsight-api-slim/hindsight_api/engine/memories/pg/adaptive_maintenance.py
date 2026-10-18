"""Adaptive-memory maintenance on the SQL store: grounding checks, the archive sweep, prune candidates.

- :func:`grounding_status` answers the mental-model retraction check in lifecycle mode: whether
  each id a document cites is absent, live, or a live memory whose claim was retired.
- :func:`archive_sweep` is the per-bank archive pass (``adaptive_memory_maintenance`` operation).
  It reads the bank's ACTIVE materialized retention rows in keyset pages over
  ``(bank_id, last_reinforced_at)`` and computes retrievability in Python with the same function
  recall scores with, so the sweep and the ranking can never disagree about R. Each page that has
  something to change locks just those rows (``FOR UPDATE SKIP LOCKED``: a row a reinforcement
  writer holds is left for the next sweep, never waited on), re-decides from the locked values and
  writes in one transaction of its own. A memory still on its lazy defaults has no row and is
  never archived: archiving needs the row to remember when R fell below the threshold.
- :func:`prune_candidates` lists the archived rows old enough to delete, in keyset pages; the engine
  deletes them through ``delete_memory_units`` (never an open-coded DELETE), which re-checks them
  under row locks with :func:`lock_prunable` inside its own transaction.

Statements are in the shared SQL dialect (``= ANY($n::uuid[])`` becomes an IN list on Oracle,
``LIMIT`` becomes ``FETCH FIRST``); id lists are chunked at the Oracle IN-list limit. Every uuid
column read keeps an ``*_id`` (or ``id``) name, which is how the Oracle driver decodes RAW(16).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ...adaptive.models import (
    GROUNDING_ABSENT,
    GROUNDING_LIVE,
    AdaptiveState,
    ArchiveResult,
    Grounding,
    PruneCursor,
    PrunePage,
    RetentionState,
    TruthStatus,
)
from ...adaptive.retention import effective_retention
from ...adaptive.settings import AdaptiveSettings
from ...db.ops_oracle import ORACLE_IN_LIST_LIMIT
from .adaptive_claims import _observations_citing, claims_for

#: The claim statuses that retire a memory from current answers (``uncertain`` is current).
_RETIRED = frozenset({TruthStatus.SUPERSEDED.value, TruthStatus.REFUTED.value})

#: The sweep never reads more than this many rows per page: the lock and update statements put
#: the page's ids in one IN list, which Oracle caps at 1000.
MAX_SWEEP_BATCH = ORACLE_IN_LIST_LIMIT

_RETENTION_COLUMNS = (
    "memory_id, importance, stability_days, last_reinforced_at, reinforcement_count, access_count, "
    "forgetting_exempt, retention_state, last_recalled_at, below_threshold_since, archived_at"
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _opt_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _as_utc(value)


def _uuid_list(values: Sequence[Any]) -> list[uuid.UUID]:
    """Distinct parseable ids in first-seen order; unparseable ones are dropped."""
    out: dict[uuid.UUID, None] = {}
    for raw in values:
        try:
            out[raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))] = None
        except (ValueError, AttributeError, TypeError):
            continue
    return list(out)


def _pieces(ids: Sequence[Any]) -> list[list[Any]]:
    items = list(ids)
    return [items[i : i + ORACLE_IN_LIST_LIMIT] for i in range(0, len(items), ORACLE_IN_LIST_LIMIT)]


# --------------------------------------------------------------------------- grounding

_GROUNDING_SQL = """
SELECT mu.id, c.truth_status, c.superseded_by_id
FROM {units} mu
LEFT JOIN {claims} c ON c.memory_id = mu.id
WHERE mu.bank_id = $1 AND mu.id = ANY($2::uuid[])
"""


async def grounding_status(
    conn, *, fq_table: Callable[[str], str], bank_id: str, ids: Sequence[Any]
) -> dict[str, Grounding]:
    """``{id: Grounding}`` for every requested id; one statement per 1000 ids."""
    wanted = list(dict.fromkeys(str(i) for i in ids))
    out: dict[str, Grounding] = dict.fromkeys(wanted, GROUNDING_ABSENT)
    parsed = _uuid_list(wanted)
    if not parsed:
        return out
    sql = _GROUNDING_SQL.format(units=fq_table("memory_units"), claims=fq_table("memory_claim_state"))
    # Keyed back by the canonical form, so an id cited in upper case still finds its row.
    canonical = {str(u): raw for raw in wanted for u in _uuid_list([raw])}
    for piece in _pieces(parsed):
        for row in await conn.fetch(sql, bank_id, piece):
            key = canonical.get(str(row["id"]), str(row["id"]))
            status = row["truth_status"]
            if status in _RETIRED:
                successor = row["superseded_by_id"]
                out[key] = Grounding(
                    state="retired",
                    truth_status=TruthStatus(status),
                    successor_id=None if successor is None else str(successor),
                )
            else:
                out[key] = GROUNDING_LIVE
    return out


# --------------------------------------------------------------------------- archive sweep


def _state_from_row(row: Any) -> RetentionState:
    return RetentionState(
        memory_id=str(row["memory_id"]),
        importance=float(row["importance"]),
        stability_days=float(row["stability_days"]),
        last_reinforced_at=_as_utc(row["last_reinforced_at"]),
        reinforcement_count=int(row["reinforcement_count"]),
        access_count=int(row["access_count"]),
        # BOOLEAN on Postgres, NUMBER(1) on Oracle.
        forgetting_exempt=bool(row["forgetting_exempt"]),
        retention_state=str(row["retention_state"]),
        last_recalled_at=_opt_utc(row["last_recalled_at"]),
        below_threshold_since=_opt_utc(row["below_threshold_since"]),
        archived_at=_opt_utc(row["archived_at"]),
    )


@dataclass(frozen=True)
class _Decision:
    """What one active row needs: a mark set, a mark cleared, or archiving (protected or not)."""

    mark: bool = False
    clear: bool = False
    archive: bool = False
    protected: bool = False


def _decide(state: RetentionState, *, settings: AdaptiveSettings, now: datetime) -> _Decision:
    """The sweep's rule for one row, from its values alone (claim evidence is checked separately).

    ``below_threshold_since`` records when a sweep first SAW R below the threshold, not when R
    crossed it: the grace period counts observed time, so changing the threshold or the bank's
    stability bounds never archives anything retroactively.
    """
    r = effective_retention(
        AdaptiveState(retention=state, claim=None),
        fact_type="",
        mentioned_at=None,
        occurred_start=None,
        event_date=None,
        settings=settings,
        now=now,
    ).retrievability
    if r >= settings.archive_threshold:
        return _Decision(clear=state.below_threshold_since is not None)
    since = state.below_threshold_since
    mark = since is None
    since = since if since is not None else now
    if now - since < timedelta(days=settings.archive_grace_days):
        return _Decision(mark=mark)
    protected = state.forgetting_exempt or state.importance >= settings.archive_protect_importance
    return _Decision(mark=mark, archive=not protected, protected=protected)


_PAGE_FIRST_SQL = """
SELECT {columns} FROM {retention}
WHERE bank_id = $1 AND retention_state = 'active'
ORDER BY last_reinforced_at, memory_id
LIMIT {limit}
"""

_PAGE_NEXT_SQL = """
SELECT {columns} FROM {retention}
WHERE bank_id = $1 AND retention_state = 'active'
  AND (last_reinforced_at > $2 OR (last_reinforced_at = $2 AND memory_id > $3))
ORDER BY last_reinforced_at, memory_id
LIMIT {limit}
"""

_LOCK_SQL = """
SELECT {columns} FROM {retention}
WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND retention_state = 'active'
FOR UPDATE SKIP LOCKED
"""

_EVIDENCE_OF_RETIRED_SQL = """
SELECT DISTINCT e.evidence_memory_id
FROM {evidence} e
JOIN {claims} c ON c.memory_id = e.memory_id
WHERE e.bank_id = $1 AND e.evidence_memory_id = ANY($2::uuid[]) AND c.truth_status <> 'active'
"""


async def _evidence_of_non_active_claims(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[uuid.UUID]
) -> set[str]:
    """The ids among ``memory_ids`` that a claim which is not ``active`` cites as evidence."""
    if not memory_ids:
        return set()
    sql = _EVIDENCE_OF_RETIRED_SQL.format(
        evidence=fq_table("memory_claim_evidence"), claims=fq_table("memory_claim_state")
    )
    found: set[str] = set()
    for piece in _pieces(memory_ids):
        found.update(str(row["evidence_memory_id"]) for row in await conn.fetch(sql, bank_id, piece))
    return found


async def _set_column(
    conn, *, table: str, bank_id: str, ids: Sequence[uuid.UUID], assignment: str, now: datetime
) -> None:
    if ids:
        await conn.execute(
            f"UPDATE {table} SET {assignment}, updated_at = $3 WHERE bank_id = $1 AND memory_id = ANY($2::uuid[])",
            bank_id,
            list(ids),
            now,
        )


async def archive_sweep(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    now: datetime,
    settings: AdaptiveSettings,
    batch: int = 500,
) -> ArchiveResult:
    """See ``MemoriesExtension.archive_sweep``. ``conn`` must not be inside a transaction the
    caller still needs open: every page commits on its own."""
    now = _as_utc(now)
    limit = max(1, min(int(batch), MAX_SWEEP_BATCH))
    retention = fq_table("memory_retention_state")
    first = _PAGE_FIRST_SQL.format(columns=_RETENTION_COLUMNS, retention=retention, limit=limit)
    following = _PAGE_NEXT_SQL.format(columns=_RETENTION_COLUMNS, retention=retention, limit=limit)
    lock_sql = _LOCK_SQL.format(columns=_RETENTION_COLUMNS, retention=retention)

    scanned = marked = cleared = archived = protected = skipped = 0
    cursor: tuple[datetime, uuid.UUID] | None = None
    while True:
        rows = (
            await conn.fetch(first, bank_id)
            if cursor is None
            else await conn.fetch(following, bank_id, cursor[0], cursor[1])
        )
        if not rows:
            break
        scanned += len(rows)
        last = rows[-1]
        cursor = (last["last_reinforced_at"], uuid.UUID(str(last["memory_id"])))

        wanted: list[uuid.UUID] = []
        for state in (_state_from_row(row) for row in rows):
            decision = _decide(state, settings=settings, now=now)
            if decision.mark or decision.clear or decision.archive:
                wanted.append(uuid.UUID(state.memory_id))
            elif decision.protected:
                protected += 1
        if wanted:
            async with conn.transaction():
                locked = {
                    s.memory_id: s
                    for s in (_state_from_row(row) for row in await conn.fetch(lock_sql, bank_id, wanted))
                }
                skipped += len(wanted) - len(locked)
                decisions = {mid: _decide(s, settings=settings, now=now) for mid, s in locked.items()}
                candidates = [uuid.UUID(mid) for mid, d in decisions.items() if d.archive]
                cited = await _evidence_of_non_active_claims(
                    conn, fq_table=fq_table, bank_id=bank_id, memory_ids=candidates
                )
                to_mark = sorted(uuid.UUID(m) for m, d in decisions.items() if d.mark)
                to_clear = sorted(uuid.UUID(m) for m, d in decisions.items() if d.clear)
                to_archive = sorted(u for u in candidates if str(u) not in cited)
                protected += sum(1 for d in decisions.values() if d.protected) + len(candidates) - len(to_archive)
                await _set_column(
                    conn,
                    table=retention,
                    bank_id=bank_id,
                    ids=to_mark,
                    assignment="below_threshold_since = $3",
                    now=now,
                )
                await _set_column(
                    conn,
                    table=retention,
                    bank_id=bank_id,
                    ids=to_clear,
                    assignment="below_threshold_since = NULL",
                    now=now,
                )
                await _set_column(
                    conn,
                    table=retention,
                    bank_id=bank_id,
                    ids=to_archive,
                    assignment="retention_state = 'archived', archived_at = $3",
                    now=now,
                )
                marked += len(to_mark)
                cleared += len(to_clear)
                archived += len(to_archive)
        if len(rows) < limit:
            break
    return ArchiveResult(
        scanned=scanned,
        marked=marked,
        cleared=cleared,
        archived=archived,
        protected=protected,
        skipped_locked=skipped,
    )


# --------------------------------------------------------------------------- prune

#: The archived rows prune may delete, before the Python checks. ``forgetting_exempt`` is filtered in
#: Python, not as ``= FALSE``: the column is BOOLEAN on Postgres and NUMBER(1) on Oracle, and no
#: boolean literal compares with both. A memory another claim names as its successor or as the
#: observation it inherited from is kept: deleting it would revert those claims to active (the
#: delete repair), bringing a retired statement back as current.
_PRUNABLE_WHERE = """r.bank_id = $1 AND r.retention_state = 'archived' AND r.archived_at <= $2
  AND NOT EXISTS (SELECT 1 FROM {evidence} e WHERE e.evidence_memory_id = r.memory_id)
  AND NOT EXISTS (
      SELECT 1 FROM {claims} c WHERE c.superseded_by_id = r.memory_id OR c.inherited_from_id = r.memory_id
  )"""

#: Keyset pages over ``(archived_at, memory_id)``: a page whose rows are all protected still moves
#: the cursor on, so protected rows at the head of the archive never hide the deletable ones behind.
_PRUNE_SQL = """
SELECT r.memory_id, r.forgetting_exempt, r.archived_at
FROM {retention} r
WHERE {where}{after}
ORDER BY r.archived_at, r.memory_id
LIMIT {limit}
"""
_PRUNE_AFTER = "\n  AND (r.archived_at > $3 OR (r.archived_at = $3 AND r.memory_id > $4))"

#: The same predicate, re-read under row locks inside the delete transaction. ``SKIP LOCKED``: a row
#: a writer holds (a hold being placed, a reinforcement unarchiving it) is left for the next run.
_LOCK_PRUNABLE_SQL = """
SELECT r.memory_id, r.forgetting_exempt
FROM {retention} r
WHERE {where} AND r.memory_id = ANY($3::uuid[])
FOR UPDATE SKIP LOCKED
"""


def _prune_cutoff(now: datetime, settings: AdaptiveSettings) -> datetime:
    return _as_utc(now) - timedelta(days=settings.prune_after_days)


def _prunable_where(fq_table: Callable[[str], str]) -> str:
    return _PRUNABLE_WHERE.format(evidence=fq_table("memory_claim_evidence"), claims=fq_table("memory_claim_state"))


async def _without_current_observation_sources(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: list[str]
) -> list[str]:
    """``memory_ids`` minus the raw facts that are a source of an observation whose claim is current
    (active, uncertain, or no claim row): deleting one would sweep that observation away."""
    if not memory_ids:
        return []
    citing = await _observations_citing(conn, fq_table=fq_table, bank_id=bank_id, unit_ids=memory_ids)
    observations = sorted({o for obs in citing.values() for o in obs})
    claims = await claims_for(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=observations)

    def _backs_current_observation(memory_id: str) -> bool:
        for observation in citing.get(memory_id, ()):
            claim = claims.get(observation)
            if claim is None or claim.truth_status in (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN):
                return True
        return False

    return [m for m in memory_ids if not _backs_current_observation(m)]


async def prune_candidates(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    now: datetime,
    settings: AdaptiveSettings,
    limit: int = 500,
    after: PruneCursor | None = None,
) -> PrunePage:
    """See ``MemoriesExtension.prune_candidates``. Beyond the interface's exclusions, a successor or
    an inherited-from observation of another claim is kept (``_PRUNABLE_WHERE``)."""
    limit = max(1, min(int(limit), MAX_SWEEP_BATCH))
    sql = _PRUNE_SQL.format(
        retention=fq_table("memory_retention_state"),
        where=_prunable_where(fq_table),
        after="" if after is None else _PRUNE_AFTER,
        limit=limit,
    )
    args: list[Any] = [bank_id, _prune_cutoff(now, settings)]
    if after is not None:
        args += [_as_utc(after.archived_at), uuid.UUID(after.memory_id)]
    rows = await conn.fetch(sql, *args)
    next_after = (
        PruneCursor(archived_at=_as_utc(rows[-1]["archived_at"]), memory_id=str(rows[-1]["memory_id"]))
        if len(rows) == limit
        else None
    )
    candidates = [str(row["memory_id"]) for row in rows if not bool(row["forgetting_exempt"])]
    memory_ids = await _without_current_observation_sources(
        conn, fq_table=fq_table, bank_id=bank_id, memory_ids=candidates
    )
    return PrunePage(memory_ids=memory_ids, next_after=next_after)


async def lock_prunable(
    conn,
    *,
    fq_table: Callable[[str], str],
    bank_id: str,
    memory_ids: Sequence[Any],
    now: datetime,
    settings: AdaptiveSettings,
) -> list[str]:
    """See ``MemoriesExtension.lock_prunable``. Runs inside the caller's (delete) transaction."""
    ids = _uuid_list(memory_ids)
    if not ids:
        return []
    sql = _LOCK_PRUNABLE_SQL.format(retention=fq_table("memory_retention_state"), where=_prunable_where(fq_table))
    cutoff = _prune_cutoff(now, settings)
    locked: list[str] = []
    for piece in _pieces(ids):
        rows = await conn.fetch(sql, bank_id, cutoff, piece)
        locked.extend(str(row["memory_id"]) for row in rows if not bool(row["forgetting_exempt"]))
    return sorted(
        await _without_current_observation_sources(conn, fq_table=fq_table, bank_id=bank_id, memory_ids=locked)
    )


# --------------------------------------------------------------------------- unarchive


async def unarchive_memories(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any], now: datetime
) -> list[str]:
    """See ``MemoriesExtension.unarchive_memories``."""
    ids = _uuid_list(memory_ids)
    if not ids:
        return []
    retention = fq_table("memory_retention_state")
    found: list[str] = []
    for piece in _pieces(ids):
        rows = await conn.fetch(
            f"SELECT memory_id FROM {retention} "
            f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND retention_state = 'archived'",
            bank_id,
            piece,
        )
        archived = [uuid.UUID(str(row["memory_id"])) for row in rows]
        if archived:
            await conn.execute(
                f"UPDATE {retention} SET retention_state = 'active', archived_at = NULL, "
                f"below_threshold_since = NULL, updated_at = $3 "
                f"WHERE bank_id = $1 AND memory_id = ANY($2::uuid[]) AND retention_state = 'archived'",
                bank_id,
                archived,
                _as_utc(now),
            )
            found.extend(str(u) for u in archived)
    return sorted(found)


async def clear_below_threshold_marks(
    conn, *, fq_table: Callable[[str], str], bank_id: str, memory_ids: Sequence[Any], now: datetime
) -> None:
    """Clear ``below_threshold_since`` on ``memory_ids``: an applied reinforcement is new use, so
    the grace period restarts from the next sweep that finds the memory below the threshold again.
    """
    retention = fq_table("memory_retention_state")
    for piece in _pieces(_uuid_list(memory_ids)):
        await _set_column(
            conn,
            table=retention,
            bank_id=bank_id,
            ids=piece,
            assignment="below_threshold_since = NULL",
            now=_as_utc(now),
        )


async def banks_with_retention_state(conn, *, fq_table: Callable[[str], str]) -> list[str]:
    """The banks of the current schema that have at least one retention row, sorted.

    One statement; per bank it is an existence probe on the ``(bank_id, ...)`` retention indexes,
    so the cost follows the number of banks, not the number of rows.
    """
    rows = await conn.fetch(
        f"SELECT b.bank_id FROM {fq_table('banks')} b "
        f"WHERE EXISTS (SELECT 1 FROM {fq_table('memory_retention_state')} r WHERE r.bank_id = b.bank_id) "
        f"ORDER BY b.bank_id"
    )
    return [str(row["bank_id"]) for row in rows]
