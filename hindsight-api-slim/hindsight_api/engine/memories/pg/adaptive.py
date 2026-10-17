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
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from ...adaptive.models import AdaptiveState, ClaimState, RetentionState, TruthStatus
from ...db.ops_oracle import ORACLE_IN_LIST_LIMIT

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
