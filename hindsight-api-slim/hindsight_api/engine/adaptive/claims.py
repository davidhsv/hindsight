"""Claim-lifecycle value types shared by the memories store and its callers (DESIGN.md §5, §7).

The writers live on ``MemoriesExtension`` (``lock_claims``, ``cas_retire``, ``set_claim``, …);
these are the values they exchange that are not a single stored row: the retired-status set the
requeue invariant and the recall filters share, the re-ingest carry-forward handle, and its result.
Plain data, no SQL.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .models import TruthStatus

#: Statuses that take a memory out of the current view and out of consolidation (DESIGN.md §7: a
#: superseded or refuted raw fact is never requeued). ``uncertain`` is a live, contested claim.
RETIRED_STATUSES: tuple[TruthStatus, ...] = (TruthStatus.SUPERSEDED, TruthStatus.REFUTED)
#: The SQL literal list of :data:`RETIRED_STATUSES`, for ``truth_status IN (…)``.
RETIRED_STATUSES_SQL = ", ".join(f"'{s.value}'" for s in RETIRED_STATUSES)

#: The ``reason`` a claim reverted by the delete repair records (DESIGN.md §7).
REVERTED_EVIDENCE_DELETED = "reverted:evidence_deleted"


def reingest_text_key(text: str | None) -> str:
    """The carry-forward match key of a fact's text: sha256 of its whitespace-collapsed, casefolded form.

    A re-ingest re-extracts the document; a fact whose text comes back the same up to spacing and
    case is the same claim and keeps its lifecycle state (DESIGN.md §7).
    """
    normalized = " ".join((text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CarryForwardResult:
    """What one ``carry_forward_reingest`` call did.

    ``matched``: outgoing facts whose state moved to a re-extracted fact in this call.
    ``unmatched``: outgoing facts with state that are still unmatched after it (on the final call,
    the ones whose state was dropped). ``reverted``: claims the final call reverted because every
    piece of their evidence was among the unmatched facts.
    """

    matched: int = 0
    unmatched: int = 0
    reverted: int = 0


@dataclass(frozen=True)
class StashedRetention:
    """An outgoing fact's ``memory_retention_state`` row."""

    importance: float
    stability_days: float
    last_reinforced_at: datetime
    reinforcement_count: int
    access_count: int
    last_recalled_at: datetime | None
    forgetting_exempt: bool
    exemption_reason: str | None


@dataclass(frozen=True)
class StashedEvent:
    """One reinforcement ledger row of an outgoing fact (its idempotency key travels with it)."""

    source_type: str
    source_key: str
    event_time: datetime
    weight: float
    independence_weight: float
    applied_gain: float
    outcome: str
    processed_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class StashedClaim:
    """An outgoing fact's own claim row and the evidence it cites."""

    truth_status: TruthStatus
    confidence: float | None
    valid_from: datetime | None
    valid_to: datetime | None
    retired_via: str | None
    superseded_by_id: str | None
    inherited_from_id: str | None
    scope_fingerprint: str | None
    reason: str | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class StashedUnit:
    """One outgoing fact that carries state, and the key it is matched on."""

    unit_id: str
    fact_type: str
    text_key: str
    claim: StashedClaim | None
    retention: StashedRetention | None
    events: tuple[StashedEvent, ...]


@dataclass(frozen=True)
class InboundReference:
    """Another claim, or a relation row, that pointed at an outgoing fact before the re-ingest."""

    kind: Literal["evidence", "superseded_by", "inherited_from", "relation_source", "relation_target"]
    #: The citing claim's memory id, or the relation row's id.
    holder_id: str
    old_unit_id: str


@dataclass
class ReingestStash:
    """The lifecycle state of a document's outgoing facts, held across a re-ingest.

    Taken by ``stash_reingest_lifecycle`` before the old facts are deleted; consumed by
    ``carry_forward_reingest`` once the re-extracted facts exist (possibly over several batch
    transactions of one streaming retain). Inbound evidence rows and claim pointers are detached
    when the stash is taken, so the delete's own repair sees none of them; the final carry-forward
    call reverts what no re-extracted fact took over.
    """

    bank_id: str
    document_id: str
    units: dict[str, StashedUnit] = field(default_factory=dict)
    inbound: list[InboundReference] = field(default_factory=list)
    #: Superseded or refuted observation id -> the outgoing facts it cited. They are dropped from its
    #: sources when the stash is taken (the delete would otherwise leave it citing deleted ids) and the
    #: re-extracted facts that match them are added back, so the retired observation keeps its row,
    #: its text and its provenance across the re-ingest.
    retired_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: old unit id -> the re-extracted unit that took its state over.
    matched: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.units and not self.inbound and not self.retired_sources
