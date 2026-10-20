"""The read view of one memory's adaptive state: ``MemoriesExtension.get_memory_lifecycle``.

What ``GET /memories/{id}`` shows under ``lifecycle`` and ``PATCH …/lifecycle`` answers with:
retention (always present: the stored row, or the lazy defaults a memory without one is scored
with, plus retrievability at the caller's ``now``), the claim row, lifecycle relations in and out,
the most recent reinforcement events, and the observe-mode shadow record. Pydantic, so the HTTP
layer can dump it as JSON as it stands (``model_dump(mode="json")``).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

#: How many reinforcement events the view lists, newest first.
RECENT_EVENTS_LIMIT = 20


class RetentionView(BaseModel):
    """A memory's retention, and its retrievability at the view's ``now``."""

    materialized: bool = Field(description="True when a retention row exists; False when every value is a default.")
    retrievability: float
    stability_days: float
    importance: float
    last_reinforced_at: datetime | None
    reinforcement_count: int
    access_count: int
    last_recalled_at: datetime | None = None
    forgetting_exempt: bool
    exemption_reason: str | None = None
    retention_state: str = "active"
    below_threshold_since: datetime | None = None
    archived_at: datetime | None = None
    updated_at: datetime | None = None


class ClaimView(BaseModel):
    """The memory's claim row: what should be believed about it, and why."""

    truth_status: str
    confidence: float | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retired_via: str | None = None
    superseded_by_id: str | None = None
    inherited_from_id: str | None = None
    reason: str | None = None
    evidence_memory_ids: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class RelationView(BaseModel):
    """One lifecycle relation. A NULL endpoint is a tombstone: that memory is gone (its snapshot stays)."""

    id: str
    relation_type: str
    source_memory_id: str | None
    target_memory_id: str | None
    source_snapshot: dict[str, Any] = Field(default_factory=dict)
    target_snapshot: dict[str, Any] = Field(default_factory=dict)
    origin: str
    reason: str | None = None
    confidence: float | None = None
    created_at: datetime | None = None


class ReinforcementEventView(BaseModel):
    """One ledger row."""

    source_type: str
    source_key: str
    event_time: datetime
    weight: float
    independence_weight: float
    applied_gain: float
    outcome: str
    processed_at: datetime | None = None


class ClaimShadowView(BaseModel):
    """What observe mode would have retired this memory as (diagnostic only)."""

    would_status: str
    deciding_snapshot: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class MemoryLifecycle(BaseModel):
    """Everything the store holds about one memory's adaptive state.

    ``archived`` memories (curation-invalidated) are read from the snapshot taken when they were
    invalidated, which carries no shadow record. ``stored`` is False when nothing at all is stored
    for the memory, so ``retention`` is all defaults and every list is empty: the GET view leaves
    the key out for such a memory while the bank's mode is off.
    """

    memory_id: str
    archived: bool
    stored: bool
    retention: RetentionView
    claim: ClaimView | None = None
    relations_out: list[RelationView] = Field(default_factory=list)
    relations_in: list[RelationView] = Field(default_factory=list)
    recent_events: list[ReinforcementEventView] = Field(default_factory=list)
    shadow: ClaimShadowView | None = None
