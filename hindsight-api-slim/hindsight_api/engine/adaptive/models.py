"""Adaptive-memory value types: modes, claim statuses, and the per-memory state rows.

The dataclasses mirror the store's ``memory_retention_state`` / ``memory_claim_state`` rows as
the engine sees them. They are frozen: a state change is a new value (``dataclasses.replace``),
computed by the pure functions in :mod:`.retention` and persisted by the store.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


class AdaptiveMode(str, Enum):
    """``adaptive_memory_mode``, the per-bank umbrella switch (DESIGN.md §2).

    Declared in rising order of effect; ``config.ADAPTIVE_MEMORY_MODES`` keeps the same tuple
    (it cannot import the engine) and a test holds the two equal.
    """

    OFF = "off"
    OBSERVE = "observe"
    RANK = "rank"
    LIFECYCLE = "lifecycle"

    @property
    def loads_state(self) -> bool:
        """Recall loads adaptive state and computes the factor (every mode but off)."""
        return self is not AdaptiveMode.OFF

    @property
    def applies_factor(self) -> bool:
        """The adaptive factor changes the recall ranking (observe only reports it)."""
        return self in (AdaptiveMode.RANK, AdaptiveMode.LIFECYCLE)

    @property
    def lifecycle(self) -> bool:
        """Claim lifecycle (labels, filtering, views) is on."""
        return self is AdaptiveMode.LIFECYCLE


class TruthStatus(str, Enum):
    """``memory_claim_state.truth_status``: what should be believed about a memory."""

    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    SUPERSEDED = "superseded"
    REFUTED = "refuted"


#: The reinforcement source types (``memory_reinforcement_events.source_type`` CHECK). They are
#: also the keys of ``adaptive_memory_source_weights``.
ReinforcementSource = Literal[
    "explicit_confirmation",
    "independent_retain_confirmation",
    "successful_reflect_use",
    "observation_new_evidence",
    "final_external_recall",
]

#: The one source that also counts as an access (``access_count`` / ``last_recalled_at``).
FINAL_EXTERNAL_RECALL: ReinforcementSource = "final_external_recall"

EventOutcomeKind = Literal["applied", "cooldown", "initial"]


@dataclass(frozen=True)
class RetentionState:
    """One materialized ``memory_retention_state`` row.

    A memory without a row is not "unknown": its retention is derived lazily from its own
    timestamps and fact type (:func:`.retention.effective_retention`). A row exists once
    something has to be remembered about the memory beyond those defaults — a reinforcement,
    an importance, an exemption.
    """

    memory_id: str
    importance: float
    stability_days: float
    last_reinforced_at: datetime
    reinforcement_count: int
    access_count: int
    forgetting_exempt: bool
    retention_state: str
    last_recalled_at: datetime | None
    below_threshold_since: datetime | None
    archived_at: datetime | None


@dataclass(frozen=True)
class ClaimState:
    """One ``memory_claim_state`` row: the memory's truth status and validity interval."""

    memory_id: str
    truth_status: TruthStatus
    confidence: float | None
    valid_from: datetime | None
    valid_to: datetime | None
    retired_via: str | None
    superseded_by_id: str | None
    inherited_from_id: str | None
    scope_fingerprint: str | None
    reason: str | None


@dataclass(frozen=True)
class AdaptiveState:
    """Everything the store holds about one memory's adaptive state. Either half may be absent."""

    retention: RetentionState | None
    claim: ClaimState | None


@dataclass(frozen=True)
class ReinforcementEvent:
    """One piece of evidence that a memory was useful (DESIGN.md §4.3).

    ``weight`` is the source weight (``adaptive_memory_source_weights[source_type]``), resolved
    by the producer; ``independence_weight`` discounts evidence whose provenance is weak (a
    server-minted document id, the ``memory:<id>`` fallback key).
    """

    memory_id: str
    source_type: ReinforcementSource
    source_key: str
    event_time: datetime
    weight: float
    independence_weight: float


@dataclass(frozen=True)
class EventOutcome:
    """What :func:`.retention.apply_events` did with one event."""

    event: ReinforcementEvent
    outcome: EventOutcomeKind
    applied_gain: float
