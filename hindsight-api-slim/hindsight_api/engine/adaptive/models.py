"""Adaptive-memory value types: modes, claim statuses, and the per-memory state rows.

The dataclasses mirror the store's ``memory_retention_state`` / ``memory_claim_state`` rows as
the engine sees them. They are frozen: a state change is a new value (``dataclasses.replace``),
computed by the pure functions in :mod:`.retention` and persisted by the store.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
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


#: What the store did with one event handed to ``MemoriesExtension.record_reinforcement``: the
#: three :data:`EventOutcomeKind` values of an event that was decided, plus the three ways an event
#: is never decided — its key is already in the ledger (``deduplicated``), its memory is not live in
#: the bank or is being deleted (``skipped_missing``), or reinforcement is off for it
#: (``disabled``: mode not rank/lifecycle, the bank's reinforcement switch, or the static recall or
#: reflect switch).
RecordedOutcomeKind = Literal["applied", "cooldown", "initial", "deduplicated", "skipped_missing", "disabled"]


@dataclass(frozen=True)
class RecordedEvent:
    """One input event's fate. ``source_key`` is the key as stored (long keys are hashed)."""

    memory_id: str
    source_type: ReinforcementSource
    source_key: str
    outcome: RecordedOutcomeKind
    applied_gain: float
    persisted: bool
    """True when the event now has a ledger row (recall/reflect cooldowns never get one)."""


@dataclass(frozen=True)
class ReinforcementResult:
    """What one ``record_reinforcement`` call did. The default is the no-op result.

    The counts are per event, except ``initialized`` (retention rows this call created).
    ``events`` follows the input order. ``states`` holds the retention row of every memory the
    call locked, after the call (unchanged when all its events were deduplicated or cooled down),
    keyed by memory id.
    """

    applied: int = 0
    cooldown: int = 0
    initial: int = 0
    deduplicated: int = 0
    initialized: int = 0
    skipped_missing: int = 0
    disabled: int = 0
    events: tuple[RecordedEvent, ...] = ()
    states: Mapping[str, RetentionState] = field(default_factory=dict)


@dataclass(frozen=True)
class RetentionPolicy:
    """An explicit retention setting for one memory; ``None`` leaves that field as it is.

    ``importance`` in [0, 1] rescales the memory's stability by the ratio of the importance
    multipliers ``(0.5 + new) / (0.5 + old)``. Turning ``forgetting_exempt`` off also clears
    ``exemption_reason``.
    """

    memory_id: str
    importance: float | None = None
    forgetting_exempt: bool | None = None
    exemption_reason: str | None = None


#: What a cited memory id is now, for a mental model's grounding check: ``absent`` (no live row in
#: the bank — deleted, invalidated, swept, or never this bank's), ``live`` (a row whose claim is
#: current: no claim row, ``active`` or ``uncertain``), ``retired`` (a live row whose claim is
#: ``superseded`` or ``refuted``).
GroundingState = Literal["absent", "live", "retired"]


@dataclass(frozen=True)
class Grounding:
    """One cited id's grounding (``MemoriesExtension.grounding_status``).

    ``truth_status`` and ``successor_id`` are set only when ``state`` is ``retired``: the claim's
    status and the memory that replaced it (``superseded_by_id``, None for a refutation that named
    no correction or a successor that is no longer live).
    """

    state: GroundingState
    truth_status: TruthStatus | None = None
    successor_id: str | None = None


GROUNDING_ABSENT = Grounding(state="absent")
GROUNDING_LIVE = Grounding(state="live")


@dataclass(frozen=True)
class ArchiveResult:
    """What one ``MemoriesExtension.archive_sweep`` call did; the default is the no-op result.

    ``scanned`` counts the active retention rows read; ``marked`` the rows whose
    ``below_threshold_since`` was set (R fell below the threshold), ``cleared`` those whose mark was
    removed (R is back at or above it), ``archived`` the rows moved to ``retention_state='archived'``,
    and ``protected`` the rows that met the retrievability and grace conditions but were kept
    (importance at or above the protection level, an exemption or hold, or evidence of a claim that
    is not active). ``skipped_locked`` counts rows another transaction held while the sweep wanted
    to change them; the next sweep sees them again.
    """

    scanned: int = 0
    marked: int = 0
    cleared: int = 0
    archived: int = 0
    protected: int = 0
    skipped_locked: int = 0


@dataclass(frozen=True)
class PruneCursor:
    """Where a ``MemoriesExtension.prune_candidates`` page ended: its last row's archive key."""

    archived_at: datetime
    memory_id: str


@dataclass(frozen=True)
class PrunePage:
    """One page of ``MemoriesExtension.prune_candidates``; the default is the empty, final page.

    ``memory_ids`` are the page's deletable memories (possibly none: every row on the page may be
    protected). ``next_after`` is the cursor for the next page, None once the bank's old enough
    archived rows are exhausted.
    """

    memory_ids: list[str] = field(default_factory=list)
    next_after: PruneCursor | None = None


#: ``exemption_reason`` of a hold (``PATCH .../lifecycle`` with ``hold: true``): an exemption from
#: forgetting that also keeps the memory from being archived or pruned, like any exemption.
HOLD_EXEMPTION_REASON = "hold"
