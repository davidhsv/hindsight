"""Reinforcement producers: which operations emit evidence that a memory was useful.

Four producers hand events to ``MemoriesExtension.record_reinforcement``:

- **consolidation** — an observation gaining a source fact with an evidence key it has not seen
  (:class:`ObservationEvidence`). Written inside the batch's apply transaction, in a savepoint of
  its own, so a failure is logged and counted and never fails the batch.
- **reflect** — the memories and observations a successful answer cited
  (``successful_reflect_use``), off the response path (:class:`BackgroundReinforcement`).
- **recall** — the final results of an HTTP or MCP recall (``final_external_recall``), off the
  response path, and only while the static recall switch is on (default off: being selected is
  not evidence of being useful). Internal recalls never pass the flag that enables it.
- **explicit confirm** — the HTTP surface's own request-path write.

Nothing is written unless the bank's effective mode is rank or lifecycle and its reinforcement
switch is on. The store enforces that for every caller; the producers check it first anyway, so a
bank with the feature off pays one config lookup and nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ..background_writes import PendingWrites
from . import clock
from .evidence import EvidenceKey, cooldown_bucket, evidence_key, reinforcement_event
from .models import FINAL_EXTERNAL_RECALL, ReinforcementEvent, ReinforcementSource
from .retention import default_anchor
from .settings import AdaptiveSettings

logger = logging.getLogger(__name__)

#: The per-bank (hierarchical) fields :meth:`AdaptiveSettings.resolve` reads from a bank config.
_BANK_FIELDS = (
    "adaptive_memory_mode",
    "adaptive_memory_score_alpha",
    "adaptive_memory_score_floor",
    "adaptive_memory_reinforcement_enabled",
    "adaptive_memory_base_stability_days",
    "adaptive_memory_max_stability_days",
    "adaptive_memory_default_view",
    "adaptive_memory_historical_cues",
)

#: Background reinforcement writes (recall, reflect) one engine keeps in flight at most. A write
#: past it is dropped and counted rather than queued: a reinforcement is evidence, not data, and a
#: backlog behind a slow database must neither grow without bound nor take the connections recall
#: and reflect themselves need.
MAX_IN_FLIGHT_WRITES = 32

#: How long engine shutdown waits for in-flight background reinforcement writes.
DRAIN_TIMEOUT_SECONDS = 5.0

UsageSource = Literal["final_external_recall", "successful_reflect_use"]


# --------------------------------------------------------------------------- gates


def reinforcement_allowed(settings: AdaptiveSettings) -> bool:
    """The bank writes reinforcement at all: rank/lifecycle mode and its reinforcement switch on."""
    return settings.effective_mode.applies_factor and settings.reinforcement_enabled


def recall_reinforcement_allowed(settings: AdaptiveSettings) -> bool:
    """Whether a final external recall may reinforce what it returned.

    The one place recall reinforcement is decided, so the context-admission track can force it off
    here: when admission filters a recall, its final results are the admitted set, and reinforcing
    them would turn admission decisions into retention: a popularity loop where what is shown stays because it was shown.
    """
    return reinforcement_allowed(settings) and settings.recall_reinforcement_enabled


def reflect_reinforcement_allowed(settings: AdaptiveSettings) -> bool:
    """Whether a successful reflect may reinforce the memories its answer cited."""
    return reinforcement_allowed(settings) and settings.reflect_reinforcement_enabled


def write_settings_from_bank_config(bank_config: Mapping[str, Any], *, bank_id: str) -> AdaptiveSettings | None:
    """The bank's adaptive settings when its effective mode writes state (rank/lifecycle), else None.

    ``bank_config`` is a resolved bank config dict (``get_bank_config``). The mode is read first,
    so the default (off) costs one lookup and nothing is resolved.
    """
    mode = bank_config.get("adaptive_memory_mode")
    if not isinstance(mode, str) or mode.strip().lower() in ("", "off"):
        return None
    from ...config import get_config
    from ..memories import get_memories

    settings = AdaptiveSettings.resolve(
        bank_config, get_config(), store_supports=get_memories().supports_adaptive_state_for(bank_id)
    )
    return settings if settings.effective_mode.applies_factor else None


def mode_settings(config: Any, *, bank_id: str) -> AdaptiveSettings | None:
    """The bank's adaptive settings whenever its effective mode is not off (observe included), else None.

    For a caller that branches on the mode itself — consolidation runs the lifecycle variant in
    lifecycle mode and records observe-mode shadow decisions. Same ``getattr`` reading as
    :func:`write_settings`, so an off bank costs one attribute lookup.
    """
    mode = getattr(config, "adaptive_memory_mode", None)
    if not isinstance(mode, str) or mode.strip().lower() in ("", "off"):
        return None
    from ...config import get_config
    from ..memories import get_memories

    settings = AdaptiveSettings.resolve(
        {name: getattr(config, name, None) for name in _BANK_FIELDS},
        get_config(),
        store_supports=get_memories().supports_adaptive_state_for(bank_id),
    )
    return settings if settings.effective_mode.loads_state else None


def write_settings(config: Any, *, bank_id: str) -> AdaptiveSettings | None:
    """:func:`write_settings_from_bank_config` for a caller holding a resolved config object.

    Retain and consolidation hold a resolved ``HindsightConfig`` rather than the dict. Read with
    ``getattr`` so a config stand-in without the adaptive fields reads as off.
    """
    if getattr(config, "adaptive_memory_mode", None) in (None, "off"):
        return None
    return write_settings_from_bank_config(
        {name: getattr(config, name, None) for name in _BANK_FIELDS}, bank_id=bank_id
    )


# --------------------------------------------------------------------------- event builders


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def as_datetime(value: datetime | str | None) -> datetime | None:
    """A timestamp from a datetime or an ISO string (recall results carry strings); None if unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    try:
        return _as_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def usage_events(
    memory_ids: Iterable[Any],
    *,
    source_type: UsageSource,
    event_time: datetime,
    settings: AdaptiveSettings,
) -> list[ReinforcementEvent]:
    """One recall or reflect event per distinct memory, all keyed by the use's cooldown window.

    ``recall:<window>`` / ``reflect:<window>``: every use of a memory inside one cooldown window is
    the same evidence, so the ledger keeps (at most) one row for it however often it is used.
    """
    prefix = "recall" if source_type == FINAL_EXTERNAL_RECALL else "reflect"
    key = f"{prefix}:{cooldown_bucket(event_time, settings.reinforcement_cooldown_hours)}"
    return [
        reinforcement_event(
            memory_id=memory_id, source_type=source_type, source_key=key, event_time=event_time, settings=settings
        )
        for memory_id in dict.fromkeys(str(m) for m in memory_ids if m)
    ]


@dataclass(frozen=True)
class FactEvidence:
    """A raw fact as reinforcement evidence: its evidence key and when it was stated."""

    key: EvidenceKey
    event_time: datetime


def fact_evidence(
    *,
    memory_id: str,
    document_id: str | None,
    document_id_minted: bool,
    metadata: Mapping[str, Any] | None,
    mentioned_at: datetime | str | None,
    occurred_start: datetime | str | None,
    event_date: datetime | str | None,
    now: datetime,
) -> FactEvidence:
    """A fact's evidence: keyed by its provenance, timed ``mentioned_at ?? occurred_start ?? event_date ?? now``."""
    stated = default_anchor(as_datetime(mentioned_at), as_datetime(occurred_start), as_datetime(event_date))
    return FactEvidence(
        key=evidence_key(
            memory_id=str(memory_id),
            document_id=document_id,
            document_id_minted=document_id_minted,
            metadata=metadata if isinstance(metadata, Mapping) else None,
        ),
        event_time=stated if stated is not None else _as_utc(now),
    )


@dataclass(frozen=True)
class ReinforcementCall:
    """The arguments of one ``record_reinforcement`` call a producer assembled."""

    events: tuple[ReinforcementEvent, ...]
    anchors: Mapping[str, datetime]
    initial_memory_ids: frozenset[str]


class ObservationEvidence:
    """What one consolidation batch did to its observations, as reinforcement events.

    An observation is reinforced by a source fact only when the fact's evidence key is new for it:
    not the key of any source it already had, nor of one attached earlier in the batch. Ten facts
    from one session therefore count once, and a fact re-stating an observation counts only when
    it comes from somewhere the observation has not heard it from.

    - :meth:`created`: a new observation, one ``initial`` event per distinct key of its creating
      facts, anchored on the earliest of them.
    - :meth:`attached`: an UPDATE. ``independent_retain_confirmation`` when the text is unchanged
      (the observation was re-stated, triage ``restates`` included), else ``observation_new_evidence``.
      An observation without a retention row is anchored on its pre-update mention time, never on
      the column the update just advanced.
    - :meth:`folded`: sources moved onto a survivor by a dedup fold or a ``merged_into`` delete,
      ``observation_new_evidence``.

    ``facts`` holds the evidence of every fact involved: this batch's and the survivors' existing
    sources. A fact with no entry cannot be keyed and is skipped. ``prior_sources`` /
    ``prior_mentioned_at`` are each survivor's sources and mention time before the batch wrote.
    """

    def __init__(
        self,
        settings: AdaptiveSettings,
        *,
        facts: Mapping[str, FactEvidence],
        prior_sources: Mapping[str, Collection[str]] | None = None,
        prior_mentioned_at: Mapping[str, datetime | None] | None = None,
    ) -> None:
        self.settings = settings
        self._facts = facts
        self._prior_sources = {k: frozenset(str(s) for s in v) for k, v in (prior_sources or {}).items()}
        self._prior_mentioned_at = dict(prior_mentioned_at or {})
        self._known: dict[str, set[str]] = {}
        self._events: list[ReinforcementEvent] = []
        self._anchors: dict[str, datetime] = {}
        self._initial: set[str] = set()

    def _known_keys(self, observation_id: str) -> set[str]:
        known = self._known.get(observation_id)
        if known is None:
            prior = self._prior_sources.get(observation_id, frozenset())
            known = {self._facts[s].key.key for s in prior if s in self._facts}
            self._known[observation_id] = known
        return known

    def _new_evidence(self, observation_id: str, source_ids: Iterable[Any]) -> list[FactEvidence]:
        """Facts among ``source_ids`` whose key is new for the observation: one per key, oldest first."""
        prior = self._prior_sources.get(observation_id, frozenset())
        candidates = sorted(
            (
                self._facts[s]
                for s in dict.fromkeys(str(sid) for sid in source_ids)
                if s in self._facts and s not in prior
            ),
            key=lambda f: f.event_time,
        )
        known = self._known_keys(observation_id)
        fresh: list[FactEvidence] = []
        for fact in candidates:
            if fact.key.key in known:
                continue
            known.add(fact.key.key)
            fresh.append(fact)
        return fresh

    def _emit(self, observation_id: str, source_type: ReinforcementSource, fact: FactEvidence) -> None:
        self._events.append(
            reinforcement_event(
                memory_id=observation_id,
                source_type=source_type,
                source_key=fact.key.key,
                event_time=fact.event_time,
                settings=self.settings,
                independence_weight=fact.key.independence_weight,
            )
        )

    def created(self, observation_id: str, source_ids: Iterable[Any]) -> None:
        observation_id = str(observation_id)
        evidence = self._new_evidence(observation_id, source_ids)
        if not evidence:
            return
        self._initial.add(observation_id)
        self._anchors[observation_id] = evidence[0].event_time
        for fact in evidence:
            self._emit(observation_id, "observation_new_evidence", fact)

    def attached(
        self,
        observation_id: str,
        source_ids: Iterable[Any],
        *,
        unchanged_text: bool,
        previous_mentioned_at: datetime | str | None,
    ) -> None:
        observation_id = str(observation_id)
        source: ReinforcementSource = (
            "independent_retain_confirmation" if unchanged_text else "observation_new_evidence"
        )
        anchor = as_datetime(previous_mentioned_at) or as_datetime(self._prior_mentioned_at.get(observation_id))
        self._attach(observation_id, source_ids, source, anchor)

    def folded(self, survivor_id: str, source_ids: Iterable[Any]) -> None:
        survivor_id = str(survivor_id)
        anchor = as_datetime(self._prior_mentioned_at.get(survivor_id))
        self._attach(survivor_id, source_ids, "observation_new_evidence", anchor)

    def _attach(
        self, observation_id: str, source_ids: Iterable[Any], source: ReinforcementSource, anchor: datetime | None
    ) -> None:
        evidence = self._new_evidence(observation_id, source_ids)
        if not evidence:
            return
        if anchor is not None:
            # Only seeds a row the store creates; a memory that already has one keeps its own.
            self._anchors.setdefault(observation_id, anchor)
        for fact in evidence:
            self._emit(observation_id, source, fact)

    def call(self) -> ReinforcementCall | None:
        """The ``record_reinforcement`` arguments, or None when the batch produced no evidence."""
        if not self._events:
            return None
        return ReinforcementCall(
            events=tuple(self._events),
            anchors=dict(self._anchors),
            initial_memory_ids=frozenset(self._initial),
        )


# --------------------------------------------------------------------------- background writes


class BackgroundReinforcement:
    """Recall and reflect reinforcement, written off the response path on the primary backend.

    Fire-and-forget through :class:`PendingWrites` (held strongly, drained at shutdown), bounded to
    ``max_in_flight`` concurrent writes. A submission past the bound is dropped and counted, never
    queued. A failed write is logged and counted; it never reaches the operation that produced it.
    """

    def __init__(
        self, backend_getter: Callable[[], Awaitable[Any]], *, max_in_flight: int = MAX_IN_FLIGHT_WRITES
    ) -> None:
        self._backend_getter = backend_getter
        self._max_in_flight = max_in_flight
        self._in_flight = 0
        self._writes = PendingWrites("adaptive reinforcement write")

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def submit(
        self,
        *,
        producer: str,
        bank_id: str,
        events: Sequence[ReinforcementEvent],
        settings: AdaptiveSettings,
    ) -> bool:
        """Schedule one ``record_reinforcement`` write; False when there was nothing to write or no room."""
        from ...metrics import get_metrics_collector

        if not events:
            return False
        if self._in_flight >= self._max_in_flight:
            logger.debug(
                "[ADAPTIVE] %s reinforcement for bank %s dropped: %d writes already in flight",
                producer,
                bank_id,
                self._in_flight,
            )
            get_metrics_collector().record_adaptive_reinforcement_dropped(
                producer=producer, reason="overflow", bank_id=bank_id
            )
            return False
        # Counted when scheduled, not when the task starts, so a burst of submissions in one tick
        # sees the bound exactly.
        self._in_flight += 1
        if not self._writes.schedule(self._write(producer, bank_id, tuple(events), settings)):
            # The coroutine was closed unstarted: its own ``finally`` never runs.
            self._in_flight -= 1
            return False
        return True

    async def _write(
        self, producer: str, bank_id: str, events: tuple[ReinforcementEvent, ...], settings: AdaptiveSettings
    ) -> None:
        from ...metrics import get_metrics_collector
        from ..db_utils import acquire_with_retry
        from ..memories import get_memories

        try:
            backend = await self._backend_getter()
            async with acquire_with_retry(backend, max_retries=1) as conn:
                await get_memories().record_reinforcement(
                    conn,
                    bank_id=bank_id,
                    events=events,
                    settings=settings,
                    now=clock.utcnow(),
                    background=True,
                )
        except Exception as exc:
            logger.warning("[ADAPTIVE] %s reinforcement for bank %s failed: %r", producer, bank_id, exc)
            get_metrics_collector().record_adaptive_reinforcement_dropped(
                producer=producer, reason="error", bank_id=bank_id
            )
        finally:
            self._in_flight -= 1

    async def drain(self) -> None:
        """Await every write already scheduled (tests; a caller about to read the state)."""
        await self._writes.drain()

    async def drain_all(self, timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
        """Await in-flight writes, bounded so a stuck database cannot hang shutdown."""
        await self._writes.drain_all(timeout)
