"""Reinforcement evidence: who said it, and how independent that source is (DESIGN.md §4.3, R10).

Pure. A producer turns a memory's provenance into an :class:`EvidenceKey` and builds a
:class:`~.models.ReinforcementEvent` from it; the store's idempotency ledger
(``UNIQUE (memory_id, source_type, source_key)``) then guarantees the same evidence never
reinforces a memory twice, however often it is re-sent.

The key is the strongest provenance the memory carries, in this order:

1. ``metadata.source_event_id`` (``event:``) — one upstream event;
2. ``metadata.session_id`` (``session:``) — one conversation session;
3. a client-supplied ``document_id`` (``document:``);
4. ``metadata.conversation_id`` (``conversation:``);
5. ``metadata.tool_execution_id`` (``tool:``);
6. a server-minted ``document_id`` (``document:``, independence 0.5);
7. ``memory:<id>`` (independence 0.5) — no provenance at all.

A server-minted document id (retain without a ``document_id`` mints one per call) sits below the
conversation and tool ids rather than at the client document's rank: every retained fact has a
document id, so at rank 3 a minted one would shadow ``conversation_id`` / ``tool_execution_id``
for every caller that never names its documents. It still keys the call, so ten facts from one
anonymous retain call count once, but only at half weight, the same as the ``memory:`` fallback.
"""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ...config import DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS
from .models import ReinforcementEvent, ReinforcementSource
from .settings import AdaptiveSettings

#: Keys whose UTF-8 encoding is longer than this many bytes are stored as a sha256 digest (a client
#: ``source_key`` or an upstream id can be anything). Bytes, not characters: the ledger column is
#: ``VARCHAR(512)`` on PostgreSQL (characters) but ``VARCHAR2(512)`` with byte semantics on Oracle,
#: where 200 CJK characters are 600 bytes and the insert would fail with ORA-12899. The prefix
#: survives, so a hashed key still says what kind of evidence it is.
MAX_KEY_LENGTH = 200

#: Independence of evidence whose provenance is weak: a server-minted document id, or none.
WEAK_INDEPENDENCE = 0.5

#: Every prefix a key can carry. A prefix is kept verbatim when a long key is hashed.
KEY_PREFIXES = ("event:", "session:", "document:", "conversation:", "tool:", "memory:")


@dataclass(frozen=True)
class EvidenceKey:
    """The ledger key of one piece of evidence and how much it counts as independent."""

    key: str
    independence_weight: float


def _text(value: Any) -> str | None:
    """A metadata value as a key component: stripped, non-empty, never a bool or a container."""
    if value is None or isinstance(value, (bool, dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def bound_key(key: str) -> str:
    """``key`` itself when it fits in :data:`MAX_KEY_LENGTH` UTF-8 bytes, else its known prefix plus
    ``sha256:<hex digest of the rest>`` (ASCII, at most 84 bytes).

    Idempotent (a bounded key is short), so the store can apply it to every key it receives,
    whichever producer built it, and a client re-sending the same long key still deduplicates.
    """
    if len(key.encode("utf-8")) <= MAX_KEY_LENGTH:
        return key
    prefix = next((p for p in KEY_PREFIXES if key.startswith(p)), "")
    digest = hashlib.sha256(key[len(prefix) :].encode("utf-8")).hexdigest()
    return f"{prefix}sha256:{digest}"


def evidence_key(
    *,
    memory_id: str,
    document_id: str | None,
    document_id_minted: bool,
    metadata: Mapping[str, Any] | None,
) -> EvidenceKey:
    """The evidence key of a raw fact (see the module docstring for the order)."""
    meta = metadata or {}
    document = _text(document_id)
    for prefix, value in (
        ("event:", _text(meta.get("source_event_id"))),
        ("session:", _text(meta.get("session_id"))),
        ("document:", document if not document_id_minted else None),
        ("conversation:", _text(meta.get("conversation_id"))),
        ("tool:", _text(meta.get("tool_execution_id"))),
    ):
        if value is not None:
            return EvidenceKey(key=bound_key(prefix + value), independence_weight=1.0)
    if document is not None:
        return EvidenceKey(key=bound_key("document:" + document), independence_weight=WEAK_INDEPENDENCE)
    return EvidenceKey(key=bound_key(f"memory:{memory_id}"), independence_weight=WEAK_INDEPENDENCE)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def cooldown_bucket(event_time: datetime, cooldown_hours: float) -> str:
    """The cooldown window ``event_time`` falls in, as the window's start (ISO 8601, UTC).

    Recall and reflect key their events ``recall:<bucket>`` / ``reflect:<bucket>``, so every use
    of a memory inside one window is the same evidence and the ledger keeps one row for it. With
    a cooldown of 0 every instant is its own window: the key is ``event_time`` itself.
    """
    moment = _as_utc(event_time)
    if not math.isfinite(cooldown_hours) or cooldown_hours <= 0:
        return moment.isoformat()
    width = timedelta(hours=cooldown_hours)
    # timedelta // timedelta is integer floor division on microseconds: exact at a boundary,
    # where float division could round an instant into the previous window.
    start = _EPOCH + width * ((moment - _EPOCH) // width)
    return start.isoformat()


def source_weight(source_type: ReinforcementSource, settings: AdaptiveSettings) -> float:
    """The configured weight of a source type (``adaptive_memory_source_weights``)."""
    weight = settings.source_weights.get(source_type)
    if weight is None:
        weight = DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS.get(source_type, 0.0)
    return float(weight)


def reinforcement_event(
    *,
    memory_id: str,
    source_type: ReinforcementSource,
    source_key: str,
    event_time: datetime,
    settings: AdaptiveSettings,
    independence_weight: float = 1.0,
) -> ReinforcementEvent:
    """A :class:`~.models.ReinforcementEvent` with its source weight resolved and its key bounded."""
    return ReinforcementEvent(
        memory_id=str(memory_id),
        source_type=source_type,
        source_key=bound_key(source_key),
        event_time=_as_utc(event_time),
        weight=source_weight(source_type, settings),
        independence_weight=independence_weight,
    )
