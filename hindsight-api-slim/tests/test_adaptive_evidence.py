"""Reinforcement evidence keys, cooldown buckets and source weights (engine/adaptive/evidence.py).

Pure: no database. The store's use of these keys (the idempotency ledger) is covered by
tests/test_adaptive_reinforcement_store.py.
"""

import dataclasses
import hashlib
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hindsight_api.engine.adaptive.evidence import (
    MAX_KEY_LENGTH,
    WEAK_INDEPENDENCE,
    EvidenceKey,
    bound_key,
    cooldown_bucket,
    evidence_key,
    reinforcement_event,
    source_weight,
)
from hindsight_api.engine.adaptive.retention import initial_stability, rescale_for_importance
from hindsight_api.engine.adaptive.settings import AdaptiveSettings

_STATIC_FIELDS = (
    "adaptive_memory_recall_reinforcement_enabled",
    "adaptive_memory_reflect_reinforcement_enabled",
    "adaptive_memory_reinforcement_cooldown_hours",
    "adaptive_memory_reinforcement_gain",
    "adaptive_memory_source_weights",
    "adaptive_memory_type_stability",
    "adaptive_memory_source_trust",
    "adaptive_memory_conflict_policy",
    "adaptive_memory_archive_enabled",
    "adaptive_memory_archive_threshold",
    "adaptive_memory_archive_grace_days",
    "adaptive_memory_archive_recall_min_score",
    "adaptive_memory_archive_protect_importance",
    "adaptive_memory_prune_enabled",
    "adaptive_memory_prune_after_days",
)


def _settings(**static) -> AdaptiveSettings:
    values = {name: None for name in _STATIC_FIELDS}
    values.update({f"adaptive_memory_{k}": v for k, v in static.items()})
    return AdaptiveSettings.resolve({"adaptive_memory_mode": "rank"}, SimpleNamespace(**values), store_supports=True)


MEMORY = "6f1c2a3e-0000-4000-8000-000000000001"
FULL = {
    "source_event_id": "evt-1",
    "session_id": "sess-1",
    "conversation_id": "conv-1",
    "tool_execution_id": "tool-1",
}


# --------------------------------------------------------------------------- evidence_key


@pytest.mark.parametrize(
    ("drop", "expected"),
    [
        ((), "event:evt-1"),
        (("source_event_id",), "session:sess-1"),
        (("source_event_id", "session_id"), "document:doc-1"),
    ],
)
def test_the_strongest_provenance_wins(drop, expected):
    metadata = {k: v for k, v in FULL.items() if k not in drop}
    key = evidence_key(memory_id=MEMORY, document_id="doc-1", document_id_minted=False, metadata=metadata)
    assert key == EvidenceKey(key=expected, independence_weight=1.0)


def test_conversation_then_tool_ids_when_there_is_no_client_document():
    metadata = {"conversation_id": "conv-1", "tool_execution_id": "tool-1"}
    assert evidence_key(memory_id=MEMORY, document_id=None, document_id_minted=False, metadata=metadata) == (
        EvidenceKey(key="conversation:conv-1", independence_weight=1.0)
    )
    assert evidence_key(
        memory_id=MEMORY, document_id=None, document_id_minted=False, metadata={"tool_execution_id": "t"}
    ) == EvidenceKey(key="tool:t", independence_weight=1.0)


def test_a_minted_document_keys_the_call_at_half_weight_below_conversation_and_tool_ids():
    """Every retained fact has a document id; a minted one must not shadow the weaker metadata ids."""
    minted = evidence_key(memory_id=MEMORY, document_id="uuid-minted", document_id_minted=True, metadata={})
    assert minted == EvidenceKey(key="document:uuid-minted", independence_weight=WEAK_INDEPENDENCE)
    with_conversation = evidence_key(
        memory_id=MEMORY, document_id="uuid-minted", document_id_minted=True, metadata={"conversation_id": "c"}
    )
    assert with_conversation == EvidenceKey(key="conversation:c", independence_weight=1.0)


def test_no_provenance_falls_back_to_the_memory_at_half_weight():
    key = evidence_key(memory_id=MEMORY, document_id=None, document_id_minted=False, metadata=None)
    assert key == EvidenceKey(key=f"memory:{MEMORY}", independence_weight=WEAK_INDEPENDENCE)


@pytest.mark.parametrize("blank", ["", "   ", None, True, ["x"], {"a": 1}])
def test_blank_or_structured_metadata_values_are_not_provenance(blank):
    key = evidence_key(
        memory_id=MEMORY, document_id="doc-1", document_id_minted=False, metadata={"source_event_id": blank}
    )
    assert key.key == "document:doc-1"


def test_metadata_values_are_stripped_and_numbers_kept():
    assert evidence_key(
        memory_id=MEMORY, document_id=None, document_id_minted=False, metadata={"session_id": "  s1 "}
    ).key == ("session:s1")
    assert evidence_key(
        memory_id=MEMORY, document_id=None, document_id_minted=False, metadata={"session_id": 42}
    ).key == ("session:42")


def test_long_keys_are_hashed_keeping_their_prefix():
    long_id = "x" * 500
    key = evidence_key(memory_id=MEMORY, document_id=long_id, document_id_minted=False, metadata={})
    assert key.key == "document:sha256:" + hashlib.sha256(long_id.encode()).hexdigest()
    assert len(key.key) <= MAX_KEY_LENGTH


def test_bound_key_is_identity_up_to_the_limit_and_idempotent_past_it():
    exact = "session:" + "a" * (MAX_KEY_LENGTH - len("session:"))
    assert bound_key(exact) == exact
    over = exact + "b"
    hashed = bound_key(over)
    assert hashed.startswith("session:sha256:") and hashed != over
    assert bound_key(hashed) == hashed
    # A key with no known prefix is hashed whole.
    assert bound_key("z" * 300) == "sha256:" + hashlib.sha256(("z" * 300).encode()).hexdigest()


def test_the_bound_counts_utf8_bytes_so_a_multibyte_key_fits_a_byte_sized_column():
    # 192 three-byte characters: a 200-character key of 584 bytes, past Oracle's VARCHAR2(512 BYTE).
    session = "あ" * 192
    key = evidence_key(memory_id=MEMORY, document_id=None, document_id_minted=False, metadata={"session_id": session})
    assert key.key == "session:sha256:" + hashlib.sha256(session.encode("utf-8")).hexdigest()
    assert len(key.key.encode("utf-8")) <= MAX_KEY_LENGTH
    assert bound_key(key.key) == key.key
    # At the limit in bytes a multibyte key is kept verbatim; one byte over, it is hashed.
    exact = "session:" + "é" * ((MAX_KEY_LENGTH - len("session:")) // 2)
    assert len(exact.encode("utf-8")) == MAX_KEY_LENGTH and bound_key(exact) == exact
    assert bound_key(exact + "a").startswith("session:sha256:")


# --------------------------------------------------------------------------- cooldown_bucket


def test_cooldown_bucket_is_the_window_start_in_utc():
    t = datetime(2026, 3, 1, 13, 59, 59, tzinfo=UTC)
    assert cooldown_bucket(t, 6) == "2026-03-01T12:00:00+00:00"
    # The same instant in another zone is the same window.
    assert cooldown_bucket(t.astimezone(timezone(timedelta(hours=-5))), 6) == "2026-03-01T12:00:00+00:00"
    # A naive time is UTC.
    assert cooldown_bucket(t.replace(tzinfo=None), 6) == "2026-03-01T12:00:00+00:00"


def test_cooldown_bucket_boundaries_are_exact():
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert cooldown_bucket(start, 6) == start.isoformat()
    assert cooldown_bucket(start - timedelta(microseconds=1), 6) == "2026-03-01T12:00:00+00:00"
    # Fractional hours work too (30-minute windows).
    assert cooldown_bucket(datetime(2026, 3, 1, 18, 44, tzinfo=UTC), 0.5) == "2026-03-01T18:30:00+00:00"


def test_zero_cooldown_keys_by_the_instant_itself():
    t = datetime(2026, 3, 1, 13, 59, 59, 123456, tzinfo=UTC)
    assert cooldown_bucket(t, 0) == t.isoformat()


# --------------------------------------------------------------------------- weights and events


def test_source_weights_follow_settings_over_the_defaults():
    assert source_weight("explicit_confirmation", _settings()) == 1.0
    assert source_weight("final_external_recall", _settings()) == 0.2
    tuned = _settings(source_weights={"final_external_recall": 0.05})
    assert source_weight("final_external_recall", tuned) == 0.05
    assert source_weight("successful_reflect_use", tuned) == 0.6


def test_reinforcement_event_resolves_weight_bounds_key_and_normalizes_time():
    event = reinforcement_event(
        memory_id=MEMORY,
        source_type="independent_retain_confirmation",
        source_key="session:" + "s" * 400,
        event_time=datetime(2026, 3, 1, 8, 0),
        settings=_settings(),
        independence_weight=0.5,
    )
    assert event.weight == 0.8
    assert event.independence_weight == 0.5
    assert event.source_key.startswith("session:sha256:")
    assert event.event_time == datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def test_the_reflect_switch_resolves_from_the_static_config():
    assert _settings().reflect_reinforcement_enabled is True
    assert _settings(reflect_reinforcement_enabled=False).reflect_reinforcement_enabled is False
    # A non-bool stored value reads as the default.
    assert _settings(reflect_reinforcement_enabled="no").reflect_reinforcement_enabled is True


# --------------------------------------------------------------------------- importance rescale


def test_importance_rescale_is_the_initial_stability_ratio():
    settings = _settings()
    s0 = initial_stability("world", 0.5, settings)
    assert rescale_for_importance(s0, 0.5, 0.9, settings) == pytest.approx(initial_stability("world", 0.9, settings))
    # Learned stability is kept: only the importance factor changes.
    assert rescale_for_importance(300.0, 0.5, 1.0, settings) == pytest.approx(450.0)
    assert rescale_for_importance(300.0, 1.0, 0.5, settings) == pytest.approx(200.0)


def test_importance_rescale_is_clamped_to_the_bank_bounds():
    settings = dataclasses.replace(_settings(), max_stability_days=100.0)
    assert rescale_for_importance(90.0, 0.0, 1.0, settings) == 100.0
    assert rescale_for_importance(0.6, 1.0, 0.0, settings) == 0.5
    assert rescale_for_importance(float("nan"), 0.5, 0.5, settings) == 100.0
