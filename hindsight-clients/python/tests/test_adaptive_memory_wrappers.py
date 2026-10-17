"""The wrapper's adaptive-memory methods send exactly what the caller named.

Confirm and the lifecycle PATCH both read an omitted field as "leave it alone" on the server, so a
wrapper that forwarded every argument (None included) would still work today; the guard here is
the other direction: a named argument must reach the request, and the retain item fields must
survive the wrapper's explicit item mapping.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture(monkeypatch, client, method: str, captured: dict):
    async def fake(bank_id, memory_id, *args, **kwargs):
        captured.update(bank_id=bank_id, memory_id=memory_id, args=args, kwargs=kwargs)
        return MagicMock()

    monkeypatch.setattr(client._memory_api, method, fake)


def test_confirm_memory_sends_the_named_fields_only(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture(monkeypatch, client, "confirm_memory", captured)
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    client.confirm_memory("bank", "mem-1", source_key="review-1", event_time=when)

    request = captured["kwargs"]["confirm_memory_request"]
    assert (captured["bank_id"], captured["memory_id"]) == ("bank", "mem-1")
    assert request.to_dict() == {"source_key": "review-1", "event_time": when}


async def test_aconfirm_memory_without_arguments_sends_an_empty_body(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture(monkeypatch, client, "confirm_memory", captured)

    await client.aconfirm_memory("bank", "mem-1", reason="checked")

    assert captured["kwargs"]["confirm_memory_request"].to_dict() == {"reason": "checked"}


def test_update_memory_lifecycle_sends_the_named_fields_only(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture(monkeypatch, client, "update_memory_lifecycle", captured)

    client.update_memory_lifecycle("bank", "mem-1", forgetting_exempt=False)

    [request] = captured["args"]
    assert request.to_dict() == {"forgetting_exempt": False}


async def test_aupdate_memory_lifecycle_forwards_importance_and_reason(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture(monkeypatch, client, "update_memory_lifecycle", captured)

    await client.aupdate_memory_lifecycle(
        "bank", "mem-1", importance=0.9, forgetting_exempt=True, exemption_reason="compliance"
    )

    [request] = captured["args"]
    assert request.to_dict() == {"importance": 0.9, "forgetting_exempt": True, "exemption_reason": "compliance"}


def _capture_retain(monkeypatch, client, captured):
    async def fake_retain(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(success=True)

    monkeypatch.setattr(client._memory_api, "retain_memories", fake_retain)


def test_retain_items_carry_importance_and_exemption(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain_batch(
        "bank",
        items=[{"content": "Pinned.", "importance": 0.8, "forgetting_exempt": True}, {"content": "Plain."}],
    )

    pinned, plain = captured["request"].items
    assert (pinned.importance, pinned.forgetting_exempt) == (0.8, True)
    assert (plain.importance, plain.forgetting_exempt) == (None, None)


async def test_single_retain_forwards_importance_and_exemption(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}
    _capture_retain(monkeypatch, client, captured)

    await client.aretain("bank", "Pinned.", importance=0.2, forgetting_exempt=False)

    item = captured["request"].items[0]
    assert (item.importance, item.forgetting_exempt) == (0.2, False)
