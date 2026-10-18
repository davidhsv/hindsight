"""The maintained wrapper threads lifecycle_view and as_of into the recall request, and the truth
fields into the lifecycle PATCH.

Kept in step with the TypeScript wrapper's recall_lifecycle_view_mapping test: a parameter only one
wrapper forwards is silently dropped for every consumer of the other.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_lifecycle_view_and_as_of(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)
    when = datetime(2026, 1, 15, tzinfo=timezone.utc)

    client.recall("test-bank", "q", lifecycle_view="historical", as_of=when)

    request = captured["request"]
    assert (request.lifecycle_view, request.as_of) == ("historical", when)
    assert request.to_dict()["lifecycle_view"] == "historical"


async def test_arecall_leaves_both_out_by_default(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    await client.arecall("test-bank", "q")

    request = captured["request"]
    assert (request.lifecycle_view, request.as_of) == (None, None)
    # Sent as null, like every optional recall field the wrapper forwards: the server's default.
    assert request.to_dict().get("lifecycle_view") is None and request.to_dict().get("as_of") is None


def test_update_memory_lifecycle_sends_the_truth_fields(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict = {}

    async def fake(bank_id, memory_id, *args, **kwargs):
        captured.update(args=args)
        return MagicMock()

    monkeypatch.setattr(client._memory_api, "update_memory_lifecycle", fake)

    client.update_memory_lifecycle("bank", "mem-1", truth_status="superseded", superseded_by="mem-2", reason="upgraded")

    [request] = captured["args"]
    assert request.to_dict() == {"truth_status": "superseded", "superseded_by": "mem-2", "reason": "upgraded"}
