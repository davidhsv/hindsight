"""The maintained wrapper threads purpose and context into the recall request.

Kept in step with the TypeScript wrapper's recall_context_admission_mapping test: a parameter only
one wrapper forwards is silently dropped for every consumer of the other.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_purpose_and_context(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall(
        "test-bank",
        "q",
        purpose="auto_context",
        context={"repository": "github.com/acme/payments", "branch": "main", "module": "auth"},
    )

    body = captured["request"].to_dict()
    assert body["purpose"] == "auto_context"
    assert body["context"] == {"repository": "github.com/acme/payments", "branch": "main", "module": "auth"}


async def test_arecall_leaves_both_out_of_the_body_by_default(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    await client.arecall("test-bank", "q")

    body = captured["request"].to_dict()
    assert "purpose" not in body and "context" not in body


async def test_arecall_rejects_an_unknown_context_key(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    with pytest.raises(ValueError, match="repo"):
        await client.arecall("test-bank", "q", context={"repo": "acme"})
    assert "request" not in captured


async def test_arecall_rejects_an_unknown_purpose(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    _capture_recall(monkeypatch, client, {})

    with pytest.raises(ValueError):
        await client.arecall("test-bank", "q", purpose="injection")  # type: ignore[arg-type]
