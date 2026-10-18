"""The recall endpoint's context-admission fields: ``purpose`` and ``context``.

The engine side (gate, modes, trace) is covered by test_context_admission*.py through
``recall_async``. This file pins the HTTP seam: the request model stays in step with the
engine's purposes and context keys, unset fields stay out of a dumped request, bad values are a
422 before the engine runs, what the caller sends reaches ``recall_async`` unchanged, and the
response carries the admission summary only while the gate filters.
"""

import uuid

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import RecallRequest, RecallTaskContext
from hindsight_api.engine.context_admission import PURPOSES, TASK_CONTEXT_KEYS


def test_request_purposes_match_the_engine():
    annotation = RecallRequest.model_fields["purpose"].annotation
    literal = next(arg for arg in annotation.__args__ if arg is not type(None))
    assert tuple(literal.__args__) == PURPOSES


def test_request_context_carries_every_engine_context_key():
    assert set(TASK_CONTEXT_KEYS) <= set(RecallTaskContext.model_fields)


def test_unset_fields_stay_out_of_a_dumped_request():
    dumped = RecallRequest(query="how do we deploy").model_dump()
    assert "purpose" not in dumped and "context" not in dumped


def test_a_misspelled_context_key_is_refused():
    with pytest.raises(ValidationError, match="repo"):
        RecallRequest(query="q", context={"repo": "github.com/acme/payments"})


def test_a_non_string_context_value_is_refused():
    with pytest.raises(ValidationError):
        RecallRequest(query="q", context={"repository": 42})


@pytest.fixture
def captured_recall(memory, monkeypatch):
    captured: dict = {}
    original = memory.recall_async

    async def spy(*args, **kwargs):
        captured.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(memory, "recall_async", spy)
    return captured


async def _new_bank(api_client) -> str:
    bank_id = f"test-ctxadm-http-{uuid.uuid4().hex[:8]}"
    response = await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    assert response.status_code == 200, response.text
    return bank_id


async def test_purpose_and_context_reach_the_engine(api_client, captured_recall):
    bank_id = await _new_bank(api_client)
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={
            "query": "How is the payments API deployed?",
            "purpose": "auto_context",
            "context": {"repository": "github.com/acme/payments", "module": "deploy", "branch": None},
        },
    )
    assert response.status_code == 200, response.text
    assert captured_recall["purpose"] == "auto_context"
    assert captured_recall["task_context"] == {"repository": "github.com/acme/payments", "module": "deploy"}
    assert captured_recall["_external"] is True


async def test_omitted_fields_reach_the_engine_as_none(api_client, captured_recall):
    bank_id = await _new_bank(api_client)
    response = await api_client.post(f"/v1/default/banks/{bank_id}/memories/recall", json={"query": "anything"})
    assert response.status_code == 200, response.text
    assert captured_recall["purpose"] is None
    assert captured_recall["task_context"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"query": "q", "purpose": "injection"},
        {"query": "q", "context": {"repo": "github.com/acme/payments"}},
        {"query": "q", "context": "github.com/acme/payments"},
    ],
)
async def test_bad_values_are_a_422_before_the_engine_runs(api_client, captured_recall, body):
    bank_id = f"test-ctxadm-http-{uuid.uuid4().hex[:8]}"
    response = await api_client.post(f"/v1/default/banks/{bank_id}/memories/recall", json=body)
    assert response.status_code == 422, response.text
    assert captured_recall == {}


async def test_the_admission_summary_is_in_the_response_only_while_the_gate_filters(api_client):
    bank_id = await _new_bank(api_client)
    url = f"/v1/default/banks/{bank_id}/memories/recall"
    body = {"query": "How is the payments API deployed?", "purpose": "auto_context"}

    off = await api_client.post(url, json=body)
    assert off.status_code == 200, off.text
    assert "context_admission" not in off.json()

    for mode, present in (("observe", False), ("filter", True), ("strict", True)):
        patched = await api_client.patch(
            f"/v1/default/banks/{bank_id}/config", json={"updates": {"context_admission_mode": mode}}
        )
        assert patched.status_code == 200, patched.text
        response = await api_client.post(url, json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        if present:
            # An empty bank: nothing was judged, which is not an abstention.
            assert payload["context_admission"] == {"abstained": False, "candidates": 0, "admitted": 0}
        else:
            assert "context_admission" not in payload
