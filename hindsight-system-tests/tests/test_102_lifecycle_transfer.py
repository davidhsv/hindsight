"""A cloned bank keeps what adaptive memory learned, re-keyed onto its own memories.

Adaptive memory mode ``lifecycle``. The billing service ran Java 17 (day 0); on day 30 it was
upgraded to Java 21 and consolidation superseded the old observation, whose raw fact inherited
the status. The upgrade fact was retained as important, and the new observation was confirmed by
hand on day 40. Then the bank is cloned. The replay gives every memory a fresh id, so none of the
rows adaptive memory keeps (claims, evidence, relations, retention) can travel verbatim. What only
this composition shows:

- the clone answers like the source: current recall without the retired runtime, history with
  it, each retired memory pointing at the CLONE's successor, never at the source's;
- the claim evidence and the supersession relation are rebuilt between the clone's own memories;
- retention travels as state: the confirmed observation keeps its learned stability and count,
  the important fact its importance, while the reinforcement events themselves stay behind.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    supersedes,
)

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
JAVA_17 = "The billing service runs on Java 17"
JAVA_21 = "The billing service was upgraded to Java 21"
OBSERVATION_17 = "The billing service runs Java 17"
OBSERVATION_21 = "The billing service runs Java 21"
QUERY = "Which Java version does the billing service run?"
TYPES = ["world", "experience", "observation"]


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _consolidate(request) -> Consolidation:
    """Day 0 creates the Java 17 observation; day 30's fact supersedes it with a labelled update."""
    if batch_fact_id(request, "Java 21") is not None:
        return supersedes(OBSERVATION_21, replacing="Java 17", citing="Java 21")(request)
    return Consolidation(creates=[Observation(text=OBSERVATION_17, source_fact_ids=batch_fact_ids(request))])


async def _ids(client, bank: str) -> dict[str, str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return {m.text: m.id for m in memories.items}


@pytest.fixture
async def clone_id(client) -> AsyncIterator[str]:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    yield bank
    await client.banks.delete_bank(bank)


@pytest.fixture
async def source(client, llm, bank_id, settled) -> dict[str, str]:
    """The source bank after the upgrade and the confirmation; maps each memory's text to its id."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Java 17").returns(extracted(fact(JAVA_17)))
    llm.on_step("extract_facts", contains="Java 21").returns(extracted(fact(JAVA_21)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.aretain(bank_id=bank_id, content=f"{JAVA_17}.", timestamp=day(0).isoformat(), document_id="inventory")
    await settled(bank_id)
    await client.aretain(
        bank_id=bank_id,
        content=f"{JAVA_21}.",
        timestamp=day(30).isoformat(),
        document_id="upgrade",
        importance=0.9,
    )
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_17, JAVA_21, OBSERVATION_17, OBSERVATION_21}, sorted(ids)
    confirmed = await client.aconfirm_memory(
        bank_id, ids[OBSERVATION_21], source_key="runtime-audit", event_time=day(40)
    )
    assert (confirmed.recorded, confirmed.outcome) == (True, "applied")
    return ids


@pytest.fixture
async def clone(client, source, bank_id, clone_id, settled) -> dict[str, str]:
    """The clone of the source bank; maps each memory's text to its id there."""
    operation_id = await client.aclone_bank(bank_id, clone_id)
    deadline = asyncio.get_running_loop().time() + 60
    while (status := await client.operations.get_operation_status(bank_id, operation_id)).status not in (
        "completed",
        "failed",
        "cancelled",
    ):
        assert asyncio.get_running_loop().time() < deadline, f"clone did not finish: {status}"
        await asyncio.sleep(0.1)
    assert status.status == "completed", status
    await settled(clone_id)

    ids = await _ids(client, clone_id)
    assert set(ids) == set(source), sorted(ids)
    assert set(ids.values()).isdisjoint(source.values()), "the replay gives every memory a fresh id"
    return ids


async def _recall(client, bank: str, **kwargs) -> dict[str, dict | None]:
    """``{text: lifecycle annotation or None}`` of what a recall returned."""
    response = await client.arecall(bank_id=bank, query=QUERY, types=TYPES, **kwargs)
    return {r.text: (r.lifecycle.to_dict() if r.lifecycle else None) for r in response.results}


async def _lifecycle(client, bank: str, memory_id: str) -> dict:
    return (await client.memory.get_memory(bank, memory_id))["lifecycle"]


async def test_the_clone_answers_with_the_new_runtime_and_keeps_the_history(client, clone_id, clone):
    assert await _recall(client, clone_id) == {OBSERVATION_21: None, JAVA_21: None}
    retired = {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": None,
        "valid_to": day(30).isoformat(),
        "superseded_by": clone[OBSERVATION_21],
    }
    history = await _recall(client, clone_id, lifecycle_view="historical")
    assert history == {OBSERVATION_21: None, JAVA_21: None, OBSERVATION_17: retired, JAVA_17: retired}


async def test_claims_evidence_and_relations_point_at_the_clones_own_memories(client, clone_id, clone):
    old = await _lifecycle(client, clone_id, clone[OBSERVATION_17])
    claim = old["claim"]
    assert (claim["truth_status"], claim["retired_via"], claim["superseded_by_id"]) == (
        "superseded",
        "supersedes",
        clone[OBSERVATION_21],
    )
    assert claim["evidence_memory_ids"] == [clone[JAVA_21]]
    [relation] = old["relations_in"]
    assert (relation["relation_type"], relation["source_memory_id"], relation["target_memory_id"]) == (
        "supersedes",
        clone[OBSERVATION_21],
        clone[OBSERVATION_17],
    )
    raw = (await _lifecycle(client, clone_id, clone[JAVA_17]))["claim"]
    assert (raw["truth_status"], raw["retired_via"], raw["inherited_from_id"]) == (
        "superseded",
        "inherited",
        clone[OBSERVATION_17],
    )


async def test_retention_travels_as_state_and_the_events_stay_behind(client, bank_id, source, clone_id, clone):
    for text in (OBSERVATION_21, JAVA_21):
        original = await _lifecycle(client, bank_id, source[text])
        copied = await _lifecycle(client, clone_id, clone[text])
        keys = ("stability_days", "importance", "reinforcement_count", "last_reinforced_at", "forgetting_exempt")
        assert {k: copied["retention"][k] for k in keys} == {k: original["retention"][k] for k in keys}, text
        assert copied["recent_events"] == [], text
    confirmed = (await _lifecycle(client, clone_id, clone[OBSERVATION_21]))["retention"]
    assert confirmed["reinforcement_count"] == 1
    assert (await _lifecycle(client, clone_id, clone[JAVA_21]))["retention"]["importance"] == pytest.approx(0.9)
