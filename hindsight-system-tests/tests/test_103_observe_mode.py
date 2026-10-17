"""Observe mode records what lifecycle mode would have done, and changes nothing a caller sees.

Two banks get the same inputs, one in mode ``off`` and one in ``observe``. The billing service
ran Java 17 (day 0); on day 30 a new session says it was upgraded to Java 21, and consolidation
answers the way it always has, without lifecycle labels (observe keeps the base prompt): a new
observation, and a delete of the one it replaced. What only this composition shows:

- the observe bank answers exactly like the off bank: the same results in the same order, the
  stale raw fact included, and no lifecycle annotation on any of them;
- the delete leaves a diagnostic on the raw fact the deleted observation stood on: GET memory
  shows that lifecycle mode would have superseded it, decided by that delete, on the evidence of
  the day-30 batch, while its claim stays unwritten.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationDelete,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    shown_observations,
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
    """Day 0 creates the Java 17 observation; day 30 creates its replacement and deletes it."""
    upgrade = batch_fact_id(request, "Java 21")
    if upgrade is None:
        return Consolidation(creates=[Observation(text=OBSERVATION_17, source_fact_ids=batch_fact_ids(request))])
    stale = shown_observations(request)[OBSERVATION_17]
    return Consolidation(
        creates=[Observation(text=OBSERVATION_21, source_fact_ids=[upgrade])],
        deletes=[ObservationDelete(observation_id=stale, reason="The billing service was upgraded")],
    )


async def _ids(client, bank: str) -> dict[str, str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return {m.text: m.id for m in memories.items}


@pytest.fixture
async def off_bank(client) -> AsyncIterator[str]:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    yield bank
    await client.banks.delete_bank(bank)


@pytest.fixture
async def banks(client, llm, bank_id, off_bank, settled) -> dict[str, dict[str, str]]:
    """``{"observe" | "off": {text: id}}`` after the upgrade, plus the deleted observation's id."""
    llm.on_step("extract_facts", contains="Java 17").returns(extracted(fact(JAVA_17)))
    llm.on_step("extract_facts", contains="Java 21").returns(extracted(fact(JAVA_21)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "observe"}})
    await client.acreate_bank(bank_id=off_bank, name="off")
    out: dict[str, dict[str, str]] = {}
    for mode, bank in (("observe", bank_id), ("off", off_bank)):
        await client.aretain(bank_id=bank, content=f"{JAVA_17}.", timestamp=day(0).isoformat(), document_id="inventory")
        await settled(bank)
        replaced = (await _ids(client, bank))[OBSERVATION_17]
        await client.aretain(bank_id=bank, content=f"{JAVA_21}.", timestamp=day(30).isoformat(), document_id="upgrade")
        await settled(bank)
        ids = await _ids(client, bank)
        assert set(ids) == {JAVA_17, JAVA_21, OBSERVATION_21}, (mode, sorted(ids))
        out[mode] = {**ids, OBSERVATION_17: replaced}
    return out


async def _recall(client, bank: str) -> list[tuple[str, object]]:
    response = await client.arecall(bank_id=bank, query=QUERY, types=TYPES)
    return [(r.text, r.lifecycle) for r in response.results]


async def test_the_observe_bank_answers_exactly_like_the_off_bank(client, bank_id, off_bank, banks):
    observed = await _recall(client, bank_id)
    assert observed == await _recall(client, off_bank)
    assert {text for text, _ in observed} == {OBSERVATION_21, JAVA_21, JAVA_17}
    assert {annotation for _, annotation in observed} == {None}


async def test_the_delete_leaves_a_shadow_on_the_raw_fact_and_nothing_else(client, bank_id, off_bank, banks):
    observe = banks["observe"]
    stale = (await client.memory.get_memory(bank_id, observe[JAVA_17]))["lifecycle"]
    assert stale["claim"] is None, "observe never writes a claim"
    shadow = stale["shadow"]
    assert (shadow["would_status"], shadow["evidence_ids"]) == ("superseded", [observe[JAVA_21]])
    snapshot = shadow["deciding_snapshot"]
    assert {k: snapshot[k] for k in ("decided_by", "observation_id", "observation_text", "retired_via")} == {
        "decided_by": "consolidation_delete",
        "observation_id": observe[OBSERVATION_17],
        "observation_text": OBSERVATION_17,
        "retired_via": "inherited",
    }
    current = (await client.memory.get_memory(bank_id, observe[JAVA_21]))["lifecycle"]
    assert (current["claim"], current["shadow"]) == (None, None)

    untouched = await client.memory.get_memory(off_bank, banks["off"][JAVA_17])
    assert "lifecycle" not in untouched, "an off bank's memory carries exactly the payload it always had"
