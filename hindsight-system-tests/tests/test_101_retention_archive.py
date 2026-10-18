"""A memory that faded is archived out of recall; the retain item's policy decides which one fades.

Adaptive memory mode ``rank``, on a server with archiving on and no grace period. Three runbook
facts were written two years ago and never reinforced since, so all three have long fallen below
the archive threshold. Their retain items set the retention policy: one is unimportant, one is
important enough to be protected, one is exempt from forgetting. The archive job runs after every
consolidation. What only this composition shows:

- the policy a client put on the retain item reaches the sweep: the same age archives the
  unimportant fact and spares the other two (the important one is seen below the threshold and
  kept; the exempt one never decays);
- an archived memory leaves recall but stays stored and readable, its state on GET memory;
- ``retention_state: "active"`` on the lifecycle PATCH brings it back into recall.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

WRITTEN = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
UNIMPORTANT = "The staging runbook restarts the cache before every deploy"
IMPORTANT = "The production runbook drains the queue before every deploy"
EXEMPT = "The release runbook tags the commit before every deploy"
QUERY = "What does the runbook do before every deploy?"
POLICIES: dict[str, dict[str, float | bool]] = {
    UNIMPORTANT: {"importance": 0.2},
    IMPORTANT: {"importance": 0.9},
    EXEMPT: {"forgetting_exempt": True},
}


@pytest.fixture
async def bank(archiving_client) -> AsyncIterator[str]:
    name = f"systest-{uuid.uuid4().hex[:12]}"
    yield name
    await archiving_client.banks.delete_bank(name)


@pytest.fixture
async def swept(archiving_client, llm, bank) -> dict[str, str]:
    """The bank after the archive job has run; maps each fact's text to its id."""
    await archiving_client.banks.update_bank_config(bank, {"updates": {"adaptive_memory_mode": "rank"}})
    for text in POLICIES:
        llm.on_step("extract_facts", contains=text).returns(extracted(fact(text)))
    llm.on_step("consolidate").returns(consolidation())

    for i, (text, policy) in enumerate(POLICIES.items()):
        await archiving_client.aretain(
            bank_id=bank, content=f"{text}.", timestamp=WRITTEN.isoformat(), document_id=f"runbook-{i}", **policy
        )
        await wait_until_settled(archiving_client, bank)

    operations = await archiving_client.operations.list_operations(bank, status="completed", limit=100)
    assert "adaptive_memory_maintenance" in {op.task_type for op in operations.operations}
    memories = await archiving_client.memory.list_memories(bank, limit=100)
    ids = {m.text: m.id for m in memories.items}
    assert set(ids) == set(POLICIES), sorted(ids)
    return ids


async def _recalled(client, bank: str) -> set[str]:
    response = await client.arecall(bank_id=bank, query=QUERY, types=["world", "experience"])
    return {r.text for r in response.results}


async def _retention(client, bank: str, memory_id: str) -> dict:
    return (await client.memory.get_memory(bank, memory_id))["lifecycle"]["retention"]


async def test_only_the_unimportant_fact_is_archived(archiving_client, bank, swept):
    states = {text: await _retention(archiving_client, bank, swept[text]) for text in POLICIES}
    assert {text: state["retention_state"] for text, state in states.items()} == {
        UNIMPORTANT: "archived",
        IMPORTANT: "active",
        EXEMPT: "active",
    }
    assert states[UNIMPORTANT]["archived_at"] is not None
    assert states[UNIMPORTANT]["importance"] == pytest.approx(0.2)
    # Protected, not unseen: the sweep saw it below the threshold and kept it for its importance.
    assert states[IMPORTANT]["below_threshold_since"] is not None
    assert states[IMPORTANT]["importance"] == pytest.approx(0.9)
    # Exempt memories never decay, so the sweep never finds them below the threshold.
    assert (states[EXEMPT]["forgetting_exempt"], states[EXEMPT]["retrievability"]) == (True, 1.0)
    assert states[EXEMPT]["below_threshold_since"] is None


async def test_the_archived_fact_leaves_recall_but_stays_stored(archiving_client, bank, swept):
    assert await _recalled(archiving_client, bank) == {IMPORTANT, EXEMPT}
    stored = await archiving_client.memory.get_memory(bank, swept[UNIMPORTANT])
    assert stored["text"] == UNIMPORTANT


async def test_returning_it_to_active_brings_it_back_into_recall(archiving_client, bank, swept):
    lifecycle = await archiving_client.aupdate_memory_lifecycle(bank, swept[UNIMPORTANT], retention_state="active")
    assert (lifecycle.retention.retention_state, lifecycle.retention.archived_at) == ("active", None)
    assert await _recalled(archiving_client, bank) == set(POLICIES)
