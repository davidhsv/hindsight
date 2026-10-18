"""In a shared-scope bank a supersession still retires the stale raw fact, though the fact is tagged
and the observation is not.

Adaptive memory mode ``lifecycle`` with ``observation_scopes="shared"``, the coding-agents shape:
every fact keeps its provenance tags (harness, session), every observation is untagged and global.
A raw fact inherits a retired observation's status only when the fact's tags cover the
observation's (its scope is not narrower), and every tag set covers the empty one — so here the
stale fact must stop answering too. A rule that wanted the tag sets to be EQUAL would never match in
such a bank, and "the deploy script uses Bash" would keep answering as current from its session
forever, in exactly the bank shape that writes the most of them.
"""

from __future__ import annotations

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

DAY0 = datetime(2026, 5, 4, 9, 0, tzinfo=UTC)
BASH = "The deploy script is written in Bash"
PYTHON = "The deploy script was rewritten in Python"
OBSERVATION_BASH = "Deploys run through a Bash script"
OBSERVATION_PYTHON = "Deploys run through a Python script"
QUERY = "What language is the deploy script written in?"
TYPES = ["world", "experience", "observation"]
HARNESS = "harness:claude-code"


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _consolidate(request) -> Consolidation:
    if batch_fact_id(request, "Python") is not None:
        return supersedes(OBSERVATION_PYTHON, replacing="Bash", citing="Python")(request)
    return Consolidation(creates=[Observation(text=OBSERVATION_BASH, source_fact_ids=batch_fact_ids(request))])


@pytest.fixture
async def rewritten(client, llm, bank_id, settled) -> dict[str, dict]:
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Bash").returns(extracted(fact(BASH)))
    llm.on_step("extract_facts", contains="Python").returns(extracted(fact(PYTHON)))
    llm.on_step("consolidate").answers_with(_consolidate)

    async def retain(text: str, session: str, when: datetime) -> None:
        await client.aretain_batch(
            bank_id=bank_id,
            items=[
                {
                    "content": f"{text}.",
                    "timestamp": when.isoformat(),
                    "tags": [HARNESS, f"session:{session}"],
                    "observation_scopes": "shared",
                }
            ],
        )
        await settled(bank_id)

    await retain(BASH, "s1", day(0))
    await retain(PYTHON, "s2", day(20))

    memories = await client.memory.list_memories(bank_id, limit=100)
    by_text = {m.text: {"id": m.id, "tags": sorted(m.tags or [])} for m in memories.items}
    assert set(by_text) == {BASH, PYTHON, OBSERVATION_BASH, OBSERVATION_PYTHON}, sorted(by_text)
    return by_text


async def test_the_bank_has_the_shared_scope_shape(rewritten):
    assert rewritten[BASH]["tags"] == [HARNESS, "session:s1"]
    assert rewritten[PYTHON]["tags"] == [HARNESS, "session:s2"]
    assert rewritten[OBSERVATION_BASH]["tags"] == [] and rewritten[OBSERVATION_PYTHON]["tags"] == []


@pytest.mark.parametrize(
    "scope", [{}, {"tags": ["session:s1"], "tags_match": "any"}], ids=["unscoped", "the-stale-session"]
)
async def test_the_tagged_stale_fact_stops_answering_as_current(client, bank_id, rewritten, scope):
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, **scope)
    texts = {r.text for r in response.results}
    assert BASH not in texts and OBSERVATION_BASH not in texts
    assert OBSERVATION_PYTHON in texts


async def test_history_shows_both_annotated_and_the_fact_inherited_from_the_untagged_observation(
    client, bank_id, rewritten
):
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, lifecycle_view="historical")
    history = {r.text: (r.lifecycle.truth_status if r.lifecycle else None) for r in response.results}
    assert history == {OBSERVATION_PYTHON: None, PYTHON: None, OBSERVATION_BASH: "superseded", BASH: "superseded"}

    claim = (await client.memory.get_memory(bank_id, rewritten[BASH]["id"]))["lifecycle"]["claim"]
    assert (claim["truth_status"], claim["retired_via"], claim["inherited_from_id"], claim["superseded_by_id"]) == (
        "superseded",
        "inherited",
        rewritten[OBSERVATION_BASH]["id"],
        rewritten[OBSERVATION_PYTHON]["id"],
    )
    assert datetime.fromisoformat(claim["valid_to"]) == day(20)
