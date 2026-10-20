"""A setting that differs between environments is two facts, not a change.

"Session tokens expire after 15 minutes" retained from production and "... after 12
hours" from the sandbox are both true at once. Where each one holds is recorded at
retain time, in the item's metadata (`environment`, `service`, `module`, `repository`).
If consolidation never sees it, the second fact reads as the first one changing: the
model rewrites the production observation with the sandbox value, and the bank answers
"12 hours" for production. That is the opposite failure to test_23_contradiction,
where retiring the old observation is the job.

The seam is between retain and consolidation. The metadata is stored on each fact and
has to come back out on both sides of the consolidation prompt: on the new fact's
line, read by the candidate query, and on the source facts of the observation it is
compared with, which arrive through a recall. Asserted here: both reach the prompt,
the rest of the metadata does not, and the two observations stay two, each backed by
its own environment's fact. Whether a real model keeps them apart is
`hindsight-api-slim/tests/test_consolidation_scope_separation_llm.py`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, batch_fact_ids, extracted, fact

pytestmark = pytest.mark.asyncio

PRODUCTION = {"environment": "production", "source": "ops-wiki"}
SANDBOX = {"environment": "sandbox", "source": "ops-wiki"}
STATED = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)

PRODUCTION_FACT = "Session tokens expire after 15 minutes"
SANDBOX_FACT = "Session tokens expire after 12 hours"
PRODUCTION_OBSERVATION = "In production, session tokens expire after 15 minutes"
SANDBOX_OBSERVATION = "In the sandbox, session tokens expire after 12 hours"


def _creates(text: str):
    """A consolidation reply that records the batch's facts as one new observation."""

    def build(request) -> Consolidation:
        return Consolidation(creates=[Observation(text=text, source_fact_ids=batch_fact_ids(request))])

    return build


@pytest.fixture
async def two_environment_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="15 minutes").returns(extracted(fact(PRODUCTION_FACT)))
    llm.on_step("extract_facts", contains="12 hours").returns(extracted(fact(SANDBOX_FACT)))
    # Each pass is told apart by the scope on its new fact's line, so a scope that never
    # reached the prompt leaves the call unscripted and fails the story.
    llm.on_step("consolidate", contains="[scope: environment=production]").answers_with(
        _creates(PRODUCTION_OBSERVATION)
    )
    llm.on_step("consolidate", contains="[scope: environment=sandbox]").answers_with(_creates(SANDBOX_OBSERVATION))

    await client.aretain(bank_id=bank_id, content=f"{PRODUCTION_FACT}.", timestamp=STATED, metadata=PRODUCTION)
    await settled(bank_id)
    await client.aretain(bank_id=bank_id, content=f"{SANDBOX_FACT}.", timestamp=STATED, metadata=SANDBOX)
    await settled(bank_id)
    return bank_id


async def _facts_by_text(client, bank: str) -> dict[str, str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return {m.text: m.id for m in memories.items if m.fact_type != "observation"}


def _new_fact_lines(prompt: str) -> list[str]:
    facts = prompt[prompt.rfind("### New facts") + len("### New facts") : prompt.rfind("### Existing observations")]
    return [line for line in facts.splitlines() if line.strip()]


def _shown_observations(prompt: str) -> list[dict]:
    shown = prompt[prompt.rfind("### Existing observations") + len("### Existing observations") :].strip()
    return json.loads(shown[: shown.rindex("]") + 1])


async def test_each_new_fact_reaches_consolidation_with_its_own_scope(client, llm, two_environment_bank):
    facts = await _facts_by_text(client, two_environment_bank)
    first, second = llm.prompts_for("consolidate")

    assert _new_fact_lines(first) == [
        f"[{facts[PRODUCTION_FACT]}] {PRODUCTION_FACT} (mentioned_at={STATED}) [scope: environment=production]"
    ]
    assert _new_fact_lines(second) == [
        f"[{facts[SANDBOX_FACT]}] {SANDBOX_FACT} (mentioned_at={STATED}) [scope: environment=sandbox]"
    ]


async def test_the_observation_it_is_compared_with_shows_where_its_evidence_holds(llm, two_environment_bank):
    """The half that arrives through recall: without it the model sees a scoped new fact
    next to an observation that looks global, and cannot tell a second environment from
    a change."""
    _, second = llm.prompts_for("consolidate")

    shown = [
        (o["text"], [(s["text"], s.get("scope")) for s in o["source_memories"]]) for o in _shown_observations(second)
    ]
    assert shown == [(PRODUCTION_OBSERVATION, [(PRODUCTION_FACT, "environment=production")])]


async def test_only_the_scope_keys_reach_consolidation(llm, two_environment_bank):
    """The rest of the metadata is provenance: it would cost tokens on every fact line and
    invite splits on what is not a place."""
    prompts = llm.prompts_for("consolidate")

    assert len(prompts) == 2
    assert not any(PRODUCTION["source"] in prompt for prompt in prompts)


async def test_the_two_values_stay_two_observations(client, two_environment_bank):
    memories = await client.memory.list_memories(two_environment_bank, limit=100)
    by_id = {m.id: m for m in memories.items}

    observations = {
        o.text: [by_id[source].metadata for source in o.source_memory_ids]
        for o in memories.items
        if o.fact_type == "observation"
    }
    assert observations == {PRODUCTION_OBSERVATION: [PRODUCTION], SANDBOX_OBSERVATION: [SANDBOX]}


async def test_recall_answers_with_both_environments(client, two_environment_bank):
    """The user-visible payoff: an agent asking how long a token lasts gets each
    environment's value, not the one retained last."""
    response = await client.arecall(
        bank_id=two_environment_bank, query="When do session tokens expire?", types=["observation"]
    )

    assert sorted(r.text for r in response.results) == sorted([PRODUCTION_OBSERVATION, SANDBOX_OBSERVATION])
