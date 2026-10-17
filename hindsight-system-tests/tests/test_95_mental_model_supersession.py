"""A knowledge page grounded on a claim that was superseded is rewritten from the new one.

Adaptive memory mode ``lifecycle``. The page about the billing runtime is written from the
"runs Java 21" observation. On day 30 an upgrade to Java 25 arrives and consolidation labels its
update ``supersedes``: the old observation stays a row (history), its claim is superseded, and
the new observation carries the current answer.

That is exactly the case the absence-based retraction check could never see: nothing the page
cites was deleted, so its grounding reads as intact, and a delta refresh would edit the page
around a claim that is no longer true. What has to hold instead, end to end:

- the refresh that consolidation triggers runs in FULL mode (``retired_grounding``), not delta;
- the page's new grounding no longer cites the superseded observation;
- its refresh trace records the cause (``superseded``) and the successor;
- the reflect calls of that refresh were shown Java 25 and never Java 21 — current recall
  excludes the superseded claim, which is what makes regenerating the right answer.

The wording of the regenerated page is the stub's; whether a real model writes "Java 25" from
that recall is the ``hs_llm_core`` judge test in api-slim
(``tests/test_mental_model_supersession_judge.py``).
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
from hindsight_system_tests.rulebook import ChatRequest, ToolCall
from hindsight_system_tests.steps import anchor_for

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
JAVA_21 = "The billing service runs on Java 21"
JAVA_25 = "The billing service was upgraded to Java 25"
OBSERVATION_21 = "The billing service runs Java 21"
OBSERVATION_25 = "The billing service runs Java 25"
PAGE_21 = "The billing service runs Java 21."
PAGE_25 = "The billing service runs Java 25."
QUERY = "Which Java version does the billing service run?"


def _consolidate(request: ChatRequest) -> Consolidation:
    if batch_fact_id(request, "Java 25") is not None:
        return supersedes(OBSERVATION_25, replacing="Java 21", citing="Java 25")(request)
    return Consolidation(creates=[Observation(text=OBSERVATION_21, source_fact_ids=batch_fact_ids(request))])


def _done(request: ChatRequest) -> ToolCall:
    """Answer from what this run was shown, citing every observation it saw (by alias)."""
    answer = PAGE_25 if OBSERVATION_25 in request.all_text else PAGE_21
    return ToolCall(name="done", arguments={"answer": answer, "observation_ids": ["o1", "o2", "o3", "o4"]})


def _reflect_prompts(llm) -> list[str]:
    return [call.all_text for call in llm.calls if anchor_for("reflect") in call.all_text]


@pytest.fixture
async def page(client, llm, bank_id, settled) -> dict[str, str]:
    """The bank after the upgrade settled; ids by text, the page's mental model, and the reflect
    prompts sent after the upgrade."""
    # The trace is what records why a refresh ran the way it did; other pages are left out so the
    # only page text a refresh can read is the evidence. Set as the bank's page default (merged
    # over the built-in one: delta over observations, refresh after consolidation) rather than on
    # the create request, where the typed client would send every trigger field's default along.
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "adaptive_memory_mode": "lifecycle",
                "knowledge_page_default_trigger": {"keep_trace": True, "exclude_mental_models": True},
            }
        },
    )
    llm.on_step("extract_facts", contains="Java 21").returns(extracted(fact(JAVA_21)))
    llm.on_step("extract_facts", contains="Java 25").returns(extracted(fact(JAVA_25)))
    llm.on_step("consolidate").answers_with(_consolidate)
    llm.on_step("reflect", tool="done").answers_with_tool_call(_done)
    llm.on_step("reflect").calls_the_offered_tool(query=QUERY)

    await client.aretain(bank_id=bank_id, content=f"{JAVA_21}.", timestamp=DAY0.isoformat(), document_id="inventory")
    await settled(bank_id)
    created = await client.knowledge_base.create_knowledge_page(
        bank_id,
        {"name": "Billing runtime", "source_query": QUERY},
    )
    await settled(bank_id)
    first = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    assert first.content.strip() == PAGE_21, "the page starts from the Java 21 observation"
    assert (first.trigger.mode, first.trigger.refresh_after_consolidation) == ("delta", True)

    shown_before = len(_reflect_prompts(llm))
    await client.aretain(
        bank_id=bank_id, content=f"{JAVA_25}.", timestamp=(DAY0 + timedelta(days=30)).isoformat(), document_id="upgrade"
    )
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    ids = {m.text: m.id for m in memories.items}
    assert {OBSERVATION_21, OBSERVATION_25} <= set(ids), sorted(ids)
    return {
        **ids,
        "mental_model_id": created.mental_model_id,
        "prompts_after_upgrade": "\n\n".join(_reflect_prompts(llm)[shown_before:]),
    }


async def test_the_supersession_triggers_a_full_refresh_that_completes(client, bank_id, page):
    operations = await client.operations.list_operations(bank_id, type="refresh_mental_model", limit=100)
    assert operations.total >= 2, "creating the page, then the refresh after the supersession"
    assert {op.status for op in operations.operations} == {"completed"}

    model = await client.mental_models.get_mental_model(bank_id, page["mental_model_id"], detail="full")
    trace = model.reflect_response["trace"]
    assert (trace["effective_mode"], trace["mode_fallback_reason"]) == ("full", "retired_grounding")
    assert model.content.strip() == PAGE_25


async def test_the_new_grounding_no_longer_cites_the_superseded_observation(client, bank_id, page):
    model = await client.mental_models.get_mental_model(bank_id, page["mental_model_id"], detail="full")
    cited = [entry["id"] for entry in model.reflect_response["based_on"].get("observation", [])]
    assert cited == [page[OBSERVATION_25]]
    assert page[OBSERVATION_21] not in cited


async def test_the_refresh_trace_records_the_cause_and_the_successor(client, bank_id, page):
    model = await client.mental_models.get_mental_model(bank_id, page["mental_model_id"], detail="full")
    retraction = model.reflect_response["trace"]["retraction"]
    assert retraction["cause"] == "superseded"
    assert retraction["fact_ids"] == [page[OBSERVATION_21]]
    assert retraction["successor_ids"] == [page[OBSERVATION_25]]
    assert retraction["applied"] is True


async def test_the_refresh_was_shown_java_25_and_never_java_21(page):
    shown = page["prompts_after_upgrade"]
    assert OBSERVATION_25 in shown, "the regeneration reads the current claim"
    assert "Java 21" not in shown, "current recall excludes the superseded claim and the fact that inherited it"
