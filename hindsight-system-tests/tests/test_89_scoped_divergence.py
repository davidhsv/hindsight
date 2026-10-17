"""Two scopes may legitimately disagree: neither claim retires the other, both stay current.

Adaptive memory mode ``lifecycle`` retires a claim only within its own scope. "The backend API runs
on Java 21" and "the legacy worker runs on Java 8" are both true at once; so are "checkout caches
sessions in Redis" in prod and "in Memcached" in test. A lifecycle that read either pair as an
upgrade would hide a true, current fact from every recall. Two kinds of scope are covered:

- tags (``backend-api`` / ``legacy-worker``): consolidation runs per tag scope, so the other scope's
  observation is not even a candidate target;
- the ``environment`` metadata key (one tag scope, two environments): the model DOES see the prod
  observation and labels the test fact as superseding it — the validation step refuses, because the
  two claims' scope fingerprints (``environment=prod`` vs ``environment=test``) differ.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    shown_observation_entries,
    supersedes,
)

pytestmark = pytest.mark.asyncio

TYPES = ["world", "experience", "observation"]

BACKEND = "The backend API runs on Java 21"
LEGACY = "The legacy worker runs on Java 8"
BACKEND_OBSERVATION = "The backend API runs Java 21"
LEGACY_OBSERVATION = "The legacy worker runs Java 8"

PROD = "Checkout caches sessions in Redis"
TEST = "Checkout caches sessions in Memcached"
PROD_OBSERVATION = "Checkout's session cache is Redis"
TEST_OBSERVATION = "Checkout's session cache is Memcached"


def _consolidate(request) -> Consolidation:
    batch = batch_fact_ids(request)
    if batch_fact_id(request, "legacy worker") is not None:
        # Labelling the legacy worker as an upgrade of the backend API would be a cross-scope
        # retirement; it can only be attempted if that observation was shown at all.
        if any("backend API" in str(o["text"]) for o in shown_observation_entries(request)):
            return supersedes(LEGACY_OBSERVATION, replacing="backend API", citing="legacy worker")(request)
        return Consolidation(creates=[Observation(text=LEGACY_OBSERVATION, source_fact_ids=batch)])
    if batch_fact_id(request, "Memcached") is not None:
        return supersedes(TEST_OBSERVATION, replacing="Redis", citing="Memcached")(request)
    if batch_fact_id(request, "backend API") is not None:
        return Consolidation(creates=[Observation(text=BACKEND_OBSERVATION, source_fact_ids=batch)])
    return Consolidation(creates=[Observation(text=PROD_OBSERVATION, source_fact_ids=batch)])


@pytest.fixture
async def diverged(client, llm, bank_id, settled) -> dict[str, str]:
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    for text in (BACKEND, LEGACY, PROD, TEST):
        llm.on_step("extract_facts", contains=f"{text}.").returns(extracted(fact(text)))
    llm.on_step("consolidate").answers_with(_consolidate)

    async def retain(text: str, **item) -> None:
        await client.aretain(bank_id=bank_id, content=f"{text}.", **item)
        await settled(bank_id)

    await retain(BACKEND, tags=["backend-api"])
    await retain(LEGACY, tags=["legacy-worker"])
    await retain(PROD, tags=["checkout"], metadata={"environment": "prod"})
    await retain(TEST, tags=["checkout"], metadata={"environment": "test"})

    memories = await client.memory.list_memories(bank_id, limit=100)
    return {m.text: m.id for m in memories.items}


async def _claim(client, bank_id: str, memory_id: str) -> dict | None:
    memory = await client.memory.get_memory(bank_id, memory_id)
    return memory.get("lifecycle", {}).get("claim")


async def test_claims_in_different_tag_scopes_both_stay_current(client, bank_id, diverged):
    assert {BACKEND, LEGACY, BACKEND_OBSERVATION, LEGACY_OBSERVATION} <= set(diverged)
    response = await client.arecall(bank_id=bank_id, query="Which Java version runs where?", types=TYPES)
    returned = {r.text: r.lifecycle for r in response.results}
    assert {BACKEND, LEGACY, BACKEND_OBSERVATION, LEGACY_OBSERVATION} <= set(returned)
    assert all(returned[t] is None for t in (BACKEND, LEGACY, BACKEND_OBSERVATION, LEGACY_OBSERVATION))
    for text in (BACKEND, BACKEND_OBSERVATION):
        claim = await _claim(client, bank_id, diverged[text])
        assert claim is None or claim["truth_status"] == "active", (text, claim)


async def test_an_environment_mismatch_refuses_the_retirement(client, bank_id, diverged):
    """The model labelled the test fact as superseding the prod observation; validation refused the
    retirement. (Whether the test fact's knowledge still lands as an observation of its own is
    consolidation's call; what must not happen is the prod claim retiring.)"""
    assert PROD_OBSERVATION in diverged
    for text in (PROD_OBSERVATION, PROD, TEST, *([TEST_OBSERVATION] if TEST_OBSERVATION in diverged else [])):
        claim = await _claim(client, bank_id, diverged[text])
        assert claim is None or claim["truth_status"] == "active", (text, claim)
    prod = await client.memory.get_memory(bank_id, diverged[PROD_OBSERVATION])
    assert prod.get("lifecycle", {}).get("relations_in", []) == [], "no supersession was recorded"

    response = await client.arecall(
        bank_id=bank_id, query="Which cache holds checkout sessions?", types=TYPES, tags=["checkout"]
    )
    returned = {r.text: r.lifecycle for r in response.results}
    assert {PROD, TEST, PROD_OBSERVATION} <= set(returned)
    assert returned[PROD_OBSERVATION] is None and returned[PROD] is None
