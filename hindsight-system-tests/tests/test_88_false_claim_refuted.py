"""A claim found to be FALSE is refuted, not superseded: it never answers again, not even as history,
and re-ingesting the document that made it does not bring it back.

Adaptive memory mode ``lifecycle``. The README says the checkout service caches sessions in Redis;
an incident review later shows it never did (sessions live in Postgres), and consolidation labels
its update ``refutes``. The difference from a supersession is the point: "it was Redis, then it
became Postgres" would be history worth showing, "it was never Redis" is not.

The composition only this story covers is the re-ingest. Re-extracting the README produces the
Redis fact again under a new id; without its carried claim it would be a fresh, unconsolidated
fact, the next consolidation would build a new current "uses Redis" observation, and the refuted
claim would be back. So the README's later revision is re-extracted for real (its one chunk
changed), and the consolidation rule fails the story if the Redis fact ever reaches it again.
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
    historical,
    refutes,
    shown_observation_entries,
)

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
README = "readme"

REDIS = "The checkout service caches sessions in Redis"
POSTGRES = "The checkout service never used Redis; its sessions are stored in Postgres"
REGION = "The checkout service runs in eu-west-1"
OBSERVATION_REDIS = "Checkout keeps sessions in Redis"
OBSERVATION_POSTGRES = "Checkout keeps sessions in Postgres"
QUERY = "Where does the checkout service keep its sessions?"
TYPES = ["world", "experience", "observation"]


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _consolidate(request) -> Consolidation:
    if batch_fact_id(request, "Postgres") is not None:
        return refutes(OBSERVATION_POSTGRES, replacing="Redis", citing="Postgres")(request)
    if batch_fact_id(request, "eu-west-1") is not None:
        # The re-extracted Redis fact carried its refuted claim, so it is not in this batch: a
        # refuted claim is never consolidated again. Its arrival here would resurrect it.
        if batch_fact_id(request, "Redis") is not None:
            raise AssertionError("the refuted Redis fact came back to consolidation after the re-ingest")
        return Consolidation()
    return Consolidation(creates=[Observation(text=OBSERVATION_REDIS, source_fact_ids=batch_fact_ids(request))])


@pytest.fixture
async def refuted(client, llm, bank_id, settled) -> dict[str, str]:
    """The bank after the refutation; maps each memory's text to its id."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Readme v1").returns(extracted(fact(REDIS)))
    llm.on_step("extract_facts", contains="Incident review").returns(extracted(fact(POSTGRES)))
    llm.on_step("extract_facts", contains="Readme v2").returns(extracted(fact(REDIS), fact(REGION)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.aretain(
        bank_id=bank_id, content=f"Readme v1. {REDIS}.", timestamp=day(0).isoformat(), document_id=README
    )
    await settled(bank_id)
    await client.aretain(
        bank_id=bank_id,
        content=f"Incident review. {POSTGRES}.",
        timestamp=day(10).isoformat(),
        document_id="incident-42",
    )
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    ids = {m.text: m.id for m in memories.items}
    assert set(ids) == {REDIS, POSTGRES, OBSERVATION_REDIS, OBSERVATION_POSTGRES}, sorted(ids)
    return ids


async def _texts(client, bank_id: str, **kwargs) -> set[str]:
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, **kwargs)
    return {r.text for r in response.results}


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ({}, {OBSERVATION_POSTGRES, POSTGRES}),
        ({"lifecycle_view": "historical"}, {OBSERVATION_POSTGRES, POSTGRES}),
        # On day 5 the bank believed Redis, but a refuted claim was never true, so even a view of
        # that day leaves it out; the correction only holds from the review (day 10) on.
        ({"as_of": DAY0 + timedelta(days=5)}, set()),
    ],
    ids=["current", "historical", "as-of-before-the-review"],
)
async def test_the_refuted_claim_answers_in_no_view(client, bank_id, refuted, view, expected):
    assert await _texts(client, bank_id, **view) == expected


async def test_the_audit_trail_keeps_the_claim_its_evidence_and_the_correction(client, bank_id, refuted):
    old = await client.memory.get_memory(bank_id, refuted[OBSERVATION_REDIS])
    assert old["text"] == OBSERVATION_REDIS
    claim = old["lifecycle"]["claim"]
    assert (claim["truth_status"], claim["retired_via"], claim["superseded_by_id"]) == (
        "refuted",
        "refutes",
        refuted[OBSERVATION_POSTGRES],
    )
    assert claim["evidence_memory_ids"] == [refuted[POSTGRES]]
    [relation] = old["lifecycle"]["relations_in"]
    assert (relation["relation_type"], relation["origin"], relation["source_memory_id"]) == (
        "refutes",
        "consolidation",
        refuted[OBSERVATION_POSTGRES],
    )
    raw = (await client.memory.get_memory(bank_id, refuted[REDIS]))["lifecycle"]["claim"]
    assert (raw["truth_status"], raw["retired_via"]) == ("refuted", "inherited")


async def test_reingesting_the_readme_does_not_resurrect_the_refuted_claim(client, bank_id, settled, refuted):
    await client.aretain(
        bank_id=bank_id,
        content=f"Readme v2. {REDIS}. {REGION}.",
        timestamp=day(20).isoformat(),
        document_id=README,
        update_mode="replace",
    )
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    redis_ids = [m.id for m in memories.items if m.text == REDIS]
    assert len(redis_ids) == 1 and redis_ids[0] != refuted[REDIS], "the README was re-extracted under a new id"
    carried = (await client.memory.get_memory(bank_id, redis_ids[0]))["lifecycle"]["claim"]
    assert carried["truth_status"] == "refuted", "the re-extracted fact carries the refuted claim forward"

    assert await _texts(client, bank_id) == {OBSERVATION_POSTGRES, POSTGRES, REGION}
    assert REDIS not in await _texts(client, bank_id, lifecycle_view="historical")
    # The refuted observation outlives the re-ingest of the README it was drawn from: same row, same
    # audit trail, now citing the re-extracted fact.
    old = await client.memory.get_memory(bank_id, refuted[OBSERVATION_REDIS])
    assert (old["text"], old["lifecycle"]["claim"]["truth_status"]) == (OBSERVATION_REDIS, "refuted")
    assert old["source_memory_ids"] == [redis_ids[0]]


CARTS = "The checkout service keeps carts in Redis"
SESSIONS_IN_REDIS = "The checkout service keeps sessions in Redis too"
OBSERVATION_BOTH = "Checkout keeps carts and sessions in Redis"
OBSERVATION_CARTS = "Checkout keeps carts in Redis"


async def test_an_operator_refutes_one_fact_and_its_observation_is_rebuilt_from_the_rest(client, llm, bank_id, settled):
    """The explicit lifecycle API, for the claim consolidation never caught: refuting one raw fact
    of a two-source observation sweeps that observation, and consolidation rebuilds it from the
    other source alone — the refuted fact is never handed to it again."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Runbook").returns(extracted(fact(CARTS), fact(SESSIONS_IN_REDIS)))

    def consolidate(request) -> Consolidation:
        if batch_fact_id(request, "sessions") is not None and batch_fact_id(request, "carts") is not None:
            return Consolidation(creates=[Observation(text=OBSERVATION_BOTH, source_fact_ids=batch_fact_ids(request))])
        if batch_fact_id(request, "sessions") is not None:
            raise AssertionError("the refuted fact came back to consolidation")
        return Consolidation(creates=[Observation(text=OBSERVATION_CARTS, source_fact_ids=batch_fact_ids(request))])

    llm.on_step("consolidate").answers_with(consolidate)
    await client.aretain(bank_id=bank_id, content=f"Runbook. {CARTS}. {SESSIONS_IN_REDIS}.", document_id="runbook")
    await settled(bank_id)
    before = {m.text: m.id for m in (await client.memory.list_memories(bank_id, limit=100)).items}
    assert set(before) == {CARTS, SESSIONS_IN_REDIS, OBSERVATION_BOTH}, sorted(before)

    view = await client.aupdate_memory_lifecycle(
        bank_id, before[SESSIONS_IN_REDIS], truth_status="refuted", reason="sessions are in Postgres"
    )
    assert (view.claim.truth_status, view.claim.retired_via, view.claim.reason) == (
        "refuted",
        "explicit",
        "sessions are in Postgres",
    )
    [relation] = view.relations_in
    assert (relation.relation_type, relation.origin, relation.source_memory_id) == ("refutes", "explicit", None)
    await settled(bank_id)

    after = {m.text: m.id for m in (await client.memory.list_memories(bank_id, limit=100)).items}
    assert set(after) == {CARTS, SESSIONS_IN_REDIS, OBSERVATION_CARTS}, sorted(after)
    response = await client.arecall(
        bank_id=bank_id, query="What does checkout keep in Redis?", types=TYPES, include_source_facts=True
    )
    assert {r.text for r in response.results} == {OBSERVATION_CARTS, CARTS}
    [rebuilt] = [r for r in response.results if r.text == OBSERVATION_CARTS]
    assert rebuilt.source_fact_ids == [after[CARTS]]


WIKI_REDIS = "Checkout caches its sessions in Redis"
OBSERVATION_REDIS_AGAIN = "Checkout keeps sessions in Redis (again)"


async def test_a_claim_an_operator_refuted_does_not_come_back_from_another_document(client, llm, bank_id, settled):
    """An operator refutes an observation without naming what is true instead, so it has no
    successor. A wiki page then states the same claim: consolidation is shown the refuted
    observation (labelled, not retirable) and attaches the page's fact to it as history, so the claim
    stays refuted and nothing about Redis answers as current."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Readme v1").returns(extracted(fact(REDIS)))
    llm.on_step("extract_facts", contains="Wiki").returns(extracted(fact(WIKI_REDIS)))

    def consolidate(request) -> Consolidation:
        if batch_fact_id(request, "caches its sessions") is not None:
            if any(
                "Redis" in str(o["text"]) and o.get("status") == "refuted" for o in shown_observation_entries(request)
            ):
                return historical(restating="Redis", citing="caches its sessions")(request)
            # What a model shown nothing related would do: a fresh, current claim.
            return Consolidation(
                creates=[Observation(text=OBSERVATION_REDIS_AGAIN, source_fact_ids=batch_fact_ids(request))]
            )
        return Consolidation(creates=[Observation(text=OBSERVATION_REDIS, source_fact_ids=batch_fact_ids(request))])

    llm.on_step("consolidate").answers_with(consolidate)
    await client.aretain(
        bank_id=bank_id, content=f"Readme v1. {REDIS}.", timestamp=day(0).isoformat(), document_id=README
    )
    await settled(bank_id)
    observation = {m.text: m.id for m in (await client.memory.list_memories(bank_id, limit=100)).items}[
        OBSERVATION_REDIS
    ]
    view = await client.aupdate_memory_lifecycle(
        bank_id, observation, truth_status="refuted", reason="sessions are in Postgres"
    )
    assert (view.claim.truth_status, view.claim.superseded_by_id) == ("refuted", None)

    await client.aretain(
        bank_id=bank_id, content=f"Wiki. {WIKI_REDIS}.", timestamp=day(30).isoformat(), document_id="wiki"
    )
    await settled(bank_id)

    ids = {m.text: m.id for m in (await client.memory.list_memories(bank_id, limit=100)).items}
    assert set(ids) == {REDIS, WIKI_REDIS, OBSERVATION_REDIS}, sorted(ids)
    wiki = (await client.memory.get_memory(bank_id, ids[WIKI_REDIS]))["lifecycle"]["claim"]
    assert (wiki["truth_status"], wiki["inherited_from_id"]) == ("refuted", observation)
    assert await _texts(client, bank_id) == set()
