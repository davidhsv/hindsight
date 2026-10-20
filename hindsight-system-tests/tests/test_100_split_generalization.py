"""A generalization that splits into narrower rules becomes history, raw fact included.

Adaptive memory mode ``lifecycle``. On day 0 the team "always uses JOIN FETCH in JPA queries"; on
day 50 one session says detail endpoints keep JOIN FETCH while paged list endpoints use EntityGraph
because JOIN FETCH breaks pagination. Consolidation answers with two creates naming the old
observation as ``split_from``. What only this composition shows:

- the over-general RAW fact stops answering as current too. Default recall returns every fact
  type, so a retired observation alone would leave "always uses JOIN FETCH" answering; the raw fact
  inherits the split (no successor: there are two facets);
- the facets stand on their own facts. They keep the parent's raw fact as claim evidence (deleting
  the split's facts can revert it), never as a source, which is what kept it current before;
- the historical view still returns both the old observation and its raw fact.
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
    shown_observations,
)

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
GENERAL = "The team always uses JOIN FETCH in JPA queries to avoid N+1 selects"
PAGING = "Paged list endpoints use EntityGraph with batch fetching because JOIN FETCH breaks pagination"
DETAIL = "Single-entity detail endpoints keep using JOIN FETCH"
GENERAL_OBSERVATION = "The team always uses JOIN FETCH in JPA queries"
PAGING_OBSERVATION = "Paged list endpoints use EntityGraph with batch fetching instead of JOIN FETCH"
DETAIL_OBSERVATION = "Single-entity detail endpoints use JOIN FETCH"
QUERY = "How should JPA queries fetch their associations?"
TYPES = ["world", "experience", "observation"]


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _consolidate(request) -> Consolidation:
    """Day 0 creates the general observation; day 50's two facts split it into two facets."""
    paging = batch_fact_id(request, "breaks pagination")
    detail = batch_fact_id(request, "detail endpoints")
    if paging is None or detail is None:
        return Consolidation(creates=[Observation(text=GENERAL_OBSERVATION, source_fact_ids=batch_fact_ids(request))])
    parent = next(i for text, i in shown_observations(request).items() if "always uses JOIN FETCH" in text)
    return Consolidation(
        creates=[
            Observation(text=PAGING_OBSERVATION, source_fact_ids=[paging], split_from=parent),
            Observation(text=DETAIL_OBSERVATION, source_fact_ids=[detail], split_from=parent),
        ]
    )


@pytest.fixture
async def split(client, llm, bank_id, settled) -> dict[str, str]:
    """The bank after the split; maps each memory's text to its id."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="always use JOIN FETCH").returns(extracted(fact(GENERAL)))
    llm.on_step("extract_facts", contains="breaks pagination").returns(extracted(fact(PAGING), fact(DETAIL)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.aretain(
        bank_id=bank_id,
        content="We always use JOIN FETCH in JPA queries to avoid N+1 selects.",
        timestamp=day(0).isoformat(),
        document_id="jf-general",
    )
    await settled(bank_id)
    await client.aretain(
        bank_id=bank_id,
        content=(
            "JOIN FETCH breaks pagination on collections, so paged list endpoints use EntityGraph with batch "
            "fetching. Single-entity detail endpoints keep using JOIN FETCH."
        ),
        timestamp=day(50).isoformat(),
        document_id="jf-split",
    )
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    ids = {m.text: m.id for m in memories.items}
    expected = {GENERAL, PAGING, DETAIL, GENERAL_OBSERVATION, PAGING_OBSERVATION, DETAIL_OBSERVATION}
    assert set(ids) == expected, sorted(ids)
    return ids


async def _recall(client, bank_id: str, **kwargs) -> dict[str, dict | None]:
    """``{text: lifecycle annotation or None}`` of what a recall returned."""
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, **kwargs)
    return {r.text: (r.lifecycle.to_dict() if r.lifecycle else None) for r in response.results}


async def test_current_recall_answers_with_the_facets_only(client, bank_id, split):
    current = await _recall(client, bank_id)
    assert {GENERAL, GENERAL_OBSERVATION}.isdisjoint(current), sorted(current)
    assert {PAGING_OBSERVATION, DETAIL_OBSERVATION, PAGING, DETAIL} <= set(current)


async def test_history_returns_the_generalization_and_its_raw_fact(client, bank_id, split):
    retired = {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": None,
        "valid_to": day(50).isoformat(),
        "superseded_by": None,
    }
    history = await _recall(client, bank_id, lifecycle_view="historical")
    assert (history.get(GENERAL_OBSERVATION), history.get(GENERAL)) == (retired, retired)


async def test_the_raw_fact_inherits_the_split_and_the_facets_keep_it_as_evidence(client, bank_id, split):
    raw = (await client.memory.get_memory(bank_id, split[GENERAL]))["lifecycle"]["claim"]
    assert (raw["truth_status"], raw["retired_via"], raw["superseded_by_id"], raw["inherited_from_id"]) == (
        "superseded",
        "splits",
        None,
        split[GENERAL_OBSERVATION],
    )
    for facet, own in ((PAGING_OBSERVATION, PAGING), (DETAIL_OBSERVATION, DETAIL)):
        memory = await client.memory.get_memory(bank_id, split[facet])
        assert memory["source_memory_ids"] == [split[own]], facet
        assert memory["lifecycle"]["claim"]["evidence_memory_ids"] == [split[GENERAL]], facet
        [relation] = memory["lifecycle"]["relations_out"]
        assert (relation["relation_type"], relation["target_memory_id"]) == ("splits", split[GENERAL_OBSERVATION])
