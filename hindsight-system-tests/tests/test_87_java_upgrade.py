"""An upgrade retires the old claim — the observation AND the raw fact it was built from — without
erasing it: current recall answers with the new state, history is still there for the asking.

Adaptive memory mode ``lifecycle``. The billing service ran Java 17 (day 0); on day 30 a new
session says it was upgraded to Java 21, and consolidation labels the UPDATE it would emit anyway
as ``supersedes``. Three things only this composition shows:

- the stale RAW fact stops answering too. Default recall returns every fact type, so a superseded
  observation alone would still leave "runs on Java 17" answering as current; it inherits the
  observation's status;
- the change instant is the evidence's own time (day 30), written on both sides, so ``as_of``
  before and after the upgrade each see exactly one runtime, raw facts included;
- the history is an audit trail, not a deletion: the old observation keeps its text, carries its
  successor and the evidence that retired it, and GET memory shows the relation.
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


@pytest.fixture
async def upgraded(client, llm, bank_id, settled) -> dict[str, str]:
    """The bank after the upgrade; maps each memory's text to its id."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Java 17").returns(extracted(fact(JAVA_17)))
    llm.on_step("extract_facts", contains="Java 21").returns(extracted(fact(JAVA_21)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.aretain(bank_id=bank_id, content=f"{JAVA_17}.", timestamp=day(0).isoformat(), document_id="inventory")
    await settled(bank_id)
    await client.aretain(bank_id=bank_id, content=f"{JAVA_21}.", timestamp=day(30).isoformat(), document_id="upgrade")
    await settled(bank_id)

    memories = await client.memory.list_memories(bank_id, limit=100)
    ids = {m.text: m.id for m in memories.items}
    assert set(ids) == {JAVA_17, JAVA_21, OBSERVATION_17, OBSERVATION_21}, sorted(ids)
    return ids


async def _recall(client, bank_id: str, **kwargs) -> dict[str, dict | None]:
    """``{text: lifecycle annotation or None}`` of what a recall returned."""
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, **kwargs)
    return {r.text: (r.lifecycle.to_dict() if r.lifecycle else None) for r in response.results}


async def test_current_recall_answers_with_the_new_runtime_only(client, bank_id, upgraded):
    assert await _recall(client, bank_id) == {OBSERVATION_21: None, JAVA_21: None}


async def test_history_returns_the_old_runtime_annotated_with_its_successor(client, bank_id, upgraded):
    retired = {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": None,
        "valid_to": day(30).isoformat(),
        "superseded_by": upgraded[OBSERVATION_21],
    }
    history = await _recall(client, bank_id, lifecycle_view="historical")
    assert history == {OBSERVATION_21: None, JAVA_21: None, OBSERVATION_17: retired, JAVA_17: retired}


@pytest.mark.parametrize(
    ("instant", "expected"),
    [(15, {OBSERVATION_17, JAVA_17}), (45, {OBSERVATION_21, JAVA_21})],
    ids=["before-the-upgrade", "after-the-upgrade"],
)
async def test_as_of_sees_exactly_one_runtime_at_each_instant(client, bank_id, upgraded, instant, expected):
    assert set(await _recall(client, bank_id, as_of=day(instant))) == expected


async def test_the_ranking_clock_is_not_the_truth_clock(client, bank_id, upgraded):
    """A historical query_timestamp anchors recency only: it never brings the old runtime back."""
    dated = await _recall(client, bank_id, query_timestamp=day(15).isoformat())
    assert set(dated) == {OBSERVATION_21, JAVA_21}


async def test_the_retired_observation_keeps_its_text_successor_and_evidence(client, bank_id, upgraded):
    old = await client.memory.get_memory(bank_id, upgraded[OBSERVATION_17])
    assert old["text"] == OBSERVATION_17, "a supersession never rewrites the old observation"
    claim = old["lifecycle"]["claim"]
    assert (claim["truth_status"], claim["retired_via"], claim["superseded_by_id"]) == (
        "superseded",
        "supersedes",
        upgraded[OBSERVATION_21],
    )
    assert datetime.fromisoformat(claim["valid_to"]) == day(30)
    assert claim["evidence_memory_ids"] == [upgraded[JAVA_21]]
    [relation] = old["lifecycle"]["relations_in"]
    assert (relation["relation_type"], relation["origin"], relation["source_memory_id"]) == (
        "supersedes",
        "consolidation",
        upgraded[OBSERVATION_21],
    )
    assert relation["source_snapshot"]["text"] == OBSERVATION_21

    stale_fact = (await client.memory.get_memory(bank_id, upgraded[JAVA_17]))["lifecycle"]["claim"]
    assert (stale_fact["truth_status"], stale_fact["retired_via"], stale_fact["inherited_from_id"]) == (
        "superseded",
        "inherited",
        upgraded[OBSERVATION_17],
    )
    successor = (await client.memory.get_memory(bank_id, upgraded[OBSERVATION_21]))["lifecycle"]["claim"]
    assert (successor["truth_status"], datetime.fromisoformat(successor["valid_from"])) == ("active", day(30))
