"""A supersession survives the edits and rebuilds around it, and ends when — only when — its
evidence is gone.

Adaptive memory mode ``lifecycle``. "The billing service runs on Java 21" (day 0) was superseded on
day 30 by "upgraded to Java 25". Observations are disposable: editing, deleting or re-extracting a
source fact deletes the observations built on it and rebuilds them under new ids. A supersession
anchored on those ids would silently end on the first rebuild and the stale Java 21 claim would
answer as current again. It is anchored on the RAW evidence instead, so:

- editing the successor's fact rebuilds the successor observation; Java 21 stays excluded, and the
  retired observation points at the rebuilt successor;
- deleting the successor's document deletes the evidence itself: Java 21 is current again;
- re-extracting the successor's document unchanged carries the evidence over to the re-extracted
  fact: Java 21 stays excluded.

The predecessor side is the mirror image: the retired Java 21 observation is history, not a summary
waiting to be rebuilt, so re-extracting, editing or invalidating-and-restoring the Java 21 fact keeps
its row, its text and its place in the historical view, and it keeps citing that fact.
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
    shown_observation_entries,
    supersedes,
)

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2026, 4, 6, 9, 0, tzinfo=UTC)
JAVA_21 = "The billing service runs on Java 21"
JAVA_25 = "The billing service was upgraded to Java 25"
JAVA_25_EDITED = "The billing service was upgraded to Java 25 LTS"
OBSERVATION_21 = "The billing service runs Java 21"
OBSERVATION_25 = "The billing service runs Java 25"
QUERY = "Which Java version does the billing service run?"
TYPES = ["world", "experience", "observation"]
UPGRADE_DOC = "upgrade-notes"


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _active(request, needle: str) -> bool:
    return any(needle in str(o["text"]) and o.get("status") == "active" for o in shown_observation_entries(request))


def _shown(request, needle: str) -> bool:
    return any(needle in str(o["text"]) for o in shown_observation_entries(request))


def _consolidate(request) -> Consolidation:
    batch = batch_fact_ids(request)
    if batch_fact_id(request, "Java 25") is not None:
        if _active(request, "Java 21"):
            return supersedes(OBSERVATION_25, replacing="Java 21", citing="Java 25")(request)
        # A rebuild: the Java 21 observation is already retired, so the Java 25 fact gets its own
        # observation back, plainly.
        return Consolidation(creates=[Observation(text=OBSERVATION_25, source_fact_ids=batch)])
    if _shown(request, "Java 21"):
        # A Java 21 fact back in the queue (its supersession was undone) next to its observation.
        return Consolidation()
    return Consolidation(creates=[Observation(text=OBSERVATION_21, source_fact_ids=batch)])


@pytest.fixture
async def superseded(client, llm, bank_id, settled) -> dict[str, str]:
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "lifecycle"}})
    llm.on_step("extract_facts", contains="Java 21").returns(extracted(fact(JAVA_21)))
    llm.on_step("extract_facts", contains="Java 25").returns(extracted(fact(JAVA_25)))
    llm.on_step("consolidate").answers_with(_consolidate)

    await client.aretain(bank_id=bank_id, content=f"{JAVA_21}.", timestamp=day(0).isoformat(), document_id="inventory")
    await settled(bank_id)
    await client.aretain(bank_id=bank_id, content=f"{JAVA_25}.", timestamp=day(30).isoformat(), document_id=UPGRADE_DOC)
    await settled(bank_id)
    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_21, JAVA_25, OBSERVATION_21, OBSERVATION_25}, sorted(ids)
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}
    return ids


async def _ids(client, bank_id: str) -> dict[str, str]:
    memories = await client.memory.list_memories(bank_id, limit=100)
    return {m.text: m.id for m in memories.items}


async def _current(client, bank_id: str) -> set[str]:
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES)
    return {r.text for r in response.results}


async def _claim(client, bank_id: str, memory_id: str) -> dict:
    return (await client.memory.get_memory(bank_id, memory_id))["lifecycle"]["claim"]


async def test_editing_the_successors_fact_keeps_java_21_retired(client, bank_id, settled, superseded):
    await client.memory.update_memory(bank_id, superseded[JAVA_25], {"text": JAVA_25_EDITED})
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_21, JAVA_25_EDITED, OBSERVATION_21, OBSERVATION_25}, sorted(ids)
    assert ids[OBSERVATION_25] != superseded[OBSERVATION_25], "the successor observation was rebuilt"
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25_EDITED}
    retired = await _claim(client, bank_id, ids[OBSERVATION_21])
    assert (retired["truth_status"], retired["superseded_by_id"]) == ("superseded", ids[OBSERVATION_25]), (
        "the retired observation points at the rebuilt successor"
    )
    assert (await _claim(client, bank_id, ids[JAVA_21]))["truth_status"] == "superseded"


async def test_deleting_the_successors_document_makes_java_21_current_again(client, bank_id, settled, superseded):
    await client.documents.delete_document(bank_id, UPGRADE_DOC)
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_21, OBSERVATION_21}, sorted(ids)
    assert ids[OBSERVATION_21] == superseded[OBSERVATION_21], "the old observation is reverted, not rebuilt"
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES)
    assert {r.text: r.lifecycle for r in response.results} == {OBSERVATION_21: None, JAVA_21: None}
    for text in (OBSERVATION_21, JAVA_21):
        claim = await _claim(client, bank_id, ids[text])
        assert (claim["truth_status"], claim["valid_to"], claim["reason"]) == (
            "active",
            None,
            "reverted:evidence_deleted",
        ), text
    # The audit trail outlives the evidence: the relation stays, its successor end a tombstone.
    old = await client.memory.get_memory(bank_id, ids[OBSERVATION_21])
    [relation] = old["lifecycle"]["relations_in"]
    assert (relation["relation_type"], relation["source_memory_id"]) == ("supersedes", None)
    assert relation["source_snapshot"]["text"] == OBSERVATION_25


async def test_reextracting_the_successors_document_unchanged_keeps_java_21_retired(
    client, bank_id, settled, superseded
):
    await client.documents.reprocess_document(bank_id, UPGRADE_DOC)
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_21, JAVA_25, OBSERVATION_21, OBSERVATION_25}, sorted(ids)
    assert ids[JAVA_25] != superseded[JAVA_25], "the document was re-extracted under new ids"
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}
    retired = await _claim(client, bank_id, ids[OBSERVATION_21])
    assert retired["truth_status"] == "superseded"
    assert retired["evidence_memory_ids"] == [ids[JAVA_25]], "the evidence moved to the re-extracted fact"
    assert (await _claim(client, bank_id, ids[JAVA_21]))["truth_status"] == "superseded"


async def _historical(client, bank_id: str) -> set[str]:
    response = await client.arecall(bank_id=bank_id, query=QUERY, types=TYPES, lifecycle_view="historical")
    return {r.text for r in response.results}


async def _sources(client, bank_id: str, memory_id: str) -> list[str]:
    return list((await client.memory.get_memory(bank_id, memory_id)).get("source_memory_ids") or [])


async def test_reextracting_the_predecessors_document_keeps_the_retired_observation(
    client, bank_id, settled, superseded
):
    await client.documents.reprocess_document(bank_id, "inventory")
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {JAVA_21, JAVA_25, OBSERVATION_21, OBSERVATION_25}, sorted(ids)
    assert ids[JAVA_21] != superseded[JAVA_21], "the document was re-extracted under new ids"
    assert ids[OBSERVATION_21] == superseded[OBSERVATION_21], "the retired observation is kept, not rebuilt"
    assert await _sources(client, bank_id, ids[OBSERVATION_21]) == [ids[JAVA_21]], "it cites the re-extracted fact"
    assert (await _claim(client, bank_id, ids[OBSERVATION_21]))["truth_status"] == "superseded"
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}
    assert {OBSERVATION_21, JAVA_21} <= await _historical(client, bank_id)


async def test_editing_the_predecessors_fact_keeps_the_retired_observation(client, bank_id, settled, superseded):
    await client.memory.update_memory(bank_id, superseded[JAVA_21], {"text": f"{JAVA_21} (LTS)"})
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert set(ids) == {f"{JAVA_21} (LTS)", JAVA_25, OBSERVATION_21, OBSERVATION_25}, sorted(ids)
    assert ids[OBSERVATION_21] == superseded[OBSERVATION_21]
    assert await _sources(client, bank_id, ids[OBSERVATION_21]) == [superseded[JAVA_21]]
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}
    assert OBSERVATION_21 in await _historical(client, bank_id)


async def test_invalidating_and_restoring_the_predecessors_fact_keeps_the_retired_observation(
    client, bank_id, settled, superseded
):
    await client.memory.update_memory(bank_id, superseded[JAVA_21], {"state": "invalidated", "reason": "audit"})
    await settled(bank_id)
    assert set(await _ids(client, bank_id)) == {JAVA_25, OBSERVATION_21, OBSERVATION_25}
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}

    await client.memory.update_memory(bank_id, superseded[JAVA_21], {"state": "valid"})
    await settled(bank_id)

    ids = await _ids(client, bank_id)
    assert ids == superseded, "nothing was rebuilt: same memories, same ids"
    assert await _sources(client, bank_id, ids[OBSERVATION_21]) == [ids[JAVA_21]]
    assert (await _claim(client, bank_id, ids[JAVA_21]))["truth_status"] == "superseded"
    assert await _current(client, bank_id) == {OBSERVATION_25, JAVA_25}
    assert {OBSERVATION_21, JAVA_21} <= await _historical(client, bank_id)
