"""Spaced, independent evidence makes a memory durable; repetition in one sitting does not.

The retention model behind adaptive memory (mode ``rank``) is the spacing effect: a claim
restated by a NEW source, after time has passed, earns stability; the same source saying it
again earns nothing, however often it says it. Two claims are built side by side in one bank,
worded so that no lexical signal can tell them apart for the cue used at the end:

- S, "Team Alpha …": stated in three different sessions, on day 0, day 30 and day 120. Each
  later session's consolidation restates the observation with unchanged text.
- R, "Team Omega …": stated ten times on day 0, all in one session.

Each step is a separate mechanism with its own tests in api-slim — evidence keys, the
reinforcement ledger, the consolidation producer, the read view, the recall factor. What only
this story shows is that they compose: the session id a client put on a retain item reaches the
ledger as the evidence key, consolidation turns the restatement into reinforcement keyed by it,
and recall months later ranks by what was learned.

R wins the proof count (ten sources against three), so the ranking at day 200 is not an artefact
of evidence volume; S's later mentions also help it through the recency boost, which is why the
story also asserts the retrievability the adaptive factor scored, read from the trace.

Ranking by retention is experimental: ``adaptive_memory_score_alpha`` defaults to 0, which leaves
every score as it is, and a positive one needs the server's decay switch. So the bank is
configured and recalled through a server started with the switch, and the story also shows the
default server's side: the same alpha is refused there, and the stored one ranks at 0, with the
retention the ledger built still scored and reported.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationUpdate,
    batch_fact_ids,
    extracted,
    fact,
    shown_observations,
)

pytestmark = pytest.mark.asyncio

DAY0 = datetime(2025, 1, 6, 9, 0, tzinfo=UTC)
SPACED = "Team Alpha deploys on Fridays"
REPEATED = "Team Omega deploys on Fridays"
#: Equal lexical overlap with both claims: neither "Alpha" nor "Omega" is in it.
CUE = "Which team deploys on Fridays?"
#: Rank mode with retention in the ranking, which only a server with the decay switch accepts.
RANK_WITH_DECAY = {"adaptive_memory_mode": "rank", "adaptive_memory_score_alpha": 0.1}


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _consolidate(request) -> Consolidation:
    """Create each claim's observation once, then restate it unchanged for every later fact."""
    batch = batch_fact_ids(request)
    text = request.user_text
    facts = text[text.rfind("### New facts") : text.rfind("### Existing observations")]
    claim = SPACED if SPACED in facts else REPEATED if REPEATED in facts else None
    if claim is None or not batch:
        return Consolidation()
    existing = shown_observations(request).get(claim)
    if existing is None:
        return Consolidation(creates=[Observation(text=claim, source_fact_ids=batch)])
    return Consolidation(
        updates=[ObservationUpdate(text=claim, observation_id=existing, source_fact_ids=batch, reason="restated")]
    )


@pytest.fixture
async def two_claims(client, experimental_decay_client, llm, bank_id, settled) -> dict[str, str]:
    """The bank after both histories; maps each claim to its observation id."""
    await experimental_decay_client.banks.update_bank_config(bank_id, {"updates": RANK_WITH_DECAY})
    llm.on_step("extract_facts", contains=SPACED).returns(extracted(fact(SPACED)))
    llm.on_step("extract_facts", contains=REPEATED).returns(extracted(fact(REPEATED)))
    llm.on_step("consolidate").answers_with(_consolidate)

    async def retain(content: str, session: str, when: datetime) -> None:
        await client.aretain(
            bank_id=bank_id, content=f"{content}.", timestamp=when.isoformat(), metadata={"session_id": session}
        )
        await settled(bank_id)

    await retain(SPACED, "s1", day(0))
    for hour in range(10):
        await retain(REPEATED, "r1", day(0) + timedelta(hours=hour))
    await retain(SPACED, "s2", day(30))
    await retain(SPACED, "s3", day(120))

    memories = await client.memory.list_memories(bank_id, type="observation", limit=100)
    observations = {m.text: m.id for m in memories.items}
    assert set(observations) == {SPACED, REPEATED}, f"expected one observation per claim, got {sorted(observations)}"
    return observations


async def _lifecycle(client, bank_id: str, memory_id: str) -> dict:
    memory = await client.memory.get_memory(bank_id, memory_id)
    assert "lifecycle" in memory, "a rank-mode bank's memory must carry its lifecycle view"
    return memory["lifecycle"]


async def test_independent_sessions_build_stability_and_one_session_does_not(client, bank_id, two_claims):
    spaced = await _lifecycle(client, bank_id, two_claims[SPACED])
    repeated = await _lifecycle(client, bank_id, two_claims[REPEATED])

    assert spaced["retention"]["stability_days"] > repeated["retention"]["stability_days"]
    assert spaced["retention"]["reinforcement_count"] == 2, "the day-30 and day-120 sessions each reinforce once"
    assert repeated["retention"]["reinforcement_count"] <= 1, "one session never re-reinforces its own claim"

    # The ledger says why: one key per independent session for S, a single key for all of R.
    spaced_keys = sorted(e["source_key"] for e in spaced["recent_events"])
    assert spaced_keys == ["session:s1", "session:s2", "session:s3"]
    assert [e["source_key"] for e in repeated["recent_events"]] == ["session:r1"]

    # Reinforcement follows the evidence's own time, not the wall clock of the retain.
    assert datetime.fromisoformat(spaced["retention"]["last_reinforced_at"]) == day(120)
    assert datetime.fromisoformat(repeated["retention"]["last_reinforced_at"]) == day(0)


async def _recall_months_later(client, bank_id: str):
    return await client.arecall(
        bank_id=bank_id, query=CUE, types=["observation"], query_timestamp=day(200).isoformat(), trace=True
    )


async def test_months_later_the_spaced_claim_outranks_the_repeated_one(experimental_decay_client, bank_id, two_claims):
    response = await _recall_months_later(experimental_decay_client, bank_id)

    assert [r.text for r in response.results] == [SPACED, REPEATED]
    assert response.trace["adaptive"]["alpha"] == 0.1
    components = {r["text"]: r["score_components"] for r in response.trace["reranked"]}
    assert components[SPACED]["retrievability"] > components[REPEATED]["retrievability"]
    assert components[SPACED]["adaptive_factor"] > components[REPEATED]["adaptive_factor"]


async def test_on_a_default_server_retention_never_moves_a_score(client, bank_id, two_claims):
    """Without the decay switch a bank cannot raise its alpha, and the one it stores ranks at 0."""
    with pytest.raises(ApiException) as refused:
        await client.banks.update_bank_config(bank_id, {"updates": RANK_WITH_DECAY})
    assert refused.value.status == 400
    assert "HINDSIGHT_API_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL" in str(refused.value.body)

    response = await _recall_months_later(client, bank_id)

    assert {r.text for r in response.results} == {SPACED, REPEATED}
    assert response.trace["adaptive"]["alpha"] == 0.0
    components = {r["text"]: r["score_components"] for r in response.trace["reranked"]}
    # Retention is still scored and reported, and still tells the claims apart...
    assert components[SPACED]["retrievability"] > components[REPEATED]["retrievability"]
    # ...but it moves no score: every factor is exactly 1 and every final score is the factor-free one.
    assert {c["adaptive_factor"] for c in components.values()} == {1.0}
    for result in response.results:
        assert result.scores.final == pytest.approx(components[result.text]["adaptive_base_score"], abs=1e-9)


async def test_an_explicit_confirmation_retried_with_its_key_counts_once(client, llm, bank_id, settled):
    """The public client's confirm, on a raw fact, against the same ledger: a retry must not double-count."""
    await client.banks.update_bank_config(bank_id, {"updates": {"adaptive_memory_mode": "rank"}})
    llm.on_step("extract_facts", contains=REPEATED).returns(extracted(fact(REPEATED)))
    llm.on_step("consolidate").returns(Consolidation())
    await client.aretain(bank_id=bank_id, content=f"{REPEATED}.", timestamp=day(0).isoformat())
    await settled(bank_id)
    [memory] = (await client.memory.list_memories(bank_id, type="world", limit=10)).items

    first = await client.aconfirm_memory(
        bank_id, memory.id, source_key="ops-review-7", event_time=day(150), reason="checked the runbook"
    )
    retry = await client.aconfirm_memory(bank_id, memory.id, source_key="ops-review-7", event_time=day(150))

    assert (first.recorded, first.deduplicated, first.outcome) == (True, False, "applied")
    assert first.reinforcement_count == 1
    assert (retry.recorded, retry.deduplicated, retry.outcome) == (False, True, "deduplicated")
    assert (retry.reinforcement_count, retry.stability_days) == (first.reinforcement_count, first.stability_days)

    lifecycle = await _lifecycle(client, bank_id, memory.id)
    assert [(e["source_type"], e["source_key"]) for e in lifecycle["recent_events"]] == [
        ("explicit_confirmation", "ops-review-7")
    ]
    assert datetime.fromisoformat(lifecycle["retention"]["last_reinforced_at"]) == day(150)
