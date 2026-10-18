"""A coding session comes back to the auth bug: context admission injects what session 1 learned
about it, and nothing from the sessions in between.

Five sessions in one bank, each retained with the repository, module and branch it ran in (the
metadata the coding-agents integration stamps): an auth token-refresh bug and its fix, a joke, a
frontend change, a database migration, and a near-twin of the auth bug from ANOTHER repository.
Session 5 is back on the auth bug, and its automatic first-prompt recall asks with
``purpose: "auto_context"`` and its own context. Three things only the composition shows:

- in ``strict`` mode the injected set is exactly session 1's memories: the joke, the frontend and
  the migration do not clear the auto threshold, and the other repository's twin, which shares
  most of the vocabulary, is rejected on hard scope;
- a task none of the memories helps with gets nothing at all, where an ungated recall hands back
  the least bad memory it has;
- the integration's default automatic recall asks for observations only, and consolidation writes
  no metadata on an observation: the gate still rejects the other repository's observation,
  because an observation's scope is that of the facts it was consolidated from;
- with the mode ``off`` the same requests answer exactly as a recall without the new fields does.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, batch_fact_id, extracted, fact

pytestmark = pytest.mark.asyncio

PAYMENTS = "github.com/acme/payments"
STOREFRONT = "github.com/acme/storefront"

AUTH_CAUSE = "The login token refresh fails because the auth service caches its signing keys for 24 hours"
AUTH_FIX = "The auth token refresh was fixed by lowering the auth service signing key cache to 5 minutes"
JOKE = "Joke of the day: the build server only starts working after its morning coffee"
FRONTEND = "The checkout page buttons now use the design system colour palette"
MIGRATION = "The orders table migration adds an index on customer_id"
OTHER_REPO_AUTH = "The storefront login token refresh fails because the auth proxy caches session cookies"

# (content, fact texts, metadata): one retain per session, in session order.
SESSIONS = [
    (
        "Debugging the login token refresh in the auth service.",
        [AUTH_CAUSE, AUTH_FIX],
        {"repository": PAYMENTS, "module": "auth", "branch": "fix/auth-timeout"},
    ),
    ("A joke about the build server.", [JOKE], {"repository": PAYMENTS, "branch": "main"}),
    ("Restyling the checkout page.", [FRONTEND], {"repository": PAYMENTS, "module": "frontend", "branch": "main"}),
    ("Adding the orders index.", [MIGRATION], {"repository": PAYMENTS, "module": "db", "branch": "main"}),
    (
        "Storefront login token refresh bug.",
        [OTHER_REPO_AUTH],
        {"repository": STOREFRONT, "module": "auth", "branch": "main"},
    ),
]

AUTH_OBSERVATION = "Login token refresh failures came from the auth service caching its signing keys for 24 hours"
OTHER_REPO_OBSERVATION = "Storefront login token refresh failures come from the auth proxy caching session cookies"

SESSION_5_TASK = "The login token refresh is failing again in the auth service. What did we change last time?"
SESSION_5_CONTEXT = {"repository": PAYMENTS, "module": "auth", "branch": "main"}
UNRELATED_TASK = "Which coffee beans should we order for the office kitchen?"


@pytest.fixture
async def coding_bank(client, llm, bank_id, settled) -> str:
    for content, facts, _ in SESSIONS:
        llm.on_step("extract_facts", contains=content).returns(extracted(*(fact(text) for text in facts)))

    def observe_the_auth_sessions(request) -> Consolidation:
        # One observation per auth session, drawn from that session's facts only, so each one's
        # provenance is a single repository.
        creates = []
        auth_ids = [i for i in (batch_fact_id(request, AUTH_CAUSE), batch_fact_id(request, AUTH_FIX)) if i]
        if auth_ids:
            creates.append(Observation(text=AUTH_OBSERVATION, source_fact_ids=auth_ids))
        if other_id := batch_fact_id(request, OTHER_REPO_AUTH):
            creates.append(Observation(text=OTHER_REPO_OBSERVATION, source_fact_ids=[other_id]))
        return Consolidation(creates=creates)

    llm.on_step("consolidate").answers_with(observe_the_auth_sessions)
    await client.acreate_bank(bank_id=bank_id, name="coding")
    for index, (content, _, metadata) in enumerate(SESSIONS):
        await client.aretain(bank_id=bank_id, content=content, document_id=f"session-{index + 1}", metadata=metadata)
    await settled(bank_id)
    return bank_id


async def _set_mode(client, bank_id: str, mode: str) -> None:
    await client.banks.update_bank_config(bank_id, {"updates": {"context_admission_mode": mode}})


async def _recall(client, bank_id: str, query: str, **kwargs):
    return await client.arecall(bank_id=bank_id, query=query, types=["world", "experience"], **kwargs)


def _texts(response) -> list[str]:
    return [r.text for r in response.results]


async def test_session_5_gets_session_1_and_nothing_in_between(client, coding_bank):
    ungated = await _recall(client, coding_bank, SESSION_5_TASK)
    ungated_texts = _texts(ungated)
    # Retrieval alone hands the agent the other repository's twin and the frontend change too:
    # what keeps them out below is the gate, not the ranking.
    assert {AUTH_CAUSE, AUTH_FIX, OTHER_REPO_AUTH, FRONTEND} <= set(ungated_texts)
    ids = {r.text: r.id for r in ungated.results}
    # The scope the gate compares is the metadata the session was retained with.
    assert next(r.metadata for r in ungated.results if r.text == AUTH_CAUSE) == SESSIONS[0][2]

    await _set_mode(client, coding_bank, "strict")
    gated = await _recall(
        client, coding_bank, SESSION_5_TASK, purpose="auto_context", context=SESSION_5_CONTEXT, trace=True
    )

    # Exactly session 1, in recall's own order, although session 5 runs on another branch:
    # branch is soft scope and only moves the score.
    assert _texts(gated) == [t for t in ungated_texts if t in (AUTH_CAUSE, AUTH_FIX)]
    admission = gated.trace["context_admission"]
    assert (admission["mode"], admission["purpose"], admission["abstained"]) == ("strict", "auto_context", False)
    # "last time" refers to shared history, which lowers the automatic threshold.
    assert admission["reference_signal"] == "last_time"
    reasons = {d["memory_id"]: d["reason"] for d in admission["decisions"]}
    assert reasons[ids[OTHER_REPO_AUTH]] == "wrong_scope"
    assert reasons[ids[FRONTEND]] == "below_threshold"
    assert reasons[ids[AUTH_CAUSE]] == reasons[ids[AUTH_FIX]] == "admitted"
    assert admission["cross_scope_candidates"] >= 1


async def test_a_task_no_memory_helps_with_gets_nothing(client, coding_bank):
    ungated = await _recall(client, coding_bank, UNRELATED_TASK)
    # Ungated, recall answers with the least bad memories it has (the coffee joke among them).
    assert JOKE in _texts(ungated)

    await _set_mode(client, coding_bank, "strict")
    gated = await _recall(
        client, coding_bank, UNRELATED_TASK, purpose="auto_context", context=SESSION_5_CONTEXT, trace=True
    )

    assert gated.results == []
    # The response says it is an abstention, not an empty bank, without a trace.
    assert gated.context_admission is not None and gated.context_admission.abstained is True
    admission = gated.trace["context_admission"]
    assert admission["abstained"] is True
    assert admission["admitted"] == 0
    assert admission["reasons"]["below_threshold"] == admission["candidates"] > 0


async def test_the_automatic_observation_recall_keeps_the_other_repository_out(client, coding_bank):
    async def observations(**kwargs):
        return await client.arecall(bank_id=coding_bank, query=SESSION_5_TASK, types=["observation"], **kwargs)

    ungated = await observations()
    # Both observations are retrieved; neither carries metadata of its own.
    assert {AUTH_OBSERVATION, OTHER_REPO_OBSERVATION} <= set(_texts(ungated))
    ids = {r.text: r.id for r in ungated.results}

    await _set_mode(client, coding_bank, "strict")
    gated = await observations(purpose="auto_context", context=SESSION_5_CONTEXT, trace=True)

    assert AUTH_OBSERVATION in _texts(gated)
    assert OTHER_REPO_OBSERVATION not in _texts(gated)
    decisions = {d["memory_id"]: d for d in gated.trace["context_admission"]["decisions"]}
    assert decisions[ids[OTHER_REPO_OBSERVATION]]["reason"] == "wrong_scope"
    assert decisions[ids[OTHER_REPO_OBSERVATION]]["scope"]["repository"] == "mismatch"
    assert decisions[ids[AUTH_OBSERVATION]]["scope"]["repository"] == "match"


async def test_mode_off_answers_as_if_the_fields_were_not_there(client, coding_bank):
    for query in (SESSION_5_TASK, UNRELATED_TASK):
        plain = await _recall(client, coding_bank, query, trace=True)
        with_fields = await _recall(
            client, coding_bank, query, purpose="auto_context", context=SESSION_5_CONTEXT, trace=True
        )
        assert [(r.id, r.text) for r in with_fields.results] == [(r.id, r.text) for r in plain.results]
        assert "context_admission" not in with_fields.trace
