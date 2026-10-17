"""The truth part of ``PATCH …/memories/{id}/lifecycle`` and the lifecycle recall fields over HTTP.

Driven over an ASGI transport against the real engine and the Postgres store (pg0, MockLLM), so
the mode gates, the claim writers (one ``lock_claims`` pass, CAS, inheritance, the invalidation
sweep for a raw fact) and the read view all run for real. The matrix: an observation superseded
with a successor (its raw source inherits), a raw fact refuted inside a multi-source observation
(the observation is rebuilt from the others), restore, uncertain, a lost CAS, every successor
validation, the mode gates, and the recall request/response fields with their 503 fail-closed
answer.

Seeding is raw SQL into ``memory_units`` (pinned texts and observation sources; the consolidation
LLM that builds observations is not under test), so the module is ``memory_backend_incompatible``.
``consolidated_at`` is read with SQL too: it is scheduler state no public read returns, and it is
the property under test (a retired fact never goes back to the queue).
"""

import uuid
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.retain import embedding_processing

pytestmark = [pytest.mark.asyncio, pytest.mark.memory_backend_incompatible]

QUERY = "Which cache does the checkout service use?"


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(memory, api_client):
    """A lifecycle-mode bank of the test's own, auto consolidation off (nothing rebuilds behind
    the assertions); deleted afterwards."""
    bank_id = f"test-alp-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    await _set_config(api_client, bank_id, adaptive_memory_mode="lifecycle", enable_auto_consolidation=False)
    yield bank_id
    await memory.delete_bank(bank_id, request_context=RequestContext())


async def _set_config(api_client: httpx.AsyncClient, bank_id: str, **updates) -> None:
    response = await api_client.patch(f"/v1/default/banks/{bank_id}/config", json={"updates": updates})
    assert response.status_code == 200, response.text


async def _insert(
    memory,
    bank_id: str,
    text: str,
    *,
    fact_type: str = "world",
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    mentioned_at: datetime = datetime(2026, 1, 10, tzinfo=UTC),
) -> str:
    embedding = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    memory_id = uuid.uuid4()
    async with acquire_with_retry(await memory._get_backend()) as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, tags, source_memory_ids, event_date,
                                      mentioned_at, created_at, updated_at, consolidated_at)
            VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $8, NOW(), NOW(), $9)
            """,
            memory_id,
            bank_id,
            text,
            fact_type,
            str(embedding[0]),
            tags or [],
            [uuid.UUID(s) for s in sources] if sources else None,
            mentioned_at,
            None if fact_type == "observation" else datetime(2026, 1, 11, tzinfo=UTC),
        )
    return str(memory_id)


async def _consolidated_at(memory, memory_id: str):
    async with acquire_with_retry(await memory._get_backend()) as conn:
        return await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", uuid.UUID(memory_id))


async def _patch(api_client: httpx.AsyncClient, bank_id: str, memory_id: str, **body) -> httpx.Response:
    return await api_client.patch(f"/v1/default/banks/{bank_id}/memories/{memory_id}/lifecycle", json=body)


async def _recall(api_client: httpx.AsyncClient, bank_id: str, **body) -> list[dict]:
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": QUERY, "types": ["world", "observation"], **body},
    )
    assert response.status_code == 200, response.text
    return response.json()["results"]


async def _get(api_client: httpx.AsyncClient, bank_id: str, memory_id: str) -> httpx.Response:
    return await api_client.get(f"/v1/default/banks/{bank_id}/memories/{memory_id}")


async def test_superseding_an_observation_retires_it_and_its_raw_source_and_records_the_relation(
    api_client, memory, bank
):
    old_fact = await _insert(memory, bank, "The checkout service caches sessions in Memcached.")
    old_obs = await _insert(memory, bank, "Checkout uses Memcached.", fact_type="observation", sources=[old_fact])
    new_fact = await _insert(
        memory,
        bank,
        "The checkout service moved its cache to Redis.",
        mentioned_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    new_obs = await _insert(
        memory,
        bank,
        "Checkout uses Redis.",
        fact_type="observation",
        sources=[new_fact],
        mentioned_at=datetime(2026, 2, 1, tzinfo=UTC),
    )

    response = await _patch(api_client, bank, old_obs, truth_status="superseded", superseded_by=new_obs, reason="moved")
    assert response.status_code == 200, response.text
    claim = response.json()["claim"]
    assert (claim["truth_status"], claim["retired_via"], claim["superseded_by_id"], claim["reason"]) == (
        "superseded",
        "explicit",
        new_obs,
        "moved",
    )
    [relation] = response.json()["relations_in"]
    assert (relation["relation_type"], relation["origin"], relation["source_memory_id"]) == (
        "supersedes",
        "explicit",
        new_obs,
    )

    # The raw source inherits (DESIGN §5.3), so the stale fact stops answering current recall.
    source = (await _get(api_client, bank, old_fact)).json()["lifecycle"]["claim"]
    assert (source["truth_status"], source["retired_via"], source["inherited_from_id"]) == (
        "superseded",
        "inherited",
        old_obs,
    )
    # The change instant is the successor's evidence time, on both sides of the edge.
    successor = (await _get(api_client, bank, new_obs)).json()["lifecycle"]["claim"]
    assert successor["valid_from"] == claim["valid_to"]
    assert datetime.fromisoformat(claim["valid_to"]) == datetime(2026, 2, 1, tzinfo=UTC)

    current = {r["id"] for r in await _recall(api_client, bank)}
    assert {new_fact, new_obs} <= current and not {old_fact, old_obs} & current
    history = {r["id"]: r for r in await _recall(api_client, bank, lifecycle_view="historical")}
    assert {old_fact, old_obs, new_fact, new_obs} <= set(history)
    assert history[old_obs]["lifecycle"]["superseded_by"] == new_obs
    assert "lifecycle" not in history[new_obs], "active results carry no lifecycle key"


async def test_refuting_one_raw_fact_of_a_multi_source_observation_rebuilds_it_from_the_others(
    api_client, memory, bank
):
    good_a = await _insert(memory, bank, "The checkout service uses Redis for its cart cache.")
    wrong = await _insert(memory, bank, "The checkout service uses Redis for sessions too.")
    good_b = await _insert(memory, bank, "Checkout's Redis cache is in eu-west-1.")
    observation = await _insert(
        memory, bank, "Checkout uses Redis.", fact_type="observation", sources=[good_a, wrong, good_b]
    )

    response = await _patch(api_client, bank, wrong, truth_status="refuted", reason="sessions are in Postgres")
    assert response.status_code == 200, response.text
    assert response.json()["claim"]["truth_status"] == "refuted"
    [relation] = response.json()["relations_in"]
    assert (relation["relation_type"], relation["origin"], relation["source_memory_id"]) == (
        "refutes",
        "explicit",
        None,
    )

    # The observation built on it is swept, exactly as invalidating the fact would; its other sources
    # go back to consolidation, the refuted fact does not.
    assert (await _get(api_client, bank, observation)).status_code == 404
    assert await _consolidated_at(memory, good_a) is None
    assert await _consolidated_at(memory, good_b) is None
    assert await _consolidated_at(memory, wrong) is not None
    # Its own claim keeps it out of every view a reader can ask for.
    for view in ({}, {"lifecycle_view": "historical"}, {"as_of": "2020-01-01T00:00:00Z"}):
        assert wrong not in {r["id"] for r in await _recall(api_client, bank, **view)}, view


async def test_restoring_a_refuted_fact_requeues_it_and_restoring_an_observation_restores_its_inheritors(
    api_client, memory, bank
):
    fact = await _insert(memory, bank, "The checkout service uses Redis for sessions.")
    assert (await _patch(api_client, bank, fact, truth_status="refuted")).status_code == 200
    restored = await _patch(api_client, bank, fact, truth_status="active", reason="it does after all")
    assert restored.status_code == 200, restored.text
    claim = restored.json()["claim"]
    assert (claim["truth_status"], claim["valid_to"], claim["retired_via"], claim["reason"]) == (
        "active",
        None,
        None,
        "it does after all",
    )
    assert await _consolidated_at(memory, fact) is None, "it is consolidated again"
    assert fact in {r["id"] for r in await _recall(api_client, bank)}
    again = await _patch(api_client, bank, fact, truth_status="active")
    assert again.status_code == 200 and again.json()["claim"]["reason"] == "it does after all", "a no-op"

    source = await _insert(memory, bank, "The checkout service runs three replicas.")
    observation = await _insert(memory, bank, "Checkout has three replicas.", fact_type="observation", sources=[source])
    assert (await _patch(api_client, bank, observation, truth_status="refuted")).status_code == 200
    assert (await _get(api_client, bank, source)).json()["lifecycle"]["claim"]["truth_status"] == "refuted"
    assert (await _patch(api_client, bank, observation, truth_status="active")).status_code == 200
    assert (await _get(api_client, bank, source)).json()["lifecycle"]["claim"]["truth_status"] == "active"


async def test_uncertain_is_a_live_claim_annotated_in_every_view(api_client, memory, bank):
    fact = await _insert(memory, bank, "The checkout service cache TTL is ten minutes.")
    response = await _patch(api_client, bank, fact, truth_status="uncertain", reason="two runbooks disagree")
    assert response.status_code == 200, response.text
    assert response.json()["claim"]["truth_status"] == "uncertain"
    results = {r["id"]: r for r in await _recall(api_client, bank)}
    assert results[fact]["lifecycle"] == {
        "truth_status": "uncertain",
        "confidence": None,
        "valid_from": None,
        "valid_to": None,
        "superseded_by": None,
    }
    assert await _consolidated_at(memory, fact) is not None, "uncertain is not a retirement"


async def test_a_retired_memory_cannot_be_retired_again(api_client, memory, bank):
    fact = await _insert(memory, bank, "The checkout service uses Memcached.")
    assert (await _patch(api_client, bank, fact, truth_status="refuted")).status_code == 200
    for status in ("refuted", "superseded"):
        lost = await _patch(api_client, bank, fact, truth_status=status)
        assert lost.status_code == 409, lost.text
        assert "already refuted" in lost.json()["detail"]
    # The 409 undid a retention change sent with it, too: one transaction.
    combined = await _patch(api_client, bank, fact, truth_status="refuted", importance=0.9)
    assert combined.status_code == 409
    assert (await _get(api_client, bank, fact)).json()["lifecycle"]["retention"]["importance"] == 0.5


async def test_a_successor_must_be_a_different_live_memory_in_the_same_scope(api_client, memory, bank):
    target = await _insert(memory, bank, "The checkout service uses Memcached.", tags=["team:payments"])
    other_scope = await _insert(memory, bank, "The checkout service uses Redis.", tags=["team:search"])
    same_scope = await _insert(memory, bank, "The checkout service uses Redis now.", tags=["team:payments"])
    built_on_target = await _insert(
        memory, bank, "Checkout cache history.", fact_type="observation", tags=["team:payments"], sources=[target]
    )

    scope = await _patch(api_client, bank, target, truth_status="superseded", superseded_by=other_scope)
    assert scope.status_code == 400 and "different scope" in scope.json()["detail"]
    itself = await _patch(api_client, bank, target, truth_status="superseded", superseded_by=target)
    assert itself.status_code == 400
    built = await _patch(api_client, bank, target, truth_status="superseded", superseded_by=built_on_target)
    assert built.status_code == 400 and "built on" in built.json()["detail"]
    missing = await _patch(api_client, bank, target, truth_status="superseded", superseded_by=str(uuid.uuid4()))
    assert missing.status_code == 404
    assert (await _get(api_client, bank, target)).json()["lifecycle"]["claim"] is None, "nothing was written"

    assert (await _patch(api_client, bank, same_scope, truth_status="refuted")).status_code == 200
    retired_successor = await _patch(api_client, bank, target, truth_status="superseded", superseded_by=same_scope)
    assert retired_successor.status_code == 400 and "itself refuted" in retired_successor.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"reason": "why"},
        {"superseded_by": "123e4567-e89b-12d3-a456-426614174000"},
        {"truth_status": "active", "superseded_by": "123e4567-e89b-12d3-a456-426614174000"},
        {"truth_status": "forgotten"},
        {"truth_status": "refuted", "reason": ""},
    ],
)
async def test_malformed_truth_changes_are_rejected(api_client, memory, bank, body):
    fact = await _insert(memory, bank, "The checkout service uses Memcached.")
    assert (await _patch(api_client, bank, fact, **body)).status_code == 422


@pytest.mark.parametrize("mode", ["off", "observe", "rank"])
async def test_a_truth_change_needs_lifecycle_mode(api_client, memory, bank, mode):
    await _set_config(api_client, bank, adaptive_memory_mode=mode)
    fact = await _insert(memory, bank, "The checkout service uses Memcached.")
    response = await _patch(api_client, bank, fact, truth_status="refuted")
    assert response.status_code == 409, response.text
    assert "needs mode 'lifecycle'" in response.json()["detail"]
    if mode == "rank":
        assert (await _patch(api_client, bank, fact, importance=0.8)).status_code == 200, "retention still works"


async def test_a_missing_memory_is_404(api_client, bank):
    assert (await _patch(api_client, bank, str(uuid.uuid4()), truth_status="refuted")).status_code == 404


async def test_recall_rejects_an_internal_view_and_answers_503_when_claims_cannot_be_read(
    api_client, memory, bank, monkeypatch
):
    await _insert(memory, bank, "The checkout service uses Redis.")
    refused = await api_client.post(
        f"/v1/default/banks/{bank}/memories/recall", json={"query": QUERY, "lifecycle_view": "all-with-status"}
    )
    assert refused.status_code == 422, "the internal view is not part of the HTTP surface"

    async def broken(conn, *, bank_id, memory_ids):
        raise ConnectionResetError("replica went away")

    monkeypatch.setattr(get_memories(), "claims_for", broken)
    response = await api_client.post(f"/v1/default/banks/{bank}/memories/recall", json={"query": QUERY})
    assert response.status_code == 503, response.text
    assert response.headers["Retry-After"] == "5"
