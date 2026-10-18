"""The adaptive-memory HTTP surface: explicit confirm, PATCH lifecycle, and GET memory's ``lifecycle`` key.

Driven over an ASGI transport against the real engine and the Postgres store (pg0, MockLLM), so the
mode gates, the reinforcement ledger and the read view all run for real. What the store tests
(test_adaptive_reinforcement_store.py) already pin — the apply math, locking, dedup under
concurrency — is only touched here as far as the surface reports it.

Seeding is raw SQL into ``memory_units`` (exact ``mentioned_at`` values), so the module is
``memory_backend_incompatible``.
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app
from hindsight_api.config import get_config
from hindsight_api.engine.adaptive.retention import initial_stability
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.retain import embedding_processing
from hindsight_api.extensions import BankWriteOperation, ValidationResult

pytestmark = [pytest.mark.asyncio, pytest.mark.memory_backend_incompatible]

#: Old enough that a confirmation now is outside the cooldown and reinforces.
SIXTY_DAYS_AGO = datetime.now(UTC) - timedelta(days=60)


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def bank(memory, api_client):
    """A bank of the test's own; deleted afterwards."""
    bank_id = f"test-am-http-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    yield bank_id
    await memory.delete_bank(bank_id, request_context=RequestContext())


async def _set_config(api_client: httpx.AsyncClient, bank_id: str, **updates) -> None:
    response = await api_client.patch(f"/v1/default/banks/{bank_id}/config", json={"updates": updates})
    assert response.status_code == 200, response.text


async def _insert_memory(
    memory, bank_id: str, text: str, *, fact_type: str = "world", mentioned_at: datetime = SIXTY_DAYS_AGO
) -> str:
    """One memory with a real embedding (curation needs it), mentioned at ``mentioned_at``."""
    embedding = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    memory_id = uuid.uuid4()
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, mentioned_at,
                                      created_at, updated_at, consolidated_at)
            VALUES ($1, $2, $3, $4, $5::vector, $6, $6, NOW(), NOW(), NOW())
            """,
            memory_id,
            bank_id,
            text,
            fact_type,
            str(embedding[0]),
            mentioned_at,
        )
    return str(memory_id)


def _settings(mode: str = "rank") -> AdaptiveSettings:
    return AdaptiveSettings.resolve({"adaptive_memory_mode": mode}, get_config(), store_supports=True)


async def _get(api_client: httpx.AsyncClient, bank_id: str, memory_id: str) -> dict:
    response = await api_client.get(f"/v1/default/banks/{bank_id}/memories/{memory_id}")
    assert response.status_code == 200, response.text
    return response.json()


async def _confirm(api_client: httpx.AsyncClient, bank_id: str, memory_id: str, **body) -> httpx.Response:
    return await api_client.post(f"/v1/default/banks/{bank_id}/memories/{memory_id}/confirm", json=body)


async def _patch_lifecycle(api_client: httpx.AsyncClient, bank_id: str, memory_id: str, **body) -> httpx.Response:
    return await api_client.patch(f"/v1/default/banks/{bank_id}/memories/{memory_id}/lifecycle", json=body)


# --------------------------------------------------------------------------- confirm


@pytest.mark.parametrize("fact_type", ["world", "observation"])
async def test_a_confirmation_reinforces_once_per_source_key(api_client, memory, bank, fact_type):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The payments service runs on Postgres 16.", fact_type=fact_type)
    base = initial_stability(fact_type, 0.5, _settings())

    first = await _confirm(api_client, bank, memory_id, source_key="review-1", reason="checked the inventory")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["memory_id"] == memory_id
    assert (body["recorded"], body["deduplicated"], body["outcome"]) == (True, False, "applied")
    assert body["reinforcement_count"] == 1
    assert body["stability_days"] > base

    again = await _confirm(api_client, bank, memory_id, source_key="review-1")
    assert again.status_code == 200, again.text
    replay = again.json()
    assert (replay["recorded"], replay["deduplicated"], replay["outcome"]) == (False, True, "deduplicated")
    assert replay["reinforcement_count"] == 1
    assert replay["stability_days"] == body["stability_days"], "a re-sent key must change nothing"

    lifecycle = (await _get(api_client, bank, memory_id))["lifecycle"]
    assert lifecycle["stored"] is True and lifecycle["archived"] is False
    assert lifecycle["retention"]["materialized"] is True
    assert lifecycle["retention"]["reinforcement_count"] == 1
    assert lifecycle["retention"]["stability_days"] == body["stability_days"]
    assert [(e["source_type"], e["source_key"], e["outcome"]) for e in lifecycle["recent_events"]] == [
        ("explicit_confirmation", "review-1", "applied")
    ]


async def test_an_unkeyed_confirmation_is_its_own_evidence_each_time(api_client, memory, bank):
    """Without a key every request is recorded; the second lands inside the first's cooldown."""
    await _set_config(api_client, bank, adaptive_memory_mode="lifecycle")
    memory_id = await _insert_memory(memory, bank, "Billing invoices are generated on the first of the month.")

    outcomes = []
    for _ in range(2):
        response = await _confirm(api_client, bank, memory_id)
        assert response.status_code == 200, response.text
        outcomes.append((response.json()["recorded"], response.json()["outcome"]))
    assert outcomes == [(True, "applied"), (True, "cooldown")]

    events = (await _get(api_client, bank, memory_id))["lifecycle"]["recent_events"]
    keys = {e["source_key"] for e in events}
    assert len(keys) == 2 and all(key.startswith("explicit:") for key in keys)


async def test_a_confirmation_without_a_body_is_accepted(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "Deploys are frozen during the last week of the quarter.")

    response = await api_client.post(f"/v1/default/banks/{bank}/memories/{memory_id}/confirm")

    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "applied"


async def test_a_historical_event_time_is_kept_and_a_naive_one_reads_as_utc(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(
        memory, bank, "The office moved to the fourth floor.", mentioned_at=SIXTY_DAYS_AGO - timedelta(days=60)
    )
    when = (SIXTY_DAYS_AGO + timedelta(days=10)).replace(microsecond=0)

    response = await _confirm(
        api_client, bank, memory_id, source_key="k", event_time=when.replace(tzinfo=None).isoformat()
    )

    assert response.status_code == 200, response.text
    lifecycle = (await _get(api_client, bank, memory_id))["lifecycle"]
    assert datetime.fromisoformat(lifecycle["recent_events"][0]["event_time"]) == when
    assert datetime.fromisoformat(lifecycle["retention"]["last_reinforced_at"]) == when


async def test_a_confirmation_from_the_future_is_refused(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The VPN endpoint is vpn.example.internal.")

    future = await _confirm(api_client, bank, memory_id, event_time=(datetime.now(UTC) + timedelta(days=1)).isoformat())
    assert future.status_code == 400, future.text

    # A client clock a little ahead of the server's is not the future: it counts as now.
    skewed = await _confirm(
        api_client, bank, memory_id, event_time=(datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    )
    assert skewed.status_code == 200, skewed.text

    events = (await _get(api_client, bank, memory_id))["lifecycle"]["recent_events"]
    assert len(events) == 1 and datetime.fromisoformat(events[0]["event_time"]) <= datetime.now(UTC)


@pytest.mark.parametrize(
    "body",
    [
        {"source_key": ""},
        {"source_key": "k" * 257},
        {"reason": "r" * 501},
        {"event_time": "not a time"},
    ],
)
async def test_a_malformed_confirmation_is_rejected(api_client, memory, bank, body):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The staging cluster is in eu-west-1.")

    response = await _confirm(api_client, bank, memory_id, **body)

    assert response.status_code == 422, response.text


async def test_a_long_source_key_still_deduplicates(api_client, memory, bank):
    """Keys over the ledger's bound are hashed; the same long key must still be the same evidence."""
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The on-call rotation changes on Mondays.")
    key = "ticket:" + "x" * 249

    first = await _confirm(api_client, bank, memory_id, source_key=key)
    second = await _confirm(api_client, bank, memory_id, source_key=key)

    assert (first.json()["outcome"], second.json()["outcome"]) == ("applied", "deduplicated")


@pytest.mark.parametrize(
    "config",
    [
        {"adaptive_memory_mode": "off"},
        {"adaptive_memory_mode": "observe"},
        {"adaptive_memory_mode": "rank", "adaptive_memory_reinforcement_enabled": False},
    ],
    ids=["off", "observe", "rank-reinforcement-off"],
)
async def test_a_confirmation_is_refused_where_reinforcement_is_not_recorded(api_client, memory, bank, config):
    await _set_config(api_client, bank, **config)
    memory_id = await _insert_memory(memory, bank, "Support tickets are triaged within four hours.")

    response = await _confirm(api_client, bank, memory_id, source_key="k")

    assert response.status_code == 409, response.text
    body = await _get(api_client, bank, memory_id)
    if config["adaptive_memory_mode"] == "off":
        assert "lifecycle" not in body
    else:
        assert body["lifecycle"]["stored"] is False, "a refused confirmation must write nothing"
        assert body["lifecycle"]["recent_events"] == []


async def test_confirming_a_missing_or_invalidated_memory_is_404(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")

    assert (await _confirm(api_client, bank, str(uuid.uuid4()))).status_code == 404
    assert (await _confirm(api_client, bank, "not-a-uuid")).status_code == 400

    memory_id = await _insert_memory(memory, bank, "The backup window is 02:00 to 04:00 UTC.")
    invalidate = await api_client.patch(f"/v1/default/banks/{bank}/memories/{memory_id}", json={"state": "invalidated"})
    assert invalidate.status_code == 200, invalidate.text

    assert (await _confirm(api_client, bank, memory_id, source_key="k")).status_code == 404
    assert (await _patch_lifecycle(api_client, bank, memory_id, importance=0.9)).status_code == 404


async def test_a_memory_in_another_bank_is_404(api_client, memory, bank):
    other = f"test-am-http-other-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=other, request_context=RequestContext())
    try:
        await _set_config(api_client, bank, adaptive_memory_mode="rank")
        foreign = await _insert_memory(memory, other, "The other bank's fact.")

        assert (await _confirm(api_client, bank, foreign, source_key="k")).status_code == 404
        assert (await _patch_lifecycle(api_client, bank, foreign, importance=0.9)).status_code == 404
    finally:
        await memory.delete_bank(other, request_context=RequestContext())


# --------------------------------------------------------------------------- PATCH lifecycle


async def test_patch_lifecycle_sets_importance_and_exemption(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "Customer data must stay in the EU region.")

    response = await _patch_lifecycle(api_client, bank, memory_id, importance=0.9)
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["memory_id"] == memory_id and view["stored"] is True
    assert view["retention"]["importance"] == pytest.approx(0.9)  # stored as a 4-byte real
    assert view["retention"]["stability_days"] == pytest.approx(initial_stability("world", 0.9, _settings()))
    assert view["retention"]["retrievability"] < 1.0

    response = await _patch_lifecycle(
        api_client, bank, memory_id, forgetting_exempt=True, exemption_reason="compliance requirement"
    )
    assert response.status_code == 200, response.text
    retention = response.json()["retention"]
    assert (retention["forgetting_exempt"], retention["exemption_reason"]) == (True, "compliance requirement")
    assert retention["retrievability"] == 1.0
    assert retention["importance"] == pytest.approx(0.9), "an omitted field stays as it was"

    # The GET view is the same one PATCH answers with.
    assert (await _get(api_client, bank, memory_id))["lifecycle"]["retention"]["forgetting_exempt"] is True

    response = await _patch_lifecycle(api_client, bank, memory_id, forgetting_exempt=False)
    assert response.status_code == 200, response.text
    retention = response.json()["retention"]
    assert (retention["forgetting_exempt"], retention["exemption_reason"]) == (False, None)


async def test_patch_lifecycle_places_and_lifts_a_hold(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The audit log is kept for seven years.")

    held = await _patch_lifecycle(api_client, bank, memory_id, hold=True)
    assert held.status_code == 200, held.text
    retention = held.json()["retention"]
    assert (retention["forgetting_exempt"], retention["exemption_reason"], retention["retrievability"]) == (
        True,
        "hold",
        1.0,
    )
    # Unarchiving a memory that is not archived leaves it as it is.
    active = await _patch_lifecycle(api_client, bank, memory_id, retention_state="active")
    assert active.status_code == 200, active.text
    assert active.json()["retention"]["retention_state"] == "active"

    lifted = (await _patch_lifecycle(api_client, bank, memory_id, hold=False)).json()["retention"]
    assert (lifted["forgetting_exempt"], lifted["exemption_reason"]) == (False, None)


async def test_importance_rescales_what_the_memory_learned(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The release train leaves every other Tuesday.")
    learned = (await _confirm(api_client, bank, memory_id, source_key="k")).json()["stability_days"]

    view = (await _patch_lifecycle(api_client, bank, memory_id, importance=1.0)).json()

    assert view["retention"]["stability_days"] == pytest.approx(learned * 1.5 / 1.0)
    assert view["retention"]["reinforcement_count"] == 1


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"importance": 1.5},
        {"importance": -0.1},
        {"importance": "high"},
        {"forgetting_exempt": False, "exemption_reason": "why"},
        {"exemption_reason": ""},
        {"hold": True, "forgetting_exempt": True},
        {"hold": False, "exemption_reason": "why"},
        {"retention_state": "archived"},
    ],
)
async def test_a_malformed_lifecycle_patch_is_rejected(api_client, memory, bank, body):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The data warehouse refreshes nightly.")

    response = await _patch_lifecycle(api_client, bank, memory_id, **body)

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("mode", ["off", "observe"])
async def test_a_lifecycle_patch_is_refused_outside_rank_and_lifecycle(api_client, memory, bank, mode):
    await _set_config(api_client, bank, adaptive_memory_mode=mode)
    memory_id = await _insert_memory(memory, bank, "The mobile app supports iOS 17 and later.")

    response = await _patch_lifecycle(api_client, bank, memory_id, importance=0.9)

    assert response.status_code == 409, response.text
    body = await _get(api_client, bank, memory_id)
    if mode == "off":
        assert "lifecycle" not in body
    else:
        assert body["lifecycle"]["stored"] is False, "a refused patch must write nothing"


async def test_patching_a_missing_memory_is_404(api_client, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")

    assert (await _patch_lifecycle(api_client, bank, str(uuid.uuid4()), importance=0.2)).status_code == 404
    assert (await _patch_lifecycle(api_client, bank, "nope", importance=0.2)).status_code == 400


# --------------------------------------------------------------------------- GET memory


async def test_get_memory_in_off_mode_has_no_lifecycle_key(api_client, memory, bank):
    memory_id = await _insert_memory(memory, bank, "The intranet is at intranet.example.internal.")

    assert "lifecycle" not in await _get(api_client, bank, memory_id)


async def test_get_memory_shows_the_lazy_defaults_once_the_mode_is_on(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="observe")
    memory_id = await _insert_memory(memory, bank, "The design system uses an 8px grid.")

    lifecycle = (await _get(api_client, bank, memory_id))["lifecycle"]

    assert (lifecycle["stored"], lifecycle["archived"], lifecycle["retention"]["materialized"]) == (False, False, False)
    assert lifecycle["retention"]["stability_days"] == pytest.approx(initial_stability("world", 0.5, _settings()))
    assert 0.0 < lifecycle["retention"]["retrievability"] < 1.0
    assert datetime.fromisoformat(lifecycle["retention"]["last_reinforced_at"]) == SIXTY_DAYS_AGO
    assert lifecycle["claim"] is None and lifecycle["recent_events"] == []


async def test_a_bank_switched_back_to_off_still_shows_what_was_learned(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The cache TTL is fifteen minutes.")
    assert (await _confirm(api_client, bank, memory_id, source_key="k")).status_code == 200

    await _set_config(api_client, bank, adaptive_memory_mode="off")

    lifecycle = (await _get(api_client, bank, memory_id))["lifecycle"]
    assert lifecycle["stored"] is True and lifecycle["retention"]["reinforcement_count"] == 1


async def test_an_invalidated_memory_shows_its_archived_state(api_client, memory, bank):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "The legacy API is served from api-v1.example.internal.")
    learned = (await _confirm(api_client, bank, memory_id, source_key="k")).json()

    response = await api_client.patch(
        f"/v1/default/banks/{bank}/memories/{memory_id}", json={"state": "invalidated", "reason": "decommissioned"}
    )
    assert response.status_code == 200, response.text
    # The curation response is the same detail view as GET, so it carries the key too.
    assert response.json()["lifecycle"]["archived"] is True

    body = await _get(api_client, bank, memory_id)
    assert body["state"] == "invalidated"
    assert body["lifecycle"]["archived"] is True
    assert body["lifecycle"]["retention"]["reinforcement_count"] == 1
    assert body["lifecycle"]["retention"]["stability_days"] == learned["stability_days"]

    # Off mode still shows it: the state is stored, in the archive's snapshot.
    await _set_config(api_client, bank, adaptive_memory_mode="off")
    assert (await _get(api_client, bank, memory_id))["lifecycle"]["retention"]["reinforcement_count"] == 1


# --------------------------------------------------------------------------- retain items


async def test_retain_items_carry_importance_and_exemption_to_the_engine(api_client, memory, bank, monkeypatch):
    captured: list[list[dict]] = []

    async def _capture(*, contents, **_kwargs):
        captured.append(contents)
        return [], None

    monkeypatch.setattr(memory, "retain_batch_async", _capture)
    response = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={
            "items": [
                {"content": "Pinned fact.", "importance": 0.8, "forgetting_exempt": True},
                {"content": "Plain fact."},
                {"content": "Unimportant fact.", "importance": 0, "forgetting_exempt": False},
            ]
        },
    )

    assert response.status_code == 200, response.text
    [contents] = captured
    assert [(c.get("importance"), c.get("forgetting_exempt")) for c in contents] == [
        (0.8, True),
        (None, None),
        (0.0, False),
    ]
    assert "importance" not in contents[1] and "forgetting_exempt" not in contents[1]


@pytest.mark.parametrize("item", [{"importance": 1.01}, {"importance": -1}, {"forgetting_exempt": "maybe"}])
async def test_a_retain_item_with_a_bad_retention_policy_is_rejected(api_client, bank, item):
    response = await api_client.post(f"/v1/default/banks/{bank}/memories", json={"items": [{"content": "x", **item}]})

    assert response.status_code == 422, response.text


# --------------------------------------------------------------------------- access control


def _validator(*, reject: BankWriteOperation | None = None, write_scope: list[str] | None = None) -> MagicMock:
    validator = MagicMock()
    validator.validate_bank_write = AsyncMock(
        side_effect=lambda ctx: ValidationResult.reject("forbidden")
        if ctx.operation is reject
        else ValidationResult.accept()
    )
    validator.validate_bank_read = AsyncMock(return_value=ValidationResult.accept())
    validator.resolve_tag_scope = AsyncMock(return_value=None)
    validator.resolve_write_tag_scope = AsyncMock(return_value=write_scope)
    return validator


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        (BankWriteOperation.CONFIRM_MEMORY, lambda c, b, m: _confirm(c, b, m, source_key="k")),
        (BankWriteOperation.UPDATE_MEMORY_LIFECYCLE, lambda c, b, m: _patch_lifecycle(c, b, m, importance=0.9)),
    ],
    ids=["confirm", "patch"],
)
async def test_the_operation_validator_gates_both_writes(api_client, memory, bank, monkeypatch, operation, call):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "Only admins may rotate the signing keys.")
    validator = _validator(reject=operation)
    monkeypatch.setattr(memory, "_operation_validator", validator)

    response = await call(api_client, bank, memory_id)

    assert response.status_code == 403, response.text
    assert [c.args[0].operation for c in validator.validate_bank_write.await_args_list] == [operation]
    monkeypatch.setattr(memory, "_operation_validator", None)
    assert (await _get(api_client, bank, memory_id))["lifecycle"]["stored"] is False


@pytest.mark.parametrize(
    "call",
    [
        lambda c, b, m: _confirm(c, b, m, source_key="k"),
        lambda c, b, m: _patch_lifecycle(c, b, m, importance=0.9),
    ],
    ids=["confirm", "patch"],
)
async def test_a_memory_outside_the_write_scope_is_403(api_client, memory, bank, monkeypatch, call):
    await _set_config(api_client, bank, adaptive_memory_mode="rank")
    memory_id = await _insert_memory(memory, bank, "An untagged memory belongs to everyone.")
    monkeypatch.setattr(memory, "_operation_validator", _validator(write_scope=["team:y"]))

    response = await call(api_client, bank, memory_id)

    assert response.status_code == 403, response.text
