"""The adaptive-memory axes stay independent of each other.

- Truth and strength: retrievability is computed from the retention row alone, whatever the claim
  says; retiring a memory leaves everything it learned untouched; and the claim view never reads
  retrievability, so a faded memory still answers as current while a superseded one that is fully
  retrievable never does.
- Event recency and reinforcement recency: confirming a memory moves its retrievability and leaves
  its event time, and the recency boost recall computes from that time, where they were.

The integration cases seed raw SQL (pinned texts, ``mentioned_at`` and retention rows), so the
module is ``memory_backend_incompatible``; claims are written through the store interface
consolidation uses, and assertions go through ``recall_async`` and the store's readers.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.adaptive.models import AdaptiveState, TruthStatus
from hindsight_api.engine.adaptive.retention import effective_retention
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories
from tests.test_adaptive_lifecycle_recall import _claim, _seed
from tests.test_adaptive_retention_math import _settings, _state, day

pytestmark = pytest.mark.memory_backend_incompatible

QDATE = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("status", list(TruthStatus))
def test_retrievability_is_the_same_whatever_the_claim_says(status):
    row = _state(last=day(0), stability=40.0, count=2)
    plain = effective_retention(
        AdaptiveState(retention=row, claim=None),
        fact_type="world",
        mentioned_at=day(0),
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(90),
    )
    claimed = effective_retention(
        AdaptiveState(retention=row, claim=_claim(status, valid_to=day(30))),
        fact_type="world",
        mentioned_at=day(0),
        occurred_start=None,
        event_date=None,
        settings=_settings(),
        now=day(90),
    )
    assert claimed == plain


# --------------------------------------------------------------------------- pg0

QUERY = "Which database does the ledger service keep its balances in?"
FADED = "The ledger service keeps its balances in PostgreSQL."
STALE = "The ledger service keeps its balances in MySQL."
SUCCESSOR = "The ledger service moved its balances from MySQL to PostgreSQL."


def _bank() -> str:
    return f"test-axes-{uuid.uuid4().hex[:8]}"


async def _seed_retention(
    memory, bank_id: str, memory_id: str, *, last: datetime, stability: float, importance: float, exempt: bool
) -> None:
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_retention_state
                (memory_id, bank_id, importance, stability_days, last_reinforced_at, reinforcement_count,
                 forgetting_exempt, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, 4, $6, $7, $7)
            """,
            uuid.UUID(memory_id),
            bank_id,
            importance,
            stability,
            last,
            exempt,
            QDATE - timedelta(days=30),
        )


async def _set_mode(memory, bank_id: str, mode: str) -> None:
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": mode})


def _adaptive(mode: str) -> AdaptiveSettings:
    return AdaptiveSettings.resolve({"adaptive_memory_mode": mode}, get_config(), store_supports=True)


async def _retention_view(memory, bank_id: str, memory_id: str):
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        view = await get_memories().get_memory_lifecycle(
            conn, bank_id=bank_id, memory_id=memory_id, now=QDATE, settings=_adaptive("lifecycle")
        )
    assert view is not None
    return view


async def _recall(memory, rc, bank_id: str, **kwargs):
    return await memory.recall_async(
        bank_id=bank_id,
        query=QUERY,
        fact_type=["world"],
        question_date=QDATE,
        enable_trace=True,
        request_context=rc,
        **kwargs,
    )


def _components(result) -> dict[str, dict]:
    return {r["node_id"]: r["score_components"] for r in result.trace["reranked"]}


async def test_retiring_a_memory_keeps_what_it_learned_and_the_view_never_reads_strength(memory, request_context):
    bank_id = _bank()
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        await _set_mode(memory, bank_id, "lifecycle")
        faded = await _seed(memory, bank_id, FADED, mentioned_at=QDATE - timedelta(days=2000))
        stale = await _seed(memory, bank_id, STALE, mentioned_at=QDATE - timedelta(days=400))
        successor = await _seed(memory, bank_id, SUCCESSOR, mentioned_at=QDATE - timedelta(days=60))
        # Faded: two thousand days since a one-day stability, R ~ 0. Stale: exempt and important, R = 1.
        await _seed_retention(
            memory, bank_id, faded, last=QDATE - timedelta(days=2000), stability=1.0, importance=0.5, exempt=False
        )
        await _seed_retention(
            memory, bank_id, stale, last=QDATE - timedelta(days=400), stability=3650.0, importance=1.0, exempt=True
        )

        learned = (await _retention_view(memory, bank_id, stale)).retention
        assert learned.materialized and learned.retrievability == 1.0
        store = get_memories()
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                await store.lock_claims(conn, bank_id=bank_id, memory_ids=[stale, successor], skip_locked=False)
                assert await store.cas_retire(
                    conn,
                    bank_id=bank_id,
                    memory_id=stale,
                    to_status=TruthStatus.SUPERSEDED,
                    retired_via="supersedes",
                    superseded_by_id=successor,
                    valid_to=QDATE - timedelta(days=60),
                    reason="test",
                    confidence=None,
                )
        retired = await _retention_view(memory, bank_id, stale)
        assert retired.claim is not None and retired.claim.truth_status == "superseded"
        assert retired.retention == learned, "a truth change leaves retention as it was"

        current = await _recall(memory, request_context, bank_id)
        shown = {r.id for r in current.results}
        assert faded in shown and successor in shown, "fading never retires a memory"
        assert stale not in shown, "being fully retrievable never makes a retired memory current"
        assert _components(current)[faded]["retrievability"] < 0.01

        history = await _recall(memory, request_context, bank_id, lifecycle_view="historical")
        by_id = {r.id: r for r in history.results}
        assert by_id[stale].lifecycle is not None and by_id[stale].lifecycle.truth_status == "superseded"

        async with acquire_with_retry(backend) as conn:
            claims = await store.claims_for(conn, bank_id=bank_id, memory_ids=[faded])
        assert faded not in claims or claims[faded].truth_status is TruthStatus.ACTIVE
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_confirmation_moves_retrievability_and_leaves_the_event_time_and_its_recency_boost(
    memory, request_context
):
    bank_id = _bank()
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        await _set_mode(memory, bank_id, "rank")
        fact = await _seed(memory, bank_id, FADED, mentioned_at=QDATE - timedelta(days=200))

        async def event_time():
            async with memory._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT mentioned_at, occurred_start, occurred_end FROM memory_units WHERE id = $1",
                    uuid.UUID(fact),
                )
            return dict(row)

        dated = await event_time()
        before = _components(await _recall(memory, request_context, bank_id))[fact]
        confirmation = await memory.confirm_memory(
            bank_id,
            fact,
            source_key="ops-review",
            event_time=QDATE - timedelta(days=1),
            request_context=request_context,
        )
        assert confirmation is not None
        after = _components(await _recall(memory, request_context, bank_id))[fact]

        assert after["retrievability"] > before["retrievability"]
        assert after["reinforcement_count"] == before["reinforcement_count"] + 1
        assert after["recency"] == before["recency"], "the event-recency boost reads the event time only"
        assert await event_time() == dated
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
