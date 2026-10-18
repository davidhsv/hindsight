"""Adaptive memory through a real recall (pg0, the ``memory`` fixture, a pinned question_date).

What the pure tests in test_adaptive_scoring.py cannot show: that mode off and observe return the
same payload from the real pipeline, that rank reorders exactly as observe predicted (also when
prefer_observations dedups over a window the counterfactual alone fills), that the two paths able
to drop a weak memory (``min_scores.final``, the token budget) still return an exact cue, that a
failed state load is isolated, that the load running beside the reranker is cancelled with it and
never starts for an order the factor does not touch, that a store without adaptive state resolves
to off, and that the loader's SQL probes the primary keys.

Seeding uses raw SQL, and the module is therefore ``memory_backend_incompatible``:
- memories go straight into ``memory_units`` because the ranking predictions need exact texts
  and ``mentioned_at`` values, which the retain pipeline (LLM extraction, ingest clock) does not
  let a test pin;
- retention and claim rows go straight into their tables so each ranking case starts from an exact
  retention or claim state, which the public writers (reinforcement, lifecycle) cannot pin.
Assertions go through ``recall_async`` and the store's public reader.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.engine.memory_engine import Budget, count_tokens
from hindsight_api.engine.response_models import MinScores
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.search.decision_reranker import PROVIDER_NAME as DECISION_MODEL_RERANKER
from hindsight_api.engine.search.reranking import RerankResult

pytestmark = pytest.mark.memory_backend_incompatible

logger = logging.getLogger(__name__)

QDATE = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

DEPLOY_QUERY = "When does the billing team deploy the payments service?"
DEPLOY_TEXTS = [
    "The billing team deploys the payments service every Thursday afternoon.",
    "Payments service deploys are done by the billing team on Thursdays.",
    "The billing team ships payments service releases weekly on Thursday.",
    "Every Thursday the payments service gets deployed by billing.",
    "The billing team's payments service deploy window is Thursday after lunch.",
]

VAULT_TEXT = "The vault recovery passphrase is kept in the blue binder on the third shelf."
DISTRACTOR_TEXTS = [
    "The cafeteria serves lasagna on Wednesdays.",
    "Quarterly planning starts next Monday.",
    "The parking garage closes at midnight.",
]
# Observations on the deploy topic, each less on point than any DEPLOY_TEXTS line, so the
# recall's top results are raw facts and an observation sits outside the dedup window.
DEPLOY_OBSERVATIONS = [
    "The billing team owns several internal services.",
    "Release coordination for finance tools happens in the billing channel.",
    "Billing engineers prefer predictable weekly routines.",
]


def _bank() -> str:
    return f"test-am-{uuid.uuid4().hex[:8]}"


def _vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def _seed_units(
    memory, rc, bank_id: str, rows: list[tuple[str, datetime]], *, fact_type: str = "world"
) -> list[str]:
    """Insert facts with pinned text and mentioned_at; returns their ids in input order."""
    await memory.ensure_bank_profile(bank_id, request_context=rc)
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text for text, _ in rows])
    ids = [str(uuid.uuid4()) for _ in rows]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for unit_id, (text, mentioned_at), embedding in zip(ids, rows, embeddings):
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, mentioned_at)
                VALUES ($1, $2, $3, $4, $5::vector, $6)
                """,
                unit_id,
                bank_id,
                text,
                fact_type,
                _vector(embedding),
                mentioned_at,
            )
    return ids


async def _seed_retention(memory, bank_id: str, memory_id: str, *, last: datetime, stability: float) -> None:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_retention_state
                (memory_id, bank_id, stability_days, last_reinforced_at, reinforcement_count, created_at, updated_at)
            VALUES ($1, $2, $3, $4, 3, $5, $5)
            """,
            uuid.UUID(memory_id),
            bank_id,
            stability,
            last,
            QDATE,
        )


async def _seed_claim(memory, bank_id: str, memory_id: str, *, status: str) -> None:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_claim_state (memory_id, bank_id, truth_status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $4)
            """,
            uuid.UUID(memory_id),
            bank_id,
            status,
            QDATE,
        )


async def _configure(memory, bank_id: str, **updates) -> None:
    await memory._config_resolver.update_bank_config(
        bank_id, {f"adaptive_memory_{key}": value for key, value in updates.items()}
    )


async def _recall(memory, rc, bank_id: str, query: str, **kwargs):
    kwargs.setdefault("max_tokens", 4096)
    kwargs.setdefault("fact_type", ["world"])
    kwargs.setdefault("budget", Budget.MID)
    return await memory.recall_async(bank_id=bank_id, query=query, question_date=QDATE, request_context=rc, **kwargs)


def _ids(result) -> list[str]:
    return [str(r.id) for r in result.results]


def _components(result) -> dict[str, dict[str, float]]:
    return {r["node_id"]: r["score_components"] for r in result.trace["reranked"]}


async def test_off_and_observe_return_the_same_recall(memory, request_context):
    bank_id = _bank()
    try:
        ids = await _seed_units(
            memory,
            request_context,
            bank_id,
            [(text, QDATE - timedelta(days=age)) for text, age in zip(DEPLOY_TEXTS, (900, 1, 60, 5, 400))],
        )
        off = await _recall(memory, request_context, bank_id, DEPLOY_QUERY)
        off_traced = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert set(_ids(off)) == set(ids)
        assert "adaptive" not in off_traced.trace
        assert not any("adaptive_factor" in c for c in _components(off_traced).values())
        assert not any(p["phase_name"].startswith("adaptive_") for p in off_traced.trace["summary"]["phase_metrics"])

        await _configure(memory, bank_id, mode="observe", score_alpha=1.0)
        observed = await _recall(memory, request_context, bank_id, DEPLOY_QUERY)
        assert observed.model_dump_json() == off.model_dump_json()

        observed_traced = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert [(str(r.id), r.scores) for r in observed_traced.results] == [(str(r.id), r.scores) for r in off.results]
        adaptive = observed_traced.trace["adaptive"]
        assert adaptive["mode"] == "observe" and adaptive["scored"] is True
        assert sorted(adaptive["simulated_final_ids"]) == sorted(ids)
        assert all("adaptive_factor" in c and "retrievability" in c for c in _components(observed_traced).values())
        phases = {p["phase_name"]: p for p in observed_traced.trace["summary"]["phase_metrics"]}
        assert phases["adaptive_state_load"]["details"] == {"diagnostic": True}
        assert phases["adaptive_scoring"]["details"] == {"diagnostic": True}
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_rank_reorders_exactly_as_observe_predicted(memory, request_context):
    bank_id = _bank()
    try:
        # One mention age for all, so recency boosts are equal and only the retention rows differ.
        ids = await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=10)) for text in DEPLOY_TEXTS]
        )
        off_order = _ids(await _recall(memory, request_context, bank_id, DEPLOY_QUERY))
        assert set(off_order) == set(ids)
        top = off_order[0]
        # The off-mode winner has faded completely; everything else keeps its lazy default (R≈0.72).
        await _seed_retention(memory, bank_id, top, last=QDATE - timedelta(days=3650), stability=1.0)

        await _configure(memory, bank_id, mode="observe", score_alpha=1.0)
        observed = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert _ids(observed) == off_order
        predicted = observed.trace["adaptive"]["simulated_final_ids"]
        assert observed.trace["adaptive"]["top_k_changed"] is True
        assert observed.trace["adaptive"]["states_loaded"] == 1
        assert _components(observed)[top]["retrievability"] == pytest.approx(0.0, abs=1e-9)

        await _configure(memory, bank_id, mode="rank")
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert _ids(ranked) == predicted
        assert _ids(ranked)[0] != top
        assert ranked.trace["adaptive"]["rerank_changed"] is True
        for result in ranked.results:
            parts = _components(ranked)[str(result.id)]
            assert result.scores.final == pytest.approx(parts["adaptive_base_score"] * parts["adaptive_factor"])
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_an_exact_cue_recovers_a_faded_memory_past_min_scores_and_the_token_budget(memory, request_context):
    bank_id = _bank()
    try:
        vault, *_ = await _seed_units(
            memory,
            request_context,
            bank_id,
            [(VAULT_TEXT, QDATE - timedelta(days=2000))] + [(t, QDATE - timedelta(days=1)) for t in DISTRACTOR_TEXTS],
        )
        await _seed_retention(memory, bank_id, vault, last=QDATE - timedelta(days=2000), stability=1.0)
        off = await _recall(memory, request_context, bank_id, VAULT_TEXT)
        assert _ids(off)[0] == vault
        off_weight = off.results[0].scores.final

        await _configure(memory, bank_id, mode="rank")  # default alpha / floor
        ranked = await _recall(memory, request_context, bank_id, VAULT_TEXT, enable_trace=True)
        assert _ids(ranked)[0] == vault
        assert _components(ranked)[vault]["adaptive_factor"] == pytest.approx(0.93)
        assert ranked.results[0].scores.final == pytest.approx(off_weight * 0.93)

        # min_scores.final: a floor the exact cue clears in off mode still lets it through.
        floored = await _recall(
            memory, request_context, bank_id, VAULT_TEXT, min_scores=MinScores(final=off_weight * 0.5)
        )
        assert vault in _ids(floored)
        # The token budget: room for exactly one fact, and the faded exact cue is that fact.
        tight = await _recall(memory, request_context, bank_id, VAULT_TEXT, max_tokens=count_tokens(VAULT_TEXT))
        assert _ids(tight) == [vault]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_failed_state_load_is_isolated(memory, request_context, monkeypatch):
    bank_id = _bank()
    try:
        await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=30)) for text in DEPLOY_TEXTS]
        )
        off = await _recall(memory, request_context, bank_id, DEPLOY_QUERY)

        async def unavailable(self, conn, *, bank_id, memory_ids):
            raise RuntimeError("adaptive state unavailable")

        collector = MagicMock()
        monkeypatch.setattr(PostgresMemories, "load_adaptive_state", unavailable)
        monkeypatch.setattr(memory_engine_module, "get_metrics_collector", lambda: collector)
        await _configure(memory, bank_id, mode="rank", score_alpha=1.0)
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)

        # Factor 1: the same answer as off, plus a trace saying why.
        assert [(str(r.id), r.scores) for r in ranked.results] == [(str(r.id), r.scores) for r in off.results]
        assert ranked.trace["adaptive"]["state_load_error"] == "RuntimeError"
        assert ranked.trace["adaptive"]["scored"] is False
        collector.record_adaptive_state_load_error.assert_called_once_with(mode="rank", bank_id=bank_id)
        collector.record_adaptive_recall.assert_not_called()
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_store_without_adaptive_state_resolves_to_off(memory, request_context, monkeypatch):
    bank_id = _bank()
    try:
        await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=30)) for text in DEPLOY_TEXTS]
        )
        off = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)

        async def must_not_load(self, conn, *, bank_id, memory_ids):
            raise AssertionError("an unsupported store must never be asked for adaptive state")

        monkeypatch.setattr(PostgresMemories, "supports_adaptive_state_for", lambda self, bank_id: False)
        monkeypatch.setattr(PostgresMemories, "load_adaptive_state", must_not_load)
        await _configure(memory, bank_id, mode="rank", score_alpha=1.0)
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert "adaptive" not in ranked.trace
        assert [(str(r.id), r.scores) for r in ranked.results] == [(str(r.id), r.scores) for r in off.results]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_rank_matches_observe_when_only_the_counterfactual_window_holds_an_observation(
    memory, request_context, monkeypatch
):
    """prefer_observations drops the raw facts of observations inside the window. Observe replays
    that over its counterfactual order, whose window can hold an observation the real one does
    not; only the extra source read for that observation lets the replay drop its source fact."""
    bank_id = _bank()
    try:
        mentioned = QDATE - timedelta(days=10)
        raw_ids = await _seed_units(memory, request_context, bank_id, [(t, mentioned) for t in DEPLOY_TEXTS])
        observation_ids = await _seed_units(
            memory, request_context, bank_id, [(t, mentioned) for t in DEPLOY_OBSERVATIONS], fact_type="observation"
        )
        # thinking_budget 2: the dedup window and the final list are both the top 4.
        await memory._config_resolver.update_bank_config(bank_id, {"recall_budget_fixed_low": 2})
        # Retrieval's per-arm limit is the same thinking budget, which would leave nothing below
        # the window; widen it so every seeded memory is a candidate.
        real_recall_unified = PostgresMemories.recall_unified

        async def every_memory(self, **kwargs):
            return await real_recall_unified(self, **{**kwargs, "limit": 50})

        monkeypatch.setattr(PostgresMemories, "recall_unified", every_memory)
        dedup = {"fact_type": ["world", "observation"], "budget": Budget.LOW, "prefer_observations": True}

        first = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True, **dedup)
        order = [r["node_id"] for r in first.trace["reranked"]]
        assert set(order) == set(raw_ids) | set(observation_ids)
        below_window = [i for i in order[4:] if i in observation_ids]
        assert below_window, order
        observation = below_window[0]
        source = next(i for i in order[:4] if i in raw_ids)
        # An observation's sources are raw SQL for the same reason the units are (module docstring).
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_units SET source_memory_ids = $1 WHERE id = $2",
                [uuid.UUID(source)],
                uuid.UUID(observation),
            )
        off = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, **dedup)
        # The observation is outside the real window, so it supersedes nothing yet.
        assert _ids(off) == _ids(first)
        assert source in _ids(off) and observation not in _ids(off)

        # Fade everything but the observation and its source: in the factor's order those two lead.
        for memory_id in order:
            if memory_id not in (observation, source):
                await _seed_retention(memory, bank_id, memory_id, last=QDATE - timedelta(days=3650), stability=1.0)

        await _configure(memory, bank_id, mode="observe", score_alpha=1.0, score_floor=0.0)
        observed = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, **dedup)
        assert observed.model_dump_json() == off.model_dump_json()
        predicted = (await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True, **dedup)).trace[
            "adaptive"
        ]["simulated_final_ids"]
        assert observation in predicted and source not in predicted

        await _configure(memory, bank_id, mode="rank")
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, **dedup)
        assert _ids(ranked) == predicted
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


class _BlockingLoader:
    """Stands in for the store's adaptive-state read: holds its connection until cancelled.

    ``release()`` lets a load nobody cancelled return, so a regression fails the test's
    assertion instead of leaving a connection checked out that the engine teardown waits on.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self._released = asyncio.Event()

    async def __call__(self, conn, *, bank_id: str, memory_ids):
        self.calls += 1
        self.started.set()
        try:
            await self._released.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {}

    def release(self) -> None:
        self._released.set()


def _connections_in_use(pool) -> int:
    return pool.get_size() - pool.get_idle_size()


async def _until_released(pool, baseline: int) -> int:
    """The cancelled load releases its connection while it unwinds; give it a moment."""
    for _ in range(200):
        if _connections_in_use(pool) == baseline:
            break
        await asyncio.sleep(0.01)
    return _connections_in_use(pool)


@pytest.mark.parametrize(
    "failure", [RuntimeError("reranker unavailable"), asyncio.CancelledError()], ids=["raises", "cancelled"]
)
async def test_a_failed_rerank_cancels_the_state_load_and_releases_its_connection(
    memory, request_context, monkeypatch, failure
):
    bank_id = _bank()
    loader = _BlockingLoader()
    try:
        await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=30)) for text in DEPLOY_TEXTS]
        )
        await _configure(memory, bank_id, mode="rank")
        pool = await memory._get_pool()
        baseline = _connections_in_use(pool)
        held_during_rerank: list[int] = []

        async def failing_rerank(query, candidates, *, now=None):
            # Fail only once the load holds its connection, so the cancel has something to undo.
            await asyncio.wait_for(loader.started.wait(), timeout=10)
            held_during_rerank.append(_connections_in_use(pool))
            raise failure

        monkeypatch.setattr(PostgresMemories, "load_adaptive_state", loader)
        monkeypatch.setattr(memory._cross_encoder_reranker, "rerank", failing_rerank)
        with pytest.raises(type(failure)):
            await _recall(memory, request_context, bank_id, DEPLOY_QUERY)

        await asyncio.wait_for(loader.cancelled.wait(), timeout=5)
        assert loader.calls == 1
        assert held_during_rerank == [baseline + 1]
        assert await _until_released(pool, baseline) == baseline
    finally:
        loader.release()
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.parametrize("order", ["interleave", "decision_model_primary"])
async def test_an_authoritative_order_never_starts_the_state_load(memory, request_context, monkeypatch, order):
    bank_id = _bank()
    try:
        await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=30)) for text in DEPLOY_TEXTS]
        )
        reranking = "interleave" if order == "interleave" else "cross_encoder"
        off = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, reranking=reranking)

        calls: list[list[str]] = []

        async def recording_loader(self, conn, *, bank_id, memory_ids):
            calls.append(list(memory_ids))
            return {}

        monkeypatch.setattr(PostgresMemories, "load_adaptive_state", recording_loader)
        if order == "decision_model_primary":
            # Configured primary only; this rerank is still served by the local model.
            monkeypatch.setattr(
                type(memory._cross_encoder_reranker.cross_encoder),
                "primary_provider_name",
                property(lambda self: DECISION_MODEL_RERANKER),
            )
        await _configure(memory, bank_id, mode="rank", score_alpha=1.0)
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, reranking=reranking, enable_trace=True)

        assert calls == []
        adaptive = ranked.trace["adaptive"]
        assert adaptive["mode"] == "rank" and adaptive["scored"] is False
        assert "states_loaded" not in adaptive and "rerank_changed" not in adaptive
        assert not any("adaptive_factor" in c for c in _components(ranked).values())
        assert [(str(r.id), r.scores) for r in ranked.results] == [(str(r.id), r.scores) for r in off.results]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_failover_onto_the_decision_model_cancels_the_state_load(memory, request_context, monkeypatch):
    bank_id = _bank()
    loader = _BlockingLoader()
    try:
        await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=30)) for text in DEPLOY_TEXTS]
        )
        await _configure(memory, bank_id, mode="rank", score_alpha=1.0)
        pool = await memory._get_pool()
        baseline = _connections_in_use(pool)
        real_rerank = memory._cross_encoder_reranker.rerank

        async def failed_over(query, candidates, *, now=None):
            reranked = await real_rerank(query, candidates, now=now)
            await asyncio.wait_for(loader.started.wait(), timeout=10)
            return RerankResult(results=reranked.results, provider_name=DECISION_MODEL_RERANKER)

        monkeypatch.setattr(PostgresMemories, "load_adaptive_state", loader)
        monkeypatch.setattr(memory._cross_encoder_reranker, "rerank", failed_over)
        ranked = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)

        await asyncio.wait_for(loader.cancelled.wait(), timeout=5)
        assert await _until_released(pool, baseline) == baseline
        assert ranked.results
        assert ranked.trace["adaptive"]["scored"] is False
        assert not any("adaptive_factor" in c for c in _components(ranked).values())
    finally:
        loader.release()
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_the_loader_returns_stored_rows_and_probes_the_primary_keys(memory, request_context):
    bank_id = _bank()
    try:
        with_retention, with_claim, plain = await _seed_units(
            memory, request_context, bank_id, [(text, QDATE - timedelta(days=1)) for text in DEPLOY_TEXTS[:3]]
        )
        await _seed_retention(memory, bank_id, with_retention, last=QDATE - timedelta(days=2), stability=42.0)
        await _seed_claim(memory, bank_id, with_claim, status="uncertain")

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            recorder = _RecordingConnection(conn)
            states = await get_memories().load_adaptive_state(
                recorder, bank_id=bank_id, memory_ids=[with_retention, with_claim, plain, "not-a-uuid"]
            )
            assert set(states) == {with_retention, with_claim}
            assert states[with_retention].retention.stability_days == 42.0
            assert states[with_retention].retention.reinforcement_count == 3
            assert states[with_retention].claim is None
            assert states[with_claim].claim.truth_status.value == "uncertain"
            assert states[with_claim].retention is None
            # Bank-scoped: the same ids under another bank read as absent.
            assert await get_memories().load_adaptive_state(conn, bank_id=_bank(), memory_ids=[with_retention]) == {}

            # One statement, one round trip: both tables in a FULL OUTER JOIN.
            assert len(recorder.statements) == 1
            for sql, args in recorder.statements:
                async with conn.transaction():
                    default_plan = "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN {sql}", *args))
                    # The test tables are tiny, so the planner is free to seq-scan them; switching
                    # that off shows which index the predicate CAN use, which is what scales.
                    await conn.execute("SET LOCAL enable_seqscan = off")
                    forced_plan = "\n".join(r[0] for r in await conn.fetch(f"EXPLAIN {sql}", *args))
                logger.info("adaptive loader plan (default):\n%s\n(seqscan off):\n%s", default_plan, forced_plan)
                print(f"\n--- {sql}\n{default_plan}\n--- enable_seqscan=off\n{forced_plan}")
                # Measured on a 50k-memory bank with ANALYZEd stats, the planner probes
                # the retention primary key with the id list; on these few rows it may prefer a
                # bank_id index instead. Either is an index probe, on both sides of the join; a
                # sequential scan the planner cannot avoid is the regression.
                assert "Seq Scan" not in forced_plan, forced_plan
                assert forced_plan.count("Index Scan") >= 2, forced_plan
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


class _RecordingConnection:
    """Passes ``fetch`` through and keeps the SQL, so the test EXPLAINs exactly what the loader ran."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.statements: list[tuple[str, tuple]] = []

    async def fetch(self, sql: str, *args):
        self.statements.append((sql, args))
        return await self._conn.fetch(sql, *args)
