"""The Jev rerank: the request it sends, the order it keeps, and what it does when Jev fails.

Unit tests over the TypeSafe client with its one exit (``_ask``) faked. The request is checked
byte for byte against the measured construction: the TypeSafe reranker's rank question over the
local top 100 in local order, with ``"\\nCurrent date: <date>"`` added to its state and
``"[Memory date: <date>] "`` before each option. The engine wiring (which recalls run it, where it
acts) is in test_recall_checks_engine.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta, timezone

import pytest

from hindsight_api import config as config_module
from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.cross_encoder import TypeSafeCrossEncoder
from hindsight_api.engine.remote_retry import RetryPolicy
from hindsight_api.engine.search import recall_checks
from hindsight_api.engine.search.jev_rerank import POOL_SIZE, jev_option, jev_rerank, memory_date, ranked_first
from hindsight_api.engine.search.recall_checks import resolve_recall_checks
from hindsight_api.engine.search.recall_guard import CircuitBreaker, GuardedRanker, RecallGuard
from hindsight_api.engine.search.reranking import rerank_document
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult

ASKED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class _Jev(TypeSafeCrossEncoder):
    """The product client with its exit faked: records each body and answers it with ``rank``
    (option position -> probability, default: later options more likely, so the answer reverses
    the order), or raises ``fail``, or answers after ``delay`` seconds."""

    def __init__(self, *, rank=None, fail: BaseException | None = None, delay: float = 0.0, **kwargs):
        super().__init__(api_key="k", **kwargs)
        self.rank_of = rank or (lambda position, n: (position + 1) / n)
        self.fail = fail
        self.delay = delay
        self.bodies: list[dict] = []

    async def _ask(self, body: dict) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        criteria = body["questions"]["rank"]["criteria"]
        n = len(criteria)
        return {"answers": {"rank": {"probabilities": {key: self.rank_of(i, n) for i, key in enumerate(criteria)}}}}


def _result(
    index: int,
    *,
    text: str | None = None,
    context: str | None = None,
    mentioned_at: datetime | None = ASKED_AT - timedelta(days=3),
    occurred_start: datetime | None = None,
    weight: float | None = None,
) -> ScoredResult:
    retrieval = RetrievalResult(
        id=f"m{index}",
        text=text if text is not None else f"memory {index}",
        fact_type="world",
        context=context,
        mentioned_at=mentioned_at,
        occurred_start=occurred_start,
    )
    return ScoredResult(
        candidate=MergedCandidate(retrieval=retrieval, rrf_score=0.0),
        cross_encoder_score=float(-index),
        cross_encoder_score_normalized=1.0 / (index + 2),
        weight=weight if weight is not None else 1.0 / (index + 1),
    )


def _pool(n: int) -> list[ScoredResult]:
    return [_result(i) for i in range(n)]


# --- the request ----------------------------------------------------------------------------


class TestRequest:
    @pytest.mark.asyncio
    async def test_the_body_is_the_measured_one_byte_for_byte(self):
        """The rank question as the reranker builds it, with the date line appended to its state
        and the date prefix before each option; key order included."""
        results = [
            _result(
                0,
                text="Alice leads the payments migration",
                context="Standup notes",
                mentioned_at=datetime(2026, 9, 28, 9, 0, tzinfo=UTC),
                occurred_start=datetime(2026, 9, 27, 0, 0, tzinfo=UTC),
            ),
            _result(1, text="Alice moved to the platform team", mentioned_at=datetime(2026, 9, 1, tzinfo=UTC)),
            _result(2, text="An undated note", mentioned_at=None, occurred_start=None),
        ]
        jev = _Jev(model="jev-1.14.0")
        await jev_rerank(jev, "What does Alice work on?", results, ASKED_AT)

        expected = {
            "state": "Question: What does Alice work on?\nCurrent date: 2026-10-01",
            "model": "jev-1.14.0",
            "questions": {
                "rank": {
                    "type": "choice",
                    "instructions": "Which candidate answers the question: What does Alice work on?",
                    "criteria": {
                        "c0": "[Memory date: 2026-09-28] [Date: September 27, 2026 (2026-09-27)] "
                        "Standup notes: Alice leads the payments migration",
                        "c1": "[Memory date: 2026-09-01] Alice moved to the platform team",
                        "c2": "An undated note",
                    },
                }
            },
        }
        assert len(jev.bodies) == 1
        assert json.dumps(jev.bodies[0], ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_without_the_dates_the_body_is_the_rerankers_own(self):
        """The two additions are the only difference from the TypeSafe reranker's rank question."""
        docs = [rerank_document(sr.retrieval) for sr in _pool(3)]
        plain, dated = _Jev(), _Jev()
        await plain._rank("q", docs, [0, 1, 2])
        await jev_rerank(dated, "q", _pool(3), ASKED_AT)

        undated = json.loads(json.dumps(dated.bodies[0]))
        undated["state"] = undated["state"].removesuffix("\nCurrent date: 2026-10-01")
        criteria = undated["questions"]["rank"]["criteria"]
        for key, option in criteria.items():
            criteria[key] = option.removeprefix("[Memory date: 2026-09-28] ")
        assert json.dumps(undated) == json.dumps(plain.bodies[0])

    @pytest.mark.asyncio
    async def test_the_pool_is_the_top_hundred_in_local_order(self):
        jev = _Jev()
        await jev_rerank(jev, "q", _pool(POOL_SIZE + 7), ASKED_AT)

        criteria = jev.bodies[0]["questions"]["rank"]["criteria"]
        assert list(criteria) == [f"c{i}" for i in range(POOL_SIZE)]
        assert [option.removeprefix("[Memory date: 2026-09-28] ") for option in criteria.values()] == [
            f"memory {i}" for i in range(POOL_SIZE)
        ]

    def test_the_memory_date_is_when_it_was_said_else_when_it_happened(self):
        said = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
        happened = datetime(2026, 8, 30, tzinfo=UTC)
        assert memory_date(_result(0, mentioned_at=said, occurred_start=happened).retrieval) == "2026-09-02"
        assert memory_date(_result(0, mentioned_at=None, occurred_start=happened).retrieval) == "2026-08-30"
        assert memory_date(_result(0, mentioned_at=None, occurred_start=None).retrieval) is None

    def test_dates_are_utc_calendar_dates(self):
        """23:30 at UTC-5 is the next day in UTC; a naive time is read as UTC."""
        late = datetime(2026, 9, 2, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
        assert memory_date(_result(0, mentioned_at=late).retrieval) == "2026-09-03"
        assert memory_date(_result(0, mentioned_at=datetime(2026, 9, 2, 23, 30)).retrieval) == "2026-09-02"

    @pytest.mark.asyncio
    async def test_the_current_date_is_the_recall_clock_in_utc(self):
        jev = _Jev()
        await jev_rerank(jev, "q", _pool(2), datetime(2026, 10, 1, 22, 0, tzinfo=timezone(timedelta(hours=-3))))
        assert jev.bodies[0]["state"] == "Question: q\nCurrent date: 2026-10-02"

    def test_an_option_is_the_cross_encoders_text_behind_its_date(self):
        result = _result(0, text="t", context="c", occurred_start=datetime(2026, 9, 27, tzinfo=UTC))
        assert jev_option(result.retrieval) == f"[Memory date: 2026-09-28] {rerank_document(result.retrieval)}"


# --- the order ------------------------------------------------------------------------------


class TestOrder:
    @pytest.mark.asyncio
    async def test_jevs_order_is_final_for_the_pool_and_the_rest_follow_in_local_order(self):
        results = _pool(POOL_SIZE + 5)
        local_ids = [sr.id for sr in results]
        out = await jev_rerank(_Jev(), "q", results, ASKED_AT)

        assert [sr.id for sr in out.results] == local_ids[:POOL_SIZE][::-1] + local_ids[POOL_SIZE:]
        assert out.ranked_ids == local_ids[:POOL_SIZE][::-1]
        assert out.error is None

    @pytest.mark.asyncio
    async def test_the_final_score_still_sorts_the_list(self):
        """The pool keeps its final scores as a set, handed out by Jev's rank; the results below
        it keep theirs, which are all lower."""
        results = _pool(POOL_SIZE + 5)
        pool_weights = sorted((sr.weight for sr in results[:POOL_SIZE]), reverse=True)
        tail_weights = [sr.weight for sr in results[POOL_SIZE:]]
        out = await jev_rerank(_Jev(), "q", results, ASKED_AT)

        weights = [sr.weight for sr in out.results]
        assert weights == sorted(weights, reverse=True)
        assert weights[:POOL_SIZE] == pool_weights
        assert weights[POOL_SIZE:] == tail_weights

    @pytest.mark.asyncio
    async def test_each_result_keeps_its_own_reranker_score(self):
        results = _pool(10)
        before = {sr.id: (sr.cross_encoder_score, sr.cross_encoder_score_normalized) for sr in results}
        out = await jev_rerank(_Jev(), "q", results, ASKED_AT)
        assert {sr.id: (sr.cross_encoder_score, sr.cross_encoder_score_normalized) for sr in out.results} == before

    @pytest.mark.asyncio
    async def test_ties_keep_the_local_order(self):
        out = await jev_rerank(_Jev(rank=lambda position, n: 1.0 / n), "q", _pool(5), ASKED_AT)
        assert [sr.id for sr in out.results] == [f"m{i}" for i in range(5)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [0, 1])
    async def test_fewer_than_two_results_send_nothing(self, n):
        jev = _Jev()
        results = _pool(n)
        out = await jev_rerank(jev, "q", results, ASKED_AT)
        assert jev.bodies == []
        assert out.results is results and out.ranked_ids == [] and out.error is None

    def test_ranked_first_applies_the_order_to_another_list_of_the_same_candidates(self):
        results = _pool(5)
        reordered = ranked_first(results, ["m3", "m0", "m9"])
        assert [sr.id for sr in reordered] == ["m3", "m0", "m1", "m2", "m4"]


# --- when Jev fails -------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            UpstreamHTTPError(400, "criteria must map 2 or more options", {}, "https://jev.test/v1/systemone"),
            RuntimeError("upstream down"),
        ],
    )
    async def test_an_error_keeps_the_local_order(self, failure):
        results = _pool(12)
        before = [(sr.id, sr.weight) for sr in results]
        out = await jev_rerank(_Jev(fail=failure), "q", results, ASKED_AT)

        assert [(sr.id, sr.weight) for sr in out.results] == before
        assert out.ranked_ids == []
        assert out.error == type(failure).__name__

    @pytest.mark.asyncio
    async def test_a_request_past_the_recall_timeout_keeps_the_local_order(self):
        results = _pool(12)
        breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0, name="test")
        ranker = GuardedRanker(_Jev(delay=1.0), RecallGuard(breaker, timeout=0.05))
        out = await jev_rerank(ranker, "q", results, ASKED_AT)
        assert [sr.id for sr in out.results] == [f"m{i}" for i in range(12)]
        assert out.error == "TimeoutError"

    @pytest.mark.asyncio
    async def test_a_skipped_request_keeps_the_local_order_without_a_warning(self, caplog):
        """The breaker is open: nothing is sent, and the breaker's own warning is the only one."""
        results = _pool(12)
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0, name="test")
        breaker.record_unavailable(breaker.admit())
        jev = _Jev()
        with caplog.at_level("WARNING"):
            out = await jev_rerank(GuardedRanker(jev, RecallGuard(breaker, timeout=2.0)), "q", results, ASKED_AT)
        assert jev.bodies == []
        assert [sr.id for sr in out.results] == [f"m{i}" for i in range(12)]
        assert (out.ranked_ids, out.error) == ([], "RecallCheckSkipped")
        assert "Jev rerank" not in caplog.text

    @pytest.mark.asyncio
    async def test_an_answer_out_of_shape_keeps_the_local_order(self):
        class _Missing(_Jev):
            async def _ask(self, body: dict) -> dict:
                return {"answers": {"rank": {"probabilities": {"c0": 0.9}}}}

        out = await jev_rerank(_Missing(), "q", _pool(4), ASKED_AT)
        assert [sr.id for sr in out.results] == ["m0", "m1", "m2", "m3"]
        assert out.error == "KeyError"

    @pytest.mark.asyncio
    async def test_a_transient_failure_is_retried(self):
        class _Flaky(_Jev):
            async def _ask(self, body: dict) -> dict:
                if not self.bodies:
                    self.bodies.append(body)
                    raise UpstreamHTTPError(503, "unavailable", {}, "https://jev.test/v1/systemone")
                return await super()._ask(body)

        jev = _Flaky()
        jev.retry_policy = RetryPolicy(max_retries=1, initial_backoff=0.0, max_backoff=0.0)
        out = await jev_rerank(jev, "q", _pool(3), ASKED_AT)
        assert out.ranked_ids == ["m2", "m1", "m0"]
        assert len(jev.bodies) == 2

    @pytest.mark.asyncio
    async def test_cancellation_is_not_swallowed(self):
        """A client that disconnects cancels the recall; the fallback is for Jev's failures."""
        task = asyncio.create_task(jev_rerank(_Jev(delay=5.0), "q", _pool(3), ASKED_AT))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# --- which banks run it ---------------------------------------------------------------------


@pytest.fixture
def server_config(monkeypatch):
    for name in list(os.environ):
        if "DECISION_MODEL" in name:
            monkeypatch.delenv(name)

    def configure(env: dict[str, str]) -> config_module.HindsightConfig:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        config = config_module.HindsightConfig.from_env()
        monkeypatch.setattr(recall_checks, "get_config", lambda: config)
        return config

    return configure


TYPESAFE = {
    "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_PROVIDER": "typesafe",
    "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_API_KEY": "k",
}


class TestResolve:
    def test_it_is_off_by_default(self, server_config):
        config = server_config(TYPESAFE)
        assert config.recall_jev_rerank is False
        assert resolve_recall_checks({}) is None

    def test_on_it_asks_the_recall_checks_setup_with_the_rank_question(self, server_config):
        server_config({**TYPESAFE, "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_BASE_URL": "http://jev.test/"})
        checks = resolve_recall_checks({"recall_jev_rerank": True})

        assert checks is not None and checks.jev_reranker is not None
        client = checks.jev_reranker.client
        assert isinstance(client, TypeSafeCrossEncoder)
        assert (checks.jev_reranker.model, client.base_url) == ("jev-1.14.0", "http://jev.test")
        # The recall timeout, not the decision model's, and one quick retry inside it.
        assert client.timeout == 2.0
        assert client.retry_policy == RetryPolicy(
            max_retries=1, initial_backoff=0.2, max_backoff=0.2, budget_seconds=2.0
        )
        # The rerank and the other checks of a recall share its guard.
        assert checks.jev_reranker.guard is checks.model.guard
        # The other checks stay off, so nothing is asked about the results afterwards.
        assert (checks.relevance, checks.conditional_recency, checks.injection_gate) == (False, False, False)
        assert checks.on_results is False

    def test_another_check_alone_builds_no_rank_client(self, server_config):
        server_config(TYPESAFE)
        checks = resolve_recall_checks({"recall_relevance_check": True})
        assert checks is not None and checks.jev_reranker is None

    def test_without_a_recall_checks_decision_model_it_is_skipped(self, server_config):
        server_config({})
        assert resolve_recall_checks({"recall_jev_rerank": True}) is None

    def test_the_env_sets_the_server_default(self, server_config):
        config = server_config({**TYPESAFE, "HINDSIGHT_API_RECALL_JEV_RERANK": "true"})
        assert config.recall_jev_rerank is True
        assert "recall_jev_rerank" in config_module.HindsightConfig.get_configurable_fields()
