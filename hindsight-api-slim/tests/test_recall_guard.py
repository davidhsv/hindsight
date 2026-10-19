"""Fast failure for the recall checks: the recall timeout, one outage per recall, and the breaker.

Unit tests over fake clients and a mocked clock. What each check asks is in test_recall_checks.py,
the Jev rerank's request in test_jev_rerank.py, and the engine wiring in
test_recall_checks_engine.py.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import aiohttp
import pytest

from hindsight_api import config as config_module
from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
from hindsight_api.engine.decision_model import DecisionAnswer, DecisionResponse
from hindsight_api.engine.remote_retry import RetryPolicy
from hindsight_api.engine.response_models import MemoryFact, RecallResult
from hindsight_api.engine.search import recall_checks
from hindsight_api.engine.search.jev_rerank import jev_rerank
from hindsight_api.engine.search.recall_checks import (
    RecallChecks,
    apply_result_checks,
    asks_current_state,
    nothing_relevant,
    planted_memory_ids,
    recency_decay_for,
    resolve_recall_checks,
)
from hindsight_api.engine.search.recall_guard import (
    Admission,
    CircuitBreaker,
    GuardedDecisionModel,
    GuardedRanker,
    RecallCheckSkipped,
    RecallGuard,
    is_unavailable,
)
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult, ScoredResult

ASKED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
URL = "https://jev.test/v1/systemone"
OPENED = "the circuit breaker is open"
CLOSED = "the circuit breaker is closed"


class _Clock:
    """The breaker's mocked clock: time moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _Endpoint:
    """Stands in for both recall-check clients: ``ask`` for the decision model, ``rank`` for the
    Jev rerank. Each request is recorded, then waits ``delay`` seconds (real time) and raises
    ``fail``, or answers: every noul 0.01 (nothing planted, nothing helps less than flagged), and the
    rank as the order it was sent."""

    model = "jev-1.14.0"
    max_request_tokens = 26_000

    def __init__(self, *, fail: BaseException | None = None, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.requests: list[str] = []

    async def _answer(self, kind: str) -> None:
        self.requests.append(kind)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail

    async def ask(self, state, questions):
        await self._answer("ask")
        return DecisionResponse(
            answers={key: DecisionAnswer(noul=0.99 if key == "flag" else 0.01) for key in questions}
        )

    async def rank(self, query, docs, *, current_date=None):
        await self._answer("rank")
        return list(range(len(docs)))


def _breaker(clock: _Clock | None = None, *, failures: int = 3, cooldown: float = 30.0) -> CircuitBreaker:
    return CircuitBreaker(
        failure_threshold=failures, cooldown_seconds=cooldown, name="Recall checks", clock=clock or _Clock()
    )


def _opened(clock: _Clock | None = None, *, cooldown: float = 30.0) -> CircuitBreaker:
    breaker = _breaker(clock, failures=1, cooldown=cooldown)
    breaker.record_unavailable(breaker.admit())
    return breaker


def _checks(endpoint: _Endpoint, guard: RecallGuard) -> RecallChecks:
    """Every check on, asked of ``endpoint`` through one recall's ``guard``."""
    return RecallChecks(
        model=GuardedDecisionModel(endpoint, guard),  # type: ignore[arg-type]
        relevance=True,
        conditional_recency=True,
        injection_gate=True,
        jev_reranker=GuardedRanker(endpoint, guard),  # type: ignore[arg-type]
    )


def _fact(fact_id: str) -> MemoryFact:
    return MemoryFact(id=fact_id, text=f"memory {fact_id}", fact_type="world")


def _scored(index: int) -> ScoredResult:
    retrieval = RetrievalResult(id=f"m{index}", text=f"memory m{index}", fact_type="world", mentioned_at=ASKED_AT)
    return ScoredResult(
        candidate=MergedCandidate(retrieval=retrieval, rrf_score=0.0),
        cross_encoder_score=float(-index),
        cross_encoder_score_normalized=1.0 / (index + 2),
        weight=1.0 / (index + 1),
    )


def _local_result() -> RecallResult:
    return RecallResult(results=[_fact(str(i)) for i in range(12)])


@dataclass(frozen=True)
class _RecallOutcome:
    """What a recall's checks left: the rerank's order, the recency decay, and the result."""

    order: list[str]
    decay: str
    result: RecallResult


async def _recall(checks: RecallChecks) -> _RecallOutcome:
    """The checks in the order a recall runs them: the routing question beside retrieval, the
    rerank, then the gate and the flag on the result."""
    route = asyncio.create_task(asks_current_state(checks.model, "What does Alice work on now?"))
    assert checks.jev_reranker is not None
    jev = await jev_rerank(checks.jev_reranker, "q", [_scored(i) for i in range(12)], ASKED_AT)
    decay = await recency_decay_for(route, "linear")
    result = await apply_result_checks(checks, "q", _local_result(), "bank")
    return _RecallOutcome([sr.id for sr in jev.results], decay, result)


LOCAL_ORDER = [f"m{i}" for i in range(12)]


def _warnings(caplog: pytest.LogCaptureFixture, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records if record.levelname == "WARNING")


# --- what counts as no answer ----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        aiohttp.ClientConnectionError("Cannot connect to host 127.0.0.1:9"),
        ConnectionRefusedError(),
        UpstreamHTTPError(503, "unavailable", {}, URL),
        UpstreamHTTPError(500, "internal", {}, URL),
        UpstreamHTTPError(429, "slow down", {}, URL),
        UpstreamHTTPError(402, "out of credit", {}, URL),
        UpstreamHTTPError(401, "bad key", {}, URL),
        UpstreamHTTPError(403, "forbidden", {}, URL),
    ],
)
def test_an_endpoint_that_cannot_answer_is_unavailable(error):
    assert is_unavailable(error)


@pytest.mark.parametrize(
    "error",
    [
        UpstreamHTTPError(400, "criteria must map 2 or more options", {}, URL),
        ValueError("out of shape"),
        KeyError("c0"),
    ],
)
def test_a_response_the_check_cannot_use_still_shows_the_endpoint_is_up(error):
    assert not is_unavailable(error)


# --- the breaker ------------------------------------------------------------------------------


class TestBreaker:
    def test_it_opens_after_n_requests_in_a_row_get_no_answer_and_warns_once(self, caplog):
        breaker = _breaker(failures=3)
        with caplog.at_level("WARNING"):
            for _ in range(2):
                breaker.record_unavailable(breaker.admit())
            assert breaker.admit() is Admission.SEND
            breaker.record_unavailable(Admission.SEND)
            assert breaker.admit() is Admission.DENY
            # A request that was already out when it opened fails too: no second warning.
            breaker.record_unavailable(Admission.SEND)
        assert _warnings(caplog, OPENED) == 1
        assert "3 decision-model requests in a row got no answer" in caplog.text

    def test_an_answer_in_between_starts_the_count_again(self):
        breaker = _breaker(failures=3)
        for _ in range(2):
            breaker.record_unavailable(breaker.admit())
        breaker.record_answer()
        for _ in range(2):
            breaker.record_unavailable(breaker.admit())
        assert breaker.admit() is Admission.SEND

    def test_open_it_sends_nothing_until_the_cool_down_is_over(self):
        clock = _Clock()
        breaker = _opened(clock, cooldown=30.0)
        clock.now += 29.9
        assert breaker.admit() is Admission.DENY
        clock.now += 0.1
        assert breaker.admit() is Admission.PROBE
        # One probe at a time: the rest wait for its answer.
        assert breaker.admit() is Admission.DENY

    def test_a_probe_that_answers_closes_it(self, caplog):
        clock = _Clock()
        breaker = _opened(clock)
        clock.now += 30.0
        with caplog.at_level("WARNING"):
            admission = breaker.admit()
            assert admission is Admission.PROBE
            breaker.record_answer()
            assert breaker.admit() is Admission.SEND
            breaker.record_answer()
        assert _warnings(caplog, CLOSED) == 1
        assert not breaker.is_open

    def test_a_probe_that_fails_opens_it_for_a_new_cool_down(self, caplog):
        clock = _Clock()
        breaker = _opened(clock, cooldown=30.0)
        clock.now += 30.0
        caplog.clear()
        with caplog.at_level("WARNING"):
            breaker.record_unavailable(breaker.admit())
        assert _warnings(caplog, OPENED) == 0, "it was already open: no second warning"
        clock.now += 29.9
        assert breaker.admit() is Admission.DENY
        clock.now += 0.1
        assert breaker.admit() is Admission.PROBE

    def test_a_cancelled_probe_lets_the_next_request_probe(self):
        clock = _Clock()
        breaker = _opened(clock)
        clock.now += 30.0
        breaker.release(breaker.admit())
        assert breaker.admit() is Admission.PROBE


# --- one recall's guard -----------------------------------------------------------------------


class TestGuard:
    async def test_a_request_past_the_timeout_fails_and_counts_as_no_answer(self):
        breaker = _breaker(failures=1)
        guard = RecallGuard(breaker, timeout=0.05)
        endpoint = _Endpoint(delay=5.0)
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await GuardedDecisionModel(endpoint, guard).ask("state", {})  # type: ignore[arg-type]
        assert asyncio.get_running_loop().time() - started < 1.0
        assert breaker.is_open

    async def test_after_a_request_gets_no_answer_the_recall_asks_nothing_more(self):
        breaker = _breaker(failures=5)
        endpoint = _Endpoint(fail=aiohttp.ClientConnectionError("refused"))
        model = GuardedDecisionModel(endpoint, RecallGuard(breaker, timeout=2.0))  # type: ignore[arg-type]
        with pytest.raises(aiohttp.ClientConnectionError):
            await model.ask("state", {})
        with pytest.raises(RecallCheckSkipped):
            await model.ask("state", {})
        assert endpoint.requests == ["ask"]
        assert not breaker.is_open, "one failure is not an outage of the endpoint"
        # The next recall has its own guard, and asks.
        with pytest.raises(aiohttp.ClientConnectionError):
            await GuardedDecisionModel(endpoint, RecallGuard(breaker, timeout=2.0)).ask("state", {})  # type: ignore[arg-type]
        assert endpoint.requests == ["ask", "ask"]

    async def test_a_response_the_check_cannot_use_does_not_stop_the_recall(self):
        breaker = _breaker(failures=1)
        endpoint = _Endpoint(fail=UpstreamHTTPError(400, "bad request", {}, URL))
        model = GuardedDecisionModel(endpoint, RecallGuard(breaker, timeout=2.0))  # type: ignore[arg-type]
        for _ in range(2):
            with pytest.raises(UpstreamHTTPError):
                await model.ask("state", {})
        assert endpoint.requests == ["ask", "ask"]
        assert not breaker.is_open

    async def test_a_cancelled_probe_is_cancelled_and_frees_the_probe(self):
        clock = _Clock()
        breaker = _opened(clock)
        clock.now += 30.0
        endpoint = _Endpoint(delay=5.0)
        task = asyncio.create_task(GuardedDecisionModel(endpoint, RecallGuard(breaker, 2.0)).ask("state", {}))  # type: ignore[arg-type]
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert breaker.admit() is Admission.PROBE


# --- each check falls back as on a failure ------------------------------------------------------


class TestFallbacks:
    async def test_past_the_timeout_each_check_falls_back(self):
        """Each check on its own recall (a fresh guard), so each waits out its own timeout."""
        endpoint = _Endpoint(delay=5.0)

        def model() -> GuardedDecisionModel:
            return GuardedDecisionModel(endpoint, RecallGuard(_breaker(), timeout=0.05))  # type: ignore[arg-type]

        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await asks_current_state(model(), "q") is None
        assert await nothing_relevant(model(), "q", [_fact("a")]) is None
        assert await planted_memory_ids(model(), "q", [_fact("a"), _fact("b")]) == set()
        ranker = GuardedRanker(endpoint, RecallGuard(_breaker(), timeout=0.05))  # type: ignore[arg-type]
        jev = await jev_rerank(ranker, "q", [_scored(i) for i in range(3)], ASKED_AT)
        assert ([sr.id for sr in jev.results], jev.error) == (["m0", "m1", "m2"], "TimeoutError")
        assert loop.time() - started < 2.0
        assert endpoint.requests == ["ask", "ask", "ask", "rank"]

    async def test_a_skipped_check_falls_back_without_a_warning(self, caplog):
        endpoint = _Endpoint()
        checks = _checks(endpoint, RecallGuard(_opened(), timeout=2.0))
        caplog.clear()
        with caplog.at_level("WARNING"):
            outcome = await _recall(checks)
        assert endpoint.requests == []
        assert (outcome.order, outcome.decay) == (LOCAL_ORDER, "linear")
        assert outcome.result == _local_result()
        assert caplog.records == []


# --- an outage, recall by recall ----------------------------------------------------------------


async def test_a_dead_endpoint_opens_the_breaker_and_a_recall_then_costs_what_the_local_path_does(caplog):
    """The endpoint refuses every connection. Each recall stops at its first failed request; the
    second failure opens the breaker, and the recalls after it send nothing and take no longer than
    a recall without the checks. After the cool-down one probe finds the endpoint back and closes it."""
    clock = _Clock()
    breaker = _breaker(clock, failures=2, cooldown=30.0)
    endpoint = _Endpoint(fail=aiohttp.ClientConnectionError("Cannot connect to host 127.0.0.1:9"))
    loop = asyncio.get_running_loop()

    with caplog.at_level("WARNING"):
        for expected_requests in (1, 2):
            outcome = await _recall(_checks(endpoint, RecallGuard(breaker, timeout=2.0)))
            assert len(endpoint.requests) == expected_requests, "a recall stops at its first failed request"
            assert (outcome.order, outcome.decay, outcome.result) == (LOCAL_ORDER, "linear", _local_result())
        assert breaker.is_open

        # Open: had a request gone out, each would have cost a second.
        endpoint.fail, endpoint.delay = None, 1.0
        clock.now += 29.0
        started = loop.time()
        local = _local_result()
        local_elapsed = loop.time() - started
        started = loop.time()
        outcome = await _recall(_checks(endpoint, RecallGuard(breaker, timeout=2.0)))
        checked_elapsed = loop.time() - started
        assert len(endpoint.requests) == 2, "nothing is sent while the breaker is open"
        assert (outcome.order, outcome.decay, outcome.result) == (LOCAL_ORDER, "linear", local)
        assert checked_elapsed < local_elapsed + 0.25

        # The cool-down is over: the recall's first request probes, finds the endpoint back, closes it.
        endpoint.delay = 0.0
        clock.now += 1.0
        outcome = await _recall(_checks(endpoint, RecallGuard(breaker, timeout=2.0)))
        assert not breaker.is_open
        # The rank goes first and is the probe; then the routing question, the gate's two
        # requests (ten memories each) and the flag.
        assert endpoint.requests[2:] == ["rank", "ask", "ask", "ask", "ask"]
        assert outcome.result.nothing_relevant is False

    assert _warnings(caplog, OPENED) == 1
    assert _warnings(caplog, CLOSED) == 1


# --- configuration ------------------------------------------------------------------------------


@pytest.fixture
def server_config(monkeypatch):
    for name in list(os.environ):
        if "DECISION_MODEL" in name or name.startswith("HINDSIGHT_API_RECALL_CHECKS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_PROVIDER", "typesafe")
    monkeypatch.setenv("HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_API_KEY", "k")

    def configure(env: dict[str, str]) -> config_module.HindsightConfig:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        config = config_module.HindsightConfig.from_env()
        monkeypatch.setattr(recall_checks, "get_config", lambda: config)
        return config

    return configure


class TestConfig:
    def test_defaults_are_server_level(self, server_config):
        config = server_config({})
        assert (
            config.recall_checks_timeout,
            config.recall_checks_breaker_failures,
            config.recall_checks_breaker_cooldown,
        ) == (2.0, 5, 30.0)
        configurable = config_module.HindsightConfig.get_configurable_fields()
        assert not {"recall_checks_timeout", "recall_checks_breaker_failures", "recall_checks_breaker_cooldown"} & (
            configurable
        )
        # The decision model's own timeout is untouched: consolidation and reflect keep it.
        assert config.decision_model_config("recall_checks").timeout == 60.0  # type: ignore[union-attr]

    def test_each_recall_gets_its_own_guard_over_the_one_breaker(self, server_config):
        server_config(
            {
                "HINDSIGHT_API_RECALL_CHECKS_TIMEOUT": "0.8",
                "HINDSIGHT_API_RECALL_CHECKS_BREAKER_FAILURES": "2",
                "HINDSIGHT_API_RECALL_CHECKS_BREAKER_COOLDOWN": "5",
            }
        )
        flags = {"recall_injection_gate": True, "recall_jev_rerank": True}
        first, second = resolve_recall_checks(flags), resolve_recall_checks(flags)
        assert first is not None and second is not None
        assert first.jev_reranker is not None and first.jev_reranker.guard is first.model.guard
        assert first.model.guard is not second.model.guard
        breaker = first.model.guard.breaker
        assert breaker is second.model.guard.breaker
        assert (breaker.failure_threshold, breaker.cooldown_seconds, first.model.guard.timeout) == (2, 5.0, 0.8)
        # Both clients time out at the recall timeout and retry once inside it.
        bounded = RetryPolicy(max_retries=1, initial_backoff=0.2, max_backoff=0.2, budget_seconds=0.8)
        assert first.model.client.retry_policy == bounded
        assert (first.jev_reranker.client.timeout, first.jev_reranker.client.retry_policy) == (0.8, bounded)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("HINDSIGHT_API_RECALL_CHECKS_TIMEOUT", "0"),
            ("HINDSIGHT_API_RECALL_CHECKS_TIMEOUT", "-1"),
            ("HINDSIGHT_API_RECALL_CHECKS_TIMEOUT", "soon"),
            ("HINDSIGHT_API_RECALL_CHECKS_BREAKER_FAILURES", "0"),
            ("HINDSIGHT_API_RECALL_CHECKS_BREAKER_COOLDOWN", "0"),
        ],
    )
    def test_a_value_that_would_disable_the_checks_fails_startup(self, server_config, name, value):
        with pytest.raises(ValueError, match=name):
            server_config({name: value})
