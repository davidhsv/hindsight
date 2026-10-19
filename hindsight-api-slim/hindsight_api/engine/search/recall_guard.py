"""Fast failure for the decision-model requests a recall waits on: the recall checks.

A recall check (``recall_checks.py``, ``jev_rerank.py``) already falls back when its request fails:
the gate returns the memories unchecked, the rerank keeps the local order, the flag is left unset
and recency stays as configured. What made an outage expensive was the wait before that fallback:
the decision model's own timeout and retry budget, sized for consolidation and reflect, held every
recall for about 16 s while the endpoint was unreachable. Three bounds keep that wait short:

- **A recall timeout** (``HINDSIGHT_API_RECALL_CHECKS_TIMEOUT``) on each request, its one retry
  included.
- **One outage per recall.** Once a request of a recall gets no answer, that recall asks nothing
  more, so a recall waits out the timeout at most about once (requests already in flight together,
  like the gate's, end within the same timeout).
- **A process-wide breaker** (:class:`CircuitBreaker`). After
  ``HINDSIGHT_API_RECALL_CHECKS_BREAKER_FAILURES`` requests in a row get no answer it opens, and
  for ``HINDSIGHT_API_RECALL_CHECKS_BREAKER_COOLDOWN`` seconds no recall sends a request: each
  check falls back at once. After the cool-down one request probes the endpoint: an answer closes
  the breaker, another failure opens it for a new cool-down. It logs one warning when it opens and
  one when it closes, and the checks it skips log nothing of their own.

"No answer" is the endpoint being unavailable, not a bad answer: a connection error, a timeout, a
5xx, a 429, or a key it refuses or has no credit left on (401, 402, 403). Any other response,
including a 400 or an answer out of shape, shows the endpoint is up, so it counts as answered.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..cross_encoder import TypeSafeCrossEncoder
from ..decision_model import DecisionModel, DecisionQuestion, DecisionResponse
from ..remote_retry import is_transient_remote_error, status_code_of

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Statuses that are not transient but still mean every request will fail the same way: a key the
# endpoint refuses, or an account out of credit. Retrying them is pointless; asking again on the
# next recall is too.
_UNUSABLE_KEY_STATUSES = frozenset({401, 402, 403})


def is_unavailable(error: BaseException) -> bool:
    """Whether ``error`` means the endpoint did not answer (see the module docstring)."""
    return is_transient_remote_error(error) or status_code_of(error) in _UNUSABLE_KEY_STATUSES


class RecallCheckSkipped(Exception):
    """A recall check was not asked: the breaker is open, or an earlier request of the same recall
    got no answer. The check falls back as on a failure, without a warning of its own."""


class Admission(enum.Enum):
    """What :meth:`CircuitBreaker.admit` lets a request do."""

    #: Closed: send it.
    SEND = "send"
    #: Open, cool-down over: send it as the one probe.
    PROBE = "probe"
    #: Open: do not send it.
    DENY = "deny"


class CircuitBreaker:
    """Stops requests to an endpoint that keeps failing, for a cool-down, then probes it once.

    Closed, every request is sent, and ``failure_threshold`` unavailable requests in a row open
    it. Open, :meth:`admit` denies every request until ``cooldown_seconds`` have passed; then it
    admits exactly one as the probe and denies the rest while the probe is out. The probe's answer
    closes the breaker; its failure opens it for a new cool-down. Any answered request closes it,
    including one sent before it opened.

    One instance is shared by every recall in the process, from every event loop (worker threads
    run their own), so its state is guarded by a ``threading.Lock`` over await-free sections.
    """

    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        name: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.name = name
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        #: When it last opened; None while closed.
        self._opened_at: float | None = None
        self._probing = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None

    def admit(self) -> Admission:
        with self._lock:
            if self._opened_at is None:
                return Admission.SEND
            if self._probing or self._clock() - self._opened_at < self.cooldown_seconds:
                return Admission.DENY
            self._probing = True
            return Admission.PROBE

    def record_answer(self) -> None:
        with self._lock:
            was_open = self._opened_at is not None
            self._failures, self._opened_at, self._probing = 0, None, False
        if was_open:
            logger.warning(f"{self.name}: the decision model answers again, the circuit breaker is closed")

    def record_unavailable(self, admission: Admission) -> None:
        with self._lock:
            self._failures += 1
            failures = self._failures
            if admission is Admission.PROBE and self._probing:
                # A failed probe opens it for a new cool-down; the warning was logged when it opened.
                self._probing = False
                self._opened_at = self._clock()
                opened = False
            else:
                opened = self._opened_at is None and failures >= self.failure_threshold
                if opened:
                    self._opened_at = self._clock()
        if opened:
            logger.warning(
                f"{self.name}: {failures} decision-model requests in a row got no answer; the circuit breaker is "
                f"open and recalls skip the checks for {self.cooldown_seconds:g}s at a time, until the decision "
                "model answers again"
            )
        elif admission is Admission.PROBE:
            logger.debug(f"{self.name}: the probe got no answer, the circuit breaker stays open")

    def release(self, admission: Admission) -> None:
        """Forget a request that ended without an answer either way (it was cancelled), so a probe
        that never finished does not keep the breaker from probing again."""
        if admission is Admission.PROBE:
            with self._lock:
                self._probing = False


class RecallGuard:
    """One recall's way to the recall checks' decision model: the shared breaker, the recall
    timeout on each request, and this recall's own record of a request that got no answer.

    Built per recall; every check of the recall goes through the same guard, on the recall's loop.
    """

    def __init__(self, breaker: CircuitBreaker, timeout: float) -> None:
        self.breaker = breaker
        self.timeout = timeout
        self._unavailable = False

    async def call(self, request: Callable[[], Awaitable[T]]) -> T:
        """``request()`` within the recall timeout, or :class:`RecallCheckSkipped` without sending
        it when the breaker is open or this recall already met an outage."""
        if self._unavailable:
            raise RecallCheckSkipped("an earlier request of this recall got no answer")
        admission = self.breaker.admit()
        if admission is Admission.DENY:
            raise RecallCheckSkipped("the circuit breaker is open")
        try:
            async with asyncio.timeout(self.timeout):
                result = await request()
        except asyncio.CancelledError:
            self.breaker.release(admission)
            raise
        except Exception as e:
            if is_unavailable(e):
                self._unavailable = True
                self.breaker.record_unavailable(admission)
            else:
                self.breaker.record_answer()
            raise
        self.breaker.record_answer()
        return result


class GuardedDecisionModel:
    """The recall checks' :class:`DecisionModel`, asked through one recall's guard."""

    def __init__(self, client: DecisionModel, guard: RecallGuard) -> None:
        self.client = client
        self.guard = guard

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def max_request_tokens(self) -> int:
        return self.client.max_request_tokens

    async def ask(self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]) -> DecisionResponse:
        return await self.guard.call(lambda: self.client.ask(state, questions))


class GuardedRanker:
    """The Jev rerank's :class:`TypeSafeCrossEncoder`, asked through one recall's guard."""

    def __init__(self, client: TypeSafeCrossEncoder, guard: RecallGuard) -> None:
        self.client = client
        self.guard = guard

    @property
    def model(self) -> str:
        return self.client.model

    async def rank(self, query: str, docs: list[str], *, current_date: str | None = None) -> list[int]:
        return await self.guard.call(lambda: self.client.rank(query, docs, current_date=current_date))
