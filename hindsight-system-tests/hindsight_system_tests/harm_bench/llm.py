"""The answering and judging models, over the Anthropic Messages REST API.

The ``anthropic`` SDK is not a dependency of this package, so calls go through ``httpx`` with
the documented request shape. ``claude-haiku-5-5`` and ``claude-sonnet-5-5`` reject non-default
sampling parameters, so no ``temperature`` is sent and repeats are unseeded. The API key is
read from ``ANTHROPIC_API_KEY`` and only ever placed in a request header.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

from ..context_bench.metrics import approx_tokens
from .model import PROJECT_NAMES, QUERY_TIME, Task

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
ENV_API_KEY = "ANTHROPIC_API_KEY"

ANSWER_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"

#: USD per million tokens: (input, output) up to the long-context threshold, then beyond it.
#: Thinking tokens are billed (and reported by the API) as output tokens.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50, 0.50, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.00, 10.00),
}
LONG_CONTEXT_TOKENS = 100_000

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_ATTEMPTS = 6


class LLMError(Exception):
    """A model call that failed for good (non-retryable status, or retries exhausted)."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Completion(BaseModel):
    model: str
    text: str
    stop_reason: str | None
    usage: Usage
    cost_usd: float
    latency_ms: float
    attempts: int
    request_id: str | None = None


def cost_usd(model: str, usage: Usage) -> float:
    """Price one call. Cache writes cost 1.25x input and reads 0.1x (none are expected here,
    since no request sets ``cache_control``)."""
    if model not in PRICES:
        raise ValueError(f"no price for model {model!r}")
    low_in, low_out, high_in, high_out = PRICES[model]
    prompt = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    price_in, price_out = (high_in, high_out) if prompt > LONG_CONTEXT_TOKENS else (low_in, low_out)
    return (
        usage.input_tokens * price_in
        + usage.cache_creation_input_tokens * price_in * 1.25
        + usage.cache_read_input_tokens * price_in * 0.1
        + usage.output_tokens * price_out
    ) / 1e6


@dataclass(frozen=True)
class ModelSettings:
    model: str
    effort: str | None
    max_tokens: int


class MessagesClient:
    """A thin async Messages API client with retry on 408/409/429/5xx/529 and connection errors."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_s: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise LLMError(f"{ENV_API_KEY} is not set")
        self._http = httpx.AsyncClient(
            timeout=timeout_s,
            transport=transport,
            headers={"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def create(
        self,
        settings: ModelSettings,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        output_config: dict[str, Any] = {}
        if settings.effort:
            output_config["effort"] = settings.effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            body["output_config"] = output_config

        loop = asyncio.get_running_loop()
        started = loop.time()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = await self._http.post(API_URL, json=body)
            except httpx.TransportError as e:
                if attempts >= MAX_ATTEMPTS:
                    raise LLMError(f"{settings.model}: transport error after {attempts} attempts: {e!r}") from e
                await asyncio.sleep(min(60.0, 2.0**attempts))
                continue
            if response.status_code == 200:
                break
            if response.status_code not in RETRY_STATUSES or attempts >= MAX_ATTEMPTS:
                raise LLMError(f"{settings.model}: HTTP {response.status_code}: {response.text[:500]}")
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2.0**attempts
            logger.warning("%s: HTTP %d, retrying in %.0fs", settings.model, response.status_code, delay)
            await asyncio.sleep(min(60.0, delay))

        payload = response.json()
        usage = Usage.model_validate(payload.get("usage") or {})
        text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
        return Completion(
            model=settings.model,
            text=text,
            stop_reason=payload.get("stop_reason"),
            usage=usage,
            cost_usd=cost_usd(settings.model, usage),
            latency_ms=round((loop.time() - started) * 1000, 1),
            attempts=attempts,
            request_id=response.headers.get("request-id"),
        )


# ---------------------------------------------------------------------------
# The answer prompt
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are an AI coding assistant pairing with a software engineer at Acme. They are working in the "
    "{repository} repository ({project}). Answer their question directly: lead with the most likely cause or the "
    "decision, then give the concrete fix, commands or steps. Keep it under 250 words."
)

#: The Claude Code integration's own preamble and block shape (hindsight-integrations/claude-code:
#: settings.json ``recallPromptPreamble``, scripts/recall.py), so the injected context is what a
#: real agent sees.
MEMORY_PREAMBLE = (
    "Relevant memories from past conversations (prioritize recent when conflicting). Only use memories that are "
    "directly useful to continue this conversation; ignore the rest:"
)


class InjectedMemory(BaseModel):
    label: str
    text: str
    type: str | None = None
    mentioned_at: str | None = None


def format_memories(memories: list[InjectedMemory]) -> str:
    """``- <text> [<type>] (<mentioned_at>)`` per memory, blank-line separated (as the integration does)."""
    lines = []
    for memory in memories:
        type_str = f" [{memory.type}]" if memory.type else ""
        date_str = f" ({memory.mentioned_at})" if memory.mentioned_at else ""
        lines.append(f"- {memory.text}{type_str}{date_str}")
    return "\n\n".join(lines)


def memory_block(memories: list[InjectedMemory], *, preamble: bool = True) -> str:
    """The injected context; empty when nothing was admitted (the integration injects nothing then).
    ``preamble=False`` drops only the integration's preamble line, leaving the block otherwise intact."""
    if not memories:
        return ""
    current_time = QUERY_TIME.strftime("%Y-%m-%d %H:%M UTC")
    head = f"{MEMORY_PREAMBLE}\n" if preamble else ""
    return (
        f"<hindsight_memories>\n{head}Current time - {current_time}\n\n"
        f"{format_memories(memories)}\n</hindsight_memories>"
    )


def answer_prompt(task: Task, memories: list[InjectedMemory], *, preamble: bool = True) -> tuple[str, str]:
    """(system, user) for one answer. Without memories the user turn is the task alone."""
    system = ANSWER_SYSTEM.format(repository=task.repository, project=PROJECT_NAMES[task.project])
    block = memory_block(memories, preamble=preamble)
    user = f"{block}\n\n{task.text}" if block else task.text
    return system, user


def injected_tokens(memories: list[InjectedMemory], *, preamble: bool = True) -> int:
    """Approximate tokens the memory block adds (context_bench's tokenizer-free count)."""
    block = memory_block(memories, preamble=preamble)
    return approx_tokens(block) if block else 0


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You grade an AI assistant's answer to a software engineering task against a rubric. Grade only what the "
    "answer says; do not reward length or hedging. A criterion counts as met when the answer states it clearly "
    "and correctly, in any wording. An answer that lists several possible causes without committing to the "
    "right one as the most likely does not meet a 'identifies the cause' criterion."
)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "harm": {"type": "boolean"},
        "unmet": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["success", "harm", "unmet", "reason"],
    "additionalProperties": False,
}


class Verdict(BaseModel):
    success: bool
    harm: bool
    unmet: list[str]
    reason: str


def judge_prompt(task: Task, answer: str) -> str:
    must = "\n".join(f"{i}. {criterion}" for i, criterion in enumerate(task.must_include, 1))
    must_not = "\n".join(f"- {criterion}" for criterion in task.must_not) or "- (none)"
    if task.harm_signature:
        harm = (
            "HARM: set harm=true if the answer's diagnosis, main recommendation or first step follows this wrong "
            f"lead: {task.harm_signature} Mentioning it only to rule it out, or as a clearly secondary afterthought "
            "after the right answer, is not harm."
        )
    else:
        harm = "HARM: this task has no designated wrong lead; always set harm=false."
    return (
        f"<task>\n{task.text}\n</task>\n\n"
        f"<reference_answer>\n{task.reference}\n</reference_answer>\n\n"
        f"<rubric>\nSUCCESS requires every MUST criterion met and no MUST-NOT criterion met.\n"
        f"MUST:\n{must}\nMUST-NOT:\n{must_not}\n{harm}\n</rubric>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        "Return success, harm, unmet (the MUST criteria not met and the MUST-NOT criteria violated, quoted briefly; "
        "empty when none) and reason (one or two sentences)."
    )


def parse_verdict(task: Task, text: str) -> Verdict:
    """The judge's JSON, with ``harm`` forced false for a task without a harm signature."""
    try:
        verdict = Verdict.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMError(f"unparseable verdict for {task.id}: {text[:300]!r}") from e
    if not task.harm_signature and verdict.harm:
        verdict = verdict.model_copy(update={"harm": False})
    return verdict
