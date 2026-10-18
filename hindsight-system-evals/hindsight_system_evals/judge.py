"""An independent LLM judge.

Two rules carried over from ``hindsight-api-slim/tests/llm_judge.py``, both
load-bearing:

* **The judge must not be the model under test.** Judging an answer with the same
  model that wrote it measures agreement with itself. It is configured separately
  from the server's provider; the fixtures warn when the two resolve to the same
  model.
* **A single temperature-0 verdict flips on borderline phrasing.** A "not met" is
  re-asked at a higher temperature and upheld only on majority agreement, so one
  noisy sample cannot fail a run. Verdicts that pass first time cost one call.

Three backends, because the environments differ: a developer usually has a Gemini
API key, while the perf workflow authenticates every quality benchmark with a
VertexAI service account; both go through ``google-genai``. A run whose model under
test is Gemini can be judged by Claude instead (``HINDSIGHT_EVAL_JUDGE_PROVIDER=anthropic``),
over plain HTTP. Never through ``hindsight_api``, so nothing here can depend on
engine internals.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from functools import cache

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

_CONFIRMATIONS = int(os.getenv("HINDSIGHT_EVAL_JUDGE_CONFIRMATIONS", "2"))
_CONFIRM_TEMPERATURE = float(os.getenv("HINDSIGHT_EVAL_JUDGE_CONFIRM_TEMPERATURE", "0.5"))
#: Anthropic only: ``output_config.effort`` when set. Grading one criterion needs little
#: reasoning, and on models that think by default the thinking is billed as output.
_EFFORT = os.getenv("HINDSIGHT_EVAL_JUDGE_EFFORT")
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
#: The Claude models that still accept a sampling temperature. Newer ones (Haiku 5.5,
#: Sonnet 5.5, Opus 4.7+, ...) reject a non-default value with a 400, so for them it is
#: omitted and a confirmation re-ask is simply another sample at the default temperature.
_ANTHROPIC_SAMPLING = re.compile(r"^claude-(3|haiku-4|sonnet-4|opus-4($|-[0-6]))")


@dataclass
class JudgeUsage:
    """What the judge spent in this process, so a run can price its grading."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


#: Process-wide; the Anthropic backend reports usage per call (Gemini's is not counted).
USAGE = JudgeUsage()

_SYSTEM = (
    "You grade whether an answer satisfies a stated criterion. Judge ONLY the criterion "
    "given — not style, length, or anything else. Reply with a single JSON object: "
    '{"meets_criteria": true|false, "reasoning": "<one sentence>"}. No prose outside it.'
)


@dataclass(frozen=True)
class Verdict:
    meets_criteria: bool
    reasoning: str


def judge_model() -> str:
    # The perf workflow names Vertex models with a "google/" prefix; the SDK wants
    # the bare name. Accepting both keeps one env convention across benchmarks.
    return os.getenv("HINDSIGHT_EVAL_JUDGE_MODEL", "gemini-2.5-flash").removeprefix("google/")


def _judge_api_key() -> str:
    return os.getenv("HINDSIGHT_EVAL_JUDGE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def _provider() -> str:
    return os.getenv("HINDSIGHT_EVAL_JUDGE_PROVIDER", "gemini" if _judge_api_key() else "vertexai")


class _AnthropicUsage(BaseModel):
    # The cache counters come back null when caching was not involved.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class _AnthropicBlock(BaseModel):
    type: str
    text: str = ""


class _AnthropicReply(BaseModel):
    content: list[_AnthropicBlock] = []
    stop_reason: str | None = None
    usage: _AnthropicUsage = _AnthropicUsage()


def _anthropic_http() -> httpx.AsyncClient:
    # A client per call, not a cached one: pytest-asyncio gives every test its own event
    # loop, and pooled connections from a closed loop fail on reuse.
    key = os.getenv("HINDSIGHT_EVAL_JUDGE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
    if not key:
        raise RuntimeError("An Anthropic judge needs HINDSIGHT_EVAL_JUDGE_API_KEY (or ANTHROPIC_API_KEY).")
    return httpx.AsyncClient(
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        timeout=httpx.Timeout(120.0),
    )


async def _anthropic_text(prompt: str, temperature: float) -> str:
    """One Messages API call; returns the reply's text.

    Raw HTTP rather than the ``anthropic`` SDK: ``httpx`` is already a dependency here,
    and this package keeps its judge dependencies to what it already ships. 429, 5xx
    and connection errors are retried with backoff, as the SDK would.
    """
    model = judge_model()
    body: dict[str, object] = {
        "model": model,
        "max_tokens": 4096,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    if _ANTHROPIC_SAMPLING.match(model):
        body["temperature"] = temperature
    if _EFFORT:
        body["output_config"] = {"effort": _EFFORT}
    for attempt in range(5):
        try:
            async with _anthropic_http() as http:
                response = await http.post(_ANTHROPIC_URL, json=body)
        except httpx.TransportError:
            if attempt == 4:
                raise
            await asyncio.sleep(2**attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 4:
                response.raise_for_status()
            await asyncio.sleep(float(response.headers.get("retry-after") or 2**attempt))
            continue
        response.raise_for_status()
        reply = _AnthropicReply.model_validate(response.json())
        USAGE.calls += 1
        u = reply.usage
        USAGE.input_tokens += (
            (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        )
        USAGE.output_tokens += u.output_tokens or 0
        # A refusal carries no verdict; the empty text fails closed in the parser below.
        if reply.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in reply.content if b.type == "text")
    raise AssertionError("unreachable: the last attempt returns or raises")


@cache
def _client() -> genai.Client:
    """An API-key client when a key is set, otherwise VertexAI from a service account."""
    if _provider() == "gemini":
        key = _judge_api_key()
        if not key:
            raise RuntimeError(
                "The judge needs HINDSIGHT_EVAL_JUDGE_API_KEY (or GEMINI_API_KEY), or "
                "HINDSIGHT_EVAL_JUDGE_PROVIDER=vertexai with a service account. It is configured "
                "separately from the server on purpose — a model grading its own output agrees with itself."
            )
        return genai.Client(api_key=key)

    from google.oauth2 import service_account

    key_file = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_SERVICE_ACCOUNT_KEY") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY", ""
    )
    project = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_PROJECT_ID") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID", ""
    )
    if not (key_file and project):
        raise RuntimeError(
            "A VertexAI judge needs a service-account key file and project id "
            "(HINDSIGHT_EVAL_JUDGE_VERTEXAI_* or the HINDSIGHT_API_LLM_VERTEXAI_* equivalents)."
        )
    credentials = service_account.Credentials.from_service_account_file(
        key_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    location = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_REGION") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_REGION", "us-central1"
    )
    return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)


async def _judge_once(response: str, criteria: str, context: str | None, temperature: float) -> Verdict:
    prompt = (
        f"## Context\n{context}\n\n" if context else ""
    ) + f"## Criterion\n{criteria}\n\n## Answer to grade\n{response}"
    if _provider() == "anthropic":
        text = await _anthropic_text(prompt, temperature)
    else:
        reply = await _client().aio.models.generate_content(
            model=judge_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        text = reply.text or ""
    try:
        parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except ValueError:
        # A judge that returned something unreadable must not silently pass the
        # thing it was asked to check.
        return Verdict(False, f"judge reply was not JSON: {text[:200]}")
    return Verdict(bool(parsed.get("meets_criteria")), str(parsed.get("reasoning", "")))


async def evaluate(response: str, criteria: str, context: str | None = None, *, confirm_met: bool = False) -> Verdict:
    """Grade ``response`` against ``criteria``, smoothing single-call judge noise.

    By default a "not met" is the verdict that gets re-asked, because a failed
    requirement is what fails a run. A *trap* criterion inverts that: there "met"
    is the failure, so ``confirm_met=True`` re-asks a "met" instead. Smoothing the
    wrong side turns noise into a hit — a correct "not met" overruled by two
    lenient re-asks at higher temperature.
    """
    primary = await _judge_once(response, criteria, context, temperature=0.0)
    if primary.meets_criteria != confirm_met or _CONFIRMATIONS <= 0:
        return primary

    confirmations = await asyncio.gather(
        *(_judge_once(response, criteria, context, _CONFIRM_TEMPERATURE) for _ in range(_CONFIRMATIONS)),
        return_exceptions=True,
    )
    verdicts = [primary] + [v for v in confirmations if isinstance(v, Verdict)]
    met = sum(1 for v in verdicts if v.meets_criteria)
    if met > len(verdicts) - met:
        return Verdict(True, f"{met}/{len(verdicts)} judges met the criteria: {primary.reasoning}")
    return Verdict(False, f"{len(verdicts) - met}/{len(verdicts)} judges did not: {primary.reasoning}")
