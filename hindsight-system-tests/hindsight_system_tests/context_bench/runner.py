"""Run the context_pollution queries against one server build and record every result.

One run = one server profile (a checkout plus server env) and a list of request
*variants*. Each variant is a recall request shape, sent once per query:

- ``plain`` — the query as an agent sends it today;
- ``tags`` — plus the task's project as a hard tag scope (``tags_match=any``, so the
  user's own untagged memories stay reachable);
- ``final@0.30`` / ``reranker@0.30`` — plus a ``min_scores`` floor on that stage;
- components join with ``+`` (``tags+final@0.30``); ``#2`` repeats a variant (determinism).

With ``send_context`` every request also carries the A49 ``context`` object and ``purpose``,
which a server without the admission layer ignores (FastAPI drops unknown fields).

Rows keep labels and scores, not judgments: the report applies client-side gates and
metrics afterwards, so a new gate never needs a new server run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from ..rulebook import Stubs
from ..server import ENV_API_DIR, ENV_DATABASE_URL, free_port, start_hindsight_server, start_stub_server
from .metrics import approx_tokens, percentile
from .model import MAIN_BANK, QUERY_TIME, SESSION_BANK, Query
from .scenarios import LABEL_BY_DOCUMENT, MEMORIES, QUERIES, split_of

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The shipping local models pinned to CPU (an MPS kernel may reorder float reductions
#: between runs), read from the local HF cache so a run never depends on the network.
LOCAL_MODEL_ENV = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_RERANKER_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU": "true",
    "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU": "true",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

#: ``none`` forces chunks-mode retain and disables observations server-wide (config.py), so
#: every retained document becomes exactly one memory with the text we wrote, and nothing
#: needs an LLM. ``UV_NO_SYNC`` keeps ``uv run`` from re-syncing the checkout's venv.
BASE_ENV = {"HINDSIGHT_API_LLM_PROVIDER": "none", "UV_NO_SYNC": "1"}

TYPESAFE_UPSTREAM = "https://api.typesafe.ai"

#: Server env whose values are never written to a run file.
_SECRET = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)", re.IGNORECASE)

DEFAULT_PG0 = "hindsight-ctxbench:15529"
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class BenchmarkError(Exception):
    """The scenario could not be set up as written; its numbers would mean nothing."""


# ---------------------------------------------------------------------------
# Request variants
# ---------------------------------------------------------------------------


def base_body(query: Query, *, max_tokens: int, send_context: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": query.text,
        "budget": "mid",
        "max_tokens": max_tokens,
        "query_timestamp": QUERY_TIME.strftime("%Y-%m-%dT%H:%M:%S"),
        "include": {"entities": None},
    }
    if send_context:
        body["context"] = query.context()
        body["purpose"] = query.purpose
    return body


def variant_body(variant: str, query: Query, *, max_tokens: int, send_context: bool) -> dict[str, Any]:
    """The recall body a variant sends for one query (see the module docstring)."""
    body = base_body(query, max_tokens=max_tokens, send_context=send_context)
    floors: dict[str, float] = {}
    for part in variant.split("#", 1)[0].split("+"):
        if part == "plain":
            continue
        if part == "tags":
            if query.project is not None:
                body["tags"] = [f"project:{query.project}"]
                body["tags_match"] = "any"
            continue
        stage, sep, value = part.partition("@")
        if sep and stage in ("final", "reranker", "semantic", "keyword"):
            floors[stage] = float(value)
            continue
        raise ValueError(f"unknown variant component {part!r} in {variant!r}")
    if floors:
        body["min_scores"] = floors
    return body


def sweep_variants(prefix: str, stage: str) -> list[str]:
    head = f"{prefix}+" if prefix != "plain" else ""
    return [f"{head}{stage}@{t:.2f}" for t in SWEEP]


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    label: str
    """Scenario label; ``?<memory id>`` for a memory the scenario did not write."""
    server_rank: int
    final: float | None = None
    reranker: float | None = None
    semantic: float | None = None
    keyword: float | None = None


class QueryRow(BaseModel):
    variant: str
    query_id: str
    split: str
    latency_ms: float
    status: int
    results: list[ResultRow]
    error: str | None = None


class TypeSafeCall(BaseModel):
    question: str
    options: int
    approx_request_tokens: int
    status: int
    latency_ms: float
    attempts: int
    usage: dict[str, Any] | None = None
    depth_score: float | None = None


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    label: str
    api_dir: str
    api_sha: str | None
    api_dirty: bool | None
    harness_sha: str | None
    server_env: dict[str, str]
    variants: list[str]
    send_context: bool
    max_tokens: int
    database_url: str
    database_schema: str
    banks: dict[str, str]
    bank_sizes: dict[str, int]
    started_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    rows: list[QueryRow] = Field(default_factory=list)
    typesafe_calls: list[TypeSafeCall] = Field(default_factory=list)
    unlabelled_texts: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    label: str
    api_dir: Path
    variants: tuple[str, ...]
    out_dir: Path
    server_env: Mapping[str, str] = field(default_factory=dict)
    local_models: bool = True
    typesafe_proxy: bool = False
    send_context: bool = False
    max_tokens: int = 4096
    pg0: str = DEFAULT_PG0
    reuse: Path | None = None
    """A previous run file whose schema and banks this run queries instead of ingesting."""
    query_ids: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# TypeSafe usage proxy
# ---------------------------------------------------------------------------


class _TypeSafeProxy:
    """Forwards ``/v1/systemone`` to TypeSafe and records each call's size, status and any
    ``usage`` block, so a run reports what it spent. The Authorization header is forwarded
    untouched and never recorded. 429 and 5xx are retried with exponential backoff, since
    the server's reranker runs with retries off and one rate-limit would fail a recall."""

    MAX_ATTEMPTS = 5

    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.calls: list[TypeSafeCall] = []
        self._lock = threading.Lock()
        app = FastAPI()
        app.add_api_route("/v1/systemone", self._forward, methods=["POST"])
        self.port = free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="typesafe-proxy")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("typesafe proxy did not start")
            time.sleep(0.05)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    async def _forward(self, request: Request) -> Response:
        payload = await request.body()
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() in ("authorization", "content-type") or name.lower().startswith("x-")
        }
        started = time.monotonic()
        attempts = 0
        async with httpx.AsyncClient(timeout=180) as client:
            while True:
                attempts += 1
                upstream = await client.post(f"{self.upstream}/v1/systemone", content=payload, headers=headers)
                if upstream.status_code not in (429, 500, 502, 503, 504) or attempts >= self.MAX_ATTEMPTS:
                    break
                await asyncio.sleep(min(30.0, 2.0**attempts))
        self._record(payload, upstream, attempts, (time.monotonic() - started) * 1000)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    def _record(self, payload: bytes, upstream: httpx.Response, attempts: int, latency_ms: float) -> None:
        question, options = "unknown", 0
        with contextlib.suppress(ValueError, AttributeError, StopIteration):
            name, spec = next(iter(json.loads(payload)["questions"].items()))
            question = f"{name}:{spec.get('type')}"
            criteria = spec.get("criteria")
            options = len(criteria) if isinstance(criteria, (dict, list)) else 0
        usage: dict[str, Any] | None = None
        depth: float | None = None
        with contextlib.suppress(ValueError, AttributeError, KeyError, TypeError):
            answer = upstream.json()
            raw_usage = answer.get("usage")
            usage = raw_usage if isinstance(raw_usage, dict) else None
            if "depth" in answer.get("answers", {}):
                depth = float(answer["answers"]["depth"]["score"])
        call = TypeSafeCall(
            question=question,
            options=options,
            approx_request_tokens=approx_tokens(payload.decode("utf-8", "replace")),
            status=upstream.status_code,
            latency_ms=round(latency_ms, 1),
            attempts=attempts,
            usage=usage,
            depth_score=depth,
        )
        with self._lock:
            self.calls.append(call)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _redacted(env: Mapping[str, str]) -> dict[str, str]:
    return {key: ("<redacted>" if _SECRET.search(key) else value) for key, value in sorted(env.items())}


def server_environment(config: RunConfig, proxy_url: str | None) -> dict[str, str]:
    env = dict(BASE_ENV)
    if config.local_models:
        env.update(LOCAL_MODEL_ENV)
    env.update(config.server_env)
    if proxy_url is not None:
        env["HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL"] = proxy_url
    return env


async def _ingest(http: httpx.AsyncClient, banks: Mapping[str, str]) -> dict[str, int]:
    for bank_id in banks.values():
        response = await http.put(f"/v1/default/banks/{bank_id}", json={})
        response.raise_for_status()
        response = await http.patch(
            f"/v1/default/banks/{bank_id}/config",
            json={"updates": {"retain_extraction_mode": "chunks", "enable_observations": False}},
        )
        response.raise_for_status()
    for bank_key, bank_id in banks.items():
        items = [
            {
                "content": memory.text,
                "document_id": memory.document_id,
                "timestamp": memory.timestamp.isoformat(),
                "tags": memory.tags,
                "metadata": memory.metadata,
            }
            for memory in MEMORIES
            if memory.bank == bank_key
        ]
        for start in range(0, len(items), 25):
            response = await http.post(
                f"/v1/default/banks/{bank_id}/memories", json={"items": items[start : start + 25]}
            )
            if response.status_code != 200:
                raise BenchmarkError(f"retain into {bank_id} -> {response.status_code}: {response.text[:500]}")
    return await _verify_banks(http, banks)


async def _verify_banks(http: httpx.AsyncClient, banks: Mapping[str, str]) -> dict[str, int]:
    """Each bank holds exactly one memory per scenario document, and nothing else."""
    sizes: dict[str, int] = {}
    for bank_key, bank_id in banks.items():
        response = await http.get(f"/v1/default/banks/{bank_id}/memories/list", params={"limit": 1000})
        response.raise_for_status()
        items = response.json()["items"]
        expected = {memory.document_id for memory in MEMORIES if memory.bank == bank_key}
        documents = [item.get("document_id") for item in items]
        if sorted(documents) != sorted(expected):
            raise BenchmarkError(
                f"bank {bank_id}: {len(items)} memories for {len(expected)} documents "
                f"(missing {len(expected - set(documents))}, extra {len(documents) - len(expected)})"
            )
        sizes[bank_key] = len(items)
    return sizes


def _parse_results(payload: Mapping[str, Any], unlabelled: dict[str, str]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for rank, result in enumerate(payload.get("results", []), start=1):
        label = LABEL_BY_DOCUMENT.get(result.get("document_id") or "")
        if label is None:
            label = f"?{result['id']}"
            unlabelled[label] = result.get("text", "")
        scores = result.get("scores") or {}
        rows.append(
            ResultRow(
                label=label,
                server_rank=rank,
                final=scores.get("final"),
                reranker=scores.get("reranker"),
                semantic=scores.get("semantic"),
                keyword=scores.get("keyword"),
            )
        )
    return rows


async def run_benchmark(config: RunConfig) -> RunRecord:
    queries = [query for query in QUERIES if config.query_ids is None or query.id in config.query_ids]
    reused = load_record(config.reuse) if config.reuse else None
    proxy = _TypeSafeProxy(TYPESAFE_UPSTREAM) if config.typesafe_proxy else None
    env = server_environment(config, proxy.url if proxy else None)
    schema = (
        reused.database_schema
        if reused
        else f"ctxb_{re.sub(r'[^a-z0-9]+', '_', config.label.lower())[:24]}_{uuid.uuid4().hex[:6]}"
    )
    nonce = uuid.uuid4().hex[:6]
    banks = reused.banks if reused else {MAIN_BANK: f"ctxb-main-{nonce}", SESSION_BANK: f"ctxb-session-{nonce}"}
    database_url = f"pg0://{config.pg0}"
    record = RunRecord(
        label=config.label,
        api_dir=str(config.api_dir),
        api_sha=_git(config.api_dir, "rev-parse", "HEAD"),
        api_dirty=bool(_git(config.api_dir, "status", "--porcelain", "--", ".")),
        harness_sha=_git(Path(__file__).resolve().parent, "rev-parse", "HEAD"),
        server_env=_redacted(env),
        variants=list(config.variants),
        send_context=config.send_context,
        max_tokens=config.max_tokens,
        database_url=database_url,
        database_schema=schema,
        banks=dict(banks),
        bank_sizes={},
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    started = time.monotonic()
    log_dir = config.out_dir / "logs" / config.label
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_API_DIR] = str(config.api_dir)
    os.environ[ENV_DATABASE_URL] = database_url
    stub = start_stub_server(Stubs())
    if proxy:
        proxy.start()
    server = None
    try:
        server = start_hindsight_server(
            stub_url=stub.url,
            log_path=log_dir / "server.log",
            extra_env={**env, "HINDSIGHT_API_DATABASE_SCHEMA": schema},
        )
        async with httpx.AsyncClient(base_url=server.url, timeout=300) as http:
            record.bank_sizes = await (_verify_banks(http, banks) if reused else _ingest(http, banks))
            logger.info("banks ready: %s", record.bank_sizes)
            for variant in config.variants:
                variant_started = time.monotonic()
                for query in queries:
                    record.rows.append(await _recall(http, banks, variant, query, config, record.unlabelled_texts))
                logger.info("%s: %d queries in %.1fs", variant, len(queries), time.monotonic() - variant_started)
                # A checkpoint per variant: a long run that dies keeps what it measured.
                write_record(record, config.out_dir / "runs" / f"{config.label}.partial.json")
    finally:
        if server is not None:
            server.stop()
        if proxy:
            proxy.stop()
            record.typesafe_calls = list(proxy.calls)
        stub.stop()
    record.finished_at = datetime.now(timezone.utc).isoformat()
    record.duration_s = round(time.monotonic() - started, 1)
    return record


async def _recall(
    http: httpx.AsyncClient,
    banks: Mapping[str, str],
    variant: str,
    query: Query,
    config: RunConfig,
    unlabelled: dict[str, str],
) -> QueryRow:
    body = variant_body(variant, query, max_tokens=config.max_tokens, send_context=config.send_context)
    started = time.monotonic()
    response = await http.post(f"/v1/default/banks/{banks[query.bank]}/memories/recall", json=body)
    latency_ms = (time.monotonic() - started) * 1000
    if response.status_code != 200:
        logger.error("%s %s -> %d: %s", variant, query.id, response.status_code, response.text[:300])
        return QueryRow(
            variant=variant,
            query_id=query.id,
            split=split_of(query.id),
            latency_ms=round(latency_ms, 1),
            status=response.status_code,
            results=[],
            error=response.text[:500],
        )
    return QueryRow(
        variant=variant,
        query_id=query.id,
        split=split_of(query.id),
        latency_ms=round(latency_ms, 1),
        status=200,
        results=_parse_results(response.json(), unlabelled),
    )


def write_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1))


def load_record(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text())


def typesafe_summary(calls: Sequence[TypeSafeCall]) -> dict[str, Any]:
    by_question: dict[str, int] = {}
    usage_totals: dict[str, float] = {}
    for call in calls:
        by_question[call.question] = by_question.get(call.question, 0) + 1
        for key, value in (call.usage or {}).items():
            if isinstance(value, (int, float)):
                usage_totals[key] = usage_totals.get(key, 0) + value
    latencies = [call.latency_ms for call in calls]
    return {
        "calls": len(calls),
        "by_question": by_question,
        "non_200": sum(1 for call in calls if call.status != 200),
        "retried_calls": sum(1 for call in calls if call.attempts > 1),
        "approx_request_tokens": sum(call.approx_request_tokens for call in calls),
        "usage_reported": any(call.usage for call in calls),
        "usage_totals": usage_totals,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def run(config: RunConfig) -> RunRecord:
    record = asyncio.run(run_benchmark(config))
    write_record(record, config.out_dir / "runs" / f"{config.label}.json")
    (config.out_dir / "runs" / f"{config.label}.partial.json").unlink(missing_ok=True)
    return record
