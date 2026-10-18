"""Run the context_pollution queries against one server build and record every result.

One run = one server profile (a checkout plus server env) and a list of request
*variants*. Each variant is a recall request shape, sent once per query:

- ``plain`` — the query as an agent sends it today;
- ``tags`` — plus the task's project as a hard tag scope (``tags_match=any``, so the
  user's own untagged memories stay reachable);
- ``final@0.30`` / ``reranker@0.30`` — plus a ``min_scores`` floor on that stage;
- ``adm=filter+ta=0.55+te=0.40`` — a context-admission setting (feature server only): before
  the variant's queries every bank's config is patched to that ``context_admission_mode`` and
  auto / explicit thresholds, and each request asks for a trace so the gate's per-candidate
  decisions are recorded (:class:`AdmissionRow`);
- components join with ``+`` (``tags+final@0.30``); ``#2`` repeats a variant (determinism).

With ``send_context`` every request also carries the request ``context`` object and ``purpose``,
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
#: Bench servers run the experimental admission arms (filter, strict, the jev and hybrid scorers),
#: which a server refuses without the experimental switch; a base checkout ignores the variable.
BASE_ENV = {
    "HINDSIGHT_API_LLM_PROVIDER": "none",
    "UV_NO_SYNC": "1",
    "HINDSIGHT_API_CONTEXT_ADMISSION_EXPERIMENTAL": "true",
}

TYPESAFE_UPSTREAM = "https://api.typesafe.ai"
#: The context-admission decision model (the Jev scorer), configured like the other operations'.
ADMISSION_DECISION_MODEL_PROVIDER = "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER"

#: Server env whose values are never written to a run file. Scope-key lists name metadata keys
#: (``..._SCOPE_KEYS=repository,module``), not credentials, and a replay needs them.
_SECRET = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)", re.IGNORECASE)
_NOT_SECRET = re.compile(r"_SCOPE_KEYS$")

DEFAULT_PG0 = "hindsight-ctxbench:15529"
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

ADMISSION_MODES = ("observe", "filter", "strict")
#: Admission variant component -> the bank config field it sets.
_ADMISSION_FIELDS = {
    "adm": "context_admission_mode",
    "ta": "context_admission_threshold_auto",
    "te": "context_admission_threshold_explicit",
}


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


def admission_updates(variant: str) -> dict[str, Any] | None:
    """The bank config an admission variant sets before its queries, or None for a variant
    without admission components. An admission variant names all three fields, so a run file
    always says which setting each row was asked under."""
    updates: dict[str, Any] = {}
    for part in variant.split("#", 1)[0].split("+"):
        name, sep, value = part.partition("=")
        if not sep:
            continue
        if name not in _ADMISSION_FIELDS:
            raise ValueError(f"unknown admission component {part!r} in {variant!r}")
        if name == "adm":
            if value not in ADMISSION_MODES:
                raise ValueError(f"admission mode must be one of {ADMISSION_MODES}, got {value!r} in {variant!r}")
            updates[_ADMISSION_FIELDS[name]] = value
        else:
            threshold = float(value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"admission threshold must be in [0, 1], got {value!r} in {variant!r}")
            updates[_ADMISSION_FIELDS[name]] = threshold
    if not updates:
        return None
    missing = sorted(set(_ADMISSION_FIELDS.values()) - set(updates))
    if missing:
        raise ValueError(f"admission variant {variant!r} must set adm=, ta= and te= (missing {missing})")
    return updates


def variant_body(
    variant: str, query: Query, *, max_tokens: int, send_context: bool, trace: bool = False
) -> dict[str, Any]:
    """The recall body a variant sends for one query (see the module docstring)."""
    body = base_body(query, max_tokens=max_tokens, send_context=send_context)
    floors: dict[str, float] = {}
    admission = admission_updates(variant) is not None
    for part in variant.split("#", 1)[0].split("+"):
        if part == "plain" or "=" in part:
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
    if trace or admission:
        body["trace"] = True
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


class AdmissionDecisionRow(BaseModel):
    """One judged candidate, as the server's ``trace.context_admission.decisions[]`` reports it,
    plus the scenario label of its memory (``?<memory id>`` for one the scenario did not write)."""

    label: str
    memory_id: str
    base_retrieval_score: float
    scope_score: float
    task_relevance: float
    expected_utility: float
    truth_status: str
    retrievability: float | None = None
    admission_score: float
    admitted: bool
    reason: str
    tokens: int
    duplicates: int = 0
    scope: dict[str, str] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)
    """In the server's order: the deterministic scorer sums them in this order."""


class AdmissionRow(BaseModel):
    """What the gate did on one recall: the response's ``context_admission`` summary (filter and
    strict only) and the trace's account. ``decisions`` are in recall order. Empty when the gate
    did not run (nothing was retrieved)."""

    summary: dict[str, Any] | None = None
    mode: str | None = None
    scorer: str | None = None
    purpose: str | None = None
    threshold: float | None = None
    historical: bool | None = None
    reference_signal: str | None = None
    candidates: int = 0
    admitted: int = 0
    abstained: bool = False
    reasons: dict[str, int] = Field(default_factory=dict)
    cross_scope_candidates: int = 0
    relevance_source: str | None = None
    jev_error: str | None = None
    embeddings_error: str | None = None
    scope_error: str | None = None
    phase_ms: float | None = None
    """The trace's ``context_admission`` phase duration in ms (summed over search retries); None
    when the trace reports no such phase."""
    decisions: list[AdmissionDecisionRow] = Field(default_factory=list)


class QueryRow(BaseModel):
    variant: str
    query_id: str
    split: str
    latency_ms: float
    status: int
    results: list[ResultRow]
    error: str | None = None
    admission: AdmissionRow | None = None


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
    memory_labels: dict[str, str] = Field(default_factory=dict)
    """Server memory id -> scenario label, from the bank listing (the trace names memories by id)."""
    admission_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Admission variant -> the resolved ``context_admission_*`` / ``adaptive_memory_*`` bank config
    the server answered the patch with (the same for every bank, checked)."""
    embeddings_file: str | None = None
    """A side file (label -> stored embedding) for replaying redundancy suppression offline."""


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
    trace: bool = False
    """Ask every recall for a trace (admission variants always do)."""
    dump_embeddings: bool = False
    """Read the banks' stored embeddings from the run's pg0 instance into a side file."""


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
    return {
        key: ("<redacted>" if _SECRET.search(key) and not _NOT_SECRET.search(key) else value)
        for key, value in sorted(env.items())
    }


def server_environment(config: RunConfig, proxy_url: str | None) -> dict[str, str]:
    env = dict(BASE_ENV)
    if config.local_models:
        env.update(LOCAL_MODEL_ENV)
    env.update(config.server_env)
    if proxy_url is not None:
        env["HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL"] = proxy_url
        if env.get(ADMISSION_DECISION_MODEL_PROVIDER) == "typesafe":
            env["HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_BASE_URL"] = proxy_url
    return env


async def _ingest(
    http: httpx.AsyncClient, banks: Mapping[str, str], memory_labels: dict[str, str] | None = None
) -> dict[str, int]:
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
    return await _verify_banks(http, banks, memory_labels)


async def _verify_banks(
    http: httpx.AsyncClient, banks: Mapping[str, str], memory_labels: dict[str, str] | None = None
) -> dict[str, int]:
    """Each bank holds exactly one memory per scenario document, and nothing else. Fills
    ``memory_labels`` (server memory id -> scenario label) when given."""
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
        if memory_labels is not None:
            for item in items:
                memory_labels[str(item["id"])] = LABEL_BY_DOCUMENT[item["document_id"]]
    return sizes


async def _apply_admission(
    http: httpx.AsyncClient, banks: Mapping[str, str], updates: Mapping[str, Any]
) -> dict[str, Any]:
    """Patch every bank to one admission setting; returns the resolved admission (and adaptive
    memory) config the server answered with, which must be the same for every bank."""
    resolved: dict[str, Any] | None = None
    for bank_id in banks.values():
        response = await http.patch(f"/v1/default/banks/{bank_id}/config", json={"updates": dict(updates)})
        if response.status_code != 200:
            raise BenchmarkError(f"admission config on {bank_id} -> {response.status_code}: {response.text[:500]}")
        config = response.json()["config"]
        relevant = {
            key: value
            for key, value in sorted(config.items())
            if key.startswith(("context_admission_", "adaptive_memory_mode"))
        }
        for key, value in updates.items():
            if relevant.get(key) != value:
                raise BenchmarkError(f"bank {bank_id}: {key} reads {relevant.get(key)!r} after setting {value!r}")
        if resolved is not None and relevant != resolved:
            raise BenchmarkError(f"bank {bank_id} resolved a different admission config than the first bank")
        resolved = relevant
    return resolved or {}


def _pg0_instance(pg0: str) -> dict[str, Any]:
    name = pg0.split(":", 1)[0]
    path = Path.home() / ".pg0" / "instances" / name / "instance.json"
    if not path.is_file():
        raise BenchmarkError(f"no pg0 instance file at {path}")
    return json.loads(path.read_text())


def dump_embeddings(
    pg0: str, schema: str, banks: Mapping[str, str], memory_labels: Mapping[str, str]
) -> dict[str, list[float]]:
    """The banks' stored embeddings, label -> floats, read with the pg0 instance's own ``psql``
    in the server's text form and parsed the way the server parses it (``float`` per component),
    so an offline replay of redundancy suppression sees the server's numbers."""
    instance = _pg0_instance(pg0)
    psql = Path(instance["installation_dir"]) / instance["version"] / "bin" / "psql"
    if not re.fullmatch(r"[a-z0-9_]+", schema):
        raise BenchmarkError(f"unexpected schema name {schema!r}")
    bank_list = ", ".join("'" + bank.replace("'", "''") + "'" for bank in banks.values())
    sql = f'SELECT id::text, embedding::text FROM "{schema}".memory_units WHERE bank_id IN ({bank_list})'
    completed = subprocess.run(
        [
            str(psql),
            "-h",
            "127.0.0.1",
            "-p",
            str(instance["port"]),
            "-U",
            instance["username"],
            "-d",
            instance["database"],
            "-At",
            "-F",
            "\t",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PGPASSWORD": instance["password"]},
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"embedding read failed: {completed.stderr.strip()[:500]}")
    return parse_embedding_rows(completed.stdout, memory_labels)


def parse_embedding_rows(text: str, memory_labels: Mapping[str, str]) -> dict[str, list[float]]:
    """``<memory id>\t[x,y,...]`` lines (pgvector's text form) -> label -> floats, one per memory."""
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        memory_id, _, vector = line.partition("\t")
        label = memory_labels.get(memory_id)
        if label is None:
            raise BenchmarkError(f"embedding for unknown memory {memory_id}")
        out[label] = [float(part) for part in vector.strip().removeprefix("[").removesuffix("]").split(",")]
    if len(out) != len(memory_labels):
        raise BenchmarkError(f"read {len(out)} embeddings for {len(memory_labels)} memories")
    return out


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
            record.bank_sizes = await (
                _verify_banks(http, banks, record.memory_labels)
                if reused
                else _ingest(http, banks, record.memory_labels)
            )
            logger.info("banks ready: %s", record.bank_sizes)
            if config.dump_embeddings:
                embeddings = dump_embeddings(config.pg0, schema, banks, record.memory_labels)
                path = config.out_dir / "runs" / f"{config.label}.embeddings.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(embeddings))
                record.embeddings_file = path.name
                logger.info("embeddings: %d memories -> %s", len(embeddings), path)
            for variant in config.variants:
                variant_started = time.monotonic()
                updates = admission_updates(variant)
                if updates is not None:
                    record.admission_settings[variant] = await _apply_admission(http, banks, updates)
                for query in queries:
                    record.rows.append(
                        await _recall(
                            http, banks, variant, query, config, record.unlabelled_texts, record.memory_labels
                        )
                    )
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
    memory_labels: Mapping[str, str],
) -> QueryRow:
    body = variant_body(
        variant, query, max_tokens=config.max_tokens, send_context=config.send_context, trace=config.trace
    )
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
    payload = response.json()
    return QueryRow(
        variant=variant,
        query_id=query.id,
        split=split_of(query.id),
        latency_ms=round(latency_ms, 1),
        status=200,
        results=_parse_results(payload, unlabelled),
        admission=parse_admission(payload, memory_labels) if admission_updates(variant) is not None else None,
    )


def parse_admission(payload: Mapping[str, Any], memory_labels: Mapping[str, str]) -> AdmissionRow:
    """The gate's account of one recall: the response summary plus the trace's decisions."""
    full_trace = payload.get("trace") or {}
    trace = full_trace.get("context_admission")
    summary = payload.get("context_admission")
    if trace is None:
        return AdmissionRow(summary=summary)
    decisions = [
        AdmissionDecisionRow(label=memory_labels.get(item["memory_id"], f"?{item['memory_id']}"), **item)
        for item in trace.get("decisions", [])
    ]
    phases = [
        phase["duration_seconds"]
        for phase in (full_trace.get("summary") or {}).get("phase_metrics") or []
        if phase.get("phase_name") == "context_admission" and isinstance(phase.get("duration_seconds"), (int, float))
    ]
    fields = {
        key: trace.get(key) for key in AdmissionRow.model_fields if key not in ("summary", "decisions", "phase_ms")
    }
    return AdmissionRow(
        summary=summary,
        decisions=decisions,
        phase_ms=round(sum(phases) * 1000, 3) if phases else None,
        **{key: value for key, value in fields.items() if value is not None},
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
