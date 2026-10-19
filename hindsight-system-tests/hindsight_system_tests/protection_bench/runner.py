"""Run one suite of the protection benchmark against one server build and record what it served.

One run is one server (a checkout plus server env) on a pg0 instance of the run's own, stopped
with the server. Suites:

- ``p1`` — the coding-project bank of :mod:`.data` is retained in chunks mode (one memory per
  document, verbatim, no LLM), then every task query is recalled as the coding-agents
  integration recalls (``purpose=auto_context``, the session's repository context, ``budget=low``,
  ``max_tokens=2000``, entities off; the bank has no observations, so the types are ``world`` and
  ``experience``, as the integration's docs tell such a bank to ask). A *pass* is one sweep of the
  queries under one bank-level override:

  * ``served`` — no override: what the server serves under its own env (BEFORE's only pass, and
    AFTER's measured one);
  * ``gate-off`` / ``gate-only`` (AFTER) — the Jev rerank off, the injection gate off / on. With
    the rerank off the list before the gate comes from the deterministic local pipeline, so the two
    passes see the same pre-gate list and their difference is exactly what the gate withheld.

- ``p2`` — the same bank and queries, ``served`` only, on a server whose TypeSafe base URL points
  at a port nothing listens on (``dead_endpoint``). BEFORE's side of P2 is BEFORE's P1 run.
- ``p3`` — each scope scenario on a bank of its own: waves retained in chunks mode with metadata,
  consolidated by the server's LLM, waited out, and the final state recorded (every memory, its
  sources, and its claim when the server keeps claims).

Every decision-model request goes through context_bench's usage-recording proxy, so a run
records how many Jev calls it made; a cap over every run file stops a run before it overspends.
The P3 Anthropic spend is read back from the server's own LLM trace and capped the same way.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..context_bench.runner import (
    BASE_ENV,
    LOCAL_MODEL_ENV,
    TYPESAFE_UPSTREAM,
    BenchmarkError,
    TypeSafeCall,
    _git,
    _redacted,
    _TypeSafeProxy,
)
from ..harm_bench.llm import PRICES
from ..rulebook import Stubs
from ..server import ENV_API_DIR, ENV_DATABASE_URL, free_port, start_hindsight_server, start_stub_server
from .data import LABEL_BY_DOCUMENT, MEMORIES, QUERIES, QUERY_TIME, SCENARIOS, P3Scenario, Query

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

Suite = Literal["p1", "p2", "p3"]
SUITES: tuple[Suite, ...] = ("p1", "p2", "p3")
Side = Literal["before", "after"]
SIDES: tuple[Side, ...] = ("before", "after")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = Path(__file__).resolve().parent / "data.py"
DEFAULT_OUT = REPO_ROOT / "tasks" / "adaptive-memory" / "eval" / "protection-bench"
FROZEN_HASH_FILE = DEFAULT_OUT / "data.sha256"

#: Bank-level overrides a P1 pass applies before its sweep (``None``: the server's own config).
PASSES: dict[str, dict[str, Any] | None] = {
    "served": None,
    "gate-off": {"recall_injection_gate": False, "recall_jev_rerank": False},
    "gate-only": {"recall_injection_gate": True, "recall_jev_rerank": False},
}
DEFAULT_PASSES: dict[tuple[str, str], tuple[str, ...]] = {
    ("p1", "before"): ("served",),
    ("p1", "after"): ("served", "gate-off", "gate-only"),
    ("p2", "after"): ("served",),
}

#: How many results a recall injects: the protocol's "returned top 10".
TOP_K = 10
#: The coding-agents integration's ``DEFAULT_RECALL_OPTIONS`` (``core/hindsight.ts``), with the
#: types widened as its docs tell a bank without observations to.
INTEGRATION_RECALL_OPTIONS: dict[str, Any] = {
    "types": ["world", "experience"],
    "budget": "low",
    "max_tokens": 2000,
    "include": {"entities": None},
}

#: The documented personal profile ("Personal profile" in adaptive-memory.mdx) plus its optional
#: recall-check lines for the injection gate and the Jev rerank. The key is read from the
#: environment (:data:`ENV_TYPESAFE_KEY`) so it never appears on a command line.
PERSONAL_PROFILE: dict[str, str] = {
    "HINDSIGHT_API_ENABLE_OBSERVATIONS": "true",
    "HINDSIGHT_API_ADAPTIVE_MEMORY_MODE": "lifecycle",
    "HINDSIGHT_API_ADAPTIVE_MEMORY_SCORE_ALPHA": "0",
    "HINDSIGHT_API_CONTEXT_ADMISSION_MODE": "observe",
    "HINDSIGHT_API_CONTEXT_ADMISSION_SCORER": "deterministic",
    "HINDSIGHT_API_CONTEXT_ADMISSION_EXPERIMENTAL": "false",
    "HINDSIGHT_API_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL": "false",
    "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_PROVIDER": "typesafe",
    "HINDSIGHT_API_RECALL_INJECTION_GATE": "true",
    "HINDSIGHT_API_RECALL_JEV_RERANK": "true",
}
ENV_TYPESAFE_KEY = "TYPESAFE_API_KEY"
RECALL_CHECKS_KEY = "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_API_KEY"

ENV_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
P3_MODEL = "claude-haiku-5-5"

#: Protocol caps, over every run file under ``<out>/runs`` (finished, partial and aborted).
DEFAULT_JEV_CAP = 3000
DEFAULT_ANTHROPIC_CAP_USD = 1.00
#: Jev calls one P1 query can make (a rerank plus a few gate batches, plus retries): a sweep
#: stops before a query that could cross the cap.
JEV_CALLS_PER_QUERY_MARGIN = 20
#: Assumed for the first P3 scenario, before any has been priced.
P3_SCENARIO_COST_ESTIMATE_USD = 0.05

SETTLE_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.5

_DECISION_PROVIDER = re.compile(r"^HINDSIGHT_API_(?:([A-Z_]+)_)?DECISION_MODEL_PROVIDER$")


# ---------------------------------------------------------------------------
# Server environment
# ---------------------------------------------------------------------------


def profile_env(profile: str, environ: Mapping[str, str]) -> dict[str, str]:
    """``none``: nothing; ``personal``: :data:`PERSONAL_PROFILE` with the TypeSafe key from ``environ``."""
    if profile == "none":
        return {}
    if profile != "personal":
        raise ValueError(f"unknown profile {profile!r}")
    key = environ.get(RECALL_CHECKS_KEY) or environ.get(ENV_TYPESAFE_KEY)
    if not key:
        raise ValueError(f"--profile personal needs {ENV_TYPESAFE_KEY} in the environment")
    return {**PERSONAL_PROFILE, RECALL_CHECKS_KEY: key}


def typesafe_base_url_keys(env: Mapping[str, str]) -> list[str]:
    """The base-URL settings of every component the env sends to TypeSafe: each decision-model
    operation (or the global one) whose provider is ``typesafe``, and a ``typesafe`` reranker."""
    keys: list[str] = []
    for name, value in sorted(env.items()):
        match = _DECISION_PROVIDER.match(name)
        if match and value.strip().lower() == "typesafe":
            operation = match.group(1)
            keys.append(
                f"HINDSIGHT_API_{operation}_DECISION_MODEL_BASE_URL"
                if operation
                else name[: -len("PROVIDER")] + "BASE_URL"
            )
    if env.get("HINDSIGHT_API_RERANKER_PROVIDER", "").strip().lower() == "typesafe":
        keys.append("HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL")
    return keys


def dead_url() -> str:
    """A loopback URL on a port that was free a moment ago and has no listener: connections are refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def server_env(suite: Suite, extra: Mapping[str, str], anthropic_key: str | None = None) -> dict[str, str]:
    """BASE/local-model env plus the run's own. P1/P2 run with no LLM (chunks-mode retain only);
    P3 consolidates with :data:`P3_MODEL` and must not inherit the provider stub's base URL."""
    env = {**LOCAL_MODEL_ENV, "UV_NO_SYNC": "1"}
    if suite == "p3":
        if not anthropic_key:
            raise ValueError(f"p3 needs {ENV_ANTHROPIC_KEY} in the environment")
        env.update(
            {
                "HINDSIGHT_API_LLM_PROVIDER": "anthropic",
                "HINDSIGHT_API_LLM_MODEL": P3_MODEL,
                "HINDSIGHT_API_LLM_API_KEY": anthropic_key,
                # Empty reads as unset: the provider's own endpoint, not the stub's.
                "HINDSIGHT_API_LLM_BASE_URL": "",
                "HINDSIGHT_API_LLM_MAX_RETRIES": "4",
            }
        )
    else:
        env.update(BASE_ENV)
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    label: str
    """Data label; ``?<memory id>`` for a memory the benchmark did not write."""
    rank: int
    final: float | None = None
    reranker: float | None = None


class RecallRow(BaseModel):
    pass_name: str
    query_id: str
    status: int
    latency_ms: float
    results: list[ResultRow] = Field(default_factory=list)
    error: str | None = None

    @property
    def labels(self) -> list[str]:
        return [r.label for r in self.results]

    @property
    def top(self) -> list[str]:
        return self.labels[:TOP_K]


class P3MemoryRow(BaseModel):
    id: str
    fact_type: str
    text: str
    document_id: str | None = None
    sources: list[str] = Field(default_factory=list)
    state: str | None = None
    """The curation state (``valid`` / ``invalidated``)."""
    truth_status: str = "active"
    """The claim's status; ``active`` when the server reports no claim (a server without claims)."""
    retired_via: str | None = None
    consolidated: bool | None = None


class P3ScenarioRow(BaseModel):
    scenario_id: str
    bank_id: str
    bank_config: dict[str, Any] = Field(default_factory=dict)
    memories: list[P3MemoryRow] = Field(default_factory=list)
    snapshots: list[list[str]] = Field(default_factory=list)
    """Current observation texts after each wave."""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    suite: str
    side: str
    label: str
    data_sha256: str
    api_dir: str
    api_sha: str | None
    api_dirty: bool | None
    harness_sha: str | None
    harness_dirty: bool | None
    profile: str
    server_env: dict[str, str]
    dead_endpoint: str | None = None
    dead_endpoint_keys: list[str] = Field(default_factory=list)
    database_url: str
    database_schema: str
    bank_id: str | None = None
    bank_size: int | None = None
    bank_config: dict[str, Any] = Field(default_factory=dict)
    """P1/P2: the bank's resolved recall-check / admission / adaptive config after ingest."""
    passes: list[str] = Field(default_factory=list)
    pass_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Pass -> the resolved recall-check config the server answered its override with."""
    recall_options: dict[str, Any] = Field(default_factory=dict)
    top_k: int = TOP_K
    started_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    stopped: str | None = None
    """Why the run stopped before finishing (a cap), or None."""
    recalls: list[RecallRow] = Field(default_factory=list)
    scenarios: list[P3ScenarioRow] = Field(default_factory=list)
    typesafe_calls: list[TypeSafeCall] = Field(default_factory=list)
    unlabelled_texts: dict[str, str] = Field(default_factory=dict)

    @property
    def jev_requests(self) -> int:
        """Requests that reached TypeSafe (each proxy retry is one)."""
        return sum(call.attempts for call in self.typesafe_calls)

    @property
    def spend_usd(self) -> float:
        return sum(row.cost_usd for row in self.scenarios)

    def recall(self, pass_name: str, query_id: str) -> RecallRow | None:
        return next((r for r in self.recalls if r.pass_name == pass_name and r.query_id == query_id), None)


@dataclass(frozen=True)
class Usage:
    """What the run files under one directory recorded: requests that reached TypeSafe, and P3's Anthropic spend."""

    jev_requests: int = 0
    anthropic_usd: float = 0.0


# --- Server payloads, parsed at the boundary (unknown keys are ignored, so an older or newer
# --- server's extra fields never break a run).


class _RecallScores(BaseModel):
    final: float | None = None
    reranker: float | None = None


class _RecallItem(BaseModel):
    id: str
    document_id: str | None = None
    text: str | None = None
    scores: _RecallScores | None = None


class _RecallPage(BaseModel):
    results: list[_RecallItem] | None = None


class _BankConfig(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    """Config field -> resolved value: the keys are the server's, not this harness's."""


class _ListedMemory(BaseModel):
    id: str
    fact_type: str | None = None
    text: str | None = None
    document_id: str | None = None
    source_memory_ids: list[str] | None = None
    state: str | None = None
    consolidated_at: str | None = None
    consolidation_failed_at: str | None = None


class _MemoryPage(BaseModel):
    items: list[_ListedMemory] = Field(default_factory=list)
    total: int | None = None


class _Operation(BaseModel):
    task_type: str | None = None
    status: str | None = None
    error_message: str | None = None


class _OperationPage(BaseModel):
    operations: list[_Operation] = Field(default_factory=list)


class _LLMRequest(BaseModel):
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


class _LLMRequestPage(BaseModel):
    items: list[_LLMRequest] = Field(default_factory=list)
    total: int | None = None


class Claim(BaseModel):
    """A memory's claim as ``GET /memories/{id}`` reports it under ``lifecycle``; a server without
    claims reports none, which reads as active."""

    truth_status: str = "active"
    retired_via: str | None = None


class _Lifecycle(BaseModel):
    claim: Claim | None = None


class _MemoryDetail(BaseModel):
    lifecycle: _Lifecycle | None = None


@dataclass(frozen=True)
class IngestedBank:
    size: int
    config: dict[str, Any]
    """The bank's resolved recall / admission / adaptive config (server keys)."""


@dataclass(frozen=True)
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def write_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1))


def load_record(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text())


def data_sha256() -> str:
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


def frozen_sha256(path: Path | None = None) -> str | None:
    """The hash in the frozen hash file (:data:`FROZEN_HASH_FILE`, read when called), or None."""
    path = path or FROZEN_HASH_FILE
    if not path.is_file():
        return None
    return path.read_text().split()[0]


def prior_usage(runs_dir: Path, exclude: Path | None = None) -> Usage:
    """What every run file under ``runs_dir`` (finished, partial, aborted) recorded."""
    jev, usd = 0, 0.0
    if not runs_dir.is_dir():
        return Usage()
    for path in sorted(runs_dir.glob("*.json")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        with contextlib.suppress(ValueError, OSError):
            record = load_record(path)
            jev += record.jev_requests
            usd += record.spend_usd
    return Usage(jev_requests=jev, anthropic_usd=usd)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    suite: Suite
    side: Side
    label: str
    api_dir: Path
    out_dir: Path = DEFAULT_OUT
    profile: str = "none"
    server_env: Mapping[str, str] = field(default_factory=dict)
    passes: tuple[str, ...] | None = None
    dead_endpoint: bool = False
    pg0_name: str | None = None
    """pg0 instance of this run's own (default ``hindsight-protbench-<suite>-<side>``)."""
    query_ids: tuple[str, ...] | None = None
    scenario_ids: tuple[str, ...] | None = None
    jev_cap: int = DEFAULT_JEV_CAP
    anthropic_cap_usd: float = DEFAULT_ANTHROPIC_CAP_USD
    resume: bool = False
    """P3: continue ``<label>.partial.json``, skipping the scenarios it already finished."""

    def resolved_passes(self) -> tuple[str, ...]:
        if self.suite == "p3":
            return ()
        passes = self.passes or DEFAULT_PASSES.get((self.suite, self.side))
        if passes is None:
            raise ValueError(f"no default passes for {self.suite}/{self.side}; name them with --passes")
        unknown = [p for p in passes if p not in PASSES]
        if unknown:
            raise ValueError(f"unknown pass(es) {unknown}; choose from {list(PASSES)}")
        return passes


def check_config(config: RunConfig, env: Mapping[str, str]) -> None:
    """Refuse a run whose numbers could not mean what the protocol says."""
    if config.suite == "p2" and not config.dead_endpoint:
        raise ValueError("p2 is the decision-model outage: it needs --dead-endpoint (BEFORE's side is its p1 run)")
    if config.dead_endpoint and config.suite != "p2":
        raise ValueError("--dead-endpoint belongs to p2")
    if config.dead_endpoint and not typesafe_base_url_keys(env):
        raise ValueError("--dead-endpoint: nothing in the server env is sent to TypeSafe")
    passes = config.resolved_passes()
    if config.side == "before" and any(PASSES[p] is not None for p in passes):
        raise ValueError("bank-level recall-check overrides need the AFTER server")
    if config.suite in ("p1", "p2") and passes and passes[0] != "served":
        raise ValueError("the served pass runs first, on the bank as ingested")


# ---------------------------------------------------------------------------
# pg0
# ---------------------------------------------------------------------------


def _pg0_pid(name: str) -> int | None:
    path = Path.home() / ".pg0" / "instances" / name / "instance.json"
    with contextlib.suppress(OSError, ValueError, KeyError):
        pid = int(json.loads(path.read_text())["pid"])
        os.kill(pid, 0)
        return pid
    return None


def stop_pg0(name: str, api_dir: Path) -> None:
    """Stop the run's pg0 instance if the server left it running (it stops one it started on a
    clean shutdown), through the checkout's own pg0 package."""
    if _pg0_pid(name) is None:
        return
    logger.info("stopping pg0 instance %s", name)
    completed = subprocess.run(
        ["uv", "run", "--project", str(api_dir), "python", "-c", f"from pg0 import Pg0; Pg0(name={name!r}).stop()"],
        capture_output=True,
        text=True,
        env={**os.environ, "UV_NO_SYNC": "1"},
        check=False,
    )
    if completed.returncode != 0 or _pg0_pid(name) is not None:
        raise BenchmarkError(f"pg0 instance {name} is still running: {completed.stderr.strip()[:300]}")


# ---------------------------------------------------------------------------
# P1 / P2
# ---------------------------------------------------------------------------


def recall_body(query: Query) -> dict[str, Any]:
    return {
        "query": query.text,
        "purpose": "auto_context",
        "context": query.context(),
        **INTEGRATION_RECALL_OPTIONS,
        # Pinned so recency is computed from the data's own "now", identically on both servers.
        "query_timestamp": QUERY_TIME.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def parse_results(payload: Mapping[str, Any], unlabelled: dict[str, str]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for rank, result in enumerate(_RecallPage.model_validate(payload).results or [], start=1):
        label = LABEL_BY_DOCUMENT.get(result.document_id or "")
        if label is None:
            label = f"?{result.id}"
            unlabelled[label] = result.text or ""
        scores = result.scores or _RecallScores()
        rows.append(ResultRow(label=label, rank=rank, final=scores.final, reranker=scores.reranker))
    return rows


def _config_of(response: httpx.Response) -> dict[str, Any]:
    return _relevant(_BankConfig.model_validate(response.json()).config)


_RELEVANT_CONFIG = ("recall_", "context_admission_", "adaptive_memory_", "enable_observations", "retain_extraction")


def _relevant(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sorted(config.items()) if key.startswith(_RELEVANT_CONFIG)}


async def _ingest_p1(http: httpx.AsyncClient, bank_id: str) -> IngestedBank:
    response = await http.put(f"/v1/default/banks/{bank_id}", json={})
    response.raise_for_status()
    response = await http.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={"updates": {"retain_extraction_mode": "chunks", "enable_observations": False}},
    )
    if response.status_code != 200:
        raise BenchmarkError(f"bank config -> {response.status_code}: {response.text[:500]}")
    items = [
        {
            "content": m.text,
            "document_id": m.document_id,
            "timestamp": m.timestamp.isoformat(),
            "tags": m.tags,
            "metadata": m.metadata,
        }
        for m in MEMORIES
    ]
    for start in range(0, len(items), 25):
        response = await http.post(f"/v1/default/banks/{bank_id}/memories", json={"items": items[start : start + 25]})
        if response.status_code != 200:
            raise BenchmarkError(f"retain -> {response.status_code}: {response.text[:500]}")
    listed = await _list_memories(http, bank_id)
    documents = sorted(item.document_id or "" for item in listed)
    if documents != sorted(m.document_id for m in MEMORIES):
        raise BenchmarkError(f"bank {bank_id} holds {len(listed)} memories for {len(MEMORIES)} documents")
    response = await http.get(f"/v1/default/banks/{bank_id}/config")
    response.raise_for_status()
    return IngestedBank(size=len(listed), config=_config_of(response))


async def _apply_pass(http: httpx.AsyncClient, bank_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    response = await http.patch(f"/v1/default/banks/{bank_id}/config", json={"updates": dict(updates)})
    if response.status_code != 200:
        raise BenchmarkError(f"pass override {dict(updates)} -> {response.status_code}: {response.text[:500]}")
    resolved = _config_of(response)
    for key, value in updates.items():
        if resolved.get(key) != value:
            raise BenchmarkError(f"{key} reads {resolved.get(key)!r} after setting {value!r}")
    return resolved


async def probe_overrides(http: httpx.AsyncClient, passes: Sequence[str]) -> None:
    """Apply every pass's override to a throwaway bank first, so a server that rejects one fails the
    run before the measured bank has spent a single decision-model request."""
    overrides = [PASSES[p] for p in passes if PASSES[p] is not None]
    if not overrides:
        return
    probe = f"pb-probe-{uuid.uuid4().hex[:6]}"
    response = await http.put(f"/v1/default/banks/{probe}", json={})
    response.raise_for_status()
    try:
        for updates in overrides:
            assert updates is not None
            await _apply_pass(http, probe, updates)
    finally:
        with contextlib.suppress(httpx.HTTPError):
            await http.delete(f"/v1/default/banks/{probe}")


def require_decision_model_reached(calls: Sequence[TypeSafeCall]) -> None:
    """After AFTER's first recall: at least one decision-model request went through and succeeded,
    or the gate and the rerank are not running and the rest of the sweep would measure nothing."""
    if not any(call.status == 200 for call in calls):
        statuses = sorted({call.status for call in calls})
        raise BenchmarkError(
            f"the first recall reached TypeSafe {len(calls)} time(s), none answered 200 (statuses {statuses}); "
            "check the recall-checks provider, key and base URL"
        )


def _check_served_config(config: RunConfig, resolved: Mapping[str, Any]) -> None:
    """AFTER's served pass must run with the gate and the rerank on, as the profile sets them."""
    if config.side != "after" or config.suite == "p3":
        return
    for key in ("recall_injection_gate", "recall_jev_rerank"):
        if key in resolved and resolved[key] is not True:
            raise BenchmarkError(f"AFTER bank resolves {key}={resolved[key]!r}; the profile turns it on")


async def _recall(http: httpx.AsyncClient, bank_id: str, pass_name: str, query: Query, record: RunRecord) -> RecallRow:
    started = time.monotonic()
    try:
        response = await http.post(f"/v1/default/banks/{bank_id}/memories/recall", json=recall_body(query))
    except httpx.HTTPError as e:
        return RecallRow(
            pass_name=pass_name,
            query_id=query.id,
            status=0,
            latency_ms=round((time.monotonic() - started) * 1000, 1),
            error=f"{type(e).__name__}: {e}",
        )
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    if response.status_code != 200:
        logger.error("%s %s -> %d: %s", pass_name, query.id, response.status_code, response.text[:300])
        return RecallRow(
            pass_name=pass_name,
            query_id=query.id,
            status=response.status_code,
            latency_ms=latency_ms,
            error=response.text[:500],
        )
    return RecallRow(
        pass_name=pass_name,
        query_id=query.id,
        status=200,
        latency_ms=latency_ms,
        results=parse_results(response.json(), record.unlabelled_texts),
    )


async def _run_recall_suite(
    config: RunConfig,
    record: RunRecord,
    http: httpx.AsyncClient,
    proxy: _TypeSafeProxy | None,
    prior_jev: int,
    partial: Path,
) -> None:
    bank_id = record.bank_id
    assert bank_id is not None
    await probe_overrides(http, record.passes)
    bank = await _ingest_p1(http, bank_id)
    record.bank_size, record.bank_config = bank.size, bank.config
    logger.info("bank %s ready: %d memories; %s", bank_id, record.bank_size, record.bank_config)
    queries = [q for q in QUERIES if config.query_ids is None or q.id in config.query_ids]
    for pass_name in record.passes:
        updates = PASSES[pass_name]
        if updates is None:
            record.pass_configs[pass_name] = dict(record.bank_config)
            _check_served_config(config, record.bank_config)
        else:
            record.pass_configs[pass_name] = await _apply_pass(http, bank_id, updates)
        started = time.monotonic()
        for query in queries:
            used = prior_jev + (sum(c.attempts for c in proxy.calls) if proxy is not None else 0)
            if used + JEV_CALLS_PER_QUERY_MARGIN > config.jev_cap:
                record.stopped = f"Jev cap: {used} requests recorded, cap {config.jev_cap}"
                logger.error(record.stopped)
                return
            record.recalls.append(await _recall(http, bank_id, pass_name, query, record))
            if proxy is not None and len(record.recalls) == 1:
                require_decision_model_reached(proxy.calls)
        logger.info("%s: %d queries in %.1fs", pass_name, len(queries), time.monotonic() - started)
        if proxy is not None:
            record.typesafe_calls = list(proxy.calls)
        write_record(record, partial)


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------


async def _list_memories(http: httpx.AsyncClient, bank_id: str) -> list[_ListedMemory]:
    items: list[_ListedMemory] = []
    offset = 0
    while True:
        response = await http.get(f"/v1/default/banks/{bank_id}/memories/list", params={"limit": 500, "offset": offset})
        response.raise_for_status()
        page = _MemoryPage.model_validate(response.json())
        items += page.items
        offset += len(page.items)
        if not page.items or offset >= (page.total if page.total is not None else offset):
            return items


async def _operations(http: httpx.AsyncClient, bank_id: str, status: str) -> list[_Operation]:
    response = await http.get(f"/v1/default/banks/{bank_id}/operations", params={"status": status, "limit": 100})
    response.raise_for_status()
    return _OperationPage.model_validate(response.json()).operations


async def _settle(http: httpx.AsyncClient, bank_id: str, timeout: float = SETTLE_TIMEOUT_S) -> None:
    """Until the bank has no queued work and every raw fact is consolidated (or marked failed),
    for two polls in a row; a failed operation is an error."""
    deadline = time.monotonic() + timeout
    quiet = 0
    waiting: list[str] = []
    while time.monotonic() < deadline:
        failed = await _operations(http, bank_id, "failed")
        if failed:
            raise BenchmarkError(f"failed operations: {[(op.task_type, op.error_message) for op in failed]}")
        busy = await _operations(http, bank_id, "pending") + await _operations(http, bank_id, "processing")
        errored = [op for op in busy if op.error_message]
        if errored:
            raise BenchmarkError(f"erroring operations: {[(op.task_type, op.error_message) for op in errored]}")
        raw = [m for m in await _list_memories(http, bank_id) if m.fact_type != "observation"]
        waiting = [f"{op.task_type}({op.status})" for op in busy] + [
            f"unconsolidated {m.document_id}" for m in raw if not m.consolidated_at and not m.consolidation_failed_at
        ]
        quiet = 0 if waiting else quiet + 1
        if quiet >= 2:
            return
        await asyncio.sleep(POLL_INTERVAL_S)
    raise BenchmarkError(f"bank {bank_id} did not settle in {timeout:.0f}s: {waiting[:5]}")


async def _claim(http: httpx.AsyncClient, bank_id: str, memory_id: str) -> Claim:
    response = await http.get(f"/v1/default/banks/{bank_id}/memories/{memory_id}")
    if response.status_code == 404:
        return Claim()
    response.raise_for_status()
    lifecycle = _MemoryDetail.model_validate(response.json()).lifecycle
    return lifecycle.claim if lifecycle is not None and lifecycle.claim is not None else Claim()


async def _state(http: httpx.AsyncClient, bank_id: str) -> list[P3MemoryRow]:
    rows: list[P3MemoryRow] = []
    for item in await _list_memories(http, bank_id):
        claim = await _claim(http, bank_id, item.id)
        rows.append(
            P3MemoryRow(
                id=item.id,
                fact_type=item.fact_type or "",
                text=item.text or "",
                document_id=item.document_id,
                sources=list(item.source_memory_ids or []),
                state=item.state,
                truth_status=claim.truth_status,
                retired_via=claim.retired_via,
                consolidated=bool(item.consolidated_at) if item.fact_type != "observation" else None,
            )
        )
    return rows


async def _llm_usage(http: httpx.AsyncClient, bank_id: str) -> LLMUsage:
    """Every LLM call the server traced for the bank. (The server's start-up connection check is
    traced to no bank and is not counted: a few tokens.)"""
    calls = tokens_in = tokens_out = 0
    offset = 0
    while True:
        response = await http.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"limit": 500, "offset": offset})
        response.raise_for_status()
        page = _LLMRequestPage.model_validate(response.json())
        for item in page.items:
            calls += 1
            # Cached prompt tokens are priced as fresh input: an upper bound.
            tokens_in += (item.input_tokens or 0) + (item.cached_tokens or 0)
            tokens_out += item.output_tokens or 0
        offset += len(page.items)
        if not page.items or offset >= (page.total if page.total is not None else offset):
            return LLMUsage(calls=calls, input_tokens=tokens_in, output_tokens=tokens_out)


def p3_cost_usd(tokens_in: int, tokens_out: int, model: str = P3_MODEL) -> float:
    """At the long-context price when the total is above it: an upper bound per scenario."""
    low_in, low_out, high_in, high_out = PRICES[model]
    price_in, price_out = (high_in, high_out) if tokens_in > 100_000 else (low_in, low_out)
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


def p3_items(scenario: P3Scenario, wave: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fact in scenario.waves[wave]:
        item: dict[str, Any] = {
            "content": fact.text,
            "document_id": scenario.document_id(fact.key),
            "timestamp": f"{fact.on}T12:00:00+00:00",
            "entities": [{"text": e} for e in fact.entities],
        }
        if fact.metadata:
            item["metadata"] = dict(fact.metadata)
        items.append(item)
    return items


async def _run_scenario(http: httpx.AsyncClient, scenario: P3Scenario, bank_id: str) -> P3ScenarioRow:
    row = P3ScenarioRow(scenario_id=scenario.id, bank_id=bank_id)
    started = time.monotonic()
    try:
        response = await http.put(f"/v1/default/banks/{bank_id}", json={})
        response.raise_for_status()
        response = await http.patch(
            f"/v1/default/banks/{bank_id}/config",
            json={
                "updates": {
                    "retain_extraction_mode": "chunks",
                    "enable_observations": True,
                    "enable_auto_consolidation": True,
                }
            },
        )
        if response.status_code != 200:
            raise BenchmarkError(f"bank config -> {response.status_code}: {response.text[:300]}")
        row.bank_config = _config_of(response)
        for wave in range(len(scenario.waves)):
            response = await http.post(
                f"/v1/default/banks/{bank_id}/memories", json={"items": p3_items(scenario, wave)}
            )
            if response.status_code != 200:
                raise BenchmarkError(f"retain wave {wave + 1} -> {response.status_code}: {response.text[:300]}")
            await _settle(http, bank_id)
            state = await _state(http, bank_id)
            row.memories = state
            # The server marks a fact whose consolidation failed for good and moves on, without
            # failing an operation; a scenario consolidated without one of its facts measures nothing.
            unconsolidated = sorted(
                m.document_id or m.id for m in state if m.fact_type != "observation" and m.consolidated is False
            )
            if unconsolidated:
                raise BenchmarkError(f"wave {wave + 1}: consolidation failed for {unconsolidated}")
            row.snapshots.append(
                [
                    m.text
                    for m in state
                    if m.fact_type == "observation" and m.truth_status not in ("superseded", "refuted")
                ]
            )
    except (BenchmarkError, httpx.HTTPError) as e:
        row.error = f"{type(e).__name__}: {e}"
        logger.error("%s: %s", scenario.id, row.error)
    with contextlib.suppress(httpx.HTTPError):
        usage = await _llm_usage(http, bank_id)
        row.llm_calls, row.input_tokens, row.output_tokens = usage.calls, usage.input_tokens, usage.output_tokens
    row.cost_usd = round(p3_cost_usd(row.input_tokens, row.output_tokens), 6)
    row.duration_s = round(time.monotonic() - started, 1)
    return row


async def _run_p3(
    config: RunConfig, record: RunRecord, http: httpx.AsyncClient, prior_usd: float, partial: Path
) -> None:
    done = {row.scenario_id for row in record.scenarios if row.error is None}
    record.scenarios = [row for row in record.scenarios if row.error is None]
    scenarios = [s for s in SCENARIOS if (config.scenario_ids is None or s.id in config.scenario_ids)]
    for index, scenario in enumerate(scenarios):
        if scenario.id in done:
            continue
        priced = [row.cost_usd for row in record.scenarios]
        estimate = max([*priced, P3_SCENARIO_COST_ESTIMATE_USD])
        if prior_usd + record.spend_usd + estimate > config.anthropic_cap_usd:
            record.stopped = (
                f"Anthropic cap: ${prior_usd + record.spend_usd:.4f} recorded, the next scenario could cost "
                f"${estimate:.4f}, cap ${config.anthropic_cap_usd:.2f}"
            )
            logger.error(record.stopped)
            return
        bank_id = f"pb3-{index:02d}-{uuid.uuid4().hex[:6]}"
        row = await _run_scenario(http, scenario, bank_id)
        record.scenarios.append(row)
        logger.info(
            "%s: %d memories, %d LLM calls, $%.4f%s",
            scenario.id,
            len(row.memories),
            row.llm_calls,
            row.cost_usd,
            f" — {row.error}" if row.error else "",
        )
        write_record(record, partial)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _new_record(config: RunConfig, env: Mapping[str, str], pg0_name: str, port: int) -> RunRecord:
    nonce = uuid.uuid4().hex[:6]
    return RunRecord(
        suite=config.suite,
        side=config.side,
        label=config.label,
        data_sha256=data_sha256(),
        api_dir=str(config.api_dir),
        api_sha=_git(config.api_dir, "rev-parse", "HEAD"),
        api_dirty=bool(_git(config.api_dir, "status", "--porcelain", "--", ".")),
        harness_sha=_git(Path(__file__).resolve().parent, "rev-parse", "HEAD"),
        harness_dirty=bool(_git(Path(__file__).resolve().parent, "status", "--porcelain", "--", ".")),
        profile=config.profile,
        server_env=_redacted(env),
        database_url=f"pg0://{pg0_name}:{port}",
        database_schema=f"protb_{config.suite}_{re.sub(r'[^a-z0-9]+', '_', config.label.lower())[:20]}_{nonce}",
        bank_id=f"pb-{config.suite}-{nonce}" if config.suite != "p3" else None,
        passes=list(config.resolved_passes()),
        recall_options=dict(INTEGRATION_RECALL_OPTIONS) if config.suite != "p3" else {},
        started_at=datetime.now(timezone.utc).isoformat(),
    )


async def run_benchmark(config: RunConfig, environ: Mapping[str, str] | None = None) -> RunRecord:
    environ = os.environ if environ is None else environ
    # The protocol freezes the data before any run: a run on anything else is refused outright.
    frozen = frozen_sha256()
    if frozen is None:
        raise BenchmarkError(f"no frozen data hash at {FROZEN_HASH_FILE}")
    if frozen != data_sha256():
        raise BenchmarkError(f"data.py hashes {data_sha256()}, the frozen hash is {frozen}")
    env = server_env(
        config.suite,
        {**profile_env(config.profile, environ), **config.server_env},
        anthropic_key=environ.get(ENV_ANTHROPIC_KEY) if config.suite == "p3" else None,
    )
    check_config(config, env)
    runs_dir = config.out_dir / "runs"
    partial = runs_dir / f"{config.label}.partial.json"
    final = runs_dir / f"{config.label}.json"
    if final.exists():
        raise BenchmarkError(f"{final} exists; choose another --label")
    resumed: RunRecord | None = None
    if partial.exists():
        if config.resume and config.suite == "p3":
            resumed = load_record(partial)
        else:
            # Kept, not overwritten: what an aborted run spent still counts against the caps.
            aborted = runs_dir / f"{config.label}.aborted-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
            partial.rename(aborted)
            logger.warning("moved an earlier partial run to %s", aborted.name)
    prior = prior_usage(runs_dir, exclude=partial if resumed is not None else None)

    dead = dead_url() if config.dead_endpoint else None
    dead_keys = typesafe_base_url_keys(env) if dead else []
    proxy_keys = [] if dead else typesafe_base_url_keys(env)
    proxy = _TypeSafeProxy(TYPESAFE_UPSTREAM) if proxy_keys else None
    if dead:
        env.update(dict.fromkeys(dead_keys, dead))
    if proxy is not None:
        env.update(dict.fromkeys(proxy_keys, proxy.url))

    pg0_name = config.pg0_name or f"hindsight-protbench-{config.suite}-{config.side}"
    if resumed is not None:
        record = resumed
        port = int(record.database_url.rsplit(":", 1)[1])
        if resumed.data_sha256 != data_sha256():
            raise BenchmarkError("the partial run was made with other data")
    else:
        port = free_port()
        record = _new_record(config, env, pg0_name, port)
    record.dead_endpoint = dead
    record.dead_endpoint_keys = dead_keys

    started = time.monotonic()
    log_dir = config.out_dir / "logs" / config.label
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_API_DIR] = str(config.api_dir)
    os.environ[ENV_DATABASE_URL] = record.database_url
    stub = start_stub_server(Stubs())
    if proxy is not None:
        proxy.start()
    server = None
    try:
        server = start_hindsight_server(
            stub_url=stub.url,
            log_path=log_dir / "server.log",
            extra_env={**env, "HINDSIGHT_API_DATABASE_SCHEMA": record.database_schema},
        )
        async with httpx.AsyncClient(base_url=server.url, timeout=600) as http:
            if config.suite == "p3":
                await _run_p3(config, record, http, prior.anthropic_usd, partial)
            else:
                await _run_recall_suite(config, record, http, proxy, prior.jev_requests, partial)
    finally:
        if server is not None:
            server.stop()
        if proxy is not None:
            proxy.stop()
            record.typesafe_calls = list(proxy.calls)
        stub.stop()
        stop_pg0(pg0_name, config.api_dir)
    record.finished_at = datetime.now(timezone.utc).isoformat()
    record.duration_s = round((record.duration_s or 0) + time.monotonic() - started, 1)
    return record


def run(config: RunConfig) -> RunRecord:
    record = asyncio.run(run_benchmark(config))
    runs_dir = config.out_dir / "runs"
    partial = runs_dir / f"{config.label}.partial.json"
    if record.stopped or any(row.error for row in record.scenarios):
        # A stopped run, or a P3 run with a scenario whose setup failed, stays partial: the report
        # never reads it as complete, and --resume re-runs only what is missing or errored.
        write_record(record, partial)
    else:
        write_record(record, runs_dir / f"{config.label}.json")
        partial.unlink(missing_ok=True)
    return record


def failed_recalls(record: RunRecord) -> list[RecallRow]:
    return [r for r in record.recalls if r.status != 200]
