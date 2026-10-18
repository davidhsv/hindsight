"""Run the harm benchmark: build the bank, recall per condition, answer, judge, record.

Conditions:

- ``none`` — the prompt is the task alone; no server is needed.
- ``ungated`` — the top-10 results of a raw recall (query = task text, budget ``mid``, no tags,
  no context) are injected as the Claude Code integration injects them.
- ``gated`` — the same recall plus ``purpose=auto_context`` and the task's request ``context``,
  against a server whose ``HINDSIGHT_API_CONTEXT_ADMISSION_MODE`` is ``strict``. An empty result
  injects nothing, so the prompt is then the no-memory prompt.
- ``ungated-nopreamble`` (exploratory) — ``ungated``'s recall and block without the integration's
  "ignore irrelevant memories" preamble, to measure what that instruction does.

One run is one server build. Recalls are deterministic, so each (task, condition) is recalled
once and its context is reused by every repeat; the repeats differ only by the model's sampling.
The server is stopped before any model call. Every answer and verdict is checkpointed, so a run
that dies — or stops at its budget — can be resumed without paying twice.

A ``gated`` recall also records what the gate did: the response's ``context_admission`` summary,
and the per-candidate decisions of a replay of the same request with ``trace`` on. The replay is
diagnostic only (what is injected always comes from the untraced request, as registered), and its
result list is kept so the report can show whether it reproduced the injected one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..context_bench.runner import (
    ADMISSION_DECISION_MODEL_PROVIDER,
    BASE_ENV,
    LOCAL_MODEL_ENV,
    TYPESAFE_UPSTREAM,
    AdmissionRow,
    BenchmarkError,
    TypeSafeCall,
    _git,
    _redacted,
    _TypeSafeProxy,
    parse_admission,
)
from ..rulebook import Stubs
from ..server import ENV_API_DIR, ENV_DATABASE_URL, start_hindsight_server, start_stub_server
from .llm import (
    ANSWER_MODEL,
    JUDGE_MODEL,
    JUDGE_SYSTEM,
    VERDICT_SCHEMA,
    Completion,
    InjectedMemory,
    LLMError,
    MessagesClient,
    ModelSettings,
    Verdict,
    answer_prompt,
    injected_tokens,
    judge_prompt,
    parse_verdict,
)
from .model import QUERY_TIME, Memory, Task
from .suites import LABEL_BY_DOCUMENT, SUITES

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

Condition = Literal["none", "ungated", "ungated-nopreamble", "gated"]
CONDITIONS: tuple[Condition, ...] = ("none", "ungated", "ungated-nopreamble", "gated")
MEMORY_CONDITIONS: tuple[Condition, ...] = ("ungated", "ungated-nopreamble", "gated")
#: Conditions that send another condition's recall request (and so can reuse its recorded results).
RECALL_AS: dict[str, Condition] = {"ungated-nopreamble": "ungated"}
NO_PREAMBLE: frozenset[str] = frozenset({"ungated-nopreamble"})
#: Not pre-registered; reported with this note next to every number.
EXPLORATORY: dict[str, str] = {"ungated-nopreamble": "exploratory, added after the ungated result was seen"}

ENV_ADMISSION_MODE = "HINDSIGHT_API_CONTEXT_ADMISSION_MODE"
GATED_MODE = "strict"
ENV_ADMISSION_DECISION_MODEL_BASE_URL = "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_BASE_URL"

DEFAULT_PG0 = "hindsight-harmbench:15541"
DEFAULT_TOP_K = 10
DEFAULT_BUDGET_USD = 6.0

#: Each model at its own default effort (Haiku 5.5: medium; Sonnet 5.5: high), stated explicitly so
#: a default change cannot move the benchmark. Thinking is the models' default (adaptive).
DEFAULT_ANSWER = ModelSettings(model=ANSWER_MODEL, effort="medium", max_tokens=8000)
DEFAULT_JUDGE = ModelSettings(model=JUDGE_MODEL, effort="high", max_tokens=8000)

#: Held back from the budget for calls already in flight and for the price of a long answer.
BUDGET_MARGIN_USD = 0.25


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    label: str
    """Scenario label; ``?<memory id>`` for a memory the scenario did not write."""
    server_rank: int
    final: float | None = None
    reranker: float | None = None


class RecallRow(BaseModel):
    task_id: str
    condition: str
    status: int
    latency_ms: float
    results: list[ResultRow]
    injected: list[InjectedMemory]
    """The first ``top_k`` results, as the prompt shows them."""
    error: str | None = None
    admission_summary: dict[str, Any] | None = None
    """``gated``: the response's ``context_admission`` summary (abstained, candidates, admitted)."""
    admission: AdmissionRow | None = None
    """``gated``: the gate's per-candidate decisions, from the traced replay of the same request."""
    traced_labels: list[str] | None = None
    """``gated``: the traced replay's result labels; equal to ``results``' when the replay reproduced
    the injected recall, so its decisions describe what was injected."""


class AnswerRow(BaseModel):
    task_id: str
    kind: str
    condition: str
    repeat: int
    injected_labels: list[str]
    injected_tokens: int
    answer: Completion | None = None
    judge: Completion | None = None
    verdict: Verdict | None = None
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.condition, self.repeat)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in (self.answer, self.judge) if c is not None)


class RunRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    label: str
    suite: str = "original"
    api_dir: str | None
    api_sha: str | None
    api_dirty: bool | None
    harness_sha: str | None
    server_env: dict[str, str]
    conditions: list[str]
    repeats: int
    top_k: int
    recall_max_tokens: int
    answer_settings: dict[str, Any]
    judge_settings: dict[str, Any]
    database_url: str | None = None
    database_schema: str | None = None
    bank_id: str | None = None
    bank_size: int | None = None
    budget_usd: float
    started_at: str
    finished_at: str | None = None
    duration_s: float | None = None
    stopped_for_budget: bool = False
    recalls: list[RecallRow] = Field(default_factory=list)
    answers: list[AnswerRow] = Field(default_factory=list)
    unlabelled_texts: dict[str, str] = Field(default_factory=dict)
    recalls_from: str | None = None
    """The run file the recalls were copied from, when they were not made by this run."""
    harness_dirty: bool | None = None
    """Whether the harness package had uncommitted changes, so ``harness_sha`` alone does not name it."""
    memory_labels: dict[str, str] = Field(default_factory=dict)
    """Server memory id -> scenario label, from the bank listing (the trace names memories by id)."""
    admission_settings: dict[str, Any] = Field(default_factory=dict)
    """The bank's resolved ``context_admission_*`` / ``adaptive_memory_mode`` config, as the server
    reported it after ingest (what the gated recalls ran under)."""
    typesafe_calls: list[TypeSafeCall] = Field(default_factory=list)
    """Decision-model requests made through the usage-recording proxy (``typesafe_proxy``)."""

    @property
    def spend_usd(self) -> float:
        return sum(row.cost_usd for row in self.answers)

    def recall_for(self, task_id: str, condition: str) -> RecallRow | None:
        return next((r for r in self.recalls if r.task_id == task_id and r.condition == condition), None)


@dataclass(frozen=True)
class RunConfig:
    label: str
    conditions: tuple[Condition, ...]
    out_dir: Path
    suite: str = "original"
    api_dir: Path | None = None
    server_env: Mapping[str, str] = field(default_factory=dict)
    repeats: int = 2
    top_k: int = DEFAULT_TOP_K
    recall_max_tokens: int = 4096
    pg0: str = DEFAULT_PG0
    budget_usd: float = DEFAULT_BUDGET_USD
    concurrency: int = 6
    answer: ModelSettings = DEFAULT_ANSWER
    judge: ModelSettings = DEFAULT_JUDGE
    task_ids: tuple[str, ...] | None = None
    resume: Path | None = None
    reuse_recalls: Path | None = None
    """A run file whose recorded recalls this run injects instead of recalling again (same bank,
    same build, byte-identical top-k); see :data:`RECALL_AS`."""
    recall_only: bool = False
    """Build the bank and record the recalls, but make no model call (a free construction check)."""
    typesafe_proxy: bool = False
    """Send the admission decision model's TypeSafe requests through context_bench's usage-recording
    proxy (retries 429/5xx), as context_bench's admission arms were run."""


def check_conditions(conditions: Sequence[str], server_env: Mapping[str, str]) -> None:
    """``gated`` needs a server in an admission mode, which would also gate ``ungated`` requests,
    so the two never share a run."""
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; choose from {CONDITIONS}")
    if "gated" in conditions and any(RECALL_AS.get(c, c) == "ungated" for c in conditions):
        raise ValueError("gated and ungated need different server configurations; run them separately")
    if "gated" in conditions and server_env.get(ENV_ADMISSION_MODE, "off") == "off":
        raise ValueError(f"gated needs {ENV_ADMISSION_MODE} (e.g. {GATED_MODE}) in the server env")
    if "gated" not in conditions and server_env.get(ENV_ADMISSION_MODE, "off") != "off":
        raise ValueError(f"{ENV_ADMISSION_MODE} is set but no gated condition was asked for")


def recall_body(task: Task, condition: str, *, max_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": task.text,
        "budget": "mid",
        "max_tokens": max_tokens,
        "query_timestamp": QUERY_TIME.strftime("%Y-%m-%dT%H:%M:%S"),
        "include": {"entities": None},
    }
    request = RECALL_AS.get(condition, condition)
    if request == "gated":
        body["purpose"] = "auto_context"
        body["context"] = task.context()
    elif request != "ungated":
        raise ValueError(f"condition {condition!r} does not recall")
    return body


# ---------------------------------------------------------------------------
# Bank + recall
# ---------------------------------------------------------------------------


async def _ingest(
    http: httpx.AsyncClient, bank_id: str, memories: Sequence[Memory], memory_labels: dict[str, str] | None = None
) -> int:
    response = await http.put(f"/v1/default/banks/{bank_id}", json={})
    response.raise_for_status()
    response = await http.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={"updates": {"retain_extraction_mode": "chunks", "enable_observations": False}},
    )
    response.raise_for_status()
    items = [
        {
            "content": memory.text,
            "document_id": memory.document_id,
            "timestamp": memory.timestamp.isoformat(),
            "tags": memory.tags,
            "metadata": memory.metadata,
        }
        for memory in memories
    ]
    for start in range(0, len(items), 25):
        response = await http.post(f"/v1/default/banks/{bank_id}/memories", json={"items": items[start : start + 25]})
        if response.status_code != 200:
            raise BenchmarkError(f"retain into {bank_id} -> {response.status_code}: {response.text[:500]}")
    return await _verify_bank(http, bank_id, memories, memory_labels)


async def _verify_bank(
    http: httpx.AsyncClient, bank_id: str, memories: Sequence[Memory], memory_labels: dict[str, str] | None = None
) -> int:
    """The bank holds exactly one memory per scenario document, and nothing else. Fills
    ``memory_labels`` (server memory id -> scenario label) when given."""
    response = await http.get(f"/v1/default/banks/{bank_id}/memories/list", params={"limit": 1000})
    response.raise_for_status()
    items = response.json()["items"]
    documents = sorted(item.get("document_id") or "" for item in items)
    expected = sorted(memory.document_id for memory in memories)
    if documents != expected:
        raise BenchmarkError(
            f"bank {bank_id}: {len(items)} memories for {len(expected)} documents "
            f"(missing {len(set(expected) - set(documents))}, extra {len(documents) - len(expected)})"
        )
    if memory_labels is not None:
        for item in items:
            memory_labels[str(item["id"])] = LABEL_BY_DOCUMENT[item["document_id"]]
    return len(items)


async def _admission_config(http: httpx.AsyncClient, bank_id: str) -> dict[str, Any]:
    """The bank's resolved admission and adaptive-memory config, as the server reports it."""
    response = await http.get(f"/v1/default/banks/{bank_id}/config")
    response.raise_for_status()
    config = response.json()["config"]
    return {
        key: value
        for key, value in sorted(config.items())
        if key.startswith(("context_admission_", "adaptive_memory_mode"))
    }


def parse_results(
    payload: Mapping[str, Any], top_k: int, unlabelled: dict[str, str]
) -> tuple[list[ResultRow], list[InjectedMemory]]:
    rows: list[ResultRow] = []
    injected: list[InjectedMemory] = []
    for rank, result in enumerate(payload.get("results", []), start=1):
        label = LABEL_BY_DOCUMENT.get(result.get("document_id") or "")
        if label is None:
            label = f"?{result['id']}"
            unlabelled[label] = result.get("text", "")
        scores = result.get("scores") or {}
        rows.append(
            ResultRow(label=label, server_rank=rank, final=scores.get("final"), reranker=scores.get("reranker"))
        )
        if rank <= top_k:
            injected.append(
                InjectedMemory(
                    label=label,
                    text=result.get("text", ""),
                    type=result.get("type"),
                    mentioned_at=result.get("mentioned_at"),
                )
            )
    return rows, injected


async def _recall(
    http: httpx.AsyncClient, bank_id: str, task: Task, condition: str, config: RunConfig, record: RunRecord
) -> RecallRow:
    body = recall_body(task, condition, max_tokens=config.recall_max_tokens)
    started = time.monotonic()
    response = await http.post(f"/v1/default/banks/{bank_id}/memories/recall", json=body)
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    if response.status_code != 200:
        logger.error("%s %s -> %d: %s", condition, task.id, response.status_code, response.text[:300])
        return RecallRow(
            task_id=task.id,
            condition=condition,
            status=response.status_code,
            latency_ms=latency_ms,
            results=[],
            injected=[],
            error=response.text[:500],
        )
    payload = response.json()
    rows, injected = parse_results(payload, config.top_k, record.unlabelled_texts)
    row = RecallRow(
        task_id=task.id, condition=condition, status=200, latency_ms=latency_ms, results=rows, injected=injected
    )
    if RECALL_AS.get(condition, condition) == "gated":
        row.admission_summary = payload.get("context_admission")
        traced = await http.post(f"/v1/default/banks/{bank_id}/memories/recall", json={**body, "trace": True})
        if traced.status_code != 200:
            raise BenchmarkError(f"traced replay of {task.id} -> {traced.status_code}: {traced.text[:300]}")
        traced_payload = traced.json()
        traced_rows, _ = parse_results(traced_payload, config.top_k, record.unlabelled_texts)
        row.traced_labels = [r.label for r in traced_rows]
        row.admission = parse_admission(traced_payload, record.memory_labels)
    return row


def _server_env(config: RunConfig) -> dict[str, str]:
    return {**BASE_ENV, **LOCAL_MODEL_ENV, **config.server_env}


async def _build_and_recall(config: RunConfig, record: RunRecord, tasks: Sequence[Task]) -> None:
    """Start the server, ingest the bank, record every (task, memory condition) recall, stop."""
    if config.api_dir is None:
        raise ValueError("a memory condition needs --api-dir")
    nonce = uuid.uuid4().hex[:6]
    record.database_url = f"pg0://{config.pg0}"
    record.database_schema = f"harmb_{re.sub(r'[^a-z0-9]+', '_', config.label.lower())[:24]}_{nonce}"
    record.bank_id = f"harmb-{nonce}"
    log_dir = config.out_dir / "logs" / config.label
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_API_DIR] = str(config.api_dir)
    os.environ[ENV_DATABASE_URL] = record.database_url
    env = _server_env(config)
    proxy = _TypeSafeProxy(TYPESAFE_UPSTREAM) if config.typesafe_proxy else None
    if proxy is not None:
        if env.get(ADMISSION_DECISION_MODEL_PROVIDER) != "typesafe":
            raise BenchmarkError(f"typesafe_proxy needs {ADMISSION_DECISION_MODEL_PROVIDER}=typesafe in the server env")
        env[ENV_ADMISSION_DECISION_MODEL_BASE_URL] = proxy.url
        record.server_env = _redacted(env)
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
        async with httpx.AsyncClient(base_url=server.url, timeout=300) as http:
            record.bank_size = await _ingest(http, record.bank_id, SUITES[config.suite].memories, record.memory_labels)
            record.admission_settings = await _admission_config(http, record.bank_id)
            logger.info("bank %s ready: %d memories; %s", record.bank_id, record.bank_size, record.admission_settings)
            for condition in config.conditions:
                if condition not in MEMORY_CONDITIONS:
                    continue
                for task in tasks:
                    record.recalls.append(await _recall(http, record.bank_id, task, condition, config, record))
    finally:
        if server is not None:
            server.stop()
        if proxy is not None:
            proxy.stop()
            record.typesafe_calls = list(proxy.calls)
        stub.stop()
    failed = [r for r in record.recalls if r.status != 200]
    if failed:
        raise BenchmarkError(f"{len(failed)} recall(s) failed, e.g. {failed[0].task_id}: {failed[0].error}")


# ---------------------------------------------------------------------------
# Answer + judge
# ---------------------------------------------------------------------------


class _Budget:
    """Stops new work once spend plus what is in flight could cross the cap."""

    def __init__(self, cap_usd: float, spent_usd: float) -> None:
        self.cap = cap_usd
        self.spent = spent_usd
        self.in_flight = 0
        self.max_pair = 0.05
        self.exhausted = False

    def admit(self) -> bool:
        if self.spent + (self.in_flight + 1) * self.max_pair + BUDGET_MARGIN_USD > self.cap:
            self.exhausted = True
            return False
        self.in_flight += 1
        return True

    def settle(self, cost: float) -> None:
        self.in_flight -= 1
        self.spent += cost
        self.max_pair = max(self.max_pair, cost)


async def answer_and_judge(
    client: MessagesClient, config: RunConfig, task: Task, condition: str, repeat: int, injected: list[InjectedMemory]
) -> AnswerRow:
    row = AnswerRow(
        task_id=task.id,
        kind=task.kind,
        condition=condition,
        repeat=repeat,
        injected_labels=[m.label for m in injected],
        injected_tokens=injected_tokens(injected, preamble=condition not in NO_PREAMBLE),
    )
    system, user = answer_prompt(task, injected, preamble=condition not in NO_PREAMBLE)
    try:
        row.answer = await client.create(config.answer, system=system, user=user)
        if row.answer.stop_reason == "refusal" or not row.answer.text.strip():
            # A refusal or an empty answer fails the task; there is nothing for the judge to grade.
            row.verdict = Verdict(
                success=False, harm=False, unmet=["no answer"], reason=f"stop_reason={row.answer.stop_reason}"
            )
            return row
        row.judge = await client.create(
            config.judge, system=JUDGE_SYSTEM, user=judge_prompt(task, row.answer.text), json_schema=VERDICT_SCHEMA
        )
        row.verdict = parse_verdict(task, row.judge.text)
    except LLMError as e:
        row.error = str(e)
        logger.error("%s/%s/%d: %s", task.id, condition, repeat, e)
    return row


def _work(config: RunConfig, tasks: Sequence[Task], done: set[tuple[str, str, int]]) -> list[tuple[Task, str, int]]:
    """Repeat-major, then task, then condition, so a budget stop leaves every condition comparable."""
    return [
        (task, condition, repeat)
        for repeat in range(1, config.repeats + 1)
        for task in tasks
        for condition in config.conditions
        if (task.id, condition, repeat) not in done
    ]


async def _answer_all(config: RunConfig, record: RunRecord, tasks: Sequence[Task], partial: Path) -> None:
    done = {row.key for row in record.answers if row.verdict is not None}
    record.answers = [row for row in record.answers if row.verdict is not None]
    work = _work(config, tasks, done)
    budget = _Budget(config.budget_usd, record.spend_usd)
    semaphore = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()
    client = MessagesClient()

    async def one(task: Task, condition: str, repeat: int) -> None:
        async with semaphore:
            if not budget.admit():
                return
            recall = record.recall_for(task.id, condition)
            if condition in MEMORY_CONDITIONS and recall is None:
                raise BenchmarkError(f"{task.id}/{condition}: no recall recorded")
            injected = recall.injected if recall is not None else []
            row = await answer_and_judge(client, config, task, condition, repeat, injected)
            budget.settle(row.cost_usd)
            async with lock:
                record.answers.append(row)
                write_record(record, partial)
            logger.info(
                "%s/%s/%d success=%s harm=%s $%.4f (total $%.3f)",
                task.id,
                condition,
                repeat,
                row.verdict.success if row.verdict else None,
                row.verdict.harm if row.verdict else None,
                row.cost_usd,
                budget.spent,
            )

    try:
        await asyncio.gather(*(one(task, condition, repeat) for task, condition, repeat in work))
    finally:
        await client.aclose()
    record.stopped_for_budget = budget.exhausted


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def reused_recalls(
    source: RunRecord, conditions: Sequence[str], task_ids: Sequence[str], suite: str = "original"
) -> list[RecallRow]:
    """``source``'s recorded recalls for each memory condition, relabelled with that condition."""
    if source.suite != suite:
        raise BenchmarkError(f"{source.label} recalled the {source.suite} suite's bank, not {suite}'s")
    rows: list[RecallRow] = []
    for condition in conditions:
        if condition not in MEMORY_CONDITIONS:
            continue
        sent_as = RECALL_AS.get(condition, condition)
        for task_id in task_ids:
            recall = source.recall_for(task_id, sent_as)
            if recall is None or recall.status != 200:
                raise BenchmarkError(f"{source.label} has no successful {sent_as} recall for {task_id}")
            rows.append(recall.model_copy(update={"condition": condition}))
    return rows


def _adopt_recalls(record: RunRecord, source_path: Path, conditions: Sequence[str], task_ids: Sequence[str]) -> None:
    source = load_record(source_path)
    record.recalls = reused_recalls(source, conditions, task_ids, record.suite)
    record.recalls_from = str(source_path)
    record.unlabelled_texts = dict(source.unlabelled_texts)
    record.api_dir = source.api_dir
    record.api_sha = source.api_sha
    record.api_dirty = source.api_dirty
    record.server_env = source.server_env
    record.database_url = source.database_url
    record.database_schema = source.database_schema
    record.bank_id = source.bank_id
    record.bank_size = source.bank_size
    record.memory_labels = dict(source.memory_labels)
    record.admission_settings = dict(source.admission_settings)


def _new_record(config: RunConfig) -> RunRecord:
    return RunRecord(
        label=config.label,
        suite=config.suite,
        api_dir=str(config.api_dir) if config.api_dir else None,
        api_sha=_git(config.api_dir, "rev-parse", "HEAD") if config.api_dir else None,
        api_dirty=bool(_git(config.api_dir, "status", "--porcelain", "--", ".")) if config.api_dir else None,
        harness_sha=_git(Path(__file__).resolve().parent, "rev-parse", "HEAD"),
        harness_dirty=bool(_git(Path(__file__).resolve().parents[1], "status", "--porcelain", "--", ".")),
        server_env=_redacted(_server_env(config)) if config.api_dir else {},
        conditions=list(config.conditions),
        repeats=config.repeats,
        top_k=config.top_k,
        recall_max_tokens=config.recall_max_tokens,
        answer_settings=asdict(config.answer),
        judge_settings=asdict(config.judge),
        budget_usd=config.budget_usd,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


async def run_benchmark(config: RunConfig) -> RunRecord:
    check_conditions(config.conditions, config.server_env)
    if config.suite not in SUITES:
        raise ValueError(f"unknown suite {config.suite!r}; choose from {list(SUITES)}")
    tasks = [task for task in SUITES[config.suite].tasks if config.task_ids is None or task.id in config.task_ids]
    started = time.monotonic()
    record = load_record(config.resume) if config.resume else _new_record(config)
    if config.resume and (
        record.conditions != list(config.conditions) or record.repeats != config.repeats or record.suite != config.suite
    ):
        raise ValueError("--resume must use the run's own suite, conditions and repeats")
    partial = config.out_dir / "runs" / f"{config.label}.partial.json"
    needs_recall = [c for c in config.conditions if c in MEMORY_CONDITIONS]
    if needs_recall and not record.recalls:
        if config.reuse_recalls:
            _adopt_recalls(record, config.reuse_recalls, config.conditions, [task.id for task in tasks])
        else:
            await _build_and_recall(config, record, tasks)
        write_record(record, partial)
    if not config.recall_only:
        await _answer_all(config, record, tasks, partial)
    record.finished_at = datetime.now(timezone.utc).isoformat()
    record.duration_s = round((record.duration_s or 0) + time.monotonic() - started, 1)
    return record


def write_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1))


def load_record(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text())


def run(config: RunConfig) -> RunRecord:
    record = asyncio.run(run_benchmark(config))
    write_record(record, config.out_dir / "runs" / f"{config.label}.json")
    (config.out_dir / "runs" / f"{config.label}.partial.json").unlink(missing_ok=True)
    return record
