"""Run the memory_lifecycle scenarios against one server build and record every row.

One run = one arm: a ``hindsight-api-slim`` checkout, a set of server environment
variables, optional per-bank config, the stub LLM answering from a payload source, and
either the shipping local embedder + cross-encoder or the stub's lexical pair. Every
scenario gets a bank of its own; the run gets a database schema of its own on a pg0
instance of its own, so neither a concurrent suite nor a previous run can leak in.

Everything the server is asked goes through its public HTTP API — retain through the
published client, recall as a raw HTTP body so arms can receive request fields the base
server does not know (``lifecycle_view``, ``as_of``; FastAPI ignores unknown fields).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from hindsight_client import Hindsight
from pydantic import BaseModel, ConfigDict, Field

from ..payloads import extracted, fact
from ..rulebook import ChatRequest, ChatRule, StubbedReply, Stubs
from ..server import API_DIR, ENV_API_DIR, ENV_DATABASE_URL, start_hindsight_server, start_stub_server
from ..steps import anchor_for
from ..waiting import wait_until_settled
from .encoding import (
    BaseEncoding,
    BatchView,
    LifecycleForm,
    NeedleReply,
    SchemaCapabilities,
    resolve,
)
from .labels import QueryLabels, RankedItem, classify
from .metrics import StateConformance, aggregate, score_query, state_conformance
from .model import Item, Query, Scenario, Step, at_day
from .scenarios import SCENARIOS, holdout_ids
from .sources import AuthoredSource, FrozenSource, PayloadError, PayloadSource, statement

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Results kept per query row. Metrics read at most the top 10; the rest is for diagnosis.
ROW_DEPTH = 20

#: Final scores closer than this are a tie. Measured: two runs of the same arm with the local
#: CPU models return identical-text raw facts at 0.266891 / 0.266892 in either order, so the
#: server's order inside such a group is float noise, not ranking. Within a tie the harness
#: orders by (text, labels) so rows are reproducible; the server's own position is kept as
#: ``server_rank``.
SCORE_TIE_TOLERANCE = 1e-5

#: The dedup adjudicator's prompt (``consolidator._DEDUP_PROMPT``). It fires when a new or
#: rewritten observation is >= 0.97 cosine-similar to an existing one — real embeddings make
#: that reachable, so the scenario has to say whether the pair is the same fact.
DEDUP_ANCHOR = "You reconcile long-term memory observations."

#: The shipping local models (``DEFAULT_EMBEDDINGS_LOCAL_MODEL`` /
#: ``DEFAULT_RERANKER_LOCAL_MODEL``), pinned to CPU: an MPS/CUDA kernel may reorder float
#: reductions between runs, and a benchmark row has to be reproducible bit for bit.
LOCAL_MODEL_ENV = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_RERANKER_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU": "true",
    "HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU": "true",
    # The models are read from the local HF cache; a run must not depend on the network.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

#: Always: ``uv run`` would otherwise sync the checkout's environment to the default extras
#: — slow, and destructive for a venv that was synced with ``--all-extras`` (test_76 does the
#: same).
BASE_ENV = {"UV_NO_SYNC": "1"}

DEFAULT_PG0_INSTANCE = "hindsight-lcbench"
DEFAULT_PG0_PORT = 15633


class BenchmarkError(Exception):
    """A scenario could not be executed as scripted; its numbers would mean nothing."""


@dataclass(frozen=True)
class RunConfig:
    label: str
    arm: str
    api_dir: Path
    encoding: BaseEncoding
    lifecycle_form: LifecycleForm
    lexical: bool = False
    server_env: Mapping[str, str] = field(default_factory=dict)
    bank_config: Mapping[str, Any] = field(default_factory=dict)
    recall_types: tuple[str, ...] | None = None
    fixtures: Path | None = None
    scenario_ids: tuple[str, ...] | None = None
    pg0_instance: str = DEFAULT_PG0_INSTANCE
    pg0_port: int = DEFAULT_PG0_PORT
    log_dir: Path = Path("lifecycle-bench-logs")
    keep_banks: bool = False
    keep_going: bool = False


# ---------------------------------------------------------------------------
# Output records (the run JSON)
# ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    rank: int
    server_rank: int
    memory_id: str
    type: str
    text: str
    docs: list[str]
    tags: list[str]
    cls: str
    score: float | None = None


class QueryRow(BaseModel):
    scenario: str
    query_id: str
    split: str
    kind: str
    text: str
    request: dict[str, Any]
    """The recall body as sent (a JSON blob, recorded verbatim)."""
    results: list[ResultRow]
    metrics: dict[str, float | None]


class CallRecord(BaseModel):
    """One consolidation call: what it showed, what the payload source decided, what was sent."""

    step: int
    batch_facts: list[str]
    fact_ids: list[str]
    pool: list[str]
    schema_found: bool
    lifecycle_schema: bool
    reply: NeedleReply
    wire: dict[str, Any]
    degradations: list[str]
    notes: list[str] = Field(default_factory=list)


class DedupRecord(BaseModel):
    step: int
    new: str
    existing: str
    action: str


class StateRecord(BaseModel):
    observed: dict[str, list[str | None]]
    """Document → truth_status of each of its raw facts, as GET memory reported it (``None``:
    no ``lifecycle`` key; ``implicit``: a lifecycle key with no claim)."""
    observation_statuses: dict[str, int]
    """Observation count by reported truth_status (``none`` when not reported)."""
    conformance: float | None = None
    retirement_recall: float | None = None
    false_retirement: float | None = None
    n_expected: int = 0
    n_reported: int = 0


class ScenarioRecord(BaseModel):
    id: str
    title: str
    split: str
    covers: list[str]
    bank_id: str
    calls: list[CallRecord] = Field(default_factory=list)
    dedup: list[DedupRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    state: StateRecord | None = None
    duration_s: float = 0.0


class MetricValue(BaseModel):
    mean: float | None
    n: int


class RunRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: int = SCHEMA_VERSION
    benchmark: str = "memory_lifecycle"
    label: str
    arm: str
    api_dir: str
    api_sha: str | None
    api_dirty: bool | None
    harness_sha: str | None
    harness_dirty: bool | None
    encoding: str
    lifecycle_form: str
    payload_source: str
    embedder: str
    server_env: dict[str, str]
    bank_config: dict[str, Any]
    recall_types: list[str] | None
    holdout: list[str]
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    invalid_scenarios: list[str] = Field(default_factory=list)
    scenarios: list[ScenarioRecord] = Field(default_factory=list)
    rows: list[QueryRow] = Field(default_factory=list)
    metrics: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    """Split (``all`` / ``dev`` / ``holdout``) → ``name@k`` → summary."""
    per_scenario: dict[str, dict[str, MetricValue]] = Field(default_factory=dict)
    state_conformance: dict[str, MetricValue] = Field(default_factory=dict)
    """Split → E-only conformance summaries; every mean is N/A when no state was reported."""


def summarize(record: RunRecord) -> None:
    """Fill the aggregate sections of ``record`` from its rows (and nothing else)."""
    valid = [row for row in record.rows if row.scenario not in record.invalid_scenarios]
    splits = {
        "all": valid,
        "dev": [r for r in valid if r.split == "dev"],
        "holdout": [r for r in valid if r.split == "holdout"],
    }
    record.metrics = {
        name: {key: MetricValue(mean=s.mean, n=s.n) for key, s in aggregate(r.metrics for r in rows).items()}
        for name, rows in splits.items()
    }
    record.per_scenario = {
        scenario.id: {
            key: MetricValue(mean=s.mean, n=s.n)
            for key, s in aggregate(r.metrics for r in valid if r.scenario == scenario.id).items()
        }
        for scenario in record.scenarios
    }
    expected = {scenario.id: scenario.expected_states for scenario in SCENARIOS}
    claims_reported = any(
        status not in (None, IMPLICIT)
        for s in record.scenarios
        if s.state is not None
        for statuses in s.state.observed.values()
        for status in statuses
    ) or any(
        key not in ("none", IMPLICIT)
        for s in record.scenarios
        if s.state is not None
        for key in s.state.observation_statuses
    )
    for s in record.scenarios:
        if s.state is None:
            continue
        implicit_as = "active" if claims_reported else None
        observed = {
            doc: [implicit_as if status == IMPLICIT else status for status in statuses]
            for doc, statuses in s.state.observed.items()
        }
        scored: StateConformance = state_conformance(expected.get(s.id, {}), observed)
        s.state.conformance = scored.conformance
        s.state.retirement_recall = scored.retirement_recall
        s.state.false_retirement = scored.false_retirement
        s.state.n_expected = scored.n_expected
        s.state.n_reported = scored.n_reported

    record.state_conformance = {}
    for name in ("all", "dev", "holdout"):
        states = [
            s.state
            for s in record.scenarios
            if s.state is not None and s.id not in record.invalid_scenarios and (name == "all" or s.split == name)
        ]
        for metric in ("conformance", "retirement_recall", "false_retirement"):
            values = [getattr(state, metric) for state in states if getattr(state, metric) is not None]
            record.state_conformance[f"{name}:{metric}"] = MetricValue(
                mean=sum(values) / len(values) if values else None, n=len(values)
            )


# ---------------------------------------------------------------------------
# Server wire views (parsed at the boundary)
# ---------------------------------------------------------------------------


class _Scores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    final: float | None = None


class _RecallItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str
    type: str | None = None
    document_id: str | None = None
    tags: list[str] | None = None
    source_fact_ids: list[str] | None = None
    scores: _Scores | None = None


class _RecallResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[_RecallItem]


class _MemoryRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    text: str = ""
    fact_type: str | None = None
    document_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)


class _MemoryList(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_MemoryRow]


class _DedupDecision(BaseModel):
    action: str
    text: str
    reason: str


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _schema_name(label: str) -> str:
    """A fresh Postgres schema per run: migrations of one checkout never meet another's."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:30]
    return f"lcb_{slug}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"


def server_environment(config: RunConfig) -> dict[str, str]:
    env = dict(BASE_ENV)
    if not config.lexical:
        env.update(LOCAL_MODEL_ENV)
    env.update(config.server_env)
    return env


def _payload_source(config: RunConfig) -> PayloadSource:
    if config.fixtures is not None:
        return FrozenSource(directory=config.fixtures)
    return AuthoredSource(encoding=config.encoding, lifecycle_form=config.lifecycle_form)


async def run_benchmark(config: RunConfig) -> RunRecord:
    scenarios = [s for s in SCENARIOS if config.scenario_ids is None or s.id in config.scenario_ids]
    if config.scenario_ids is not None:
        unknown = set(config.scenario_ids) - {s.id for s in scenarios}
        if unknown:
            raise BenchmarkError(f"unknown scenarios: {sorted(unknown)}")
    holdout = holdout_ids()
    harness_dir = Path(__file__).resolve().parents[2]
    env = server_environment(config)
    source = _payload_source(config)
    started = time.monotonic()
    record = RunRecord(
        label=config.label,
        arm=config.arm,
        api_dir=str(config.api_dir),
        api_sha=_git(config.api_dir, "rev-parse", "HEAD"),
        api_dirty=_dirty(config.api_dir),
        harness_sha=_git(harness_dir, "rev-parse", "HEAD"),
        harness_dirty=_dirty(harness_dir),
        encoding=config.encoding,
        lifecycle_form=config.lifecycle_form,
        payload_source=source.name if config.fixtures is None else f"frozen:{config.fixtures}",
        embedder="lexical-stub" if config.lexical else "local (bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2, CPU)",
        server_env=env,
        bank_config=dict(config.bank_config),
        recall_types=list(config.recall_types) if config.recall_types else None,
        holdout=sorted(holdout),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    stubs = Stubs()
    stub = start_stub_server(stubs)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_API_DIR] = str(config.api_dir)
    os.environ[ENV_DATABASE_URL] = f"pg0://{config.pg0_instance}:{config.pg0_port}"
    schema = _schema_name(config.label)
    logger.info("starting %s from %s (schema %s)", config.label, config.api_dir, schema)
    server = start_hindsight_server(
        stub_url=stub.url,
        log_path=config.log_dir / "server.log",
        extra_env={**env, "HINDSIGHT_API_DATABASE_SCHEMA": schema},
    )
    nonce = uuid.uuid4().hex[:6]
    try:
        client = Hindsight(base_url=server.url)
        async with httpx.AsyncClient(base_url=server.url, timeout=180) as http:
            try:
                for scenario in scenarios:
                    split = "holdout" if scenario.id in holdout else "dev"
                    runner = _ScenarioRunner(
                        config=config,
                        scenario=scenario,
                        split=split,
                        stubs=stubs,
                        client=client,
                        http=http,
                        source=source,
                    )
                    outcome = await runner.run(bank_id=f"lcb-{scenario.id.replace('_', '-')}-{nonce}")
                    scenario_record = outcome.record
                    record.scenarios.append(scenario_record)
                    record.rows.extend(outcome.rows)
                    if scenario_record.errors:
                        record.invalid_scenarios.append(scenario.id)
                        logger.error("%s invalid: %s", scenario.id, "; ".join(scenario_record.errors))
                        if not config.keep_going:
                            raise BenchmarkError(f"{scenario.id}: {'; '.join(scenario_record.errors)}")
                    logger.info("%s done in %.1fs", scenario.id, scenario_record.duration_s)
            finally:
                await client.aclose()
    finally:
        server.stop()
        stub.stop()

    record.finished_at = datetime.now(timezone.utc).isoformat()
    record.duration_s = round(time.monotonic() - started, 1)
    summarize(record)
    return record


def _dirty(path: Path) -> bool | None:
    status = _git(path, "status", "--porcelain", "--", ".")
    return None if status is None else bool(status)


@dataclass
class ScenarioOutcome:
    record: ScenarioRecord
    rows: list[QueryRow]


@dataclass
class _ScenarioRunner:
    config: RunConfig
    scenario: Scenario
    split: str
    stubs: Stubs
    client: Hindsight
    http: httpx.AsyncClient
    source: PayloadSource
    _foreign: dict[str, str] = field(default_factory=dict)
    _step: int = 0

    async def run(self, *, bank_id: str) -> ScenarioOutcome:
        started = time.monotonic()
        record = ScenarioRecord(
            id=self.scenario.id,
            title=self.scenario.title,
            split=self.split,
            covers=list(self.scenario.covers),
            bank_id=bank_id,
        )
        rows: list[QueryRow] = []
        try:
            await self.client.acreate_bank(bank_id=bank_id, name=self.scenario.title)
            if self.config.bank_config:
                await self.client.banks.update_bank_config(bank_id, {"updates": dict(self.config.bank_config)})
            for index, step in enumerate(self.scenario.steps):
                await self._run_step(bank_id, index, step, record)
                if record.errors:
                    return ScenarioOutcome(record=record, rows=rows)
            memories = await self._memories(bank_id)
            record.state = await self._states(bank_id, memories)
            for query in self.scenario.queries:
                rows.append(await self._query(bank_id, query, memories))
        except Exception as exc:  # noqa: BLE001 - recorded: it invalidates the scenario, loudly
            record.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            record.duration_s = round(time.monotonic() - started, 1)
            if not self.config.keep_banks:
                with contextlib.suppress(Exception):
                    await self.client.banks.delete_bank(bank_id)
        return ScenarioOutcome(record=record, rows=rows)

    # -- steps ---------------------------------------------------------------

    async def _run_step(self, bank_id: str, index: int, step: Step, record: ScenarioRecord) -> None:
        self.stubs.reset()
        self._step = index
        step_errors: list[str] = []
        llm = self.stubs.llm
        for item in step.items:
            llm.on_step("extract_facts", contains=item.content).returns(
                extracted(*(fact(text, entities=list(item.entities)) for text in item.facts))
            )
        llm.add_rule(
            ChatRule(contains=(anchor_for("consolidate"),), respond=self._consolidation_responder(record, step_errors))
        )
        llm.add_rule(ChatRule(contains=(DEDUP_ANCHOR,), respond=self._dedup_responder(record)))

        # Error-injection decisions name observations the call will not show; resolve them
        # against the whole bank as it stands before this step.
        self._foreign = {row.text: row.id for row in await self._memories(bank_id) if row.fact_type == "observation"}

        await self.client.aretain_batch(bank_id=bank_id, items=[_retain_item(item) for item in step.items])
        try:
            await wait_until_settled(self.client, bank_id)
        except AssertionError as exc:
            step_errors.append(f"step {index}: {exc}")

        for call in llm.unmatched:
            step_errors.append(f"step {index}: unmatched LLM call: {call.suggestion()}")
        for rejection in self.stubs.rejected_requests:
            step_errors.append(f"step {index}: stub rejected a request: {rejection}")
        expected_calls = [c for c in llm.calls if anchor_for("extract_facts") in c.all_text]
        if len(expected_calls) < len(step.items):
            step_errors.append(f"step {index}: {len(expected_calls)} extraction calls for {len(step.items)} items")
        record.errors.extend(step_errors)

    def _consolidation_responder(self, record: ScenarioRecord, errors: list[str]):
        def respond(request: ChatRequest) -> StubbedReply:
            try:
                batch = BatchView.from_request(request)
                caps = SchemaCapabilities.from_request(request)
                needles = self.source.consolidation(self.scenario, self._step, batch, caps)
                resolution = resolve(needles, batch, self._foreign)
            except (PayloadError, ValueError) as exc:
                errors.append(f"step {self._step}: consolidation payload: {exc}")
                return StubbedReply(message={}, finish_reason="error", error_status=400)
            body = resolution.reply.to_json()
            asked = frozenset(f.id for f in batch.facts)
            if any(frozenset(c.fact_ids) >= asked for c in record.calls if c.step == self._step):
                # The server re-asks facts a reply left unaddressed: the scripted reply did not
                # cover what the call needed. Not fatal (the server copes), but never silent.
                resolution.degradations.append("re-asked: an earlier reply in this step left these facts unaddressed")
            record.calls.append(
                CallRecord(
                    step=self._step,
                    batch_facts=[statement(f.text) for f in batch.facts],
                    fact_ids=[f.id for f in batch.facts],
                    pool=[f"[{entry.status or 'active'}] {entry.text}" for entry in batch.pool],
                    schema_found=caps.found,
                    lifecycle_schema=caps.lifecycle,
                    reply=needles,
                    wire=json.loads(body),
                    degradations=resolution.degradations,
                    notes=resolution.notes,
                )
            )
            return StubbedReply(message={"role": "assistant", "content": body}, finish_reason="stop")

        return respond

    def _dedup_responder(self, record: ScenarioRecord):
        def respond(request: ChatRequest) -> StubbedReply:
            text = request.all_text
            new = _line_after(text, "[NEW] ")
            existing = _line_after(text, "[EXISTING] ")
            same = any(
                (a in new and b in existing) or (b in new and a in existing) for a, b in self.scenario.dedup_merges
            )
            decision = _DedupDecision(
                action="merge" if same else "keep",
                text=existing if same else "",
                reason="the scenario declares these the same fact" if same else "they state different facts",
            )
            record.dedup.append(DedupRecord(step=self._step, new=new, existing=existing, action=decision.action))
            return StubbedReply(
                message={"role": "assistant", "content": decision.model_dump_json()}, finish_reason="stop"
            )

        return respond

    # -- reads ---------------------------------------------------------------

    async def _memories(self, bank_id: str) -> list[_MemoryRow]:
        response = await self.http.get(f"/v1/default/banks/{bank_id}/memories/list", params={"limit": 1000})
        response.raise_for_status()
        return _MemoryList.model_validate(response.json()).items

    async def _states(self, bank_id: str, memories: Sequence[_MemoryRow]) -> StateRecord:
        observed: dict[str, list[str | None]] = {}
        observation_statuses: dict[str, int] = {}
        for memory in memories:
            response = await self.http.get(f"/v1/default/banks/{bank_id}/memories/{memory.id}")
            response.raise_for_status()
            status = _truth_status(response.json())
            if memory.fact_type == "observation":
                key = status or "none"
                observation_statuses[key] = observation_statuses.get(key, 0) + 1
            elif memory.document_id is not None:
                observed.setdefault(memory.document_id, []).append(status)
        # Scored in ``summarize``, once the run as a whole says whether claims are reported.
        return StateRecord(observed=observed, observation_statuses=observation_statuses)

    async def _query(self, bank_id: str, query: Query, memories: Sequence[_MemoryRow]) -> QueryRow:
        body = recall_body(query, self.config.recall_types)
        response = await self.http.post(f"/v1/default/banks/{bank_id}/memories/recall", json=body)
        if response.status_code != 200:
            raise BenchmarkError(f"recall {query.id} -> {response.status_code}: {response.text[:500]}")
        recalled = _RecallResponse.model_validate(response.json()).results
        by_id = {memory.id: memory for memory in memories}
        doc_of = {memory.id: memory.document_id for memory in memories if memory.fact_type != "observation"}
        server_items = [_ranked(rank, result, by_id, doc_of) for rank, result in enumerate(recalled, start=1)]
        items = canonical_order(server_items)[:ROW_DEPTH]
        labels = QueryLabels.for_query(self.scenario, query)
        return QueryRow(
            scenario=self.scenario.id,
            query_id=query.id,
            split=self.split,
            kind=query.kind,
            text=query.text,
            request=body,
            results=[
                ResultRow(
                    rank=position,
                    server_rank=item.rank,
                    memory_id=item.memory_id,
                    type=item.type,
                    text=item.text,
                    docs=list(item.docs),
                    tags=list(item.tags),
                    cls=classify(item, labels),
                    score=None if item.score is None else round(item.score, 6),
                )
                for position, item in enumerate(items, start=1)
            ],
            metrics=score_query(items, labels),
        )


def canonical_order(items: Sequence[RankedItem]) -> list[RankedItem]:
    """The server's order, with each run of near-equal final scores (``SCORE_TIE_TOLERANCE``)
    ordered by (text, labels). Items without a score keep their server position."""
    out: list[RankedItem] = []
    group: list[RankedItem] = []

    def flush() -> None:
        out.extend(sorted(group, key=lambda item: (item.text, item.type, item.docs)))
        group.clear()

    for item in items:
        previous = group[-1] if group else None
        if (
            previous is not None
            and item.score is not None
            and previous.score is not None
            and abs(previous.score - item.score) <= SCORE_TIE_TOLERANCE
        ):
            group.append(item)
            continue
        flush()
        group.append(item)
    flush()
    return out


def recall_body(query: Query, recall_types: Sequence[str] | None) -> dict[str, Any]:
    """The recall request for one query. Historical and as_of queries carry the lifecycle
    request fields, which a server without the feature ignores."""
    body: dict[str, Any] = {
        "query": query.text,
        "query_timestamp": _iso(at_day(query.day)),
        "budget": "mid",
        "max_tokens": 4096,
        # Entities are an add-on section and never change the ranked results; source facts
        # make every observation name its citations, which is how it is labelled.
        "include": {"entities": None, "source_facts": {"max_tokens": -1}},
    }
    if recall_types:
        body["types"] = list(recall_types)
    if query.kind == "historical":
        body["lifecycle_view"] = "historical"
    if query.kind == "as_of" and query.as_of is not None:
        body["as_of"] = at_day(query.as_of).isoformat()
    if query.tags:
        body["tags"] = list(query.tags)
        body["tags_match"] = query.tags_match
    return body


def _retain_item(item: Item) -> dict[str, Any]:
    metadata = {"session_id": item.session}
    if item.source:
        metadata["source"] = item.source
    payload: dict[str, Any] = {
        "content": item.content,
        "timestamp": item.timestamp.isoformat(),
        "document_id": item.doc,
        "metadata": metadata,
    }
    if item.tags:
        payload["tags"] = list(item.tags)
    if item.observation_scopes:
        payload["observation_scopes"] = item.observation_scopes
    return payload


def _ranked(
    rank: int, result: _RecallItem, by_id: Mapping[str, _MemoryRow], doc_of: Mapping[str, str | None]
) -> RankedItem:
    row = by_id.get(result.id)
    kind = result.type or (row.fact_type if row else None) or "unknown"
    if kind == "observation":
        sources = result.source_fact_ids or (row.source_memory_ids if row else [])
        docs = tuple(sorted({doc for sid in sources if (doc := doc_of.get(sid)) is not None}))
    else:
        doc = result.document_id or (row.document_id if row else None)
        docs = (doc,) if doc else ()
    return RankedItem(
        rank=rank,
        memory_id=result.id,
        type=kind,
        text=result.text,
        docs=docs,
        tags=tuple(sorted(result.tags or (row.tags if row else []))),
        score=result.scores.final if result.scores else None,
    )


#: GET memory carried a ``lifecycle`` key but no claim: active by default *if* the server
#: tracks claims at all, which only the run as a whole can tell (see ``summarize``).
IMPLICIT = "implicit"


def _truth_status(memory: Mapping[str, Any]) -> str | None:
    """The claim state GET memory reports under ``lifecycle``, wherever the shape puts it.

    The ``lifecycle`` key is the feature's (DESIGN §5.6) and its exact nesting is not frozen
    yet, so both a nested ``claim.truth_status`` and a flat ``truth_status`` are read. A
    ``lifecycle`` key without a claim is ``IMPLICIT``; no key at all is ``None`` (no state).
    """
    lifecycle = memory.get("lifecycle")
    if not isinstance(lifecycle, dict):
        return None
    claim = lifecycle.get("claim")
    if isinstance(claim, dict) and isinstance(claim.get("truth_status"), str):
        return claim["truth_status"]
    status = lifecycle.get("truth_status")
    return status if isinstance(status, str) else IMPLICIT


def _line_after(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("\n", start)
    return text[start:] if end < 0 else text[start:end]


def default_api_dir() -> Path:
    return API_DIR


def write_record(record: RunRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1))


def load_record(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text())


def run(config: RunConfig) -> RunRecord:
    return asyncio.run(run_benchmark(config))
