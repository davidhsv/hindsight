"""Golden transcripts: everything a server says to one fixed scenario, written down so a
later build can be held to it byte for byte.

A feature that ships switched off has one promise to keep: with the switch off,
nothing a caller can see changes. Not the recall payload, not the trace, not the
prompt and JSON schema consolidation sends the model, not the MCP tool list, not
the transfer archive. No story asserts that, because no story looks at everything
the server says. This module records all of it from one server, so another can be
compared with it.

The scenario is the suite's usual one: the published client, the stub LLM, and
timestamps that drive time. Two rules are added:

- **Raw bodies, not client models.** Responses are read with the generated
  client's ``*_without_preload_content`` calls, so a key the server starts sending
  (or stops sending, or starts sending as ``null``) shows up even where the client
  model would have dropped it.
- **Canonical, not sorted.** What the wall clock, the UUID generator and the host
  decide is replaced by placeholders, and an order the server takes from a UUID is
  replaced by a stable one (``canonicalize`` lists every rule). Every other value,
  and every order the server actually chose, is kept as sent, because ranking order
  is half of what is being compared.

The base server is not deterministic everywhere, and three of its nondeterminisms
cannot be undone after the fact, so the scenario is built not to trigger them:

- **No two candidates tie in a retrieval arm.** Fusion scores a candidate by its rank
  in each arm, and the database orders equal scores by physical row position, which
  moves whenever a row is rewritten. Keyword scores come in steps of 0.1 per matched
  term, so the facts share few words and the queries avoid those; ``ties`` fails a
  capture that has one.
- **No entity is shared between facts.** Graph retrieval stable-sorts a ``set`` of
  fact ids by score, so equal graph scores come out in the order of their random
  UUIDs' hashes. Facts that share no entity give the graph arm nothing to tie on.
- **No updated observation is shown to a later consolidation.** An update stores its
  sources through ``array_agg(DISTINCT ...)``, which orders them by UUID, and the next
  prompt lists them in that order. So each update is the last consolidation in its
  scope.

The server also runs with ``PYTHONHASHSEED=0``: recall iterates a ``frozenset`` of fact
types, so under a random seed the trace lists the fact-type arms in a different order
in every process.

CLI::

    uv run python -m hindsight_system_tests.golden capture --out DIR [--api-dir PATH] [--schema NAME]
    uv run python -m hindsight_system_tests.golden compare A B [--report FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import NotFoundException
from hindsight_client_api.models.recall_request import RecallRequest
from hindsight_client_api.models.retain_request import RetainRequest

from .payloads import (
    Consolidation,
    ExtractedFacts,
    Observation,
    ObservationDelete,
    ObservationUpdate,
    batch_fact_id,
    extracted,
    fact,
    shown_observations,
)
from .rulebook import ChatRequest, Stubs
from .server import ENV_API_DIR, ENV_DATABASE_URL, api_dir, start_hindsight_server, start_stub_server
from .steps import STEP_ANCHORS
from .waiting import wait_until_settled

#: Fixed, so the id is the same in every capture: it travels in the export archive and
#: the MCP paths. The suite's prefix, so a run killed mid-capture is swept like any other.
GOLDEN_BANK = "systest-golden-off-mode"

#: Every recall pins its clock. Unpinned, recency scoring reads the wall clock and every
#: score in the transcript moves with it.
QUERY_TIMESTAMP = "2025-08-01T12:00:00+00:00"

#: A pg0 instance of the capture's own, so a capture never shares a database with a
#: concurrent suite run.
DEFAULT_DATABASE_URL = "pg0://hindsight-golden:15519"

#: Not part of the comparison: where and when a capture was taken.
MANIFEST = "manifest.json"


def fresh_schema() -> str:
    """A schema nothing has written to yet, for one capture.

    Fresh rather than reused, for two reasons. Where scores tie in a recall no trace
    shows — the one consolidation runs to pick the observations it shows the model —
    rows come back in physical order, and a schema that has had rows deleted hands their
    space to new ones. And a schema migrated by a newer server version cannot be opened
    by an older one, so a base and a feature capture must never share one.
    """
    return f"systest_golden_{uuid.uuid4().hex[:8]}"


def server_env(schema: str) -> dict[str, str]:
    """The server-level settings a golden capture adds to the suite's ``stub_environment``."""
    return {"HINDSIGHT_API_DATABASE_SCHEMA": schema, "PYTHONHASHSEED": "0"}


# --- The scenario -----------------------------------------------------------------------

# Worded so that no query below scores two candidates equally in any retrieval arm (see
# ``ties``): two facts matching a query on the same number of terms tie on keyword score.
# Hence few words shared between facts, and queries that avoid the ones that are. Entity
# names are indexed for keyword search too, which is why "Java 21" counts twice.
JAVA_21 = "The Atlas backend runs on Java 21"
KUBERNETES = "Every Atlas service is deployed to Kubernetes"
CLUSTER = "Atlas workloads share one cluster"
REDIS = "Atlas caches user sessions in Redis"
PYTHON = "Beacon is written in Python 3.12"
POSTGRES = "Beacon stores its records in Postgres"
NO_REDIS = "The earlier Redis claim about Atlas was wrong"
MEMCACHED = "Session state for Atlas has always lived in Memcached"
JAVA_25 = "The Atlas team upgraded the backend from Java 21 to Java 25"
HELM = "Atlas deploys with Helm charts onto the cluster"

# Observations avoid the facts' shared words too: consolidation shows the model the
# observations its own recall finds for the new facts, in that recall's order, so a tie
# there would reorder the prompt.
OBS_JAVA_21 = "Backend services run on Java 21"
OBS_JAVA_25 = "Backend services run on Java 25 since the June 2025 upgrade"
OBS_KUBERNETES = "Every service is deployed to the Kubernetes cluster"
OBS_KUBERNETES_HELM = "Every service is deployed with Helm charts to one shared Kubernetes cluster"
OBS_CLUSTER = "Workloads share a single cluster"
OBS_REDIS = "User sessions are cached in Redis"
OBS_MEMCACHED = "User sessions live in Memcached, never in Redis"
OBS_BEACON = "Beacon is a Python 3.12 service backed by Postgres"

#: Three observation scopes. Atlas's runtime and cache facts share one; its platform facts
#: carry a second tag, which (tags taken together) makes a scope of their own.
ATLAS = "project:atlas"
PLATFORM = "team:platform"
BEACON = "project:beacon"


def _fact_id(request: ChatRequest, text: str) -> str:
    found = batch_fact_id(request, text)
    if found is None:
        raise AssertionError(f"golden scenario: fact {text!r} is not in this consolidation batch")
    return found


def _observation_id(request: ChatRequest, text: str) -> str:
    shown = shown_observations(request)
    if text not in shown:
        raise AssertionError(f"golden scenario: observation {text!r} was not shown; shown: {sorted(shown)}")
    return shown[text]


def _creates(*observations: tuple[str, list[str]]) -> Callable[[ChatRequest], Consolidation]:
    """One new observation per ``(text, [source fact text, ...])``."""

    def build(request: ChatRequest) -> Consolidation:
        return Consolidation(
            creates=[
                Observation(text=text, source_fact_ids=[_fact_id(request, source) for source in sources])
                for text, sources in observations
            ]
        )

    return build


def _refute_redis(request: ChatRequest) -> Consolidation:
    """The correction retires the Redis observation outright instead of amending it."""
    return Consolidation(
        creates=[
            Observation(text=OBS_MEMCACHED, source_fact_ids=[_fact_id(request, NO_REDIS), _fact_id(request, MEMCACHED)])
        ],
        deletes=[
            ObservationDelete(
                observation_id=_observation_id(request, OBS_REDIS), reason="refuted: Atlas never used Redis"
            )
        ],
    )


def _upgrade_java(request: ChatRequest) -> Consolidation:
    """The supersession case: Java 21 becomes Java 25, updated in place as the base prompt asks."""
    return Consolidation(
        updates=[
            ObservationUpdate(
                text=OBS_JAVA_25,
                observation_id=_observation_id(request, OBS_JAVA_21),
                source_fact_ids=[_fact_id(request, JAVA_25)],
                reason="state change: the backend moved to Java 25",
            )
        ]
    )


def _merge_clusters(request: ChatRequest) -> Consolidation:
    """An update plus a ``merged_into`` delete: the near-duplicate folds into the survivor."""
    survivor = _observation_id(request, OBS_KUBERNETES)
    return Consolidation(
        updates=[
            ObservationUpdate(
                text=OBS_KUBERNETES_HELM,
                observation_id=survivor,
                source_fact_ids=[_fact_id(request, HELM)],
                reason="same deployment facet, more detail",
            )
        ],
        deletes=[
            ObservationDelete(
                observation_id=_observation_id(request, OBS_CLUSTER),
                merged_into=survivor,
                reason="near-duplicate of the deployment observation",
            )
        ],
    )


@dataclass(frozen=True)
class Step:
    """One retained document: its content, what extraction finds in it, and what the
    consolidation that follows does with those facts."""

    document_id: str
    timestamp: str
    tags: list[str]
    content: str
    facts: ExtractedFacts
    consolidate: Callable[[ChatRequest], Consolidation]


# One entity per fact and none shared (see the module docstring), named so no two are
# similar enough for entity resolution to merge them.
STEPS: list[Step] = [
    Step(
        document_id="atlas-runtime",
        timestamp="2025-01-10T09:00:00+00:00",
        tags=[ATLAS],
        content=f"{JAVA_21}.",
        facts=extracted(fact(JAVA_21, entities=["Java 21"])),
        consolidate=_creates((OBS_JAVA_21, [JAVA_21])),
    ),
    Step(
        document_id="atlas-platform",
        timestamp="2025-01-20T14:00:00+00:00",
        tags=[ATLAS, PLATFORM],
        content=f"{KUBERNETES}. {CLUSTER}.",
        facts=extracted(fact(KUBERNETES, entities=["Kubernetes"]), fact(CLUSTER, entities=["cluster"])),
        consolidate=_creates((OBS_KUBERNETES, [KUBERNETES]), (OBS_CLUSTER, [CLUSTER])),
    ),
    Step(
        document_id="atlas-cache",
        timestamp="2025-02-14T10:30:00+00:00",
        tags=[ATLAS],
        content=f"{REDIS}.",
        facts=extracted(fact(REDIS, entities=["Redis"])),
        consolidate=_creates((OBS_REDIS, [REDIS])),
    ),
    Step(
        document_id="beacon-stack",
        timestamp="2025-03-01T08:00:00+00:00",
        tags=[BEACON],
        content=f"{PYTHON}. {POSTGRES}.",
        facts=extracted(fact(PYTHON, entities=["Python"]), fact(POSTGRES, entities=["Postgres"])),
        consolidate=_creates((OBS_BEACON, [PYTHON, POSTGRES])),
    ),
    Step(
        document_id="atlas-cache-correction",
        timestamp="2025-04-02T16:45:00+00:00",
        tags=[ATLAS],
        content=f"Correction: {NO_REDIS}. {MEMCACHED}.",
        facts=extracted(fact(NO_REDIS, entities=["correction"]), fact(MEMCACHED, entities=["Memcached"])),
        consolidate=_refute_redis,
    ),
    Step(
        document_id="atlas-upgrade",
        timestamp="2025-06-20T11:15:00+00:00",
        tags=[ATLAS],
        content=f"{JAVA_25} this week.",
        facts=extracted(
            fact(
                JAVA_25,
                entities=["upgrade"],
                fact_kind="event",
                when="June 2025",
                occurred_start="2025-06-20T00:00:00+00:00",
                occurred_end="2025-06-20T23:59:59+00:00",
            )
        ),
        consolidate=_upgrade_java,
    ),
    Step(
        document_id="atlas-deploy",
        timestamp="2025-07-05T13:00:00+00:00",
        tags=[ATLAS, PLATFORM],
        content=f"{HELM}.",
        facts=extracted(fact(HELM, entities=["Helm"])),
        consolidate=_merge_clusters,
    ),
]


@dataclass(frozen=True)
class Query:
    """One recall, named so a diff says which one moved. Each runs twice, with and without
    a trace: the trace is what ``ties`` reads, and the untraced payload is what callers get."""

    name: str
    params: dict[str, Any]


JAVA_QUESTION = "Which Java version does the backend run?"

QUERIES: list[Query] = [
    Query("java", {"query": JAVA_QUESTION}),
    Query("java-observations", {"query": JAVA_QUESTION, "types": ["observation"]}),
    # Before the upgrade: same question, earlier clock (recency is scored against it).
    Query("java-historical", {"query": JAVA_QUESTION, "query_timestamp": "2025-03-15T00:00:00+00:00"}),
    Query("cache-world", {"query": "How are user sessions cached?", "types": ["world"]}),
    Query(
        "cache-world-observations",
        {
            "query": "How are user sessions cached?",
            "types": ["world", "observation"],
            "include": {"source_facts": {"max_tokens": 4096}},
        },
    ),
    Query(
        "deploy-platform-tags",
        {
            "query": "How are services deployed?",
            "tags": [ATLAS, PLATFORM],
            "tags_match": "all_strict",
            "include": {"entities": {"max_tokens": 500}, "chunks": {"max_tokens": 2000}},
        },
    ),
    Query("beacon-tags", {"query": "Where are Beacon records stored?", "tags": [BEACON], "tags_match": "any_strict"}),
    Query("min-scores", {"query": "backend Java upgrade", "min_scores": {"final": 0.3, "reranker": 0.2}}),
    Query("semantic-floor", {"query": "Helm charts on Kubernetes", "min_scores": {"semantic": 0.2}}),
    # The only query with a date in it, so the only one with a temporal arm. Scoped to
    # Beacon: temporal retrieval spreads from its entry points along links at one flat
    # score, and in the whole bank that ties every fact it reaches.
    Query(
        "temporal-beacon",
        {"query": "What did Beacon store on March 1, 2025?", "tags": [BEACON], "tags_match": "any_strict"},
    ),
]


# --- Capture ----------------------------------------------------------------------------


class CaptureFailed(AssertionError):
    """The scenario did not run as scripted, so there is nothing worth comparing."""


@dataclass(frozen=True)
class Capture:
    """A raw transcript, keyed by output file, and the wall-clock window it was taken in.

    The window is how canonicalization tells a stamp the server took *now* from a date
    the scenario supplied, wherever the stamp appears: a field, a JSON document inside a
    string field, a prompt.
    """

    files: dict[str, Any]
    started: datetime
    finished: datetime


async def _json(response: Any) -> Any:
    """The body of a raw generated-client response, failing loudly on an error status."""
    body = await response.read()
    if response.status >= 400:
        raise CaptureFailed(f"{response.method} {response.url} answered {response.status}: {body[:2000]!r}")
    return json.loads(body)


def _guard(stubs: Stubs, where: str) -> None:
    """The suite's two silent-success guards, run per step: the rulebook is reset between
    steps, so the fixture-level guard would only ever see the last one."""
    if stubs.llm.unmatched:
        suggestions = "\n".join(f"    {call.suggestion()}" for call in stubs.llm.unmatched)
        raise CaptureFailed(f"{where}: {len(stubs.llm.unmatched)} LLM call(s) matched no rule:\n{suggestions}")
    if stubs.rejected_requests:
        raise CaptureFailed(f"{where}: the stub rejected request(s) as malformed: {stubs.rejected_requests}")


def _step_of(request: ChatRequest) -> str | None:
    return next((name for name, anchor in STEP_ANCHORS.items() if anchor in request.all_text), None)


async def _retain_step(client: Hindsight, stubs: Stubs, step: Step) -> dict[str, Any]:
    """Retain one document and wait for its consolidation, recording every LLM request.

    One document at a time, settled in between, so each consolidation sees exactly one
    step's facts and the calls arrive in an order the scenario decides.
    """
    stubs.llm.reset()
    stubs.llm.on_step("extract_facts", contains=step.content).returns(step.facts)
    stubs.llm.on_step("consolidate").answers_with(step.consolidate)

    request = RetainRequest.from_dict(
        {
            "items": [
                {
                    "content": step.content,
                    "timestamp": step.timestamp,
                    "tags": step.tags,
                    "document_id": step.document_id,
                }
            ],
            "async": False,
        }
    )
    assert request is not None  # from_dict returns None only for a None input
    retained = await _json(await client.memory.retain_memories_without_preload_content(GOLDEN_BANK, request))
    await wait_until_settled(client, GOLDEN_BANK)
    _guard(stubs, f"step {step.document_id}")

    return {
        "document_id": step.document_id,
        "retain": retained,
        "llm_calls": [{"step": _step_of(call), "body": call.body} for call in stubs.llm.calls],
    }


async def _memories(client: Hindsight) -> dict[str, Any]:
    listed = await _json(await client.memory.list_memories_without_preload_content(GOLDEN_BANK, limit=100))
    got = []
    histories = []
    for item in listed["items"]:
        got.append(await _json(await client.memory.get_memory_without_preload_content(GOLDEN_BANK, item["id"])))
        if item["fact_type"] == "observation":
            history = await client.memory.get_observation_history_without_preload_content(GOLDEN_BANK, item["id"])
            histories.append({"id": item["id"], "history": await _json(history)})
    return {"list": listed, "get": got, "observation_history": histories}


async def _recalls(client: Hindsight) -> list[dict[str, Any]]:
    out = []
    for query in QUERIES:
        for trace in (True, False):
            params = {"query_timestamp": QUERY_TIMESTAMP, **query.params, "trace": trace}
            request = RecallRequest.from_dict(params)
            assert request is not None  # from_dict returns None only for a None input
            response = await client.memory.recall_memories_without_preload_content(GOLDEN_BANK, request)
            out.append({"name": query.name, "request": params, "response": await _json(response)})
    return out


_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _jsonrpc(response: httpx.Response) -> Any:
    """A JSON-RPC reply, whether it came as plain JSON or as one server-sent event."""
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        data = [line[len("data:") :].strip() for line in response.text.splitlines() if line.startswith("data:")]
        if len(data) != 1:
            raise CaptureFailed(f"expected one MCP event, got {len(data)}: {response.text[:2000]!r}")
        return json.loads(data[0])
    return response.json()


async def _mcp_session(http: httpx.AsyncClient, path: str) -> dict[str, Any]:
    """``initialize`` then ``tools/list`` on one MCP endpoint, over streamable HTTP, the
    way an MCP client opens a session."""
    headers = dict(_MCP_HEADERS)
    initialize = await http.post(
        path,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "hindsight-golden", "version": "1"},
            },
        },
    )
    initialized = _jsonrpc(initialize)
    session = initialize.headers.get("mcp-session-id")
    if session:
        headers["Mcp-Session-Id"] = session
    try:
        notified = await http.post(
            path, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        notified.raise_for_status()
        listed = await http.post(path, headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = _jsonrpc(listed)
    finally:
        if session:
            with contextlib.suppress(httpx.HTTPError):
                await http.delete(path, headers=headers)
    return {"initialize": initialized, "tools/list": tools}


async def _mcp(base_url: str) -> dict[str, Any]:
    """Both MCP surfaces: the multi-bank root and the single-bank path."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as http:
        return {
            "multi_bank": await _mcp_session(http, "/mcp/"),
            "single_bank": await _mcp_session(http, f"/mcp/{GOLDEN_BANK}/"),
        }


def _archive(archive: bytes) -> dict[str, Any]:
    """An export ZIP as its file list, in archive order, and each file's parsed content."""
    files: dict[str, Any] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in zipped.namelist():
            raw = zipped.read(name)
            if name.endswith(".json"):
                files[name] = json.loads(raw)
            elif name.endswith((".jsonl", ".ndjson")):
                files[name] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            else:
                files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {"files": list(files), "contents": files}


async def capture(base_url: str, stubs: Stubs) -> Capture:
    """Run the scenario against the server at ``base_url``, answered from ``stubs``.

    The server must have been started with ``server_env`` on top of the suite's
    environment. The result is raw; ``canonicalize`` it before writing or comparing.
    """
    client = Hindsight(base_url=base_url)
    started = datetime.now(timezone.utc)
    try:
        files = await _capture(client, base_url, stubs)
    finally:
        await client.aclose()
    return Capture(files=files, started=started, finished=datetime.now(timezone.utc))


async def _capture(client: Hindsight, base_url: str, stubs: Stubs) -> dict[str, Any]:
    # Only a reused schema can still hold the bank; on a fresh one this is a 404.
    with contextlib.suppress(NotFoundException):
        await client.banks.delete_bank(GOLDEN_BANK)
    await client.acreate_bank(bank_id=GOLDEN_BANK, name="Golden off-mode transcript")

    steps = [await _retain_step(client, stubs, step) for step in STEPS]
    stubs.llm.reset()

    files: dict[str, Any] = {"steps.json": steps}
    files["memories.json"] = await _memories(client)
    files["recall.json"] = await _recalls(client)
    files["mcp.json"] = await _mcp(base_url)
    # The bank export first: it carries the operations log, which the document export
    # would otherwise have added to.
    files["export_bank.json"] = _archive(await client.aexport_bank(GOLDEN_BANK, poll_interval=0.2))
    files["export_documents.json"] = _archive(
        await client.aexport_documents(
            GOLDEN_BANK, include_observations=True, include_knowledge_base=True, poll_interval=0.2
        )
    )
    _guard(stubs, "reads")
    return files


# --- Canonicalization -------------------------------------------------------------------

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_PLACEHOLDER = re.compile(r"<uuid-(\d+)>")

#: Measured by the server's own clock, so different on every run.
_DURATION_KEY = re.compile(r"(^|_)duration_seconds$")

#: Values that name the machine the server ran on rather than anything it did.
_HOST_KEYS = frozenset({"worker_id"})

#: Lists of memory ids the server stores through ``array_agg(DISTINCT ...)``, which orders
#: them by UUID: a random order, unlike every other id list here.
_ID_SET_KEYS = frozenset({"source_memory_ids", "source_fact_ids", "new_source_memory_ids"})

#: How far outside the capture a stamp may fall and still be the server's "now": covers
#: a naive local time read as UTC and a retry scheduled after the capture ended. Every
#: date the scenario supplies is in 2025, nowhere near.
_WALL_CLOCK_SLACK = timedelta(days=1)

WALL_CLOCK = "<wall-clock>"
DURATION = "<duration>"
HOST = "<host>"


@dataclass
class Canonicalizer:
    """Replaces what the wall clock, the UUID generator and the host decide, and nothing else.

    - A timestamp inside the capture window becomes ``<wall-clock>``; one outside it came
      from the scenario and is kept.
    - A ``*duration_seconds`` number becomes ``<duration>``.
    - ``worker_id`` (the host name) becomes ``<host>``.
    - A UUID becomes ``<uuid-N>``, numbered by first appearance in transcript order. The
      first file is the consolidation prompts, in scenario order, so every fact is
      numbered in the order it was extracted; an id that moves in a later ranking
      therefore shows up as a moved placeholder instead of renumbering everything.

    Applied to keys as well as values: the recall response keys some maps by memory id.
    """

    started: datetime
    finished: datetime
    ids: dict[str, str] = field(default_factory=dict)

    def _is_wall_clock(self, stamp: str) -> bool:
        try:
            parsed = datetime.fromisoformat(stamp.replace(" ", "T", 1))
        except ValueError:
            return False
        # A naive stamp may be UTC or the server's local time; it is the clock's if either is.
        readings = [parsed] if parsed.tzinfo else [parsed.replace(tzinfo=timezone.utc), parsed.astimezone()]
        low, high = self.started - _WALL_CLOCK_SLACK, self.finished + _WALL_CLOCK_SLACK
        return any(low <= reading <= high for reading in readings)

    def _timestamp(self, match: re.Match[str]) -> str:
        return WALL_CLOCK if self._is_wall_clock(match.group(0)) else match.group(0)

    def _uuid(self, match: re.Match[str]) -> str:
        key = match.group(0).lower()
        if key not in self.ids:
            self.ids[key] = f"<uuid-{len(self.ids) + 1}>"
        return self.ids[key]

    def text(self, value: str) -> str:
        return _UUID.sub(self._uuid, _TIMESTAMP.sub(self._timestamp, value))

    def value(self, value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {self.text(k): self.value(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v, key) for v in value]
        if isinstance(value, str):
            return HOST if key in _HOST_KEYS else self.text(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and _DURATION_KEY.search(key):
            return DURATION
        return value


def _sort_entities(node: Any, key: str = "") -> Any:
    """Entity names in alphabetical order, wherever the server lists them.

    A memory's entities, and the entity map of a recall response, come out in whatever
    order the database joined them — for an observation, the union of its sources'.
    Run before UUIDs are numbered: the recall entity map is where entity ids first
    appear, so its order decides their placeholders.
    """
    if isinstance(node, list):
        if key == "entities" and all(isinstance(v, str) for v in node):
            return sorted(node)
        return [_sort_entities(v, key) for v in node]
    if isinstance(node, str) and key == "entities":
        return ", ".join(sorted(node.split(", "))) if node else node
    if not isinstance(node, dict):
        return node
    items = node.items()
    if key == "entities" and all(isinstance(v, dict) and "entity_id" in v for v in node.values()):
        items = sorted(items)
    return {k: _sort_entities(v, k) for k, v in items}


def _placeholder_order(value: Any) -> int:
    """Sort key for a canonical id: its placeholder number, i.e. its first appearance."""
    match = _PLACEHOLDER.fullmatch(value) if isinstance(value, str) else None
    return int(match.group(1)) if match else sys.maxsize


def _sort_id_sets(node: Any) -> Any:
    """``_ID_SET_KEYS`` lists, and the recall ``source_facts`` map keyed by the same ids,
    in first-appearance order instead of UUID order."""
    if isinstance(node, list):
        return [_sort_id_sets(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _ID_SET_KEYS and isinstance(value, list):
            value = sorted(value, key=_placeholder_order)
        elif key == "source_facts" and isinstance(value, dict):
            value = {k: value[k] for k in sorted(value, key=_placeholder_order)}
        elif key == "source_facts" and isinstance(value, list) and all(isinstance(v, dict) for v in value):
            # The observation history resolves its source ids to facts, in the same UUID order.
            value = sorted(value, key=lambda v: _placeholder_order(v.get("id")))
        out[key] = _sort_id_sets(value)
    return out


def _order_archive(archive: dict[str, Any]) -> None:
    """Undo the UUID order of an export archive's rows, in place.

    Each document's facts and the observations file come out ordered by id. An
    observation cites its sources as ``{document_id, fact_index}`` — a position in that
    random order — so the facts are put in first-appearance order and every citation is
    renumbered to match.
    """
    contents = archive["contents"]
    positions: dict[str, dict[int, int]] = {}
    for name, document in contents.items():
        if not (name.startswith("documents/") and isinstance(document, dict) and "facts" in document):
            continue
        order = sorted(
            range(len(document["facts"])), key=lambda i: _placeholder_order(document["facts"][i].get("source_id"))
        )
        document["facts"] = [document["facts"][i] for i in order]
        positions[document["id"]] = {old: new for new, old in enumerate(order)}

    observations = contents.get("observations.json")
    if isinstance(observations, list):
        for observation in observations:
            for source in observation.get("sources") or []:
                remap = positions.get(source.get("document_id"), {})
                source["fact_index"] = remap.get(source.get("fact_index"), source.get("fact_index"))
            observation["sources"] = sorted(
                observation.get("sources") or [], key=lambda s: (s.get("document_id") or "", s.get("fact_index") or 0)
            )
        observations.sort(key=lambda o: _placeholder_order(o.get("source_id")))


def canonicalize(capture: Capture) -> dict[str, Any]:
    """The transcript with every wall-clock, host and UUID value replaced (see
    ``Canonicalizer``), entity names alphabetical (``_sort_entities``), and every
    UUID-derived order replaced by first-appearance order (``_sort_id_sets``,
    ``_order_archive``).

    Raises when a value that still reads as the wall clock survives — a date with no time,
    an epoch number — because a golden file carrying one would pass on the day it was
    captured and fail on every other.
    """
    canon = Canonicalizer(started=capture.started, finished=capture.finished)
    files = {name: _sort_id_sets(canon.value(_sort_entities(content))) for name, content in capture.files.items()}
    for name in ("export_bank.json", "export_documents.json"):
        if name in files:
            _order_archive(files[name])

    leaks = list(_wall_clock_leaks(files, capture.started, capture.finished))
    if leaks:
        raise CaptureFailed("wall-clock values survived canonicalization:\n  " + "\n  ".join(leaks[:50]))
    return files


def _wall_clock_leaks(files: dict[str, Any], started: datetime, finished: datetime) -> Iterator[str]:
    days = {moment.date() for moment in (started, finished, started.astimezone(), finished.astimezone())}
    dates = {form for day in days for form in (day.isoformat(), f"{day:%B} {day.day}, {day.year}", f"{day:%B %d, %Y}")}
    low = (started - _WALL_CLOCK_SLACK).timestamp()
    high = (finished + _WALL_CLOCK_SLACK).timestamp()

    def walk(node: Any, where: str) -> Iterator[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(key, f"{where}.<key>")
                yield from walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, f"{where}[{index}]")
        elif isinstance(node, str):
            yield from (f"{where}: contains the capture date {date!r}" for date in dates if date in node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if low <= node <= high or low * 1000 <= node <= high * 1000:
                yield f"{where}: {node} reads as an epoch time inside the capture"

    for name, content in files.items():
        yield from walk(content, name)


def ties(files: dict[str, Any]) -> list[str]:
    """Every retrieval arm, in every traced recall, that scored two candidates equally.

    Each arm hands fusion a ranked list, and fusion scores a candidate by its rank. When
    two candidates tie, the database decides which ranks first — by row order, which
    moves whenever a row is rewritten — so the fused scores, and the order everything
    downstream is shown in, change from run to run. A transcript with a tie in it cannot
    be a golden file; the scenario has to be reworded until there is none.
    """
    found = []
    for recall in files.get("recall.json", []):
        trace = recall["response"].get("trace") or {}
        for arm in trace.get("retrieval_results") or []:
            scores = [result["score"] for result in arm["results"]]
            tied = sorted({score for score in scores if scores.count(score) > 1})
            if tied:
                texts = [result["text"] for result in arm["results"] if result["score"] in tied]
                found.append(f"{recall['name']}: {arm['method_name']}/{arm['fact_type']} ties at {tied}: {texts}")
    return found


# --- Files and comparison ---------------------------------------------------------------


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def write(directory: Path, transcript: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Write one canonical transcript, replacing the transcript the directory held.

    Refuses a non-empty directory that holds no transcript, rather than clearing out
    files it did not write.
    """
    if directory.is_dir() and any(directory.iterdir()) and not (directory / MANIFEST).is_file():
        raise FileExistsError(f"{directory} is not empty and holds no {MANIFEST}; not overwriting it")
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.json"):
        stale.unlink()
    for name, content in transcript.items():
        (directory / name).write_text(_dumps(content))
    (directory / MANIFEST).write_text(_dumps(manifest))


def load(directory: Path) -> dict[str, Any]:
    files = {p.name: json.loads(p.read_text()) for p in sorted(directory.glob("*.json")) if p.name != MANIFEST}
    if not files:
        raise FileNotFoundError(f"no golden transcript in {directory}")
    return files


def _brief(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 300 else f"{text[:300]}…"


def _walk(path: str, a: Any, b: Any) -> Iterator[str]:
    """One readable entry per differing leaf; a multi-line string as a line diff."""
    if type(a) is not type(b):
        yield f"{path}: {_brief(a)} != {_brief(b)}"
    elif isinstance(a, dict):
        only_a = [k for k in a if k not in b]
        only_b = [k for k in b if k not in a]
        if only_a or only_b:
            yield f"{path}: keys only in A {only_a}, only in B {only_b}"
        elif list(a) != list(b):
            yield f"{path}: key order differs: {list(a)} != {list(b)}"
        for key in a:
            if key in b:
                yield from _walk(f"{path}.{key}", a[key], b[key])
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            yield from _walk(f"{path}[{index}]", x, y)
    elif a != b:
        if isinstance(a, str) and ("\n" in a or "\n" in b):
            lines = difflib.unified_diff(a.splitlines(), b.splitlines(), "A", "B", lineterm="", n=1)
            yield f"{path}: text differs\n    " + "\n    ".join(lines)
        else:
            yield f"{path}: {_brief(a)} != {_brief(b)}"


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Every difference between two canonical transcripts; empty means identical."""
    differences = []
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            differences.append(f"{name}: missing from B")
        elif name not in expected:
            differences.append(f"{name}: missing from A")
        else:
            differences.extend(_walk(name, expected[name], actual[name]))
    return differences


def report(a: str, b: str, differences: list[str]) -> str:
    verdict = "IDENTICAL" if not differences else f"{len(differences)} difference(s)"
    return f"A: {a}\nB: {b}\n{verdict}\n" + "".join(f"\n- {d}" for d in differences)


# --- CLI --------------------------------------------------------------------------------


def _git(directory: Path, *args: str) -> str | None:
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        return subprocess.run(
            ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
        ).stdout.rstrip()
    return None


async def _capture_cli(args: argparse.Namespace) -> int:
    if args.api_dir:
        os.environ[ENV_API_DIR] = str(args.api_dir)
    os.environ[ENV_DATABASE_URL] = args.database_url
    server_dir = api_dir()
    schema = args.schema or fresh_schema()

    log_dir = Path(tempfile.mkdtemp(prefix="hindsight-golden-"))
    stubs = Stubs()
    stub = start_stub_server(stubs)
    try:
        server = start_hindsight_server(
            stub_url=stub.url, log_path=log_dir / "server.log", extra_env=server_env(schema)
        )
        try:
            raw = await capture(server.url, stubs)
        finally:
            server.stop()
    finally:
        stub.stop()

    # Kept beside the server log: when a rule needs adding, the raw transcript shows what
    # the server actually sent before any placeholder replaced it.
    raw_path = log_dir / "raw.json"
    raw_path.write_text(
        _dumps({"started": raw.started.isoformat(), "finished": raw.finished.isoformat(), "files": raw.files})
    )
    transcript = canonicalize(raw)
    tied = ties(transcript)
    if tied:
        raise CaptureFailed("ties in retrieval would make this transcript flaky:\n  " + "\n  ".join(tied))

    status = _git(server_dir, "status", "--porcelain", "--", ".")
    manifest = {
        "api_dir": str(server_dir),
        "api_git_head": _git(server_dir, "rev-parse", "HEAD"),
        "api_tree_changes": None if status is None else status.splitlines(),
        "database_url": args.database_url,
        "schema": schema,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "server_log": str(log_dir / "server.log"),
        "raw_transcript": str(raw_path),
    }
    write(args.out, transcript, manifest)
    print(f"captured {len(transcript)} files into {args.out} (server log and raw transcript in {log_dir})")
    return 0


def _compare_cli(args: argparse.Namespace) -> int:
    differences = compare(load(args.a), load(args.b))
    text = report(str(args.a), str(args.b), differences)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0 if not differences else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hindsight_system_tests.golden", description=__doc__.split("\n\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser(
        "capture", help="run the scenario against a fresh server, write the transcript"
    )
    capture_parser.add_argument("--out", type=Path, required=True, help="directory to write the transcript into")
    capture_parser.add_argument("--api-dir", type=Path, help=f"hindsight-api-slim checkout to run (sets {ENV_API_DIR})")
    capture_parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL, help="database the server runs on")
    capture_parser.add_argument("--schema", help="database schema (default: a fresh one per capture)")

    compare_parser = commands.add_parser("compare", help="compare two transcript directories; exit 1 on any difference")
    compare_parser.add_argument("a", type=Path)
    compare_parser.add_argument("b", type=Path)
    compare_parser.add_argument("--report", type=Path, help="also write the comparison to this file")

    args = parser.parse_args(argv)
    if args.command == "capture":
        return asyncio.run(_capture_cli(args))
    return _compare_cli(args)


if __name__ == "__main__":
    sys.exit(main())
