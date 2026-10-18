"""Claim-lifecycle views in recall (adaptive memory ``lifecycle`` mode), the historical-cue detector,
and the reflect surface that reads them.

Pure parts first (view rules, view resolution, the refill loop, the cue patterns, the reflect tool
schema and prompt), then the real pipeline on pg0: a supersession laid down the way the
consolidation writer lays it down (one ``lock_claims`` pass, CAS, successor and asserting-fact
claim rows, raw inheritance) and recalled through every view, the fail-closed path, the refill
keeping the trim full, the cue detector, the downgrade semantics, and reflect's expand annotation.

Seeding is raw SQL into ``memory_units`` (pinned texts, ``mentioned_at`` and observation sources —
the consolidation LLM that would build them is not under test), so the module is
``memory_backend_incompatible``. Claims are written through the store interface consolidation and
the PATCH API use; assertions go through ``recall_async``, the tools, and the store's readers.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry, generate_latest

from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine.adaptive.cues import CUE_NAMES, MAX_CUE_QUERY_TOKENS, detect_historical_cue
from hindsight_api.engine.adaptive.models import ClaimState, TruthStatus
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.reflect.agent import ReflectToolTokenLimits, _execute_tool
from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools
from hindsight_api.engine.reflect.tools import tool_expand, tool_recall
from hindsight_api.engine.reflect.tools_schema import TOOL_RECALL, get_reflect_tools
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.search.lifecycle_filter import (
    ClaimStateUnavailable,
    LifecycleView,
    filter_candidates_by_claims,
    in_trim_order,
    lifecycle_annotation,
    resolve_view,
)
from hindsight_api.engine.search.recall_boost import trim_merged_candidates
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.extensions.operation_validator import OperationValidationError

pytestmark = pytest.mark.memory_backend_incompatible

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)
ACTIVE = TruthStatus.ACTIVE
UNCERTAIN = TruthStatus.UNCERTAIN
SUPERSEDED = TruthStatus.SUPERSEDED
REFUTED = TruthStatus.REFUTED


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _claim(
    status: TruthStatus,
    *,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    retired_via: str | None = None,
    superseded_by: str | None = None,
) -> ClaimState:
    return ClaimState(
        memory_id="m",
        truth_status=status,
        confidence=None,
        valid_from=valid_from,
        valid_to=valid_to,
        retired_via=retired_via,
        superseded_by_id=superseded_by,
        inherited_from_id=None,
        scope_fingerprint=None,
        reason=None,
    )


# --------------------------------------------------------------------------- the view rules (pure)

CURRENT = LifecycleView(name="current")
HISTORICAL = LifecycleView(name="historical")
ALL = LifecycleView(name="all-with-status")

CLAIMS = {
    "no_row": None,
    "active": _claim(ACTIVE),
    "uncertain": _claim(UNCERTAIN),
    "superseded": _claim(SUPERSEDED, valid_to=day(30), retired_via="supersedes", superseded_by="s"),
    "inherited": _claim(SUPERSEDED, valid_to=day(30), retired_via="inherited", superseded_by="s"),
    "split": _claim(SUPERSEDED, retired_via="splits"),
    "merged": _claim(SUPERSEDED, valid_to=day(30), retired_via="merges"),
    "refuted": _claim(REFUTED, valid_to=day(30), retired_via="explicit"),
    "refuted_inherited": _claim(REFUTED, valid_to=day(30), retired_via="inherited"),
}


@pytest.mark.parametrize(
    ("view", "shown"),
    [
        (CURRENT, {"no_row", "active", "uncertain"}),
        (HISTORICAL, {"no_row", "active", "uncertain", "superseded", "inherited", "split"}),
        (ALL, set(CLAIMS)),
    ],
    ids=["current", "historical", "all-with-status"],
)
def test_each_named_view_shows_exactly_its_statuses(view, shown):
    assert {name for name, claim in CLAIMS.items() if view.keeps(claim)} == shown


def test_as_of_keeps_a_claim_iff_its_interval_contains_the_instant():
    at = LifecycleView(name="as_of", as_of=day(30))
    assert at.keeps(None), "no row: active since before anything recorded"
    assert at.keeps(_claim(ACTIVE)), "an active claim with no interval holds at every instant"
    assert at.keeps(_claim(ACTIVE, valid_from=day(30))), "valid_from is inclusive"
    assert not at.keeps(_claim(ACTIVE, valid_from=day(31)))
    assert not at.keeps(_claim(SUPERSEDED, valid_to=day(30), retired_via="supersedes")), "valid_to is exclusive"
    assert at.keeps(_claim(SUPERSEDED, valid_to=day(31), retired_via="supersedes"))
    assert at.keeps(_claim(SUPERSEDED, valid_from=day(1), valid_to=day(31), retired_via="inherited"))
    assert not at.keeps(_claim(SUPERSEDED, valid_to=None, retired_via="merges")), "merge-retired never"
    assert not at.keeps(_claim(REFUTED, valid_to=day(99))), "refuted never, whatever its interval"
    # A naive valid_from (a store that hands one back) reads as UTC, not as an error.
    assert at.keeps(_claim(ACTIVE, valid_from=datetime(2026, 1, 31)))


def test_the_view_resolves_as_of_then_the_named_view_then_a_cue_then_the_bank_default():
    assert resolve_view(requested=None, as_of=None, default_view="current") == CURRENT
    assert resolve_view(requested=None, as_of=None, default_view="historical") == HISTORICAL
    assert resolve_view(requested="current", as_of=None, default_view="historical") == CURRENT
    assert resolve_view(requested="all-with-status", as_of=None, default_view="current") == ALL
    naive = datetime(2026, 2, 1)
    assert resolve_view(requested="historical", as_of=naive, default_view="current") == LifecycleView(
        name="as_of", as_of=naive.replace(tzinfo=UTC)
    )
    assert resolve_view(requested=None, as_of=None, default_view="current", cue_name="used_to") == LifecycleView(
        name="historical", cue="used_to"
    )
    assert resolve_view(
        requested=None, as_of=None, default_view="current", cue_name="before_date", cue_as_of=day(5)
    ) == LifecycleView(name="as_of", as_of=day(5), cue="before_date")
    with pytest.raises(ValueError, match="Unknown lifecycle_view"):
        resolve_view(requested="everything", as_of=None, default_view="current")


def test_only_a_claim_that_is_not_plain_active_is_annotated():
    assert lifecycle_annotation(None) is None
    assert lifecycle_annotation(_claim(ACTIVE, valid_from=day(3))) is None
    note = lifecycle_annotation(_claim(SUPERSEDED, valid_from=day(1), valid_to=day(30), superseded_by="s"))
    assert note is not None
    assert note.model_dump() == {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": day(1).isoformat(),
        "valid_to": day(30).isoformat(),
        "superseded_by": "s",
    }
    assert lifecycle_annotation(_claim(UNCERTAIN)).truth_status == "uncertain"


# --------------------------------------------------------------------------- the refill loop (pure)


def _candidates(n: int) -> list[MergedCandidate]:
    return [
        MergedCandidate(
            retrieval=RetrievalResult(id=f"c{i}", text=f"t{i}", fact_type="world"),
            rrf_score=1.0 / (i + 1),
            rrf_rank=i + 1,
            source_ranks={"semantic_rank": i + 1},
        )
        for i in range(n)
    ]


class _Loader:
    """Claims by id, recording each read's ids."""

    def __init__(self, claims: dict[str, ClaimState]):
        self.claims = claims
        self.reads: list[list[str]] = []

    async def __call__(self, ids: list[str]) -> dict[str, ClaimState]:
        self.reads.append(ids)
        return {i: self.claims[i] for i in ids if i in self.claims}


async def test_removals_are_refilled_from_the_tail_until_the_trim_is_full():
    retired = _claim(SUPERSEDED, retired_via="supersedes")
    # The first read (target 3 + slack 1 = 4 candidates) loses c0..c2; refills read the shortfall
    # plus the slack until three are kept.
    loader = _Loader({"c0": retired, "c1": retired, "c2": retired, "c4": retired})
    outcome = await filter_candidates_by_claims(_candidates(10), view=CURRENT, target=3, load_claims=loader, slack=1)
    assert [mc.id for mc in outcome.kept] == ["c3", "c5", "c6"]
    assert [r.memory_id for r in outcome.removed] == ["c0", "c1", "c2", "c4"]
    assert loader.reads == [["c0", "c1", "c2", "c3"], ["c4", "c5", "c6"]]
    assert (outcome.claims_checked, outcome.refill_rounds, outcome.unchecked) == (7, 1, 3)


async def test_after_the_last_refill_the_unread_tail_is_dropped_not_passed_on_unchecked():
    retired = _claim(REFUTED)
    loader = _Loader({f"c{i}": retired for i in range(20)})
    outcome = await filter_candidates_by_claims(
        _candidates(20), view=CURRENT, target=2, load_claims=loader, slack=1, max_refills=3
    )
    assert outcome.kept == []
    assert len(loader.reads) == 4, "the first read and three refills"
    assert outcome.refill_rounds == 3
    assert outcome.claims_checked + outcome.unchecked == 20
    assert outcome.unchecked > 0


async def test_kept_candidates_carry_their_annotation_and_a_load_failure_propagates():
    loader = _Loader({"c1": _claim(UNCERTAIN)})
    outcome = await filter_candidates_by_claims(_candidates(3), view=CURRENT, target=5, load_claims=loader)
    assert [mc.id for mc in outcome.kept] == ["c0", "c1", "c2"]
    assert [mc.lifecycle.truth_status if mc.lifecycle else None for mc in outcome.kept] == [None, "uncertain", None]

    async def broken(ids):
        raise ConnectionError("replica gone")

    with pytest.raises(ConnectionError):
        await filter_candidates_by_claims(_candidates(3), view=CURRENT, target=5, load_claims=broken)


def test_claims_are_read_in_the_order_the_trim_keeps_candidates():
    pool = _candidates(6)
    pool.reverse()  # fusion order no longer matches rrf_score
    assert in_trim_order(pool, 10, {}) is pool, "under the cap the trim keeps fusion order"
    ordered = in_trim_order(list(pool), 4, {"graph": "high"})
    assert [mc.id for mc in ordered] == [f"c{i}" for i in range(6)], "over it, by (boosted) RRF score"
    assert [mc.id for mc in ordered[:4]] == [
        mc.id for mc in trim_merged_candidates(list(pool), 4, {"graph": "high"}).kept
    ]


# --------------------------------------------------------------------------- the cue detector (pure)

POSITIVE_CUES = [
    ("What database did we use before the migration?", "what_before"),
    ("Which CI system did the team used to run?", "used_to"),
    ("what did we use for queues", "what_did_we_use"),
    ("What was the runtime before?", "what_before"),
    ("Show the config before the upgrade", "before_change"),
    ("What was the previous deploy target?", "previously"),
    ("Previously, where did logs go?", "previously"),
    ("The service formerly known as Atlas", "formerly"),
    ("What was originally chosen for auth?", "originally"),
    ("Docs for the earlier version of the API", "earlier_version"),
    ("Give me the history of the payments schema", "history_of"),
    ("What has the timeout changed from?", "changed_from"),
]
NEGATIVE_CUES = [
    "Which Java version does the billing service run?",
    "What do we use for queues?",
    "Before you answer, list the services",  # "before" with no date resolved and no phrase
    "How is the deploy pipeline configured?",
    "useful tools for the migration",
    "The history shows nothing",
]


@pytest.mark.parametrize(("query", "cue"), POSITIVE_CUES)
def test_a_historical_phrase_fires_its_cue(query, cue):
    fired = detect_historical_cue(query)
    assert fired is not None and (fired.cue, fired.as_of) == (cue, None)


@pytest.mark.parametrize("query", NEGATIVE_CUES)
def test_a_present_tense_question_fires_no_cue(query):
    assert detect_historical_cue(query) is None


def test_before_a_resolved_date_views_the_claims_as_of_that_date():
    start = datetime(2024, 3, 1)
    cue = detect_historical_cue("What database did we use before March 2024?", start)
    assert (cue.cue, cue.as_of) == ("what_before", start)
    bare = detect_historical_cue("Which runtime ran before 2024-03-01?", start)
    assert (bare.cue, bare.as_of) == ("before_date", start)
    # A resolved date without "before" does not move the instant.
    plain = detect_historical_cue("What was the previous runtime in March 2024?", start)
    assert (plain.cue, plain.as_of) == ("previously", None)
    assert "before_date" in CUE_NAMES


def test_a_long_query_never_fires_a_cue():
    long_query = "previously " + " ".join(["word"] * MAX_CUE_QUERY_TOKENS)
    assert detect_historical_cue(long_query) is None
    assert detect_historical_cue(" ".join(["previously", *["w"] * (MAX_CUE_QUERY_TOKENS - 1)])) is not None


# --------------------------------------------------------------------------- reflect surface (pure)


def _recall_tool(tools: list[dict]) -> dict:
    return next(t for t in tools if t["function"]["name"] == "recall")


def test_the_reflect_recall_tool_is_byte_identical_outside_lifecycle_mode():
    default = get_reflect_tools()
    assert json.dumps(default, sort_keys=True) == json.dumps(get_reflect_tools(lifecycle=False), sort_keys=True)
    assert json.dumps(_recall_tool(default)) == json.dumps(TOOL_RECALL)
    assert "include_history" not in json.dumps(default)


def test_lifecycle_mode_adds_include_history_and_changes_nothing_else():
    lifecycle_tool = _recall_tool(get_reflect_tools(lifecycle=True))
    properties = lifecycle_tool["function"]["parameters"]["properties"]
    assert properties["include_history"]["type"] == "boolean"
    stripped = json.loads(json.dumps(lifecycle_tool))
    del stripped["function"]["parameters"]["properties"]["include_history"]
    assert stripped == TOOL_RECALL
    assert "include_history" not in TOOL_RECALL["function"]["parameters"]["properties"], "the constant is shared"
    others = [t for t in get_reflect_tools(lifecycle=True) if t["function"]["name"] != "recall"]
    assert others == [t for t in get_reflect_tools() if t["function"]["name"] != "recall"]


def test_the_uncertain_rule_is_only_in_the_lifecycle_prompt():
    profile = {"name": "Bank", "mission": "testing"}
    base = build_system_prompt_for_tools(profile)
    assert build_system_prompt_for_tools(profile, lifecycle=False) == base
    lifecycle = build_system_prompt_for_tools(profile, lifecycle=True)
    assert "NOT resolved by recency" in lifecycle and "NOT resolved by recency" not in base
    # Pure addition: removing the clause gives back the base prompt.
    assert (
        lifecycle.replace(lifecycle[lifecycle.index("Some results carry") : lifecycle.index("## Showing")], "") == base
    )


async def test_the_recall_tool_passes_include_history_only_when_the_agent_asked():
    calls: list[tuple] = []

    async def recall_fn(query, max_tokens, max_chunk_tokens, **kwargs):
        calls.append((query, kwargs))
        return {"memories": []}

    async def unused(*args, **kwargs):
        raise AssertionError("not this tool")

    limits = ReflectToolTokenLimits(
        recall_max_tokens=2048,
        recall_chunk_max_tokens=1000,
        observations_max_tokens=5000,
        mental_models_read_max_tokens=6000,
    )

    async def run(args):
        return await _execute_tool("recall", args, unused, unused, unused, recall_fn, unused, limits, 100_000)

    await run({"query": "q1", "reason": "r"})
    await run({"query": "q2", "reason": "r", "include_history": True})
    await run({"query": "q3", "reason": "r", "include_history": "true"})
    await run({"query": "q4", "reason": "r", "include_history": False})
    assert calls == [("q1", {}), ("q2", {"include_history": True}), ("q3", {"include_history": True}), ("q4", {})]


# --------------------------------------------------------------------------- the real pipeline (pg0)

QUERY = "Which Java version does the billing service run?"
F1 = "The billing service runs on Java 17."
F2 = "The billing service was upgraded to Java 21."
O1 = "The billing service runs Java 17."
O2 = "The billing service runs Java 21."
R = "The billing service runs on Java 8."
M = "Billing service Java runtime is version 17."


@asynccontextmanager
async def _tx(memory):
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        async with conn.transaction():
            yield conn


async def _seed(memory, bank_id: str, text: str, *, fact_type: str = "world", mentioned_at=DAY0, sources=()) -> str:
    """One memory with a real embedding and pinned text/mentioned_at (observations get sources)."""
    (embedding,) = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text])
    memory_id = str(uuid.uuid4())
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, mentioned_at, source_memory_ids, "
            "consolidated_at) VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8)",
            uuid.UUID(memory_id),
            bank_id,
            text,
            fact_type,
            "[" + ",".join(str(v) for v in embedding) + "]",
            mentioned_at,
            [uuid.UUID(s) for s in sources] if sources else None,
            None if fact_type == "observation" else day(1),
        )
    return memory_id


async def _configure(memory, bank_id: str, **updates) -> None:
    await memory._config_resolver.update_bank_config(
        bank_id, {f"adaptive_memory_{key}": value for key, value in updates.items()}
    )


@pytest.fixture
async def java_upgrade(memory, request_context):
    """Java 17 → 21, superseded the way consolidation does it, plus a refuted and a merged memory."""
    bank_id = f"test-alr-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    await _configure(memory, bank_id, mode="lifecycle")
    ids = {"F1": await _seed(memory, bank_id, F1, mentioned_at=day(0))}
    ids["O1"] = await _seed(memory, bank_id, O1, fact_type="observation", sources=[ids["F1"]])
    ids["F2"] = await _seed(memory, bank_id, F2, mentioned_at=day(30))
    ids["O2"] = await _seed(memory, bank_id, O2, fact_type="observation", mentioned_at=day(30), sources=[ids["F2"]])
    ids["R"] = await _seed(memory, bank_id, R, mentioned_at=day(5))
    ids["M"] = await _seed(memory, bank_id, M, fact_type="observation", sources=[ids["F1"]])
    store = get_memories()
    async with _tx(memory) as conn:
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=list(ids.values()), skip_locked=False)
        for successor_side in ("O2", "F2"):
            await store.set_claim(
                conn, bank_id=bank_id, memory_id=ids[successor_side], truth_status=ACTIVE, valid_from=day(30)
            )
        assert await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=ids["O1"],
            to_status=SUPERSEDED,
            retired_via="supersedes",
            superseded_by_id=ids["O2"],
            valid_to=day(30),
            reason="test",
            confidence=None,
        )
        await store.add_claim_evidence(conn, bank_id=bank_id, memory_id=ids["O1"], evidence_ids=[ids["F2"]])
        assert await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=ids["M"],
            to_status=SUPERSEDED,
            retired_via="merges",
            superseded_by_id=ids["O1"],
            valid_to=day(2),
            reason="test",
            confidence=None,
        )
        # M (merged away) cites F1 too, but is retired, so it does not hold F1 up.
        inherited = await store.inherit_from_observation(
            conn,
            bank_id=bank_id,
            observation_id=ids["O1"],
            to_status=SUPERSEDED,
            successor_id=ids["O2"],
            valid_to=day(30),
            evidence_ids=[ids["F2"]],
            candidate_source_ids=[ids["F1"]],
        )
        assert inherited == [ids["F1"]]
        assert await store.cas_retire(
            conn,
            bank_id=bank_id,
            memory_id=ids["R"],
            to_status=REFUTED,
            retired_via="explicit",
            superseded_by_id=None,
            valid_to=day(6),
            reason="test",
            confidence=None,
        )
    yield SimpleNamespace(bank_id=bank_id, ids=ids, names={v: k for k, v in ids.items()})
    await memory.delete_bank(bank_id, request_context=request_context)


async def _names(memory, request_context, scenario, **kwargs) -> dict[str, object]:
    """``{name: lifecycle annotation dict or None}`` of the scenario memories a recall returned."""
    kwargs.setdefault("fact_type", ["world", "observation"])
    result = await memory.recall_async(scenario.bank_id, QUERY, request_context=request_context, **kwargs)
    return {
        scenario.names[r.id]: (r.lifecycle.model_dump() if r.lifecycle else None)
        for r in result.results
        if r.id in scenario.names
    }


async def test_current_view_drops_superseded_and_refuted_memories_of_every_type(memory, request_context, java_upgrade):
    result = await memory.recall_async(
        java_upgrade.bank_id,
        QUERY,
        fact_type=["world", "observation"],
        enable_trace=True,
        request_context=request_context,
    )
    shown = {java_upgrade.names[r.id] for r in result.results if r.id in java_upgrade.names}
    assert shown == {"O2", "F2"}, "the inherited raw Java 17 fact must not answer as current"
    assert all(r.lifecycle is None for r in result.results), "active memories carry no annotation"
    assert result.trace["lifecycle"] == {"view": "current", "claims_checked": 6, "refill_rounds": 0}
    filtered = {java_upgrade.names[f["node_id"]]: f for f in result.trace["lifecycle_filtered"]}
    assert set(filtered) == {"O1", "F1", "R", "M"}
    assert filtered["O1"] == {
        "node_id": java_upgrade.ids["O1"],
        "truth_status": "superseded",
        "retired_via": "supersedes",
        "superseded_by": java_upgrade.ids["O2"],
    }
    assert filtered["F1"]["retired_via"] == "inherited"
    assert filtered["R"] == {"node_id": java_upgrade.ids["R"], "truth_status": "refuted", "retired_via": "explicit"}
    phase = next(p for p in result.trace["summary"]["phase_metrics"] if p["phase_name"] == "lifecycle_claim_load")
    assert not phase["details"].get("diagnostic"), "a partitioning phase, counted in the timeline"
    assert phase["details"]["removed"] == 4
    # It partitions the timeline without overlapping its neighbours (the coverage test's rule).
    timeline = sum(
        p["duration_seconds"] for p in result.trace["summary"]["phase_metrics"] if not p["details"].get("diagnostic")
    )
    assert timeline <= result.trace["summary"]["total_duration_seconds"] + 0.01


async def test_historical_view_annotates_superseded_memories_and_never_shows_refuted_or_merged(
    memory, request_context, java_upgrade
):
    shown = await _names(memory, request_context, java_upgrade, lifecycle_view="historical")
    assert set(shown) == {"O1", "F1", "O2", "F2"}
    assert shown["O2"] is None and shown["F2"] is None
    assert shown["O1"] == {
        "truth_status": "superseded",
        "confidence": None,
        "valid_from": None,
        "valid_to": day(30).isoformat(),
        "superseded_by": java_upgrade.ids["O2"],
    }
    assert shown["F1"]["superseded_by"] == java_upgrade.ids["O2"]


async def test_as_of_shows_the_claims_valid_at_the_instant(memory, request_context, java_upgrade):
    before = await _names(memory, request_context, java_upgrade, as_of=day(15))
    assert set(before) == {"O1", "F1"}, "the successor and its asserting fact begin at day 30"
    after = await _names(memory, request_context, java_upgrade, as_of=day(45))
    assert set(after) == {"O2", "F2"}
    # as_of=now excludes the inherited raw fact exactly as the current view does, and wins over a
    # named view sent with it.
    now = await _names(memory, request_context, java_upgrade, as_of=datetime.now(UTC), lifecycle_view="historical")
    assert set(now) == {"O2", "F2"}
    # The ranking clock is not the truth clock: a historical query_timestamp changes nothing here.
    dated = await _names(memory, request_context, java_upgrade, question_date=day(15))
    assert set(dated) == {"O2", "F2"}


async def test_the_internal_view_shows_everything_labelled(memory, request_context, java_upgrade):
    shown = await _names(memory, request_context, java_upgrade, lifecycle_view="all-with-status")
    assert set(shown) == {"O1", "F1", "O2", "F2", "R", "M"}
    assert {name: (s or {}).get("truth_status") for name, s in shown.items()} == {
        "O1": "superseded",
        "F1": "superseded",
        "O2": None,
        "F2": None,
        "R": "refuted",
        "M": "superseded",
    }


async def test_the_bank_default_view_applies_when_the_caller_names_none(memory, request_context, java_upgrade):
    await _configure(memory, java_upgrade.bank_id, default_view="historical")
    assert set(await _names(memory, request_context, java_upgrade)) == {"O1", "F1", "O2", "F2"}
    assert set(await _names(memory, request_context, java_upgrade, lifecycle_view="current")) == {"O2", "F2"}


async def test_after_a_downgrade_claims_are_neither_applied_nor_shown_and_come_back_on_upgrade(
    memory, request_context, java_upgrade
):
    await _configure(memory, java_upgrade.bank_id, mode="rank")
    ranked = await memory.recall_async(
        java_upgrade.bank_id,
        QUERY,
        fact_type=["world", "observation"],
        lifecycle_view="current",
        enable_trace=True,
        request_context=request_context,
    )
    assert {java_upgrade.names[r.id] for r in ranked.results} == {"O1", "F1", "O2", "F2", "R", "M"}
    assert all(r.lifecycle is None for r in ranked.results)
    assert "lifecycle" not in ranked.trace and "lifecycle_filtered" not in ranked.trace
    assert all("lifecycle" not in r.model_dump() for r in ranked.results)
    # The claim rows were kept: back in lifecycle mode they apply again.
    await _configure(memory, java_upgrade.bank_id, mode="lifecycle")
    assert set(await _names(memory, request_context, java_upgrade)) == {"O2", "F2"}


async def test_removals_before_the_trim_keep_the_candidate_budget_full(memory, request_context, monkeypatch):
    bank_id = f"test-alr-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    await _configure(memory, bank_id, mode="lifecycle")
    ids = [
        await _seed(memory, bank_id, f"The billing service deploy runbook, revision {n}.", mentioned_at=day(n))
        for n in range(8)
    ]
    store = get_memories()
    async with _tx(memory) as conn:
        await store.lock_claims(conn, bank_id=bank_id, memory_ids=ids[:5], skip_locked=False)
        for memory_id in ids[:5]:
            await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=memory_id,
                to_status=SUPERSEDED,
                retired_via="explicit",
                superseded_by_id=None,
                valid_to=day(9),
                reason="test",
                confidence=None,
            )
    monkeypatch.setattr(memory_engine_module, "_resolve_reranker_max_candidates", lambda config, budget: 3)
    try:
        result = await memory.recall_async(
            bank_id,
            "billing service deploy runbook",
            fact_type=["world"],
            enable_trace=True,
            request_context=request_context,
        )
        assert sorted(r.id for r in result.results) == sorted(ids[5:]), "the three current revisions fill the trim"
        assert result.trace["lifecycle"]["claims_checked"] == 8
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_claim_load_failure_fails_the_recall_closed(memory, request_context, java_upgrade, monkeypatch):
    async def broken(conn, *, bank_id, memory_ids):
        raise ConnectionResetError("replica went away")

    monkeypatch.setattr(get_memories(), "claims_for", broken)
    for view in (None, "historical", "all-with-status"):
        with pytest.raises(ClaimStateUnavailable) as raised:
            await memory.recall_async(java_upgrade.bank_id, QUERY, lifecycle_view=view, request_context=request_context)
        assert isinstance(raised.value.__cause__, ConnectionResetError)
    # Outside lifecycle mode nothing reads claims, so the same failure cannot reach the recall.
    await _configure(memory, java_upgrade.bank_id, mode="rank")
    assert (await memory.recall_async(java_upgrade.bank_id, QUERY, request_context=request_context)).results


async def test_an_unknown_view_is_refused(memory, request_context, java_upgrade):
    with pytest.raises(OperationValidationError) as raised:
        await memory.recall_async(
            java_upgrade.bank_id, QUERY, lifecycle_view="everything", request_context=request_context
        )
    assert raised.value.status_code == 422


async def test_the_cue_detector_turns_an_external_history_question_historical(memory, request_context, java_upgrade):
    history_question = "What Java version did the billing service use before the upgrade?"

    async def shown(**kwargs):
        result = await memory.recall_async(
            java_upgrade.bank_id,
            history_question,
            fact_type=["world", "observation"],
            enable_trace=True,
            request_context=request_context,
            **kwargs,
        )
        return {java_upgrade.names[r.id] for r in result.results if r.id in java_upgrade.names}, result.trace[
            "lifecycle"
        ]

    names, trace = await shown(_external=True)
    assert (names, trace["view"], trace.get("historical_cue")) == ({"O2", "F2"}, "current", None), "flag off"
    await _configure(memory, java_upgrade.bank_id, historical_cues=True)
    names, trace = await shown(_external=True)
    assert names == {"O1", "F1", "O2", "F2"}
    assert (trace["view"], trace["historical_cue"]) == ("historical", "what_before")
    names, trace = await shown()
    assert (names, trace["view"]) == ({"O2", "F2"}, "current"), "internal recalls never read cues"
    names, trace = await shown(_external=True, lifecycle_view="current")
    assert (names, trace.get("historical_cue")) == ({"O2", "F2"}, None), "a named view wins over a cue"


async def test_reflect_tools_read_history_annotated_and_expand_never_shows_a_retired_memory_as_current(
    memory, request_context, java_upgrade
):
    current = await tool_recall(memory, java_upgrade.bank_id, QUERY, request_context, include_chunks=False)
    assert {java_upgrade.names[m["id"]] for m in current["memories"]} == {"F2"}
    history = await tool_recall(
        memory, java_upgrade.bank_id, QUERY, request_context, include_chunks=False, lifecycle_view="historical"
    )
    by_name = {java_upgrade.names[m["id"]]: m for m in history["memories"]}
    assert set(by_name) == {"F1", "F2"}
    assert by_name["F1"]["lifecycle"] == {
        "truth_status": "superseded",
        "valid_to": day(30).isoformat(),
        "superseded_by": java_upgrade.ids["O2"],
    }, "null-pruned like the rest of the result"
    assert "lifecycle" not in by_name["F2"]

    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        ids = [java_upgrade.ids["F1"], java_upgrade.ids["F2"]]
        expanded = await tool_expand(
            conn, java_upgrade.bank_id, ids, "chunk", tags=None, tags_match="any", tag_groups=None, lifecycle=True
        )
        plain = await tool_expand(
            conn, java_upgrade.bank_id, ids, "chunk", tags=None, tags_match="any", tag_groups=None
        )
    annotated = {r["memory_id"]: r["memory"].get("lifecycle") for r in expanded["results"]}
    assert annotated[java_upgrade.ids["F1"]]["truth_status"] == "superseded"
    assert annotated[java_upgrade.ids["F2"]] is None
    assert all("lifecycle" not in r["memory"] for r in plain["results"])


# --------------------------------------------------------------------------- metrics


def test_lifecycle_metrics_export_under_their_exact_prometheus_names():
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    from hindsight_api.metrics import MetricsCollector

    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    provider = MeterProvider(metric_readers=[reader])
    config = SimpleNamespace(
        metrics_include_bank_id=False,
        metrics_include_tenant=False,
        recall_diagnostic_phases=True,
        recall_phase_sample_every=1,
    )
    try:
        with (
            patch("hindsight_api.metrics.get_meter", return_value=provider.get_meter("test-lifecycle")),
            patch("hindsight_api.config.get_config", return_value=config),
        ):
            collector = MetricsCollector()
        collector.record_adaptive_lifecycle_filtered(bank_id="b", view="current", superseded=2, refuted=1)
        collector.record_adaptive_lifecycle_filtered(bank_id="b", view="historical", superseded=0, refuted=0)
        collector.record_adaptive_historical_cue(bank_id="b", cue="used_to")
        for transition in ("supersession", "refutation", "uncertain", "split", "merge"):
            collector.record_adaptive_claim_transition(bank_id="b", transition=transition, origin="explicit")
        text = generate_latest(registry).decode()
    finally:
        provider.shutdown()
    lines = text.splitlines()

    def sample(name: str, label: str) -> str:
        """The one sample line of ``name`` carrying ``label`` (labels export sorted by name)."""
        return next(line for line in lines if line.startswith(name + "{") and label in line)

    assert sample("hindsight_adaptive_memory_recall_superseded_filtered_total", 'view="current"').endswith(" 2.0")
    assert sample("hindsight_adaptive_memory_recall_refuted_filtered_total", 'view="current"').endswith(" 1.0")
    assert sample("hindsight_adaptive_memory_historical_cue_fired_total", 'cue="used_to"').endswith(" 1.0")
    for name in ("supersessions", "refutations", "uncertain_claims", "splits", "merges"):
        assert sample(f"hindsight_adaptive_memory_{name}_total", 'origin="explicit"').endswith(" 1.0"), name
    assert 'view="historical"' not in text, "zero counts add no series"
    assert 'bank_id="b"' not in text
