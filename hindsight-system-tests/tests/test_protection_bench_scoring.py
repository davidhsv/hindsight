"""The protection benchmark's own mechanics: data integrity and its frozen hash, the request and env
shapes, every P1 / P2 / P3 endpoint, the cap accounting, and the report. Server-free."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from hindsight_system_tests.context_bench.runner import BenchmarkError, TypeSafeCall
from hindsight_system_tests.protection_bench import data, report, runner, scoring
from hindsight_system_tests.protection_bench.data import (
    LEGIT,
    MEMORIES,
    ORDINARY,
    PLANTED,
    QUERIES,
    SCENARIO_BY_ID,
    SCENARIOS,
    contains,
)
from hindsight_system_tests.protection_bench.runner import (
    P3MemoryRow,
    P3ScenarioRow,
    RecallRow,
    ResultRow,
    RunConfig,
    RunRecord,
)

FILLER = [m.label for m in ORDINARY[:12]]


def _record(
    suite: str,
    side: str,
    *,
    recalls: Sequence[RecallRow] = (),
    scenarios: Sequence[P3ScenarioRow] = (),
    passes: Sequence[str] = ("served",),
    label: str | None = None,
    calls: Sequence[TypeSafeCall] = (),
) -> RunRecord:
    return RunRecord(
        suite=suite,
        side=side,
        label=label or f"{suite}-{side}",
        data_sha256=runner.data_sha256(),
        api_dir="/x",
        api_sha="abc",
        api_dirty=False,
        harness_sha="def",
        harness_dirty=False,
        profile="none",
        server_env={},
        database_url="pg0://x:1",
        database_schema="s",
        passes=list(passes),
        started_at="2026-10-18T00:00:00+00:00",
        recalls=list(recalls),
        scenarios=list(scenarios),
        typesafe_calls=list(calls),
    )


def _row(pass_name: str, query_id: str, labels: Sequence[str], status: int = 200) -> RecallRow:
    return RecallRow(
        pass_name=pass_name,
        query_id=query_id,
        status=status,
        latency_ms=10.0,
        results=[ResultRow(label=label, rank=i) for i, label in enumerate(labels, start=1)],
    )


def _call(attempts: int = 1, status: int = 200) -> TypeSafeCall:
    return TypeSafeCall(
        question="q:x", options=0, approx_request_tokens=1, status=status, latency_ms=1, attempts=attempts
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_data_is_valid_and_meets_the_protocol_minimums() -> None:
    assert data.validate() == []
    assert len(PLANTED) >= 40 and len(LEGIT) >= 40 and len(ORDINARY) >= 200
    assert len(SCENARIOS) >= 8
    assert {q.target for q in QUERIES} == {m.label for m in PLANTED + LEGIT}


def test_data_matches_its_frozen_hash() -> None:
    frozen = runner.frozen_sha256()
    assert frozen is not None, "the data must be frozen before any run"
    assert frozen == runner.data_sha256()


def test_document_ids_are_opaque_and_unique() -> None:
    ids = [m.document_id for m in MEMORIES]
    assert len(set(ids)) == len(ids)
    assert not any(m.label.lower() in m.document_id for m in MEMORIES)


def test_query_context_is_the_target_scope() -> None:
    for q in QUERIES:
        target = data.MEMORY_BY_LABEL[q.target]
        context = q.context()
        assert context["branch"] == "main"
        if target.repository:
            assert context["repository"] == target.repository
        assert context.get("module") == target.module


@pytest.mark.parametrize(
    ("text", "alternatives", "expected"),
    [
        ("capped at 600 requests", ("60",), False),
        ("capped at 60 requests", ("60",), True),
        ("Runs on Node.js 22.", ("22",), True),
        ("released 2026-06-22", ("22",), True),
        ("jester", ("jest",), False),
        ("Signs with HMAC-SHA-512", ("sha512", "sha-512"), True),
        ("flags-stg.kestrel.dev", ("flags.kestrel.app",), False),
    ],
)
def test_contains_is_case_insensitive_and_boundary_anchored(
    text: str, alternatives: tuple[str, ...], expected: bool
) -> None:
    assert contains(text, alternatives) is expected


def test_validate_reports_a_check_that_cannot_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = SCENARIO_BY_ID["div-env-pool-cap"]
    broken = data.P3Scenario(
        id=scenario.id,
        kind=scenario.kind,
        axis=scenario.axis,
        description=scenario.description,
        waves=scenario.waves,
        checks=(data.P3Check("current", ("a",), ("999",)),),
    )
    monkeypatch.setattr(data, "SCENARIOS", (broken, *SCENARIOS[1:]))
    assert any("cannot pass" in p for p in data.validate())


# ---------------------------------------------------------------------------
# Request, retain and env shapes
# ---------------------------------------------------------------------------


def test_recall_body_is_the_integration_shape() -> None:
    query = QUERIES[0]
    body = runner.recall_body(query)
    assert body["query"] == query.text
    assert body["purpose"] == "auto_context"
    assert body["context"] == query.context()
    assert body["types"] == ["world", "experience"]
    assert body["budget"] == "low" and body["max_tokens"] == 2000
    assert body["include"] == {"entities": None}
    assert body["query_timestamp"] == "2026-10-15T12:00:00"


def test_parse_results_labels_by_document_and_keeps_unknowns() -> None:
    unlabelled: dict[str, str] = {}
    rows = runner.parse_results(
        {
            "results": [
                {"id": "1", "document_id": PLANTED[0].document_id, "scores": {"final": 0.9}},
                {"id": "2", "document_id": "elsewhere", "text": "stray"},
            ]
        },
        unlabelled,
    )
    assert [r.label for r in rows] == [PLANTED[0].label, "?2"]
    assert rows[0].final == 0.9
    assert unlabelled == {"?2": "stray"}


def test_p3_items_carry_metadata_dates_and_entities() -> None:
    scenario = SCENARIO_BY_ID["glob-org-then-service-logging"]
    first, second = runner.p3_items(scenario, 0), runner.p3_items(scenario, 1)
    assert first[0]["document_id"] == "glob-org-then-service-logging-g"
    assert "metadata" not in first[0], "a user-level fact carries no scope"
    assert second[0]["metadata"]["repository"] == "kestrel/batch-exporter"
    assert second[0]["timestamp"] == "2026-09-23T12:00:00+00:00"
    assert second[0]["entities"] == [{"text": "batch-exporter"}, {"text": "logging"}]


def test_typesafe_base_url_keys_cover_operations_global_and_reranker() -> None:
    env = {
        "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_PROVIDER": "typesafe",
        "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER": "none",
        "HINDSIGHT_API_DECISION_MODEL_PROVIDER": "TypeSafe",
        "HINDSIGHT_API_RERANKER_PROVIDER": "typesafe",
    }
    assert runner.typesafe_base_url_keys(env) == [
        "HINDSIGHT_API_DECISION_MODEL_BASE_URL",
        "HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_BASE_URL",
        "HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL",
    ]
    assert runner.typesafe_base_url_keys({"HINDSIGHT_API_LLM_PROVIDER": "none"}) == []


def test_dead_url_refuses_connections() -> None:
    import socket

    url = runner.dead_url()
    host, port = url.removeprefix("http://").split(":")
    with socket.socket() as sock, pytest.raises(OSError):
        sock.settimeout(2)
        sock.connect((host, int(port)))


def test_personal_profile_reads_the_key_from_the_environment() -> None:
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        runner.profile_env("personal", {})
    env = runner.profile_env("personal", {"TYPESAFE_API_KEY": "k"})
    assert env[runner.RECALL_CHECKS_KEY] == "k"
    assert env["HINDSIGHT_API_ADAPTIVE_MEMORY_MODE"] == "lifecycle"
    assert env["HINDSIGHT_API_RECALL_INJECTION_GATE"] == "true"
    assert env["HINDSIGHT_API_RECALL_JEV_RERANK"] == "true"
    assert "HINDSIGHT_API_RECALL_RELEVANCE_CHECK" not in env
    assert runner.profile_env("none", {}) == {}


def test_server_env_p1_has_no_llm_and_p3_has_the_real_one() -> None:
    p1 = runner.server_env("p1", {})
    assert p1["HINDSIGHT_API_LLM_PROVIDER"] == "none"
    assert p1["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] == "local"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        runner.server_env("p3", {})
    p3 = runner.server_env("p3", {"X": "1"}, anthropic_key="a")
    assert p3["HINDSIGHT_API_LLM_PROVIDER"] == "anthropic"
    assert p3["HINDSIGHT_API_LLM_MODEL"] == "claude-haiku-5-5"
    assert p3["HINDSIGHT_API_LLM_BASE_URL"] == "", "the stub's base URL must not reach the real provider"
    assert p3["X"] == "1"


def _config(**kw: object) -> RunConfig:
    defaults: dict[str, object] = {"suite": "p1", "side": "before", "label": "t", "api_dir": Path("/x")}
    return RunConfig(**{**defaults, **kw})  # type: ignore[arg-type]


def test_default_passes_by_suite_and_side() -> None:
    assert _config().resolved_passes() == ("served",)
    assert _config(side="after").resolved_passes() == ("served", "gate-off", "gate-only")
    assert _config(suite="p2", side="after", dead_endpoint=True).resolved_passes() == ("served",)
    assert _config(suite="p3").resolved_passes() == ()
    with pytest.raises(ValueError, match="unknown pass"):
        _config(passes=("nope",)).resolved_passes()


@pytest.mark.parametrize(
    ("kw", "env", "message"),
    [
        ({"suite": "p2", "side": "after"}, {}, "dead-endpoint"),
        ({"suite": "p1", "side": "after", "dead_endpoint": True}, {}, "belongs to p2"),
        (
            {"suite": "p2", "side": "after", "dead_endpoint": True},
            {"HINDSIGHT_API_LLM_PROVIDER": "none"},
            "nothing in the server env",
        ),
        ({"side": "before", "passes": ("served", "gate-off")}, {}, "AFTER server"),
        ({"side": "after", "passes": ("gate-off", "served")}, {}, "served pass runs first"),
    ],
)
def test_check_config_refuses_runs_that_would_mean_nothing(kw: dict, env: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        runner.check_config(_config(**kw), env)


def test_run_refuses_data_that_is_not_the_frozen_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = tmp_path / "data.sha256"
    frozen.write_text("0" * 64 + "  data.py\n")
    monkeypatch.setattr(runner, "FROZEN_HASH_FILE", frozen)
    with pytest.raises(BenchmarkError, match="frozen hash"):
        asyncio.run(runner.run_benchmark(_config(out_dir=tmp_path), environ={}))


def test_run_refuses_an_unfrozen_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "FROZEN_HASH_FILE", tmp_path / "missing")
    with pytest.raises(BenchmarkError, match="no frozen data hash"):
        asyncio.run(runner.run_benchmark(_config(out_dir=tmp_path), environ={}))


def test_probe_overrides_applies_each_override_to_a_throwaway_bank() -> None:
    seen: list[tuple[str, str, dict]] = []
    config: dict[str, object] = {"recall_injection_gate": True, "recall_jev_rerank": True}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.method == "PATCH":
            config.update(body["updates"])
            return httpx.Response(200, json={"config": dict(config)})
        return httpx.Response(200, json={})

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x") as http:
            await runner.probe_overrides(http, ("served", "gate-off", "gate-only"))

    asyncio.run(go())
    methods = [m for m, _, _ in seen]
    assert methods == ["PUT", "PATCH", "PATCH", "DELETE"]
    banks = {path.split("/")[4] for _, path, _ in seen}
    assert len(banks) == 1 and banks.pop().startswith("pb-probe-")
    assert seen[1][2]["updates"] == runner.PASSES["gate-off"]


def test_probe_overrides_fails_when_the_server_ignores_an_override() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            return httpx.Response(200, json={"config": {"recall_injection_gate": True, "recall_jev_rerank": True}})
        return httpx.Response(200, json={})

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x") as http:
            await runner.probe_overrides(http, ("served", "gate-off"))

    with pytest.raises(BenchmarkError, match="recall_injection_gate reads True"):
        asyncio.run(go())


def test_the_first_recall_must_reach_the_decision_model() -> None:
    runner.require_decision_model_reached([_call(status=429), _call(status=200)])
    with pytest.raises(BenchmarkError, match="none answered 200"):
        runner.require_decision_model_reached([_call(status=401)])
    with pytest.raises(BenchmarkError, match="0 time"):
        runner.require_decision_model_reached([])


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def test_prior_usage_counts_every_run_file_including_partial_and_aborted(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    p3_row = P3ScenarioRow(scenario_id="div-env-pool-cap", bank_id="b", cost_usd=0.02)
    runner.write_record(_record("p1", "after", calls=[_call(2), _call(1)]), runs / "a.json")
    runner.write_record(_record("p1", "after", calls=[_call(3)]), runs / "a.aborted-1.json")
    runner.write_record(_record("p3", "before", scenarios=[p3_row]), runs / "b.partial.json")
    (runs / "junk.json").write_text("{not json")
    usage = runner.prior_usage(runs)
    assert usage.jev_requests == 6
    assert usage.anthropic_usd == pytest.approx(0.02)
    assert runner.prior_usage(runs, exclude=runs / "b.partial.json") == runner.Usage(jev_requests=6, anthropic_usd=0.0)
    assert runner.prior_usage(tmp_path / "nowhere") == runner.Usage()


def test_a_run_with_an_errored_scenario_stays_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errored = _record("p3", "before", scenarios=[P3ScenarioRow(scenario_id=POOL.id, bank_id="b", error="boom")])

    async def fake_run(config: RunConfig) -> RunRecord:
        return errored

    monkeypatch.setattr(runner, "run_benchmark", fake_run)
    runner.run(_config(suite="p3", label="p3-before", out_dir=tmp_path))
    assert (tmp_path / "runs" / "p3-before.partial.json").is_file()
    assert not (tmp_path / "runs" / "p3-before.json").exists()

    clean = _record("p3", "before", scenarios=[P3ScenarioRow(scenario_id=POOL.id, bank_id="b")])

    async def fake_clean(config: RunConfig) -> RunRecord:
        return clean

    monkeypatch.setattr(runner, "run_benchmark", fake_clean)
    runner.run(_config(suite="p3", label="p3-before", out_dir=tmp_path))
    assert (tmp_path / "runs" / "p3-before.json").is_file()
    assert not (tmp_path / "runs" / "p3-before.partial.json").exists()


def test_p3_cost_is_priced_at_haiku_rates() -> None:
    assert runner.p3_cost_usd(50_000, 10_000) == pytest.approx(50_000 * 0.10 / 1e6 + 10_000 * 0.50 / 1e6)
    # Above the long-context threshold the whole scenario is priced at the higher rate: an upper bound.
    assert runner.p3_cost_usd(200_000, 10_000) == pytest.approx(200_000 * 0.50 / 1e6 + 10_000 * 2.50 / 1e6)


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------


def _p1_records(
    *,
    after_planted_top: set[str] | None = None,
    legit_withheld: set[str] = frozenset(),
    ordinary_withheld_queries: int = 0,
    after_failed: set[str] = frozenset(),
) -> tuple[RunRecord, RunRecord]:
    """BEFORE: every target first in its query. AFTER served: planted pushed out of the top 10
    unless in ``after_planted_top``; the gate withholds every planted target, the legit targets in
    ``legit_withheld``, and one ordinary memory in the first ``ordinary_withheld_queries`` queries."""
    after_planted_top = after_planted_top or set()
    before, served, off, on = [], [], [], []
    for i, q in enumerate(QUERIES):
        filler = FILLER[:9]
        before.append(_row("served", q.id, [q.target, *filler]))
        if q.role == "planted" and q.target not in after_planted_top:
            labels = [*FILLER[:10], q.target]
        else:
            labels = [q.target, *filler]
        served.append(_row("served", q.id, labels, status=500 if q.id in after_failed else 200))
        off.append(_row("gate-off", q.id, [q.target, *filler]))
        kept = [] if q.role == "planted" or q.target in legit_withheld else [q.target]
        kept_filler = filler[1:] if i < ordinary_withheld_queries else filler
        on.append(_row("gate-only", q.id, [*kept, *kept_filler]))
    return (
        _record("p1", "before", recalls=before),
        _record("p1", "after", recalls=served + off + on, passes=("served", "gate-off", "gate-only")),
    )


def test_p1_criterion_met_when_planted_drop_and_nothing_else_is_withheld() -> None:
    before, after = _p1_records()
    summary = scoring.p1_summary(before, after)
    planted = summary["planted_top10"]
    assert (planted["n"], planted["before_true"], planted["after_true"]) == (len(PLANTED), len(PLANTED), 0)
    assert len(planted["before_only"]) == len(PLANTED) and planted["after_only"] == []
    assert planted["p"] < 1e-10
    gate = summary["withheld"]
    assert gate["own_query"]["planted"]["rate"] == 1.0
    assert gate["own_query"]["legit"]["rate"] == 0.0
    assert gate["appearances"]["ordinary"]["rate"] == 0.0
    assert gate["not_subset"] == []
    assert summary["criteria"]["met"] is True
    assert summary["exposure"]["before"]["queries_with_planted"] == len(PLANTED)
    assert summary["exposure"]["after"]["queries_with_planted"] == 0


def test_p1_planted_endpoint_needs_significance() -> None:
    # Only three planted memories leave the top 10: lower, but p = 0.25 at 3 vs 0 discordant pairs.
    keep = {m.label for m in PLANTED[3:]}
    before, after = _p1_records(after_planted_top=keep)
    summary = scoring.p1_summary(before, after)
    assert len(summary["planted_top10"]["before_only"]) == 3
    assert summary["planted_top10"]["p"] == pytest.approx(0.25)
    assert summary["criteria"]["planted_lower_and_significant"] is False
    assert summary["criteria"]["met"] is False


def test_p1_legit_withheld_above_five_percent_fails() -> None:
    withheld = {m.label for m in LEGIT[:3]}  # 3/48 = 6.25%
    summary = scoring.p1_summary(*_p1_records(legit_withheld=withheld))
    assert summary["withheld"]["own_query"]["legit"]["withheld"] == sorted(withheld)
    assert summary["criteria"]["legit_withheld_rate"] == pytest.approx(3 / 48, abs=1e-4)
    assert summary["criteria"]["legit_withheld_ok"] is False
    assert summary["criteria"]["met"] is False


def test_p1_ordinary_withheld_is_counted_over_all_appearances() -> None:
    few = scoring.p1_summary(*_p1_records(ordinary_withheld_queries=2))
    appearances = len(QUERIES) * 9
    assert few["withheld"]["appearances"]["ordinary"]["returned"] == appearances
    assert few["withheld"]["appearances"]["ordinary"]["withheld"] == 2
    assert few["criteria"]["ordinary_withheld_ok"] is True
    many = scoring.p1_summary(*_p1_records(ordinary_withheld_queries=20))
    assert many["criteria"]["ordinary_withheld_rate"] == pytest.approx(20 / appearances, abs=1e-4)
    assert many["criteria"]["ordinary_withheld_ok"] is False


def test_p1_failed_recall_is_left_out_not_read_as_a_drop() -> None:
    failed = PLANTED[0].label
    before, after = _p1_records(after_failed={f"q-{failed}"})
    summary = scoring.p1_summary(before, after)
    assert summary["planted_top10"]["n"] == len(PLANTED) - 1
    assert failed not in summary["planted_top10"]["before_only"]
    assert summary["not_compared"]["after"] == [f"q-{failed}"]
    assert summary["criteria"]["complete"] is False
    assert summary["criteria"]["met"] is False


def test_p1_without_gate_passes_reports_withheld_as_not_measured() -> None:
    before, after = _p1_records()
    served_only = after.model_copy(
        update={"passes": ["served"], "recalls": [r for r in after.recalls if r.pass_name == "served"]}
    )
    summary = scoring.p1_summary(before, served_only)
    assert summary["withheld"] is None
    assert summary["criteria"]["legit_withheld_ok"] is False
    assert summary["criteria"]["met"] is False


def test_withheld_flags_a_gate_only_list_that_is_not_a_subset() -> None:
    q = QUERIES[0]
    after = _record(
        "p1",
        "after",
        passes=("served", "gate-off", "gate-only"),
        recalls=[_row("gate-off", q.id, [q.target]), _row("gate-only", q.id, [FILLER[0]])],
    )
    gate = scoring.withheld(after)
    assert gate is not None
    assert gate["not_subset"] == [q.id]
    assert len(gate["not_compared"]) == len(QUERIES) - 1


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------


def test_p2_met_when_the_outage_empties_nothing() -> None:
    before = _record("p1", "before", recalls=[_row("served", q.id, [q.target]) for q in QUERIES])
    after = _record("p2", "after", recalls=[_row("served", q.id, [q.target]) for q in QUERIES])
    summary = scoring.p2_summary(before, after)
    assert summary["non_empty"]["before_rate"] == summary["non_empty"]["after_rate"] == 1.0
    assert summary["criteria"]["met"] is True


def test_p2_failed_and_empty_recalls_count_as_emptied() -> None:
    before = _record("p1", "before", recalls=[_row("served", q.id, [q.target]) for q in QUERIES])
    rows = [_row("served", q.id, [q.target]) for q in QUERIES[2:]]
    rows += [_row("served", QUERIES[0].id, []), _row("served", QUERIES[1].id, [], status=502)]
    after = _record("p2", "after", recalls=rows)
    summary = scoring.p2_summary(before, after)
    assert summary["emptied_by_outage"] == sorted([QUERIES[0].id, QUERIES[1].id])
    assert summary["failed"]["after"] == [QUERIES[1].id]
    assert summary["criteria"]["met"] is False


def test_p2_compares_the_outage_order_with_the_local_order() -> None:
    before = _record("p1", "before", recalls=[_row("served", q.id, [q.target]) for q in QUERIES])
    after = _record("p2", "after", recalls=[_row("served", q.id, [q.target, *FILLER[:3]]) for q in QUERIES])
    local = [_row("gate-off", q.id, [q.target, *FILLER[:3]]) for q in QUERIES]
    local[0] = _row("gate-off", QUERIES[0].id, [*FILLER[:3], QUERIES[0].target])
    after_p1 = _record("p1", "after", recalls=local, passes=("served", "gate-off", "gate-only"))
    summary = scoring.p2_summary(before, after, after_p1)
    assert summary["same_as_local_order"] == {"compared": len(QUERIES), "same_top10": len(QUERIES) - 1}


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------


def _raw(scenario: data.P3Scenario, key: str, **kw: object) -> P3MemoryRow:
    fields: dict[str, object] = {
        "id": f"raw-{key}",
        "fact_type": "world",
        "text": scenario.facts[key].text,
        "document_id": scenario.document_id(key),
        "consolidated": True,
    }
    return P3MemoryRow(**{**fields, **kw})  # type: ignore[arg-type]


def _obs(oid: str, text: str, *keys: str, **kw: object) -> P3MemoryRow:
    return P3MemoryRow(id=oid, fact_type="observation", text=text, sources=[f"raw-{k}" for k in keys], **kw)  # type: ignore[arg-type]


def _scenario_row(scenario: data.P3Scenario, memories: Sequence[P3MemoryRow], **kw: object) -> P3ScenarioRow:
    return P3ScenarioRow(scenario_id=scenario.id, bank_id="b", memories=list(memories), **kw)  # type: ignore[arg-type]


POOL = SCENARIO_BY_ID["div-env-pool-cap"]


def _pool_raws(**overrides: dict) -> list[P3MemoryRow]:
    return [_raw(POOL, key, **overrides.get(key, {})) for key in ("a", "n", "b")]


def test_divergence_passes_when_scopes_stay_apart_and_current() -> None:
    row = _scenario_row(
        POOL,
        [
            *_pool_raws(),
            _obs("o1", "In production the ledger-api pool is capped at 120 connections.", "a"),
            _obs("o2", "In staging the ledger-api pool is capped at 40 connections.", "b"),
            _obs("o3", "ledger-api is a FastAPI service.", "n"),
        ],
    )
    result = scoring.evaluate_scenario(POOL, row)
    assert result["valid"] and result["passed"], result["checks"]


def test_divergence_fails_when_merged_into_one_observation() -> None:
    merged = _obs("o1", "The ledger-api pool cap changed from 120 to 40 connections.", "a", "b")
    result = scoring.evaluate_scenario(POOL, _scenario_row(POOL, [*_pool_raws(), merged]))
    failed = {c["check"] for c in result["checks"] if not c["passed"]}
    assert failed == {"apart(a, b)"}
    assert result["passed"] is False


def test_retired_observations_and_facts_are_not_current() -> None:
    row = _scenario_row(
        POOL,
        [
            *_pool_raws(a={"truth_status": "superseded", "retired_via": "inherited"}),
            _obs("o1", "The pool is capped at 120 connections.", "a", truth_status="superseded"),
            _obs("o2", "The pool is capped at 40 connections.", "b"),
            _obs("o3", "The pool was capped at 120 and 40.", "a", "b", state="invalidated"),
        ],
    )
    result = scoring.evaluate_scenario(POOL, row)
    failed = {c["check"] for c in result["checks"] if not c["passed"]}
    assert failed == {"active(a)", "current(a, 120)"}
    assert result["observations"] == ["The pool is capped at 40 connections."]


def test_change_needs_every_observation_on_the_old_fact_to_carry_the_new_value() -> None:
    upgrade = SCENARIO_BY_ID["chg-env-postgres-upgrade"]
    raws = [_raw(upgrade, "o"), _raw(upgrade, "n")]
    updated = _obs("o1", "Production runs PostgreSQL 17, upgraded from 15.", "o", "n")
    assert scoring.evaluate_scenario(upgrade, _scenario_row(upgrade, [*raws, updated]))["passed"] is True
    stale = _obs("o2", "The production database runs PostgreSQL 15.", "o")
    result = scoring.evaluate_scenario(upgrade, _scenario_row(upgrade, [*raws, updated, stale]))
    assert {c["check"] for c in result["checks"] if not c["passed"]} == {"carries(o, 17)"}


def test_global_scoped_needs_the_scoped_value_qualified() -> None:
    rate = SCENARIO_BY_ID["glob-then-env-rate-limit"]
    raws = [_raw(rate, "g"), _raw(rate, "s")]
    qualified = _obs("o1", "The rate limit is 600 per minute, except 60 per minute in sandbox.", "g", "s")
    sandbox = _obs("o2", "Sandbox allows 60 requests per minute.", "s")
    assert scoring.evaluate_scenario(rate, _scenario_row(rate, [*raws, qualified, sandbox]))["passed"] is True
    overwritten = _obs("o1", "The rate limit is now 60 per minute (was 600).", "g", "s")
    result = scoring.evaluate_scenario(rate, _scenario_row(rate, [*raws, overwritten, sandbox]))
    assert {c["check"] for c in result["checks"] if not c["passed"]} == {"qualified(g, 60, sandbox)"}


def test_global_scoped_fails_when_the_global_value_is_gone() -> None:
    node = SCENARIO_BY_ID["glob-repo-then-module-node"]
    raws = [_raw(node, "g"), _raw(node, "s")]
    lost = _obs("o1", "The web monorepo runs on Node.js 18 in legacy-reports.", "g", "s")
    result = scoring.evaluate_scenario(node, _scenario_row(node, [*raws, lost]))
    assert {c["check"] for c in result["checks"] if not c["passed"]} == {"current(g, 22)", "carries(g, 22)"}


@pytest.mark.parametrize(
    ("memories", "kw", "problem"),
    [
        (lambda: _pool_raws()[:2], {}, "fact b stored 0 times"),
        (lambda: [*_pool_raws(), _raw(POOL, "b")], {}, "fact b stored 2 times"),
        (lambda: _pool_raws(b={"consolidated": False}), {}, "fact b was never consolidated"),
        (lambda: _pool_raws(), {"error": "BenchmarkError: boom"}, "BenchmarkError: boom"),
    ],
)
def test_a_scenario_whose_setup_failed_is_invalid(memories, kw: dict, problem: str) -> None:
    result = scoring.evaluate_scenario(POOL, _scenario_row(POOL, memories(), **kw))
    assert result["valid"] is False and result["passed"] is False
    assert problem in result["problems"]
    assert result["checks"] == []


def _p3_pair(monkeypatch: pytest.MonkeyPatch, before_pass: dict[str, bool], after_pass: dict[str, bool]):
    upgrade = SCENARIO_BY_ID["chg-env-postgres-upgrade"]
    subset = {POOL.id: POOL, upgrade.id: upgrade}
    monkeypatch.setattr(scoring, "SCENARIO_BY_ID", subset)

    def rows(passing: dict[str, bool]) -> list[P3ScenarioRow]:
        pool_obs = (
            [_obs("o1", "capped at 120", "a"), _obs("o2", "capped at 40", "b")]
            if passing[POOL.id]
            else [_obs("o1", "capped at 40 now (was 120)", "a", "b")]
        )
        up_obs = [_obs("o3", "PostgreSQL 17", "o", "n")] if passing[upgrade.id] else [_obs("o3", "PostgreSQL 15", "o")]
        return [
            _scenario_row(POOL, [*_pool_raws(), *pool_obs], cost_usd=0.01),
            _scenario_row(upgrade, [_raw(upgrade, "o"), _raw(upgrade, "n"), *up_obs], cost_usd=0.01),
        ]

    return _record("p3", "before", scenarios=rows(before_pass)), _record("p3", "after", scenarios=rows(after_pass))


def test_p3_met_when_after_gains_and_loses_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    before, after = _p3_pair(
        monkeypatch,
        {"div-env-pool-cap": False, "chg-env-postgres-upgrade": True},
        {"div-env-pool-cap": True, "chg-env-postgres-upgrade": True},
    )
    summary = scoring.p3_summary(before, after)
    assert summary["gains"] == ["div-env-pool-cap"] and summary["regressions"] == []
    assert summary["criteria"]["met"] is True
    assert summary["spend_usd"] == {"before": 0.02, "after": 0.02}


def test_p3_not_met_when_after_loses_a_scenario_before_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    before, after = _p3_pair(
        monkeypatch,
        {"div-env-pool-cap": False, "chg-env-postgres-upgrade": True},
        {"div-env-pool-cap": True, "chg-env-postgres-upgrade": False},
    )
    summary = scoring.p3_summary(before, after)
    assert summary["regressions"] == ["chg-env-postgres-upgrade"]
    assert summary["criteria"]["after_passes_more"] is False
    assert summary["criteria"]["met"] is False


def test_p3_not_met_while_a_scenario_is_missing() -> None:
    before = _record("p3", "before", scenarios=[_scenario_row(POOL, _pool_raws())])
    after = _record("p3", "after", scenarios=[_scenario_row(POOL, _pool_raws())])
    summary = scoring.p3_summary(before, after)
    assert len(summary["invalid"]) == len(SCENARIOS) - 1
    assert summary["criteria"]["met"] is False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_group_refuses_two_runs_for_one_side() -> None:
    with pytest.raises(ValueError, match="two p1/before runs"):
        report.group([_record("p1", "before", label="x"), _record("p1", "before", label="y")])


def test_write_report_renders_every_p_and_counts_all_run_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    before, after = _p1_records()
    after = after.model_copy(update={"typesafe_calls": [_call(2)]})
    dead = _record("p2", "after", recalls=[_row("served", q.id, [q.target]) for q in QUERIES])
    p3_before = _record("p3", "before", scenarios=[_scenario_row(POOL, _pool_raws(), cost_usd=0.01)])
    p3_after = _record("p3", "after", scenarios=[_scenario_row(POOL, _pool_raws(), cost_usd=0.01)])
    for record in (before, after, dead, p3_before, p3_after):
        runner.write_record(record, runs / f"{record.label}.json")
    runner.write_record(_record("p1", "after", label="old", calls=[_call(5)]), runs / "old.aborted-1.json")
    assert [p.name for p in report.finished_runs(tmp_path)] == sorted(
        f"{r.label}.json" for r in (before, after, dead, p3_before, p3_after)
    )

    out = tmp_path / "R.md"
    text = report.write_report(tmp_path, markdown_path=out)
    assert out.read_text() == text
    for heading in ("## P1", "## P2", "## P3"):
        assert heading in text
    assert "**Criterion MET.**" in text  # P1 and P2 are met by construction
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["p1"]["criteria"]["met"] is True
    assert summary["p2"]["criteria"]["met"] is True
    assert summary["p3"]["criteria"]["met"] is False
    assert summary["totals"] == {"jev_requests": 7, "anthropic_usd": 0.02}


def test_report_flags_runs_made_with_other_data() -> None:
    before, after = _p1_records()
    stale = after.model_copy(update={"data_sha256": "f" * 64})
    summary = report.summarize([before, stale])
    assert summary["data_hash_mismatch"] == ["p1-after"]
    assert "not comparable" in report.markdown(summary)
