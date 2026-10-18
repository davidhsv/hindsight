"""The harm benchmark's own mechanics: scenario integrity, request and prompt shapes, the
Messages API client, verdict parsing, pricing, the budget guard, and every metric. Server-free."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from hindsight_system_tests.context_bench.runner import BenchmarkError
from hindsight_system_tests.context_bench.scenarios import MEMORIES as CONTEXT_BENCH_MEMORIES
from hindsight_system_tests.harm_bench import scoring
from hindsight_system_tests.harm_bench.h2 import GROUPS, H2_MEMORIES, H2_TASKS, validate_h2
from hindsight_system_tests.harm_bench.llm import (
    MEMORY_PREAMBLE,
    VERDICT_SCHEMA,
    Completion,
    InjectedMemory,
    LLMError,
    MessagesClient,
    ModelSettings,
    Usage,
    Verdict,
    answer_prompt,
    cost_usd,
    injected_tokens,
    judge_prompt,
    parse_verdict,
)
from hindsight_system_tests.harm_bench.runner import (
    AnswerRow,
    RecallRow,
    RunConfig,
    RunRecord,
    _Budget,
    _work,
    check_conditions,
    parse_results,
    recall_body,
    reused_recalls,
)
from hindsight_system_tests.harm_bench.scenario import MEMORIES, MEMORY_BY_LABEL, TASK_BY_ID, TASKS, validate
from hindsight_system_tests.harm_bench.suites import SUITES
from hindsight_system_tests.harm_bench.suites import TASK_BY_ID as ALL_TASKS

H = TASK_BY_ID["H01"]
U = TASK_BY_ID["U01"]
N = TASK_BY_ID["N01"]


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def test_scenario_is_consistent() -> None:
    assert validate() == []
    assert len(TASKS) == 24
    assert {kind: sum(1 for t in TASKS if t.kind == kind) for kind in "HUN"} == {"H": 8, "U": 8, "N": 8}
    assert len(MEMORIES) >= 150
    assert {m.project for m in MEMORIES} == {"o", "p", "d", None}


def test_h2_suite_is_consistent_and_leaves_the_original_bank_alone() -> None:
    assert validate_h2() == []
    assert {g: sum(1 for t in H2_TASKS if t.group == g) for g in GROUPS} == {g: 4 for g in GROUPS}
    assert sum(1 for t in H2_TASKS if t.kind == "H") == 12 and sum(1 for t in H2_TASKS if t.kind == "U") == 4
    original, h2 = SUITES["original"], SUITES["h2"]
    assert original.memories == MEMORIES and original.tasks == TASKS and original.exploratory is None
    assert h2.memories[: len(MEMORIES)] == MEMORIES and h2.memories[len(MEMORIES) :] == H2_MEMORIES
    assert h2.exploratory and "after the base result" in h2.exploratory
    # Stale traps are superseded only in ground truth; nothing in their text says so.
    for task in H2_TASKS:
        if task.group == "stale":
            (stale,) = task.trap_ids
            memory = next(mem for mem in H2_MEMORIES if mem.label == stale)
            assert memory.status == "superseded" and "old" not in memory.text.lower().split()


def test_bank_is_distinct_from_context_bench() -> None:
    assert not {m.text for m in MEMORIES} & {m.text for m in CONTEXT_BENCH_MEMORIES}
    assert not {m.document_id for m in MEMORIES} & {m.document_id for m in CONTEXT_BENCH_MEMORIES}


def test_validate_catches_broken_tasks() -> None:
    broken = (
        dataclasses.replace(H, trap_ids=("nope",)),
        dataclasses.replace(U, useful_ids=()),
        dataclasses.replace(N, harm_signature="x"),
    )
    problems = validate(MEMORIES, broken)
    assert any("unknown memory nope" in p for p in problems)
    assert any("U01: a U task needs useful_ids" in p for p in problems)
    assert any("N01: only H tasks have traps" in p for p in problems)
    assert any("tasks of kind H" in p for p in problems)


def test_memory_scope_metadata_and_tags() -> None:
    trap = MEMORY_BY_LABEL["o.trap.pool20"]
    assert trap.metadata == {
        "branch": "main",
        "repository": "acme/orders-service",
        "module": "api",
        "environment": "prod",
    }
    assert trap.tags == ["project:o"]
    assert MEMORY_BY_LABEL["g.humor.vpn"].metadata == {"branch": "main"}
    assert MEMORY_BY_LABEL["g.humor.vpn"].tags == []


# ---------------------------------------------------------------------------
# Requests and prompts
# ---------------------------------------------------------------------------


def test_recall_body_per_condition() -> None:
    ungated = recall_body(TASK_BY_ID["H03"], "ungated", max_tokens=4096)
    assert ungated["query"] == TASK_BY_ID["H03"].text
    assert ungated["budget"] == "mid"
    assert "purpose" not in ungated and "context" not in ungated and "tags" not in ungated
    gated = recall_body(TASK_BY_ID["H03"], "gated", max_tokens=4096)
    assert gated["purpose"] == "auto_context"
    assert gated["context"] == {
        "repository": "acme/orders-service",
        "branch": "main",
        "module": "api",
        "environment": "prod",
    }
    with pytest.raises(ValueError):
        recall_body(H, "none", max_tokens=4096)


def test_condition_combinations() -> None:
    check_conditions(("none", "ungated"), {})
    check_conditions(("gated",), {"HINDSIGHT_API_CONTEXT_ADMISSION_MODE": "strict"})
    with pytest.raises(ValueError, match="separately"):
        check_conditions(("ungated", "gated"), {"HINDSIGHT_API_CONTEXT_ADMISSION_MODE": "strict"})
    with pytest.raises(ValueError, match="needs"):
        check_conditions(("gated",), {})
    with pytest.raises(ValueError, match="no gated"):
        check_conditions(("ungated",), {"HINDSIGHT_API_CONTEXT_ADMISSION_MODE": "strict"})
    with pytest.raises(ValueError, match="unknown"):
        check_conditions(("bogus",), {})


def test_parse_results_labels_and_cuts_at_top_k() -> None:
    known = MEMORY_BY_LABEL["o.trap.jackson"]
    payload = {
        "results": [
            {
                "id": "a",
                "text": known.text,
                "document_id": known.document_id,
                "type": "world",
                "scores": {"final": 0.9},
            },
            {"id": "b", "text": "stray", "document_id": "other"},
            {"id": "c", "text": "third", "document_id": None},
        ]
    }
    unlabelled: dict[str, str] = {}
    rows, injected = parse_results(payload, 2, unlabelled)
    assert [r.label for r in rows] == ["o.trap.jackson", "?b", "?c"]
    assert rows[0].final == 0.9
    assert [m.label for m in injected] == ["o.trap.jackson", "?b"]
    assert injected[0].type == "world"
    assert unlabelled == {"?b": "stray", "?c": "third"}


def test_answer_prompt_without_and_with_memories() -> None:
    system, user = answer_prompt(H, [])
    assert user == H.text
    assert "acme/orders-service" in system and "orders-service" in system
    memories = [InjectedMemory(label="x", text="Remember this.", type="world", mentioned_at="2026-01-01")]
    _, user = answer_prompt(H, memories)
    assert user.startswith("<hindsight_memories>\n" + MEMORY_PREAMBLE)
    assert "- Remember this. [world] (2026-01-01)" in user
    assert user.endswith(H.text)
    assert injected_tokens([]) == 0
    assert injected_tokens(memories) > 0


def test_nopreamble_condition_drops_only_the_preamble() -> None:
    memories = [InjectedMemory(label="x", text="Remember this.", type="world", mentioned_at="2026-01-01")]
    _, with_preamble = answer_prompt(H, memories)
    _, without = answer_prompt(H, memories, preamble=False)
    assert MEMORY_PREAMBLE not in without
    assert without == with_preamble.replace(MEMORY_PREAMBLE + "\n", "")
    assert without.startswith("<hindsight_memories>\nCurrent time - ")
    assert "- Remember this. [world] (2026-01-01)" in without and without.endswith(H.text)
    assert answer_prompt(H, [], preamble=False)[1] == H.text
    assert injected_tokens(memories, preamble=False) < injected_tokens(memories)
    # The same recall request as ungated, so the two can share recorded results.
    assert recall_body(H, "ungated-nopreamble", max_tokens=4096) == recall_body(H, "ungated", max_tokens=4096)
    with pytest.raises(ValueError, match="separately"):
        check_conditions(("ungated-nopreamble", "gated"), {"HINDSIGHT_API_CONTEXT_ADMISSION_MODE": "strict"})


def test_reused_recalls_copy_the_ungated_top_k() -> None:
    def recall(task_id: str, condition: str, status: int = 200) -> RecallRow:
        return RecallRow(
            task_id=task_id,
            condition=condition,
            status=status,
            latency_ms=1.0,
            results=[],
            injected=[InjectedMemory(label=f"{task_id}.m", text="t")],
        )

    source = _record("base", ["none", "ungated"], [])
    source.recalls = [recall("H01", "ungated"), recall("U01", "ungated")]
    rows = reused_recalls(source, ["ungated-nopreamble"], ["H01", "U01"])
    assert [(r.task_id, r.condition) for r in rows] == [("H01", "ungated-nopreamble"), ("U01", "ungated-nopreamble")]
    assert rows[0].injected == source.recalls[0].injected
    assert source.recalls[0].condition == "ungated"  # the source record is left as it was
    assert reused_recalls(source, ["none"], ["H01"]) == []
    with pytest.raises(BenchmarkError, match="N01"):
        reused_recalls(source, ["ungated-nopreamble"], ["H01", "N01"])
    source.recalls = [recall("H01", "ungated", status=500)]
    with pytest.raises(BenchmarkError):
        reused_recalls(source, ["ungated-nopreamble"], ["H01"])


def test_judge_prompt_carries_rubric_and_harm_definition() -> None:
    prompt = judge_prompt(H, "the answer")
    for criterion in (*H.must_include, *H.must_not):
        assert criterion in prompt
    assert H.harm_signature and H.harm_signature in prompt
    assert H.reference in prompt
    assert "always set harm=false" in judge_prompt(U, "x")


def test_parse_verdict() -> None:
    raw = json.dumps({"success": True, "harm": True, "unmet": [], "reason": "ok"})
    assert parse_verdict(H, raw).harm is True
    # A task without a harm signature can never be harmed.
    assert parse_verdict(U, raw).harm is False
    with pytest.raises(LLMError):
        parse_verdict(H, "not json")
    with pytest.raises(LLMError):
        parse_verdict(H, json.dumps({"success": True}))


# ---------------------------------------------------------------------------
# The Messages API client
# ---------------------------------------------------------------------------


async def test_client_sends_documented_shape_and_retries_overload() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        if len(seen) == 1:
            return httpx.Response(529, headers={"retry-after": "0"}, json={"type": "error"})
        return httpx.Response(
            200,
            headers={"request-id": "req_1"},
            json={
                "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"success": true}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        )

    client = MessagesClient("test-key", transport=httpx.MockTransport(handler))
    try:
        completion = await client.create(
            ModelSettings("claude-sonnet-5-5", "medium", 4000), system="sys", user="hi", json_schema=VERDICT_SCHEMA
        )
    finally:
        await client.aclose()
    assert len(seen) == 2 and completion.attempts == 2
    body = seen[-1]["body"]
    assert body["model"] == "claude-sonnet-5-5"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}}
    assert "temperature" not in body and "thinking" not in body
    assert seen[-1]["headers"]["x-api-key"] == "test-key"
    assert seen[-1]["headers"]["anthropic-version"] == "2023-06-01"
    assert completion.text == '{"success": true}'
    assert completion.cost_usd == pytest.approx(0.007)
    assert completion.request_id == "req_1"


async def test_client_fails_fast_on_a_bad_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "temperature is not supported"}})

    client = MessagesClient("k", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError, match="HTTP 400"):
            await client.create(ModelSettings("claude-haiku-5-5", None, 100), system="s", user="u")
    finally:
        await client.aclose()
    assert calls == 1


def test_client_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        MessagesClient()


def test_pricing() -> None:
    assert cost_usd("claude-haiku-5-5", Usage(input_tokens=100_000, output_tokens=100_000)) == pytest.approx(0.06)
    assert cost_usd("claude-sonnet-5-5", Usage(input_tokens=1000, output_tokens=500)) == pytest.approx(0.007)
    # Beyond 100K prompt tokens Haiku is priced at the long-context rate.
    assert cost_usd("claude-haiku-5-5", Usage(input_tokens=200_000, output_tokens=0)) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        cost_usd("claude-unknown", Usage())


def test_budget_guard_counts_in_flight_work() -> None:
    budget = _Budget(cap_usd=1.0, spent_usd=0.0)
    assert budget.admit()  # 0 + 1 x 0.05 + 0.25 <= 1
    budget.settle(0.30)
    assert budget.max_pair == pytest.approx(0.30)
    assert budget.admit()  # 0.30 + 1 x 0.30 + 0.25 = 0.85
    assert not budget.admit()  # 0.30 + 2 x 0.30 + 0.25 > 1
    assert budget.exhausted


def test_work_order_is_repeat_major_and_skips_done(tmp_path: Path) -> None:
    config = RunConfig(label="t", conditions=("none", "ungated"), out_dir=tmp_path, repeats=2)
    work = _work(config, [H, U], {("H01", "none", 1)})
    keys = [(t.id, c, r) for t, c, r in work]
    assert keys[0] == ("H01", "ungated", 1)
    assert keys[:3] == [("H01", "ungated", 1), ("U01", "none", 1), ("U01", "ungated", 1)]
    assert len(keys) == 7


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _completion(model: str = "claude-haiku-5-5", cost: float = 0.001, stop: str = "end_turn") -> Completion:
    return Completion(
        model=model,
        text="x",
        stop_reason=stop,
        usage=Usage(input_tokens=100, output_tokens=50),
        cost_usd=cost,
        latency_ms=1.0,
        attempts=1,
    )


def ans(
    task_id: str, condition: str, repeat: int, ok: bool | None, harmed: bool = False, injected: tuple[str, ...] = ()
) -> AnswerRow:
    task = ALL_TASKS[task_id]
    return AnswerRow(
        task_id=task_id,
        kind=task.kind,
        condition=condition,
        repeat=repeat,
        injected_labels=list(injected),
        injected_tokens=10 * len(injected),
        answer=_completion(),
        judge=_completion("claude-sonnet-5-5", 0.01) if ok is not None else None,
        verdict=Verdict(success=ok, harm=harmed, unmet=[], reason="r") if ok is not None else None,
        error=None if ok is not None else "boom",
    )


def test_rate_excludes_failed_calls_and_is_undefined_when_empty() -> None:
    rows = [ans("H01", "none", 1, True), ans("H01", "none", 2, False), ans("H02", "none", 1, None)]
    assert scoring.rate(rows, scoring.success) == pytest.approx(0.5)
    assert scoring.rate([], scoring.success) is None
    assert scoring.rate([ans("H01", "none", 1, None)], scoring.success) is None


def test_per_task_majority_and_split_rule() -> None:
    rows = [
        ans("H01", "none", 1, True),
        ans("H01", "none", 2, False),
        ans("H02", "none", 1, True),
        ans("H02", "none", 2, True),
        ans("H03", "none", 1, False),
        ans("H03", "none", 2, False),
    ]
    assert scoring.per_task(rows, scoring.success) == {"H01": False, "H02": True, "H03": False}
    assert scoring.per_task(rows, scoring.success, split_counts_as=True) == {"H01": True, "H02": True, "H03": False}
    three = [ans("H01", "none", r, ok) for r, ok in ((1, True), (2, True), (3, False))]
    assert scoring.per_task(three, scoring.success) == {"H01": True}


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [(0, 0, 1.0), (5, 0, 0.0625), (0, 5, 0.0625), (6, 1, 0.125), (10, 0, 2 / 1024), (3, 3, 1.0)],
)
def test_mcnemar_exact(b: int, c: int, p: float) -> None:
    assert scoring.mcnemar_exact(b, c) == pytest.approx(p)


def test_paired_counts_only_shared_tasks() -> None:
    result = scoring.paired({"H01": True, "H02": False, "H03": True}, {"H01": False, "H02": False, "H04": True})
    assert result["n"] == 2
    assert result["a_only"] == ["H01"] and result["b_only"] == []
    assert result["p"] == pytest.approx(1.0)


def test_injection_checks_traps_any_and_useful_all() -> None:
    def recall(task_id: str, labels: tuple[str, ...]) -> RecallRow:
        return RecallRow(
            task_id=task_id,
            condition="ungated",
            status=200,
            latency_ms=1.0,
            results=[],
            injected=[InjectedMemory(label=label, text="t") for label in labels],
        )

    recalls = [
        recall("H02", ("g.humor.vpn", "o.oncall")),  # one of two traps is enough
        recall("H01", ("o.oncall",)),
        recall("U01", ("p.use.deploy",)),
    ]
    assert scoring.injection(recalls, "H") == {"tasks": 2, "hit": 1, "rate": 0.5, "hit_tasks": ["H02"]}
    assert scoring.injection(recalls, "U")["rate"] == 1.0
    assert scoring.injection(recalls, "N") == {}


def _record(label: str, conditions: list[str], answers: list[AnswerRow], suite: str = "original") -> RunRecord:
    return RunRecord(
        label=label,
        suite=suite,
        api_dir=None,
        api_sha="abc",
        api_dirty=False,
        harness_sha=None,
        server_env={},
        conditions=conditions,
        repeats=2,
        top_k=10,
        recall_max_tokens=4096,
        answer_settings={"model": "claude-haiku-5-5", "effort": "medium", "max_tokens": 8000},
        judge_settings={"model": "claude-sonnet-5-5", "effort": "medium", "max_tokens": 6000},
        budget_usd=6.0,
        started_at="2026-10-17T00:00:00+00:00",
        answers=answers,
    )


def test_summary_rates_benefit_and_comparisons() -> None:
    trap = ("o.trap.jackson",)
    none = [
        ans("H01", "none", 1, True),
        ans("H01", "none", 2, True),
        ans("U01", "none", 1, False),
        ans("U01", "none", 2, False),
        ans("N01", "none", 1, True),
        ans("N01", "none", 2, None),
    ]
    ungated = [
        ans("H01", "ungated", 1, False, harmed=True, injected=trap),
        ans("H01", "ungated", 2, False, harmed=True, injected=trap),
        ans("U01", "ungated", 1, True, injected=("p.use.deploy",)),
        ans("U01", "ungated", 2, False, injected=("p.use.deploy",)),
        ans("N01", "ungated", 1, True, injected=("o.money",)),
        ans("N01", "ungated", 2, True, injected=("o.money",)),
    ]
    summary = scoring.summarize([_record("base", ["none", "ungated"], none + ungated)])
    s_none, s_ung = summary["conditions"]["none"], summary["conditions"]["ungated"]
    assert s_none["judged"] == 5 and s_none["failed"] == 1
    assert s_none["success_by_kind"] == {"H": 1.0, "U": 0.0, "N": 1.0}
    assert s_none["harm_H"] == 0.0
    assert s_ung["harm_H"] == 1.0
    assert s_ung["harm_H_trap_injected"] == 1.0 and s_ung["harm_H_trap_injected_n"] == 2
    assert s_ung["useful_benefit_U"] == pytest.approx(0.5)
    assert s_none["useful_benefit_U"] == 0.0
    assert s_ung["mean_injected_memories"] == 1.0
    assert s_none["mean_injected_tokens"] == 0.0
    assert summary["spend_usd"] == pytest.approx(12 * 0.001 + 11 * 0.01)
    harm_pair = summary["comparisons"]["none_vs_ungated"]["harm_H"]["majority"]
    assert harm_pair["b_only"] == ["H01"] and harm_pair["p"] == 1.0
    success_u = summary["comparisons"]["none_vs_ungated"]["success_U"]
    # U01 split 1-1 under ungated: false by majority, true when a split counts as true.
    assert success_u["majority"]["b_true"] == 0 and success_u["any"]["b_true"] == 1
    assert summary["per_task"]["H01"]["ungated"]["harm"] == [True, True]
    report = scoring.markdown(summary)
    assert "| none |" in report and "| ungated |" in report and "| H01 | H |" in report


def test_report_marks_exploratory_conditions() -> None:
    base = _record("base", ["none", "ungated"], [ans("H01", "none", 1, True), ans("H01", "ungated", 1, True)])
    extra = _record("base-nopreamble", ["ungated-nopreamble"], [ans("H01", "ungated-nopreamble", 1, False, True)])
    summary = scoring.summarize([base, extra])
    assert list(summary["conditions"]) == ["none", "ungated", "ungated-nopreamble"]
    assert summary["exploratory"] == {"ungated-nopreamble": "exploratory, added after the ungated result was seen"}
    assert "ungated_vs_ungated-nopreamble" in summary["comparisons"]
    report = scoring.markdown(summary)
    assert "| ungated-nopreamble (exploratory) |" in report
    assert "| ungated vs ungated-nopreamble (exploratory) | harm_H |" in report
    assert "exploratory, added after the ungated result was seen" in report
    assert "(exploratory)" not in scoring.markdown(scoring.summarize([base]))


def test_suites_are_summarised_apart_with_group_tables() -> None:
    base = _record("base", ["none", "ungated"], [ans("H01", "none", 1, True), ans("H01", "ungated", 1, True)])
    h2 = _record(
        "base-h2",
        ["none", "ungated"],
        [
            ans("S01", "none", 1, False),
            ans("S01", "ungated", 1, False, harmed=True, injected=("o.h2.timeout.old", "o.h2.timeout.new")),
            ans("S01", "ungated", 2, True, injected=("o.h2.timeout.old", "o.h2.timeout.new")),
            ans("M01", "ungated", 1, True, injected=("o.oncall",)),
            ans("C01", "ungated", 1, True, injected=("o.h2.actuator",)),
        ],
        suite="h2",
    )
    with pytest.raises(ValueError, match="one suite"):
        scoring.summarize([base, h2])
    summaries = scoring.summarize_suites([h2, base])
    assert list(summaries) == ["original", "h2"]
    assert "by_group" not in summaries["original"]["conditions"]["ungated"]
    groups = summaries["h2"]["conditions"]["ungated"]["by_group"]
    assert groups["stale"] == {
        "answers": 2,
        "success": 0.5,
        "harm": 0.5,
        "trap_injected_answers": 2,
        "harm_trap_injected": 0.5,
        "useful_injected_answers": 2,
    }
    assert groups["cross_module"]["trap_injected_answers"] == 0
    assert groups["control"]["harm"] is None and groups["control"]["useful_injected_answers"] == 1
    assert list(summaries["h2"]["per_task"]) == [t.id for t in H2_TASKS]
    report = scoring.report_markdown(summaries)
    assert "## Suite `original`" in report and "## Suite `h2` — **exploratory" in report
    assert "### Per group" in report and "| S01 | H (stale) |" in report
    assert "success_N" not in report  # no N tasks in either suite here, so no empty comparison rows


def test_reused_recalls_must_come_from_the_same_suite() -> None:
    source = _record("base", ["ungated"], [])
    with pytest.raises(BenchmarkError, match="suite"):
        reused_recalls(source, ["ungated"], ["S01"], "h2")


def test_summary_refuses_a_condition_from_two_runs() -> None:
    a = _record("a", ["none"], [ans("H01", "none", 1, True)])
    b = _record("b", ["none"], [ans("H01", "none", 1, True)])
    with pytest.raises(ValueError, match="both"):
        scoring.summarize([a, b])
