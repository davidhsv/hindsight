"""The context_pollution benchmark's own arithmetic: metric definitions, the scenario's
integrity, the holdout split, request variants and tie canonicalisation. Server-free."""

from __future__ import annotations

import pytest

from hindsight_system_tests.context_bench import metrics
from hindsight_system_tests.context_bench.metrics import Admitted
from hindsight_system_tests.context_bench.model import Memory, Query
from hindsight_system_tests.context_bench.report import Config, admitted_labels, canonical
from hindsight_system_tests.context_bench.runner import QueryRow, ResultRow, variant_body
from hindsight_system_tests.context_bench.scenarios import HOLDOUT, MEMORIES, QUERIES, holdout_ids, validate

MEMS = {
    "a.fact": Memory("a.fact", "durable_fact", "Payments fact", "a"),
    "a.old": Memory("a.old", "durable_fact", "Payments old fact", "a", status="superseded"),
    "a.wrong": Memory("a.wrong", "speculation", "Payments guess", "a", status="refuted"),
    "a.joke": Memory("a.joke", "humor", "Payments joke", "a"),
    "b.fact": Memory("b.fact", "durable_fact", "FlowKeeper fact", "b"),
    "g.pref": Memory("g.pref", "preference", "User preference", None),
}
TASK = Query(
    "q1", "cross_project", "task", "a", relevant=("a.fact",), acceptable=("g.pref",), hard_negatives=("b.fact",)
)
EMPTY = Query("q2", "none_relevant", "nothing", "a")
HISTORY = Query("q3", "exact_historical_cue", "what was it", "a", relevant=("a.old",), purpose="historical_search")


def rows(*pairs: tuple[Query, tuple[str, ...]]) -> list[Admitted]:
    return [Admitted(query, labels) for query, labels in pairs]


def test_precision_excludes_acceptable_and_pools_queries() -> None:
    data = rows((TASK, ("a.fact", "g.pref", "b.fact")), (EMPTY, ("a.joke",)))
    # relevant 1; irrelevant b.fact + a.joke; g.pref is acceptable and on neither side.
    assert metrics.context_precision(data) == pytest.approx(1 / 3)


def test_precision_undefined_when_nothing_admitted() -> None:
    assert metrics.context_precision(rows((TASK, ()), (EMPTY, ()))) is None


def test_recall_ignores_abstention_queries() -> None:
    data = rows((TASK, ("b.fact",)), (EMPTY, ("a.joke",)), (HISTORY, ("a.old",)))
    assert metrics.useful_memory_recall(data) == pytest.approx(1 / 2)
    assert metrics.false_negative_admission_rate(data) == pytest.approx(1 / 2)


def test_pollution_counts_acceptable_in_the_total() -> None:
    data = rows((TASK, ("a.fact", "g.pref", "b.fact", "a.joke")))
    assert metrics.context_pollution_rate(data) == pytest.approx(2 / 4)
    assert metrics.polluted_queries_rate(data) == 1.0
    assert metrics.polluted_queries_rate(rows((TASK, ("a.fact", "g.pref")))) == 0.0


def test_abstention_accuracy_and_false_abstention() -> None:
    data = rows((EMPTY, ()), (EMPTY, ("a.joke",)), (TASK, ()), (TASK, ("a.fact",)))
    assert metrics.abstention_accuracy(data) == pytest.approx(1 / 2)
    assert metrics.false_abstention_rate(data) == pytest.approx(1 / 2)
    assert metrics.abstention_accuracy(rows((TASK, ()))) is None


def test_false_positive_rate_uses_every_wrong_memory_in_the_bank() -> None:
    data = rows((TASK, ("a.fact", "b.fact")), (EMPTY, ("a.joke", "a.old")))
    # TASK can wrongly admit 10 - 1 relevant - 1 acceptable = 8; EMPTY can wrongly admit 10.
    assert metrics.false_positive_admission_rate(data, {"main": 10}) == pytest.approx(3 / 18)


def test_hard_negative_admission_rate() -> None:
    assert metrics.hard_negative_admission_rate(rows((TASK, ("b.fact",)))) == 1.0
    assert metrics.hard_negative_admission_rate(rows((TASK, ("a.joke",)))) == 0.0
    assert metrics.hard_negative_admission_rate(rows((EMPTY, ("a.joke",)))) is None


def test_cross_scope_leakage_ignores_unscoped_memories_and_respects_k() -> None:
    data = rows((TASK, ("a.fact", "g.pref", "b.fact")))
    assert metrics.cross_scope_leakage(data, MEMS) == pytest.approx(1 / 3)
    assert metrics.cross_scope_leakage(data, MEMS, k=2) == 0.0
    unscoped = Query("q4", "humor_casual", "music?", None)
    assert metrics.cross_scope_leakage(rows((unscoped, ("b.fact",))), MEMS) is None


def test_stale_and_refuted_rates_skip_historical_queries() -> None:
    data = rows((TASK, ("a.old", "a.wrong", "a.fact", "a.joke")), (HISTORY, ("a.old", "a.old")))
    assert metrics.stale_admission_rate(data, MEMS) == pytest.approx(1 / 4)
    assert metrics.refuted_admission_rate(data, MEMS) == pytest.approx(1 / 4)


def test_token_overhead_is_a_mean_per_query() -> None:
    data = rows((TASK, ("a.fact", "b.fact")), (EMPTY, ()))
    assert metrics.token_overhead(data, lambda label: 5) == pytest.approx(5.0)
    assert metrics.approx_tokens("statement_timeout = '5s'.") == 6


def test_f_beta_and_percentile() -> None:
    assert metrics.f_beta(None, 0.5) is None
    assert metrics.f_beta(0.0, 0.0) == 0.0
    assert metrics.f_beta(1.0, 1.0) == pytest.approx(1.0)
    assert metrics.f_beta(0.5, 1.0) == pytest.approx(1.25 * 0.5 / (0.25 * 0.5 + 1.0))
    assert metrics.percentile([3.0, 1.0, 2.0, 4.0], 50) == 2.0
    assert metrics.percentile([3.0, 1.0, 2.0, 4.0], 95) == 4.0
    assert metrics.percentile([], 50) is None


def test_aggregate_reports_every_metric() -> None:
    out = metrics.aggregate(rows((TASK, ("a.fact",))), memories=MEMS, bank_sizes={"main": 6}, tokens_of=len)
    assert set(metrics.METRIC_NAMES) <= set(out)
    assert out["context_precision"] == 1.0 and out["n"] == 1.0


def test_scenario_is_consistent_and_meets_the_spec_sizes() -> None:
    assert validate() == []
    assert sum(1 for memory in MEMORIES if memory.bank == "main") >= 300
    assert len(QUERIES) >= 120
    categories = {query.category for query in QUERIES}
    assert {"all_candidates_irrelevant", "coding_session", "none_relevant"} <= categories
    # Ground truth never reaches the server: document ids and metadata carry no label.
    for memory in MEMORIES:
        assert memory.label not in memory.document_id
        assert memory.category not in memory.metadata.values()


def test_holdout_is_deterministic_and_stratified() -> None:
    assert holdout_ids() == HOLDOUT
    for category in {query.category for query in QUERIES}:
        ids = [query.id for query in QUERIES if query.category == category]
        held = [qid for qid in ids if qid in HOLDOUT]
        assert len(held) == (len(ids) + 2) // 3


def test_variant_bodies() -> None:
    task = next(query for query in QUERIES if query.project == "c")
    body = variant_body("tags+final@0.40", task, max_tokens=4096, send_context=False)
    assert body["tags"] == ["project:c"] and body["tags_match"] == "any"
    assert body["min_scores"] == {"final": 0.4}
    assert "context" not in body and "purpose" not in body
    assert variant_body("plain#2", task, max_tokens=4096, send_context=False) == variant_body(
        "plain", task, max_tokens=4096, send_context=False
    )
    with_context = variant_body("plain", task, max_tokens=4096, send_context=True)
    assert with_context["context"]["repository"] == "acme/audit-api" and with_context["purpose"] == task.purpose
    unscoped = next(query for query in QUERIES if query.project is None)
    assert "tags" not in variant_body("tags", unscoped, max_tokens=4096, send_context=False)
    with pytest.raises(ValueError):
        variant_body("bogus@1", task, max_tokens=4096, send_context=False)


def _row(*results: tuple[str, float, float | None]) -> QueryRow:
    return QueryRow(
        variant="plain",
        query_id="q1",
        split="dev",
        latency_ms=1.0,
        status=200,
        results=[
            ResultRow(label=label, server_rank=i, final=final, semantic=semantic)
            for i, (label, final, semantic) in enumerate(results, start=1)
        ],
    )


def test_ties_are_ordered_by_label_and_gates_apply_before_the_cap() -> None:
    row = _row(("z", 0.9, 0.8), ("b", 0.5, 0.3), ("a", 0.5 + 1e-9, 0.9), ("c", 0.1, None))
    assert [result.label for result in canonical(row.results)] == ["z", "a", "b", "c"]
    ungated = Config("r", "r:plain", "plain", None, None, 2)
    assert admitted_labels(row, ungated) == ("z", "a")
    gated = Config("r", "r:plain+client_semantic", "plain", 0.5, "semantic", 10)
    # A missing semantic score never clears the floor.
    assert admitted_labels(row, gated) == ("z", "a")
