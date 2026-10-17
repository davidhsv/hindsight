"""Adaptive memory in recall ranking (engine/adaptive/scoring.py), tested over ScoredResult objects.

Pure: no database, no model. The integration path (state load, trace, metrics through a real
recall) is tests/test_adaptive_recall_integration.py.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry, generate_latest

from hindsight_api.config import ADAPTIVE_MEMORY_MODES
from hindsight_api.engine.adaptive.models import AdaptiveMode, AdaptiveState, RetentionState
from hindsight_api.engine.adaptive.scoring import (
    AdaptiveRecall,
    CachedTokenCounter,
    apply_adaptive_factor,
    attach_adaptive,
    final_selection,
    reweighted,
)
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.memories.base import MemoriesExtension
from hindsight_api.engine.search.trace import (
    AdaptiveTraceInfo,
    QueryInfo,
    RerankedResult,
    SearchSummary,
    SearchTrace,
)
from hindsight_api.engine.search.tracer import SearchTracer
from hindsight_api.engine.search.types import ArmScores, MergedCandidate, RetrievalResult, ScoredResult

NOW = datetime(2026, 6, 1, tzinfo=UTC)

_STATIC_FIELDS = (
    "adaptive_memory_recall_reinforcement_enabled",
    "adaptive_memory_reinforcement_cooldown_hours",
    "adaptive_memory_reinforcement_gain",
    "adaptive_memory_source_weights",
    "adaptive_memory_type_stability",
    "adaptive_memory_source_trust",
    "adaptive_memory_conflict_policy",
    "adaptive_memory_archive_enabled",
    "adaptive_memory_archive_threshold",
    "adaptive_memory_archive_grace_days",
    "adaptive_memory_archive_recall_min_score",
    "adaptive_memory_archive_protect_importance",
    "adaptive_memory_prune_enabled",
    "adaptive_memory_prune_after_days",
)


def _static(**values) -> SimpleNamespace:
    return SimpleNamespace(**{**{name: None for name in _STATIC_FIELDS}, **values})


def _settings(mode: str = "rank", **bank) -> AdaptiveSettings:
    return AdaptiveSettings.resolve(
        {"adaptive_memory_mode": mode, **{f"adaptive_memory_{k}": v for k, v in bank.items()}},
        _static(),
        store_supports=True,
    )


def _result(
    memory_id: str,
    weight: float,
    *,
    age_days: float | None = 0.0,
    fact_type: str = "world",
    text: str | None = None,
    ce: float | None = None,
    sources: list[str] | None = None,
    semantic: float | None = None,
) -> ScoredResult:
    retrieval = RetrievalResult(
        id=memory_id,
        text=text if text is not None else f"fact {memory_id}",
        fact_type=fact_type,
        mentioned_at=None if age_days is None else NOW - timedelta(days=age_days),
        source_memory_ids=sources,
    )
    return ScoredResult(
        candidate=MergedCandidate(retrieval=retrieval, rrf_score=0.01, arm_scores=ArmScores(semantic=semantic)),
        cross_encoder_score=ce if ce is not None else weight,
        cross_encoder_score_normalized=ce if ce is not None else weight,
        combined_score=weight,
        weight=weight,
    )


def _row(memory_id: str, *, last: datetime, stability: float = 30.0, importance: float = 0.5) -> AdaptiveState:
    return AdaptiveState(
        retention=RetentionState(
            memory_id=memory_id,
            importance=importance,
            stability_days=stability,
            last_reinforced_at=last,
            reinforcement_count=2,
            access_count=0,
            forgetting_exempt=False,
            retention_state="active",
            last_recalled_at=None,
            below_threshold_since=None,
            archived_at=None,
        ),
        claim=None,
    )


def _ranked(results: list[ScoredResult]) -> list[str]:
    return [sr.id for sr in sorted(results, key=lambda sr: sr.weight, reverse=True)]


def _words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- ranking


def test_an_old_relevant_memory_stays_recoverable():
    old_precise = _result("old", 0.9, age_days=3650)
    recent_vague = _result("recent", 0.6, age_days=0)
    results = [old_precise, recent_vague]
    attach_adaptive(results, {}, _settings(), NOW)
    apply_adaptive_factor(results)
    assert _ranked(results) == ["old", "recent"]
    assert old_precise.retrievability == pytest.approx(0.0, abs=1e-12)
    assert old_precise.weight == pytest.approx(0.9 * 0.93)


def test_an_irrelevant_recent_memory_does_not_beat_a_precise_old_one():
    results = [_result("old", 0.8, age_days=400), _result("irrelevant", 0.2, age_days=0)]
    attach_adaptive(results, {}, _settings(), NOW)
    apply_adaptive_factor(results)
    assert _ranked(results) == ["old", "irrelevant"]


def test_a_weak_memory_is_downranked_on_a_weak_cue():
    # Near-tied relevance: the faded memory loses to the retrievable one.
    faded, fresh = _result("faded", 0.50, age_days=365), _result("fresh", 0.49, age_days=1)
    results = [faded, fresh]
    attach_adaptive(results, {}, _settings(), NOW)
    apply_adaptive_factor(results)
    assert _ranked(results) == ["fresh", "faded"]


def test_a_reinforced_row_beats_the_lazy_age_default():
    # Same mention age, same relevance; the stored reinforcement decides.
    results = [_result("lazy", 0.5, age_days=200), _result("reinforced", 0.5, age_days=200)]
    attach_adaptive(
        results, {"reinforced": _row("reinforced", last=NOW - timedelta(days=1), stability=400)}, _settings(), NOW
    )
    apply_adaptive_factor(results)
    assert _ranked(results) == ["reinforced", "lazy"]
    assert results[1].reinforcement_count == 2 and results[0].reinforcement_count == 0


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("floor", [0.0, 0.3, 1.0])
def test_the_factor_never_leaves_its_bound(alpha, floor):
    settings = _settings(score_alpha=alpha, score_floor=floor)
    results = [_result(str(i), 1.0, age_days=age) for i, age in enumerate((0, 1, 30, 365, 3650, None))]
    attach_adaptive(results, {}, settings, NOW)
    lower = 1 - alpha * (1 - floor)
    for sr in results:
        assert lower - 1e-12 <= sr.adaptive_factor <= 1.0 + 1e-12


def test_importance_lifts_the_floor():
    results = [_result("plain", 1.0, age_days=3650), _result("important", 1.0, age_days=3650)]
    states = {
        "plain": _row("plain", last=NOW - timedelta(days=3650), stability=1),
        "important": _row("important", last=NOW - timedelta(days=3650), stability=1, importance=1.0),
    }
    attach_adaptive(results, states, _settings(), NOW)
    assert results[1].retention_factor > results[0].retention_factor


def test_observe_leaves_weights_byte_identical_and_simulates_rank():
    def fresh() -> list[ScoredResult]:
        return [
            _result("a", 0.90, age_days=900),
            _result("b", 0.88, age_days=0),
            _result("c", 0.70, age_days=10),
            _result("d", 0.10, age_days=0),
        ]

    observed = fresh()
    weights_before = [sr.weight for sr in observed]
    observe = AdaptiveRecall(
        settings=_settings("observe", score_alpha=1.0), now=NOW, count_tokens=CachedTokenCounter(_words)
    )
    observe.states = {}
    assert observe.score(observed) is True
    assert [sr.weight for sr in observed] == weights_before  # exact, not approx
    observed.sort(key=lambda sr: sr.weight, reverse=True)
    actual = final_selection(
        observed,
        min_reranker=None,
        min_final=None,
        prefer_observations=False,
        rerank_limit=10,
        max_tokens=100,
        count_tokens=_words,
    )
    simulated = observe.finish(
        actual.ids, min_reranker=None, min_final=None, prefer_observations=False, rerank_limit=10, max_tokens=100
    )

    ranked = fresh()
    rank = AdaptiveRecall(settings=_settings("rank", score_alpha=1.0), now=NOW, count_tokens=CachedTokenCounter(_words))
    rank.states = {}
    assert rank.score(ranked) is True
    ranked.sort(key=lambda sr: sr.weight, reverse=True)
    rank_ids = final_selection(
        ranked,
        min_reranker=None,
        min_final=None,
        prefer_observations=False,
        rerank_limit=10,
        max_tokens=100,
        count_tokens=_words,
    ).ids
    counterfactual = rank.finish(
        rank_ids, min_reranker=None, min_final=None, prefer_observations=False, rerank_limit=10, max_tokens=100
    )

    assert actual.ids == ["a", "b", "c", "d"]
    assert simulated is not None and simulated.ids == rank_ids == ["b", "c", "a", "d"]
    assert simulated.changed is True
    # Rank's counterfactual is the factor-free order, i.e. what observe actually returned.
    assert counterfactual is not None and counterfactual.ids == actual.ids and counterfactual.changed is True


def test_a_failed_state_load_scores_nothing():
    results = [_result("a", 0.5)]
    adaptive = AdaptiveRecall(settings=_settings(), now=NOW, count_tokens=CachedTokenCounter(_words))
    assert adaptive.score(results) is False
    assert results[0].weight == 0.5 and results[0].adaptive_factor is None
    assert (
        adaptive.finish(
            ["a"], min_reranker=None, min_final=None, prefer_observations=False, rerank_limit=2, max_tokens=10
        )
        is None
    )


def test_reweighted_copies_share_retrieval_and_sort_stably():
    originals = [_result("x", 0.5), _result("y", 0.5), _result("z", 0.9)]
    copies = reweighted(originals, lambda sr: sr.weight)
    assert [sr.id for sr in copies] == ["z", "x", "y"]
    assert all(copy is not original for copy, original in zip(copies, sorted(originals, key=lambda s: -s.weight)))
    assert copies[1].retrieval is originals[0].retrieval
    copies[0].weight = 0.0
    assert originals[2].weight == 0.9


def test_the_token_counter_counts_each_text_once():
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return len(text)

    counter = CachedTokenCounter(counting)
    assert counter("abc") == 3 and counter("abc") == 3 and counter("de") == 2
    assert calls == ["abc", "de"]


# --------------------------------------------------------------------------- the recall tail


def test_final_selection_applies_the_min_scores_floors():
    results = [_result("a", 0.9, ce=0.2), _result("b", 0.6, ce=0.8), _result("c", 0.3, ce=0.9)]
    assert final_selection(
        results,
        min_reranker=None,
        min_final=0.5,
        prefer_observations=False,
        rerank_limit=10,
        max_tokens=99,
        count_tokens=_words,
    ).ids == ["a", "b"]
    assert final_selection(
        results,
        min_reranker=0.5,
        min_final=0.5,
        prefer_observations=False,
        rerank_limit=10,
        max_tokens=99,
        count_tokens=_words,
    ).ids == ["b"]


def test_final_selection_dedups_raw_facts_behind_an_observation_in_the_window():
    results = [
        _result("obs", 0.9, fact_type="observation", sources=["s1"]),
        _result("s1", 0.8),
        _result("s2", 0.7),
        _result("far-obs", 0.1, fact_type="observation", sources=["s2"]),
    ]
    common = dict(min_reranker=None, min_final=None, max_tokens=99, count_tokens=_words)
    assert final_selection(results, prefer_observations=True, rerank_limit=2, **common).ids == ["obs", "s2"]
    # Off, or with the far observation outside the window, nothing else is dropped.
    assert final_selection(results, prefer_observations=False, rerank_limit=2, **common).ids == ["obs", "s1"]
    assert final_selection(results, prefer_observations=True, rerank_limit=10, **common).ids == ["obs", "far-obs"]


def test_final_selection_truncates_then_spends_the_token_budget_in_rank_order():
    results = [
        _result("a", 0.9, text="one two three"),
        _result("big", 0.8, text="w " * 50),
        _result("b", 0.7, text="four five"),
        _result("tail", 0.6, text="six"),
    ]
    selection = final_selection(
        results,
        min_reranker=None,
        min_final=None,
        prefer_observations=False,
        rerank_limit=3,
        max_tokens=6,
        count_tokens=_words,
    )
    # The oversized fact skips itself only; "tail" was truncated away before the budget.
    assert selection.ids == ["a", "b"]
    assert selection.total_tokens == 5 and selection.truncated is True


def test_a_too_tight_budget_still_returns_the_top_fact():
    results = [_result("a", 0.9, text="a b c d"), _result("b", 0.8, text="e f g h")]
    assert final_selection(
        results,
        min_reranker=None,
        min_final=None,
        prefer_observations=False,
        rerank_limit=4,
        max_tokens=1,
        count_tokens=_words,
    ).ids == ["a"]


# --------------------------------------------------------------------------- settings


def test_the_modes_mirror_the_config_tuple_and_their_capabilities():
    assert tuple(m.value for m in AdaptiveMode) == ADAPTIVE_MEMORY_MODES
    assert [m.loads_state for m in AdaptiveMode] == [False, True, True, True]
    assert [m.applies_factor for m in AdaptiveMode] == [False, False, True, True]
    assert [m.lifecycle for m in AdaptiveMode] == [False, False, False, True]


@pytest.mark.parametrize("stored", [None, "", "bogus", 3, "RANKING"])
def test_a_missing_or_unknown_mode_is_off(stored):
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": stored}, _static(), store_supports=True)
    assert settings.mode is AdaptiveMode.OFF and settings.effective_mode is AdaptiveMode.OFF
    assert AdaptiveSettings.resolve({}, _static(), store_supports=True).effective_mode is AdaptiveMode.OFF


def test_a_store_without_adaptive_state_forces_off():
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": "lifecycle"}, _static(), store_supports=False)
    assert settings.mode is AdaptiveMode.LIFECYCLE
    assert settings.effective_mode is AdaptiveMode.OFF


def test_the_interface_default_store_does_not_hold_adaptive_state():
    assert MemoriesExtension.supports_adaptive_state_for(MagicMock(), "bank") is False


def test_stored_values_are_clamped_defensively():
    settings = AdaptiveSettings.resolve(
        {
            "adaptive_memory_mode": "Rank",
            "adaptive_memory_score_alpha": 7.0,
            "adaptive_memory_score_floor": -1.0,
            "adaptive_memory_base_stability_days": 50.0,
            "adaptive_memory_max_stability_days": 10.0,
            "adaptive_memory_default_view": "sideways",
            "adaptive_memory_reinforcement_enabled": "yes",
        },
        _static(adaptive_memory_reinforcement_gain=float("nan")),
        store_supports=True,
    )
    assert settings.effective_mode is AdaptiveMode.RANK
    assert settings.score_alpha == 1.0 and settings.score_floor == 0.0
    assert settings.max_stability_days == 50.0  # never below base
    assert settings.default_view == "current"
    assert settings.reinforcement_enabled is True  # a non-bool reads as the default
    assert settings.reinforcement_gain == 1.0


def test_none_maps_mean_the_defaults_and_partial_maps_merge():
    settings = AdaptiveSettings.resolve(
        {"adaptive_memory_mode": "observe"},
        _static(adaptive_memory_type_stability={"world": 3.0}),
        store_supports=True,
    )
    assert settings.type_stability == {"experience": 0.5, "world": 3.0, "observation": 2.0}
    assert settings.source_weights["explicit_confirmation"] == 1.0
    assert dict(settings.source_trust) == {}
    with pytest.raises(TypeError):
        settings.type_stability["world"] = 9.0  # read-only


# --------------------------------------------------------------------------- trace + serialization


def test_scored_result_dict_carries_adaptive_keys_only_when_scored():
    sr = _result("a", 0.8, semantic=0.7)
    plain = sr.to_dict()
    assert not {"retrievability", "adaptive_factor", "semantic_score", "adaptive_base_score"} & plain.keys()
    attach_adaptive([sr], {}, _settings(), NOW)
    scored = sr.to_dict()
    assert scored["adaptive_base_score"] == 0.8
    assert scored["semantic_score"] == 0.7
    assert {
        "retrievability",
        "stability_days",
        "importance",
        "reinforcement_count",
        "retention_factor",
    } <= scored.keys()
    assert {k: v for k, v in scored.items() if k in plain} == plain


def test_reranked_score_components_gain_adaptive_floats_only_when_present():
    tracer = SearchTracer("q", 10, 100)
    plain, scored = _result("p", 0.5), _result("s", 0.5, semantic=0.4)
    attach_adaptive([scored], {}, _settings(), NOW)
    tracer.add_reranked([plain.to_dict(), scored.to_dict()], [])
    plain_components, scored_components = (r.score_components for r in tracer.reranked)
    assert "adaptive_factor" not in plain_components and "semantic_score" not in plain_components
    assert scored_components["adaptive_factor"] == pytest.approx(scored.adaptive_factor)
    assert scored_components["semantic_score"] == 0.4
    assert scored_components["reinforcement_count"] == 0.0


def test_new_trace_fields_are_omitted_when_none():
    reranked = RerankedResult(node_id="n", text="t", rerank_score=1.0, rerank_rank=1, rrf_rank=1, rank_change=0)
    assert "lifecycle" not in reranked.model_dump()
    trace = SearchTrace(
        query=QueryInfo(query_text="q", query_embedding=[], timestamp=NOW, budget=1, max_tokens=1),
        summary=SearchSummary(
            total_nodes_visited=0,
            entry_points_found=0,
            budget_used=0,
            budget_remaining=1,
            total_duration_seconds=0.0,
            results_returned=0,
        ),
        final_results=[],
    )
    assert "adaptive" not in trace.to_dict()
    trace.adaptive = AdaptiveTraceInfo(mode="rank", alpha=0.1, floor=0.3, scored=True, rerank_changed=False)
    assert trace.to_dict()["adaptive"] == {
        "mode": "rank",
        "alpha": 0.1,
        "floor": 0.3,
        "scored": True,
        "rerank_changed": False,
    }
    assert "adaptive" in trace.model_dump_json()


# --------------------------------------------------------------------------- metrics


def test_adaptive_metrics_export_under_their_exact_prometheus_names():
    registry = CollectorRegistry()
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    from hindsight_api.metrics import MetricsCollector

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
            patch("hindsight_api.metrics.get_meter", return_value=provider.get_meter("test-adaptive")),
            patch("hindsight_api.config.get_config", return_value=config),
        ):
            collector = MetricsCollector()
        collector.record_adaptive_state_load_error(mode="rank", bank_id="b")
        collector.record_adaptive_recall(
            mode="observe",
            bank_id="b",
            retrievability=[0.07, 1.0],
            stability_days=[30.0, 4000.0],
            rerank_changed=None,
            top_k_changed=True,
        )
        collector.record_adaptive_recall(
            mode="rank",
            bank_id="b",
            retrievability=[0.5],
            stability_days=[2.0],
            rerank_changed=True,
            top_k_changed=None,
        )
        text = generate_latest(registry).decode()
    finally:
        provider.shutdown()

    for name in (
        "hindsight_adaptive_memory_state_load_errors_total",
        "hindsight_adaptive_memory_rerank_changes_total",
        "hindsight_adaptive_memory_top_k_changed_total",
        "hindsight_adaptive_memory_retrievability_bucket",
        "hindsight_adaptive_memory_stability_days_bucket",
    ):
        assert f"\n{name}{{" in text, name
    lines = text.splitlines()

    def sample(prefix: str) -> str:
        return next(line for line in lines if line.startswith(prefix))

    # The explicit buckets, not the SDK defaults: 0.07 lands in (0.05, 0.1], 4000 days past 3650.
    assert sample('hindsight_adaptive_memory_retrievability_bucket{le="0.05",mode="observe",').endswith(" 0.0")
    assert sample('hindsight_adaptive_memory_retrievability_bucket{le="0.1",mode="observe",').endswith(" 1.0")
    assert sample('hindsight_adaptive_memory_stability_days_bucket{le="3650.0",mode="observe",').endswith(" 1.0")
    assert sample('hindsight_adaptive_memory_top_k_changed_total{mode="observe",').endswith(" 1.0")
    # No bank_id label unless opted in.
    assert 'bank_id="b"' not in text
