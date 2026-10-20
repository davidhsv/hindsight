"""The context_pollution benchmark's admission tooling: admission variants, trace parsing, the
offline replay of the gate, the logistic fit and the dev selection rule. Server-free."""

from __future__ import annotations

import math
import random

import pytest

from hindsight_system_tests.context_bench import admission as adm
from hindsight_system_tests.context_bench.admission_dev import (
    ENGINE_DEFAULT_WEIGHTS,
    HoldoutRowError,
    dev_rows,
    fitted_weights,
    server_env_for,
)
from hindsight_system_tests.context_bench.report import T2_HOLDOUT_HEADING, insert_t2_section
from hindsight_system_tests.context_bench.runner import (
    AdmissionDecisionRow,
    AdmissionRow,
    BenchmarkError,
    QueryRow,
    ResultRow,
    RunRecord,
    _redacted,
    admission_updates,
    parse_admission,
    parse_embedding_rows,
    variant_body,
)
from hindsight_system_tests.context_bench.scenarios import HOLDOUT, QUERIES

WEIGHTS = {"bias": -1.0, "reranker": 4.0, "rank": 0.5, "scope_mismatch.repository": -3.0, "token_cost": -0.1}


def settings(**overrides: object) -> adm.GateSettings:
    base = dict(
        scorer="deterministic",
        mode="filter",
        weights=WEIGHTS,
        threshold_auto=0.6,
        threshold_explicit=0.35,
        reference_bonus=0.15,
        uncertain_margin=0.1,
        jev_weight=0.5,
        mmr_lambda=0.7,
        dup_similarity=0.92,
        max_memories=10,
        max_tokens=2000,
        hard_scope_keys=("repository",),
    )
    base.update(overrides)
    return adm.GateSettings(**base)  # type: ignore[arg-type]


def decision(
    label: str,
    reranker: float,
    *,
    rank: int = 0,
    status: str = "active",
    scope: dict[str, str] | None = None,
    tokens: int = 20,
    final: float = 0.5,
    weights: dict[str, float] = WEIGHTS,
    probability: float | None = None,
    scorer: str = "deterministic",
    jev_weight: float = 0.5,
) -> AdmissionDecisionRow:
    """A recorded candidate whose score parts are what a server with ``weights`` would report."""
    scope = scope or {}
    features = {"bias": 1.0, "reranker": reranker, "rank": 1.0 / (1.0 + rank), "token_cost": tokens / 100.0}
    for key, match in scope.items():
        if match != "unknown":
            features[f"scope_{match}.{key}"] = 1.0
    det = adm.deterministic_score(features, weights)
    relevance, score = det.task_relevance, det.admission_score
    if probability is not None:
        relevance = probability
        score = probability if scorer == "jev" else jev_weight * probability + (1 - jev_weight) * det.admission_score
    return AdmissionDecisionRow(
        label=label,
        memory_id=f"id-{label}",
        base_retrieval_score=final,
        scope_score=det.scope_score,
        task_relevance=relevance,
        expected_utility=det.expected_utility,
        truth_status=status,
        admission_score=score,
        admitted=False,
        reason="below_threshold",
        tokens=tokens,
        scope=scope,
        features=features,
    )


def row(*decisions: AdmissionDecisionRow, purpose: str = "auto_context", **kw: object) -> AdmissionRow:
    return AdmissionRow(purpose=purpose, decisions=list(decisions), **kw)  # type: ignore[arg-type]


def run_gate(r: AdmissionRow, s: adm.GateSettings, embeddings: dict[str, list[float]] | None = None) -> list:
    probabilities = [None] * len(r.decisions)
    scores = adm.candidate_scores(r, s, probabilities)
    return adm.replay(
        r, s, scores=scores, similarity=adm.similarity_function([d.label for d in r.decisions], embeddings or {})
    )


# --------------------------------------------------------------------------- variants and parsing


def test_admission_variant_sets_the_three_fields_and_asks_for_a_trace() -> None:
    assert admission_updates("adm=filter+ta=0.55+te=0.40") == {
        "context_admission_mode": "filter",
        "context_admission_threshold_auto": 0.55,
        "context_admission_threshold_explicit": 0.40,
    }
    assert admission_updates("plain") is None and admission_updates("tags+final@0.30") is None
    task = next(query for query in QUERIES if query.project == "c")
    body = variant_body("adm=strict+ta=0.6+te=0.35", task, max_tokens=4096, send_context=True)
    assert body["trace"] is True and body["purpose"] == task.purpose and "min_scores" not in body
    assert "trace" not in variant_body("plain", task, max_tokens=4096, send_context=True)
    assert variant_body("plain", task, max_tokens=4096, send_context=True, trace=True)["trace"] is True


@pytest.mark.parametrize(
    "variant",
    ["adm=filter+ta=0.5", "adm=bogus+ta=0.5+te=0.5", "adm=filter+ta=1.5+te=0.5", "adm=filter+tx=0.5+te=0.5+ta=0.5"],
)
def test_admission_variant_errors(variant: str) -> None:
    with pytest.raises(ValueError):
        admission_updates(variant)


def test_parse_admission_maps_memory_ids_to_labels() -> None:
    item = decision("a.fact", 0.9).model_dump(exclude={"label"})
    other = {**item, "memory_id": "unknown-id"}
    payload = {
        "results": [],
        "context_admission": {"abstained": False, "candidates": 2, "admitted": 1},
        "trace": {
            "context_admission": {
                "mode": "strict",
                "scorer": "hybrid",
                "purpose": "auto_context",
                "threshold": 0.45,
                "historical": False,
                "reference_signal": "last_time",
                "candidates": 2,
                "admitted": 1,
                "abstained": False,
                "reasons": {"below_threshold": 1},
                "cross_scope_candidates": 0,
                "decisions": [item, other],
                "relevance_source": "reranker",
            }
        },
    }
    parsed = parse_admission(payload, {"id-a.fact": "a.fact"})
    assert [d.label for d in parsed.decisions] == ["a.fact", "?unknown-id"]
    assert parsed.summary == {"abstained": False, "candidates": 2, "admitted": 1}
    assert parsed.reference_signal == "last_time" and parsed.historical is False and parsed.threshold == 0.45
    assert list(parsed.decisions[0].features) == ["bias", "reranker", "rank", "token_cost"]
    empty = parse_admission({"results": [], "context_admission": {"candidates": 0}}, {})
    assert empty.decisions == [] and empty.summary == {"candidates": 0}
    assert parsed.phase_ms is None


def test_parse_admission_records_the_gate_phase_duration() -> None:
    payload = {
        "results": [],
        "trace": {
            "context_admission": {"mode": "filter", "candidates": 0, "decisions": []},
            "summary": {
                "phase_metrics": [
                    {"phase_name": "retrieval", "duration_seconds": 1.5},
                    {"phase_name": "context_admission", "duration_seconds": 0.0125},
                    # A retried search adds a second gate pass; both count.
                    {"phase_name": "context_admission", "duration_seconds": 0.002},
                    {"phase_name": "context_admission", "duration_seconds": "n/a"},
                ]
            },
        },
    }
    assert parse_admission(payload, {}).phase_ms == pytest.approx(14.5)
    del payload["trace"]["summary"]
    assert parse_admission(payload, {}).phase_ms is None


def test_embedding_rows_and_redaction() -> None:
    parsed = parse_embedding_rows("id1\t[0.5,-1.25]\nid2\t[1e-3,2]\n", {"id1": "a", "id2": "b"})
    assert parsed == {"a": [0.5, -1.25], "b": [0.001, 2.0]}
    with pytest.raises(BenchmarkError):
        parse_embedding_rows("id1\t[0.5]\n", {"id1": "a", "id2": "b"})
    with pytest.raises(BenchmarkError):
        parse_embedding_rows("zz\t[0.5]\n", {"zz2": "a"})
    redacted = _redacted({"X_DECISION_MODEL_API_KEY": "s", "X_HARD_SCOPE_KEYS": "repository", "X_TOKEN": "t"})
    assert redacted == {
        "X_DECISION_MODEL_API_KEY": "<redacted>",
        "X_HARD_SCOPE_KEYS": "repository",
        "X_TOKEN": "<redacted>",
    }


def test_server_env_states_every_static_setting() -> None:
    env = server_env_for("hybrid", ENGINE_DEFAULT_WEIGHTS)
    assert env["HINDSIGHT_API_CONTEXT_ADMISSION_SCORER"] == "hybrid"
    assert env["HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER"] == "typesafe"
    assert env["HINDSIGHT_API_ADAPTIVE_MEMORY_MODE"] == "lifecycle"
    assert "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER" not in server_env_for("deterministic", {})
    assert not any("API_KEY" in key for key in env)


# --------------------------------------------------------------------------- scores


def test_deterministic_score_parts() -> None:
    features = {"bias": 1.0, "reranker": 0.5, "scope_match.module": 1.0, "token_cost": 0.2, "unweighted": 3.0}
    weights = {"bias": -1.0, "reranker": 4.0, "scope_match.module": 0.6, "token_cost": -0.1}
    score = adm.deterministic_score(features, weights)
    assert score.task_relevance == pytest.approx(adm.sigmoid(1.0))
    assert score.scope_score == pytest.approx(adm.sigmoid(0.6))
    assert score.expected_utility == pytest.approx(adm.sigmoid(-0.02))
    assert score.admission_score == pytest.approx(adm.sigmoid(1.0 + 0.6 - 0.02))
    assert adm.sigmoid(-800.0) == 0.0 and adm.sigmoid(800.0) == 1.0


def test_decision_model_probabilities_recovered_and_checked() -> None:
    source = settings(scorer="hybrid")
    answered = decision("a", 0.3, probability=0.9, scorer="hybrid")
    unanswered = decision("b", 0.3)
    assert adm.decision_model_probabilities(row(answered, unanswered), source) == [0.9, None]
    jev = settings(scorer="jev")
    assert adm.decision_model_probabilities(row(decision("a", 0.3, probability=0.2, scorer="jev")), jev) == [0.2]
    # A request that failed falls back to the deterministic score for every candidate.
    assert adm.decision_model_probabilities(row(unanswered, jev_error="TimeoutError"), source) == [None]
    # Recorded numbers the replay cannot reproduce are an error, not a silently wrong sweep.
    with pytest.raises(adm.ReplayError):
        adm.decision_model_probabilities(row(decision("a", 0.3, weights={**WEIGHTS, "bias": -2.0})), source)
    tampered = answered.model_copy(update={"admission_score": answered.admission_score + 0.01})
    with pytest.raises(adm.ReplayError):
        adm.decision_model_probabilities(row(tampered), source)


def test_candidate_scores_per_scorer() -> None:
    d = decision("a", 0.3)
    det = adm.deterministic_score(d.features, WEIGHTS).admission_score
    r = row(d, d.model_copy(update={"label": "b"}))
    assert adm.candidate_scores(r, settings(), [None, None]) == [det, det]
    assert adm.candidate_scores(r, settings(scorer="jev"), [0.8, None]) == [0.8, det]
    assert adm.candidate_scores(r, settings(scorer="hybrid", jev_weight=0.25), [0.8, None]) == [
        pytest.approx(0.25 * 0.8 + 0.75 * det),
        det,
    ]


# --------------------------------------------------------------------------- the gate


def test_currentness_threshold_purpose_and_margin() -> None:
    s = settings()
    strong, weak = 0.75, 0.0  # scores 0.92 and 0.37 with WEIGHTS
    r = row(
        decision("ok", strong),
        decision("weak", weak),
        decision("old", strong, status="superseded"),
        decision("wrong", strong, status="refuted"),
        decision("unsure", strong, status="uncertain"),
    )
    reasons = {d.label: d.reason for d in run_gate(r, s)}
    assert reasons == {
        "ok": "admitted",
        "weak": "below_threshold",
        "old": "superseded",
        "wrong": "refuted",
        "unsure": "admitted",
    }
    unsure = adm.deterministic_score(r.decisions[4].features, WEIGHTS).admission_score
    tight = settings(threshold_auto=unsure - 0.05)
    assert {d.label: d.reason for d in run_gate(r, tight)}["unsure"] == "below_threshold"
    historical = row(*r.decisions, historical=True, purpose="historical_search")
    reasons = {d.label: d.reason for d in run_gate(historical, s)}
    assert reasons["old"] == "admitted" and reasons["wrong"] == "refuted"
    # explicit_search / historical_search read the explicit threshold; a reference phrase lowers the auto one.
    weak_score = adm.deterministic_score(r.decisions[1].features, WEIGHTS).admission_score
    assert s.threshold("explicit_search", reference_signal=True) == 0.35
    assert s.threshold("auto_context", reference_signal=True) == pytest.approx(0.45)
    assert settings(threshold_auto=0.1).threshold("auto_context", reference_signal=True) == 0.0
    lowered = settings(threshold_auto=weak_score + 0.1)
    assert {d.label: d.reason for d in run_gate(row(*r.decisions, reference_signal="last_time"), lowered)}[
        "weak"
    ] == "admitted"


def test_hard_scope_only_in_strict_mode() -> None:
    other = decision("other", 0.95, scope={"repository": "mismatch"})
    unknown = decision("mine", 0.95, scope={"repository": "unknown"})
    filtered = {d.label: d.reason for d in run_gate(row(other, unknown), settings())}
    strict = {d.label: d.reason for d in run_gate(row(other, unknown), settings(mode="strict"))}
    # The mismatch weight alone (−3) already rejects it in filter; strict names the reason.
    assert filtered == {"other": "below_threshold", "mine": "admitted"}
    assert strict == {"other": "wrong_scope", "mine": "admitted"}
    lenient = settings(weights={**WEIGHTS, "scope_mismatch.repository": 0.0})
    assert {d.label: d.reason for d in run_gate(row(other), lenient)}["other"] == "admitted"


def test_strict_suppresses_near_duplicates_and_keeps_order() -> None:
    a = decision("a", 0.95, rank=1, final=0.9)
    twin = decision("twin", 0.94, rank=0, final=0.95)
    b = decision("b", 0.9, rank=2, final=0.8)
    embeddings = {"a": [1.0, 0.0], "twin": [0.999, 0.01], "b": [0.0, 1.0]}
    decisions = run_gate(row(twin, a, b), settings(mode="strict"), embeddings)
    by_label = {d.label: d for d in decisions}
    # twin scores higher (rank 0) so it is kept; a is its near-duplicate.
    assert by_label["twin"].admitted and by_label["twin"].duplicates == 1
    assert by_label["a"].reason == "redundant" and by_label["b"].admitted
    assert [r.label for r in adm.admitted_results(decisions)] == ["twin", "b"]
    # Without embeddings nothing is a duplicate; filter mode never suppresses.
    assert all(d.admitted for d in run_gate(row(twin, a, b), settings(mode="strict")))
    assert all(d.admitted for d in run_gate(row(twin, a, b), settings(), embeddings))


def test_budget_count_binds_by_value() -> None:
    weights = {"bias": 0.0, "reranker": 1.0}
    big = decision("big", 0.0, tokens=400, weights=weights)
    smalls = [decision(f"s{i}", 0.0, tokens=30, weights=weights) for i in range(10)]
    big = big.model_copy(update={"features": {"bias": 1.0, "reranker": 4.6}})
    for i, s in enumerate(smalls):
        smalls[i] = s.model_copy(update={"features": {"bias": 1.0, "reranker": -0.4}})
    s = settings(mode="strict", weights=weights, threshold_auto=0.3)
    decisions = run_gate(row(big, *smalls), s)
    kept = [d.label for d in decisions if d.admitted]
    # Eleven fit the tokens, so the count binds: the most valuable memory stays.
    assert "big" in kept and len(kept) == 10
    assert sum(1 for d in decisions if d.reason == "budget") == 1
    tokens = settings(mode="strict", weights=weights, threshold_auto=0.3, max_tokens=300)
    decisions = run_gate(row(big, *smalls), tokens)
    assert [d.label for d in decisions if d.admitted] == [f"s{i}" for i in range(10)]


def test_twin_readmitted_when_its_canonical_is_dropped_by_the_budget() -> None:
    weights = {"bias": 0.0, "reranker": 1.0}
    long = decision("long", 0.0, tokens=150, weights=weights).model_copy(
        update={"features": {"bias": 1.0, "reranker": 3.0}}
    )
    short = decision("short", 0.0, tokens=20, weights=weights).model_copy(
        update={"features": {"bias": 1.0, "reranker": 2.9}}
    )
    other = decision("other", 0.0, tokens=50, weights=weights).model_copy(
        update={"features": {"bias": 1.0, "reranker": 2.0}}
    )
    embeddings = {"long": [1.0, 0.0], "short": [0.99, 0.05], "other": [0.0, 1.0]}
    s = settings(mode="strict", weights=weights, threshold_auto=0.5, max_tokens=100)
    by_label = {d.label: d for d in run_gate(row(long, short, other), s, embeddings)}
    assert by_label["long"].reason == "budget" and by_label["long"].duplicates == 0
    assert by_label["short"].admitted and by_label["other"].admitted
    no_room = settings(mode="strict", weights=weights, threshold_auto=0.5, max_tokens=60)
    by_label = {d.label: d for d in run_gate(row(long, short, other), no_room, embeddings)}
    assert not by_label["short"].admitted and by_label["other"].admitted


def test_similarity_function_edge_cases() -> None:
    sim = adm.similarity_function(
        ["a", "b", "c", "d", "z"], {"a": [3.0, 4.0], "b": [4.0, 3.0], "c": [0.0, 0.0], "d": [1.0]}
    )
    assert sim("a", "b") == pytest.approx(24 / 25) and sim("b", "a") == sim("a", "b")
    assert sim("a", "c") == 0.0 and sim("a", "d") == 0.0 and sim("a", "z") == 0.0
    assert sim("a", "a") == pytest.approx(1.0)


def test_verify_replay_counts_exact_and_reports_mismatches() -> None:
    s = settings()
    recorded = row(decision("ok", 0.75), decision("weak", 0.0))
    decisions = run_gate(recorded, s)
    live = row(
        *(
            d.model_copy(update={"admitted": r.admitted, "reason": r.reason})
            for d, r in zip(recorded.decisions, decisions)
        )
    )
    check = adm.ReplayCheck()
    adm.verify_replay([ResultRow(label="ok", server_rank=1, final=0.5)], live, decisions, check, query_id="q1")
    assert check.all_exact and check.reasons == {"admitted": 1, "below_threshold": 1}
    adm.verify_replay([], live, decisions, check, query_id="q2")
    assert check.exact == 1 and check.mismatches[0]["query"] == "q2" and not check.all_exact


# --------------------------------------------------------------------------- the fit


def test_logistic_fit_recovers_a_known_model() -> None:
    rng = random.Random(7)
    truth = {"bias": -1.0, "x": 2.0, "y": -1.5}
    samples = []
    for _ in range(4000):
        x, y = rng.uniform(-2, 2), rng.uniform(0, 3)
        p = adm.sigmoid(truth["bias"] + truth["x"] * x + truth["y"] * y)
        samples.append(({"bias": 1.0, "x": x, "y": y, "const": 1.0, "never": 0.0}, int(rng.random() < p)))
    fit = adm.fit_logistic(samples, l2=1e-6)
    assert fit.fitted == ("x", "y") and fit.constant == ("const",) and fit.absent == ("never",)
    assert fit.weights["const"] == 0.0 and "never" not in fit.weights
    for key, value in truth.items():
        assert fit.weights[key] == pytest.approx(value, abs=0.2)
    # At the optimum the gradient of the penalized loss vanishes (checked on the raw scale for
    # the bias, which carries no penalty).
    grad_bias = sum(
        adm.sigmoid(fit.weights["bias"] + fit.weights["x"] * f["x"] + fit.weights["y"] * f["y"]) - label
        for f, label in samples
    )
    assert abs(grad_bias) < 1e-6
    shrunk = adm.fit_logistic(samples, l2=1000.0)
    assert abs(shrunk.weights["x"]) < abs(fit.weights["x"])


def test_logistic_fit_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        adm.fit_logistic([({"x": 1.0}, 1), ({"x": 2.0}, 1)])
    with pytest.raises(ValueError):
        adm.fit_logistic([])


def test_fitted_weights_complete_the_map() -> None:
    fit = adm.LogisticFit(
        weights={"bias": -2.0, "reranker": 5.0, "scope_match.branch": 0.0},
        fitted=("reranker",),
        constant=("scope_match.branch",),
        absent=("uncertain",),
        l2=1.0,
        iterations=3,
        samples=10,
        positives=2,
        log_loss=0.3,
    )
    weights = fitted_weights(fit)
    assert set(weights) == set(ENGINE_DEFAULT_WEIGHTS)
    assert weights["reranker"] == 5.0 and weights["scope_match.branch"] == 0.0
    assert weights["uncertain"] == ENGINE_DEFAULT_WEIGHTS["uncertain"]


# --------------------------------------------------------------------------- selection


def result(
    auto: float, explicit: float, recall: float, pollution: float | None, abstention: float
) -> adm.SettingResult:
    return adm.SettingResult(
        "w",
        auto,
        explicit,
        {"useful_memory_recall": recall, "context_pollution_rate": pollution, "abstention_accuracy": abstention},
    )


def test_selection_rule() -> None:
    feasible = [
        result(0.5, 0.5, 0.90, 0.30, 0.5),
        result(0.6, 0.5, 0.86, 0.20, 0.5),
        result(0.7, 0.5, 0.86, 0.20, 0.7),
        result(0.9, 0.9, 0.40, 0.00, 1.0),
    ]
    # Lowest pollution among recall >= 0.85, tie broken by abstention accuracy.
    assert adm.select(feasible).threshold_auto == 0.7
    tie = [result(0.6, 0.5, 0.86, 0.20, 0.7), result(0.6, 0.6, 0.86, 0.20, 0.7)]
    assert adm.select(tie).threshold_explicit == 0.6
    infeasible = [result(0.5, 0.5, 0.80, 0.50, 0.0), result(0.6, 0.6, 0.70, 0.30, 0.0), result(0.9, 0.9, 0.0, None, 1)]
    assert adm.select(infeasible).threshold_auto == 0.6
    with pytest.raises(ValueError):
        adm.select([])


def test_threshold_grid() -> None:
    assert adm.THRESHOLD_GRID[0] == 0.3 and adm.THRESHOLD_GRID[-1] == 0.9 and len(adm.THRESHOLD_GRID) == 13
    assert all(math.isclose(b - a, 0.05) for a, b in zip(adm.THRESHOLD_GRID, adm.THRESHOLD_GRID[1:]))


# --------------------------------------------------------------------------- holdout guard and report


def _record(*query_ids: str) -> RunRecord:
    return RunRecord(
        label="r",
        api_dir="x",
        api_sha=None,
        api_dirty=None,
        harness_sha=None,
        server_env={},
        variants=["v"],
        send_context=True,
        max_tokens=4096,
        database_url="pg0://x",
        database_schema="s",
        banks={},
        bank_sizes={},
        started_at="now",
        rows=[
            QueryRow(
                variant="v",
                query_id=qid,
                split="holdout" if qid in HOLDOUT else "dev",
                latency_ms=1.0,
                status=200,
                results=[],
                admission=AdmissionRow(),
            )
            for qid in query_ids
        ],
    )


def test_dev_tooling_refuses_holdout_rows() -> None:
    dev_id = next(q.id for q in QUERIES if q.id not in HOLDOUT)
    held_id = next(iter(sorted(HOLDOUT)))
    assert list(dev_rows(_record(dev_id), "v")) == [dev_id]
    with pytest.raises(HoldoutRowError):
        dev_rows(_record(dev_id, held_id), "v")
    with pytest.raises(ValueError):
        dev_rows(_record(dev_id), "other")


def test_insert_t2_section(tmp_path) -> None:
    report = tmp_path / "REPORT.md"
    report.write_text("# r\n\n## Determinism\n\n- x\n\n## Arms not run\n\n- y\n")
    insert_t2_section(report, "## T2 dev\n\nfirst\n")
    assert report.read_text() == "# r\n\n## Determinism\n\n- x\n\n## T2 dev\n\nfirst\n\n## Arms not run\n\n- y\n"
    insert_t2_section(report, "## T2 dev\n\nsecond")
    assert report.read_text() == "# r\n\n## Determinism\n\n- x\n\n## T2 dev\n\nsecond\n\n## Arms not run\n\n- y\n"
    # A holdout section goes after the dev one; regenerating either keeps the other.
    insert_t2_section(report, "## T2 holdout\n\nh1\n", heading=T2_HOLDOUT_HEADING)
    both = "# r\n\n## Determinism\n\n- x\n\n## T2 dev\n\nsecond\n\n## T2 holdout\n\nh1\n\n## Arms not run\n\n- y\n"
    assert report.read_text() == both
    insert_t2_section(report, "## T2 dev\n\nthird\n")
    assert report.read_text() == both.replace("second", "third")
    insert_t2_section(report, "## T2 holdout\n\nh2", heading=T2_HOLDOUT_HEADING)
    assert report.read_text() == both.replace("second", "third").replace("h1", "h2")


def test_balanced_prior_forms_move_the_intercept_not_the_order() -> None:
    rng = random.Random(3)
    samples = []
    for _ in range(3000):
        x = rng.uniform(-3, 3)
        samples.append(({"bias": 1.0, "x": x}, int(rng.random() < adm.sigmoid(-4.0 + 1.5 * x))))
    plain = adm.fit_logistic(samples, l2=1e-6)
    balanced = adm.fit_logistic(samples, l2=1e-6, class_balanced=True)
    positives = sum(label for _, label in samples)
    assert balanced.class_balanced and not plain.class_balanced
    # A rare positive class: balancing raises the intercept by about log(negatives / positives).
    shift = math.log((len(samples) - positives) / positives)
    assert balanced.weights["bias"] > plain.weights["bias"] + 0.5 * shift
    shifted = plain.prior_shifted()
    assert shifted["bias"] == pytest.approx(plain.weights["bias"] + shift)
    assert shifted["x"] == plain.weights["x"]
    # Under balanced weights the mean predicted probability of each class straddles 0.5.
    mean_pos = sum(adm.sigmoid(balanced.weights["bias"] + balanced.weights["x"] * f["x"]) for f, y in samples if y)
    assert mean_pos / positives > 0.5


def test_settings_of_needs_a_fully_stated_run() -> None:
    from hindsight_system_tests.context_bench.admission_dev import settings_of

    record = _record()
    with pytest.raises(ValueError):
        settings_of(record, "v")  # nothing stated
    record.server_env = server_env_for("hybrid", ENGINE_DEFAULT_WEIGHTS)
    with pytest.raises(ValueError):
        settings_of(record, "v")  # no resolved bank config for the variant
    record.admission_settings["v"] = {
        "context_admission_mode": "strict",
        "context_admission_threshold_auto": 0.6,
        "context_admission_threshold_explicit": 0.35,
        "context_admission_reference_bonus": 0.15,
        "context_admission_mmr_lambda": 0.7,
        "context_admission_dup_similarity": 0.92,
        "context_admission_max_memories": 10,
        "context_admission_max_tokens": 2000,
    }
    s = settings_of(record, "v")
    assert (s.scorer, s.mode, s.hard_scope_keys, s.jev_weight) == ("hybrid", "strict", ("repository",), 0.5)
    record.server_env = server_env_for("hybrid", {"bias": 1.0})
    with pytest.raises(ValueError):
        settings_of(record, "v")  # a partial weight map would leave weights to the server's defaults
