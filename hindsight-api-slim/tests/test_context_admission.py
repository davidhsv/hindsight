"""Context admission, pure: features, the scorers, the gate (currentness, hard scope, threshold,
abstention), redundancy suppression, the budget, the reference-signal detector and the observe
replay. No database and no model: the decision model is a fake that records its requests."""

import math
from dataclasses import replace

import pytest

from hindsight_api.config import get_config
from hindsight_api.engine.adaptive.producers import recall_reinforcement_allowed
from hindsight_api.engine.adaptive.scoring import CachedTokenCounter
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.context_admission import (
    JEV_QUESTION,
    AdmissionMode,
    ContextAdmissionRecall,
    ContextAdmissionSettings,
    Purpose,
    Reason,
    ScopeMatch,
    ScorerKind,
    clean_task_context,
    detect_reference_signal,
)
from hindsight_api.engine.context_admission.models import ContextAdmissionDecision
from hindsight_api.engine.context_admission.scoring import deterministic_score, observation_scope, scope_matches
from hindsight_api.engine.context_admission.selection import (
    apply_budget,
    currentness_rejection,
    passes_threshold,
    readmit_twins,
    similarity_matrix,
    suppress_redundant,
)
from hindsight_api.engine.decision_model import DecisionAnswer, DecisionResponse
from hindsight_api.engine.response_models import RecallLifecycle
from hindsight_api.engine.search.types import ArmScores, MergedCandidate, RetrievalResult, ScoredResult

REPO = "github.com/acme/payments"


def _settings(**overrides) -> ContextAdmissionSettings:
    base = ContextAdmissionSettings.resolve({"context_admission_mode": "filter"}, None)
    return replace(base, **overrides)


def _sr(
    memory_id: str,
    *,
    ce: float = 0.9,
    semantic: float | None = 0.8,
    keyword: bool = True,
    arms: int = 2,
    metadata: dict[str, str] | None = None,
    status: str | None = None,
    retrievability: float | None = None,
    importance: float | None = None,
    text: str | None = None,
    fact_type: str = "world",
) -> ScoredResult:
    retrieval = RetrievalResult(
        id=memory_id, text=text or f"memory {memory_id}", fact_type=fact_type, metadata=metadata
    )
    candidate = MergedCandidate(
        retrieval=retrieval,
        rrf_score=0.01,
        source_ranks={name: 1 for name in ("semantic_rank", "bm25_rank", "graph_rank", "temporal_rank")[:arms]},
        arm_scores=ArmScores(semantic=semantic, keyword=1.0 if keyword else None),
        lifecycle=RecallLifecycle(truth_status=status) if status else None,
    )
    return ScoredResult(
        candidate=candidate,
        cross_encoder_score_normalized=ce,
        weight=ce,
        retrievability=retrievability,
        importance=importance,
    )


def _recall(settings: ContextAdmissionSettings, *, purpose=Purpose.EXPLICIT_SEARCH, task_context=None, query="q"):
    return ContextAdmissionRecall(
        settings=settings,
        purpose=purpose,
        task_context=task_context or {},
        query=query,
        reference_signal=detect_reference_signal(query) if purpose is Purpose.AUTO_CONTEXT else None,
        count_tokens=CachedTokenCounter(lambda text: len(text.split())),
    )


async def _no_embeddings(ids):
    return {}


class FakeDecisionModel:
    """Answers every noul with a fixed probability per memory text; records each request."""

    def __init__(self, by_text: dict[str, float] | None = None, *, fail: Exception | None = None):
        self.by_text = by_text or {}
        self.fail = fail
        self.calls: list[tuple[object, dict]] = []

    async def ask(self, state, questions):
        self.calls.append((state, questions))
        if self.fail is not None:
            raise self.fail
        answers = {
            key: DecisionAnswer(noul=self.by_text.get(q.instructions["memory"], 0.0)) for key, q in questions.items()
        }
        return DecisionResponse(answers=answers)


# --------------------------------------------------------------------------- the scorer


async def test_a_stable_important_but_irrelevant_memory_is_rejected():
    """Importance 1.0, a long stability and R = 0.99 never stand in for relevance."""
    sr = _sr("stable", ce=0.01, semantic=0.3, keyword=False, arms=1, retrievability=0.99, importance=1.0)
    settings = _settings()
    run = _recall(settings)
    outcome = await run.decide([sr], historical=False, load_embeddings=_no_embeddings, jev=None)
    decision = outcome.decisions[0]
    assert decision.reason is Reason.BELOW_THRESHOLD
    assert decision.admission_score < settings.threshold_explicit
    assert outcome.abstained


async def test_a_weak_but_precise_memory_is_admitted():
    """R = 0.15 does not keep out the exact answer to an exact question."""
    sr = _sr("workaround", ce=0.95, semantic=0.85, retrievability=0.15, importance=0.3)
    run = _recall(_settings(), purpose=Purpose.AUTO_CONTEXT, query="What exact workaround did we use for FIN-1602?")
    outcome = await run.decide([sr], historical=False, load_embeddings=_no_embeddings, jev=None)
    assert outcome.decisions[0].admitted
    assert outcome.admitted_ids == ["workaround"]


def test_deterministic_score_is_the_sigmoid_of_the_weighted_sum():
    weights = {"bias": -1.0, "reranker": 2.0, "scope_mismatch.repository": -3.0, "importance": 1.0}
    vector = {"bias": 1.0, "reranker": 0.5, "scope_mismatch.repository": 1.0, "importance": 0.2, "rank": 1.0}
    score = deterministic_score(vector, weights)

    def sig(x):
        return 1 / (1 + math.exp(-x))

    assert score.task_relevance == pytest.approx(sig(-1.0 + 1.0))
    assert score.scope_score == pytest.approx(sig(-3.0))
    assert score.expected_utility == pytest.approx(sig(0.2))
    assert score.admission_score == pytest.approx(sig(-1.0 + 1.0 - 3.0 + 0.2))


def test_scope_matching_is_per_key_and_normalized():
    scope = scope_matches(
        {"repository": " GitHub.com/Acme/Payments/ ", "module": "deploy", "branch": "main"},
        {"repository": "github.com/acme/payments", "module": "frontend"},
        ("repository", "module", "branch", "environment"),
    )
    assert scope == {
        "repository": ScopeMatch.MATCH,
        "module": ScopeMatch.MISMATCH,
        "branch": ScopeMatch.UNKNOWN,
        "environment": ScopeMatch.UNKNOWN,
    }


async def test_soft_scope_scores_rather_than_filters():
    """A module mismatch lowers the score; it does not reject on its own (filter mode)."""
    same = _sr("same", metadata={"repository": REPO, "module": "deploy"})
    other = _sr("other", metadata={"repository": REPO, "module": "frontend"})
    run = _recall(_settings(), task_context={"repository": REPO, "module": "deploy"})
    outcome = await run.decide([same, other], historical=False, load_embeddings=_no_embeddings, jev=None)
    by_id = {d.memory_id: d for d in outcome.decisions}
    assert by_id["other"].admission_score < by_id["same"].admission_score
    assert by_id["other"].scope_score < 0.5 < by_id["same"].scope_score
    assert by_id["other"].admitted and by_id["same"].admitted


# --------------------------------------------------------------------------- the gate


@pytest.mark.parametrize(
    ("status", "historical", "expected"),
    [
        ("active", False, None),
        ("uncertain", False, None),
        ("superseded", False, Reason.SUPERSEDED),
        ("superseded", True, None),
        ("refuted", False, Reason.REFUTED),
        ("refuted", True, Reason.REFUTED),
    ],
)
def test_currentness_matrix(status, historical, expected):
    assert currentness_rejection(status, historical=historical) is expected


def test_an_uncertain_claim_needs_the_margin():
    assert passes_threshold(0.65, threshold=0.6, uncertain=False, uncertain_margin=0.1)
    assert not passes_threshold(0.65, threshold=0.6, uncertain=True, uncertain_margin=0.1)
    assert passes_threshold(0.7, threshold=0.6, uncertain=True, uncertain_margin=0.1)


async def test_currentness_reasons_reach_the_decisions_and_the_abstention_counts():
    candidates = [_sr("old", status="superseded"), _sr("wrong", status="refuted"), _sr("doubt", status="uncertain")]
    run = _recall(_settings())
    outcome = await run.decide(candidates, historical=False, load_embeddings=_no_embeddings, jev=None)
    reasons = {d.memory_id: d.reason for d in outcome.decisions}
    assert reasons == {"old": Reason.SUPERSEDED, "wrong": Reason.REFUTED, "doubt": Reason.ADMITTED}
    assert {d.memory_id: d.uncertain for d in outcome.decisions}["doubt"]

    history = await run.decide(candidates, historical=True, load_embeddings=_no_embeddings, jev=None)
    assert {d.memory_id: d.reason for d in history.decisions}["old"] is Reason.ADMITTED


async def test_abstention_admits_nothing_and_counts_the_reasons():
    candidates = [
        _sr("joke", ce=0.02, semantic=0.4, keyword=False, arms=1),
        _sr("other-repo", metadata={"repository": "github.com/acme/web"}),
        _sr("old", status="superseded"),
    ]
    run = _recall(_settings(mode=AdmissionMode.STRICT), task_context={"repository": REPO})
    outcome = await run.decide(candidates, historical=False, load_embeddings=_no_embeddings, jev=None)
    assert outcome.admitted_ids == []
    assert outcome.abstained
    counts = outcome.reason_counts()
    assert counts["below_threshold"] == 1 and counts["wrong_scope"] == 1 and counts["superseded"] == 1
    assert counts["refuted"] == counts["redundant"] == counts["budget"] == 0
    assert outcome.cross_scope_candidates(("repository",)) == 1


async def test_hard_scope_rejects_only_in_the_full_gate():
    other = _sr("other-repo", ce=0.99, semantic=0.95, metadata={"repository": "github.com/acme/web"})
    unknown = _sr("no-repo", ce=0.99, semantic=0.95)
    context = {"repository": REPO}
    strict = await _recall(_settings(mode=AdmissionMode.STRICT), task_context=context).decide(
        [other, unknown], historical=False, load_embeddings=_no_embeddings, jev=None
    )
    assert {d.memory_id: d.reason for d in strict.decisions} == {
        "other-repo": Reason.WRONG_SCOPE,
        "no-repo": Reason.ADMITTED,
    }
    # filter: the mismatch is a (heavy) score term, not a gate
    filtered = await _recall(_settings(mode=AdmissionMode.FILTER), task_context=context).decide(
        [other], historical=False, load_embeddings=_no_embeddings, jev=None
    )
    assert filtered.decisions[0].reason is not Reason.WRONG_SCOPE


async def test_a_hard_scope_key_gates_even_when_it_is_not_a_soft_scope_key():
    """Scope is compared on the soft keys and the hard keys together, so narrowing the soft keys
    does not switch the hard gate off."""
    settings = _settings(mode=AdmissionMode.STRICT, scope_keys=("module", "branch"), hard_scope_keys=("repository",))
    assert settings.compared_scope_keys == ("module", "branch", "repository")
    other = _sr("other-repo", ce=0.99, semantic=0.95, metadata={"repository": "github.com/other/y"})
    outcome = await _recall(settings, task_context={"repository": REPO}).decide(
        [other], historical=False, load_embeddings=_no_embeddings, jev=None
    )
    decision = outcome.decisions[0]
    assert decision.reason is Reason.WRONG_SCOPE
    assert decision.scope["repository"] is ScopeMatch.MISMATCH

    # A hard key nobody weighs gates only: the mismatch adds nothing to the score.
    tenant = _settings(mode=AdmissionMode.STRICT, scope_keys=("module",), hard_scope_keys=("tenant",))
    foreign = _sr("foreign", ce=0.99, semantic=0.95, metadata={"tenant": "b"})
    unstamped = _sr("unstamped", ce=0.99, semantic=0.95)
    run = _recall(tenant, task_context={"tenant": "a"})
    (gated,) = (await run.decide([foreign], historical=False, load_embeddings=_no_embeddings, jev=None)).decisions
    (kept,) = (await run.decide([unstamped], historical=False, load_embeddings=_no_embeddings, jev=None)).decisions
    assert gated.reason is Reason.WRONG_SCOPE and kept.admitted
    assert gated.admission_score == pytest.approx(kept.admission_score)


def test_an_observation_takes_its_scope_from_its_sources():
    keys = ("repository", "module")
    sources = [{"repository": "github.com/Other/y/"}, {"repository": "github.com/other/y", "module": "api"}, {}]
    assert observation_scope({}, sources, keys) == {"repository": ("github.com/other/y",), "module": ("api",)}
    # Its own metadata wins where it names a key.
    assert observation_scope({"repository": REPO}, sources, keys)["repository"] == REPO
    mixed = observation_scope({}, [{"repository": REPO}, {"repository": "github.com/other/y"}], keys)
    assert scope_matches({"repository": REPO}, mixed, keys)["repository"] is ScopeMatch.MATCH
    assert scope_matches({"repository": "github.com/third/z"}, mixed, keys)["repository"] is ScopeMatch.MISMATCH
    assert scope_matches({"repository": REPO}, {}, keys)["repository"] is ScopeMatch.UNKNOWN


async def test_strict_rejects_an_observation_consolidated_from_another_repositorys_facts():
    other_obs = _sr("obs-other", ce=0.99, semantic=0.95, fact_type="observation")
    own_obs = _sr("obs-own", ce=0.99, semantic=0.95, fact_type="observation")
    fact = _sr("fact", ce=0.99, semantic=0.95)
    asked: list[list[str]] = []

    async def sources(observations):
        asked.append([sr.id for sr in observations])
        return {
            "obs-other": [{"repository": "github.com/other/y"}, {"repository": "github.com/other/y"}],
            "obs-own": [{"repository": REPO}],
        }

    run = _recall(_settings(mode=AdmissionMode.STRICT), purpose=Purpose.AUTO_CONTEXT, task_context={"repository": REPO})
    outcome = await run.decide(
        [other_obs, own_obs, fact],
        historical=False,
        load_embeddings=_no_embeddings,
        jev=None,
        load_source_metadata=sources,
    )
    assert asked == [["obs-other", "obs-own"]]  # one read, observations only
    reasons = {d.memory_id: d.reason for d in outcome.decisions}
    assert reasons == {"obs-other": Reason.WRONG_SCOPE, "obs-own": Reason.ADMITTED, "fact": Reason.ADMITTED}

    # No scope key in the request: nothing to compare, so no read.
    asked.clear()
    await _recall(_settings(mode=AdmissionMode.STRICT)).decide(
        [other_obs], historical=False, load_embeddings=_no_embeddings, jev=None, load_source_metadata=sources
    )
    assert asked == []

    async def broken(observations):
        raise ConnectionError("gone")

    degraded = await run.decide(
        [other_obs], historical=False, load_embeddings=_no_embeddings, jev=None, load_source_metadata=broken
    )
    assert degraded.scope_error == "ConnectionError"
    assert degraded.decisions[0].scope["repository"] is ScopeMatch.UNKNOWN


async def test_a_rank_scored_reranker_is_not_read_as_relevance():
    """A passthrough (or rank-scored) reranker seeds the top candidates with high positional
    scores; the deterministic scorer must read the semantic similarity instead and abstain."""
    # What apply_combined_scoring seeds for a passthrough: 1.0 - 0.9 * rank / (n - 1).
    n = 30
    pool = [_sr(f"m{rank}", ce=1.0 - 0.9 * rank / (n - 1), semantic=0.05, keyword=False, arms=1) for rank in range(n)]
    run = _recall(_settings(), purpose=Purpose.AUTO_CONTEXT)
    as_relevance = await run.decide(pool, historical=False, load_embeddings=_no_embeddings, jev=None)
    assert not as_relevance.abstained  # read as relevance, positions admit the top of any pool
    outcome = await run.decide(
        pool, historical=False, load_embeddings=_no_embeddings, jev=None, rank_based_relevance=True
    )
    assert outcome.abstained
    assert outcome.relevance_source == "semantic"
    top = outcome.decisions[0]
    assert top.features["reranker"] == pytest.approx(0.05)
    assert "rank" not in top.features

    # A semantically close memory still clears the threshold without a reranker score.
    close = _sr("close", ce=1.0, semantic=0.85, keyword=True, arms=2)
    admitted = await run.decide(
        [close], historical=False, load_embeddings=_no_embeddings, jev=None, rank_based_relevance=True
    )
    assert admitted.admitted_ids == ["close"]


def test_thresholds_follow_the_purpose_and_the_reference_signal():
    settings = _settings(threshold_auto=0.6, threshold_explicit=0.3, reference_bonus=0.2)
    assert settings.threshold(Purpose.AUTO_CONTEXT, reference_signal=False) == 0.6
    assert settings.threshold(Purpose.AUTO_CONTEXT, reference_signal=True) == pytest.approx(0.4)
    assert settings.threshold(Purpose.EXPLICIT_SEARCH, reference_signal=True) == 0.3
    assert settings.threshold(Purpose.HISTORICAL_SEARCH, reference_signal=False) == 0.3


async def test_embedding_similarity_alone_does_not_inject_a_memory():
    """Two candidates equally close to the query in embedding space and on BM25; the reranker reads
    only one as an answer, and only that one is injected automatically."""
    answer = _sr("answer", ce=0.9, semantic=0.9)
    lookalike = _sr("lookalike", ce=0.02, semantic=0.9)
    outcome = await _recall(_settings(), purpose=Purpose.AUTO_CONTEXT).decide(
        [answer, lookalike], historical=False, load_embeddings=_no_embeddings, jev=None
    )
    assert outcome.admitted_ids == ["answer"]
    assert {d.memory_id: d.reason for d in outcome.decisions}["lookalike"] is Reason.BELOW_THRESHOLD


async def test_a_search_someone_asked_for_admits_what_automatic_injection_rejects():
    """One candidate between the default thresholds: explicit and historical searches get it, an
    automatic injection does not, unless the task refers to shared history."""
    candidate = _sr("mid", ce=0.02, semantic=0.9)
    settings = _settings()

    async def decide(purpose, query="How is the payments API deployed?"):
        (decision,) = (
            await _recall(settings, purpose=purpose, query=query).decide(
                [candidate], historical=purpose is Purpose.HISTORICAL_SEARCH, load_embeddings=_no_embeddings, jev=None
            )
        ).decisions
        return decision

    auto = await decide(Purpose.AUTO_CONTEXT)
    lowered = settings.threshold(Purpose.AUTO_CONTEXT, reference_signal=True)
    assert lowered <= auto.admission_score < settings.threshold_auto
    assert auto.admission_score >= settings.threshold_explicit
    assert auto.reason is Reason.BELOW_THRESHOLD
    assert (await decide(Purpose.EXPLICIT_SEARCH)).admitted
    assert (await decide(Purpose.HISTORICAL_SEARCH)).admitted
    assert (await decide(Purpose.AUTO_CONTEXT, "What did we decide about deploying the payments API?")).admitted


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What did we decide about the retry policy?", "what_did_we_decide"),
        ("Do you remember the flaky test fix?", "remember"),
        ("Use the same approach as last time", "last_time"),
        ("We discussed a lock ordering issue", "we_discussed"),
        ("previously the deploy used blue/green", "previously"),
        ("as we said, keep the API stable", "as_mentioned"),
        ("Remember to run the tests before pushing", None),
        ("How is the Payments API deployed?", None),
    ],
)
def test_reference_signal_detector(query, expected):
    assert detect_reference_signal(query) == expected


def test_task_context_is_cleaned():
    assert clean_task_context(None) == {}
    cleaned = clean_task_context({"repository": f"  {REPO} ", "branch": "", "module": 3, " ": "x", "x" * 3: "y" * 999})
    assert cleaned == {"repository": REPO, "xxx": "y" * 256}


# --------------------------------------------------------------------------- redundancy and budget


def _decision(memory_id: str, score: float, tokens: int = 10) -> ContextAdmissionDecision:
    return ContextAdmissionDecision(
        memory_id=memory_id,
        base_retrieval_score=score,
        scope_score=0.5,
        task_relevance=score,
        expected_utility=0.5,
        truth_status="active",
        retrievability=None,
        admission_score=score,
        admitted=True,
        reason=Reason.ADMITTED,
        tokens=tokens,
    )


def test_similarity_edge_cases():
    ids = ["a", "same", "orthogonal", "zero", "short", "nan", "missing"]
    embeddings = {
        "a": [1.0, 0.0],
        "same": [2.0, 0.0],
        "orthogonal": [0.0, 1.0],
        "zero": [0.0, 0.0],
        "short": [1.0],
        "nan": [math.nan, 1.0],
    }
    sims = similarity_matrix(ids, embeddings)
    assert sims[0, 1] == pytest.approx(1.0)
    assert sims[0, 2] == pytest.approx(0.0)
    assert sims[0, 3] == 0.0  # a zero vector: nothing known
    assert sims[0, 4] == 0.0  # different lengths
    assert sims[0, 5] == 0.0  # a non-finite vector
    assert sims[0, 6] == 0.0  # no stored embedding


def test_near_duplicates_are_suppressed_and_counted_on_the_canonical():
    decisions = [_decision("a", 0.9), _decision("a-twin", 0.85), _decision("b", 0.8), _decision("no-vector", 0.7)]
    embeddings = {"a": [1.0, 0.0, 0.0], "a-twin": [0.99, 0.05, 0.0], "b": [0.0, 1.0, 0.0]}
    redundancy = suppress_redundant(decisions, embeddings, mmr_lambda=0.7, dup_similarity=0.92)
    by_id = {d.memory_id: d for d in decisions}
    assert by_id["a-twin"].reason is Reason.REDUNDANT and not by_id["a-twin"].admitted
    assert by_id["a"].duplicates == 1
    assert by_id["b"].admitted and by_id["no-vector"].admitted
    assert set(redundancy.values) == {"a", "b", "no-vector"}
    assert redundancy.values["a"] == pytest.approx(0.7 * 0.9)
    assert [t.memory_id for t in redundancy.twins["a"]] == ["a-twin"]


def test_mmr_prefers_novelty_over_a_slightly_better_near_twin():
    """lambda trades score for novelty: a distinct memory is taken before a similar, higher one."""
    decisions = [_decision("a", 0.9), _decision("a-ish", 0.88), _decision("b", 0.8)]
    embeddings = {"a": [1.0, 0.0], "a-ish": [0.9, math.sqrt(1 - 0.81)], "b": [0.0, 1.0]}
    values = suppress_redundant(decisions, embeddings, mmr_lambda=0.5, dup_similarity=0.99).values
    assert values["b"] > values["a-ish"]


def _reference_mmr(decisions, embeddings, *, mmr_lambda, dup_similarity):
    """The plain greedy definition, every max_sim recomputed: what the incremental pass must equal."""

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm > 0 else 0.0

    remaining = sorted(decisions, key=lambda d: -d.admission_score)
    kept, out = [], {}
    while remaining:
        best = max(
            range(len(remaining)),
            key=lambda i: (
                mmr_lambda * remaining[i].admission_score
                - (1 - mmr_lambda)
                * max([0.0, *(cos(embeddings[remaining[i].memory_id], embeddings[k]) for k in kept)]),
                -i,
            ),
        )
        chosen = remaining.pop(best)
        sim = max([0.0, *(cos(embeddings[chosen.memory_id], embeddings[k]) for k in kept)])
        if sim >= dup_similarity:
            out[chosen.memory_id] = "redundant"
        else:
            kept.append(chosen.memory_id)
            out[chosen.memory_id] = "kept"
    return out


def test_redundancy_is_quadratic_and_equals_the_plain_greedy_definition():
    """Thirty candidates at the default embedding size: one similarity per pair, computed in one
    matrix product, and the same decisions as recomputing every max_sim at every step."""
    import random
    import time

    rng = random.Random(7)
    base = [[rng.gauss(0, 1) for _ in range(384)] for _ in range(10)]
    embeddings = {}
    for i in range(30):
        # Three near-copies of each of ten topics, so some pairs cross the duplicate threshold.
        embeddings[f"m{i}"] = [x + rng.gauss(0, 0.15) for x in base[i % 10]]
    scores = {f"m{i}": rng.uniform(0.4, 0.99) for i in range(30)}

    def fresh():
        return [_decision(m, scores[m]) for m in embeddings]

    expected = _reference_mmr(fresh(), embeddings, mmr_lambda=0.7, dup_similarity=0.92)
    decisions = fresh()
    start = time.perf_counter()
    suppress_redundant(decisions, embeddings, mmr_lambda=0.7, dup_similarity=0.92)
    elapsed = time.perf_counter() - start
    got = {d.memory_id: "redundant" if d.reason is Reason.REDUNDANT else "kept" for d in decisions}
    assert got == expected
    assert "redundant" in got.values() and "kept" in got.values()
    # About a millisecond in practice; the bound only catches a return to per-step recomputation.
    assert elapsed < 0.05


def test_budget_fills_tokens_by_value_per_token():
    decisions = [
        _decision("long", 0.9, tokens=900),
        _decision("short", 0.6, tokens=50),
        _decision("mid", 0.8, tokens=200),
    ]
    apply_budget(decisions, {}, max_memories=10, max_tokens=300)
    reasons = {d.memory_id: d.reason for d in decisions}
    assert reasons == {"long": Reason.BUDGET, "short": Reason.ADMITTED, "mid": Reason.ADMITTED}

    capped = [_decision("a", 0.9), _decision("b", 0.8), _decision("c", 0.7)]
    apply_budget(capped, {}, max_memories=2, max_tokens=10_000)
    assert [d.reason for d in capped] == [Reason.ADMITTED, Reason.ADMITTED, Reason.BUDGET]


def test_when_the_count_cap_binds_the_most_valuable_memory_is_not_pushed_out():
    """Ten short weak memories fit the count and leave most tokens unused; the best one stays."""
    decisions = [_decision("best", 0.99, tokens=400)] + [_decision(f"weak{i}", 0.40, tokens=30) for i in range(10)]
    apply_budget(decisions, {}, max_memories=10, max_tokens=2000)
    by_id = {d.memory_id: d for d in decisions}
    assert by_id["best"].admitted
    assert sum(d.admitted for d in decisions) == 10
    assert sum(d.reason is Reason.BUDGET for d in decisions) == 1


def test_a_twin_takes_the_place_of_a_canonical_the_budget_dropped():
    decisions = [_decision("long", 0.9, tokens=500), _decision("short-twin", 0.85, tokens=50), _decision("b", 0.8)]
    embeddings = {"long": [1.0, 0.0], "short-twin": [1.0, 0.01], "b": [0.0, 1.0]}
    redundancy = suppress_redundant(decisions, embeddings, mmr_lambda=0.7, dup_similarity=0.92)
    by_id = {d.memory_id: d for d in decisions}
    assert by_id["short-twin"].reason is Reason.REDUNDANT
    apply_budget(decisions, redundancy.values, max_memories=5, max_tokens=100)
    assert by_id["long"].reason is Reason.BUDGET
    readmit_twins(decisions, redundancy, max_memories=5, max_tokens=100)
    assert by_id["short-twin"].admitted and by_id["short-twin"].reason is Reason.ADMITTED
    assert by_id["b"].admitted
    assert by_id["long"].duplicates == 0 and by_id["short-twin"].duplicates == 0

    # No room left: the twin stays redundant.
    tight = [_decision("long", 0.9, tokens=500), _decision("short-twin", 0.85, tokens=50), _decision("b", 0.8)]
    redundancy = suppress_redundant(tight, embeddings, mmr_lambda=0.7, dup_similarity=0.92)
    apply_budget(tight, redundancy.values, max_memories=1, max_tokens=100)
    readmit_twins(tight, redundancy, max_memories=1, max_tokens=100)
    assert {d.memory_id: d.reason for d in tight} == {
        "long": Reason.BUDGET,
        "short-twin": Reason.REDUNDANT,
        "b": Reason.ADMITTED,
    }


async def test_strict_runs_redundancy_and_budget_and_keeps_recall_order():
    candidates = [_sr("a", text="alpha one"), _sr("b", text="beta two"), _sr("a2", text="alpha one again")]

    async def embeddings(ids):
        assert sorted(ids) == ["a", "a2", "b"]
        return {"a": [1.0, 0.0], "a2": [1.0, 0.01], "b": [0.0, 1.0]}

    run = _recall(_settings(mode=AdmissionMode.STRICT, max_memories=5))
    outcome = await run.decide(candidates, historical=False, load_embeddings=embeddings, jev=None)
    assert outcome.admitted_ids == ["a", "b"]
    assert {d.memory_id: d.reason for d in outcome.decisions}["a2"] is Reason.REDUNDANT

    async def broken(ids):
        raise ConnectionError("gone")

    degraded = await run.decide(candidates, historical=False, load_embeddings=broken, jev=None)
    assert degraded.embeddings_error == "ConnectionError"
    assert degraded.admitted_ids == ["a", "b", "a2"]


# --------------------------------------------------------------------------- the Jev scorer


async def test_jev_asks_once_for_every_candidate_and_uses_the_probability():
    candidates = [_sr(f"m{i}", text=f"text {i}") for i in range(5)]
    model = FakeDecisionModel({"text 0": 0.9, "text 1": 0.1, "text 2": 0.8})
    run = _recall(
        _settings(scorer=ScorerKind.JEV),
        purpose=Purpose.AUTO_CONTEXT,
        task_context={"repository": REPO},
        query="How is the Payments API deployed?",
    )
    outcome = await run.decide(candidates, historical=False, load_embeddings=_no_embeddings, jev=model)
    assert len(model.calls) == 1
    state, questions = model.calls[0]
    assert state["task"] == "How is the Payments API deployed?" and state["purpose"] == "auto_context"
    assert REPO in state["task_context"]
    assert len(questions) == 5
    assert all(q.type == "noul" and q.instructions["question"] == JEV_QUESTION for q in questions.values())
    scores = {d.memory_id: d.admission_score for d in outcome.decisions}
    assert scores["m0"] == pytest.approx(0.9) and scores["m1"] == pytest.approx(0.1)
    assert outcome.admitted_ids == ["m0", "m2"]


async def test_hybrid_blends_the_probability_with_the_deterministic_score():
    sr = _sr("m", text="the text")
    settings = _settings(scorer=ScorerKind.HYBRID, jev_weight=0.25)
    det = await _recall(_settings()).decide([sr], historical=False, load_embeddings=_no_embeddings, jev=None)
    hybrid = await _recall(settings).decide(
        [sr], historical=False, load_embeddings=_no_embeddings, jev=FakeDecisionModel({"the text": 0.2})
    )
    expected = 0.25 * 0.2 + 0.75 * det.decisions[0].admission_score
    assert hybrid.decisions[0].admission_score == pytest.approx(expected)
    assert hybrid.decisions[0].task_relevance == pytest.approx(0.2)


async def test_a_failed_or_missing_decision_model_falls_back_to_the_deterministic_score():
    sr = _sr("m")
    det = await _recall(_settings()).decide([sr], historical=False, load_embeddings=_no_embeddings, jev=None)
    failed = await _recall(_settings(scorer=ScorerKind.JEV)).decide(
        [sr], historical=False, load_embeddings=_no_embeddings, jev=FakeDecisionModel(fail=TimeoutError())
    )
    missing = await _recall(_settings(scorer=ScorerKind.HYBRID)).decide(
        [sr], historical=False, load_embeddings=_no_embeddings, jev=None
    )
    assert failed.jev_error == "TimeoutError" and missing.jev_error == "not_configured"
    for outcome in (failed, missing):
        assert outcome.decisions[0].admission_score == pytest.approx(det.decisions[0].admission_score)


async def test_the_gates_hold_under_the_jev_scorer():
    """A probability of 1.0 does not admit a refuted memory or another repository's (strict)."""
    refuted = _sr("refuted", text="r", status="refuted")
    other = _sr("other", text="o", metadata={"repository": "github.com/acme/web"})
    run = _recall(_settings(mode=AdmissionMode.STRICT, scorer=ScorerKind.JEV), task_context={"repository": REPO})
    outcome = await run.decide(
        [refuted, other], historical=False, load_embeddings=_no_embeddings, jev=FakeDecisionModel({"r": 1.0, "o": 1.0})
    )
    assert {d.memory_id: d.reason for d in outcome.decisions} == {
        "refuted": Reason.REFUTED,
        "other": Reason.WRONG_SCOPE,
    }


# --------------------------------------------------------------------------- apply and observe


async def test_filter_returns_only_admitted_judged_candidates():
    """The unjudged tail never backfills a rejected candidate, so [] is reachable."""
    candidates = [_sr("good"), _sr("bad", ce=0.0, semantic=0.1, keyword=False, arms=1), _sr("tail")]
    run = _recall(_settings(candidates=2))
    outcome = await run.decide(candidates, historical=False, load_embeddings=_no_embeddings, jev=None)
    assert [d.memory_id for d in outcome.decisions] == ["good", "bad"]
    assert [sr.id for sr in run.apply(candidates)] == ["good"]
    assert run.shadow is None


async def test_observe_changes_nothing_and_replays_the_tail_over_the_shadow():
    candidates = [_sr("good", text="a b"), _sr("bad", ce=0.0, semantic=0.1, keyword=False, arms=1, text="c d")]
    run = _recall(_settings(mode=AdmissionMode.OBSERVE))
    await run.decide(candidates, historical=False, load_embeddings=_no_embeddings, jev=None)
    assert run.apply(candidates) is candidates
    observed = run.finish(["good", "bad"], prefer_observations=False, rerank_limit=20, max_tokens=4096)
    assert observed.shadow_ids == ["good"]
    assert observed.would_drop == 1 and observed.would_change_top_k and not observed.would_abstain

    all_bad = [_sr("bad", ce=0.0, semantic=0.1, keyword=False, arms=1)]
    await run.decide(all_bad, historical=False, load_embeddings=_no_embeddings, jev=None)
    run.apply(all_bad)
    assert run.finish(["bad"], prefer_observations=False, rerank_limit=20, max_tokens=4096).would_abstain


def test_recall_reinforcement_is_forced_off_while_admission_filters():
    settings = replace(
        AdaptiveSettings.resolve({"adaptive_memory_mode": "rank"}, get_config(), store_supports=True),
        recall_reinforcement_enabled=True,
    )
    assert recall_reinforcement_allowed(settings)
    assert recall_reinforcement_allowed(settings, context_admission_mode="observe")
    assert not recall_reinforcement_allowed(settings, context_admission_mode="filter")
    assert not recall_reinforcement_allowed(settings, context_admission_mode="strict")
