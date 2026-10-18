"""The memory_lifecycle metrics, labelling rule and aggregation — pure, no server.

The benchmark's numbers are only as good as these definitions, and they were fixed before
any arm ran, so each one is pinned here on hand-built ranked lists: the success case, the
failure case, and the N/A case that must never read as a zero.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.lifecycle_bench.labels import QueryLabels, RankedItem, classify
from hindsight_system_tests.lifecycle_bench.metrics import (
    K_VALUES,
    METRICS,
    aggregate,
    current_precision,
    current_recall,
    historical_recall,
    independent_reinforcement_accuracy,
    mixed_rate,
    refuted_leakage,
    scope_accuracy,
    score_query,
    stale_contamination,
    state_conformance,
    supersession_accuracy,
    uncertainty_accuracy,
)
from hindsight_system_tests.lifecycle_bench.model import ConflictSide, Item, Query, Scenario, Step
from hindsight_system_tests.lifecycle_bench.scenarios import SCENARIOS, holdout_ids, validate


def _scenario(
    *queries: Query, groups: dict[str, tuple[str, ...]] | None = None, tags: dict[str, tuple[str, ...]] | None = None
) -> Scenario:
    docs = {"old", "new", "other", "claim", "fix", "a", "b", "s1", "s2", "n1", "n2", "web", "mobile"}
    tags = tags or {}
    items = tuple(Item(doc=d, content=d, facts=(d,), day=0, session="s", tags=tags.get(d, ())) for d in sorted(docs))
    return Scenario(
        id="t", title="t", covers=(), steps=(Step(items=items, decisions=()),), queries=queries, groups=groups or {}
    )


def _labels(query: Query, **kwargs) -> QueryLabels:
    return QueryLabels.for_query(_scenario(query, **kwargs), query)


def raw(doc: str, rank: int = 0, tags: tuple[str, ...] = ()) -> RankedItem:
    return RankedItem(rank=rank, memory_id=f"raw-{doc}", type="world", text=f"fact {doc}", docs=(doc,), tags=tags)


def obs(text: str, *docs: str, tags: tuple[str, ...] = ()) -> RankedItem:
    return RankedItem(rank=0, memory_id=f"obs-{text}", type="observation", text=text, docs=tuple(docs), tags=tags)


JAVA_NOW = Query(
    id="q",
    text="Which Java?",
    kind="current",
    day=0,
    required_active=("new",),
    forbidden_stale=("old",),
    current_pattern=r"\b25\b",
    stale_pattern=r"\b21\b",
)
JAVA_BEFORE = Query(
    id="h",
    text="Which Java before?",
    kind="historical",
    day=0,
    allowed_historical=("old",),
    current_context=("new",),
    current_pattern=r"\b25\b",
    stale_pattern=r"\b21\b",
)


# ---------------------------------------------------------------------------
# The labelling rule
# ---------------------------------------------------------------------------


class TestClassify:
    def test_a_raw_fact_is_labelled_by_its_document(self):
        labels = _labels(JAVA_NOW)
        assert classify(raw("new"), labels) == "current"
        assert classify(raw("old"), labels) == "stale"
        assert classify(raw("other"), labels) == "other"

    def test_an_observation_needs_a_current_citation_and_the_current_value(self):
        labels = _labels(JAVA_NOW)
        assert classify(obs("runs on Java 25", "new"), labels) == "current"
        # Cites the upgrade fact but states the old value only.
        assert classify(obs("runs on Java 21", "new"), labels) == "stale"

    def test_all_stale_citations_make_it_stale_whatever_the_text(self):
        assert classify(obs("the runtime", "old"), _labels(JAVA_NOW)) == "stale"

    def test_the_in_place_update_that_folds_history_in_is_mixed(self):
        folded = obs("runs on Java 25; it was upgraded from Java 21", "old", "new")
        assert classify(folded, _labels(JAVA_NOW)) == "mixed"

    def test_refuted_citations_or_refuted_value_make_it_refuted(self):
        query = Query(
            id="r",
            text="cache?",
            kind="current",
            day=0,
            required_active=("fix",),
            forbidden_refuted=("claim",),
            current_pattern=r"no cache",
            refuted_pattern=r"uses Redis",
        )
        labels = _labels(query)
        assert classify(obs("The API uses Redis as a cache", "claim"), labels) == "refuted"
        assert classify(obs("The API uses Redis", "fix"), labels) == "refuted"
        # An in-place correction that keeps the refuted fact as a source is still the correction.
        assert classify(obs("The API has no cache; Redis was never deployed", "claim", "fix"), labels) == "current"

    def test_without_patterns_citations_decide(self):
        query = Query(id="c", text="?", kind="current", day=0, required_active=("a", "b"))
        labels = _labels(query)
        assert classify(obs("both", "a", "b"), labels) == "current"
        assert classify(obs("unrelated", "other"), labels) == "other"

    def test_unrelated_citations_do_not_dilute_a_stale_observation(self):
        assert classify(obs("runs on Java 21", "old", "other"), _labels(JAVA_NOW)) == "stale"


# ---------------------------------------------------------------------------
# Current-state metrics
# ---------------------------------------------------------------------------


class TestCurrentMetrics:
    def test_precision_counts_current_and_mixed_over_the_cut(self):
        items = [obs("runs on Java 25", "new"), raw("old"), obs("Java 25; earlier 21", "old", "new"), raw("other")]
        assert current_precision(items, _labels(JAVA_NOW), 4) == pytest.approx(2 / 4)
        assert current_precision(items, _labels(JAVA_NOW), 1) == 1.0

    def test_precision_is_undefined_on_an_empty_answer_and_on_history_queries(self):
        assert current_precision([], _labels(JAVA_NOW), 5) is None
        assert current_precision([raw("old")], _labels(JAVA_BEFORE), 5) is None

    def test_recall_counts_a_label_present_through_a_raw_fact_or_a_current_observation(self):
        labels = _labels(JAVA_NOW)
        assert current_recall([raw("other"), raw("new")], labels, 1) == 0.0
        assert current_recall([raw("other"), raw("new")], labels, 2) == 1.0
        assert current_recall([obs("Java 25; earlier 21", "old", "new")], labels, 1) == 1.0
        # Cites the successor but presents the old value: the successor is not present.
        assert current_recall([obs("runs on Java 21", "new")], labels, 1) == 0.0

    def test_recall_of_a_group_label_is_satisfied_by_any_member(self):
        query = Query(id="g", text="?", kind="current", day=0, required_active=("moved", "other"))
        labels = _labels(query, groups={"moved": ("a", "b")})
        assert current_recall([raw("b")], labels, 5) == 0.5
        assert current_recall([raw("b"), raw("other")], labels, 5) == 1.0

    def test_recall_needs_required_labels(self):
        query = Query(id="n", text="?", kind="current", day=0)
        assert current_recall([raw("a")], _labels(query), 5) is None


class TestStaleAndRefuted:
    def test_contamination_is_the_stale_share_split_into_raw_and_observation_parts(self):
        items = [raw("old"), obs("runs on Java 21", "old"), obs("runs on Java 25", "new"), raw("new")]
        labels = _labels(JAVA_NOW)
        combined = stale_contamination(items, labels, 4)
        raw_part = stale_contamination(items, labels, 4, "raw")
        obs_part = stale_contamination(items, labels, 4, "observation")
        assert (combined, raw_part, obs_part) == (0.5, 0.25, 0.25)
        assert raw_part + obs_part == combined

    def test_a_mixed_observation_is_not_contamination_but_is_reported(self):
        items = [obs("Java 25; earlier 21", "old", "new")]
        labels = _labels(JAVA_NOW)
        assert stale_contamination(items, labels, 5) == 0.0
        assert mixed_rate(items, labels, 5) == 1.0

    def test_contamination_of_an_empty_answer_is_zero_and_history_queries_are_exempt(self):
        assert stale_contamination([], _labels(JAVA_NOW), 5) == 0.0
        assert stale_contamination([raw("old")], _labels(JAVA_BEFORE), 5) is None

    def test_contamination_needs_a_superseded_state(self):
        query = Query(id="n", text="?", kind="current", day=0, required_active=("a",))
        assert stale_contamination([raw("a")], _labels(query), 5) is None

    def test_refuted_leakage_applies_to_every_query_kind(self):
        query = Query(
            id="r", text="?", kind="historical", day=0, forbidden_refuted=("claim",), allowed_historical=("old",)
        )
        items = [raw("claim"), raw("old"), raw("other"), raw("fix")]
        labels = _labels(query)
        assert refuted_leakage(items, labels, 4) == 0.25
        assert refuted_leakage(items, labels, 4, "observation") == 0.0
        assert refuted_leakage(items, _labels(JAVA_NOW), 4) is None


class TestHistorical:
    def test_the_earlier_state_is_present_through_a_raw_fact_a_retired_or_a_mixed_observation(self):
        labels = _labels(JAVA_BEFORE)
        assert historical_recall([raw("old")], labels, 1) == 1.0
        assert historical_recall([obs("runs on Java 21", "old")], labels, 1) == 1.0
        assert historical_recall([obs("Java 25; earlier 21", "old", "new")], labels, 1) == 1.0

    def test_an_observation_that_only_states_the_current_value_does_not_carry_history(self):
        # A rewrite that dropped the old value but kept the old fact as a source.
        assert historical_recall([obs("runs on Java 25", "old", "new"), raw("new")], _labels(JAVA_BEFORE), 5) == 0.0

    def test_not_scored_on_current_queries(self):
        assert historical_recall([raw("old")], _labels(JAVA_NOW), 5) is None


class TestScope:
    QUERY = Query(id="s", text="?", kind="current", day=0, required_active=("web",), scope_tags=("team:web",))
    TAGS = {"web": ("team:web",), "mobile": ("team:mobile",)}

    def test_results_outside_the_scope_count_against_it(self):
        labels = _labels(self.QUERY, tags=self.TAGS)
        items = [raw("web", tags=("team:web",)), raw("mobile", tags=("team:mobile",))]
        assert scope_accuracy(items, labels, 2) == 0.5

    def test_an_observation_that_absorbed_another_scopes_evidence_is_outside(self):
        labels = _labels(self.QUERY, tags=self.TAGS)
        polluted = obs("deploys", "web", "mobile", tags=("team:web",))
        assert scope_accuracy([polluted], labels, 1) == 0.0
        assert scope_accuracy([obs("deploys", "web", tags=("team:web",))], labels, 1) == 1.0

    def test_only_scope_queries_with_results_are_scored(self):
        assert scope_accuracy([], _labels(self.QUERY, tags=self.TAGS), 5) is None
        assert scope_accuracy([raw("new")], _labels(JAVA_NOW), 5) is None


class TestSupersession:
    def test_successor_present_and_predecessor_absent(self):
        labels = _labels(JAVA_NOW)
        assert supersession_accuracy([obs("runs on Java 25", "new"), raw("other")], labels, 5) == 1.0
        assert supersession_accuracy([obs("runs on Java 25", "new"), raw("old")], labels, 5) == 0.0
        assert supersession_accuracy([raw("other")], labels, 5) == 0.0

    def test_a_mixed_observation_is_the_successor_not_the_predecessor(self):
        assert supersession_accuracy([obs("Java 25; earlier 21", "old", "new")], _labels(JAVA_NOW), 5) == 1.0

    def test_the_cut_decides(self):
        items = [obs("runs on Java 25", "new"), raw("old")]
        assert supersession_accuracy(items, _labels(JAVA_NOW), 1) == 1.0
        assert supersession_accuracy(items, _labels(JAVA_NOW), 2) == 0.0


class TestUncertainty:
    QUERY = Query(
        id="u",
        text="Node?",
        kind="current",
        day=0,
        required_active=("a", "b"),
        conflict_sides=(ConflictSide(("a",), r"Node 18"), ConflictSide(("b",), r"Node 20")),
    )

    def test_both_sides_must_be_present(self):
        labels = _labels(self.QUERY)
        assert uncertainty_accuracy([raw("a"), raw("b")], labels, 2) == 1.0
        assert uncertainty_accuracy([raw("a"), raw("other")], labels, 2) == 0.0

    def test_an_observation_presents_a_side_only_when_it_states_its_value(self):
        labels = _labels(self.QUERY)
        both = obs("README says Node 18, CI runs Node 20", "a", "b")
        assert uncertainty_accuracy([both], labels, 1) == 1.0
        last_write_wins = obs("The frontend uses Node 20", "a", "b")
        assert uncertainty_accuracy([last_write_wins], labels, 1) == 0.0

    def test_only_conflict_queries_are_scored(self):
        assert uncertainty_accuracy([raw("a")], _labels(JAVA_NOW), 5) is None


class TestIndependentReinforcement:
    QUERY = Query(id="i", text="?", kind="current", day=0, spaced=("spaced",), contrast=("burst",))
    GROUPS = {"spaced": ("s1", "s2"), "burst": ("n1", "n2")}

    def test_the_spaced_memory_must_rank_above_the_burst(self):
        labels = _labels(self.QUERY, groups=self.GROUPS)
        assert independent_reinforcement_accuracy([raw("s2"), raw("n1")], labels, 5) == 1.0
        assert independent_reinforcement_accuracy([obs("burst", "n1", "n2"), raw("s1")], labels, 5) == 0.0

    def test_absence_ranks_below_everything(self):
        labels = _labels(self.QUERY, groups=self.GROUPS)
        assert independent_reinforcement_accuracy([raw("other"), raw("s1")], labels, 5) == 1.0
        assert independent_reinforcement_accuracy([raw("other")], labels, 5) == 0.0
        # Outside the cut counts as absent.
        assert independent_reinforcement_accuracy([raw("n1"), raw("s1")], labels, 1) == 0.0


# ---------------------------------------------------------------------------
# Aggregation and state conformance
# ---------------------------------------------------------------------------


def test_score_query_reports_every_metric_at_every_k():
    scores = score_query([raw("new")], _labels(JAVA_NOW))
    assert set(scores) == {f"{name}@{k}" for name in METRICS for k in K_VALUES}
    assert scores["historical_recall@5"] is None


def test_aggregate_excludes_not_applicable_values_and_counts_the_rest():
    summary = aggregate([{"current_recall@5": 1.0}, {"current_recall@5": 0.0}, {"current_recall@5": None}])
    assert summary["current_recall@5"].mean == 0.5
    assert summary["current_recall@5"].n == 2
    assert summary["historical_recall@5"].mean is None
    assert summary["historical_recall@5"].n == 0


class TestStateConformance:
    EXPECTED = {"old": "superseded", "new": "active", "claim": "refuted"}

    def test_no_reported_state_is_not_applicable_never_zero(self):
        result = state_conformance(self.EXPECTED, {"old": [None], "new": [None]})
        assert (result.conformance, result.retirement_recall, result.false_retirement) == (None, None, None)
        assert result.n_reported == 0

    def test_conformance_retirement_and_false_retirement(self):
        result = state_conformance(self.EXPECTED, {"old": ["superseded"], "new": ["superseded"], "claim": ["active"]})
        assert result.conformance == pytest.approx(1 / 3)
        assert result.retirement_recall == 0.5
        assert result.false_retirement == 1.0

    def test_every_raw_fact_of_a_document_must_agree(self):
        result = state_conformance({"old": "superseded"}, {"old": ["superseded", "active"]})
        assert result.conformance == 0.0


# ---------------------------------------------------------------------------
# The catalogue itself
# ---------------------------------------------------------------------------


def test_the_catalogue_is_valid_and_covers_r34():
    assert validate() == []
    covered = {c for s in SCENARIOS for c in s.covers}
    for name in (
        "preference reversal",
        "software-version upgrade",
        "infrastructure migration",
        "false fact correction",
        "historically true but retired state",
        "conflicting sources",
        "scope-specific divergence",
        "repeated same-session noise",
        "independent reconfirmation",
        "low-value fact decay",
        "exact historical cue",
        "split/generalization",
        "merged duplicate evidence",
    ):
        assert name in covered, name
    assert sum(len(s.queries) for s in SCENARIOS) >= 60


def test_the_holdout_is_a_fixed_third_of_whole_scenarios():
    holdout = holdout_ids()
    assert holdout == holdout_ids(tuple(reversed(SCENARIOS))), "the split must not depend on catalogue order"
    assert len(holdout) == round(len(SCENARIOS) / 3)
    assert holdout <= {s.id for s in SCENARIOS}
