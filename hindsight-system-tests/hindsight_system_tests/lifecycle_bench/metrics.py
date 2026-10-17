"""The memory_lifecycle metrics: pure functions of one query's ranked recall output.

Each takes the ranked results (best first), the query's expanded labels and a cut-off ``k``,
and returns a value in [0, 1], or ``None`` when the metric does not apply to that query.
``None`` is excluded from every mean and counted separately as ``n``: a metric that did not
apply is not a zero. Nothing reads server state, so every arm — including one with no
lifecycle feature at all — is scored by exactly the same arithmetic.

"Present" (used by the recall-style metrics) means: a raw fact of one of the label's
documents is in the top k, or an observation in the top k cites one of them and its class
(``labels.classify``) presents that state.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .labels import ItemClass, QueryLabels, RankedItem, carries, classify
from .model import TruthStatus

K_VALUES = (1, 3, 5, 10)

Part = Literal["all", "raw", "observation"]

_CURRENT_LIKE: frozenset[ItemClass] = frozenset({"current", "mixed"})
_HISTORICAL_LIKE: frozenset[ItemClass] = frozenset({"stale", "mixed"})
_ANY: frozenset[ItemClass] = frozenset({"current", "stale", "refuted", "mixed", "other"})


def _present(
    top: Sequence[RankedItem], docs: frozenset[str], classes: frozenset[ItemClass], labels: QueryLabels
) -> bool:
    return any(carries(item, docs, classes, labels) for item in top)


def _share(top: Sequence[RankedItem], labels: QueryLabels, wanted: frozenset[ItemClass], part: Part = "all") -> float:
    """Share of ``top`` whose class is in ``wanted``, counting only ``part`` in the numerator.

    The denominator is always every result in the cut, so the raw and observation shares of
    one metric add up to its combined value.
    """
    if not top:
        return 0.0
    hits = sum(
        1
        for item in top
        if classify(item, labels) in wanted and (part == "all" or (part == "observation") == item.is_observation)
    )
    return hits / len(top)


def _is_current_query(labels: QueryLabels) -> bool:
    return labels.query.kind == "current"


def current_precision(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """current_precision@k — on current queries: the share of the top-k results that present
    the current state (class current or mixed). ``None`` for a non-current query, or when the
    recall returned nothing (precision of an empty list is undefined; current_recall still
    scores it as a miss)."""
    if not _is_current_query(labels):
        return None
    top = items[:k]
    return _share(top, labels, _CURRENT_LIKE) if top else None


def current_recall(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """current_recall@k — on current queries with required_active labels: the share of those
    labels present in the top k (a mixed observation counts)."""
    required = labels.query.required_active
    if not _is_current_query(labels) or not required:
        return None
    top = items[:k]
    return sum(_present(top, labels.docs_of(label), _CURRENT_LIKE, labels) for label in required) / len(required)


def _states_stale_things(labels: QueryLabels) -> bool:
    return bool(labels.query.forbidden_stale) or labels.query.stale_pattern is not None


def stale_contamination(items: Sequence[RankedItem], labels: QueryLabels, k: int, part: Part = "all") -> float | None:
    """stale_contamination@k — on current queries that have a superseded state: the share of
    the top-k results classed stale, i.e. superseded state presented as an answer to a
    question about now. ``part`` restricts the numerator to raw facts or to observations (the
    denominator stays every result, so raw + observation = combined). An empty result shows
    nothing stale: 0.0."""
    if not _is_current_query(labels) or not _states_stale_things(labels):
        return None
    return _share(items[:k], labels, frozenset({"stale"}), part)


def mixed_rate(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """mixed_rate@k — on current queries: the share of the top-k results that are mixed
    observations (current value and the value it replaced in one text). Reported beside
    precision and recall, which count these as current."""
    if not _is_current_query(labels):
        return None
    return _share(items[:k], labels, frozenset({"mixed"}))


def refuted_leakage(items: Sequence[RankedItem], labels: QueryLabels, k: int, part: Part = "all") -> float | None:
    """refuted_leakage@k — on every query whose bank holds a refuted claim, whatever the query
    kind (a claim shown to be false is never a valid answer, not even historically): the share
    of the top-k results classed refuted. ``part`` as for stale_contamination."""
    if not labels.query.forbidden_refuted:
        return None
    return _share(items[:k], labels, frozenset({"refuted"}), part)


def historical_recall(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """historical_recall@k — on historical and as_of queries: the share of allowed_historical
    labels present in the top k. An observation presents an earlier state when it classes
    stale (the retired observation itself) or mixed (a current one that kept the history)."""
    allowed = labels.query.allowed_historical
    if labels.query.kind == "current" or not allowed:
        return None
    top = items[:k]
    return sum(_present(top, labels.docs_of(label), _HISTORICAL_LIKE, labels) for label in allowed) / len(allowed)


def scope_accuracy(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """scope_accuracy@k — on scope queries (``scope_tags`` set): the share of the top-k results
    inside the query's scope. A result is inside when its tags include every scope tag and,
    for an observation, every document it cites is itself inside the scope — so an
    observation that absorbed another scope's evidence counts as outside. ``None`` when
    nothing was returned."""
    scope = frozenset(labels.query.scope_tags)
    if not scope:
        return None
    top = items[:k]
    if not top:
        return None

    def inside(item: RankedItem) -> bool:
        if not scope <= frozenset(item.tags):
            return False
        return not item.is_observation or all(labels.in_scope(doc) for doc in item.docs)

    return sum(inside(item) for item in top) / len(top)


def supersession_accuracy(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """supersession_accuracy@k — on current queries with both a successor (required_active)
    and a predecessor (forbidden_stale): 1.0 when every successor label is present in the top
    k AND no top-k result is classed stale, else 0.0. A mixed observation is the successor
    stating its history, not the predecessor."""
    query = labels.query
    if not _is_current_query(labels) or not query.required_active or not query.forbidden_stale:
        return None
    top = items[:k]
    successors = all(_present(top, labels.docs_of(label), _CURRENT_LIKE, labels) for label in query.required_active)
    predecessor = any(classify(item, labels) == "stale" for item in top)
    return float(successors and not predecessor)


def uncertainty_accuracy(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """uncertainty_accuracy@k — on queries over an unresolved conflict (two or more
    ``conflict_sides``, no trust differential configured): 1.0 when every side is present in
    the top k, else 0.0. A side is present through a raw fact of its documents, or through an
    observation that cites one of them and whose text states the side's value."""
    sides = labels.query.conflict_sides
    if len(sides) < 2:
        return None
    top = items[:k]

    def present(side_docs: frozenset[str], pattern: str | None) -> bool:
        for item in top:
            if not side_docs & frozenset(item.docs):
                continue
            if not item.is_observation or pattern is None or re.search(pattern, item.text, re.IGNORECASE):
                return True
        return False

    return float(all(present(frozenset(side.docs), side.pattern) for side in sides))


def independent_reinforcement_accuracy(items: Sequence[RankedItem], labels: QueryLabels, k: int) -> float | None:
    """independent_reinforcement_accuracy@k — on queries pairing a memory confirmed in spaced,
    independent sessions (``spaced``) with one repeated inside a single session
    (``contrast``) under an equal cue: 1.0 when the best-ranked result carrying the spaced
    memory ranks above the best-ranked one carrying the contrast, within the top k (absent
    ranks below everything). 0.0 when neither is in the top k."""
    query = labels.query
    if not query.spaced or not query.contrast:
        return None
    top = items[:k]
    spaced = frozenset(doc for label in query.spaced for doc in labels.docs_of(label))
    contrast = frozenset(doc for label in query.contrast for doc in labels.docs_of(label))

    def best(docs: frozenset[str]) -> float:
        return next((index for index, item in enumerate(top) if carries(item, docs, _ANY, labels)), math.inf)

    spaced_rank, contrast_rank = best(spaced), best(contrast)
    return float(spaced_rank < contrast_rank)


MetricFn = Callable[[Sequence[RankedItem], QueryLabels, int], float | None]

#: Every behavioural metric, by its report name. Order is report order.
METRICS: dict[str, MetricFn] = {
    "current_precision": current_precision,
    "current_recall": current_recall,
    "stale_contamination": stale_contamination,
    "stale_contamination_raw": lambda items, labels, k: stale_contamination(items, labels, k, "raw"),
    "stale_contamination_obs": lambda items, labels, k: stale_contamination(items, labels, k, "observation"),
    "mixed_rate": mixed_rate,
    "refuted_leakage": refuted_leakage,
    "refuted_leakage_raw": lambda items, labels, k: refuted_leakage(items, labels, k, "raw"),
    "refuted_leakage_obs": lambda items, labels, k: refuted_leakage(items, labels, k, "observation"),
    "historical_recall": historical_recall,
    "scope_accuracy": scope_accuracy,
    "supersession_accuracy": supersession_accuracy,
    "uncertainty_accuracy": uncertainty_accuracy,
    "independent_reinforcement_accuracy": independent_reinforcement_accuracy,
}


def score_query(items: Sequence[RankedItem], labels: QueryLabels) -> dict[str, float | None]:
    """Every metric at every k for one query, keyed ``name@k``."""
    return {f"{name}@{k}": fn(items, labels, k) for name, fn in METRICS.items() for k in K_VALUES}


@dataclass(frozen=True)
class MetricSummary:
    mean: float | None
    """Mean over the queries the metric applied to; ``None`` (N/A) when it applied to none."""
    n: int


def aggregate(per_query: Iterable[Mapping[str, float | None]]) -> dict[str, MetricSummary]:
    """Mean of each ``name@k`` over the queries where it applied."""
    values: dict[str, list[float]] = {f"{name}@{k}": [] for name in METRICS for k in K_VALUES}
    for scores in per_query:
        for key, value in scores.items():
            if value is not None:
                values.setdefault(key, []).append(value)
    return {
        key: MetricSummary(mean=(sum(found) / len(found)) if found else None, n=len(found))
        for key, found in values.items()
    }


@dataclass(frozen=True)
class StateConformance:
    """E-only claim-state conformance, read from GET memory ``lifecycle``.

    Every field is ``None`` (N/A, never 0) when the server reported no lifecycle state for any
    labelled document — an arm without the feature has nothing to conform.
    """

    conformance: float | None
    """Share of documents with an expected state whose raw facts all report exactly it."""
    retirement_recall: float | None
    """Share of documents expected superseded/refuted whose raw facts report that status."""
    false_retirement: float | None
    """Share of documents expected active whose raw facts report anything else."""
    n_expected: int
    n_reported: int


def state_conformance(
    expected: Mapping[str, TruthStatus], observed: Mapping[str, Sequence[str | None]]
) -> StateConformance:
    """Score reported claim states against a scenario's expectations.

    ``observed`` maps a document to the truth_status of each of its raw facts, ``None`` where
    the memory carried no ``lifecycle`` key. A document counts as reported when any of its raw
    facts reported a state; a document whose facts were not found at all counts as reported
    nothing.
    """
    reported = {doc: statuses for doc, statuses in observed.items() if any(s is not None for s in statuses)}
    if not reported:
        return StateConformance(None, None, None, n_expected=len(expected), n_reported=0)

    def holds(doc: str, status: str) -> bool:
        statuses = observed.get(doc, ())
        return bool(statuses) and all(s == status for s in statuses)

    docs = list(expected)
    retired = [doc for doc in docs if expected[doc] in ("superseded", "refuted")]
    active = [doc for doc in docs if expected[doc] == "active"]
    return StateConformance(
        conformance=sum(holds(doc, expected[doc]) for doc in docs) / len(docs) if docs else None,
        retirement_recall=sum(holds(doc, expected[doc]) for doc in retired) / len(retired) if retired else None,
        false_retirement=(
            sum(any(s not in (None, "active") for s in observed.get(doc, ())) for doc in active) / len(active)
            if active
            else None
        ),
        n_expected=len(docs),
        n_reported=len(reported),
    )
