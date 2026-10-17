"""Context-admission metrics (A28-A32): pure functions over what an arm admitted per query.

Vocabulary, per query ``q``:

- ``S`` — the admitted list: the memory labels the arm would inject, in order, after its cap.
- ``R`` — ``q.relevant``: memories that should be injected (may be empty).
- ``Acc`` — ``q.acceptable``: related memories that are neither credited nor penalised.
- ``Irr(S)`` — admitted memories in neither ``R`` nor ``Acc``. Every memory a query does not
  name as relevant or acceptable is irrelevant to it; ``q.hard_negatives`` is the subset that
  shares the task's vocabulary or technology, tracked on its own.
- An *abstention query* is one with ``R`` empty: injecting nothing is the right answer.

All rates are pooled (micro-averaged) over the queries they are defined on, so a query with
more admitted memories weighs more. A rate whose denominator is zero is ``None`` — "not
defined", never 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .model import Memory, Query

_TOKEN = re.compile(r"\w+|[^\w\s]")


def approx_tokens(text: str) -> int:
    """A fixed, dependency-free token count: words plus punctuation marks. Over this
    scenario's memory texts it counts 0.93x what tiktoken cl100k_base counts (5251 vs 5665,
    measured); used only to compare arms with each other."""
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class Admitted:
    """One query's admitted list."""

    query: Query
    labels: tuple[str, ...]
    """Admitted memory labels in injection order; an unlabelled memory appears as ``?<id>``."""


def _irrelevant(query: Query, label: str) -> bool:
    return label not in query.relevant and label not in query.acceptable


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def context_precision(rows: Iterable[Admitted]) -> float | None:
    """Relevant admitted / (relevant admitted + irrelevant admitted), pooled. Acceptable
    memories are excluded from both sides."""
    hit = miss = 0
    for row in rows:
        hit += sum(1 for label in row.labels if label in row.query.relevant)
        miss += sum(1 for label in row.labels if _irrelevant(row.query, label))
    return _ratio(hit, hit + miss)


def useful_memory_recall(rows: Iterable[Admitted]) -> float | None:
    """(context_recall) Relevant admitted / relevant, pooled over queries with ``R`` non-empty."""
    hit = total = 0
    for row in rows:
        if not row.query.relevant:
            continue
        admitted = set(row.labels)
        hit += sum(1 for label in row.query.relevant if label in admitted)
        total += len(row.query.relevant)
    return _ratio(hit, total)


def context_pollution_rate(rows: Iterable[Admitted]) -> float | None:
    """Irrelevant admitted / total admitted, pooled (acceptable memories count in the total)."""
    irrelevant = total = 0
    for row in rows:
        irrelevant += sum(1 for label in row.labels if _irrelevant(row.query, label))
        total += len(row.labels)
    return _ratio(irrelevant, total)


def polluted_queries_rate(rows: Iterable[Admitted]) -> float | None:
    """Fraction of queries whose admitted list contains at least one irrelevant memory."""
    rows = list(rows)
    polluted = sum(1 for row in rows if any(_irrelevant(row.query, label) for label in row.labels))
    return _ratio(polluted, len(rows))


def abstention_accuracy(rows: Iterable[Admitted]) -> float | None:
    """On abstention queries (``R`` empty): fraction whose admitted list is empty."""
    target = [row for row in rows if row.query.expected_abstain]
    return _ratio(sum(1 for row in target if not row.labels), len(target))


def false_abstention_rate(rows: Iterable[Admitted]) -> float | None:
    """On queries with ``R`` non-empty: fraction whose admitted list is empty."""
    target = [row for row in rows if not row.query.expected_abstain]
    return _ratio(sum(1 for row in target if not row.labels), len(target))


def false_positive_admission_rate(rows: Iterable[Admitted], bank_sizes: Mapping[str, int]) -> float | None:
    """Irrelevant admitted / irrelevant memories in the query's bank, pooled. The denominator
    is ``bank size − |R| − |Acc|``: every memory the query could wrongly admit."""
    admitted = negatives = 0
    for row in rows:
        admitted += sum(1 for label in row.labels if _irrelevant(row.query, label))
        negatives += bank_sizes[row.query.bank] - len(row.query.relevant) - len(row.query.acceptable)
    return _ratio(admitted, negatives)


def false_negative_admission_rate(rows: Iterable[Admitted]) -> float | None:
    """Relevant memories not admitted / relevant, pooled over queries with ``R`` non-empty
    (``1 − useful_memory_recall``)."""
    recall = useful_memory_recall(rows)
    return None if recall is None else 1.0 - recall


def hard_negative_admission_rate(rows: Iterable[Admitted]) -> float | None:
    """Admitted hard negatives / hard negatives, pooled over queries that name any."""
    admitted = total = 0
    for row in rows:
        chosen = set(row.labels)
        admitted += sum(1 for label in row.query.hard_negatives if label in chosen)
        total += len(row.query.hard_negatives)
    return _ratio(admitted, total)


def cross_scope_leakage(rows: Iterable[Admitted], memories: Mapping[str, Memory], k: int | None = None) -> float | None:
    """cross_scope_leakage@k: on project-scoped queries, admitted memories (within the first
    ``k``) that belong to another project / admitted memories (within the first ``k``), pooled.
    A memory with no project (the user's own) is never leakage."""
    leaked = total = 0
    for row in rows:
        if row.query.project is None:
            continue
        window = row.labels if k is None else row.labels[:k]
        for label in window:
            memory = memories.get(label)
            total += 1
            if memory is not None and memory.project not in (None, row.query.project):
                leaked += 1
    return _ratio(leaked, total)


def _status_admission_rate(rows: Iterable[Admitted], memories: Mapping[str, Memory], status: str) -> float | None:
    hits = total = 0
    for row in rows:
        if not row.query.present_state:
            continue
        for label in row.labels:
            total += 1
            memory = memories.get(label)
            if memory is not None and memory.status == status:
                hits += 1
    return _ratio(hits, total)


def stale_admission_rate(rows: Iterable[Admitted], memories: Mapping[str, Memory]) -> float | None:
    """On present-state queries (purpose ≠ historical_search): superseded admitted / admitted."""
    return _status_admission_rate(rows, memories, "superseded")


def refuted_admission_rate(rows: Iterable[Admitted], memories: Mapping[str, Memory]) -> float | None:
    """On present-state queries: refuted admitted / admitted."""
    return _status_admission_rate(rows, memories, "refuted")


def token_overhead(rows: Iterable[Admitted], tokens_of: Callable[[str], int]) -> float | None:
    """Mean over queries of the summed tokens of the admitted memories (``approx_tokens``)."""
    rows = list(rows)
    return _ratio(sum(sum(tokens_of(label) for label in row.labels) for row in rows), len(rows))


def mean_admitted(rows: Iterable[Admitted]) -> float | None:
    rows = list(rows)
    return _ratio(sum(len(row.labels) for row in rows), len(rows))


def f_beta(precision: float | None, recall: float | None, beta: float = 0.5) -> float | None:
    """F-beta of pooled precision and recall; beta < 1 favours precision (auto-context)."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100])."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


METRIC_NAMES = (
    "context_precision",
    "useful_memory_recall",
    "context_pollution_rate",
    "polluted_queries_rate",
    "abstention_accuracy",
    "false_abstention_rate",
    "false_positive_admission_rate",
    "false_negative_admission_rate",
    "hard_negative_admission_rate",
    "cross_scope_leakage@5",
    "cross_scope_leakage@10",
    "stale_admission_rate",
    "refuted_admission_rate",
    "token_overhead",
    "mean_admitted",
    "f0.5",
)


def aggregate(
    rows: Sequence[Admitted],
    *,
    memories: Mapping[str, Memory],
    bank_sizes: Mapping[str, int],
    tokens_of: Callable[[str], int],
) -> dict[str, float | None]:
    """Every metric above over one set of rows, plus ``n`` (queries) and ``n_abstain``."""
    precision = context_precision(rows)
    recall = useful_memory_recall(rows)
    return {
        "n": float(len(rows)),
        "n_abstain": float(sum(1 for row in rows if row.query.expected_abstain)),
        "context_precision": precision,
        "useful_memory_recall": recall,
        "context_pollution_rate": context_pollution_rate(rows),
        "polluted_queries_rate": polluted_queries_rate(rows),
        "abstention_accuracy": abstention_accuracy(rows),
        "false_abstention_rate": false_abstention_rate(rows),
        "false_positive_admission_rate": false_positive_admission_rate(rows, bank_sizes),
        "false_negative_admission_rate": false_negative_admission_rate(rows),
        "hard_negative_admission_rate": hard_negative_admission_rate(rows),
        "cross_scope_leakage@5": cross_scope_leakage(rows, memories, 5),
        "cross_scope_leakage@10": cross_scope_leakage(rows, memories, 10),
        "stale_admission_rate": stale_admission_rate(rows, memories),
        "refuted_admission_rate": refuted_admission_rate(rows, memories),
        "token_overhead": token_overhead(rows, tokens_of),
        "mean_admitted": mean_admitted(rows),
        "f0.5": f_beta(precision, recall),
    }
