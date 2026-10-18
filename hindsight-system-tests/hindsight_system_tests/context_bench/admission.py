"""Offline replay of the context-admission gate over recorded traces, the logistic fit of the
deterministic scorer's weights, and the dev selection rule.

A feature-server run with an admission variant records, per recall, every judged candidate's
features, scope, truth status, tokens and scores (:class:`~.runner.AdmissionRow`). Given those
and the stored embeddings, the gate's decision under another threshold or weight vector is a pure
function, mirrored here from the server's documented rules:

1. score: ``deterministic`` = sigmoid of the weighted feature sum (the relevance, scope and
   utility partial sums are accumulated in the order the trace lists the features, then added);
   ``jev`` = the decision model's probability; ``hybrid`` = ``jev_weight * p + (1 - jev_weight) *
   deterministic``. A candidate the decision model did not answer (or a recall whose decision-model
   request failed) falls back to the deterministic score;
2. threshold by purpose: ``auto_context`` uses ``threshold_auto``, lowered by ``reference_bonus``
   when the query carries a memory-reference phrase (clamped to [0, 1]); the other purposes use
   ``threshold_explicit``;
3. candidate gate, first rejection wins: refuted; superseded unless the task is historical; in
   strict mode a hard-scope mismatch; below the threshold (an uncertain claim needs the threshold
   plus ``uncertain_margin``);
4. strict mode only: greedy MMR redundancy suppression, the admission budget (tokens by value per
   token; when more than ``max_memories`` fit, by value), then readmission of a budget-dropped
   memory's suppressed twin.

Admitted memories keep recall order. :func:`verify_replay` compares a replay with the live
decisions of the same recall; a sweep built on the replay is only used once that comparison is
exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .runner import AdmissionDecisionRow, AdmissionRow, ResultRow

_RELEVANCE_TERMS = frozenset({"bias", "reranker", "semantic", "bm25", "arm_agreement", "rank"})
_SCOPE_PREFIXES = ("scope_match.", "scope_mismatch.")
#: Two recorded floats this close are the same number written by the same arithmetic.
SCORE_TOLERANCE = 1e-12

SCORERS = ("deterministic", "jev", "hybrid")
GATED_MODES = ("filter", "strict")


class ReplayError(Exception):
    """A recorded trace is not what the replay's model of the gate says it must be."""


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class DeterministicScore:
    task_relevance: float
    scope_score: float
    expected_utility: float
    admission_score: float


def deterministic_score(features: Mapping[str, float], weights: Mapping[str, float]) -> DeterministicScore:
    """``sigmoid(sum(weight * feature))`` and its three parts, each the sigmoid of its own sum."""
    relevance = scope = utility = 0.0
    for key, value in features.items():
        contribution = weights.get(key, 0.0) * value
        if key in _RELEVANCE_TERMS:
            relevance += contribution
        elif key.startswith(_SCOPE_PREFIXES):
            scope += contribution
        else:
            utility += contribution
    return DeterministicScore(
        task_relevance=sigmoid(relevance),
        scope_score=sigmoid(scope),
        expected_utility=sigmoid(utility),
        admission_score=sigmoid(relevance + scope + utility),
    )


@dataclass(frozen=True)
class GateSettings:
    """Everything the gate applies to one recall."""

    scorer: str
    mode: str
    weights: Mapping[str, float]
    threshold_auto: float
    threshold_explicit: float
    reference_bonus: float
    uncertain_margin: float
    jev_weight: float
    mmr_lambda: float
    dup_similarity: float
    max_memories: int
    max_tokens: int
    hard_scope_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {self.scorer!r}")
        if self.mode not in GATED_MODES:
            raise ValueError(f"mode must be one of {GATED_MODES}, got {self.mode!r}")

    def threshold(self, purpose: str | None, *, reference_signal: bool) -> float:
        if purpose == "auto_context":
            return min(1.0, max(0.0, self.threshold_auto - (self.reference_bonus if reference_signal else 0.0)))
        return self.threshold_explicit

    def with_thresholds(self, auto: float, explicit: float) -> GateSettings:
        return replace(self, threshold_auto=auto, threshold_explicit=explicit)


# --------------------------------------------------------------------------- recorded scores


def decision_model_probabilities(row: AdmissionRow, source: GateSettings) -> list[float | None]:
    """The decision model's probability per recorded candidate (None: not asked, not answered,
    or the request failed), recovered from a run whose scorer was ``source.scorer``.

    The trace reports, per candidate, the deterministic scope and utility parts, ``task_relevance``
    (the probability when the model answered, else the deterministic relevance part) and the
    admission score. Recomputing the deterministic score from the recorded features with the
    run's weights tells the two cases apart, and every recorded number is checked against it:
    a mismatch raises :class:`ReplayError` rather than feeding a sweep a wrong model of the run.
    """
    out: list[float | None] = []
    for decision in row.decisions:
        det = deterministic_score(decision.features, source.weights)
        _check(decision, "scope_score", decision.scope_score, det.scope_score)
        _check(decision, "expected_utility", decision.expected_utility, det.expected_utility)
        unanswered = _close(decision.task_relevance, det.task_relevance) and _close(
            decision.admission_score, det.admission_score
        )
        if source.scorer == "deterministic" or row.jev_error is not None or unanswered:
            _check(decision, "task_relevance", decision.task_relevance, det.task_relevance)
            _check(decision, "admission_score", decision.admission_score, det.admission_score)
            out.append(None)
            continue
        probability = decision.task_relevance
        expected = (
            probability
            if source.scorer == "jev"
            else source.jev_weight * probability + (1.0 - source.jev_weight) * det.admission_score
        )
        _check(decision, "admission_score", decision.admission_score, expected)
        out.append(probability)
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= SCORE_TOLERANCE


def _check(decision: AdmissionDecisionRow, name: str, recorded: float, recomputed: float) -> None:
    if not _close(recorded, recomputed):
        raise ReplayError(f"{decision.label}: recorded {name} {recorded!r} but the replay computes {recomputed!r}")


# --------------------------------------------------------------------------- the gate


@dataclass
class Decision:
    """One candidate's replayed decision (mutable while the strict steps run)."""

    label: str
    score: float
    truth_status: str
    scope: Mapping[str, str]
    tokens: int
    final: float
    admitted: bool = False
    reason: str = "below_threshold"
    duplicates: int = 0


def candidate_scores(row: AdmissionRow, settings: GateSettings, probabilities: Sequence[float | None]) -> list[float]:
    """Each recorded candidate's admission score under ``settings`` (thresholds play no part)."""
    if len(probabilities) != len(row.decisions):
        raise ValueError("one probability (or None) per recorded candidate")
    out: list[float] = []
    for decision, probability in zip(row.decisions, probabilities):
        det = deterministic_score(decision.features, settings.weights).admission_score
        if settings.scorer == "deterministic" or probability is None:
            out.append(det)
        elif settings.scorer == "jev":
            out.append(probability)
        else:
            out.append(settings.jev_weight * probability + (1.0 - settings.jev_weight) * det)
    return out


def replay(
    row: AdmissionRow,
    settings: GateSettings,
    *,
    scores: Sequence[float],
    similarity: Callable[[str, str], float],
) -> list[Decision]:
    """The gate's decisions for one recorded recall under ``settings``, in recall order.

    ``scores`` come from :func:`candidate_scores`; ``similarity`` from :func:`similarity_function`
    over the recall's candidates (only strict mode reads it).
    """
    if len(scores) != len(row.decisions):
        raise ValueError("one score per recorded candidate")
    threshold = settings.threshold(row.purpose, reference_signal=row.reference_signal is not None)
    historical = bool(row.historical)
    strict = settings.mode == "strict"
    decisions: list[Decision] = []
    for recorded, score in zip(row.decisions, scores):
        decision = Decision(
            label=recorded.label,
            score=score,
            truth_status=recorded.truth_status,
            scope=recorded.scope,
            tokens=recorded.tokens,
            final=recorded.base_retrieval_score,
        )
        decision.reason = _rejection(decision, settings, threshold=threshold, historical=historical, strict=strict)
        decision.admitted = decision.reason == "admitted"
        decisions.append(decision)
    if strict:
        # The server reads embeddings only when two or more candidates survive the gate; with
        # fewer there is no pair to compare, so the similarity function is never consulted.
        values, twins = suppress_redundant(
            decisions, similarity, mmr_lambda=settings.mmr_lambda, dup_similarity=settings.dup_similarity
        )
        apply_budget(decisions, values, max_memories=settings.max_memories, max_tokens=settings.max_tokens)
        readmit_twins(
            decisions,
            values,
            twins,
            similarity,
            dup_similarity=settings.dup_similarity,
            max_memories=settings.max_memories,
            max_tokens=settings.max_tokens,
        )
    return decisions


def _rejection(decision: Decision, settings: GateSettings, *, threshold: float, historical: bool, strict: bool) -> str:
    if decision.truth_status == "refuted":
        return "refuted"
    if decision.truth_status == "superseded" and not historical:
        return "superseded"
    if strict and any(decision.scope.get(key) == "mismatch" for key in settings.hard_scope_keys):
        return "wrong_scope"
    margin = settings.uncertain_margin if decision.truth_status == "uncertain" else 0.0
    return "admitted" if decision.score >= threshold + margin else "below_threshold"


def _unit(vector: Sequence[float] | None) -> list[float] | None:
    if not vector or not all(math.isfinite(x) for x in vector):
        return None
    norm = math.sqrt(math.fsum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else None


def similarity_function(
    labels: Sequence[str], embeddings: Mapping[str, Sequence[float]]
) -> Callable[[str, str], float]:
    """Cosine between two labels' embeddings (cached); 0.0 when either is missing, zero,
    non-finite, or the two differ in length (nothing is known about the pair)."""
    units = {label: _unit(embeddings.get(label)) for label in labels}
    cache: dict[tuple[str, str], float] = {}

    def similarity(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        if key not in cache:
            ua, ub = units.get(a), units.get(b)
            cache[key] = (
                0.0 if ua is None or ub is None or len(ua) != len(ub) else math.fsum(x * y for x, y in zip(ua, ub))
            )
        return cache[key]

    return similarity


def suppress_redundant(
    decisions: Sequence[Decision],
    similarity: Callable[[str, str], float],
    *,
    mmr_lambda: float,
    dup_similarity: float,
) -> tuple[dict[str, float], dict[str, list[Decision]]]:
    """Greedy MMR over the admitted decisions (best ``lambda * score - (1 - lambda) * max_sim``
    first, ties to the higher score); one whose closest kept memory reaches ``dup_similarity`` is
    rejected as redundant. Returns each kept memory's MMR value and the twins each kept memory
    suppressed. Mutates ``decisions``."""
    remaining = sorted((d for d in decisions if d.admitted), key=lambda d: -d.score)
    n = len(remaining)
    max_sim = [0.0] * n
    nearest = [-1] * n
    alive = [True] * n
    values: dict[str, float] = {}
    twins: dict[str, list[Decision]] = {}
    for _ in range(n):
        best = -1
        best_value = -math.inf
        for i in range(n):
            if not alive[i]:
                continue
            value = mmr_lambda * remaining[i].score - (1.0 - mmr_lambda) * max_sim[i]
            if value > best_value:
                best, best_value = i, value
        alive[best] = False
        chosen = remaining[best]
        if nearest[best] >= 0 and max_sim[best] >= dup_similarity:
            canonical = remaining[nearest[best]]
            chosen.admitted = False
            chosen.reason = "redundant"
            canonical.duplicates += 1
            twins.setdefault(canonical.label, []).append(chosen)
            continue
        values[chosen.label] = best_value
        for i in range(n):
            if alive[i]:
                sim = similarity(chosen.label, remaining[i].label)
                if sim > max_sim[i]:
                    max_sim[i] = sim
                    nearest[i] = best
    return values, twins


def _fill(ordered: Sequence[Decision], *, max_tokens: int, max_memories: int | None) -> list[Decision]:
    chosen: list[Decision] = []
    used = 0
    for decision in ordered:
        if max_memories is not None and len(chosen) >= max_memories:
            break
        if used + decision.tokens <= max_tokens:
            chosen.append(decision)
            used += decision.tokens
    return chosen


def apply_budget(
    decisions: Sequence[Decision], values: Mapping[str, float], *, max_memories: int, max_tokens: int
) -> None:
    """Fill the token budget by value per token; when more than ``max_memories`` fit, take them by
    value (ties by value per token) instead. The rest are rejected as ``budget``."""
    admitted = [d for d in decisions if d.admitted]

    def value(d: Decision) -> float:
        return max(values.get(d.label, d.score), 0.0)

    def density(d: Decision) -> float:
        return (value(d) + 1e-9) / max(d.tokens, 1)

    chosen = _fill(sorted(admitted, key=lambda d: -density(d)), max_tokens=max_tokens, max_memories=None)
    if len(chosen) > max_memories:
        chosen = _fill(
            sorted(admitted, key=lambda d: (-value(d), -density(d))), max_tokens=max_tokens, max_memories=max_memories
        )
    kept = {id(d) for d in chosen}
    for decision in admitted:
        if id(decision) not in kept:
            decision.admitted = False
            decision.reason = "budget"


def readmit_twins(
    decisions: Sequence[Decision],
    values: Mapping[str, float],
    twins: Mapping[str, list[Decision]],
    similarity: Callable[[str, str], float],
    *,
    dup_similarity: float,
    max_memories: int,
    max_tokens: int,
) -> None:
    """A budget-dropped memory's best suppressed twin that fits the count and tokens left, and is
    not a near-duplicate of an admitted memory, takes its place (most valuable canonical first)."""
    admitted = [d for d in decisions if d.admitted]
    used = sum(d.tokens for d in admitted)
    dropped = [d for d in decisions if d.reason == "budget" and twins.get(d.label)]
    dropped.sort(key=lambda d: -values.get(d.label, d.score))
    for canonical in dropped:
        if len(admitted) >= max_memories:
            return
        for twin in sorted(twins[canonical.label], key=lambda d: -d.score):
            if used + twin.tokens > max_tokens:
                continue
            if any(similarity(twin.label, d.label) >= dup_similarity for d in admitted):
                continue
            twin.admitted = True
            twin.reason = "admitted"
            twin.duplicates += canonical.duplicates - 1
            canonical.duplicates = 0
            admitted.append(twin)
            used += twin.tokens
            break


def admitted_results(decisions: Sequence[Decision]) -> list[ResultRow]:
    """The admitted memories as result rows, in recall order, carrying the recall score the
    report's tie canonicalisation reads."""
    return [
        ResultRow(label=d.label, server_rank=rank, final=d.final)
        for rank, d in enumerate((d for d in decisions if d.admitted), start=1)
    ]


# --------------------------------------------------------------------------- live vs replay


@dataclass
class ReplayCheck:
    """How a replay compares with the live decisions of the same recalls."""

    queries: int = 0
    exact: int = 0
    candidates: int = 0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    #: Recalls on which a strict-only step changed a decision (it was exercised, not just idle).
    redundant_queries: int = 0
    budget_queries: int = 0
    wrong_scope_queries: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def all_exact(self) -> bool:
        return self.queries > 0 and self.exact == self.queries


def verify_replay(
    row_results: Sequence[ResultRow],
    row: AdmissionRow,
    decisions: Sequence[Decision],
    check: ReplayCheck,
    *,
    query_id: str,
) -> None:
    """Compare one recall's replayed decisions with the live trace (admitted flag, reason and
    duplicate count per candidate) and the replayed admitted list with the live results."""
    check.queries += 1
    check.candidates += len(decisions)
    problems: list[str] = []
    if len(decisions) != len(row.decisions):
        problems.append(f"{len(decisions)} replayed candidates for {len(row.decisions)} recorded")
    for replayed, live in zip(decisions, row.decisions):
        if replayed.label != live.label:
            problems.append(f"candidate order: {replayed.label} vs {live.label}")
            continue
        if (replayed.admitted, replayed.reason, replayed.duplicates) != (live.admitted, live.reason, live.duplicates):
            problems.append(
                f"{live.label}: live {live.reason}/{live.duplicates} replay {replayed.reason}/{replayed.duplicates}"
            )
    replayed_labels = [r.label for r in admitted_results(decisions)]
    live_labels = [r.label for r in row_results]
    if replayed_labels != live_labels:
        problems.append(f"results: live {live_labels} replay {replayed_labels}")
    reasons = {d.reason for d in row.decisions}
    check.redundant_queries += "redundant" in reasons
    check.budget_queries += "budget" in reasons
    check.wrong_scope_queries += "wrong_scope" in reasons
    for d in row.decisions:
        check.reasons[d.reason] = check.reasons.get(d.reason, 0) + 1
    if problems:
        check.mismatches.append({"query": query_id, "problems": problems})
    else:
        check.exact += 1


# --------------------------------------------------------------------------- logistic fit


@dataclass(frozen=True)
class LogisticFit:
    """An L2-regularised logistic regression on standardized features, reported on the raw scale
    (``sigmoid(bias + sum(weights[k] * x[k]))``) so the server can use the weights as they are."""

    weights: dict[str, float]
    """Raw-scale weights of the fitted features, plus ``bias``."""
    fitted: tuple[str, ...]
    constant: tuple[str, ...]
    """Features with one non-zero value on every sample: collinear with the bias, weight 0."""
    absent: tuple[str, ...]
    """Features that are 0 on every sample: the data says nothing about them; not fitted."""
    l2: float
    iterations: int
    samples: int
    positives: int
    log_loss: float
    #: Each class weighted by ``n / (2 * n_class)``: the fit then models a 50% base rate, which
    #: moves the intercept (and the operating region of a threshold) without other prior.
    class_balanced: bool = False

    def prior_shifted(self) -> dict[str, float]:
        """The weights with the intercept moved from the sample's base rate to a 50% prior
        (``bias + log(negatives / positives)``): the same ranking as the fit, its probabilities
        read as if the classes were balanced."""
        out = dict(self.weights)
        out["bias"] = self.weights["bias"] + math.log((self.samples - self.positives) / self.positives)
        return out


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the systems here are a few dozen wide)."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        out[r] = (a[r][n] - sum(a[r][c] * out[c] for c in range(r + 1, n))) / a[r][r]
    return out


def _penalized_loss(
    z: Sequence[Sequence[float]], y: Sequence[int], s: Sequence[float], beta: Sequence[float], l2: float
) -> float:
    total = 0.0
    for row, label, weight in zip(z, y, s):
        margin = sum(b * x for b, x in zip(beta, row))
        # log(1 + exp(-m)) for y = 1 and log(1 + exp(m)) for y = 0, computed stably.
        signed = margin if label else -margin
        total += weight * (math.log1p(math.exp(-signed)) if signed > 0 else -signed + math.log1p(math.exp(signed)))
    return total + 0.5 * l2 * sum(b * b for b in beta[1:])


def fit_logistic(
    samples: Sequence[tuple[Mapping[str, float], int]],
    *,
    l2: float = 1.0,
    class_balanced: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> LogisticFit:
    """Fit ``P(relevant) = sigmoid(bias + w . x)`` by Newton's method with an L2 penalty
    ``l2 / 2 * |w|^2`` on the standardized weights (not the bias), the loss summed over samples
    (each weighted ``n / (2 * n_class)`` when ``class_balanced``, so the weights still sum to n).

    ``bias`` in a sample is ignored (it is the intercept). A feature missing from a sample is 0.
    """
    if not samples:
        raise ValueError("no samples")
    labels = [int(y) for _, y in samples]
    if not 0 < sum(labels) < len(labels):
        raise ValueError("both classes are needed")
    names = sorted({key for features, _ in samples for key in features if key != "bias"})
    columns = {name: [float(features.get(name, 0.0)) for features, _ in samples] for name in names}
    fitted: list[str] = []
    constant: list[str] = []
    absent: list[str] = []
    for name in names:
        values = columns[name]
        if all(v == values[0] for v in values):
            (absent if values[0] == 0.0 else constant).append(name)
        else:
            fitted.append(name)
    n = len(samples)
    positives = sum(labels)
    sample_weights = (
        [n / (2.0 * (positives if label else n - positives)) for label in labels] if class_balanced else [1.0] * n
    )
    means = {name: math.fsum(columns[name]) / n for name in fitted}
    stds = {name: math.sqrt(math.fsum((v - means[name]) ** 2 for v in columns[name]) / n) for name in fitted}
    z = [[1.0] + [(columns[name][i] - means[name]) / stds[name] for name in fitted] for i in range(n)]
    width = len(fitted) + 1
    beta = [0.0] * width
    loss = _penalized_loss(z, labels, sample_weights, beta, l2)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]
        for row, label, weight in zip(z, labels, sample_weights):
            p = sigmoid(sum(b * x for b, x in zip(beta, row)))
            residual = weight * (p - label)
            w = weight * p * (1.0 - p)
            for j in range(width):
                gradient[j] += residual * row[j]
                wj = w * row[j]
                for k in range(j, width):
                    hessian[j][k] += wj * row[k]
        for j in range(1, width):
            gradient[j] += l2 * beta[j]
            hessian[j][j] += l2
        for j in range(width):
            for k in range(j):
                hessian[j][k] = hessian[k][j]
        step = _solve(hessian, gradient)
        scale = 1.0
        while True:
            candidate = [b - scale * s for b, s in zip(beta, step)]
            candidate_loss = _penalized_loss(z, labels, sample_weights, candidate, l2)
            if candidate_loss <= loss + 1e-12 or scale < 1e-6:
                break
            scale /= 2.0
        beta, loss = candidate, candidate_loss
        if max(abs(scale * s) for s in step) < tolerance:
            break
    weights = {name: beta[i + 1] / stds[name] for i, name in enumerate(fitted)}
    weights["bias"] = beta[0] - sum(beta[i + 1] * means[name] / stds[name] for i, name in enumerate(fitted))
    for name in constant:
        weights[name] = 0.0
    return LogisticFit(
        weights=weights,
        fitted=tuple(fitted),
        constant=tuple(constant),
        absent=tuple(absent),
        l2=l2,
        iterations=iterations,
        samples=n,
        positives=positives,
        log_loss=loss / n,
        class_balanced=class_balanced,
    )


# --------------------------------------------------------------------------- selection


#: The pre-registered dev threshold grid, 0.30 to 0.90 by 0.05.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))
RECALL_FLOOR = 0.85


@dataclass(frozen=True)
class SettingResult:
    """One candidate setting of an arm and its dev metrics."""

    weights_name: str
    threshold_auto: float
    threshold_explicit: float
    metrics: Mapping[str, float | None]


def _ordering(result: SettingResult, feasible: bool) -> tuple[float, ...]:
    m = result.metrics
    recall = m.get("useful_memory_recall") or 0.0
    pollution = m.get("context_pollution_rate") or 0.0
    abstention = m.get("abstention_accuracy") or 0.0
    tail = (result.threshold_auto, result.threshold_explicit)
    if feasible:
        # Lower pollution first; then higher abstention accuracy, recall, thresholds.
        return (-pollution, abstention, recall, *tail)
    return (recall - pollution, abstention, recall, *tail)


def select(results: Sequence[SettingResult], *, recall_floor: float = RECALL_FLOOR) -> SettingResult:
    """The dev selection rule. Pre-registered: the lowest context_pollution_rate among settings with
    useful_memory_recall >= ``recall_floor``, ties by higher abstention_accuracy; with no such
    setting, the highest useful_memory_recall - context_pollution_rate. Not pre-registered: remaining
    ties go to higher recall, then the higher auto and explicit thresholds (the more conservative
    gate), so the choice is deterministic. An undefined pollution (nothing admitted) counts as 0."""
    if not results:
        raise ValueError("nothing to select from")
    feasible = [r for r in results if (r.metrics.get("useful_memory_recall") or 0.0) >= recall_floor]
    pool = feasible or list(results)
    return max(pool, key=lambda r: _ordering(r, bool(feasible)))
