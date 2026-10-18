"""The gate: currentness, hard scope, threshold, then redundancy (MMR) and the admission budget.

Pure: every input is passed in (scores, embeddings, token counts), so the whole decision for a
recall can be replayed and tested without a database or a model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .models import ContextAdmissionDecision, Reason, ScopeMatch


def currentness_rejection(truth_status: str, *, historical: bool) -> Reason | None:
    """The currentness rule for a present-state task, or None when the claim may be admitted.

    Refuted is never admitted (it was wrong). Superseded is admitted only when the task asks about
    history. Uncertain and active pass here; uncertain pays a higher threshold (:func:`passes_threshold`).
    """
    if truth_status == "refuted":
        return Reason.REFUTED
    if truth_status == "superseded" and not historical:
        return Reason.SUPERSEDED
    return None


def hard_scope_rejection(scope: Mapping[str, ScopeMatch], hard_scope_keys: Sequence[str]) -> Reason | None:
    """Reject when the request and the memory both name a hard-scope key and disagree on it."""
    if any(scope.get(key) is ScopeMatch.MISMATCH for key in hard_scope_keys):
        return Reason.WRONG_SCOPE
    return None


def passes_threshold(score: float, *, threshold: float, uncertain: bool, uncertain_margin: float) -> bool:
    """Inclusive; an uncertain claim must clear the threshold by ``uncertain_margin``."""
    return score >= threshold + (uncertain_margin if uncertain else 0.0)


def similarity_matrix(ids: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Pairwise cosine similarities of ``ids``, computed once in one matrix product per dimension.

    A pair is 0.0 when either side has no stored embedding, a zero (or non-finite) vector, or the
    two embeddings differ in length: nothing is known about it.
    """
    n = len(ids)
    sims = np.zeros((n, n))
    by_dim: dict[int, list[int]] = {}
    for index, memory_id in enumerate(ids):
        vector = embeddings.get(memory_id)
        if vector is not None and len(vector) > 0:
            by_dim.setdefault(len(vector), []).append(index)
    for indices in by_dim.values():
        matrix = np.asarray([embeddings[ids[i]] for i in indices], dtype=float)
        matrix[~np.isfinite(matrix).all(axis=1)] = 0.0
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0
        matrix[nonzero] /= norms[nonzero][:, None]
        sims[np.ix_(indices, indices)] = matrix @ matrix.T
    return sims


@dataclass
class Redundancy:
    """What redundancy suppression decided, kept for the budget step that follows it."""

    #: Each kept memory's MMR value at the moment it was kept, which the budget orders by.
    values: dict[str, float]
    #: The near-duplicates each kept memory suppressed (canonical id -> twins, suppression order).
    twins: dict[str, list[ContextAdmissionDecision]]
    dup_similarity: float
    #: Memory id -> its row in ``similarities`` (the pairwise cosines the pass used).
    positions: dict[str, int] = field(default_factory=dict)
    similarities: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)

    def similarity(self, a: str, b: str) -> float:
        i, j = self.positions.get(a), self.positions.get(b)
        return float(self.similarities[i, j]) if i is not None and j is not None else 0.0


def suppress_redundant(
    decisions: Sequence[ContextAdmissionDecision],
    embeddings: Mapping[str, Sequence[float]],
    *,
    mmr_lambda: float,
    dup_similarity: float,
) -> Redundancy:
    """Greedy MMR over the admitted decisions; near-duplicates are rejected as redundant.

    Candidates are taken best-first by ``lambda * admission_score - (1 - lambda) * max_sim`` where
    ``max_sim`` is the cosine to the closest one already kept. One whose ``max_sim`` reaches
    ``dup_similarity`` is rejected (``redundant``) and the kept memory it duplicates counts it in
    ``duplicates``. A candidate without a stored embedding is never a duplicate. Each remaining
    candidate's ``max_sim`` is updated only against the memory just kept, so the whole pass is
    quadratic in the candidates, on similarities computed once. Mutates ``decisions``.
    """
    remaining = sorted((d for d in decisions if d.admitted), key=lambda d: -d.admission_score)
    ids = [d.memory_id for d in remaining]
    sims = similarity_matrix(ids, embeddings)
    n = len(remaining)
    scores = np.asarray([d.admission_score for d in remaining], dtype=float)
    max_sim = np.zeros(n)
    nearest = np.full(n, -1)
    alive = np.ones(n, dtype=bool)
    values: dict[str, float] = {}
    twins: dict[str, list[ContextAdmissionDecision]] = {}
    for _ in range(n):
        mmr = np.where(alive, mmr_lambda * scores - (1.0 - mmr_lambda) * max_sim, -np.inf)
        # argmax returns the first maximum: ties go to the higher score, as the sort put it first.
        best = int(np.argmax(mmr))
        alive[best] = False
        chosen = remaining[best]
        canonical_index = int(nearest[best])
        if canonical_index >= 0 and max_sim[best] >= dup_similarity:
            canonical = remaining[canonical_index]
            chosen.admitted = False
            chosen.reason = Reason.REDUNDANT
            canonical.duplicates += 1
            twins.setdefault(canonical.memory_id, []).append(chosen)
            continue
        values[chosen.memory_id] = float(mmr[best])
        row = sims[best]
        closer = alive & (row > max_sim)
        max_sim[closer] = row[closer]
        nearest[closer] = best
    return Redundancy(
        values=values,
        twins=twins,
        dup_similarity=dup_similarity,
        positions={memory_id: i for i, memory_id in enumerate(ids)},
        similarities=sims,
    )


def _fill(
    ordered: Sequence[ContextAdmissionDecision], *, max_tokens: int, max_memories: int | None
) -> list[ContextAdmissionDecision]:
    """Take ``ordered`` greedily, skipping a memory that does not fit the tokens left."""
    chosen: list[ContextAdmissionDecision] = []
    used = 0
    for decision in ordered:
        if max_memories is not None and len(chosen) >= max_memories:
            break
        if used + decision.tokens <= max_tokens:
            chosen.append(decision)
            used += decision.tokens
    return chosen


def apply_budget(
    decisions: Sequence[ContextAdmissionDecision],
    values: Mapping[str, float],
    *,
    max_memories: int,
    max_tokens: int,
) -> None:
    """Admit within ``max_tokens`` and ``max_memories``; the rest are rejected as ``budget``.

    The token budget is filled by value per token, skipping a memory that does not fit the tokens
    left so a smaller one after it may still fit. When that fill holds more than ``max_memories``,
    the count is what binds, and the memories are taken by value instead (ties by value per
    token), still skipping one that does not fit: many short weak memories must not push out the
    most valuable one. Mutates ``decisions``.
    """
    admitted = [d for d in decisions if d.admitted]

    def value(d: ContextAdmissionDecision) -> float:
        return max(values.get(d.memory_id, d.admission_score), 0.0)

    def density(d: ContextAdmissionDecision) -> float:
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
            decision.reason = Reason.BUDGET


def readmit_twins(
    decisions: Sequence[ContextAdmissionDecision],
    redundancy: Redundancy,
    *,
    max_memories: int,
    max_tokens: int,
) -> None:
    """Give a budget-rejected memory's suppressed twin the place it left, when the twin fits.

    Redundancy runs before the budget, so a long canonical memory can be dropped as ``budget``
    after its short near-duplicate was dropped as ``redundant``, losing both copies. Here, for each
    such canonical (most valuable first), its best twin that fits the count and tokens left, and is
    not itself a near-duplicate of an admitted memory, is admitted; the canonical's other twins
    are then counted on it. Mutates ``decisions``.
    """
    admitted = [d for d in decisions if d.admitted]
    used = sum(d.tokens for d in admitted)
    dropped = [d for d in decisions if d.reason is Reason.BUDGET and redundancy.twins.get(d.memory_id)]
    dropped.sort(key=lambda d: -redundancy.values.get(d.memory_id, d.admission_score))
    for canonical in dropped:
        if len(admitted) >= max_memories:
            return
        for twin in sorted(redundancy.twins[canonical.memory_id], key=lambda d: -d.admission_score):
            if used + twin.tokens > max_tokens:
                continue
            if any(redundancy.similarity(twin.memory_id, d.memory_id) >= redundancy.dup_similarity for d in admitted):
                continue
            twin.admitted = True
            twin.reason = Reason.ADMITTED
            twin.duplicates += canonical.duplicates - 1
            canonical.duplicates = 0
            admitted.append(twin)
            used += twin.tokens
            break


@dataclass(frozen=True)
class GateRules:
    """What the gate applies on one recall."""

    threshold: float
    historical: bool
    uncertain_margin: float
    #: Strict (and observe, which simulates strict): hard scope, redundancy and budget too.
    full: bool
    hard_scope_keys: tuple[str, ...]


def gate(decision: ContextAdmissionDecision, rules: GateRules) -> None:
    """Set ``admitted`` and ``reason`` from the candidate-level rules, first rejection wins:
    currentness, then hard scope (full gate only), then the threshold."""
    rejection = currentness_rejection(decision.truth_status, historical=rules.historical)
    if rejection is None and rules.full:
        rejection = hard_scope_rejection(decision.scope, rules.hard_scope_keys)
    if rejection is None and not passes_threshold(
        decision.admission_score,
        threshold=rules.threshold,
        uncertain=decision.uncertain,
        uncertain_margin=rules.uncertain_margin,
    ):
        rejection = Reason.BELOW_THRESHOLD
    decision.admitted = rejection is None
    decision.reason = rejection or Reason.ADMITTED
