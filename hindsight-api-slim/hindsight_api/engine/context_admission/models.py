"""The context-admission vocabulary: modes, purposes, scorers, decisions and outcomes."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class AdmissionMode(str, Enum):
    """What the gate does on a bank's external recalls.

    ``off``: nothing runs. ``observe``: decisions and the would-be admitted set are computed and
    reported (trace, metrics), the results are untouched. ``filter``: rejected candidates are
    dropped and the result may be empty. ``strict``: filter plus the hard scope gate, redundancy
    suppression and the admission budget.
    """

    OFF = "off"
    OBSERVE = "observe"
    FILTER = "filter"
    STRICT = "strict"

    @property
    def changes_results(self) -> bool:
        return self in (AdmissionMode.FILTER, AdmissionMode.STRICT)


class Purpose(str, Enum):
    """Why the caller recalls, which sets the trade-off the gate makes.

    ``auto_context``: memories injected into an agent's context nobody asked for (precision first).
    ``explicit_search``: someone asked for memories (recall first). ``historical_search``: the
    caller wants how things were (superseded memories are admissible, and the claim view is
    historical unless the request names one).
    """

    AUTO_CONTEXT = "auto_context"
    EXPLICIT_SEARCH = "explicit_search"
    HISTORICAL_SEARCH = "historical_search"


PURPOSES: tuple[str, ...] = tuple(p.value for p in Purpose)


class ScorerKind(str, Enum):
    """``deterministic``: the logistic feature score. ``jev``: one decision-model probability per
    candidate, asked in one batch. ``hybrid``: the two blended. The gates (currentness, hard scope)
    are deterministic under every scorer."""

    DETERMINISTIC = "deterministic"
    JEV = "jev"
    HYBRID = "hybrid"


class ScopeMatch(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class Reason(str, Enum):
    """Why a candidate was admitted or rejected; the rejection values are the abstention counts."""

    ADMITTED = "admitted"
    WRONG_SCOPE = "wrong_scope"
    BELOW_THRESHOLD = "below_threshold"
    SUPERSEDED = "superseded"
    REFUTED = "refuted"
    REDUNDANT = "redundant"
    BUDGET = "budget"


REJECTION_REASONS: tuple[str, ...] = tuple(r.value for r in Reason if r is not Reason.ADMITTED)

#: The keys of a request's task context. The scope keys among them are compared with memory
#: metadata; ``task_intent`` and ``workspace`` reach the decision model only.
TASK_CONTEXT_KEYS: tuple[str, ...] = ("repository", "branch", "module", "environment", "task_intent", "workspace")


@dataclass(frozen=True)
class CandidateFeatures:
    """What the gate knows about one candidate, all of it available without a model call."""

    memory_id: str
    rank: int
    base_retrieval_score: float
    reranker: float
    semantic: float | None
    bm25: bool
    arms: int
    scope: Mapping[str, ScopeMatch]
    truth_status: str
    retrievability: float | None
    importance: float | None
    fact_type: str
    tokens: int
    #: The recall's reranker scores by rank (a passthrough, or a provider whose scores are
    #: positions), so ``reranker`` says nothing about how relevant a candidate is.
    rank_based_relevance: bool = False

    def vector(self) -> dict[str, float]:
        """The deterministic scorer's inputs, keyed as ``context_admission_weights`` names them.

        When relevance is rank-based, the ``reranker`` slot carries the semantic similarity instead
        (the one absolute relevance signal left, so the hand-set threshold keeps its meaning) and
        ``rank`` is left out: read as relevance, both would admit the top of any pool.
        """
        semantic = self.semantic if self.semantic is not None else 0.0
        out = {
            "bias": 1.0,
            "reranker": semantic if self.rank_based_relevance else self.reranker,
            "semantic": semantic,
            "bm25": 1.0 if self.bm25 else 0.0,
            "arm_agreement": max(0, self.arms - 1) / 3.0,
            "rank": 1.0 / (1.0 + self.rank),
            "retrievability": self.retrievability - 0.5 if self.retrievability is not None else 0.0,
            "importance": self.importance - 0.5 if self.importance is not None else 0.0,
            "uncertain": 1.0 if self.truth_status == "uncertain" else 0.0,
            "token_cost": self.tokens / 100.0,
            f"fact_type.{self.fact_type}": 1.0,
        }
        if self.rank_based_relevance:
            del out["rank"]
        for key, match in self.scope.items():
            if match is ScopeMatch.MATCH:
                out[f"scope_match.{key}"] = 1.0
            elif match is ScopeMatch.MISMATCH:
                out[f"scope_mismatch.{key}"] = 1.0
        return out


@dataclass
class ContextAdmissionDecision:
    """One candidate's admission decision, inspectable and benchmarkable.

    ``task_relevance``, ``scope_score`` and ``expected_utility`` are the score's three parts on
    [0, 1] (0.5 = neutral for scope and utility); under the jev scorer ``task_relevance`` is the
    decision model's probability. ``admission_score`` is what the threshold is compared with.
    """

    memory_id: str
    base_retrieval_score: float
    scope_score: float
    task_relevance: float
    expected_utility: float
    truth_status: str
    retrievability: float | None
    admission_score: float
    admitted: bool
    reason: Reason
    scope: Mapping[str, ScopeMatch] = field(default_factory=dict)
    features: Mapping[str, float] = field(default_factory=dict)
    tokens: int = 0
    duplicates: int = 0
    uncertain: bool = False


@dataclass
class ContextAdmissionOutcome:
    """The gate's answer for one recall: the admitted ids (recall order) and every decision."""

    admitted_ids: list[str]
    decisions: list[ContextAdmissionDecision]
    threshold: float
    reference_signal: str | None
    scorer: ScorerKind
    jev_error: str | None = None
    embeddings_error: str | None = None
    #: What the deterministic ``reranker`` feature read: ``reranker`` (the cross-encoder's score)
    #: or ``semantic`` (the reranker scored by rank, see :class:`CandidateFeatures`).
    relevance_source: str = "reranker"
    #: Why observations' scopes could not be derived from their source facts (they read unknown).
    scope_error: str | None = None

    @property
    def abstained(self) -> bool:
        """Candidates were judged and none was admitted: the legitimate empty answer."""
        return bool(self.decisions) and not self.admitted_ids

    def reason_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(REJECTION_REASONS, 0)
        for decision in self.decisions:
            if decision.reason is not Reason.ADMITTED:
                counts[decision.reason.value] += 1
        return counts

    def cross_scope_candidates(self, hard_scope_keys: tuple[str, ...]) -> int:
        """Candidates whose metadata disagrees with the request on a hard scope key."""
        return sum(1 for d in self.decisions if any(d.scope.get(k) is ScopeMatch.MISMATCH for k in hard_scope_keys))
