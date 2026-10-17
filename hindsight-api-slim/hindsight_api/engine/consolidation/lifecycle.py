"""Lifecycle consolidation: an LLM response's lifecycle labels in, validated claim transitions out.

Adaptive memory in ``lifecycle`` mode keeps a retired belief instead of rewriting or deleting it
(DESIGN.md §5). The model labels what it would have done anyway — an UPDATE that ``supersedes`` or
``refutes`` its target, a CREATE naming what it replaces, refines, contradicts or splits, an old
fact attached to a retired observation as ``historical`` — and this module turns those labels into
claim transitions the store applies:

- :func:`plan_lifecycle` normalises and validates one response deterministically, before anything
  is written: an invalid label is dropped (logged and counted per reason), the rest is applied.
- the ``apply_*`` coroutines write a validated plan inside the batch's apply transaction, after the
  observations exist, through the store's claim interface (``lock_claims`` once, then CAS).

Nothing here calls an LLM or an embedder, and nothing runs outside lifecycle mode except
:func:`would_inherit`, which observe mode uses to record shadow retirements.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from ...metrics import ClaimTransition, get_metrics_collector
from ..adaptive.claims import RETIRED_STATUSES
from ..adaptive.models import ClaimState, TruthStatus

if TYPE_CHECKING:
    from ..adaptive.settings import AdaptiveSettings
    from ..memories.base import MemoriesExtension, StoredMemory
    from ..response_models import MemoryFact

logger = logging.getLogger(__name__)

#: ``updates[].relation`` values.
RELATIONS = ("supersedes", "refutes", "historical")
#: What a retiring label does to its target.
_RETIRING_STATUS = {"supersedes": TruthStatus.SUPERSEDED, "refutes": TruthStatus.REFUTED}
_LIVE = (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN)
ORIGIN = "consolidation"
#: ``memory_claim_state.scope_fingerprint`` is VARCHAR2(1024) on Oracle; a longer one is hashed.
_FINGERPRINT_MAX_BYTES = 1024


# --------------------------------------------------------------------------- lenient field readers


def clean_ids(value: Any) -> list[str]:
    """A label's id list as the model wrote it, minus anything that is not a non-empty string.

    Lenient on purpose: one malformed entry must not reject the whole response (#4152). Whether
    an id names something the model was shown is checked later, after miscopied ids are repaired.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        logger.warning("[CONSOLIDATION][lifecycle] ignoring a non-list id field %r", value)
        return []
    return list(dict.fromkeys(v.strip() for v in value if isinstance(v, str) and v.strip()))


def clean_id(value: Any) -> str | None:
    """A single-id label (``split_from``): a non-empty string, the first of a list, or None."""
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if isinstance(v, str) and v.strip()), None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def clean_relation(value: Any) -> str | None:
    """``updates[].relation``: one of :data:`RELATIONS`, case-insensitive; anything else is None."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in RELATIONS:
            return normalized
        if normalized in ("", "none", "null"):
            return None
    if value is not None:
        logger.warning("[CONSOLIDATION][lifecycle] ignoring unknown update relation %r", value)
    return None


# --------------------------------------------------------------------------- pure helpers


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None or value.utcoffset() is None else value.astimezone(UTC)


def fact_instant(fact: Mapping[str, Any]) -> datetime | None:
    """When a fact asserted what it says: ``mentioned_at``, else ``occurred_start``, else ``event_date``."""
    for key in ("mentioned_at", "occurred_start", "event_date"):
        if (value := _utc(fact.get(key))) is not None:
            return value
    return None


def change_instant(facts: Iterable[Mapping[str, Any]], now: datetime) -> datetime:
    """``t*``: the earliest instant at which the cited facts assert the new state, else ``now``."""
    return min((i for i in (fact_instant(f) for f in facts) if i is not None), default=now)


def _earliest_mention(facts: Iterable[Mapping[str, Any]]) -> datetime | None:
    return min((m for m in (_utc(f.get("mentioned_at")) for f in facts) if m is not None), default=None)


def _newest(values: Iterable[Any]) -> datetime | None:
    return max((v for v in (_utc(x) for x in values) if v is not None), default=None)


def scope_fingerprint(metadatas: Sequence[Mapping[str, Any] | None], keys: Sequence[str]) -> str | None:
    """Sorted ``k=v`` over ``keys`` from the facts' metadata, when all the facts agree; else None.

    A fact without a key disagrees with one that has it, so mixed provenance yields None (no
    fingerprint, nothing to compare) rather than the fingerprint of whichever fact came first.
    Bounded to the column's 1024 bytes by hashing.
    """
    if not metadatas or not keys:
        return None
    parts: list[str] = []
    for key in keys:
        values = {None if not md or md.get(key) in (None, "") else str(md.get(key)) for md in metadatas}
        if len(values) > 1:
            return None
        (value,) = values
        if value is not None:
            parts.append(f"{key}={value}")
    if not parts:
        return None
    fingerprint = ",".join(sorted(parts))
    if len(fingerprint.encode("utf-8")) > _FINGERPRINT_MAX_BYTES:
        return "sha256:" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return fingerprint


def top_trust(metadatas: Iterable[Mapping[str, Any] | None], trust: Mapping[str, float]) -> float:
    """The highest ``source_trust`` tier among the facts' metadata ``source`` values (0 when none is known)."""
    return max(
        (float(trust.get(str(md["source"]), 0.0)) for md in metadatas if md and md.get("source") not in (None, "")),
        default=0.0,
    )


@dataclass(frozen=True)
class ConflictOutcome:
    """How a ``conflicts_with`` label resolves (DESIGN.md §5.4). ``winner`` None: both sides uncertain."""

    winner: Literal["new", "existing"] | None
    new_confidence: float
    existing_confidence: float


def resolve_conflict(
    *,
    new_trust: float,
    new_newest: datetime | None,
    existing_trust: float,
    existing_newest: datetime | None,
    policy: str,
) -> ConflictOutcome:
    """Settle a conflict once, from both sides' source trust and evidence recency.

    ``evidence`` (default): a side wins only with a strictly higher top trust tier AND the more
    recent evidence, so with every trust at its default (none) a conflict never auto-resolves and
    repetition never decides truth. ``latest`` (an ablation): the side with the more recent evidence
    wins. Confidence is each side's share of the combined trust (0.5 each when there is none).
    """
    total = new_trust + existing_trust
    share = 0.5 if total <= 0 else new_trust / total
    newer: Literal["new", "existing"] | None = None
    if new_newest is not None and existing_newest is not None and new_newest != existing_newest:
        newer = "new" if new_newest > existing_newest else "existing"
    if policy == "latest":
        winner = newer
    elif newer == "new" and new_trust > existing_trust:
        winner = "new"
    elif newer == "existing" and existing_trust > new_trust:
        winner = "existing"
    else:
        winner = None
    return ConflictOutcome(winner=winner, new_confidence=share, existing_confidence=1.0 - share)


# --------------------------------------------------------------------------- the pool


@dataclass(frozen=True)
class PoolEntry:
    """One observation the model was shown, with the claim state the pool rendered for it."""

    id: str
    text: str
    status: TruthStatus
    successor: str | None
    tags: frozenset[str]
    source_ids: tuple[str, ...]
    mentioned_at: datetime | None
    valid_from: datetime | None
    claim_fingerprint: str | None
    #: May be named as a retirement / split / conflict target: live, and in exactly this scope.
    retirable: bool

    @property
    def retired(self) -> bool:
        return self.status in RETIRED_STATUSES


def pool_entry(observation: MemoryFact, claim: ClaimState | None, scope_tags: frozenset[str]) -> PoolEntry:
    status = claim.truth_status if claim is not None else TruthStatus.ACTIVE
    tags = frozenset(observation.tags or [])
    return PoolEntry(
        id=str(observation.id),
        text=observation.text,
        status=status,
        successor=claim.superseded_by_id if claim is not None else None,
        tags=tags,
        source_ids=tuple(dict.fromkeys(str(s) for s in (observation.source_fact_ids or []))),
        mentioned_at=_utc(observation.mentioned_at),
        valid_from=claim.valid_from if claim is not None else None,
        claim_fingerprint=claim.scope_fingerprint if claim is not None else None,
        retirable=status in _LIVE and tags == scope_tags,
    )


# --------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class RetireTarget:
    """A validated supersede/refute of one pool observation by a successor this response creates."""

    target_id: str
    status: TruthStatus
    relation: Literal["supersedes", "refutes"]
    #: ``retires_source_ids`` when the model named valid ones; None: the default rule (the target's
    #: sources minus kept ones and facts this response cites again).
    explicit_sources: tuple[str, ...] | None


@dataclass(frozen=True)
class ConflictPlan:
    target_id: str
    outcome: ConflictOutcome


@dataclass
class SuccessorPlan:
    """What one CREATE (or labelled UPDATE turned into one) does to the claims around it."""

    cited: list[str]
    t_star: datetime
    fingerprint: str | None
    retires: list[RetireTarget] = field(default_factory=list)
    keeps: list[str] = field(default_factory=list)
    refines: list[str] = field(default_factory=list)
    conflicts: list[ConflictPlan] = field(default_factory=list)
    #: The observation this create is one facet of (two or more creates name it).
    split_from: str | None = None

    @property
    def labelled(self) -> bool:
        return bool(self.retires or self.refines or self.conflicts or self.split_from)


@dataclass(frozen=True)
class HistoricalAttach:
    """Old facts restating a retired observation: attached to it as history, retired with it."""

    target_id: str
    fact_ids: tuple[str, ...]


@dataclass
class Drops:
    """Lifecycle labels dropped by validation, per reason: logged, counted, reported in the stats."""

    bank_id: str
    counts: Counter[str] = field(default_factory=Counter)

    def drop(self, reason: str, detail: str) -> None:
        self.counts[reason] += 1
        logger.info("[CONSOLIDATION][lifecycle] bank=%s dropped a label (%s): %s", self.bank_id, reason, detail)
        get_metrics_collector().record_adaptive_lifecycle_dropped(reason=reason, bank_id=self.bank_id)


@dataclass
class LifecyclePlan:
    """Everything one response's lifecycle labels decided, validated."""

    #: ``id(create action)`` -> its plan. Creates without a plan are ordinary creates.
    successors: dict[int, SuccessorPlan] = field(default_factory=dict)
    historical: list[HistoricalAttach] = field(default_factory=list)
    #: Plain deletes (no ``merged_into``), retired without a successor instead of deleted.
    plain_retires: list[str] = field(default_factory=list)
    #: Facts whose only action was dropped onto a retired observation: counted as acted on (they
    #: are old news), never re-asked.
    acted_on: set[str] = field(default_factory=set)
    #: Every batch fact the response cites anywhere (excluded from default raw inheritance).
    recited: set[str] = field(default_factory=set)

    def for_create(self, create: Any) -> SuccessorPlan | None:
        return self.successors.get(id(create))

    def retire_target_ids(self) -> set[str]:
        return {r.target_id for plan in self.successors.values() for r in plan.retires}


def _recency_ok(cited_facts: Sequence[Mapping[str, Any]], target: PoolEntry, t_star: datetime) -> bool:
    """The cited facts are at least as recent as the target's newest source, and ``t*`` is not
    before the target's own ``valid_from``. Unknown times on either side pass."""
    earliest = _earliest_mention(cited_facts)
    if earliest is not None and target.mentioned_at is not None and earliest < target.mentioned_at:
        return False
    return target.valid_from is None or t_star >= target.valid_from


def _fingerprints_differ(target_fp: str | None, successor_fp: str | None) -> bool:
    return target_fp is not None and successor_fp is not None and target_fp != successor_fp


@dataclass(frozen=True)
class PlanInputs:
    """What :func:`plan_lifecycle` reads, all of it gathered before the apply transaction."""

    pool: Mapping[str, PoolEntry]
    #: The batch's facts by id (the dicts consolidation reads them into).
    facts: Mapping[str, Mapping[str, Any]]
    scope_tags: frozenset[str]
    settings: AdaptiveSettings
    now: datetime
    #: Each lifecycle target's source facts, as the store holds them (fingerprint and trust).
    target_sources: Mapping[str, Sequence[StoredMemory]]


def target_fingerprint(entry: PoolEntry, inputs: PlanInputs) -> str | None:
    """The target's claim fingerprint, or — for an observation never written as a successor — the
    one its own sources' metadata gives."""
    if entry.claim_fingerprint is not None:
        return entry.claim_fingerprint
    sources = inputs.target_sources.get(entry.id) or []
    return scope_fingerprint([s.metadata for s in sources], inputs.settings.scope_keys)


def _validate_retire(
    target_id: str,
    relation: Literal["supersedes", "refutes"],
    cited_facts: Sequence[Mapping[str, Any]],
    plan: SuccessorPlan,
    inputs: PlanInputs,
    drops: Drops,
    claimed: dict[str, str],
    claimant: str,
) -> Literal["ok", "historical", "dropped", "recency"]:
    entry = inputs.pool.get(target_id)
    if entry is None:
        drops.drop("target_not_in_pool", f"{relation} {target_id}")
        return "dropped"
    if entry.retired:
        if not _recency_ok(cited_facts, entry, plan.t_star):
            return "historical"
        drops.drop("target_not_active", f"{relation} {target_id} ({entry.status.value})")
        return "dropped"
    if entry.tags != inputs.scope_tags:
        drops.drop("scope_tags", f"{relation} {target_id}: tags {sorted(entry.tags)} != {sorted(inputs.scope_tags)}")
        return "dropped"
    if _fingerprints_differ(target_fingerprint(entry, inputs), plan.fingerprint):
        drops.drop("scope_fingerprint", f"{relation} {target_id}")
        return "dropped"
    if not _recency_ok(cited_facts, entry, plan.t_star):
        drops.drop("recency", f"{relation} {target_id}: cited facts are older than the target")
        return "recency"
    if target_id in claimed and claimed[target_id] != claimant:
        drops.drop("duplicate_retirement", f"{relation} {target_id} already retired by {claimed[target_id]}")
        return "dropped"
    claimed[target_id] = claimant
    return "ok"


def _new_plan(cited: list[str], inputs: PlanInputs) -> SuccessorPlan:
    facts = [inputs.facts[f] for f in cited]
    return SuccessorPlan(
        cited=cited,
        t_star=change_instant(facts, inputs.now),
        fingerprint=scope_fingerprint([f.get("metadata") for f in facts], inputs.settings.scope_keys),
    )


def plan_lifecycle(
    llm_result: Any,
    inputs: PlanInputs,
    drops: Drops,
    *,
    new_create: Callable[[str, list[str], str], Any],
) -> LifecyclePlan:
    """Normalise and validate one response's lifecycle labels; mutates ``llm_result`` to what applies.

    Normalisation (DESIGN.md §5.2, P3 contract): a labelled UPDATE becomes a successor CREATE
    (``new_create(text, fact_ids, reason)``) and the target keeps its text; a delete of a retirement
    target is absorbed into the retirement; an unlabelled update of a retirement target loses its
    text change (its facts join the successor); updates onto a retired observation are dropped (the
    facts count as acted on) unless they are ``historical``. Validation drops, per label: a target
    not in the pool, not live, in another scope (tags, or a differing scope fingerprint), newer than
    the cited facts (then an attach as history when the target is already retired), or already
    retired by another label of this response (the first wins, unless both are ``split_from``).
    """
    plan = LifecyclePlan()
    pool, facts = inputs.pool, inputs.facts
    claimed: dict[str, str] = {}
    for action in [*llm_result.creates, *llm_result.updates]:
        plan.recited.update(f for f in action.source_fact_ids if f in facts)

    # --- labelled updates -> successor creates (or history) -----------------------------------
    kept_updates: list[Any] = []
    converted: list[Any] = []
    for update in llm_result.updates:
        relation = getattr(update, "relation", None)
        if relation is None:
            kept_updates.append(update)
            continue
        target = pool.get(update.observation_id)
        cited = [f for f in update.source_fact_ids if f in facts]
        if target is None:
            drops.drop("target_not_in_pool", f"{relation} update of {update.observation_id}")
            continue
        if not cited:
            continue  # cites nothing in the batch: the base path discards it as unresolved
        if relation == "historical":
            if target.retired:
                plan.historical.append(HistoricalAttach(target_id=target.id, fact_ids=tuple(cited)))
            else:
                # History of a live observation is just evidence for it.
                update.relation = None
                kept_updates.append(update)
            continue
        if _norm(update.text) == _norm(target.text):
            # Nothing changed: the label has nothing to retire, so it is ordinary evidence.
            update.relation = None
            kept_updates.append(update)
            continue
        successor_plan = _new_plan(cited, inputs)
        outcome = _validate_retire(
            target.id,
            relation,
            [facts[f] for f in cited],
            successor_plan,
            inputs,
            drops,
            claimed,
            claimant=f"update:{target.id}",
        )
        if outcome == "historical":
            plan.historical.append(HistoricalAttach(target_id=target.id, fact_ids=tuple(cited)))
            continue
        if outcome == "recency" or (outcome == "dropped" and target.retired):
            plan.acted_on.update(cited)
            continue
        create = new_create(update.text, list(update.source_fact_ids), update.reason or f"labelled update ({relation})")
        if outcome == "ok":
            explicit = [s for s in getattr(update, "retires_source_ids", []) or [] if s in target.source_ids]
            if len(explicit) != len(getattr(update, "retires_source_ids", []) or []):
                drops.drop("source_not_of_target", f"retires_source_ids of {target.id}")
            successor_plan.retires.append(
                RetireTarget(
                    target_id=target.id,
                    status=_RETIRING_STATUS[relation],
                    relation=relation,
                    explicit_sources=tuple(explicit) or None,
                )
            )
            successor_plan.keeps = _valid_keeps(getattr(update, "keeps_source_ids", []), [target], drops)
            plan.successors[id(create)] = successor_plan
        # Out of scope or a second successor: the new state is a claim of its own in this scope.
        converted.append(create)
    llm_result.updates = kept_updates

    # --- creates' own labels -------------------------------------------------------------------
    split_counts = Counter(clean_id(getattr(c, "split_from", None)) for c in llm_result.creates)
    kept_creates: list[Any] = []
    for index, create in enumerate(llm_result.creates):
        cited = [f for f in create.source_fact_ids if f in facts]
        labels = {
            name: clean_ids(getattr(create, name, None))
            for name in ("supersedes", "refutes", "refines", "conflicts_with")
        }
        split_from = clean_id(getattr(create, "split_from", None))
        if not cited or not (any(labels.values()) or split_from):
            kept_creates.append(create)
            continue
        claimant = f"create:{index}"
        successor_plan = _new_plan(cited, inputs)
        named: set[str] = set()
        retire_outcomes: list[str] = []
        for relation in ("supersedes", "refutes"):
            for target_id in labels[relation]:
                if target_id in named:
                    drops.drop("self_reference", f"{target_id} named twice by one create")
                    continue
                named.add(target_id)
                outcome = _validate_retire(
                    target_id, relation, [facts[f] for f in cited], successor_plan, inputs, drops, claimed, claimant
                )
                retire_outcomes.append(outcome)
                if outcome == "ok":
                    successor_plan.retires.append(
                        RetireTarget(
                            target_id=target_id,
                            status=_RETIRING_STATUS[relation],
                            relation=relation,
                            explicit_sources=None,
                        )
                    )
                elif outcome == "historical":
                    plan.historical.append(HistoricalAttach(target_id=target_id, fact_ids=tuple(cited)))
        for target_id in labels["refines"]:
            entry = pool.get(target_id)
            if target_id in named or entry is None or entry.retired:
                reason = (
                    "self_reference"
                    if target_id in named
                    else "target_not_in_pool"
                    if entry is None
                    else "target_not_active"
                )
                drops.drop(reason, f"refines {target_id}")
                continue
            named.add(target_id)
            successor_plan.refines.append(target_id)
        for target_id in labels["conflicts_with"]:
            entry = pool.get(target_id)
            reason = _live_in_scope(entry, target_id, named, inputs, successor_plan)
            if reason is not None:
                drops.drop(reason, f"conflicts_with {target_id}")
                continue
            assert entry is not None
            named.add(target_id)
            successor_plan.conflicts.append(
                ConflictPlan(target_id=target_id, outcome=_conflict_outcome(entry, cited, inputs))
            )
        if split_from is not None:
            entry = pool.get(split_from)
            if split_counts[split_from] < 2:
                # One facet named alone is a refinement: the parent stays true.
                if entry is not None and not entry.retired and split_from not in named:
                    successor_plan.refines.append(split_from)
                else:
                    drops.drop("target_not_active", f"split_from {split_from}")
            else:
                reason = _live_in_scope(entry, split_from, named, inputs, successor_plan)
                if reason is None and claimed.get(split_from, "split") != "split":
                    reason = "duplicate_retirement"
                if reason is not None:
                    drops.drop(reason, f"split_from {split_from}")
                else:
                    claimed[split_from] = "split"
                    successor_plan.split_from = split_from
        targets = [pool[r.target_id] for r in successor_plan.retires]
        successor_plan.keeps = _valid_keeps(getattr(create, "keeps_source_ids", []), targets, drops)
        if (
            retire_outcomes
            and not successor_plan.labelled
            and all(o in ("recency", "historical") for o in retire_outcomes)
        ):
            # Every replacement it claimed is newer than its facts: older news, not a new state (it
            # is history of a retired target, or of nothing).
            if "recency" in retire_outcomes:
                plan.acted_on.update(cited)
            continue
        if successor_plan.labelled:
            plan.successors[id(create)] = successor_plan
        kept_creates.append(create)
    llm_result.creates = [*kept_creates, *converted]

    # --- deletes: absorbed, merged or retired without a successor -----------------------------
    retire_targets = plan.retire_target_ids() | {p.split_from for p in plan.successors.values() if p.split_from}
    kept_deletes: list[Any] = []
    for delete in llm_result.deletes:
        entry = pool.get(delete.observation_id)
        if delete.observation_id in retire_targets:
            logger.info(
                "[CONSOLIDATION][lifecycle] delete of %s absorbed into its retirement", delete.observation_id[:8]
            )
            continue
        if entry is not None and entry.retired:
            drops.drop("target_not_active", f"delete of {delete.observation_id}")
            continue
        if delete.merged_into:
            survivor = pool.get(delete.merged_into)
            if survivor is not None and survivor.retired:
                drops.drop("target_not_active", f"merge into {delete.merged_into}")
                continue
            kept_deletes.append(delete)
            continue
        if entry is None:
            kept_deletes.append(delete)  # rejected downstream exactly as before (not in the pool)
            continue
        if entry.tags != inputs.scope_tags:
            drops.drop("scope_tags", f"retire of {delete.observation_id}")
            continue
        plan.plain_retires.append(delete.observation_id)
    llm_result.deletes = kept_deletes

    # --- unlabelled updates: of a retirement target, or of a retired observation --------------
    successor_of = {
        r.target_id: create
        for create in llm_result.creates
        if (p := plan.for_create(create)) is not None
        for r in p.retires
    }
    final_updates: list[Any] = []
    for update in llm_result.updates:
        entry = pool.get(update.observation_id)
        if update.observation_id in successor_of:
            # The supersede wins; the update's text change is dropped and its facts join the successor.
            successor = successor_of[update.observation_id]
            extra = [f for f in update.source_fact_ids if f in facts and f not in successor.source_fact_ids]
            successor.source_fact_ids = [*successor.source_fact_ids, *extra]
            successor_plan = plan.for_create(successor)
            if successor_plan is not None and extra:
                successor_plan.cited.extend(extra)
                successor_plan.t_star = min(
                    successor_plan.t_star, change_instant([facts[f] for f in extra], inputs.now)
                )
            logger.info(
                "[CONSOLIDATION][lifecycle] update of %s dropped: the same response retires it",
                update.observation_id[:8],
            )
            continue
        if entry is not None and entry.retired:
            plan.acted_on.update(f for f in update.source_fact_ids if f in facts)
            drops.drop("target_not_active", f"update of {update.observation_id} ({entry.status.value})")
            continue
        final_updates.append(update)
    llm_result.updates = final_updates
    return plan


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _live_in_scope(
    entry: PoolEntry | None, target_id: str, named: set[str], inputs: PlanInputs, plan: SuccessorPlan
) -> str | None:
    """Why ``entry`` cannot be a split or conflict target, or None when it can."""
    if entry is None:
        return "target_not_in_pool"
    if target_id in named:
        return "self_reference"
    if entry.retired:
        return "target_not_active"
    if entry.tags != inputs.scope_tags:
        return "scope_tags"
    if _fingerprints_differ(target_fingerprint(entry, inputs), plan.fingerprint):
        return "scope_fingerprint"
    return None


def _valid_keeps(raw: Any, targets: Sequence[PoolEntry], drops: Drops) -> list[str]:
    keeps = clean_ids(raw)
    allowed = {s for t in targets for s in t.source_ids}
    valid = [k for k in keeps if k in allowed]
    if len(valid) != len(keeps):
        drops.drop("source_not_of_target", f"keeps_source_ids {sorted(set(keeps) - allowed)}")
    return valid


def _conflict_outcome(entry: PoolEntry, cited: Sequence[str], inputs: PlanInputs) -> ConflictOutcome:
    trust = inputs.settings.source_trust
    sources = inputs.target_sources.get(entry.id) or []
    new_facts = [inputs.facts[f] for f in cited]
    return resolve_conflict(
        new_trust=top_trust([f.get("metadata") for f in new_facts], trust),
        new_newest=_newest(f.get("mentioned_at") for f in new_facts),
        existing_trust=top_trust([s.metadata for s in sources], trust),
        existing_newest=_newest([*(s.mentioned_at for s in sources), entry.mentioned_at]),
        policy=inputs.settings.conflict_policy,
    )


def lifecycle_target_ids(llm_result: Any, pool: Mapping[str, PoolEntry]) -> list[str]:
    """Pool observations a response's lifecycle labels or plain deletes name (whose sources the
    plan reads for fingerprints and trust)."""
    ids: list[str] = []
    for create in llm_result.creates:
        for name in ("supersedes", "refutes", "conflicts_with"):
            ids.extend(clean_ids(getattr(create, name, None)))
        if (split_from := clean_id(getattr(create, "split_from", None))) is not None:
            ids.append(split_from)
    ids.extend(u.observation_id for u in llm_result.updates if getattr(u, "relation", None) in _RETIRING_STATUS)
    return [i for i in dict.fromkeys(ids) if i in pool]


# --------------------------------------------------------------------------- application


@dataclass
class ApplyState:
    """Claim state read under the batch's one ``lock_claims`` pass, plus what the apply records."""

    bank_id: str
    store: MemoriesExtension
    fq_table: Callable[[str], str]
    scope_tags: frozenset[str]
    now: datetime
    drops: Drops
    locked: set[str] = field(default_factory=set)
    claims: dict[str, ClaimState] = field(default_factory=dict)
    #: ``observation id -> its source ids`` read before the lock pass (inheritance candidates).
    sources: dict[str, list[str]] = field(default_factory=dict)
    transitions: Counter[str] = field(default_factory=Counter)
    retired_tags: set[str] = field(default_factory=set)

    def live(self, memory_id: str) -> bool:
        claim = self.claims.get(memory_id)
        return memory_id in self.locked and (claim is None or claim.truth_status in _LIVE)

    def count(self, transition: ClaimTransition) -> None:
        """One transition whose compare-and-set won (metric ``hindsight.adaptive_memory.<kind>s``)."""
        self.transitions[transition] += 1
        get_metrics_collector().record_adaptive_claim_transition(
            bank_id=self.bank_id, transition=transition, origin=ORIGIN
        )


async def lock(conn, state: ApplyState, ids: Iterable[str]) -> None:
    """The batch's ONE ``lock_claims`` pass over every pre-existing memory it may touch.

    ``skip_locked``: a memory a deleter holds is skipped, never waited for; nothing is written for it.
    """
    wanted = sorted({i for i in ids if _is_uuid(i)})
    if not wanted:
        return
    state.locked = await state.store.lock_claims(
        conn, bank_id=state.bank_id, memory_ids=wanted, skip_locked=True, now=state.now
    )
    state.claims = await state.store.claims_for(conn, bank_id=state.bank_id, memory_ids=sorted(state.locked))


async def lock_new(conn, state: ApplyState, ids: Iterable[str]) -> None:
    """Claim rows for observations this transaction just inserted: nobody else can see them, so a
    second pass over them alone cannot invert the lock order of the first."""
    fresh = sorted({i for i in ids if _is_uuid(i)} - state.locked)
    if not fresh:
        return
    state.locked |= await state.store.lock_claims(
        conn, bank_id=state.bank_id, memory_ids=fresh, skip_locked=False, now=state.now
    )


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


async def in_scope(conn, state: ApplyState, observation_id: str) -> bool:
    """Re-check the target's tags under its row lock: an edit since the pool was read moves it out."""
    tags = await state.store.lock_observation_tags(
        conn=conn, fq_table=state.fq_table, bank_id=state.bank_id, observation_id=observation_id
    )
    if tags is None or frozenset(tags) != state.scope_tags:
        state.drops.drop("scope_changed", f"{observation_id} (tags now {tags})")
        return False
    state.retired_tags.update(tags)
    return True


async def retire(
    conn,
    state: ApplyState,
    *,
    target_id: str,
    status: TruthStatus,
    retired_via: str,
    successor_id: str | None,
    valid_to: datetime | None,
    reason: str,
    relation: str | None,
    evidence_ids: Sequence[str],
    prechecked: bool = False,
) -> bool:
    """CAS-retire one locked observation, then write its relation and evidence. True when it won.

    ``prechecked``: the caller already re-checked the target's liveness and scope (a merge, before
    folding its evidence away; a conflict's losing side this transaction just created).
    """
    if not prechecked:
        if not state.live(target_id):
            state.drops.drop("cas_lost", f"{target_id} is no longer live")
            return False
        if not await in_scope(conn, state, target_id):
            return False
    won = await state.store.cas_retire(
        conn,
        bank_id=state.bank_id,
        memory_id=target_id,
        to_status=status,
        retired_via=retired_via,
        superseded_by_id=successor_id,
        valid_to=valid_to,
        reason=reason,
        confidence=None,
        now=state.now,
    )
    if not won:
        state.drops.drop("cas_lost", f"{target_id}")
        return False
    if relation is not None:
        await state.store.insert_relation(
            conn,
            bank_id=state.bank_id,
            relation_type=relation,
            source_id=successor_id,
            target_id=target_id,
            origin=ORIGIN,
            reason=reason,
            confidence=None,
            evidence={"fact_ids": list(evidence_ids)},
            now=state.now,
        )
    if evidence_ids:
        await state.store.add_claim_evidence(
            conn, bank_id=state.bank_id, memory_id=target_id, evidence_ids=list(evidence_ids), now=state.now
        )
    state.claims[target_id] = _retired_claim(state.claims.get(target_id), target_id, status, retired_via, successor_id)
    return True


def _retired_claim(
    prior: ClaimState | None, memory_id: str, status: TruthStatus, retired_via: str, successor_id: str | None
) -> ClaimState:
    return ClaimState(
        memory_id=memory_id,
        truth_status=status,
        confidence=None,
        valid_from=prior.valid_from if prior is not None else None,
        valid_to=None,
        retired_via=retired_via,
        superseded_by_id=successor_id,
        inherited_from_id=None,
        scope_fingerprint=prior.scope_fingerprint if prior is not None else None,
        reason=None,
    )


async def assert_current(
    conn, state: ApplyState, *, memory_id: str, valid_from: datetime, fingerprint: str | None = None
) -> None:
    """Give a successor (or a fact asserting the new state) its explicit ``valid_from``.

    Only a live claim that carries no validity yet is written: a fact that is already contested or
    retired, or that an earlier change already dated, keeps what it has.
    """
    claim = state.claims.get(memory_id)
    if memory_id not in state.locked or (
        claim is not None and (claim.truth_status is not TruthStatus.ACTIVE or claim.valid_from is not None)
    ):
        return
    await state.store.set_claim(
        conn,
        bank_id=state.bank_id,
        memory_id=memory_id,
        truth_status=TruthStatus.ACTIVE,
        valid_from=valid_from,
        confidence=claim.confidence if claim is not None else None,
        scope_fingerprint=fingerprint if fingerprint is not None else (claim.scope_fingerprint if claim else None),
        now=state.now,
    )


async def inherit(
    conn,
    state: ApplyState,
    *,
    target: RetireTarget,
    successor_id: str | None,
    valid_to: datetime,
    evidence_ids: Sequence[str],
    keeps: Sequence[str],
    recited: set[str],
) -> list[str]:
    """Pass a retired observation's status down to the raw facts that only it stood on (DESIGN.md §5.3)."""
    if target.explicit_sources is not None:
        candidates = list(target.explicit_sources)
    else:
        candidates = [s for s in state.sources.get(target.target_id, []) if s not in set(keeps) and s not in recited]
    if not candidates:
        return []
    return await state.store.inherit_from_observation(
        conn,
        bank_id=state.bank_id,
        observation_id=target.target_id,
        to_status=target.status,
        successor_id=successor_id,
        valid_to=valid_to,
        evidence_ids=list(evidence_ids),
        candidate_source_ids=candidates,
        now=state.now,
    )


async def attach_historical(
    conn, state: ApplyState, attach: HistoricalAttach, live_fact_ids: list[uuid.UUID]
) -> list[str]:
    """Old facts restating a retired observation join its sources as history and inherit its status.

    The observation's text and ``updated_at`` stay as they are; each fact copies the observation's
    status, successor and ``valid_to``, with the observation's own evidence (so deleting that
    evidence reverts both). Returns the facts retired; dated facts and experiences stay current,
    exactly as for any inheritance.
    """
    claim = state.claims.get(attach.target_id)
    if attach.target_id not in state.locked or claim is None or claim.truth_status not in RETIRED_STATUSES:
        state.drops.drop("target_not_active", f"historical attach to {attach.target_id} (no longer retired)")
        return []
    if not live_fact_ids or not await in_scope(conn, state, attach.target_id):
        return []
    attached = await state.store.attach_sources_to_observation(
        conn=conn,
        fq_table=state.fq_table,
        bank_id=state.bank_id,
        observation_id=attach.target_id,
        source_memory_ids=live_fact_ids,
    )
    if not attached:
        return []
    evidence = (await state.store.claim_evidence(conn, bank_id=state.bank_id, memory_ids=[attach.target_id])).get(
        attach.target_id, []
    )
    return await state.store.inherit_from_observation(
        conn,
        bank_id=state.bank_id,
        observation_id=attach.target_id,
        to_status=claim.truth_status,
        successor_id=claim.superseded_by_id,
        valid_to=claim.valid_to or state.now,
        evidence_ids=evidence,
        candidate_source_ids=[str(f) for f in live_fact_ids],
        reason=f"historical:{claim.truth_status.value}",
        now=state.now,
    )


async def mark_uncertain(
    conn, state: ApplyState, *, memory_id: str, confidence: float, valid_from: datetime | None
) -> bool:
    """Set a live, locked claim to ``uncertain`` with its trust share, keeping its validity and scope."""
    claim = state.claims.get(memory_id)
    if not state.live(memory_id):
        return False
    await state.store.set_claim(
        conn,
        bank_id=state.bank_id,
        memory_id=memory_id,
        truth_status=TruthStatus.UNCERTAIN,
        valid_from=claim.valid_from if claim is not None and claim.valid_from is not None else valid_from,
        valid_to=None,
        confidence=confidence,
        scope_fingerprint=claim.scope_fingerprint if claim is not None else None,
        reason="conflict",
        now=state.now,
    )
    state.count("uncertain")
    return True


# --------------------------------------------------------------------------- observe-mode shadow


async def would_inherit(
    conn, store: MemoriesExtension, fq_table: Callable[[str], str], ops: Any, *, bank_id: str, observation_id: str
) -> list[str]:
    """The raw sources a lifecycle retirement of ``observation_id`` would pass its status down to.

    Read-only, for observe mode's shadow records: the same exclusions as the store's inheritance
    (experience and dated facts, sources in a narrower scope than the observation, sources another
    live observation still cites).
    """
    sources = (await store.observation_sources(conn, bank_id=bank_id, observation_ids=[observation_id])).get(
        observation_id, []
    )
    if not sources:
        return []
    observation = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
    obs_tags = set(observation[0].tags or []) if observation else set()
    rows = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sources)
    eligible = [
        m.unit_id
        for m in rows
        if m.fact_type == "world" and m.occurred_start is None and obs_tags.issubset(set(m.tags or []))
    ]
    if not eligible:
        return []
    citing = await store.observations_for_sources(
        conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=eligible
    )
    others = {o.unit_id for o in citing if o.unit_id != observation_id}
    other_claims = await store.claims_for(conn, bank_id=bank_id, memory_ids=sorted(others))
    live_others = {o for o in others if (c := other_claims.get(o)) is None or c.truth_status in _LIVE}
    cited_elsewhere = {s for o in citing if o.unit_id in live_others for s in o.source_memory_ids}
    return sorted(f for f in eligible if f not in cited_elsewhere)
