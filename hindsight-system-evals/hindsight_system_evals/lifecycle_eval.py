"""Classify what lifecycle-mode consolidation did, from the server's state, deterministically.

A bank in ``adaptive_memory_mode=lifecycle`` does not rewrite or delete a belief that
stopped being true: it writes a successor and retires the old observation with a claim
(``truth_status`` superseded / refuted, ``retired_via``, ``superseded_by_id``), passes
the retirement down to the raw facts the old observation stood on (``retired_via =
inherited``), and records relation rows (``supersedes``, ``refutes``, ``conflicts``,
``refines``, ...). All of it is readable through ``GET /memories/{id}`` (its
``lifecycle`` key), so whether consolidation took the right action can be graded
without a judge.

Each scenario declares what should happen (:class:`LifecycleSpec`):

* a **change** (:class:`ExpectedChange`): the beliefs resting on its ``stale`` facts
  should be superseded, or refuted, by a successor citing its ``new`` facts — or, for a
  ``conflict``, the two sides marked uncertain;
* every other memory is **protected**: a truth retirement of a raw fact outside the
  changes' ``allowed`` facts, or of an observation citing none of them, is a false
  retirement. A control scenario has no changes, so nothing in it may be retired; a
  lookalike bank (many near-identical entities, a few of which change) counts towards
  the control set too, on its protected memories.

Merges and splits retire a row too, but not its truth (the content lives on in the
survivor or the facets); they are counted apart and are never false retirements.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from hindsight_client import Hindsight
from pydantic import BaseModel, Field

from hindsight_system_evals.consolidation import RetiredFn, Scenario, _list

Action = Literal["supersede", "refute", "conflict"]
#: Claim states that take a memory out of the current view.
RETIRED = frozenset({"superseded", "refuted"})
#: ``retired_via`` values that retire a row without declaring its content untrue.
_STRUCTURAL = frozenset({"merges", "splits"})
_DIRECT_KIND: dict[str, Action] = {"supersedes": "supersede", "refutes": "refute"}


@dataclass(frozen=True)
class FactRef:
    """A fact of a scenario by wave (1-based) and exact text — seeds are shuffled, so not by index."""

    wave: int
    text: str


@dataclass(frozen=True)
class ExpectedChange:
    label: str
    action: Action
    #: The facts whose belief should be retired (for a conflict: one side).
    stale: tuple[FactRef, ...]
    #: The facts carrying the newer or correcting belief (for a conflict: the other side).
    new: tuple[FactRef, ...]
    #: Further facts whose observations may legitimately be retired with this change
    #: (a past event of the same tenure, say) without counting as a false retirement.
    allowed: tuple[FactRef, ...] = ()


Kind = Literal["control", "change", "lookalike"]
#: The scenario kinds whose false retirements make up the pre-registered control rate.
CONTROL_KINDS = frozenset({"control", "lookalike"})


@dataclass(frozen=True)
class LifecycleSpec:
    #: ``control``: nothing may be retired. ``change``: the listed changes are expected.
    #: ``lookalike``: changes are expected, and the many neighbours they resemble are controls.
    kind: Kind
    changes: tuple[ExpectedChange, ...] = ()


# --- server state ------------------------------------------------------------------


class Claim(BaseModel):
    truth_status: str = "active"
    retired_via: str | None = None
    superseded_by_id: str | None = None
    inherited_from_id: str | None = None
    reason: str | None = None
    evidence_memory_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    id: str
    relation_type: str
    source_memory_id: str | None = None
    target_memory_id: str | None = None
    origin: str | None = None
    reason: str | None = None


class _Lifecycle(BaseModel):
    claim: Claim | None = None
    relations_out: list[Relation] = Field(default_factory=list)
    relations_in: list[Relation] = Field(default_factory=list)


class _MemoryPayload(BaseModel):
    lifecycle: _Lifecycle | None = None


class MemoryState(BaseModel):
    id: str
    fact_type: str
    text: str
    sources: list[str] = Field(default_factory=list)
    document_id: str | None = None
    occurred_start: str | None = None
    #: None: no claim row, which the store reads as active.
    claim: Claim | None = None

    @property
    def status(self) -> str:
        return self.claim.truth_status if self.claim else "active"

    @property
    def retired(self) -> bool:
        return self.status in RETIRED

    @property
    def truth_retired(self) -> bool:
        """Retired as untrue or outdated — not as a merged duplicate or a split parent."""
        return self.retired and (self.claim is None or self.claim.retired_via not in _STRUCTURAL)


class BankState(BaseModel):
    memories: dict[str, MemoryState]
    relations: list[Relation]


async def _get_lifecycle(http: httpx.AsyncClient, bank_id: str, memory_id: str) -> _Lifecycle | None:
    response = await http.get(f"/v1/default/banks/{bank_id}/memories/{memory_id}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return _MemoryPayload.model_validate(response.json()).lifecycle


def _http(base_url: str, api_key: str | None) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(60.0))


async def _lifecycles(base_url: str, api_key: str | None, bank_id: str, ids: Iterable[str]) -> dict[str, _Lifecycle]:
    # Bounded: a noisy bank holds ~250 memories, one GET each.
    gate = asyncio.Semaphore(16)
    async with _http(base_url, api_key) as http:

        async def one(memory_id: str) -> tuple[str, _Lifecycle | None]:
            async with gate:
                return memory_id, await _get_lifecycle(http, bank_id, memory_id)

        pairs = await asyncio.gather(*(one(i) for i in ids))
    return {memory_id: lc for memory_id, lc in pairs if lc is not None}


def retired_fn(base_url: str, api_key: str | None) -> RetiredFn:
    """The :data:`RetiredFn` ``run_scenario`` uses to grade a lifecycle bank's current view."""

    async def retired(bank_id: str, ids: list[str]) -> set[str]:
        found = await _lifecycles(base_url, api_key, bank_id, ids)
        return {i for i, lc in found.items() if lc.claim is not None and lc.claim.truth_status in RETIRED}

    return retired


async def fetch_state(client: Hindsight, base_url: str, api_key: str | None, bank_id: str) -> BankState:
    """Every memory of the bank with its claim, and every lifecycle relation row."""
    memories: dict[str, MemoryState] = {}
    for kind in ("world", "experience", "observation"):
        for item in await _list(client, bank_id, kind):
            memories[item.id] = MemoryState(
                id=item.id,
                fact_type=kind,
                text=item.text or "",
                sources=list(item.source_memory_ids or []),
                document_id=item.document_id,
                occurred_start=item.occurred_start,
            )
    relations: dict[str, Relation] = {}
    for memory_id, lc in (await _lifecycles(base_url, api_key, bank_id, list(memories))).items():
        memories[memory_id].claim = lc.claim
        for relation in (*lc.relations_out, *lc.relations_in):
            relations[relation.id] = relation
    return BankState(memories=memories, relations=sorted(relations.values(), key=lambda r: r.id))


# --- classification ------------------------------------------------------------------


class RetirementEvent(BaseModel):
    memory_id: str
    fact_type: str
    status: str
    retired_via: str | None
    successor_id: str | None
    text: str
    #: Labels of the changes whose allowed facts this memory is (raw) or cites (observation).
    matched: list[str]
    protected: bool


class ChangeOutcome(BaseModel):
    label: str
    action: Action
    #: An observation citing a stale fact was retired with the expected kind.
    hit: bool
    #: ... with any kind (supersede for a refute counts here, not in ``hit``).
    hit_any_kind: bool
    #: Kinds of the direct retirements on this change's stale observations.
    observed: list[str]
    #: The successor of the hit cites one of the change's new facts.
    successor_cites_new: bool | None
    #: A current observation cites a stale fact and none of the new ones: the old belief
    #: is still current in the observation layer (a deterministic proxy for the trap).
    stale_current: bool
    #: A current observation cites both a stale and a new fact (history kept in place).
    history_in_place: bool
    #: Stale raw facts cited by a hit's target that were left active (not inherited).
    stale_raw_left_active: int
    stale_raw_total: int


class RawLeak(BaseModel):
    """A directly retired observation's sources that nothing else stands on, and their fate."""

    observation_id: str
    candidates: int
    left_active: int
    left_active_details: list[str]


class Classification(BaseModel):
    changes: list[ChangeOutcome]
    retirements: list[RetirementEvent]
    structural_retirements: int
    uncertain: list[str]
    conflict_relations: int
    relation_types: dict[str, int]
    false_retirement: bool
    false_retirements: list[RetirementEvent]
    raw_leaks: list[RawLeak]
    observations_total: int
    observations_current: int
    observations_deleted: int


def _resolve(scenario: Scenario, state: BankState, refs: Iterable[FactRef]) -> set[str]:
    ids: set[str] = set()
    for ref in refs:
        facts = scenario.waves[ref.wave - 1].facts
        j = next((j for j, f in enumerate(facts) if f.text == ref.text), None)
        if j is None:
            raise ValueError(f"{scenario.id}: no fact in wave {ref.wave} reads {ref.text!r}")
        doc = f"{scenario.id}-w{ref.wave}-{j}"
        ids |= {m.id for m in state.memories.values() if m.document_id == doc and m.fact_type != "observation"}
    return ids


def classify(
    scenario: Scenario, spec: LifecycleSpec, state: BankState, snapshot_ids: Iterable[str] = ()
) -> Classification:
    mem = state.memories
    stale = {c.label: _resolve(scenario, state, c.stale) for c in spec.changes}
    new = {c.label: _resolve(scenario, state, c.new) for c in spec.changes}
    allowed = {
        c.label: stale[c.label]
        | _resolve(scenario, state, c.allowed)
        | (new[c.label] if c.action == "conflict" else set())
        for c in spec.changes
    }
    observations = [m for m in mem.values() if m.fact_type == "observation"]
    current = [o for o in observations if not o.retired]

    def matched(m: MemoryState) -> list[str]:
        cited = set(m.sources) if m.fact_type == "observation" else {m.id}
        return [label for label, ids in allowed.items() if cited & ids]

    retirements = [
        RetirementEvent(
            memory_id=m.id,
            fact_type=m.fact_type,
            status=m.status,
            retired_via=m.claim.retired_via if m.claim else None,
            successor_id=m.claim.superseded_by_id if m.claim else None,
            text=m.text[:200],
            matched=matched(m),
            protected=not matched(m),
        )
        for m in mem.values()
        if m.truth_retired
    ]
    direct = [r for r in retirements if r.fact_type == "observation" and r.retired_via in _DIRECT_KIND]

    outcomes: list[ChangeOutcome] = []
    for c in spec.changes:
        s, n = stale[c.label], new[c.label]
        if c.action == "conflict":
            sides = s | n
            touched = {o.id for o in observations if set(o.sources) & sides}
            hit = any(mem[i].status == "uncertain" for i in touched) or any(
                r.relation_type == "conflicts" and {r.source_memory_id, r.target_memory_id} & touched
                for r in state.relations
            )
            outcomes.append(
                ChangeOutcome(
                    label=c.label,
                    action=c.action,
                    hit=hit,
                    hit_any_kind=hit,
                    observed=sorted(
                        {_DIRECT_KIND[r.retired_via] for r in direct if r.memory_id in touched and r.retired_via}
                    ),
                    successor_cites_new=None,
                    stale_current=False,
                    history_in_place=False,
                    stale_raw_left_active=0,
                    stale_raw_total=len(sides),
                )
            )
            continue
        targets = [r for r in direct if set(mem[r.memory_id].sources) & s]
        kinds = [_DIRECT_KIND[r.retired_via] for r in targets if r.retired_via]
        hits = [r for r, k in zip(targets, kinds, strict=True) if k == c.action]
        first = hits[0] if hits else (targets[0] if targets else None)
        successor = mem.get(first.successor_id) if first and first.successor_id else None
        stale_cited = {f for r in targets for f in mem[r.memory_id].sources if f in s}
        outcomes.append(
            ChangeOutcome(
                label=c.label,
                action=c.action,
                hit=bool(hits),
                hit_any_kind=bool(targets),
                observed=sorted(set(kinds)),
                successor_cites_new=(bool(set(successor.sources) & n) if successor else None) if first else None,
                stale_current=any(set(o.sources) & s and not set(o.sources) & n for o in current),
                history_in_place=any(set(o.sources) & s and set(o.sources) & n for o in current),
                stale_raw_left_active=sum(1 for f in stale_cited if f in mem and not mem[f].retired),
                stale_raw_total=len(stale_cited),
            )
        )

    leaks: list[RawLeak] = []
    for r in direct:
        target = mem[r.memory_id]
        kept = set(mem[r.successor_id].sources) if r.successor_id and r.successor_id in mem else set()
        standing = {f for o in current for f in o.sources}
        candidates = [f for f in dict.fromkeys(target.sources) if f not in kept and f not in standing and f in mem]
        active = [mem[f] for f in candidates if not mem[f].retired]
        leaks.append(
            RawLeak(
                observation_id=r.memory_id,
                candidates=len(candidates),
                left_active=len(active),
                left_active_details=[
                    f"{f.fact_type}{' occurred_start' if f.occurred_start else ''}: {f.text[:80]}" for f in active
                ],
            )
        )

    uncertain = [o.id for o in observations if o.status == "uncertain"]
    types: dict[str, int] = {}
    for rel in state.relations:
        types[rel.relation_type] = types.get(rel.relation_type, 0) + 1
    false = [r for r in retirements if r.protected]
    return Classification(
        changes=outcomes,
        retirements=retirements,
        structural_retirements=sum(1 for m in mem.values() if m.retired and not m.truth_retired),
        uncertain=uncertain,
        conflict_relations=types.get("conflicts", 0),
        relation_types=types,
        false_retirement=bool(false),
        false_retirements=false,
        raw_leaks=leaks,
        observations_total=len(observations),
        observations_current=len(current),
        observations_deleted=len(set(snapshot_ids) - set(mem)),
    )


# --- the run record and the report ----------------------------------------------------


class RunCost(BaseModel):
    calls: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    llm_seconds: float
    wall_seconds: float
    judge_calls: int
    judge_input_tokens: int
    judge_output_tokens: int


class RunRecord(BaseModel):
    scenario: str
    arm: str
    rep: int
    bank_id: str
    kind: Kind
    multi_wave: bool
    judged_passed: bool
    traps_hit: list[str]
    missed: list[str]
    problems: list[str]
    cost: RunCost
    lifecycle: Classification


#: Server and judge prices, USD per MTok (Haiku 5.5 / Sonnet 5.5, prompts <= 100K tokens).
#: Cache reads bill at 0.1x input; the trace's input count includes them.
_SERVER_PRICE = (0.10, 0.50)
_JUDGE_PRICE = (2.00, 10.00)


@dataclass(frozen=True)
class Spend:
    server: float
    judge: float


def spend(c: RunCost) -> Spend:
    return Spend(
        server=(
            (c.input_tokens - c.cached_tokens) * _SERVER_PRICE[0]
            + c.cached_tokens * _SERVER_PRICE[0] * 0.1
            + c.output_tokens * _SERVER_PRICE[1]
        )
        / 1e6,
        judge=(c.judge_input_tokens * _JUDGE_PRICE[0] + c.judge_output_tokens * _JUDGE_PRICE[1]) / 1e6,
    )


def _binom_cdf(x: int, n: int, p: float) -> float:
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))


def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper bound of a binomial rate: the p with P(X <= x | n, p) = alpha."""
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    lo, hi = x / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(x, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> Interval:
    """Two-sided (1 - alpha) interval, from the two one-sided bounds at alpha / 2."""
    if n == 0:
        return Interval(0.0, 1.0)
    upper = clopper_pearson_upper(x, n, alpha / 2)
    # The lower bound of x successes is one minus the upper bound of n - x failures.
    return Interval(0.0 if x == 0 else 1 - clopper_pearson_upper(n - x, n, alpha / 2), upper)


@dataclass
class _Tally:
    tp: int = 0
    fp: int = 0
    expected: int = 0
    hit: int = 0
    fp_details: list[str] = field(default_factory=list)


def _rate(x: int, n: int) -> str:
    if n == 0:
        return "n/a (0)"
    ci = clopper_pearson(x, n)
    return f"{x}/{n} = {x / n:.2f} [{ci.low:.2f}, {ci.high:.2f}]"


def report(records: list[RunRecord]) -> str:
    lines: list[str] = []
    arms = sorted({r.arm for r in records})
    lines.append("## Per arm: runs, judged pass, cost\n")
    lines.append(
        "| arm | runs | judged pass | trap hit | missed must | det. problems | calls | input tok | "
        "cached | output tok | server $ | judge calls | judge $ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        rs = [r for r in records if r.arm == arm]
        usd = [spend(r.cost) for r in rs]
        lines.append(
            f"| {arm} | {len(rs)} | {sum(r.judged_passed for r in rs)} | {sum(bool(r.traps_hit) for r in rs)} | "
            f"{sum(bool(r.missed) for r in rs)} | {sum(bool(r.problems) for r in rs)} | "
            f"{sum(r.cost.calls for r in rs)} | {sum(r.cost.input_tokens for r in rs)} | "
            f"{sum(r.cost.cached_tokens for r in rs)} | {sum(r.cost.output_tokens for r in rs)} | "
            f"{sum(u.server for u in usd):.3f} | {sum(r.cost.judge_calls for r in rs)} | {sum(u.judge for u in usd):.3f} |"
        )

    for arm in arms:
        rs = [r for r in records if r.arm == arm]
        lines.append(
            f"\n## Arm {arm}: lifecycle actions (unit: expected change for recall, retirement for precision)\n"
        )
        tallies: dict[str, _Tally] = {a: _Tally() for a in ("supersede", "refute", "conflict")}
        for r in rs:
            cl = r.lifecycle
            labels = {c.label: c.action for c in cl.changes}
            for c in cl.changes:
                tallies[c.action].expected += 1
                tallies[c.action].hit += c.hit
            for e in cl.retirements:
                if e.fact_type != "observation" or e.retired_via not in _DIRECT_KIND:
                    continue
                kind = _DIRECT_KIND[e.retired_via]
                if any(labels[m] == kind for m in e.matched):
                    tallies[kind].tp += 1
                else:
                    tallies[kind].fp += 1
                    tallies[kind].fp_details.append(f"{r.scenario}#{r.rep}: {e.text[:90]!r} matched={e.matched}")
            conflict_changes = [c for c in cl.changes if c.action == "conflict"]
            conflicts_seen = len(cl.uncertain) + cl.conflict_relations
            if conflicts_seen:
                if any(c.hit for c in conflict_changes):
                    tallies["conflict"].tp += 1
                else:
                    tallies["conflict"].fp += 1
                    tallies["conflict"].fp_details.append(
                        f"{r.scenario}#{r.rep}: {len(cl.uncertain)} uncertain, {cl.conflict_relations} conflict rows"
                    )
        lines.append("| action | recall (hit / expected) [95% CP] | precision (TP / retirements) [95% CP] |")
        lines.append("|---|---|---|")
        for action, t in tallies.items():
            unit = " (unit: run with a conflict)" if action == "conflict" else ""
            lines.append(f"| {action}{unit} | {_rate(t.hit, t.expected)} | {_rate(t.tp, t.tp + t.fp)} |")
        for action, t in tallies.items():
            for d in t.fp_details:
                lines.append(f"- {action} not matching an expected {action}: {d}")

        controls = [r for r in rs if r.kind in CONTROL_KINDS]
        multi = [r for r in controls if r.multi_wave]
        pure = [r for r in controls if r.kind == "control"]
        fr = sum(r.lifecycle.false_retirement for r in controls)
        fr_multi = sum(r.lifecycle.false_retirement for r in multi)
        changes = [r for r in rs if r.kind == "change"]
        collateral = sum(r.lifecycle.false_retirement for r in changes)
        lines.append("\n**False retirement (unit: scenario-run; any protected memory truth-retired)**\n")
        lines.append("| set | false / runs | rate | one-sided 95% CP upper |")
        lines.append("|---|---|---|---|")
        for name, x, n in (
            ("controls (primary)", fr, len(controls)),
            ("controls with >= 2 waves", fr_multi, len(multi)),
            ("pure controls (no expected change)", sum(r.lifecycle.false_retirement for r in pure), len(pure)),
            ("change scenarios (collateral)", collateral, len(changes)),
        ):
            rate = f"{x / n:.3f}" if n else "n/a"
            lines.append(f"| {name} | {x}/{n} | {rate} | {clopper_pearson_upper(x, n):.3f} |")
        for r in rs:
            for e in r.lifecycle.false_retirements:
                lines.append(
                    f"- false retirement {r.scenario}#{r.rep} ({e.fact_type}, {e.retired_via}): {e.text[:100]!r}"
                )
        # Off mode has no claims, so its way of dropping a belief is deleting the observation.
        deleted = [r for r in controls if r.lifecycle.observations_deleted]
        lines.append(
            f"- control runs where an observation was hard-deleted between waves (the off-mode analog): "
            f"{_rate(len(deleted), len(controls))}"
        )
        retiring = [c for r in rs for c in r.lifecycle.changes if c.action != "conflict"]
        lines.append(
            "- stale belief still current (a current observation cites a stale fact and no new one), per expected "
            f"supersede/refute: {_rate(sum(c.stale_current for c in retiring), len(retiring))}"
        )

        leaks = [lk for r in rs for lk in r.lifecycle.raw_leaks]
        leaked = [lk for lk in leaks if lk.left_active]
        cand = sum(lk.candidates for lk in leaks)
        left = sum(lk.left_active for lk in leaks)
        stale_left = sum(c.stale_raw_left_active for r in rs for c in r.lifecycle.changes if c.hit_any_kind)
        stale_tot = sum(c.stale_raw_total for r in rs for c in r.lifecycle.changes if c.hit_any_kind)
        lines.append("\n**Raw-fact leak (a directly retired observation's unshared sources left active)**\n")
        lines.append(f"- retired observations with >= 1 source left active: {_rate(len(leaked), len(leaks))}")
        lines.append(f"- source facts left active: {_rate(left, cand)}")
        lines.append(f"- stale facts of a retired change left active: {_rate(stale_left, stale_tot)}")
        details = sorted({d for lk in leaked for d in lk.left_active_details})
        for d in details[:20]:
            lines.append(f"  - {d}")

    lines.append("\n## Per scenario (each arm: runs passing judge; lifecycle per run)\n")
    lines.append(
        "| scenario | kind | arm | judged pass | per run: change hits (H=hit, h=other kind, -=miss, "
        "S=stale still current), F=false retirement |"
    )
    lines.append("|---|---|---|---|---|")
    for scenario in sorted({r.scenario for r in records}):
        for arm in arms:
            rs = sorted((r for r in records if r.scenario == scenario and r.arm == arm), key=lambda r: r.rep)
            if not rs:
                continue
            runs = []
            for r in rs:
                marks = "".join(
                    ("H" if c.hit else "h" if c.hit_any_kind else "-") + ("S" if c.stale_current else "")
                    for c in r.lifecycle.changes
                )
                runs.append((marks or "·") + (" F" if r.lifecycle.false_retirement else ""))
            lines.append(
                f"| {scenario} | {rs[0].kind} | {arm} | {sum(r.judged_passed for r in rs)}/{len(rs)} | "
                f"{' / '.join(runs)} |"
            )
    return "\n".join(lines)


def load(path: Path) -> list[RunRecord]:
    return [RunRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    """``python -m hindsight_system_evals.lifecycle_eval RUNS.jsonl [...]`` prints the markdown report."""
    records = [r for p in sys.argv[1:] for r in load(Path(p))]
    print(report(records))
    print("\n<!-- summary " + json.dumps({"runs": len(records)}) + " -->")


if __name__ == "__main__":
    main()
