"""The lifecycle classifier and its interval arithmetic, on hand-built bank states (no server)."""

from __future__ import annotations

import pytest

from hindsight_system_evals.consolidation import Fact, Scenario, Wave
from hindsight_system_evals.lifecycle_eval import (
    BankState,
    Claim,
    ExpectedChange,
    FactRef,
    LifecycleSpec,
    MemoryState,
    classify,
    clopper_pearson,
    clopper_pearson_upper,
)

_OLD = "The service listens on port 8443."
_NEW = "The service moved to port 9443."
_OTHER = "The service is written in Go."
_SCENARIO = Scenario(
    id="s",
    origin="test",
    waves=(Wave((Fact(_OLD, "2026-01-01"), Fact(_OTHER, "2026-01-01"))), Wave((Fact(_NEW, "2026-02-01"),))),
)
_CHANGE = LifecycleSpec("change", (ExpectedChange("port", "supersede", (FactRef(1, _OLD),), (FactRef(2, _NEW),)),))


def _raw(memory_id: str, doc: str, text: str, claim: Claim | None = None) -> MemoryState:
    return MemoryState(id=memory_id, fact_type="world", text=text, document_id=doc, claim=claim)


def _obs(memory_id: str, sources: list[str], claim: Claim | None = None) -> MemoryState:
    return MemoryState(id=memory_id, fact_type="observation", text=memory_id, sources=sources, claim=claim)


def _state(*memories: MemoryState) -> BankState:
    return BankState(memories={m.id: m for m in memories}, relations=[])


def _superseded(via: str, successor: str | None = None) -> Claim:
    return Claim(truth_status="superseded", retired_via=via, superseded_by_id=successor)


def test_cp_bounds_match_the_closed_forms() -> None:
    # x = 0 has the closed form 1 - alpha ** (1 / n).
    assert clopper_pearson_upper(0, 51) == pytest.approx(1 - 0.05 ** (1 / 51), abs=1e-9)
    ci = clopper_pearson(5, 10)
    assert ci.low == pytest.approx(0.1871, abs=1e-4) and ci.high == pytest.approx(0.8129, abs=1e-4)
    assert clopper_pearson_upper(3, 3) == 1.0 and clopper_pearson_upper(0, 0) == 1.0


def test_a_correct_supersede_is_a_hit_with_its_stale_raw_fact_inherited() -> None:
    state = _state(
        _raw("old", "s-w1-0", _OLD, _superseded("inherited", "succ")),
        _raw("other", "s-w1-1", _OTHER),
        _raw("new", "s-w2-0", _NEW),
        _obs("target", ["old"], _superseded("supersedes", "succ")),
        _obs("go", ["other"]),
        _obs("succ", ["new"]),
    )
    cl = classify(_SCENARIO, _CHANGE, state)
    [change] = cl.changes
    assert change.hit and change.successor_cites_new and not change.stale_current
    assert not cl.false_retirement
    assert [(lk.candidates, lk.left_active) for lk in cl.raw_leaks] == [(1, 0)]


def test_a_neighbour_retired_is_false_and_a_skipped_inheritance_is_a_leak() -> None:
    state = _state(
        _raw("old", "s-w1-0", _OLD),
        _raw("other", "s-w1-1", _OTHER),
        _raw("new", "s-w2-0", _NEW),
        _obs("target", ["old"], _superseded("supersedes", "succ")),
        _obs("go", ["other"], _superseded("supersedes", "succ")),
        _obs("succ", ["new"]),
    )
    cl = classify(_SCENARIO, _CHANGE, state)
    assert cl.false_retirement and [e.memory_id for e in cl.false_retirements] == ["go"]
    assert sum(lk.left_active for lk in cl.raw_leaks) == 2


def test_wrong_kind_and_merges_and_controls() -> None:
    refuted = Claim(truth_status="refuted", retired_via="refutes", superseded_by_id="succ")
    state = _state(
        _raw("old", "s-w1-0", _OLD),
        _raw("other", "s-w1-1", _OTHER),
        _raw("new", "s-w2-0", _NEW),
        _obs("target", ["old"], refuted),
        _obs("dup", ["other"], _superseded("merges")),
        _obs("succ", ["new"]),
    )
    cl = classify(_SCENARIO, _CHANGE, state)
    [change] = cl.changes
    assert not change.hit and change.hit_any_kind and change.observed == ["refute"]
    # A merge retires the row, not its truth: never a false retirement.
    assert cl.structural_retirements == 1 and not cl.false_retirement
    # The same state graded as a control: the refuted target is now protected.
    assert classify(_SCENARIO, LifecycleSpec("control"), state).false_retirement
