"""Lifecycle consolidation (adaptive memory, lifecycle mode): labels in, claim transitions out.

The store's claim writers are covered in test_adaptive_claims_store.py and the recall views in the
recall tests; these drive consolidation itself:

- pure: the label normalisation and every validation rejection (``lifecycle.plan_lifecycle``), the
  conflict and fingerprint rules, the triage redirect, the response model (lenient, enum-pinned,
  strict-schema valid, its output-token overhead), the prompt variant, and off/observe/rank prompt
  and schema byte identity against the base-SHA golden;
- pg0, through ``run_consolidation_job`` with a scripted LLM that answers through the request's own
  response model: every action type and the claim state, relations, evidence and raw inheritance
  each writes; the pool's status rendering and budget; the triage guard; dedup against retired
  twins; the cap; observe-mode shadow records.

Seeding uses raw SQL for exact timestamps, tags and metadata, so the module is
``memory_backend_incompatible``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.config import _get_raw_config, get_config
from hindsight_api.engine.adaptive import clock
from hindsight_api.engine.adaptive.models import TruthStatus
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.adaptive.views import ClaimView, MemoryLifecycle
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation import lifecycle as claim_lifecycle
from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _build_response_model,
    _CreateAction,
    _DeleteAction,
    _LifecycleCreateAction,
    _LifecycleUpdateAction,
    _UpdateAction,
    run_consolidation_job,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.consolidation.triage import TriageResult
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import count_tokens
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.response_models import LLMCallResult, MemoryFact, TokenUsage
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.structured_output import provider_json_schema, strict_json_schema

pytestmark = pytest.mark.memory_backend_incompatible

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)
NOW = DAY0 + timedelta(days=400)
ACTIVE, UNCERTAIN = TruthStatus.ACTIVE, TruthStatus.UNCERTAIN
SUPERSEDED, REFUTED = TruthStatus.SUPERSEDED, TruthStatus.REFUTED
#: The base-SHA (5331da2f4) consolidation system prompt and response schemas, generated from that
#: checkout: what every mode but lifecycle must still send, byte for byte.
GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "consolidation_off_mode_golden.json").read_text())


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _settings(**overrides: Any) -> AdaptiveSettings:
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": "lifecycle"}, get_config(), store_supports=True)
    return dataclasses.replace(settings, **overrides)


@pytest.fixture(autouse=True)
def _observations_on_and_dedup_off(monkeypatch):
    """Consolidation runs; semantic dedup (an LLM verdict per near-twin) only where a test turns it on."""
    config = _get_raw_config()
    monkeypatch.setattr(config, "enable_observations", True)
    monkeypatch.setattr(config, "consolidation_dedup_threshold", 1.0)


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(clock, "utcnow", lambda: NOW)


# =========================================================================== pure: the plan

SCOPE = frozenset[str]()


def _entry(
    entry_id: str,
    text: str = "An observation.",
    *,
    status: TruthStatus = ACTIVE,
    tags: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
    mentioned_at: datetime | None = None,
    successor: str | None = None,
    fingerprint: str | None = None,
    valid_from: datetime | None = None,
) -> claim_lifecycle.PoolEntry:
    return claim_lifecycle.PoolEntry(
        id=entry_id,
        text=text,
        status=status,
        successor=successor,
        tags=frozenset(tags),
        source_ids=sources,
        mentioned_at=mentioned_at if mentioned_at is not None else day(10),
        valid_from=valid_from,
        claim_fingerprint=fingerprint,
        retirable=status in (ACTIVE, UNCERTAIN) and frozenset(tags) == SCOPE,
    )


def _f(fact_id: str, at: datetime | None = None, **metadata: str) -> dict[str, Any]:
    return {"id": fact_id, "text": fact_id, "mentioned_at": at if at is not None else day(40), "metadata": metadata}


def _drops() -> claim_lifecycle.Drops:
    return claim_lifecycle.Drops(bank_id="b")


def _plan(
    drops: claim_lifecycle.Drops,
    result: _BatchLLMResult,
    pool: list[claim_lifecycle.PoolEntry],
    facts: list[dict[str, Any]],
    *,
    settings: AdaptiveSettings | None = None,
    target_sources: dict[str, list[Any]] | None = None,
) -> claim_lifecycle.LifecyclePlan:
    """Plan ``result`` against ``pool``; ``drops`` collects what validation dropped."""
    return claim_lifecycle.plan_lifecycle(
        result,
        claim_lifecycle.PlanInputs(
            pool={e.id: e for e in pool},
            facts={f["id"]: f for f in facts},
            scope_tags=SCOPE,
            settings=settings or _settings(),
            now=NOW,
            target_sources=target_sources or {},
        ),
        drops,
        new_create=lambda text, ids, reason: _CreateAction(text=text, source_fact_ids=ids, reason=reason),
    )


def _update(target: str, text: str, facts: list[str], **labels: Any) -> _LifecycleUpdateAction:
    return _LifecycleUpdateAction(observation_id=target, text=text, source_fact_ids=facts, **labels)


def _create(text: str, facts: list[str], **labels: Any) -> _LifecycleCreateAction:
    return _LifecycleCreateAction(text=text, source_fact_ids=facts, **labels)


def test_a_labelled_update_becomes_a_successor_create_and_leaves_the_target_as_it_is():
    result = _BatchLLMResult(updates=[_update("o1", "Runs Java 25.", ["f1"], relation="supersedes")])
    drops = _drops()
    plan = _plan(drops, result, [_entry("o1", "Runs Java 21.", sources=("s1", "s2"))], [_f("f1", day(40))])

    assert result.updates == []
    (successor,) = result.creates
    assert (successor.text, successor.source_fact_ids) == ("Runs Java 25.", ["f1"])
    successor_plan = plan.for_create(successor)
    assert successor_plan is not None
    assert successor_plan.retires == [
        claim_lifecycle.RetireTarget(target_id="o1", status=SUPERSEDED, relation="supersedes", explicit_sources=None)
    ]
    assert (successor_plan.t_star, successor_plan.keeps) == (day(40), [])
    assert not drops.counts


def test_retires_and_keeps_name_only_the_targets_sources_and_an_empty_list_means_the_default_rule():
    named = _update(
        "o1",
        "Runs Java 25.",
        ["f1"],
        relation="refutes",
        retires_source_ids=["s1", "zz"],
        keeps_source_ids=["s2", "yy"],
    )
    omitted = _update("o2", "Uses Redis 8.", ["f1"], relation="supersedes", retires_source_ids=[])
    result = _BatchLLMResult(updates=[named, omitted])
    drops = _drops()
    plan = _plan(drops, result, [_entry("o1", sources=("s1", "s2")), _entry("o2", sources=("s3",))], [_f("f1")])

    plans = [plan.for_create(c) for c in result.creates]
    assert [p.retires[0].explicit_sources for p in plans if p] == [("s1",), None]
    assert plans[0] is not None and plans[0].keeps == ["s2"] and plans[0].retires[0].status is REFUTED
    assert drops.counts["source_not_of_target"] == 2


def test_a_delete_of_a_retirement_target_is_absorbed_and_a_plain_delete_retires_without_a_successor():
    result = _BatchLLMResult(
        creates=[_create("Runs Java 25.", ["f1"], supersedes=["o1"])],
        deletes=[
            _DeleteAction(observation_id="o1"),
            _DeleteAction(observation_id="o2"),
            _DeleteAction(observation_id="o3", merged_into="o4"),
        ],
    )
    plan = _plan(_drops(), result, [_entry(i) for i in ("o1", "o2", "o3", "o4")], [_f("f1")])

    assert [d.observation_id for d in result.deletes] == ["o3"]  # the merge stays a merge
    assert plan.plain_retires == ["o2"]
    assert plan.retire_target_ids() == {"o1"}


def test_an_update_and_a_supersede_of_one_target_keep_the_supersede_and_its_facts():
    successor = _create("Runs Java 25.", ["f1"], supersedes=["o1"])
    result = _BatchLLMResult(
        creates=[successor], updates=[_UpdateAction(observation_id="o1", text="X", source_fact_ids=["f2"])]
    )
    plan = _plan(_drops(), result, [_entry("o1")], [_f("f1", day(40)), _f("f2", day(35))])

    assert result.updates == []
    assert successor.source_fact_ids == ["f1", "f2"]
    successor_plan = plan.for_create(successor)
    assert successor_plan is not None and successor_plan.cited == ["f1", "f2"] and successor_plan.t_star == day(35)


def test_updates_of_a_retired_observation_are_old_news_except_a_historical_attach():
    retired = _entry("o1", "Runs Java 21.", status=SUPERSEDED, successor="o2")
    result = _BatchLLMResult(
        updates=[
            _UpdateAction(observation_id="o1", text="Runs Java 21.", source_fact_ids=["f1"]),
            _update("o1", "Runs Java 21.", ["f2"], relation="historical"),
        ]
    )
    drops = _drops()
    plan = _plan(drops, result, [retired, _entry("o2")], [_f("f1"), _f("f2")])

    assert result.updates == [] and result.creates == []
    assert plan.historical == [claim_lifecycle.HistoricalAttach(target_id="o1", fact_ids=("f2",))]
    assert plan.acted_on == {"f1"}
    assert drops.counts["target_not_active"] == 1


def test_historical_onto_a_live_observation_is_ordinary_evidence():
    update = _update("o1", "Runs Java 25.", ["f1"], relation="historical")
    result = _BatchLLMResult(updates=[update])
    plan = _plan(_drops(), result, [_entry("o1")], [_f("f1")])
    assert result.updates == [update] and update.relation is None and not plan.historical


def test_an_unchanged_text_under_a_retiring_label_is_ordinary_evidence():
    update = _update("o1", "Runs  Java 21.", ["f1"], relation="supersedes")
    result = _BatchLLMResult(updates=[update])
    _plan(_drops(), result, [_entry("o1", "Runs Java 21.")], [_f("f1")])
    assert result.updates == [update] and update.relation is None and result.creates == []


@dataclass
class _Rejection:
    pool: list[claim_lifecycle.PoolEntry]
    facts: list[dict[str, Any]]
    result: Callable[[], _BatchLLMResult]
    reason: str
    #: What survives: the successor as a plain create, nothing at all, or history.
    survives: str
    target_sources: dict[str, list[Any]] = field(default_factory=dict)


_REJECTIONS = {
    "target_not_in_pool": _Rejection(
        [],
        [_f("f1")],
        lambda: _BatchLLMResult(creates=[_create("New.", ["f1"], supersedes=["nope"])]),
        "target_not_in_pool",
        "plain_create",
    ),
    "target_not_active": _Rejection(
        [_entry("o1", status=REFUTED)],
        [_f("f1")],
        lambda: _BatchLLMResult(creates=[_create("New.", ["f1"], supersedes=["o1"])]),
        "target_not_active",
        "plain_create",
    ),
    "scope_tags": _Rejection(
        [_entry("o1", tags=("team:other",))],
        [_f("f1")],
        lambda: _BatchLLMResult(updates=[_update("o1", "New.", ["f1"], relation="supersedes")]),
        "scope_tags",
        "plain_create",
    ),
    "scope_fingerprint": _Rejection(
        [_entry("o1", fingerprint="environment=prod")],
        [_f("f1", environment="test")],
        lambda: _BatchLLMResult(updates=[_update("o1", "New.", ["f1"], relation="supersedes")]),
        "scope_fingerprint",
        "plain_create",
    ),
    "scope_fingerprint_from_sources": _Rejection(
        [_entry("o1", sources=("s1",))],
        [_f("f1", environment="test")],
        lambda: _BatchLLMResult(creates=[_create("New.", ["f1"], refutes=["o1"])]),
        "scope_fingerprint",
        "plain_create",
        target_sources={"o1": [SimpleNamespace(metadata={"environment": "prod"}, mentioned_at=day(10))]},
    ),
    "recency_on_a_live_target": _Rejection(
        [_entry("o1", mentioned_at=day(30))],
        [_f("f1", day(20))],
        lambda: _BatchLLMResult(updates=[_update("o1", "Old state.", ["f1"], relation="supersedes")]),
        "recency",
        "nothing",
    ),
    "recency_before_the_targets_validity": _Rejection(
        [_entry("o1", mentioned_at=None, valid_from=day(50))],
        [_f("f1", day(40))],
        lambda: _BatchLLMResult(creates=[_create("Old state.", ["f1"], supersedes=["o1"])]),
        "recency",
        "nothing",
    ),
    "self_reference": _Rejection(
        [_entry("o1")],
        [_f("f1")],
        lambda: _BatchLLMResult(creates=[_create("New.", ["f1"], supersedes=["o1"], refutes=["o1"])]),
        "self_reference",
        "successor",
    ),
}


@pytest.mark.parametrize("case", list(_REJECTIONS))
def test_every_validation_rejection_drops_the_label_and_keeps_the_rest(case):
    rejection = _REJECTIONS[case]
    result = rejection.result()
    drops = _drops()
    plan = _plan(drops, result, rejection.pool, rejection.facts, target_sources=rejection.target_sources)

    assert drops.counts[rejection.reason] == 1
    labelled = [c for c in result.creates if (p := plan.for_create(c)) is not None and p.labelled]
    if rejection.survives == "plain_create":
        assert len(result.creates) == 1 and not labelled
    elif rejection.survives == "nothing":
        assert result.creates == [] and plan.acted_on == {"f1"}
    else:
        assert len(labelled) == 1


def test_an_older_fact_naming_a_retired_target_becomes_history_of_it():
    result = _BatchLLMResult(creates=[_create("Runs Java 21.", ["f1"], supersedes=["o1"])])
    plan = _plan(
        _drops(), result, [_entry("o1", status=SUPERSEDED, successor="o2", mentioned_at=day(30))], [_f("f1", day(5))]
    )
    assert result.creates == []
    assert plan.historical == [claim_lifecycle.HistoricalAttach(target_id="o1", fact_ids=("f1",))]


def test_a_target_retired_twice_keeps_the_first_successor_and_the_second_becomes_a_claim_of_its_own():
    result = _BatchLLMResult(
        creates=[_create("Runs Java 26.", ["f2"], supersedes=["o1"])],
        updates=[_update("o1", "Runs Java 25.", ["f1"], relation="supersedes")],
    )
    drops = _drops()
    plan = _plan(drops, result, [_entry("o1")], [_f("f1"), _f("f2")])
    winners = [c.text for c in result.creates if (p := plan.for_create(c)) is not None and p.retires]
    assert winners == ["Runs Java 25."]  # updates are planned first: the first claim wins
    assert {c.text for c in result.creates} == {"Runs Java 25.", "Runs Java 26."}
    assert drops.counts["duplicate_retirement"] == 1


def test_two_creates_naming_one_split_from_split_it_and_one_alone_refines_it():
    split = _BatchLLMResult(
        creates=[_create("Facet A.", ["f1"], split_from="o1"), _create("Facet B.", ["f2"], split_from="o1")]
    )
    plan = _plan(_drops(), split, [_entry("o1")], [_f("f1"), _f("f2")])
    assert [plan.for_create(c).split_from for c in split.creates] == ["o1", "o1"]  # type: ignore[union-attr]

    refine = _BatchLLMResult(creates=[_create("Facet A.", ["f1"], split_from="o1")])
    plan = _plan(_drops(), refine, [_entry("o1")], [_f("f1")])
    child = plan.for_create(refine.creates[0])
    assert child is not None and child.split_from is None and child.refines == ["o1"]


def test_a_split_parent_also_superseded_keeps_the_first_claim():
    result = _BatchLLMResult(
        creates=[
            _create("Replacement.", ["f3"], supersedes=["o1"]),
            _create("Facet A.", ["f1"], split_from="o1"),
            _create("Facet B.", ["f2"], split_from="o1"),
        ]
    )
    drops = _drops()
    plan = _plan(drops, result, [_entry("o1")], [_f("f1"), _f("f2"), _f("f3")])
    assert plan.retire_target_ids() == {"o1"}
    assert all(p.split_from is None for c in result.creates if (p := plan.for_create(c)) is not None)
    assert drops.counts["duplicate_retirement"] == 2


@pytest.mark.parametrize(
    ("new", "existing", "policy", "winner"),
    [
        ((0.0, day(40)), (0.0, day(10)), "evidence", None),  # no trust differential: never auto-resolves
        ((0.9, day(40)), (0.1, day(10)), "evidence", "new"),
        ((0.9, day(5)), (0.1, day(10)), "evidence", None),  # more trusted but older: no winner
        ((0.1, day(5)), (0.9, day(10)), "evidence", "existing"),
        ((0.0, day(40)), (0.9, day(10)), "latest", "new"),
        ((0.5, None), (0.5, day(10)), "latest", None),
    ],
)
def test_a_conflict_needs_a_higher_trust_tier_and_newer_evidence_unless_the_policy_is_latest(
    new, existing, policy, winner
):
    outcome = claim_lifecycle.resolve_conflict(
        new_trust=new[0], new_newest=new[1], existing_trust=existing[0], existing_newest=existing[1], policy=policy
    )
    assert outcome.winner == winner
    assert outcome.new_confidence + outcome.existing_confidence == pytest.approx(1.0)


def test_the_scope_fingerprint_is_set_only_when_every_cited_fact_agrees():
    keys = ("environment", "service", "module")
    assert claim_lifecycle.scope_fingerprint([{"environment": "prod", "service": "api"}] * 2, keys) == (
        "environment=prod,service=api"
    )
    assert claim_lifecycle.scope_fingerprint([{"environment": "prod"}, {"environment": "test"}], keys) is None
    assert claim_lifecycle.scope_fingerprint([{"environment": "prod"}, {}], keys) is None
    assert claim_lifecycle.scope_fingerprint([{"other": "x"}], keys) is None
    long = claim_lifecycle.scope_fingerprint([{"module": "m" * 2000}], keys)
    assert long is not None and long.startswith("sha256:") and len(long) < 100


def test_triage_never_lands_a_restatement_on_an_observation_the_response_retires():
    pool = [MemoryFact(id="o1", text="Runs Java 21.", fact_type="observation")]
    result = _BatchLLMResult(updates=[_update("o1", "Runs Java 25.", ["f1"], relation="supersedes")])
    consolidator._apply_triage(result, TriageResult(restates={"f2": "o1"}), pool, retiring=["o1"])
    assert [u.source_fact_ids for u in result.updates] == [["f1"]]


# =========================================================================== pure: schema and prompt


def _walk_strict(schema: dict[str, Any], path: str = "schema") -> None:
    """OpenAI strict-mode rules, as hindsight-system-tests/validation.py enforces them on requests."""
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, path
        assert set(schema.get("properties") or {}) == set(schema.get("required") or []), path
        for name, child in (schema.get("properties") or {}).items():
            _walk_strict(child, f"{path}.{name}")
    if isinstance(schema.get("items"), dict):
        _walk_strict(schema["items"], f"{path}[]")
    for keyword in ("anyOf", "oneOf", "allOf"):
        for index, child in enumerate(schema.get(keyword) or []):
            _walk_strict(child, f"{path}.{keyword}[{index}]")
    for name, child in (schema.get("$defs") or {}).items():
        _walk_strict(child, f"{path}.$defs.{name}")


def _lifecycle_model(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        max_creates=3, fact_ids=["f1", "f2"], lifecycle=True, target_ids=["o1", "o2"], source_ids=["s1"]
    )
    kwargs.update(overrides)
    return _build_response_model(**kwargs)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with every ``$ref`` to a non-object ``$defs`` entry replaced by that entry, and
    those entries dropped: the shape a schema has when nothing is shared between fields."""
    defs = schema.get("$defs") or {}
    shared = {f"#/$defs/{name}": d for name, d in defs.items() if d.get("type") != "object"}

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {"$ref"} and node["$ref"] in shared:
                return dict(shared[node["$ref"]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    out = inline(schema)
    out["$defs"] = {name: d for name, d in out["$defs"].items() if f"#/$defs/{name}" not in shared}
    return out


def test_the_lifecycle_schema_is_strict_valid_and_pins_every_label_to_what_was_shown():
    schema = strict_json_schema(_lifecycle_model())
    _walk_strict(schema)
    inline = _inline_refs(schema)
    create = inline["$defs"]["_Create"]["properties"]
    update = inline["$defs"]["_Update"]["properties"]
    for name in ("supersedes", "refutes", "refines", "conflicts_with"):
        assert create[name]["items"]["enum"] == ["o1", "o2"]
    assert create["split_from"]["anyOf"][0]["enum"] == ["o1", "o2"]
    assert create["keeps_source_ids"]["items"]["enum"] == ["s1"]
    assert update["retires_source_ids"]["items"]["enum"] == ["s1"]
    assert create["source_fact_ids"]["items"]["enum"] == update["source_fact_ids"]["items"]["enum"] == ["f1", "f2"]
    assert update["relation"]["anyOf"][0]["enum"] == ["supersedes", "refutes", "historical"]
    # Lifecycle labels come after the base fields, in generation order.
    assert list(create)[:3] == ["text", "source_fact_ids", "reason"]
    # Nothing retirable in the pool: the label lists can only be empty.
    empty = strict_json_schema(_lifecycle_model(target_ids=[], source_ids=[]))["$defs"]["_Create"]["properties"]
    assert empty["supersedes"]["maxItems"] == 0 and "enum" not in empty["supersedes"].get("items", {})


@pytest.mark.parametrize("serialize", [strict_json_schema, provider_json_schema], ids=["strict", "provider"])
def test_the_lifecycle_schema_spells_each_pinned_id_once_and_constrains_exactly_what_inline_pinning_did(
    serialize, monkeypatch
):
    """The schema is input on every call: an id repeated on every label field made it grow several
    times over with the pool. Each id list is one ``$defs`` entry; resolving the references gives
    back the schema with the lists inline, so the constraint is the same."""
    targets = [f"{i:08d}-0000-4000-8000-000000000000" for i in range(25)]
    sources = [f"{i:08d}-1111-4000-8000-000000000000" for i in range(30)]
    facts = [f"{i:08d}-2222-4000-8000-000000000000" for i in range(10)]
    model = _build_response_model(5, fact_ids=facts, lifecycle=True, target_ids=targets, source_ids=sources)
    schema = serialize(model)
    text = json.dumps(schema)
    assert all(text.count(i) == 1 for i in [*targets, *sources, *facts])
    shared = {name: d["enum"] for name, d in schema["$defs"].items() if d.get("type") == "string"}
    assert shared == {"BatchFactId": facts, "RetirableObservationId": targets, "ShownSourceFactId": sources}
    if serialize is strict_json_schema:
        _walk_strict(schema)
    monkeypatch.setattr(consolidator, "_hoist_id_enums", lambda schema, enums, **_: schema)
    inline = serialize(model)
    assert _inline_refs(schema) == inline
    assert count_tokens(json.dumps(schema)) < count_tokens(json.dumps(inline)) / 2


def test_the_lifecycle_model_is_lenient_a_bad_label_is_dropped_not_fatal():
    response = _lifecycle_model().model_validate(
        {
            "creates": [
                {
                    "text": "x",
                    "source_fact_ids": ["f1"],
                    "supersedes": ["o1", 5, ""],
                    "split_from": ["o2"],
                    "refutes": "o2",
                },
                {"text": "y", "source_fact_ids": ["f2"], "keeps_source_ids": {"bad": 1}},
            ],
            "updates": [
                {"observation_id": "o1", "text": "t", "source_fact_ids": ["f2"], "relation": " SUPERSEDES "},
                {"observation_id": "o2", "text": "t", "source_fact_ids": ["f2"], "relation": "rewrites"},
            ],
        }
    )
    first, second = response.creates
    assert (first.supersedes, first.refutes, first.split_from, second.keeps_source_ids) == (["o1"], ["o2"], "o2", [])
    assert [u.relation for u in response.updates] == ["supersedes", None]


def test_the_lifecycle_label_overhead_in_output_tokens(capsys):
    """Measured, not asserted against a target: the strict schema makes every label mandatory to
    emit, so an action with no label still spends tokens on empty ones."""
    base = _build_response_model(3, fact_ids=["f1", "f2"])
    lifecycle_model = _lifecycle_model()
    schema_tokens = {
        "base": count_tokens(json.dumps(strict_json_schema(base))),
        "lifecycle": count_tokens(json.dumps(strict_json_schema(lifecycle_model))),
    }
    empty_create = lifecycle_model.model_validate({"creates": [{"text": "x", "source_fact_ids": ["f1"]}]}).creates[0]
    empty_update = lifecycle_model.model_validate(
        {"updates": [{"observation_id": "o1", "text": "x", "source_fact_ids": ["f1"]}]}
    ).updates[0]
    per_action = {
        "create": count_tokens(empty_create.model_dump_json())
        - count_tokens(_CreateAction(text="x", source_fact_ids=["f1"]).model_dump_json()),
        "update": count_tokens(empty_update.model_dump_json())
        - count_tokens(_UpdateAction(observation_id="o1", text="x", source_fact_ids=["f1"]).model_dump_json()),
    }
    with capsys.disabled():
        print(f"\n[lifecycle schema overhead] schema tokens {schema_tokens}; empty labels per action {per_action}")
    assert 0 < per_action["create"] <= 40 and 0 < per_action["update"] <= 25
    assert schema_tokens["lifecycle"] > schema_tokens["base"]


def test_the_lifecycle_prompt_replaces_the_in_place_rules_and_keeps_the_stubs_headings():
    base = build_consolidation_system_prompt()
    variant = build_consolidation_system_prompt(lifecycle=True)
    assert variant != base
    for heading in ("### Existing observations", "### New facts", "## PROCESSING RULES", "## OUTPUT FORMAT"):
        assert heading in variant
    for replaced in (
        "4. STATE CHANGES — UPDATE CONCISELY",
        "never DELETE them. Only delete an observation when it is restated identically",
        '("favorite is Svelte; earlier it was React")',
        "- `deletes`: only when an observation is directly superseded or contradicted",
        "the MISSION takes priority.",
        "Alice owned a 2019 Honda Civic; sold it on March 15, 2025.",
    ):
        assert replaced in base and replaced not in variant
    for added in (
        'relation: "supersedes"',
        '"relation": "historical"',
        "`split_from`",
        '"retirable": false',
        "`status`",
    ):
        assert added in variant
    # The language rule and the output-language directive compose exactly as for the base prompt.
    assert build_consolidation_system_prompt("Chinese", lifecycle=True).endswith(
        build_consolidation_system_prompt("Chinese")[-200:]
    )


def test_off_mode_prompt_and_schemas_are_byte_identical_to_the_base_sha():
    assert build_consolidation_system_prompt() == GOLDEN["prompts"]["default"]
    assert build_consolidation_system_prompt("Chinese") == GOLDEN["prompts"]["chinese"]
    for name, case in GOLDEN["schemas"].items():
        model = _build_response_model(**case["kwargs"])
        assert model.model_json_schema() == case["schema"], name
        assert strict_json_schema(model) == case["strict"], name


def test_mock_llm_default_consolidation_still_parses_the_lifecycle_model():
    model = _lifecycle_model(fact_ids=["11111111-1111-1111-1111-111111111111"])
    response = MockLLM._build_mock_consolidation(
        [{"role": "user", "content": "### New facts\n[11111111-1111-1111-1111-111111111111] A fact.\n"}], model
    )
    assert isinstance(response, model)
    (create,) = response.creates
    assert create.source_fact_ids == ["11111111-1111-1111-1111-111111111111"] and create.supersedes == []


# =========================================================================== pg0: seeding and the scripted LLM

_SEQ = iter(range(1_000_000))


@asynccontextmanager
async def _conn(memory):
    async with acquire_with_retry(await memory._get_backend()) as conn:
        yield conn


async def _sql(memory, sql: str, *args):
    async with _conn(memory) as conn:
        return await conn.fetch(sql, *args)


async def _bank(memory, request_context, mode: str = "lifecycle", **config: Any) -> str:
    bank_id = f"test-alc-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": mode, **config})
    return bank_id


async def _unit(
    memory,
    bank_id: str,
    text: str,
    *,
    at: datetime,
    fact_type: str = "world",
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    occurred_start: datetime | None = None,
    consolidated: bool = False,
    sources: list[str] | None = None,
) -> str:
    """A memory with exact provenance, embedded so recall finds it; consolidation reads facts in insertion order."""
    memory_id = uuid.uuid4()
    (embedding,) = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text])
    async with _conn(memory) as conn:
        await conn.execute(
            """
            INSERT INTO memory_units
                (id, bank_id, text, fact_type, mentioned_at, event_date, occurred_start, metadata, tags, embedding,
                 consolidated_at, source_memory_ids, proof_count, created_at)
            VALUES ($1, $2, $3, $4, $5, $5, $6, $7::jsonb, $8, $9::vector, $10, $11, $12,
                    now() + make_interval(secs => $13))
            """,
            memory_id,
            bank_id,
            text,
            fact_type,
            at,
            occurred_start,
            json.dumps(metadata or {}),
            tags or [],
            "[" + ",".join(str(v) for v in embedding) + "]",
            day(1) if consolidated else None,
            [uuid.UUID(s) for s in sources] if sources is not None else None,
            len(sources) if sources else 1,
            float(next(_SEQ)) / 1000.0,
        )
    return str(memory_id)


async def _observation(memory, bank_id: str, text: str, sources: list[str], **kwargs: Any) -> str:
    return await _unit(memory, bank_id, text, fact_type="observation", sources=sources, **kwargs)


async def _old_fact(memory, bank_id: str, text: str, **kwargs: Any) -> str:
    return await _unit(memory, bank_id, text, consolidated=True, **kwargs)


async def _retire(memory, bank_id: str, target: str, *, successor: str | None, status=SUPERSEDED, at=day(20)):
    store = get_memories()
    async with _conn(memory) as conn:
        async with conn.transaction():
            await store.lock_claims(
                conn, bank_id=bank_id, memory_ids=[target, *([successor] if successor else [])], skip_locked=False
            )
            assert await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=target,
                to_status=status,
                retired_via="supersedes",
                superseded_by_id=successor,
                valid_to=at,
                reason="seed",
                confidence=None,
            )


@dataclass
class _View:
    """Reads a bank's memories through the engine's read API (``get_memory_unit`` and its
    ``lifecycle`` view, ``list_memory_units``), the way a caller sees them."""

    memory: Any
    bank_id: str
    request_context: Any

    async def unit(self, memory_id: str) -> dict[str, Any] | None:
        return await self.memory.get_memory_unit(self.bank_id, memory_id, request_context=self.request_context)

    async def lifecycle(self, memory_id: str) -> MemoryLifecycle:
        unit = await self.unit(memory_id)
        assert unit is not None and "lifecycle" in unit, memory_id
        return MemoryLifecycle.model_validate(unit["lifecycle"])

    async def claim(self, memory_id: str) -> ClaimView | None:
        return (await self.lifecycle(memory_id)).claim

    async def status(self, memory_id: str) -> str:
        claim = await self.claim(memory_id)
        return "active" if claim is None else claim.truth_status

    async def evidence(self, memory_id: str) -> list[str]:
        claim = await self.claim(memory_id)
        return sorted(claim.evidence_memory_ids) if claim is not None else []

    async def sources(self, observation_id: str) -> list[str]:
        unit = await self.unit(observation_id)
        assert unit is not None, observation_id
        return list(unit.get("source_memory_ids") or [])

    async def text(self, memory_id: str) -> str | None:
        unit = await self.unit(memory_id)
        return None if unit is None else unit["text"]

    async def observations(self) -> dict[str, str]:
        """The bank's observations, text -> id."""
        page = await self.memory.list_memory_units(
            self.bank_id, fact_type="observation", limit=1000, request_context=self.request_context
        )
        return {item["text"]: str(item["id"]) for item in page["items"]}

    async def relations(self) -> list[tuple[str, str | None, str | None]]:
        """Every lifecycle relation touching one of the bank's observations, as (type, source, target)."""
        found: set[tuple[str, str | None, str | None]] = set()
        for observation_id in (await self.observations()).values():
            view = await self.lifecycle(observation_id)
            for relation in [*view.relations_out, *view.relations_in]:
                found.add((relation.relation_type, relation.source_memory_id, relation.target_memory_id))
        return sorted(found, key=str)


async def _stamps(memory, memory_id: str):
    """``updated_at`` and ``consolidated_at``: bookkeeping no read API exposes, which the history
    attach must leave alone (``updated_at``) and the inheritance must set (``consolidated_at``)."""
    rows = await _sql(
        memory, "SELECT updated_at, consolidated_at FROM memory_units WHERE id = $1", uuid.UUID(memory_id)
    )
    return rows[0]


_FACT_LINE = re.compile(r"^\[([0-9a-f-]{36})\] (.*)$")


@dataclass
class _Prompt:
    """One consolidation request, as the scripted LLM saw it."""

    system: str
    user: str
    response_format: type

    def facts(self) -> dict[str, str]:
        section = self.user[self.user.rfind("### New facts") : self.user.rfind("### Existing observations")]
        return {m.group(2): m.group(1) for line in section.splitlines() if (m := _FACT_LINE.match(line))}

    def fact(self, needle: str) -> str:
        (fact_id,) = [fid for text, fid in self.facts().items() if needle in text]
        return fact_id

    def shown(self) -> list[dict[str, Any]]:
        section = self.user[self.user.rfind("### Existing observations") + len("### Existing observations") :].strip()
        return json.loads(section[: section.rindex("]") + 1]) if section.startswith("[") else []

    def obs(self, needle: str) -> dict[str, Any]:
        (entry,) = [o for o in self.shown() if needle in o["text"]]
        return entry


class _ScriptedLLM:
    """Stands in for the consolidation LLM: records each request and answers it through the
    request's own response model, so the model (lifecycle or base) that validates is the real one."""

    def __init__(self, respond: Callable[[_Prompt], dict[str, Any]]):
        self.respond = respond
        self.prompts: list[_Prompt] = []

    def with_config(self, *args: Any, **kwargs: Any) -> "_ScriptedLLM":
        return self

    async def call(self, *, messages, response_format, scope, **kwargs: Any) -> LLMCallResult:
        usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        if scope == "consolidation_dedup":
            return LLMCallResult(content=response_format(action="keep", reason="scripted"), usage=usage)
        prompt = _Prompt(system=messages[0]["content"], user=messages[1]["content"], response_format=response_format)
        self.prompts.append(prompt)
        return LLMCallResult(content=response_format.model_validate(self.respond(prompt)), usage=usage)


@dataclass
class _Run:
    """One consolidation job: what the scripted LLM saw, and the job's result."""

    llm: _ScriptedLLM
    result: dict[str, Any]


async def _consolidate(memory, bank_id: str, request_context, respond) -> _Run:
    llm = _ScriptedLLM(respond)
    with patch.object(memory, "_consolidation_llm_config", llm), patch.object(memory, "submit_async_consolidation"):
        result = await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)
    return _Run(llm=llm, result=result)


def _nothing(_prompt: _Prompt) -> dict[str, Any]:
    return {}


async def _recall_ids(memory, bank_id: str, request_context, query: str, **kwargs: Any) -> set[str]:
    result = await memory.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["world", "observation"],
        max_tokens=4096,
        request_context=request_context,
        **kwargs,
    )
    return {str(r.id) for r in result.results}


# =========================================================================== pg0: action types


@pytest.mark.asyncio
async def test_a_labelled_update_with_no_retires_source_ids_supersedes_and_retires_the_stale_raw_fact(
    memory, request_context
):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    java21 = await _old_fact(memory, bank_id, "The payments service runs Java 21.", at=day(10))
    k8s = await _old_fact(memory, bank_id, "The payments service is deployed on Kubernetes.", at=day(10))
    target = await _observation(memory, bank_id, "The payments service runs Java 21.", [java21], at=day(10))
    await _observation(memory, bank_id, "The payments service is deployed on Kubernetes.", [k8s], at=day(10))
    java25 = await _unit(memory, bank_id, "The payments service was upgraded to Java 25.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        shown = prompt.obs("Java 21")
        assert (shown["status"], shown["retirable"]) == ("active", True)
        assert shown["source_memories"][0]["id"] == java21  # retirable: its source ids are shown
        return {
            "updates": [
                {
                    "observation_id": shown["id"],
                    "text": "The payments service runs Java 25.",
                    "source_fact_ids": [prompt.fact("Java 25")],
                    "relation": "supersedes",
                }
            ]
        }

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert (
        "RECENCY IS `mentioned_at`" in run.llm.prompts[0].system
        and 'relation: "supersedes"' in run.llm.prompts[0].system
    )
    successor = (await v.observations())["The payments service runs Java 25."]

    target_claim = await v.claim(target)
    assert (target_claim.truth_status, target_claim.retired_via) == ("superseded", "supersedes")
    assert (str(target_claim.superseded_by_id), target_claim.valid_to) == (successor, day(40))
    assert await v.text(target) == "The payments service runs Java 21."  # history kept
    assert await v.evidence(target) == [java25]
    assert await v.sources(successor) == [java25]  # never the target's sources
    successor_claim = await v.claim(successor)
    assert (successor_claim.truth_status, successor_claim.valid_from) == ("active", day(40))
    assert (await v.claim(java25)).valid_from == day(40)
    # The stale raw fact stops answering as current; the unrelated one is untouched.
    stale = await v.claim(java21)
    assert (stale.truth_status, stale.retired_via, str(stale.inherited_from_id)) == (
        "superseded",
        "inherited",
        target,
    )
    assert (str(stale.superseded_by_id), stale.valid_to) == (successor, day(40))
    assert await v.status(k8s) == "active"
    assert ("supersedes", successor, target) in await v.relations()
    assert run.result["lifecycle_supersession"] == 1 and run.result["lifecycle_inherited"] == 1
    # Recall: the current view no longer returns the stale fact or the old observation.
    current = await _recall_ids(memory, bank_id, request_context, "Which Java version does the payments service run?")
    assert {java21, target}.isdisjoint(current) and successor in current


@pytest.mark.asyncio
async def test_refute_is_not_supersede_and_the_views_tell_them_apart(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    redis = await _old_fact(memory, bank_id, "The cache is Redis.", at=day(10))
    java = await _old_fact(memory, bank_id, "The billing service runs Java 17.", at=day(10))
    refuted = await _observation(memory, bank_id, "The cache is Redis.", [redis], at=day(10))
    superseded = await _observation(memory, bank_id, "The billing service runs Java 17.", [java], at=day(10))
    await _unit(memory, bank_id, "The cache was never Redis; it has always been Memcached.", at=day(40))
    await _unit(memory, bank_id, "The billing service moved to Java 21.", at=day(41))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        creates = []
        if "Memcached" in "".join(prompt.facts()):
            creates.append(
                {
                    "text": "The cache is Memcached.",
                    "source_fact_ids": [prompt.fact("Memcached")],
                    "refutes": [prompt.obs("Redis")["id"]],
                }
            )
        if "Java 21" in "".join(prompt.facts()):
            creates.append(
                {
                    "text": "The billing service runs Java 21.",
                    "source_fact_ids": [prompt.fact("Java 21")],
                    "supersedes": [prompt.obs("Java 17")["id"]],
                }
            )
        return {"creates": creates}

    await _consolidate(memory, bank_id, request_context, respond)
    assert (await v.status(refuted), await v.status(superseded)) == ("refuted", "superseded")
    assert (await v.status(redis), await v.status(java)) == ("refuted", "superseded")
    historical = await _recall_ids(
        memory, bank_id, request_context, "cache and billing service", lifecycle_view="historical"
    )
    assert superseded in historical and refuted not in historical
    current = await _recall_ids(memory, bank_id, request_context, "cache and billing service")
    assert {superseded, refuted}.isdisjoint(current)


@pytest.mark.asyncio
async def test_a_supersede_chain_a_to_b_to_c(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    v1 = await _old_fact(memory, bank_id, "The API gateway is Kong.", at=day(10))
    a = await _observation(memory, bank_id, "The API gateway is Kong.", [v1], at=day(10))

    def replace_with(new_text: str, needle: str, old_needle: str):
        def respond(prompt: _Prompt) -> dict[str, Any]:
            return {
                "updates": [
                    {
                        "observation_id": prompt.obs(old_needle)["id"],
                        "text": new_text,
                        "source_fact_ids": [prompt.fact(needle)],
                        "relation": "supersedes",
                    }
                ]
            }

        return respond

    await _unit(memory, bank_id, "The API gateway moved to Envoy.", at=day(30))
    await _consolidate(memory, bank_id, request_context, replace_with("The API gateway is Envoy.", "Envoy", "Kong"))
    b = (await v.observations())["The API gateway is Envoy."]
    await _unit(memory, bank_id, "The API gateway moved to Traefik.", at=day(60))

    def second(prompt: _Prompt) -> dict[str, Any]:
        kong = prompt.obs("Kong")
        assert (kong["status"], kong["successor"], kong["retirable"]) == ("superseded", b, False)
        assert "source_memories" not in kong  # history is shown, its evidence is not
        return replace_with("The API gateway is Traefik.", "Traefik", "Envoy")(prompt)

    await _consolidate(memory, bank_id, request_context, second)
    c = (await v.observations())["The API gateway is Traefik."]
    assert (str((await v.claim(a)).superseded_by_id), str((await v.claim(b)).superseded_by_id)) == (
        b,
        c,
    )
    assert ((await v.claim(b)).valid_from, (await v.claim(b)).valid_to) == (day(30), day(60))
    assert await v.status(c) == "active"


@pytest.mark.asyncio
async def test_an_out_of_order_old_fact_is_attached_to_the_retired_observation_as_history(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The team uses Jenkins for CI.", at=day(10))
    new = await _old_fact(memory, bank_id, "The team uses GitHub Actions for CI.", at=day(30))
    retired = await _observation(memory, bank_id, "The team uses Jenkins for CI.", [old], at=day(10))
    successor = await _observation(memory, bank_id, "The team uses GitHub Actions for CI.", [new], at=day(30))
    await _retire(memory, bank_id, retired, successor=successor, at=day(30))
    before = await _stamps(memory, retired)
    late = await _unit(memory, bank_id, "The team said it runs its CI on Jenkins.", at=day(5))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        shown = prompt.obs("Jenkins")
        assert (shown["status"], shown["successor"]) == ("superseded", successor)
        return {
            "updates": [
                {
                    "observation_id": shown["id"],
                    "text": shown["text"],
                    "source_fact_ids": [prompt.fact("Jenkins")],
                    "relation": "historical",
                }
            ]
        }

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert late in await v.sources(retired)
    assert await v.text(retired) == "The team uses Jenkins for CI."
    assert (await _stamps(memory, retired))["updated_at"] == before["updated_at"]  # history moves nothing
    late_claim = await v.claim(late)
    assert (late_claim.truth_status, late_claim.retired_via, str(late_claim.inherited_from_id)) == (
        "superseded",
        "inherited",
        retired,
    )
    assert (str(late_claim.superseded_by_id), late_claim.valid_to) == (successor, day(30))
    assert (await _stamps(memory, late))["consolidated_at"] is not None
    assert run.result["lifecycle_historical_attached"] == 1


@pytest.mark.asyncio
async def test_a_split_retires_the_parent_and_its_raw_sources_which_the_facets_keep_as_evidence(
    memory, request_context
):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The data team owns ingestion and reporting.", at=day(10))
    parent = await _observation(memory, bank_id, "The data team owns ingestion and reporting.", [s1], at=day(10))
    split = await _unit(
        memory, bank_id, "Ingestion moved to the platform team; the data team keeps reporting.", at=day(40)
    )

    def respond(prompt: _Prompt) -> dict[str, Any]:
        fact = prompt.fact("Ingestion moved")
        parent_id = prompt.obs("ingestion and reporting")["id"]
        return {
            "creates": [
                {"text": "The platform team owns ingestion.", "source_fact_ids": [fact], "split_from": parent_id},
                {"text": "The data team owns reporting.", "source_fact_ids": [fact], "split_from": parent_id},
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)
    observations = await v.observations()
    children = [observations["The platform team owns ingestion."], observations["The data team owns reporting."]]
    parent_claim = await v.claim(parent)
    assert (parent_claim.truth_status, parent_claim.retired_via, parent_claim.superseded_by_id) == (
        "superseded",
        "splits",
        None,
    )
    for child in children:
        assert await v.sources(child) == [split]
        assert await v.evidence(child) == [s1]
        assert ("splits", child, parent) in await v.relations()
    raw = await v.claim(s1)
    assert raw is not None
    assert (raw.truth_status, raw.retired_via, raw.superseded_by_id, str(raw.inherited_from_id)) == (
        "superseded",
        "splits",
        None,
        parent,
    )
    assert raw.valid_to == day(40)
    query = "Who owns ingestion and reporting?"
    current = await _recall_ids(memory, bank_id, request_context, query)
    assert {s1, parent}.isdisjoint(current) and set(children) <= current
    historical = await _recall_ids(memory, bank_id, request_context, query, lifecycle_view="historical")
    assert {s1, parent} <= historical


@pytest.mark.asyncio
async def test_a_merge_retires_the_duplicate_into_its_survivor_and_history_does_not_show_it(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "Dana leads the mobile team.", at=day(10))
    s2 = await _old_fact(memory, bank_id, "Dana is the mobile team lead.", at=day(11))
    survivor = await _observation(memory, bank_id, "Dana leads the mobile team.", [s1], at=day(10))
    duplicate = await _observation(memory, bank_id, "Dana is the mobile team lead.", [s2], at=day(11))
    await _unit(memory, bank_id, "Dana still leads mobile.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        keep, gone = prompt.obs("leads the mobile")["id"], prompt.obs("team lead")["id"]
        return {
            "updates": [
                {
                    "observation_id": keep,
                    "text": "Dana leads the mobile team.",
                    "source_fact_ids": [prompt.fact("still")],
                }
            ],
            "deletes": [{"observation_id": gone, "merged_into": keep}],
        }

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert await v.unit(duplicate) is not None  # kept, not hard-deleted
    claim = await v.claim(duplicate)
    assert (claim.truth_status, claim.retired_via, str(claim.superseded_by_id)) == (
        "superseded",
        "merges",
        survivor,
    )
    assert s2 in await v.sources(survivor)
    assert ("merges", survivor, duplicate) in await v.relations()
    assert await v.status(s2) == "active"  # no inheritance for a merge
    assert run.result["observations_deleted"] == 0 and run.result["lifecycle_merge"] == 1
    historical = await _recall_ids(memory, bank_id, request_context, "Who leads mobile?", lifecycle_view="historical")
    assert duplicate not in historical


@pytest.mark.asyncio
async def test_a_plain_delete_retires_without_a_successor(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The office is on Market Street.", at=day(10))
    target = await _observation(memory, bank_id, "The office is on Market Street.", [s1], at=day(10))
    fact = await _unit(memory, bank_id, "The office lease on Market Street ended.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {"deletes": [{"observation_id": prompt.obs("Market")["id"]}]}

    run = await _consolidate(memory, bank_id, request_context, respond)
    claim = await v.claim(target)
    assert (claim.truth_status, claim.retired_via, claim.superseded_by_id, claim.valid_to) == (
        "superseded",
        "supersedes",
        None,
        day(40),
    )
    assert await v.evidence(target) == [fact]
    assert run.result["lifecycle_retire_without_successor"] == 1


@pytest.mark.asyncio
async def test_an_unresolved_conflict_makes_both_sides_uncertain_and_trust_can_settle_it(
    memory, request_context, monkeypatch
):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The launch is on May 3.", at=day(10), metadata={"source": "chat"})
    existing = await _observation(memory, bank_id, "The launch is on May 3.", [s1], at=day(10))
    await _unit(memory, bank_id, "The launch is on May 10.", at=day(40), metadata={"source": "chat"})

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {
            "creates": [
                {
                    "text": "The launch is on May 10.",
                    "source_fact_ids": [prompt.fact("May 10")],
                    "conflicts_with": [prompt.obs("May 3")["id"]],
                }
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)
    new = (await v.observations())["The launch is on May 10."]
    for memory_id in (existing, new):
        claim = await v.claim(memory_id)
        assert (claim.truth_status, claim.confidence) == ("uncertain", pytest.approx(0.5))
    assert ("conflicts", new, existing) in await v.relations()

    # With a trust differential the more trusted, more recent side wins outright.
    monkeypatch.setattr(_get_raw_config(), "adaptive_memory_source_trust", {"crm": 0.9, "chat": 0.1})
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The launch is on May 3.", at=day(10), metadata={"source": "chat"})
    existing = await _observation(memory, bank_id, "The launch is on May 3.", [s1], at=day(10))
    await _unit(memory, bank_id, "The launch is on May 10.", at=day(40), metadata={"source": "crm"})
    await _consolidate(memory, bank_id, request_context, respond)
    new = (await v.observations())["The launch is on May 10."]
    assert (await v.status(existing), await v.status(new)) == ("superseded", "active")
    assert str((await v.claim(existing)).superseded_by_id) == new


# =========================================================================== pg0: validation through the batch


@pytest.mark.asyncio
async def test_divergent_scope_fingerprints_keep_both_claims_current(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    prod = await _old_fact(
        memory, bank_id, "The service runs Postgres 15.", at=day(10), metadata={"environment": "prod"}
    )
    target = await _observation(memory, bank_id, "The service runs Postgres 15.", [prod], at=day(10))
    await _unit(memory, bank_id, "The service runs Postgres 17.", at=day(40), metadata={"environment": "test"})

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {
            "updates": [
                {
                    "observation_id": prompt.obs("Postgres 15")["id"],
                    "text": "The service runs Postgres 17.",
                    "source_fact_ids": [prompt.fact("Postgres 17")],
                    "relation": "supersedes",
                }
            ]
        }

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert await v.status(target) == "active" and await v.status(prod) == "active"
    assert "The service runs Postgres 17." in await v.observations()  # a claim of its own scope
    assert await v.text(target) == "The service runs Postgres 15."
    assert run.result["lifecycle_dropped_scope_fingerprint"] == 1


@pytest.mark.asyncio
async def test_an_observation_of_another_scope_is_shown_but_not_retirable(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "Builds use Gradle.", at=day(10), tags=["team:a", "team:b"])
    wider = await _observation(memory, bank_id, "Builds use Gradle.", [s1], at=day(10), tags=["team:a", "team:b"])
    await _unit(memory, bank_id, "Builds moved to Bazel.", at=day(40), tags=["team:a"])

    def respond(prompt: _Prompt) -> dict[str, Any]:
        shown = prompt.obs("Gradle")
        assert shown["retirable"] is False and "id" not in shown["source_memories"][0]
        return {
            "creates": [
                {"text": "Builds use Bazel.", "source_fact_ids": [prompt.fact("Bazel")], "supersedes": [shown["id"]]}
            ]
        }

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert await v.status(wider) == "active"
    assert run.result["lifecycle_dropped_scope_tags"] == 1


# =========================================================================== pg0: the pool, triage, dedup, the cap


@pytest.mark.asyncio
async def test_a_long_supersession_chain_never_crowds_the_live_successor_out_of_the_pool(
    memory, request_context, monkeypatch
):
    bank_id = await _bank(memory, request_context)
    monkeypatch.setattr(_get_raw_config(), "consolidation_max_tokens", 40)
    chain: list[str] = []
    for version in range(1, 7):
        fact = await _old_fact(memory, bank_id, f"The search cluster runs Elasticsearch {version}.", at=day(version))
        chain.append(
            await _observation(
                memory, bank_id, f"The search cluster runs Elasticsearch {version}.", [fact], at=day(version)
            )
        )
    for older, newer in zip(chain, chain[1:]):
        await _retire(memory, bank_id, older, successor=newer, at=day(10))
    await _unit(memory, bank_id, "The search cluster now runs Elasticsearch 7.", at=day(40))

    run = await _consolidate(memory, bank_id, request_context, _nothing)
    shown = run.llm.prompts[0].shown()
    live = [o for o in shown if o["status"] == "active"]
    retired = [o for o in shown if o["status"] != "active"]
    assert chain[-1] in {o["id"] for o in live}
    # Only the immediate predecessor of a shown successor, once.
    assert [o["id"] for o in retired] == [chain[-2]]
    assert shown.index(next(o for o in shown if o["id"] == chain[-1])) < shown.index(retired[0])


@pytest.mark.asyncio
async def test_retired_near_twins_never_crowd_the_live_observation_out_of_the_over_fetch(
    memory, request_context, monkeypatch
):
    """Retired observations closer to the fact than the live one fill the over-fetched recall; the
    live one is recalled again without them, so it is still shown (and still the one to retire)."""
    bank_id = await _bank(memory, request_context)
    monkeypatch.setattr(_get_raw_config(), "consolidation_max_tokens", 40)
    chain: list[str] = []
    for version in range(1, 16):
        fact = await _old_fact(memory, bank_id, f"The search cluster runs Elasticsearch {version}.", at=day(version))
        chain.append(
            await _observation(
                memory, bank_id, f"The search cluster runs Elasticsearch {version}.", [fact], at=day(version)
            )
        )
    head_fact = await _old_fact(memory, bank_id, "OpenSearch 2 powers search now.", at=day(20))
    head = await _observation(memory, bank_id, "OpenSearch 2 powers search now.", [head_fact], at=day(20))
    for older, newer in zip(chain, [*chain[1:], head]):
        await _retire(memory, bank_id, older, successor=newer, at=day(20))
    await _unit(memory, bank_id, "The search cluster runs Elasticsearch 16.", at=day(40))

    run = await _consolidate(memory, bank_id, request_context, _nothing)

    shown = run.llm.prompts[0].shown()
    live = [o for o in shown if o["status"] == "active"]
    assert [o["id"] for o in live] == [head]
    assert [o["id"] for o in shown if o["status"] != "active"] == [chain[-1]], "its predecessor, once"


@pytest.mark.asyncio
async def test_a_refuted_claim_with_no_successor_is_shown_so_a_restating_fact_stays_history(memory, request_context):
    """A claim refuted with no correcting memory (an operator refute, a plain retire) has no
    successor to be shown next to; it is shown anyway, so a later fact restating it is attached to
    it as history instead of becoming a current claim of its own."""
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The checkout service caches sessions in Redis.", at=day(10))
    redis = await _observation(memory, bank_id, "Checkout keeps sessions in Redis.", [old], at=day(10))
    await _retire(memory, bank_id, redis, successor=None, status=REFUTED)
    restated = await _unit(memory, bank_id, "Checkout caches its sessions in Redis.", at=day(30))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        shown = prompt.obs("Redis")
        assert (shown["id"], shown["status"], shown["retirable"]) == (redis, "refuted", False)
        assert "successor" not in shown
        return {
            "updates": [
                {
                    "observation_id": shown["id"],
                    "text": shown["text"],
                    "source_fact_ids": [prompt.fact("caches its sessions")],
                    "relation": "historical",
                }
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)

    assert await v.status(restated) == "refuted"
    assert restated in await v.sources(redis)
    assert set(await v.observations()) == {"Checkout keeps sessions in Redis."}, "no new current claim"
    assert {redis, restated}.isdisjoint(
        await _recall_ids(memory, bank_id, request_context, "Where does checkout keep its sessions?")
    )


@pytest.mark.asyncio
async def test_the_relink_of_a_rebuilt_successor_takes_no_claim_lock_outside_the_batch_s_one_pass(
    memory, request_context, monkeypatch
):
    """The retired claim a create re-links is locked in the batch's single sorted pass; the re-link
    after the creates locks nothing but the new observation's own claim row."""
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The queue is RabbitMQ.", at=day(10))
    retired = await _observation(memory, bank_id, "The queue is RabbitMQ.", [old], at=day(10))
    evidence = await _unit(memory, bank_id, "The queue moved to Kafka.", at=day(30))
    store = get_memories()
    async with _conn(memory) as conn:
        async with conn.transaction():
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=[retired, evidence], skip_locked=False)
            assert await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=retired,
                to_status=SUPERSEDED,
                retired_via="supersedes",
                superseded_by_id=None,
                valid_to=day(30),
                reason="its successor was rebuilt",
                confidence=None,
            )
            await store.add_claim_evidence(conn, bank_id=bank_id, memory_id=retired, evidence_ids=[evidence])
    from hindsight_api.engine.memories.pg import adaptive_claims

    passes: list[set[str]] = []
    real = adaptive_claims.lock_claims

    async def spy(conn, **kwargs):
        passes.append({str(i) for i in kwargs["memory_ids"]})
        return await real(conn, **kwargs)

    # The SQL writer every claim lock goes through, the store's own method and its internal callers alike.
    monkeypatch.setattr(adaptive_claims, "lock_claims", spy)

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {"creates": [{"text": "The queue is Kafka.", "source_fact_ids": [prompt.fact("Kafka")]}]}

    await _consolidate(memory, bank_id, request_context, respond)

    rebuilt = (await v.observations())["The queue is Kafka."]
    assert str((await v.claim(retired)).superseded_by_id) == rebuilt
    holding = [i for i, ids in enumerate(passes) if retired in ids]
    assert holding == [0], f"the re-linked claim is locked in the first pass only: {passes}"
    assert all(ids <= {rebuilt} for ids in passes[1:]), passes


@pytest.mark.asyncio
async def test_triage_never_settles_a_fact_whose_candidates_include_a_retired_observation(memory, request_context):
    bank_id = await _bank(memory, request_context)
    old = await _old_fact(memory, bank_id, "The wiki is Confluence.", at=day(10))
    retired = await _observation(memory, bank_id, "The wiki is Confluence.", [old], at=day(10))
    successor = await _observation(memory, bank_id, "The wiki is Notion.", [old], at=day(20))
    await _retire(memory, bank_id, retired, successor=successor)
    await _unit(memory, bank_id, "The wiki is Confluence, said an old doc.", at=day(5))
    seen: list[list[str]] = []

    async def fake_triage(memories, candidates, model, **kwargs):
        seen.append([str(m["id"]) for m in memories])
        return TriageResult()

    with (
        patch.object(consolidator, "triage_model", lambda config: object()),
        patch.object(consolidator, "triage_facts", fake_triage),
    ):
        run = await _consolidate(memory, bank_id, request_context, _nothing)
    assert seen == [] and len(run.llm.prompts) == 1  # straight to the lifecycle prompt


@pytest.mark.asyncio
async def test_dedup_never_folds_into_a_retired_twin_and_a_labelled_create_skips_it(
    memory, request_context, monkeypatch
):
    monkeypatch.setattr(_get_raw_config(), "consolidation_dedup_threshold", 0.5)
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The deploy tool is Spinnaker.", at=day(10))
    retired_twin = await _observation(memory, bank_id, "The deploy tool is Spinnaker.", [old], at=day(10))
    successor = await _observation(memory, bank_id, "The deploy tool is Argo CD.", [old], at=day(20))
    await _retire(memory, bank_id, retired_twin, successor=successor)
    await _unit(memory, bank_id, "Deploys use Spinnaker pipelines again for the legacy app.", at=day(40))
    probes: list[dict[str, Any]] = []
    real = consolidator._dedup_adjudicate

    async def spy(*args, **kwargs):
        probes.append(kwargs)
        return await real(*args, **kwargs)

    def respond(prompt: _Prompt) -> dict[str, Any]:
        fact = prompt.fact("legacy app")
        return {
            "creates": [
                {"text": "The legacy app deploys with Spinnaker.", "source_fact_ids": [fact]},
                {
                    "text": "The deploy tool is Spinnaker for the legacy app.",
                    "source_fact_ids": [fact],
                    "supersedes": [prompt.obs("Argo")["id"]],
                },
            ]
        }

    with patch.object(consolidator, "_dedup_adjudicate", spy):
        await _consolidate(memory, bank_id, request_context, respond)
    assert len(probes) == 1 and probes[0]["exclude_non_active"] is True  # only the unlabelled create
    observations = await v.observations()
    assert "The legacy app deploys with Spinnaker." in observations
    assert "The deploy tool is Spinnaker for the legacy app." in observations  # a new row, never folded
    assert await v.text(retired_twin) == "The deploy tool is Spinnaker."


@pytest.mark.asyncio
async def test_a_fold_into_a_twin_retired_since_the_probe_lands_the_create_as_a_new_row(
    memory, request_context, monkeypatch
):
    monkeypatch.setattr(_get_raw_config(), "consolidation_dedup_threshold", 0.5)
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The pager tool is OpsGenie.", at=day(10))
    twin = await _observation(memory, bank_id, "The pager tool is OpsGenie.", [old], at=day(10))
    await _unit(memory, bank_id, "On-call paging runs on OpsGenie.", at=day(40))

    async def stale_verdict(*args, **kwargs):
        # The probe saw a live twin; by the time the batch writes, it is retired.
        await _retire(memory, bank_id, twin, successor=None)
        return consolidator._DedupOutcome(
            best_id=twin, merged_text="merged", should_merge=True, best_text="The pager tool is OpsGenie."
        )

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {"creates": [{"text": "Paging runs on OpsGenie.", "source_fact_ids": [prompt.fact("OpsGenie")]}]}

    with patch.object(consolidator, "_dedup_adjudicate", stale_verdict):
        await _consolidate(memory, bank_id, request_context, respond)
    assert "Paging runs on OpsGenie." in await v.observations()
    assert await v.text(twin) == "The pager tool is OpsGenie."


@pytest.mark.asyncio
async def test_a_scope_at_its_cap_still_supersedes(memory, request_context):
    bank_id = await _bank(memory, request_context, max_observations_per_scope=1)
    v = _View(memory, bank_id, request_context)
    old = await _old_fact(memory, bank_id, "The queue is RabbitMQ.", at=day(10), tags=["svc:orders"])
    target = await _observation(memory, bank_id, "The queue is RabbitMQ.", [old], at=day(10), tags=["svc:orders"])
    await _unit(memory, bank_id, "The queue moved to Kafka.", at=day(40), tags=["svc:orders"])

    def respond(prompt: _Prompt) -> dict[str, Any]:
        assert "OBSERVATION LIMIT REACHED" in prompt.user
        return {
            "updates": [
                {
                    "observation_id": prompt.obs("RabbitMQ")["id"],
                    "text": "The queue is Kafka.",
                    "source_fact_ids": [prompt.fact("Kafka")],
                    "relation": "supersedes",
                }
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)
    assert await v.status(target) == "superseded"
    assert "The queue is Kafka." in await v.observations()
    async with _conn(memory) as conn:
        assert (
            await get_memories().count_active_observations(
                conn, bank_id=bank_id, tags=["svc:orders"], tags_match="all_strict"
            )
            == 1
        )


# =========================================================================== pg0: other modes


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "observe", "rank"])
async def test_off_observe_and_rank_send_the_base_prompt_and_schema(memory, request_context, mode):
    bank_id = await _bank(memory, request_context, mode=mode)
    old = await _old_fact(memory, bank_id, "The CDN is Fastly.", at=day(10))
    await _observation(memory, bank_id, "The CDN is Fastly.", [old], at=day(10))
    fact = await _unit(memory, bank_id, "The CDN moved to Cloudflare.", at=day(40))

    run = await _consolidate(memory, bank_id, request_context, _nothing)
    (prompt,) = run.llm.prompts
    assert prompt.system == GOLDEN["prompts"]["default"]
    assert (
        prompt.response_format.model_json_schema()
        == _build_response_model(
            None, fact_ids=[fact], supports_max_items=get_config().llm_supports_max_items
        ).model_json_schema()
    )
    assert all(
        set(o) <= {"id", "text", "proof_count", "occurred_start", "occurred_end", "mentioned_at", "source_memories"}
        for o in prompt.shown()
    )
    assert not any(key.startswith("lifecycle_") for key in run.result)


@pytest.mark.asyncio
async def test_observe_mode_records_what_a_plain_delete_would_have_retired(memory, request_context):
    bank_id = await _bank(memory, request_context, mode="observe")
    v = _View(memory, bank_id, request_context)
    stale = await _old_fact(memory, bank_id, "The VPN is OpenVPN.", at=day(10))
    dated = await _old_fact(memory, bank_id, "The VPN was installed.", at=day(10), occurred_start=day(9))
    target = await _observation(memory, bank_id, "The VPN is OpenVPN.", [stale, dated], at=day(10))
    fact = await _unit(memory, bank_id, "The VPN is now WireGuard.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {"deletes": [{"observation_id": prompt.obs("OpenVPN")["id"]}]}

    run = await _consolidate(memory, bank_id, request_context, respond)
    assert await v.unit(target) is None  # observe changes nothing: the delete happened
    shadow = (await v.lifecycle(stale)).shadow
    assert shadow is not None and (shadow.would_status, shadow.evidence_ids) == ("superseded", [fact])
    assert shadow.deciding_snapshot["observation_id"] == target
    assert (await v.lifecycle(dated)).shadow is None  # the dated event would stay current
    assert await v.claim(stale) is None  # no claim was written
    assert run.result["lifecycle_shadow_retirements"] == 1


def test_scope_keys_are_a_static_comma_list_with_the_documented_default(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.delenv("HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS", raising=False)
    assert HindsightConfig.from_env().adaptive_memory_scope_keys == ["environment", "service", "module"]
    monkeypatch.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS", " tenant , region,,")
    config = HindsightConfig.from_env()
    assert config.adaptive_memory_scope_keys == ["tenant", "region"]
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": "lifecycle"}, config, store_supports=True)
    assert settings.scope_keys == ("tenant", "region")


@pytest.mark.asyncio
async def test_one_facet_named_alone_refines_its_parent_which_stays_true(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The mobile app ships weekly.", at=day(10))
    parent = await _observation(memory, bank_id, "The mobile app ships weekly.", [s1], at=day(10))
    await _unit(memory, bank_id, "The mobile app ships weekly, on Thursdays.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {
            "creates": [
                {
                    "text": "The mobile app's weekly release is on Thursdays.",
                    "source_fact_ids": [prompt.fact("Thursdays")],
                    "split_from": prompt.obs("ships weekly")["id"],
                }
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)
    child = (await v.observations())["The mobile app's weekly release is on Thursdays."]
    assert await v.status(parent) == "active"
    assert ("refines", child, parent) in await v.relations()


@pytest.mark.asyncio
async def test_an_update_folded_into_its_twin_is_retired_as_merged_not_deleted(memory, request_context, monkeypatch):
    monkeypatch.setattr(_get_raw_config(), "consolidation_dedup_threshold", 0.5)
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "Priya owns the billing roadmap.", at=day(10))
    s2 = await _old_fact(memory, bank_id, "Priya is in charge of billing planning.", at=day(11))
    twin = await _observation(memory, bank_id, "Priya owns the billing roadmap.", [s1], at=day(10))
    rewritten = await _observation(memory, bank_id, "Priya is in charge of billing planning.", [s2], at=day(11))
    await _unit(memory, bank_id, "Priya confirmed she owns billing planning.", at=day(40))

    async def merge_verdict(
        pool, memory_engine, bank_id_, config, llm_config, anchor_text, emb, tags, exclude_id, **kw
    ):
        assert kw == {"exclude_non_active": True}
        if exclude_id != rewritten:
            return consolidator._DedupOutcome(best_id=None, merged_text="", should_merge=False)
        return consolidator._DedupOutcome(
            best_id=twin,
            merged_text="Priya owns the billing roadmap.",
            should_merge=True,
            best_text="Priya owns the billing roadmap.",
        )

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {
            "updates": [
                {
                    "observation_id": prompt.obs("in charge")["id"],
                    "text": "Priya owns billing planning.",
                    "source_fact_ids": [prompt.fact("confirmed")],
                }
            ]
        }

    with patch.object(consolidator, "_dedup_adjudicate", merge_verdict):
        run = await _consolidate(memory, bank_id, request_context, respond)
    assert await v.unit(rewritten) is not None
    claim = await v.claim(rewritten)
    assert (claim.truth_status, claim.retired_via, str(claim.superseded_by_id)) == (
        "superseded",
        "merges",
        twin,
    )
    assert s2 in await v.sources(twin)
    assert ("merges", twin, rewritten) in await v.relations()
    assert run.result["lifecycle_merge"] == 1


@pytest.mark.asyncio
async def test_a_merge_survivor_keeps_the_larger_stability_of_the_pair(memory, request_context):
    bank_id = await _bank(memory, request_context)
    v = _View(memory, bank_id, request_context)
    s1 = await _old_fact(memory, bank_id, "The store opens at 9.", at=day(10))
    s2 = await _old_fact(memory, bank_id, "Opening time is 9 am.", at=day(11))
    survivor = await _observation(memory, bank_id, "The store opens at 9.", [s1], at=day(10))
    duplicate = await _observation(memory, bank_id, "Opening time is 9 am.", [s2], at=day(11))
    async with _conn(memory) as conn:
        await conn.execute(
            "INSERT INTO memory_retention_state (memory_id, bank_id, stability_days, last_reinforced_at, created_at, "
            "updated_at) VALUES ($1, $2, 900, $3, $3, $3)",
            uuid.UUID(duplicate),
            bank_id,
            day(11),
        )
    await _unit(memory, bank_id, "Someone said the store opens at 9.", at=day(40))

    def respond(prompt: _Prompt) -> dict[str, Any]:
        return {
            "deletes": [
                {"observation_id": prompt.obs("Opening time")["id"], "merged_into": prompt.obs("opens at 9")["id"]}
            ]
        }

    await _consolidate(memory, bank_id, request_context, respond)
    assert (await v.lifecycle(survivor)).retention.stability_days >= 900


def test_collapsing_duplicate_updates_keeps_their_lifecycle_labels():
    (merged,) = consolidator._dedupe_updates(
        [
            _update("o1", "Runs Java 25.", ["f1"], relation="supersedes", retires_source_ids=["s1"]),
            _UpdateAction(observation_id="o1", text="Runs Java 25 on ARM.", source_fact_ids=["f2"]),
        ],
        batch_label="test",
    )
    assert isinstance(merged, _LifecycleUpdateAction)
    assert (merged.text, merged.source_fact_ids, merged.relation, merged.retires_source_ids) == (
        "Runs Java 25 on ARM.",
        ["f1", "f2"],
        "supersedes",
        ["s1"],
    )


def test_miscopied_label_ids_are_repaired_against_the_pool_and_its_source_ids():
    pool_ids = {"d0a3effa-1795-4c48-9d2e-0123456789ab"}
    sources = {"aaaaaaaa-1111-4c48-9d2e-0123456789ab"}
    create = _create(
        "x",
        ["f1"],
        supersedes=["d0a3effa-1795-4ef0-0000-000000000000"],
        keeps_source_ids=["aaaaaaaa-1111-4fff-0000-000000000000"],
    )
    update = _update("o1", "y", ["f1"], relation="refutes", retires_source_ids=["aaaaaaaa-1111-4abc-0000-000000000000"])
    consolidator._resolve_miscopied_ids(_BatchLLMResult(creates=[create], updates=[update]), pool_ids, sources)
    assert create.supersedes == list(pool_ids) and create.keeps_source_ids == list(sources)
    assert update.retires_source_ids == list(sources)
