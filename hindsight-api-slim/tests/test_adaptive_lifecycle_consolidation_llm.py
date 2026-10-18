"""The lifecycle consolidation prompt gets a real model to label state changes (adaptive memory).

Whether a model labels a state change (``relation: "supersedes"`` on its UPDATE, or a CREATE naming
the old observation in ``supersedes``) instead of rewriting the observation in place, and whether a
MISSION that asks for in-place updates still gets labelled ones, is prompt-following that MockLLM
cannot simulate — so these run the real batch call and judge the result. The deterministic halves
(the prompt variant's text, the schema, how labels become claim transitions) are unit-tested in
test_adaptive_lifecycle_consolidation.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.adaptive.models import TruthStatus
from hindsight_api.engine.consolidation.consolidator import _BatchLLMResult, _consolidate_batch_with_llm
from hindsight_api.engine.consolidation.lifecycle import PoolEntry
from hindsight_api.engine.response_models import MemoryFact
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

#: The coding-agents integration's mission wording: on a superseded fact, UPDATE in place.
_IN_PLACE_MISSION = (
    "Track the conventions, tools and decisions of this codebase. When a new fact supersedes what an "
    "observation says, UPDATE that observation in place to the new state."
)


def _summary(result: _BatchLLMResult) -> str:
    lines = [
        f"- CREATE: {c.text} | supersedes={getattr(c, 'supersedes', [])} refutes={getattr(c, 'refutes', [])}"
        for c in result.creates
    ]
    lines += [
        f"- UPDATE {u.observation_id}: {u.text} | relation={getattr(u, 'relation', None)}" for u in result.updates
    ]
    lines += [f"- DELETE {d.observation_id} merged_into={d.merged_into}" for d in result.deletes]
    return "\n".join(lines) or "(no actions)"


def _pool_entry(observation: MemoryFact, status: TruthStatus = TruthStatus.ACTIVE, successor: str | None = None):
    return PoolEntry(
        id=str(observation.id),
        text=observation.text,
        status=status,
        successor=successor,
        tags=frozenset(),
        source_ids=tuple(observation.source_fact_ids or []),
        mentioned_at=datetime.fromisoformat(observation.mentioned_at) if observation.mentioned_at else None,
        valid_from=None,
        claim_fingerprint=None,
        retirable=status in (TruthStatus.ACTIVE, TruthStatus.UNCERTAIN),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mission", [None, _IN_PLACE_MISSION], ids=["default_mission", "in_place_mission"])
async def test_a_real_model_labels_a_state_change_instead_of_rewriting_it(mission):
    then = datetime.now(timezone.utc) - timedelta(days=60)
    now = datetime.now(timezone.utc)
    observation_id, source_id = str(uuid.uuid4()), str(uuid.uuid4())
    observation = MemoryFact(
        id=observation_id,
        text="The payments service runs on Java 21.",
        fact_type="observation",
        mentioned_at=then.isoformat(),
        source_fact_ids=[source_id],
    )
    source = MemoryFact(
        id=source_id, text="The payments service runs on Java 21.", fact_type="world", mentioned_at=then.isoformat()
    )
    memories = [
        {
            "id": uuid.uuid4(),
            "text": "The payments service was upgraded from Java 21 to Java 25 last week.",
            "tags": [],
            "occurred_start": None,
            "occurred_end": None,
            "mentioned_at": now,
        }
    ]
    config = _get_raw_config()
    if mission is not None:
        config = type(config)(
            **{**{f: getattr(config, f) for f in config.__dataclass_fields__}, "observations_mission": mission}
        )

    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=[observation],
        union_source_facts={source_id: source},
        config=config,
        lifecycle_pool={observation_id: _pool_entry(observation)},
    )

    assert result.failed is False, "the lifecycle consolidation batch call failed outright"
    await assert_meets_criteria(
        response=_summary(result),
        criteria=(
            "The Java upgrade is recorded as a labelled state change: either an UPDATE of observation "
            f"{observation_id} with relation 'supersedes' (or 'refutes') whose text states the service now "
            f"runs Java 25, or a CREATE stating Java 25 that names {observation_id} in its supersedes list. "
            "It is NOT an unlabelled update (relation None) that rewrites the observation to the new version, "
            "and nothing is deleted."
        ),
        context=(
            "A lifecycle-mode consolidation model saw one active, retirable observation 'The payments service "
            f"runs on Java 21.' (id {observation_id}) and one new fact: the service was upgraded to Java 25. "
            + (f"The bank's mission said: {mission!r}." if mission else "The bank used the default mission.")
        ),
    )


@pytest.mark.asyncio
async def test_a_real_model_attaches_an_old_restatement_of_a_retired_observation_as_history():
    long_ago = datetime.now(timezone.utc) - timedelta(days=200)
    later = datetime.now(timezone.utc) - timedelta(days=60)
    retired_id, current_id = str(uuid.uuid4()), str(uuid.uuid4())
    retired = MemoryFact(
        id=retired_id, text="The team uses Jenkins for CI.", fact_type="observation", mentioned_at=long_ago.isoformat()
    )
    current = MemoryFact(
        id=current_id,
        text="The team uses GitHub Actions for CI.",
        fact_type="observation",
        mentioned_at=later.isoformat(),
    )
    memories = [
        {
            "id": uuid.uuid4(),
            "text": "The team said its CI runs on Jenkins.",
            "tags": [],
            "occurred_start": None,
            "occurred_end": None,
            "mentioned_at": long_ago - timedelta(days=10),
        }
    ]

    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=[current, retired],
        union_source_facts={},
        config=_get_raw_config(),
        lifecycle_pool={
            current_id: _pool_entry(current),
            retired_id: _pool_entry(retired, TruthStatus.SUPERSEDED, successor=current_id),
        },
    )

    assert result.failed is False, "the lifecycle consolidation batch call failed outright"
    await assert_meets_criteria(
        response=_summary(result),
        criteria=(
            f"The old Jenkins fact is treated as history: either an UPDATE of the superseded observation {retired_id} "
            "with relation 'historical' that keeps its text, or no action at all. The current observation "
            f"{current_id} (GitHub Actions) is not superseded, refuted or rewritten back to Jenkins, and no new "
            "observation claims the team currently uses Jenkins."
        ),
        context=(
            "A lifecycle-mode consolidation model saw an active observation 'The team uses GitHub Actions for CI.' "
            f"(id {current_id}) and a superseded one 'The team uses Jenkins for CI.' (id {retired_id}, its successor is "
            f"{current_id}), plus one new fact stated before both: the team said its CI runs on Jenkins."
        ),
    )
