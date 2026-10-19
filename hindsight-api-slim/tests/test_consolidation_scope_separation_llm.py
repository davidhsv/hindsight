"""A real model keeps one setting's values in two services as two observations.

The service is named either as ``service`` metadata or only as the ``repository`` the coding-agents
integration stamps; both reach the prompt as the fact's ``scope``.

The deterministic half (the scope reaching the prompt) is ``test_consolidation_scope_separation.py``.
Whether the model then treats a differing ``scope`` as a separate observation rather than a state
change is prompt-following behaviour MockLLM cannot simulate, so this runs the real batch call
and judges the actions it returns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.response_models import MemoryFact
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@dataclass(frozen=True)
class _Scope:
    """The metadata key that names where each cap holds, and its two values."""

    key: str
    first: str
    second: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [_Scope("service", "media", "archive"), _Scope("repository", "media-uploads", "archive-uploads")],
    ids=lambda scope: scope.key,
)
async def test_real_model_keeps_another_services_value_apart_from_the_observation(scope: _Scope):
    media_obs_id, media_fact_id, archive_fact_id = str(uuid.uuid4()), str(uuid.uuid4()), uuid.uuid4()
    stated = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
    later = stated + timedelta(days=14)

    observation = MemoryFact(
        id=media_obs_id,
        text="Uploads are capped at 25 MB.",
        fact_type="observation",
        mentioned_at=stated.isoformat(),
        source_fact_ids=[media_fact_id],
    )
    media_fact = MemoryFact(
        id=media_fact_id,
        text="Uploads are capped at 25 MB.",
        fact_type="world",
        mentioned_at=stated.isoformat(),
        metadata={scope.key: scope.first},
    )
    memories = [
        {
            "id": archive_fact_id,
            "text": "Uploads are capped at 100 MB.",
            "tags": [],
            "occurred_start": None,
            "occurred_end": None,
            "mentioned_at": later,
            "metadata": {scope.key: scope.second},
        }
    ]

    result = await _consolidate_batch_with_llm(
        llm_config=LLMConfig.from_env(),
        memories=memories,
        union_observations=[observation],
        union_source_facts={media_fact_id: media_fact},
        config=_get_raw_config(),
    )

    assert result.failed is False, "the consolidation batch call failed outright"
    actions_summary = "\n".join(
        [f"- CREATE: {c.text}" for c in result.creates]
        + [f"- UPDATE {u.observation_id}: {u.text}" for u in result.updates]
        + [f"- DELETE {d.observation_id}: {d.reason}" for d in result.deletes]
    )
    await assert_meets_criteria(
        response=actions_summary or "(no actions)",
        criteria=(
            f"The {scope.first} observation {media_obs_id} still says uploads are capped at 25 MB: no DELETE "
            "removes it and no UPDATE of it says the cap is 100 MB, was raised, or changed. The 100 MB cap is "
            f"recorded for {scope.second} by a CREATE (an UPDATE of the {scope.first} observation that keeps "
            f"25 MB for {scope.first} and adds 100 MB for {scope.second} also meets this)."
        ),
        context=(
            f"A consolidation model was given one existing observation, 'Uploads are capped at 25 MB.' "
            f"(id {media_obs_id}), whose source fact holds in the {scope.key} {scope.first}, and one new fact, "
            f"'Uploads are capped at 100 MB.', which holds in the {scope.key} {scope.second}. Both caps are true "
            f"at once, each in its own {scope.key}."
        ),
    )
