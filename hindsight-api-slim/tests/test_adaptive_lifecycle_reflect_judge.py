"""Reflect in a lifecycle-mode bank does not settle an UNCERTAIN claim by recency (real LLM + judge).

The deterministic half — that the clause and the ``include_history`` flag appear only in lifecycle
mode, and that the tool passes the flag through — is pinned without a model in
test_adaptive_lifecycle_recall.py. What only a real model can show is that it follows the clause:
the reflect prompt otherwise tells it the latest ``mentioned_at`` wins a conflict, so two
contested claims, the later one newer, are exactly the case where the base rule picks a side and
the lifecycle rule must not.

The recall tool is stubbed so the evidence is controlled; the agent loop and the synthesis are the
real ones. Needs LLM credentials (``hs_llm_core``).
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_UNCERTAIN = {"truth_status": "uncertain"}

CONTESTED = [
    {
        "id": "mem-redis",
        "text": "The checkout service stores sessions in Redis (platform runbook).",
        "fact_type": "world",
        "mentioned_at": "2026-02-01T09:00:00+00:00",
        "lifecycle": _UNCERTAIN,
    },
    {
        "id": "mem-postgres",
        "text": "The checkout service stores sessions in Postgres (team wiki).",
        "fact_type": "world",
        "mentioned_at": "2026-03-01T09:00:00+00:00",
        "lifecycle": _UNCERTAIN,
    },
]


async def test_uncertain_claims_are_reported_as_a_conflict_not_resolved_by_recency(llm_config):
    async def recall_fn(query, max_tokens, max_chunk_tokens, **kwargs):
        return {"query": query, "memories": CONTESTED, "chunks": {}}

    result = await run_reflect_agent(
        llm_config=llm_config,
        bank_id="test-bank",
        query="Where does the checkout service store its sessions?",
        bank_profile={"name": "Platform", "mission": "Answer questions about the platform from memory."},
        search_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        read_mental_models_fn=AsyncMock(return_value={"mental_models": []}),
        search_observations_fn=AsyncMock(return_value={"observations": []}),
        recall_fn=recall_fn,
        expand_fn=AsyncMock(return_value={"results": []}),
        has_mental_models=False,
        include_observations=False,
        include_recall=True,
        budget="low",
        max_iterations=4,
        lifecycle=True,
    )

    assert result.text
    await assert_meets_criteria(
        response=result.text,
        criteria=(
            "The answer says the evidence conflicts about where sessions are stored, names BOTH Redis and "
            "Postgres as what the sources claim, and does NOT present Postgres (the more recent statement) "
            "as the settled answer."
        ),
        context=(
            "Two memories, both marked uncertain (unresolved conflicting evidence): the platform runbook says "
            "Redis (mentioned 2026-02-01), the team wiki says Postgres (mentioned 2026-03-01)."
        ),
    )
