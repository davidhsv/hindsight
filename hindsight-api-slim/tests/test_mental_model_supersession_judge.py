"""A page grounded on a superseded claim is regenerated from the current one (real LLM + judge).

The deterministic half — that a retired citation escalates the refresh to full, records the cause
and the successor, and that the regeneration reads current recall — is pinned without a model in
test_mental_model_retractions.py and the system story test_95_mental_model_supersession.py. What
only a real model can show is the wording: given current recall (Java 25 only), the regenerated
page states Java 25 as the runtime and does not keep Java 21 as the current answer.

The bank is seeded directly (two observations; the old one superseded through the store's claim
writers), so the only LLM work is the refresh itself. Needs LLM credentials (``hs_llm_core``) and
an explicit opt-in, ``HINDSIGHT_TEST_ALLOW_REAL_LLM=1``.
"""

import json
import os
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.adaptive.models import TruthStatus
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.engine.schema import fq_store_table
from tests.llm_judge import assert_meets_criteria

pytestmark = [
    pytest.mark.hs_llm_core,
    pytest.mark.memory_backend_incompatible,
    pytest.mark.skipif(
        os.getenv("HINDSIGHT_TEST_ALLOW_REAL_LLM") != "1",
        reason="real-LLM judge test: set HINDSIGHT_TEST_ALLOW_REAL_LLM=1",
    ),
]

OLD = "The billing service runs Java 21."
NEW = "The billing service was upgraded and now runs Java 25."
QUERY = "Which Java version does the billing service run?"


async def _seed_observations(memory, bank_id: str) -> str:
    """Seed OLD superseded by NEW; the superseded observation's id."""
    texts = [OLD, NEW]
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, texts)
    ids = [str(uuid.uuid4()) for _ in texts]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for unit_id, text, embedding, when in zip(
            ids, texts, embeddings, (datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 2, 4, tzinfo=UTC))
        ):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, mentioned_at, proof_count) "
                "VALUES ($1, $2, $3, 'observation', $4::vector, $5, 1)",
                unit_id,
                bank_id,
                text,
                "[" + ",".join(str(v) for v in embedding) + "]",
                when,
            )
        store = get_memories()
        async with conn.transaction():
            await store.lock_claims(conn, bank_id=bank_id, memory_ids=ids, skip_locked=False)
            assert await store.cas_retire(
                conn,
                bank_id=bank_id,
                memory_id=ids[0],
                to_status=TruthStatus.SUPERSEDED,
                retired_via="explicit",
                superseded_by_id=ids[1],
                valid_to=datetime(2026, 2, 4, tzinfo=UTC),
                reason=None,
                confidence=None,
            )
    return ids[0]


async def test_a_page_grounded_on_a_superseded_claim_is_regenerated_from_the_successor(memory_real_llm):
    memory = memory_real_llm
    rc = RequestContext()
    bank_id = f"test-mm-supersession-judge-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=rc)
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "lifecycle"})
    old = await _seed_observations(memory, bank_id)

    content = "## Runtime\n\nThe billing service runs Java 21.\n"
    model = await memory.create_mental_model(
        bank_id=bank_id,
        name="Billing runtime",
        source_query=QUERY,
        content=content,
        request_context=rc,
        trigger={"mode": "delta", "fact_types": ["observation"], "exclude_mental_models": True, "keep_trace": True},
    )
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {fq_store_table('mental_models')} SET content = $3, reflect_response = $4::jsonb, "
            f"last_memory_seen_at = now() WHERE bank_id = $1 AND id = $2",
            bank_id,
            model["id"],
            content,
            json.dumps(
                {
                    "text": content,
                    "based_on": {"observation": [{"id": old, "text": OLD, "type": "observation", "context": None}]},
                    "mental_models": [],
                }
            ),
        )

    try:
        refreshed = await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=model["id"], request_context=rc)
        assert refreshed is not None and refreshed["content"].strip()
        await assert_meets_criteria(
            response=refreshed["content"],
            criteria=(
                "The document says the billing service runs Java 25 now. It does NOT state Java 21 as the "
                "current runtime (mentioning Java 21 only as the earlier version is acceptable)."
            ),
            context=(
                "The page was written when the billing service ran Java 21; that claim was then superseded by "
                "an upgrade to Java 25, and the refresh read only the current memories."
            ),
        )
    finally:
        await memory.delete_bank(bank_id, request_context=rc)
