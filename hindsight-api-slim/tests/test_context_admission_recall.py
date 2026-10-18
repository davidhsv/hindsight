"""Context admission through a real recall (pg0, the ``memory`` fixture, local embeddings and
cross-encoder, a pinned question_date).

What the pure tests cannot show: that a bank with the mode off answers exactly as before (request
fields included), that observe changes nothing a caller sees (observations, the prefer-observations
dedup and adaptive scoring included), that filter can answer with no memories and says so, that
strict enforces the hard scope (on observations too, through their source facts) and suppresses a
stored near-duplicate through the store's embeddings, that a gated recall is never offered to a
store's whole-recall path, that a passthrough reranker is not read as relevance, that recall
reinforcement is forced off while the gate filters, that internal recalls are never gated, and the
trace's shape.

Memories are inserted with raw SQL (exact texts, metadata and embeddings), so the module is
``memory_backend_incompatible``. Assertions go through ``recall_async``.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from hindsight_api.config import _get_raw_config, clear_config_cache
from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.extensions.operation_validator import OperationValidationError

pytestmark = pytest.mark.memory_backend_incompatible

QDATE = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
REPO_A = "github.com/acme/payments"
REPO_B = "github.com/acme/storefront"

QUERY = "How is the payments API deployed to production?"
ROWS = [
    ("The payments API is deployed to production with a blue/green rollout on Kubernetes.", REPO_A, "deploy"),
    ("The payments API is deployed to production with a blue/green rollout on Kubernetes.", REPO_A, "deploy"),
    ("Payments API production deploys run from the release pipeline after the canary passes.", REPO_A, "deploy"),
    ("The storefront API is deployed to production with a blue/green rollout on Kubernetes.", REPO_B, "deploy"),
    ("The office coffee machine is descaled every Friday.", None, None),
]


def _bank() -> str:
    return f"test-ctxadm-{uuid.uuid4().hex[:8]}"


async def _seed(memory, rc, bank_id: str) -> list[str]:
    await memory.ensure_bank_profile(bank_id, request_context=rc)
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [r[0] for r in ROWS])
    ids = [str(uuid.uuid4()) for _ in ROWS]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for index, (unit_id, (text, repo, module), embedding) in enumerate(zip(ids, ROWS, embeddings)):
            metadata = {k: v for k, v in (("repository", repo), ("module", module)) if v}
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, mentioned_at, metadata)
                VALUES ($1, $2, $3, 'world', $4::vector, $5, $6::jsonb)
                """,
                unit_id,
                bank_id,
                text,
                "[" + ",".join(str(v) for v in embedding) + "]",
                QDATE - timedelta(days=10 + index),
                json.dumps(metadata),
            )
    return ids


OBSERVATIONS = [
    # (text, indexes of the ROWS it was consolidated from)
    ("Payments API production deploys use a blue/green rollout after the canary passes.", (0, 2)),
    ("Storefront API production deploys use a blue/green rollout on Kubernetes.", (3,)),
]


async def _seed_observations(memory, bank_id: str, fact_ids: list[str]) -> list[str]:
    """Observations as consolidation writes them: no metadata, only their source ids."""
    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [o[0] for o in OBSERVATIONS])
    ids = [str(uuid.uuid4()) for _ in OBSERVATIONS]
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for unit_id, (text, sources), embedding in zip(ids, OBSERVATIONS, embeddings):
            await conn.execute(
                """
                INSERT INTO memory_units
                    (id, bank_id, text, fact_type, embedding, mentioned_at, proof_count, source_memory_ids)
                VALUES ($1, $2, $3, 'observation', $4::vector, $5, $6, $7::uuid[])
                """,
                unit_id,
                bank_id,
                text,
                "[" + ",".join(str(v) for v in embedding) + "]",
                QDATE - timedelta(days=5),
                len(sources),
                [uuid.UUID(fact_ids[i]) for i in sources],
            )
    return ids


async def _configure(memory, bank_id: str, **updates) -> None:
    await memory._config_resolver.update_bank_config(bank_id, updates)


async def _recall(memory, rc, bank_id: str, **kwargs):
    kwargs.setdefault("max_tokens", 4096)
    kwargs.setdefault("fact_type", ["world"])
    kwargs.setdefault("budget", Budget.MID)
    kwargs.setdefault("_external", True)
    return await memory.recall_async(bank_id=bank_id, query=QUERY, question_date=QDATE, request_context=rc, **kwargs)


def _ids(result) -> list[str]:
    return [str(r.id) for r in result.results]


def _phases(result) -> set[str]:
    return {p["phase_name"] for p in result.trace["summary"]["phase_metrics"]}


async def test_mode_off_answers_exactly_as_before_with_the_new_request_fields(memory, request_context):
    bank_id = _bank()
    try:
        await _seed(memory, request_context, bank_id)
        plain = await _recall(memory, request_context, bank_id)
        assert plain.context_admission is None and "context_admission" not in plain.model_dump_json()
        with_fields = await _recall(
            memory, request_context, bank_id, purpose="auto_context", task_context={"repository": REPO_A}
        )
        assert with_fields.model_dump_json() == plain.model_dump_json()
        traced = await _recall(memory, request_context, bank_id, enable_trace=True, purpose="auto_context")
        assert "context_admission" not in traced.trace
        assert "context_admission" not in _phases(traced)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_observe_changes_nothing_visible_and_reports_the_shadow(memory, request_context):
    bank_id = _bank()
    try:
        ids = await _seed(memory, request_context, bank_id)
        off = await _recall(
            memory, request_context, bank_id, purpose="auto_context", task_context={"repository": REPO_A}
        )
        await _configure(memory, bank_id, context_admission_mode="observe")
        observed = await _recall(
            memory, request_context, bank_id, purpose="auto_context", task_context={"repository": REPO_A}
        )
        assert observed.model_dump_json() == off.model_dump_json()

        traced = await _recall(
            memory,
            request_context,
            bank_id,
            enable_trace=True,
            purpose="auto_context",
            task_context={"repository": REPO_A},
        )
        assert _ids(traced) == _ids(off)
        trace = traced.trace["context_admission"]
        assert trace["mode"] == "observe" and trace["purpose"] == "auto_context"
        assert trace["candidates"] == len(ids)
        shadow = trace["shadow_final_ids"]
        # The other repository's memory and the duplicate never reach the shadow (observe simulates strict).
        assert ids[3] not in shadow and not {ids[0], ids[1]} <= set(shadow)
        assert ids[4] not in shadow
        assert trace["would_change_top_k"] is True and trace["would_drop"] >= 2
        assert trace["cross_scope_candidates"] == 1
        decision = trace["decisions"][0]
        assert set(decision) >= {
            "memory_id",
            "base_retrieval_score",
            "scope_score",
            "task_relevance",
            "expected_utility",
            "truth_status",
            "admission_score",
            "admitted",
            "reason",
            "features",
            "scope",
        }
        assert "context_admission" in _phases(traced)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.parametrize("scorer", ["jev", "hybrid"])
async def test_observe_never_asks_the_decision_model_even_with_a_model_scorer_configured(
    memory, request_context, context_admission_experimental, monkeypatch, scorer
):
    """Observe is telemetry: it scores deterministically, so no memory text leaves the server and
    no recall waits on a decision model. The same server's strict recall does ask (the spy works)."""
    context_admission_experimental.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_SCORER", scorer)
    clear_config_cache()
    asked: list[str] = []

    def spy_jev_client():
        asked.append("jev_client")
        return None  # not configured: strict then falls back to the deterministic score

    monkeypatch.setattr(memory_engine_module, "jev_client", spy_jev_client)
    bank_id = _bank()
    try:
        await _seed(memory, request_context, bank_id)
        off = await _recall(memory, request_context, bank_id, purpose="auto_context")
        await _configure(memory, bank_id, context_admission_mode="observe")
        observed = await _recall(memory, request_context, bank_id, purpose="auto_context", enable_trace=True)
        assert asked == []
        assert _ids(observed) == _ids(off)
        trace = observed.trace["context_admission"]
        assert (trace["mode"], trace["scorer"]) == ("observe", "deterministic")
        assert "jev_error" not in trace

        await _configure(memory, bank_id, context_admission_mode="strict")
        gated = await _recall(memory, request_context, bank_id, purpose="auto_context", enable_trace=True)
        assert asked == ["jev_client"]
        assert (gated.trace["context_admission"]["scorer"], gated.trace["context_admission"]["jev_error"]) == (
            scorer,
            "not_configured",
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_observe_changes_nothing_visible_with_observations_and_adaptive_scoring(memory, request_context):
    """The shadow's observations ride on the prefer-observations dedup read and adaptive scoring
    runs beside the gate; neither may change a byte of what the caller gets."""
    bank_id = _bank()
    try:
        fact_ids = await _seed(memory, request_context, bank_id)
        await _seed_observations(memory, bank_id, fact_ids)
        await _configure(memory, bank_id, adaptive_memory_mode="rank")
        kwargs = {
            "fact_type": ["world", "observation"],
            "prefer_observations": True,
            "purpose": "auto_context",
            "task_context": {"repository": REPO_A},
        }
        off = await _recall(memory, request_context, bank_id, **kwargs)
        assert any(r.fact_type == "observation" for r in off.results)
        await _configure(memory, bank_id, context_admission_mode="observe")
        observed = await _recall(memory, request_context, bank_id, **kwargs)
        assert observed.model_dump_json() == off.model_dump_json()
        assert observed.context_admission is None
        traced = await _recall(memory, request_context, bank_id, enable_trace=True, **kwargs)
        assert _ids(traced) == _ids(off)
        assert traced.trace["context_admission"]["shadow_final_ids"] is not None
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_strict_rejects_another_repositorys_observation_on_automatic_recall(
    memory, request_context, context_admission_experimental
):
    """The coding-agents default: observations only, purpose auto_context. Observations carry no
    metadata, so their scope comes from the facts they were consolidated from."""
    bank_id = _bank()
    try:
        fact_ids = await _seed(memory, request_context, bank_id)
        own, other = await _seed_observations(memory, bank_id, fact_ids)
        await _configure(memory, bank_id, context_admission_mode="strict", context_admission_threshold_auto=0.0)
        result = await _recall(
            memory,
            request_context,
            bank_id,
            enable_trace=True,
            fact_type=["observation"],
            purpose="auto_context",
            task_context={"repository": REPO_A},
        )
        assert _ids(result) == [own]
        by_id = {d["memory_id"]: d for d in result.trace["context_admission"]["decisions"]}
        assert by_id[other]["reason"] == "wrong_scope"
        assert by_id[other]["scope"]["repository"] == "mismatch"
        assert by_id[own]["scope"]["repository"] == "match"
        assert result.trace["context_admission"]["cross_scope_candidates"] == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_gated_recall_is_never_offered_to_the_store_whole(
    memory, request_context, monkeypatch, context_admission_experimental
):
    """A store answering ``full_recall`` would skip the gate, so the engine does not ask it."""
    from hindsight_api.engine.memories import get_memories

    bank_id = _bank()
    store = get_memories()
    offered: list[object] = []
    original = store.full_recall

    async def spy(request):
        offered.append(request)
        return await original(request)

    monkeypatch.setattr(store, "full_recall", spy)
    try:
        await _seed(memory, request_context, bank_id)
        await _recall(memory, request_context, bank_id)
        assert len(offered) == 1
        for mode in ("observe", "filter", "strict"):
            offered.clear()
            await _configure(memory, bank_id, context_admission_mode=mode)
            await _recall(memory, request_context, bank_id)
            assert offered == [], mode
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_passthrough_reranker_is_scored_on_semantic_similarity(
    memory, request_context, context_admission_experimental
):
    bank_id = _bank()
    try:
        await _seed(memory, request_context, bank_id)
        await _configure(memory, bank_id, context_admission_mode="filter")
        rrf = await _recall(memory, request_context, bank_id, enable_trace=True, reranking="rrf")
        trace = rrf.trace["context_admission"]
        assert trace["relevance_source"] == "semantic"
        for decision in trace["decisions"]:
            assert "rank" not in decision["features"]
        crossed = await _recall(memory, request_context, bank_id, enable_trace=True)
        assert crossed.trace["context_admission"]["relevance_source"] == "reranker"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_filter_can_answer_with_no_memories(memory, request_context, context_admission_experimental):
    bank_id = _bank()
    try:
        await _seed(memory, request_context, bank_id)
        await _configure(memory, bank_id, context_admission_mode="filter", context_admission_threshold_explicit=1.0)
        empty = await _recall(memory, request_context, bank_id)
        assert empty.results == []
        # The summary tells the abstention from an empty bank without a trace.
        assert empty.context_admission is not None and empty.context_admission.abstained is True
        assert empty.context_admission.admitted == 0 and empty.context_admission.candidates > 0
        traced = await _recall(memory, request_context, bank_id, enable_trace=True)
        trace = traced.trace["context_admission"]
        assert trace["abstained"] is True and trace["admitted"] == 0
        assert trace["reasons"]["below_threshold"] == trace["candidates"] > 0
        # Abstention reasons live in the trace only, never in the results payload.
        assert "reasons" not in empty.model_dump_json()

        # An internal recall (reflect tools, consolidation) is never gated.
        internal = await _recall(memory, request_context, bank_id, _external=False)
        assert internal.results
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_strict_enforces_the_hard_scope_and_suppresses_the_stored_duplicate(
    memory, request_context, context_admission_experimental
):
    bank_id = _bank()
    try:
        ids = await _seed(memory, request_context, bank_id)
        await _configure(memory, bank_id, context_admission_mode="strict", context_admission_threshold_explicit=0.0)
        result = await _recall(memory, request_context, bank_id, enable_trace=True, task_context={"repository": REPO_A})
        returned = set(_ids(result))
        assert ids[3] not in returned
        assert len({ids[0], ids[1]} & returned) == 1
        assert ids[2] in returned
        trace = result.trace["context_admission"]
        by_id = {d["memory_id"]: d for d in trace["decisions"]}
        assert by_id[ids[3]]["reason"] == "wrong_scope"
        assert by_id[ids[3]]["scope"]["repository"] == "mismatch"
        twin = ids[1] if ids[0] in returned else ids[0]
        canonical = ids[0] if twin == ids[1] else ids[1]
        assert by_id[twin]["reason"] == "redundant" and by_id[canonical]["duplicates"] == 1
        assert trace["reasons"]["wrong_scope"] == 1 and trace["reasons"]["redundant"] == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_recall_reinforcement_is_forced_off_while_the_gate_filters(
    memory, request_context, context_admission_experimental
):
    bank_id = _bank()
    config = _get_raw_config()
    original = config.adaptive_memory_recall_reinforcement_enabled
    config.adaptive_memory_recall_reinforcement_enabled = True
    submit = MagicMock()
    memory._adaptive_reinforcement.submit = submit
    try:
        await _seed(memory, request_context, bank_id)
        await _configure(memory, bank_id, adaptive_memory_mode="rank", context_admission_mode="observe")
        observed = await _recall(memory, request_context, bank_id)
        assert observed.results and submit.call_count == 1

        submit.reset_mock()
        await _configure(memory, bank_id, context_admission_mode="filter", context_admission_threshold_explicit=0.0)
        filtered = await _recall(memory, request_context, bank_id)
        assert filtered.results
        submit.assert_not_called()
    finally:
        config.adaptive_memory_recall_reinforcement_enabled = original
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_an_unknown_purpose_is_rejected(memory, request_context):
    bank_id = _bank()
    try:
        await _seed(memory, request_context, bank_id)
        with pytest.raises(OperationValidationError, match="Invalid purpose"):
            await _recall(memory, request_context, bank_id, purpose="gossip")
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
