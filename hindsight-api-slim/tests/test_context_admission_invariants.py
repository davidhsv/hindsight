"""Context admission end to end, for the invariants the pure tests show only in parts (pg0, the
``memory`` fixture, local embeddings and cross-encoder).

- Currentness on real claims (adaptive ``lifecycle`` mode, the Java 17 -> 21 supersession laid
  down the way consolidation writes it): a present-state recall rejects the superseded memories a
  historical default view shows the gate, a historical search reads the history view and admits
  them, and a refuted memory is rejected even where the view shows it.
- A joke is admitted for the question that asks for it and the gate leaves it in the bank: routing,
  not deletion. Rejecting it for a technical task is an expected failure of the deterministic
  scorer (it has no casual signal and reads the cross-encoder as relevance), kept as a strict xfail.
- Strength is not relevance, on real cross-encoder scores: a weak memory that answers the question
  exactly is admitted and an important, stable memory that does not is rejected.
- The admission budget caps what a strict recall returns, by count and by tokens.
- Gated recalls write no reinforcement, so admission cannot feed its own retrievability; observe,
  which leaves the results alone, still reinforces what it returns like a recall without the gate.
- No mode of either feature makes a recall call a generative model.

Memories are inserted with raw SQL (exact texts and embeddings), so the module is
``memory_backend_incompatible``. Assertions go through ``recall_async`` and the store's readers.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import Budget, count_tokens
from tests.test_adaptive_lifecycle_recall import QUERY as JAVA_QUERY
from tests.test_adaptive_lifecycle_recall import _seed, java_upgrade  # noqa: F401 — fixture

#: Every invariant here is about a gate that filters (filter, strict), which is experimental.
pytestmark = [pytest.mark.memory_backend_incompatible, pytest.mark.usefixtures("context_admission_experimental")]

QDATE = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

DEPLOY_QUERY = "How is the payments API deployed to production?"
JOKE_QUERY = "What is the running team joke about deploying the payments API?"
DEPLOY = "The payments API is deployed to production by Argo CD from the payments-deploy repository."
CANARY = "Payments API production releases go through an Argo Rollouts canary at 10 percent for 15 minutes."
JOKE = "Running team joke: we don't deploy the payments API, we sacrifice it to the Kubernetes gods."
COFFEE = "The office coffee machine is descaled every Friday."

BUDGET_TEXTS = [
    "The payments API is deployed to production by Argo CD from the payments-deploy repository.",
    "Payments API production releases go through an Argo Rollouts canary at 10 percent for 15 minutes.",
    "Rolling back a payments API production deploy means reverting the image tag in payments-deploy.",
    "Payments API deploys to production are frozen on Fridays after 14:00 UTC.",
    "The payments API production deploy pipeline runs integration tests against a staging cluster first.",
]


def _bank() -> str:
    return f"test-ctxinv-{uuid.uuid4().hex[:8]}"


async def _configure(memory, bank_id: str, **updates) -> None:
    await memory._config_resolver.update_bank_config(bank_id, updates)


async def _recall(memory, rc, bank_id: str, query: str, **kwargs):
    kwargs.setdefault("max_tokens", 4096)
    kwargs.setdefault("fact_type", ["world"])
    kwargs.setdefault("budget", Budget.MID)
    kwargs.setdefault("question_date", QDATE)
    kwargs.setdefault("_external", True)
    return await memory.recall_async(bank_id=bank_id, query=query, request_context=rc, **kwargs)


def _decisions(result) -> dict[str, dict]:
    """The gate's decisions by memory id ({} when the gate did not run)."""
    admission = result.trace.get("context_admission") or {}
    return {d["memory_id"]: d for d in admission.get("decisions", [])}


# --------------------------------------------------------------------------- currentness on real claims

ADMIT_ALL = {"context_admission_threshold_auto": 0.0, "context_admission_threshold_explicit": 0.0}


async def _java(memory, rc, scenario, **kwargs) -> tuple[set[str], dict[str, dict], dict]:
    """(names returned, {name: admission decision}, the trace) for one recall of the Java scenario."""
    kwargs.setdefault("fact_type", ["world", "observation"])
    result = await memory.recall_async(
        scenario.bank_id, JAVA_QUERY, request_context=rc, enable_trace=True, _external=True, **kwargs
    )
    shown = {scenario.names[r.id] for r in result.results if r.id in scenario.names}
    decisions = {scenario.names[m]: d for m, d in _decisions(result).items() if m in scenario.names}
    return shown, decisions, result.trace


async def test_a_present_state_recall_rejects_the_superseded_memories_a_historical_view_shows(
    memory, request_context, java_upgrade
):
    """The bank's default view is historical, so the view hands the gate Java 17 too; a present-state
    task must still get only the current runtime, and the trace says why the rest was rejected."""
    await _configure(
        memory,
        java_upgrade.bank_id,
        adaptive_memory_default_view="historical",
        context_admission_mode="filter",
        **ADMIT_ALL,
    )
    shown, decisions, trace = await _java(memory, request_context, java_upgrade, purpose="auto_context")
    assert trace["lifecycle"]["view"] == "historical"
    assert trace["context_admission"]["historical"] is False
    assert shown == {"O2", "F2"}
    assert {name: d["reason"] for name, d in decisions.items()} == {
        "O1": "superseded",
        "F1": "superseded",
        "O2": "admitted",
        "F2": "admitted",
    }
    assert decisions["O1"]["truth_status"] == "superseded"
    assert trace["context_admission"]["reasons"]["superseded"] == 2


async def test_a_historical_search_reads_the_history_view_and_admits_superseded_memories(
    memory, request_context, java_upgrade
):
    await _configure(memory, java_upgrade.bank_id, context_admission_mode="filter", **ADMIT_ALL)
    shown, decisions, trace = await _java(memory, request_context, java_upgrade, purpose="historical_search")
    assert trace["lifecycle"]["view"] == "historical"
    assert trace["context_admission"]["historical"] is True
    assert shown == {"O1", "F1", "O2", "F2"}
    assert decisions["O1"]["reason"] == decisions["F1"]["reason"] == "admitted"
    assert "R" not in shown and "M" not in shown

    # A view the request names wins over the purpose's default.
    current, _, trace = await _java(
        memory, request_context, java_upgrade, purpose="historical_search", lifecycle_view="current"
    )
    assert trace["lifecycle"]["view"] == "current" and current == {"O2", "F2"}

    # The purpose picks the history view whatever the admission mode, the gate's own off included.
    await _configure(memory, java_upgrade.bank_id, context_admission_mode="off")
    ungated, _, trace = await _java(memory, request_context, java_upgrade, purpose="historical_search")
    assert trace["lifecycle"]["view"] == "historical" and "context_admission" not in trace
    assert ungated == {"O1", "F1", "O2", "F2"}


async def test_a_refuted_memory_is_rejected_even_where_the_view_shows_it(memory, request_context, java_upgrade):
    # The full gate, with duplicates only at cosine 1.0 so the Java 17 / 21 rewordings all stay.
    await _configure(
        memory,
        java_upgrade.bank_id,
        context_admission_mode="strict",
        context_admission_dup_similarity=1.0,
        **ADMIT_ALL,
    )
    for purpose in ("auto_context", "explicit_search", "historical_search"):
        shown, decisions, trace = await _java(
            memory, request_context, java_upgrade, purpose=purpose, lifecycle_view="all-with-status"
        )
        assert trace["lifecycle"]["view"] == "all-with-status", purpose
        assert decisions["R"]["reason"] == "refuted" and decisions["R"]["truth_status"] == "refuted", purpose
        assert "R" not in shown, purpose
        assert {"O2", "F2"} <= shown, purpose


# --------------------------------------------------------------------------- humor is routed, not deleted


async def _humor_bank(memory, rc) -> tuple[str, dict[str, str]]:
    bank_id = _bank()
    await memory.ensure_bank_profile(bank_id, request_context=rc)
    rows = (("deploy", DEPLOY), ("canary", CANARY), ("joke", JOKE), ("coffee", COFFEE))
    ids = {
        name: await _seed(memory, bank_id, text, mentioned_at=QDATE - timedelta(days=10 + i))
        for i, (name, text) in enumerate(rows)
    }
    return bank_id, ids


async def test_a_joke_the_task_asks_for_is_admitted_and_the_gate_never_removes_it(memory, request_context):
    bank_id, ids = await _humor_bank(memory, request_context)
    try:
        ungated = await _recall(memory, request_context, bank_id, DEPLOY_QUERY)
        assert ids["joke"] in {r.id for r in ungated.results}, "retrieval alone hands the joke over"

        await _configure(memory, bank_id, context_admission_mode="strict")
        await _recall(memory, request_context, bank_id, DEPLOY_QUERY, purpose="auto_context")
        asked = await _recall(memory, request_context, bank_id, JOKE_QUERY, purpose="auto_context", enable_trace=True)
        assert ids["joke"] in {r.id for r in asked.results}
        assert _decisions(asked)[ids["joke"]]["reason"] == "admitted"

        # The gate writes nothing: after gated recalls the joke is a live memory, and an ungated
        # recall returns it exactly as before.
        unit = await memory.get_memory_unit(bank_id, ids["joke"], request_context=request_context)
        assert unit is not None and unit["text"] == JOKE
        await _configure(memory, bank_id, context_admission_mode="off")
        again = await _recall(memory, request_context, bank_id, DEPLOY_QUERY)
        assert [r.id for r in again.results] == [r.id for r in ungated.results]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The deterministic scorer has no casual/humor signal: its relevance is the cross-encoder's, "
        "which scores the Kubernetes-gods joke as relevant to the deploy question (reranker 0.81, "
        "admission score 0.86 against the 0.6 auto threshold at the default weights)."
    ),
)
async def test_the_deterministic_scorer_rejects_a_joke_for_a_technical_task(memory, request_context):
    """The requirements' own example: semantic similarity is not relevance."""
    bank_id, ids = await _humor_bank(memory, request_context)
    try:
        await _configure(memory, bank_id, context_admission_mode="strict")
        technical = await _recall(
            memory, request_context, bank_id, DEPLOY_QUERY, purpose="auto_context", enable_trace=True
        )
        names = {v: k for k, v in ids.items()}
        scores = {names[m]: round(d["admission_score"], 3) for m, d in _decisions(technical).items() if m in names}
        assert ids["deploy"] in {r.id for r in technical.results}, scores
        joke = _decisions(technical)[ids["joke"]]
        assert joke["reason"] == "below_threshold", f"{scores} {joke['features']}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- strength is not relevance

WORKAROUND_QUERY = "What exact workaround did we use for FIN-1602?"
WORKAROUND = "FIN-1602 workaround: disabled HTTP/2 on the payments API's outbound client to the card network."
ONCALL = "The payments team's on-call rotation hands over every Monday at 09:00 UTC."


async def _seed_strength(
    memory, bank_id: str, memory_id: str, *, days_since: float, stability: float, importance: float
):
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_retention_state
                (memory_id, bank_id, importance, stability_days, last_reinforced_at, reinforcement_count,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, 3, $6, $6)
            """,
            uuid.UUID(memory_id),
            bank_id,
            importance,
            stability,
            QDATE - timedelta(days=days_since),
            QDATE,
        )


async def test_with_real_scores_a_weak_exact_answer_is_admitted_and_a_strong_irrelevant_memory_is_not(
    memory, request_context
):
    """Retrievability 0.15 (stability 10 days, 19 days since) on the exact answer; importance 1.0,
    stability 1000 days and retrievability 0.99 on a memory that does not answer the question. Scored
    by the local cross-encoder, not hand-picked, at the default automatic threshold."""
    bank_id = _bank()
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        weak = await _seed(memory, bank_id, WORKAROUND, mentioned_at=QDATE - timedelta(days=19))
        strong = await _seed(memory, bank_id, ONCALL, mentioned_at=QDATE - timedelta(days=10))
        await _seed_strength(memory, bank_id, weak, days_since=19, stability=10.0, importance=0.3)
        await _seed_strength(memory, bank_id, strong, days_since=10, stability=1000.0, importance=1.0)
        await _configure(memory, bank_id, adaptive_memory_mode="rank", context_admission_mode="strict")

        result = await _recall(
            memory, request_context, bank_id, WORKAROUND_QUERY, purpose="auto_context", enable_trace=True
        )
        decisions = _decisions(result)
        assert decisions[weak]["retrievability"] == pytest.approx(0.15, abs=0.005)
        assert decisions[strong]["retrievability"] == pytest.approx(0.99, abs=0.005)
        assert [r.id for r in result.results] == [weak]
        assert decisions[strong]["reason"] == "below_threshold"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- the budget through recall


async def test_the_admission_budget_caps_a_strict_recall_by_count_and_by_tokens(memory, request_context):
    bank_id = _bank()
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        for i, text in enumerate(BUDGET_TEXTS):
            await _seed(memory, bank_id, text, mentioned_at=QDATE - timedelta(days=10 + i))
        # Thresholds at 0 and duplicates only at cosine 1.0: everything is admissible, so the
        # budget is the only thing that can cut.
        await _configure(
            memory, bank_id, context_admission_mode="strict", context_admission_dup_similarity=1.0, **ADMIT_ALL
        )
        everything = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert len(everything.results) == len(BUDGET_TEXTS)

        await _configure(memory, bank_id, context_admission_max_memories=2)
        capped = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert len(capped.results) == 2
        assert capped.context_admission.admitted == 2
        assert capped.trace["context_admission"]["reasons"]["budget"] == len(BUDGET_TEXTS) - 2

        shortest = min(count_tokens(t) for t in BUDGET_TEXTS)
        await _configure(memory, bank_id, context_admission_max_memories=10, context_admission_max_tokens=shortest)
        tight = await _recall(memory, request_context, bank_id, DEPLOY_QUERY, enable_trace=True)
        assert len(tight.results) == 1
        assert sum(count_tokens(r.text) for r in tight.results) <= shortest
        assert tight.trace["context_admission"]["reasons"]["budget"] == len(BUDGET_TEXTS) - 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- no reinforcement loop


async def _ledger_rows(memory, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM memory_reinforcement_events WHERE bank_id = $1", bank_id)


async def _retention_rows(memory, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM memory_retention_state WHERE bank_id = $1", bank_id)


async def test_repeated_gated_recalls_never_reinforce_what_they_admit(memory, request_context, monkeypatch):
    """Recall reinforcement switched on, three recalls 30 days apart (past the cooldown): a gated
    recall writes no event and no retention row, so the retrievability the gate scores a memory on
    only decays. The observe bank, whose results are a plain recall's, reinforces them, and the same
    memory comes back more retrievable there: the loop the guard cuts is real."""
    monkeypatch.setattr(_get_raw_config(), "adaptive_memory_recall_reinforcement_enabled", True)
    dates = [QDATE + timedelta(days=30 * k) for k in range(3)]
    gated, observed = _bank(), _bank()
    tracked: dict[str, str] = {}
    try:
        for bank_id, mode in ((gated, "strict"), (observed, "observe")):
            await memory.ensure_bank_profile(bank_id, request_context=request_context)
            seeded = [
                await _seed(memory, bank_id, text, mentioned_at=QDATE - timedelta(days=10 + i))
                for i, text in enumerate(BUDGET_TEXTS[:3])
            ]
            tracked[bank_id] = seeded[0]
            await _configure(memory, bank_id, adaptive_memory_mode="rank", context_admission_mode=mode, **ADMIT_ALL)

        retrievability: dict[str, list[float]] = {gated: [], observed: []}
        for when in dates:
            for bank_id in (gated, observed):
                result = await _recall(
                    memory, request_context, bank_id, DEPLOY_QUERY, question_date=when, enable_trace=True
                )
                assert tracked[bank_id] in {r.id for r in result.results}
                retrievability[bank_id].append(_decisions(result)[tracked[bank_id]]["retrievability"])
                await memory._adaptive_reinforcement.drain()

        assert await _ledger_rows(memory, gated) == 0
        assert await _retention_rows(memory, gated) == 0
        assert retrievability[gated] == sorted(retrievability[gated], reverse=True), retrievability
        assert retrievability[gated][0] > retrievability[gated][-1], retrievability

        assert await _ledger_rows(memory, observed) > 0
        assert await _retention_rows(memory, observed) > 0
        assert retrievability[observed][-1] > retrievability[gated][-1], retrievability
    finally:
        for bank_id in (gated, observed):
            await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- no generative call on recall


def _llm_calls(memory) -> list[dict]:
    impls = {
        id(c._provider_impl): c._provider_impl
        for c in (
            memory._llm_config,
            memory._retain_llm_config,
            memory._reflect_llm_config,
            memory._consolidation_llm_config,
        )
    }
    return [call for impl in impls.values() for call in impl.get_mock_calls()]


@pytest.mark.parametrize(
    ("adaptive_mode", "admission_mode"),
    [("off", "off"), ("observe", "observe"), ("rank", "filter"), ("lifecycle", "strict")],
)
async def test_no_recall_calls_a_generative_model_in_any_mode(memory, request_context, adaptive_mode, admission_mode):
    bank_id = _bank()
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
        for i, text in enumerate((DEPLOY, CANARY, JOKE, COFFEE)):
            await _seed(memory, bank_id, text, mentioned_at=QDATE - timedelta(days=10 + i))
        await _configure(
            memory,
            bank_id,
            adaptive_memory_mode=adaptive_mode,
            adaptive_memory_historical_cues=True,
            context_admission_mode=admission_mode,
        )
        before = len(_llm_calls(memory))
        for purpose in ("auto_context", "explicit_search", "historical_search"):
            for query in (DEPLOY_QUERY, "How was the payments API deployed before the Argo CD migration?"):
                await _recall(memory, request_context, bank_id, query, purpose=purpose, enable_trace=True)
        assert _llm_calls(memory)[before:] == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
