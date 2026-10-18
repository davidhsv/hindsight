"""The adaptive-memory reinforcement producers: consolidation, reflect, recall, and retain-time policy.

The store's writer is covered in test_adaptive_reinforcement_store.py; these tests drive the engine
operations that produce events and check what lands in the ledger and the retention rows:

- consolidation (scripted LLM response, pg0): a new observation's creating evidence anchors it on
  the earliest fact; a later fact from a new session that re-states it reinforces it, seeded from
  the pre-update mention time; a fact from a session it has already heard reinforces nothing;
  server-minted documents count half; a failed write never costs the batch;
- reflect: the success / failure / refresh / document / ``answer_found=false`` matrix;
- recall: only a final external recall (the flag) with the static switch on, never an internal one;
- retain: a minted document id is marked (and the mark survives an append), importance and the
  forgetting exemption materialize retention rows, only in rank/lifecycle.

Seeding uses raw SQL for exact timestamps and provenance, so the module is
``memory_backend_incompatible``.
"""

import asyncio
import dataclasses
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from hindsight_api.api import create_app
from hindsight_api.config import _get_raw_config, get_config
from hindsight_api.engine.adaptive import producers
from hindsight_api.engine.adaptive.evidence import WEAK_INDEPENDENCE, evidence_key
from hindsight_api.engine.adaptive.models import RetentionState
from hindsight_api.engine.adaptive.producers import (
    BackgroundReinforcement,
    FactEvidence,
    ObservationEvidence,
    recall_reinforcement_allowed,
    usage_events,
)
from hindsight_api.engine.adaptive.retention import apply_events, initial_stability
from hindsight_api.engine.adaptive.settings import AdaptiveSettings
from hindsight_api.engine.consolidation import consolidator
from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _CreateAction,
    _DeleteAction,
    _UpdateAction,
    run_consolidation_job,
)
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.reflect.agent import _process_done_tool
from hindsight_api.engine.reflect.models import ReflectAgentResult, TokenUsageSummary, ToolCall
from hindsight_api.engine.retain import embedding_utils
from tests.test_mcp_tools import mock_memory  # noqa: F401 — fixture

pytestmark = pytest.mark.memory_backend_incompatible

DAY0 = datetime(2026, 1, 1, tzinfo=UTC)


def day(n: float) -> datetime:
    return DAY0 + timedelta(days=n)


def _settings(mode: str = "rank", **overrides) -> AdaptiveSettings:
    settings = AdaptiveSettings.resolve({"adaptive_memory_mode": mode}, get_config(), store_supports=True)
    return dataclasses.replace(settings, **overrides)


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


@pytest.fixture
def observations_off(monkeypatch):
    """Retain without the consolidation it would otherwise trigger."""
    monkeypatch.setattr(_get_raw_config(), "enable_observations", False)


@pytest.fixture
def recall_switch_on(monkeypatch):
    monkeypatch.setattr(_get_raw_config(), "adaptive_memory_recall_reinforcement_enabled", True)


async def _bank(memory, request_context, mode: str | None = "rank") -> str:
    bank_id = f"test-amp-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    if mode is not None:
        await _set_mode(memory, bank_id, mode)
    return bank_id


async def _set_mode(memory, bank_id: str, mode: str) -> None:
    await memory._config_resolver.update_bank_config(bank_id, {"adaptive_memory_mode": mode})


_SEQ = iter(range(1_000_000))


async def _fact(
    memory,
    bank_id: str,
    text: str,
    *,
    at: datetime,
    session: str | None = None,
    document_id: str | None = None,
    minted: bool = False,
) -> str:
    """A raw world fact with exact provenance, embedded so recall finds it; consolidation picks
    facts up in insertion order."""
    memory_id = uuid.uuid4()
    (embedding,) = await embedding_utils.generate_embeddings_batch(memory.embeddings, [text])
    async with memory._pool.acquire() as conn:
        if document_id is not None:
            await conn.execute(
                "INSERT INTO documents (id, bank_id, retain_params) VALUES ($1, $2, $3::jsonb) "
                "ON CONFLICT (id, bank_id) DO NOTHING",
                document_id,
                bank_id,
                json.dumps({"document_id_minted": True} if minted else {}),
            )
        await conn.execute(
            """
            INSERT INTO memory_units
                (id, bank_id, text, fact_type, mentioned_at, metadata, document_id, embedding, created_at)
            VALUES ($1, $2, $3, 'world', $4, $5::jsonb, $6, $7::vector, now() + make_interval(secs => $8))
            """,
            memory_id,
            bank_id,
            text,
            at,
            json.dumps({"session_id": session} if session else {}),
            document_id,
            "[" + ",".join(str(v) for v in embedding) + "]",
            float(next(_SEQ)) / 1000.0,
        )
    return str(memory_id)


async def _consolidate(memory, bank_id: str, request_context, respond) -> None:
    """One consolidation job whose LLM response is ``respond(memories, union_observations)``."""

    async def scripted(**kwargs) -> _BatchLLMResult:
        return respond(kwargs["memories"], kwargs["union_observations"])

    with (
        patch.object(consolidator, "_consolidate_batch_with_llm", scripted),
        patch.object(memory, "submit_async_consolidation"),
    ):
        await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)


def _create(text: str):
    def respond(memories, _pool):
        return _BatchLLMResult(creates=[_CreateAction(text=text, source_fact_ids=[str(m["id"]) for m in memories])])

    return respond


def _update_first(text: str | None = None):
    """UPDATE the one observation the recall pooled, with ``text`` (None keeps its text)."""

    def respond(memories, pool):
        assert pool, "the observation was not recalled for the new fact"
        observation = pool[0]
        return _BatchLLMResult(
            updates=[
                _UpdateAction(
                    observation_id=str(observation.id),
                    text=text if text is not None else observation.text,
                    source_fact_ids=[str(m["id"]) for m in memories],
                )
            ]
        )

    return respond


async def _rows(memory, sql: str, *args):
    async with memory._pool.acquire() as conn:
        return await conn.fetch(sql, *args)


async def _observations(memory, bank_id: str, request_context) -> dict[str, str]:
    """The bank's observations, text -> id."""
    page = await memory.list_memory_units(bank_id, fact_type="observation", request_context=request_context)
    return {item["text"]: str(item["id"]) for item in page["items"]}


async def _observation_id(memory, bank_id: str, request_context) -> str:
    (observation_id,) = (await _observations(memory, bank_id, request_context)).values()
    return observation_id


async def _unit(memory, bank_id: str, memory_id: str, request_context) -> dict:
    unit = await memory.get_memory_unit(bank_id, memory_id, request_context=request_context)
    assert unit is not None
    return unit


async def _retention(memory, memory_id: str):
    rows = await _rows(memory, "SELECT * FROM memory_retention_state WHERE memory_id = $1", uuid.UUID(memory_id))
    return rows[0] if rows else None


async def _ledger(memory, memory_id: str):
    return await _rows(
        memory,
        "SELECT source_type, source_key, event_time, outcome, applied_gain, independence_weight, weight "
        "FROM memory_reinforcement_events WHERE memory_id = $1 ORDER BY event_time, source_key",
        uuid.UUID(memory_id),
    )


async def _bank_ledger(memory, bank_id: str):
    return await _rows(
        memory,
        "SELECT memory_id, source_type, source_key FROM memory_reinforcement_events WHERE bank_id = $1",
        bank_id,
    )


def _fresh(memory_id: str, anchor: datetime, fact_type: str) -> RetentionState:
    return RetentionState(
        memory_id=memory_id,
        importance=0.5,
        stability_days=initial_stability(fact_type, 0.5, _settings()),
        last_reinforced_at=anchor,
        reinforcement_count=0,
        access_count=0,
        forgetting_exempt=False,
        retention_state="active",
        last_recalled_at=None,
        below_threshold_since=None,
        archived_at=None,
    )


# --------------------------------------------------------------------------- pure: event building


def _evidence(key: str, at: datetime, weight: float = 1.0) -> FactEvidence:
    return FactEvidence(key=SimpleNamespace(key=key, independence_weight=weight), event_time=at)  # type: ignore[arg-type]


def test_creating_evidence_is_one_initial_event_per_key_anchored_on_the_earliest():
    plan = ObservationEvidence(
        _settings(),
        facts={
            "a": _evidence("session:s1", day(10)),
            "b": _evidence("session:s1", day(12)),  # same session: counted once, by its earliest
            "c": _evidence("session:s2", day(40)),
        },
    )
    plan.created("obs", ["c", "b", "a"])
    call = plan.call()
    assert call is not None
    assert [(e.source_type, e.source_key, e.event_time) for e in call.events] == [
        ("observation_new_evidence", "session:s1", day(10)),
        ("observation_new_evidence", "session:s2", day(40)),
    ]
    assert call.anchors == {"obs": day(10)}
    assert call.initial_memory_ids == frozenset({"obs"})


def test_an_update_counts_only_keys_the_observation_has_not_heard():
    plan = ObservationEvidence(
        _settings(),
        facts={
            "old": _evidence("session:s1", day(0)),
            "same": _evidence("session:s1", day(30)),
            "new": _evidence("session:s2", day(30), weight=WEAK_INDEPENDENCE),
            "again": _evidence("session:s2", day(31)),
        },
        prior_sources={"obs": ["old"]},
        prior_mentioned_at={"obs": day(5)},
    )
    plan.attached("obs", ["same"], unchanged_text=True, previous_mentioned_at=day(0))
    assert plan.call() is None, "a session the observation was built from re-states nothing new"

    plan.attached("obs", ["new", "again", "old"], unchanged_text=True, previous_mentioned_at=day(0))
    call = plan.call()
    assert call is not None
    assert [(e.source_type, e.source_key, e.independence_weight) for e in call.events] == [
        ("independent_retain_confirmation", "session:s2", WEAK_INDEPENDENCE)
    ]
    # Seeded from the pre-update mention time the history snapshot holds, not the prior read.
    assert call.anchors == {"obs": day(0)}
    assert call.initial_memory_ids == frozenset()


def test_a_rewrite_or_a_fold_is_new_evidence_not_a_confirmation():
    plan = ObservationEvidence(
        _settings(),
        facts={"x": _evidence("session:a", day(1)), "y": _evidence("session:b", day(2))},
        prior_mentioned_at={"twin": day(-3)},
    )
    plan.attached("obs", ["x"], unchanged_text=False, previous_mentioned_at=None)
    plan.folded("twin", ["y", "unknown-fact"])  # a fact with no provenance cannot be keyed: skipped
    call = plan.call()
    assert call is not None
    assert [(e.memory_id, e.source_type) for e in call.events] == [
        ("obs", "observation_new_evidence"),
        ("twin", "observation_new_evidence"),
    ]
    assert call.anchors == {"twin": day(-3)}


def test_usage_events_share_one_key_per_cooldown_window():
    settings = _settings(reinforcement_cooldown_hours=6.0)
    at = datetime(2026, 3, 1, 13, 30, tzinfo=UTC)
    events = usage_events(["m1", "m2", "m1", ""], source_type="final_external_recall", event_time=at, settings=settings)
    assert [e.memory_id for e in events] == ["m1", "m2"]
    assert {e.source_key for e in events} == {"recall:2026-03-01T12:00:00+00:00"}
    assert events[0].weight == settings.source_weights["final_external_recall"]
    reflect = usage_events(["m1"], source_type="successful_reflect_use", event_time=at, settings=settings)
    assert reflect[0].source_key.startswith("reflect:")


def test_the_recall_gate_needs_the_mode_the_bank_switch_and_the_static_switch():
    on = _settings(recall_reinforcement_enabled=True)
    assert recall_reinforcement_allowed(on)
    assert not recall_reinforcement_allowed(dataclasses.replace(on, recall_reinforcement_enabled=False))
    assert not recall_reinforcement_allowed(dataclasses.replace(on, reinforcement_enabled=False))
    assert not recall_reinforcement_allowed(_settings("observe", recall_reinforcement_enabled=True))


def test_write_settings_reads_off_and_unknown_modes_as_none():
    assert producers.write_settings(SimpleNamespace(), bank_id="b") is None
    assert producers.write_settings(SimpleNamespace(adaptive_memory_mode="off"), bank_id="b") is None
    assert producers.write_settings(SimpleNamespace(adaptive_memory_mode="observe"), bank_id="b") is None
    assert producers.write_settings(SimpleNamespace(adaptive_memory_mode="bogus"), bank_id="b") is None
    settings = producers.write_settings(SimpleNamespace(adaptive_memory_mode="lifecycle"), bank_id="b")
    assert settings is not None and settings.effective_mode.applies_factor


# --------------------------------------------------------------------------- background writer


def _metric_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    class _Collector:
        def record_adaptive_reinforcement_dropped(self, **kwargs):
            calls.append(kwargs)

        def record_adaptive_reinforcement(self, **kwargs):
            pass

    monkeypatch.setattr("hindsight_api.metrics.get_metrics_collector", lambda: _Collector())
    return calls


@pytest.mark.asyncio
async def test_background_writes_past_the_bound_are_dropped_and_counted(monkeypatch):
    calls = _metric_calls(monkeypatch)
    release = asyncio.Event()

    async def slow_backend():
        await release.wait()
        raise RuntimeError("stop here")

    writer = BackgroundReinforcement(slow_backend, max_in_flight=1)
    events = usage_events(["m"], source_type="successful_reflect_use", event_time=day(1), settings=_settings())
    assert writer.submit(producer="reflect", bank_id="b", events=events, settings=_settings())
    assert not writer.submit(producer="reflect", bank_id="b", events=events, settings=_settings())
    assert not writer.submit(producer="reflect", bank_id="b", events=[], settings=_settings())
    assert calls == [{"producer": "reflect", "reason": "overflow", "bank_id": "b"}]

    release.set()
    await writer.drain()
    # The failed write is logged and counted, never raised; its slot is free again.
    assert calls[-1] == {"producer": "reflect", "reason": "error", "bank_id": "b"}
    assert writer.in_flight == 0


# --------------------------------------------------------------------------- consolidation


@pytest.mark.asyncio
async def test_a_new_observation_is_anchored_on_its_earliest_creating_evidence(memory, request_context):
    bank_id = await _bank(memory, request_context)
    await _fact(memory, bank_id, "The payments team deploys on Fridays.", at=day(10), session="s1")
    await _fact(memory, bank_id, "The payments team deploys on Fridays.", at=day(40), session="s2")

    await _consolidate(memory, bank_id, request_context, _create("The payments team deploys on Fridays."))

    obs = await _observation_id(memory, bank_id, request_context)
    ledger = await _ledger(memory, obs)
    assert [(r["source_type"], r["source_key"], r["outcome"]) for r in ledger] == [
        ("observation_new_evidence", "session:s1", "initial"),
        ("observation_new_evidence", "session:s2", "initial"),
    ]
    assert ledger[0]["applied_gain"] == 0.0 and ledger[1]["applied_gain"] > 0
    expected = apply_events(
        _fresh(obs, day(10), "observation"),
        [
            producers.reinforcement_event(
                memory_id=obs,
                source_type="observation_new_evidence",
                source_key=key,
                event_time=at,
                settings=_settings(),
            )
            for key, at in (("session:s1", day(10)), ("session:s2", day(40)))
        ],
        _settings(),
        initial=True,
    )
    row = await _retention(memory, obs)
    assert row["last_reinforced_at"] == day(40)
    assert row["reinforcement_count"] == 1
    assert row["stability_days"] == pytest.approx(expected.state.stability_days)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_an_independent_update_a_month_later_reinforces_from_the_pre_update_mention(memory, request_context):
    # Built while the bank was off: no retention row, and the update below advances the
    # observation's mentioned_at to day 30, which would read as "reinforced just now".
    bank_id = await _bank(memory, request_context, mode=None)
    text = "Alice leads the platform team."
    await _fact(memory, bank_id, text, at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create(text))
    obs = await _observation_id(memory, bank_id, request_context)
    assert await _retention(memory, obs) is None

    await _set_mode(memory, bank_id, "rank")
    await _fact(memory, bank_id, text, at=day(30), session="s2")
    await _consolidate(memory, bank_id, request_context, _update_first())

    (event,) = await _ledger(memory, obs)
    assert (event["source_type"], event["source_key"], event["outcome"]) == (
        "independent_retain_confirmation",
        "session:s2",
        "applied",
    )
    s0 = initial_stability("observation", 0.5, _settings())
    gain = _settings().source_weights["independent_retain_confirmation"] * (1 - math.exp(-30 / s0))
    assert event["applied_gain"] == pytest.approx(gain)
    row = await _retention(memory, obs)
    assert row["last_reinforced_at"] == day(30)
    assert row["reinforcement_count"] == 1
    assert row["stability_days"] == pytest.approx(s0 * (1 + gain))
    mentioned = (await _unit(memory, bank_id, obs, request_context))["mentioned_at"]
    assert datetime.fromisoformat(mentioned) == day(30), "the column the anchor must not be read from moved"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_an_update_from_a_session_already_heard_records_nothing(memory, request_context):
    bank_id = await _bank(memory, request_context)
    text = "Bob prefers tea over coffee."
    await _fact(memory, bank_id, text, at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create(text))
    obs = await _observation_id(memory, bank_id, request_context)
    before = await _retention(memory, obs)

    await _fact(memory, bank_id, text, at=day(30), session="s1")
    await _consolidate(memory, bank_id, request_context, _update_first())

    ledger = await _ledger(memory, obs)
    assert [(r["source_key"], r["outcome"]) for r in ledger] == [("session:s1", "initial")]
    after = await _retention(memory, obs)
    assert after["reinforcement_count"] == 0
    assert after["updated_at"] == before["updated_at"]
    unit = await _unit(memory, bank_id, obs, request_context)
    assert len(unit["source_memory_ids"]) == 2, "the update itself still landed"
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_rewriting_update_is_new_evidence(memory, request_context):
    bank_id = await _bank(memory, request_context)
    await _fact(memory, bank_id, "Carol works in Lisbon.", at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create("Carol works in Lisbon."))
    obs = await _observation_id(memory, bank_id, request_context)

    await _fact(memory, bank_id, "Carol works in Lisbon.", at=day(60), session="s2")
    await _consolidate(memory, bank_id, request_context, _update_first("Carol works in Lisbon, remotely."))

    ledger = await _ledger(memory, obs)
    assert [(r["source_type"], r["source_key"], r["outcome"]) for r in ledger] == [
        ("observation_new_evidence", "session:s1", "initial"),
        ("observation_new_evidence", "session:s2", "applied"),
    ]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_merged_into_delete_moves_its_evidence_onto_the_survivor(memory, request_context):
    bank_id = await _bank(memory, request_context)
    await _fact(memory, bank_id, "Gus runs the nightly backups.", at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create("Gus runs the nightly backups."))
    await _fact(memory, bank_id, "The weekly restore drill is owned by Gus.", at=day(5), session="s2")
    await _consolidate(memory, bank_id, request_context, _create("The weekly restore drill is owned by Gus."))
    by_text = await _observations(memory, bank_id, request_context)
    survivor, merged = by_text["Gus runs the nightly backups."], by_text["The weekly restore drill is owned by Gus."]

    await _fact(memory, bank_id, "Gus runs the nightly backups.", at=day(40), session="s3")

    def merge(memories, pool):
        assert {str(o.id) for o in pool} == {survivor, merged}
        return _BatchLLMResult(
            updates=[
                _UpdateAction(
                    observation_id=survivor,
                    text="Gus runs the nightly backups.",
                    source_fact_ids=[str(m["id"]) for m in memories],
                )
            ],
            deletes=[_DeleteAction(observation_id=merged, merged_into=survivor)],
        )

    await _consolidate(memory, bank_id, request_context, merge)

    assert [(r["source_type"], r["source_key"], r["outcome"]) for r in await _ledger(memory, survivor)] == [
        ("observation_new_evidence", "session:s1", "initial"),
        ("observation_new_evidence", "session:s2", "applied"),  # moved over from the merged one
        ("independent_retain_confirmation", "session:s3", "applied"),
    ]
    assert (await _retention(memory, survivor))["reinforcement_count"] == 2
    await memory.delete_bank(bank_id, request_context=request_context)


def _dedup_verdict(*, anchor_excluding: str | None, twin_id: str, twin_text: str):
    """A ``_dedup_adjudicate`` that calls the anchor a duplicate of ``twin_id`` (keeping the twin's
    text) for the action it names: a CREATE (``anchor_excluding=None``) or the UPDATE of that
    observation. The real one needs a cosine >= 0.97 and an LLM verdict; the folds' wiring is the
    part under test."""

    async def adjudicate(pool, memory_engine, bank_id, config, llm_config, anchor_text, emb, tags, exclude_id):
        if exclude_id != anchor_excluding:
            return consolidator._DedupOutcome(best_id=None, merged_text="", should_merge=False)
        return consolidator._DedupOutcome(
            best_id=twin_id, merged_text=twin_text, should_merge=True, best_text=twin_text
        )

    return adjudicate


@pytest.mark.asyncio
async def test_a_create_folded_into_its_twin_is_new_evidence_for_the_twin(memory, request_context):
    bank_id = await _bank(memory, request_context)
    text = "Gus runs the nightly backups."
    await _fact(memory, bank_id, text, at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create(text))
    twin = await _observation_id(memory, bank_id, request_context)

    await _fact(memory, bank_id, "Every night Gus runs the backups.", at=day(40), session="s2")
    with patch.object(
        consolidator, "_dedup_adjudicate", _dedup_verdict(anchor_excluding=None, twin_id=twin, twin_text=text)
    ):
        await _consolidate(memory, bank_id, request_context, _create("Gus runs the backups every night."))

    assert list((await _observations(memory, bank_id, request_context)).values()) == [twin], "the CREATE folded"
    assert [(r["source_type"], r["source_key"], r["outcome"]) for r in await _ledger(memory, twin)] == [
        ("observation_new_evidence", "session:s1", "initial"),
        ("observation_new_evidence", "session:s2", "applied"),
    ]
    assert (await _retention(memory, twin))["reinforcement_count"] == 1
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_an_update_folded_into_its_twin_moves_all_its_evidence_to_the_twin(memory, request_context):
    bank_id = await _bank(memory, request_context)
    twin_text = "Gus runs the nightly backups."
    await _fact(memory, bank_id, twin_text, at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create(twin_text))
    await _fact(memory, bank_id, "The weekly restore drill is owned by Gus.", at=day(5), session="s2")
    await _consolidate(memory, bank_id, request_context, _create("The weekly restore drill is owned by Gus."))
    by_text = await _observations(memory, bank_id, request_context)
    twin, rewritten = by_text[twin_text], by_text["The weekly restore drill is owned by Gus."]

    await _fact(memory, bank_id, twin_text, at=day(40), session="s3")

    def rewrite(memories, pool):
        assert rewritten in {str(o.id) for o in pool}
        return _BatchLLMResult(
            updates=[
                _UpdateAction(
                    observation_id=rewritten,
                    text="Gus runs the nightly backups and the restore drill.",
                    source_fact_ids=[str(m["id"]) for m in memories],
                )
            ]
        )

    with patch.object(
        consolidator, "_dedup_adjudicate", _dedup_verdict(anchor_excluding=rewritten, twin_id=twin, twin_text=twin_text)
    ):
        await _consolidate(memory, bank_id, request_context, rewrite)

    assert list((await _observations(memory, bank_id, request_context)).values()) == [twin], "the rewrite folded"
    # The survivor is the twin: the deleted rewritten observation's own key and the new one land there.
    assert [(r["source_type"], r["source_key"], r["outcome"]) for r in await _ledger(memory, twin)] == [
        ("observation_new_evidence", "session:s1", "initial"),
        ("observation_new_evidence", "session:s2", "applied"),
        ("observation_new_evidence", "session:s3", "applied"),
    ]
    assert (await _retention(memory, twin))["reinforcement_count"] == 2
    assert await _ledger(memory, rewritten) == [] and await _retention(memory, rewritten) is None
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_server_minted_documents_count_half(memory, request_context):
    bank_id = await _bank(memory, request_context)
    text = "The billing service runs on Postgres 16."
    minted_doc = f"minted-{uuid.uuid4().hex[:6]}"
    client_doc = f"client-{uuid.uuid4().hex[:6]}"
    await _fact(memory, bank_id, text, at=day(0), document_id=minted_doc, minted=True)
    await _fact(memory, bank_id, text, at=day(50), document_id=client_doc)

    await _consolidate(memory, bank_id, request_context, _create(text))

    obs = await _observation_id(memory, bank_id, request_context)
    weights = {r["source_key"]: r["independence_weight"] for r in await _ledger(memory, obs)}
    assert weights == {f"document:{minted_doc}": WEAK_INDEPENDENCE, f"document:{client_doc}": 1.0}
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_off_observe_and_a_disabled_switch_record_nothing(memory, request_context):
    for mode, overrides in (("off", {}), ("observe", {}), ("rank", {"adaptive_memory_reinforcement_enabled": False})):
        bank_id = await _bank(memory, request_context, mode=mode)
        if overrides:
            await memory._config_resolver.update_bank_config(bank_id, overrides)
        await _fact(memory, bank_id, "Dana owns the release train.", at=day(0), session="s1")
        await _consolidate(memory, bank_id, request_context, _create("Dana owns the release train."))
        assert await _bank_ledger(memory, bank_id) == [], mode
        assert await _retention(memory, await _observation_id(memory, bank_id, request_context)) is None, mode
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_failed_reinforcement_never_fails_the_batch(memory, request_context, monkeypatch):
    bank_id = await _bank(memory, request_context)
    fact = await _fact(memory, bank_id, "Erin rotates the on-call pager.", at=day(0), session="s1")
    dropped: list[dict] = []
    real_collector = consolidator.get_metrics_collector()

    class _Collector:
        def __getattr__(self, name):
            return getattr(real_collector, name)

        def record_adaptive_reinforcement_dropped(self, **kwargs):
            dropped.append(kwargs)

    monkeypatch.setattr(consolidator, "get_metrics_collector", lambda: _Collector())
    monkeypatch.setattr(
        type(get_memories()), "record_reinforcement", AsyncMock(side_effect=RuntimeError("ledger is down"))
    )

    await _consolidate(memory, bank_id, request_context, _create("Erin rotates the on-call pager."))

    obs = await _observation_id(memory, bank_id, request_context)
    assert await _retention(memory, obs) is None
    done = await memory.list_memory_units(
        bank_id, fact_type="world", consolidation_state="done", request_context=request_context
    )
    assert [item["id"] for item in done["items"]] == [fact], "the batch committed, stamps included"
    assert dropped == [{"producer": "consolidation", "reason": "error", "bank_id": bank_id}]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_consolidations_internal_recall_never_reinforces(memory, request_context, recall_switch_on):
    bank_id = await _bank(memory, request_context)
    text = "Frank maintains the search cluster."
    await _fact(memory, bank_id, text, at=day(0), session="s1")
    await _consolidate(memory, bank_id, request_context, _create(text))
    await _fact(memory, bank_id, text, at=day(9), session="s1")
    await _consolidate(memory, bank_id, request_context, _update_first())  # recalls the observation
    await memory._adaptive_reinforcement.drain()

    assert {r["source_type"] for r in await _bank_ledger(memory, bank_id)} == {"observation_new_evidence"}
    await memory.delete_bank(bank_id, request_context=request_context)


# --------------------------------------------------------------------------- reflect


def _agent_result(**kwargs) -> ReflectAgentResult:
    return ReflectAgentResult(text="An answer.", **kwargs)


def _retrieved(memory_ids: list[str], observation_ids: list[str] | None = None) -> list[ToolCall]:
    """A tool trace in which the agent's recall and observation search handed back these ids.

    Everything retrieved is what ``based_on`` falls back to when the agent cites nothing, so a
    reinforcement read from it instead of from the declared ids would show up here.
    """
    trace = [
        ToolCall(
            tool="recall",
            input={"query": "Who owns what?"},
            output={"memories": [{"id": m, "text": "A retrieved fact.", "fact_type": "world"} for m in memory_ids]},
            duration_ms=1,
        )
    ]
    if observation_ids:
        trace.append(
            ToolCall(
                tool="search_observations",
                input={"query": "Who owns what?"},
                output={
                    "observations": [
                        {"id": o, "text": "A retrieved observation.", "fact_type": "observation"}
                        for o in observation_ids
                    ]
                },
                duration_ms=1,
            )
        )
    return trace


async def _reflect(memory, bank_id, request_context, monkeypatch, result, **kwargs):
    async def fake_run_reflect_agent(**agent_kwargs):
        if isinstance(result, Exception):
            raise result
        return result(agent_kwargs) if callable(result) else result

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)
    await memory.reflect_async(bank_id=bank_id, query="Who owns what?", request_context=request_context, **kwargs)
    await memory._adaptive_reinforcement.drain()


@pytest.mark.asyncio
async def test_reflect_reinforces_only_what_a_successful_answer_cited(memory, request_context, monkeypatch):
    bank_id = await _bank(memory, request_context)
    cited = await _fact(memory, bank_id, "Gina approves the budget.", at=day(0))
    observation = await _fact(memory, bank_id, "Gina is the finance lead.", at=day(0))
    uncited = await _fact(memory, bank_id, "Retrieved but never cited.", at=day(0))
    uncited_observation = await _fact(memory, bank_id, "Also retrieved, never cited.", at=day(0))

    await _reflect(
        memory,
        bank_id,
        request_context,
        monkeypatch,
        _agent_result(
            used_memory_ids=[cited],
            used_observation_ids=[observation],
            answer_found=True,
            tool_trace=_retrieved([cited, uncited], [observation, uncited_observation]),
        ),
    )

    by_memory = {str(r["memory_id"]): r for r in await _bank_ledger(memory, bank_id)}
    assert set(by_memory) == {cited, observation}
    assert {r["source_type"] for r in by_memory.values()} == {"successful_reflect_use"}
    assert all(r["source_key"].startswith("reflect:") for r in by_memory.values())
    row = await _retention(memory, cited)
    assert row["reinforcement_count"] == 1 and row["access_count"] == 0
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["answer_not_found", "refresh", "document", "failure", "nothing_cited", "static_switch_off", "observe_mode"],
)
async def test_reflect_that_must_not_reinforce(memory, request_context, monkeypatch, case):
    bank_id = await _bank(memory, request_context, mode="observe" if case == "observe_mode" else "rank")
    cited = await _fact(memory, bank_id, "Hana signs off releases.", at=day(0))
    result: object = _agent_result(used_memory_ids=[cited], answer_found=True)
    kwargs: dict = {}
    if case == "answer_not_found":
        result = _agent_result(used_memory_ids=[cited], answer_found=False)
    elif case == "refresh":
        kwargs["_operation_label"] = "refresh_mental_model"
    elif case == "document":
        kwargs["answer_as_document"] = True
    elif case == "failure":
        result = RuntimeError("the agent failed")
    elif case == "nothing_cited":
        # It retrieved the fact and cited nothing: based_on then holds everything retrieved.
        result = _agent_result(answer_found=True, tool_trace=_retrieved([cited]))
    elif case == "static_switch_off":
        monkeypatch.setattr(_get_raw_config(), "adaptive_memory_reflect_reinforcement_enabled", False)

    if case == "failure":
        with pytest.raises(RuntimeError):
            await _reflect(memory, bank_id, request_context, monkeypatch, result, **kwargs)
        await memory._adaptive_reinforcement.drain()
    else:
        await _reflect(memory, bank_id, request_context, monkeypatch, result, **kwargs)

    assert await _bank_ledger(memory, bank_id) == []
    assert await _retention(memory, cited) is None
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_reflects_recall_tool_never_reinforces(memory, request_context, monkeypatch, recall_switch_on):
    bank_id = await _bank(memory, request_context)
    await _fact(memory, bank_id, "Ivan runs the data pipeline.", at=day(0))
    seen: list = []

    async def calls_its_recall_tool(agent_kwargs):
        seen.append(await agent_kwargs["recall_fn"]("Who runs the data pipeline?", 2000, 0))
        return _agent_result(answer_found=False)

    async def fake_run_reflect_agent(**agent_kwargs):
        return await calls_its_recall_tool(agent_kwargs)

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_run_reflect_agent)
    await memory.reflect_async(bank_id=bank_id, query="Who runs it?", request_context=request_context)
    await memory._adaptive_reinforcement.drain()

    assert seen, "the tool ran"
    assert await _bank_ledger(memory, bank_id) == []
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_the_done_call_carries_answer_found_onto_the_result():
    from hindsight_api.engine.response_models import LLMToolCall

    async def done(arguments):
        return await _process_done_tool(
            LLMToolCall(id="1", name="done", arguments={"answer": "Nothing known.", **arguments}),
            {"m1"},
            set(),
            set(),
            1,
            1,
            [],
            [],
            TokenUsageSummary(),
            lambda *_args, **_kwargs: None,
            "reflect-test",
            [],
            presenter=SimpleNamespace(drop_unseen_ids=lambda text: text),
        )

    assert (await done({"answer_found": False, "memory_ids": ["m1"]})).answer_found is False
    assert (await done({"answer_found": "false"})).answer_found is False
    assert (await done({})).answer_found is True, "a done without the field reads as found"
    assert ReflectAgentResult(text="forced").answer_found is None


# --------------------------------------------------------------------------- recall


async def _recall(memory, bank_id, request_context, *, max_tokens: int = 2048, **kwargs):
    result = await memory.recall_async(
        bank_id=bank_id,
        query="Who maintains the deploy scripts?",
        fact_type=["world"],
        budget=Budget.LOW,
        max_tokens=max_tokens,
        request_context=request_context,
        **kwargs,
    )
    await memory._adaptive_reinforcement.drain()
    return result


@pytest.mark.asyncio
async def test_only_a_flagged_recall_with_the_static_switch_on_reinforces(memory, request_context, monkeypatch):
    bank_id = await _bank(memory, request_context)
    fact = await _fact(memory, bank_id, "Jo maintains the deploy scripts.", at=day(0))

    result = await _recall(memory, bank_id, request_context, _external=True)
    assert [m.id for m in result.results] == [fact]
    assert await _bank_ledger(memory, bank_id) == [], "the static switch is off by default"

    monkeypatch.setattr(_get_raw_config(), "adaptive_memory_recall_reinforcement_enabled", True)
    await _recall(memory, bank_id, request_context)
    assert await _bank_ledger(memory, bank_id) == [], "an unflagged (internal) recall"

    await _recall(memory, bank_id, request_context, _external=True, question_date=day(100))
    (event,) = await _ledger(memory, fact)
    assert (event["source_type"], event["event_time"], event["outcome"]) == (
        "final_external_recall",
        day(100),
        "applied",
    )
    row = await _retention(memory, fact)
    assert (row["access_count"], row["last_recalled_at"], row["last_reinforced_at"]) == (1, day(100), day(100))
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_recall_reinforces_exactly_the_results_it_returned(memory, request_context, recall_switch_on):
    bank_id = await _bank(memory, request_context)
    facts = {
        await _fact(memory, bank_id, f"Jo maintains the deploy scripts of the {name} service.", at=day(0))
        for name in ("billing", "search", "ledger", "gateway", "mailer")
    }
    everyone = await _recall(memory, bank_id, request_context)
    assert {m.id for m in everyone.results} == facts, "every fact is a candidate for the query"

    # The token budget trims the final results below the candidate pool.
    trimmed = await _recall(memory, bank_id, request_context, max_tokens=24, _external=True)
    returned = {m.id for m in trimmed.results}
    assert returned and returned < facts
    assert {str(r["memory_id"]) for r in await _bank_ledger(memory, bank_id)} == returned
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_future_query_timestamp_is_capped_at_the_wall_clock(memory, request_context, recall_switch_on):
    bank_id = await _bank(memory, request_context)
    fact = await _fact(memory, bank_id, "Jo maintains the deploy scripts.", at=day(0))
    wall = datetime(2026, 6, 1, tzinfo=UTC)
    with patch("hindsight_api.engine.adaptive.clock.utcnow", return_value=wall):
        await _recall(memory, bank_id, request_context, _external=True, question_date=wall + timedelta(days=90))
    (event,) = await _ledger(memory, fact)
    assert event["event_time"] == wall
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_the_http_recall_endpoint_is_a_final_external_recall(memory, request_context, recall_switch_on):
    bank_id = await _bank(memory, request_context)
    fact = await _fact(memory, bank_id, "Kim maintains the deploy scripts.", at=day(0))
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/v1/default/banks/{bank_id}/memories/recall",
            json={"query": "Who maintains the deploy scripts?", "types": ["world"], "budget": "low"},
        )
    assert response.status_code == 200, response.text
    await memory._adaptive_reinforcement.drain()
    assert [r["source_type"] for r in await _ledger(memory, fact)] == ["final_external_recall"]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_both_mcp_recall_tools_flag_their_recall(mock_memory):
    from tests.test_mcp_tools import _make_mcp_server, _tools

    for include_bank_id in (True, False):
        mock_memory.recall_async.reset_mock()
        mcp = _make_mcp_server(mock_memory, {"recall"}, include_bank_id=include_bank_id)
        await _tools(mcp)["recall"].fn(query="q")
        assert mock_memory.recall_async.call_args.kwargs["_external"] is True


# --------------------------------------------------------------------------- retain


async def _document(memory, bank_id: str, document_id: str):
    (row,) = await _rows(
        memory, "SELECT retain_params FROM documents WHERE bank_id = $1 AND id = $2", bank_id, document_id
    )
    params = row["retain_params"]
    return json.loads(params) if isinstance(params, str) else (params or {})


async def _document_ids(memory, bank_id: str) -> list[str]:
    return [r["id"] for r in await _rows(memory, "SELECT id FROM documents WHERE bank_id = $1", bank_id)]


@pytest.mark.asyncio
async def test_a_minted_document_id_is_marked_and_the_mark_survives_an_append_and_a_reprocess(
    memory, request_context, observations_off
):
    bank_id = await _bank(memory, request_context)
    await memory.retain_async(
        bank_id=bank_id, content="Lena joined the security team.", request_context=request_context
    )
    (minted,) = await _document_ids(memory, bank_id)
    assert (await _document(memory, bank_id, minted)).get("document_id_minted") is True

    # The caller now names the server's id itself: still the server's id, so still weak evidence.
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Lena now leads it.", "document_id": minted, "update_mode": "append"}],
        request_context=request_context,
    )
    assert (await _document(memory, bank_id, minted)).get("document_id_minted") is True

    # A reprocess rebuilds its retain from retain_params, so it replays the mark.
    await memory.reprocess_document(bank_id, minted, request_context=request_context)
    assert (await _document(memory, bank_id, minted)).get("document_id_minted") is True

    await memory.retain_async(
        bank_id=bank_id, content="Mo joined too.", document_id="client-named", request_context=request_context
    )
    assert "document_id_minted" not in await _document(memory, bank_id, "client-named")
    key = evidence_key(memory_id="x", document_id=minted, document_id_minted=True, metadata=None)
    assert key.independence_weight == WEAK_INDEPENDENCE
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_an_off_bank_retains_byte_identical_documents(memory, request_context, observations_off):
    bank_id = await _bank(memory, request_context, mode=None)
    await memory.retain_async(bank_id=bank_id, content="Nia joined the data team.", request_context=request_context)
    (doc,) = await _document_ids(memory, bank_id)
    assert "document_id_minted" not in await _document(memory, bank_id, doc)
    await memory.delete_bank(bank_id, request_context=request_context)


async def _http_retain_with_attachment(memory, bank_id: str, text: str, metadata: dict) -> None:
    """One HTTP retain of an item with an image and no document_id: the route mints ``retain_<uuid>``."""
    from tests.test_retain_attachment_ingress import _image_block, _text_block

    app = create_app(memory, initialize_memory=False)
    # Consolidation is driven by the test, never by the retain.
    with patch.object(memory, "submit_async_consolidation"):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/v1/default/banks/{bank_id}/memories",
                json={
                    "items": [{"content": [_text_block(text), _image_block()], "metadata": metadata}],
                    "async": False,
                },
            )
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_attachment_retains_of_one_conversation_are_one_piece_of_evidence(memory, request_context):
    bank_id = await _bank(memory, request_context)
    text = "Omar keeps the incident runbook up to date."
    for _ in range(2):
        await _http_retain_with_attachment(memory, bank_id, text, {"conversation_id": "c1"})
    documents = await _document_ids(memory, bank_id)
    assert len(documents) == 2 and all(d.startswith("retain_") for d in documents)
    for document_id in documents:
        assert (await _document(memory, bank_id, document_id)).get("document_id_minted") is True

    await _consolidate(memory, bank_id, request_context, _create(text))

    # Not two full-weight `document:retain_…` keys: the server's ids rank below the conversation.
    obs = await _observation_id(memory, bank_id, request_context)
    assert [(r["source_key"], r["independence_weight"]) for r in await _ledger(memory, obs)] == [
        ("conversation:c1", 1.0)
    ]
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_an_off_bank_attachment_retain_records_no_minted_mark(memory, request_context):
    bank_id = await _bank(memory, request_context, mode=None)
    await _http_retain_with_attachment(memory, bank_id, "Pia owns the on-call rota.", {"conversation_id": "c1"})
    (document_id,) = await _document_ids(memory, bank_id)
    assert document_id.startswith("retain_")
    assert "document_id_minted" not in await _document(memory, bank_id, document_id)
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["rank", None], ids=["rank", "off"])
async def test_a_file_retain_marks_the_document_id_it_mints(memory, request_context, observations_off, mode):
    bank_id = await _bank(memory, request_context, mode=mode)
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/v1/default/banks/{bank_id}/files/retain",
            files=[
                ("files", ("anonymous.txt", b"Quinn runs the quarterly audit.", "text/plain")),
                ("files", ("named.txt", b"Rae runs the yearly audit.", "text/plain")),
            ],
            data={"request": json.dumps({"async": True, "files_metadata": [{}, {"document_id": "client-file"}]})},
        )
    assert response.status_code == 200, response.text

    documents = set(await _document_ids(memory, bank_id))
    (minted,) = documents - {"client-file"}
    assert minted.startswith("file_") and "client-file" in documents
    assert (await _document(memory, bank_id, minted)).get("document_id_minted") is (True if mode else None)
    assert "document_id_minted" not in await _document(memory, bank_id, "client-file")
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_time_importance_and_exemption_materialize_retention(memory, request_context, observations_off):
    bank_id = await _bank(memory, request_context)
    (important,) = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {
                "content": "Omar holds the root credentials escrow.",
                "document_id": "escrow",
                "importance": 0.9,
                "forgetting_exempt": True,
            }
        ],
        request_context=request_context,
    )
    (plain,) = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Omar had a sandwich for lunch.", "document_id": "lunch"}],
        request_context=request_context,
    )
    assert important and plain
    for memory_id in important:
        row = await _retention(memory, memory_id)
        fact_type = (await _unit(memory, bank_id, memory_id, request_context))["type"]
        assert (row["importance"], row["forgetting_exempt"]) == (pytest.approx(0.9), True)
        assert row["stability_days"] == pytest.approx(initial_stability(fact_type, 0.9, _settings()))
    for memory_id in plain:
        assert await _retention(memory, memory_id) is None
    # Recorded like the item's other fields, so a reprocess re-applies it.
    assert (await _document(memory, bank_id, "escrow"))["importance"] == 0.9
    assert "importance" not in await _document(memory, bank_id, "lunch")
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_time_policy_is_ignored_outside_rank_and_lifecycle(memory, request_context, observations_off):
    bank_id = await _bank(memory, request_context, mode="observe")
    (ids,) = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Pia holds the backup keys.", "importance": 1.0, "forgetting_exempt": True}],
        request_context=request_context,
    )
    assert ids
    for memory_id in ids:
        assert await _retention(memory, memory_id) is None
    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item", [{"importance": 1.5}, {"importance": "high"}, {"importance": math.nan}, {"forgetting_exempt": "yes"}]
)
async def test_retain_rejects_a_malformed_policy_before_extracting(memory, request_context, observations_off, item):
    bank_id = await _bank(memory, request_context)
    with pytest.raises(ValueError):
        await memory.retain_batch_async(
            bank_id=bank_id, contents=[{"content": "Quinn rotates keys.", **item}], request_context=request_context
        )
    assert (await memory.list_memory_units(bank_id, request_context=request_context))["total"] == 0
    await memory.delete_bank(bank_id, request_context=request_context)


def test_a_dropped_reinforcement_exports_under_its_exact_prometheus_name():
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from prometheus_client import CollectorRegistry, generate_latest

    from hindsight_api.metrics import MetricsCollector

    registry = CollectorRegistry()
    provider = MeterProvider(metric_readers=[PrometheusMetricReader(registry=registry)])
    config = SimpleNamespace(
        metrics_include_bank_id=False,
        metrics_include_tenant=False,
        recall_diagnostic_phases=True,
        recall_phase_sample_every=1,
    )
    try:
        with (
            patch("hindsight_api.metrics.get_meter", return_value=provider.get_meter("test-adaptive-producers")),
            patch("hindsight_api.config.get_config", return_value=config),
        ):
            collector = MetricsCollector()
        collector.record_adaptive_reinforcement_dropped(producer="recall", reason="overflow", bank_id="b")
        collector.record_adaptive_reinforcement_dropped(producer="consolidation", reason="error", bank_id="b")
        text = generate_latest(registry).decode()
    finally:
        provider.shutdown()

    samples = [
        line for line in text.splitlines() if line.startswith("hindsight_adaptive_memory_reinforcement_dropped_total{")
    ]
    assert len(samples) == 2
    assert any('producer="recall"' in s and 'reason="overflow"' in s and s.endswith(" 1.0") for s in samples)
    assert any('producer="consolidation"' in s and 'reason="error"' in s for s in samples)
    assert all("bank_id" not in s for s in samples), "bank_id is opt-in"
