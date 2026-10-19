"""Consolidation engine for automatic observation creation from memories.

The consolidation engine runs as a background job after retain operations complete.
It processes new memories and either:
- Creates new observations from novel facts
- Updates existing observations when new evidence supports/contradicts/refines them

Observations are stored in memory_units with fact_type='observation' and include:
- proof_count: Number of supporting memories
- source_memory_ids: Array of memory UUIDs that contribute to this observation
- history: JSONB tracking changes over time

NOTE: Observations are distinct from mental models (pinned reflections).
- Observations: auto-generated bottom-up by this engine from raw facts (memory_units table, fact_type='observation')
- Mental models: user-defined queries stored in the mental_models table, refreshed on demand via reflect
"""

import asyncio
import copy
import json
import logging
import math
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from fnmatch import fnmatchcase
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, cast

import asyncpg
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode

from ...config import get_config
from ...metrics import get_metrics_collector
from ...worker.stage import set_stage
from ..adaptive import clock as adaptive_clock
from ..adaptive.claims import RETIRED_STATUSES
from ..adaptive.models import AdaptiveMode, ClaimState, TruthStatus
from ..adaptive.producers import (
    FactEvidence,
    ObservationEvidence,
    fact_evidence,
    mode_settings,
    reinforcement_allowed,
    write_settings,
)
from ..adaptive.settings import lifecycle_scope_keys
from ..db import DatabaseBackend
from ..db_utils import acquire_with_retry
from ..fact_budget import select_facts_within_budget
from ..llm_interface import OutputTooLongError, PromptCachePrefix, ProviderRateLimitResetError
from ..llm_trace import (
    record_created_memory_ids,
    record_source_memory_ids,
    reset_trace_context,
    set_trace_context,
    trace_context_of,
)
from ..llm_wrapper import sanitize_llm_output
from ..memories import StoredMemory, get_memories
from ..memory_engine import Budget, count_tokens, fq_table
from ..prompt_utils import truncate_context_for_prompt
from ..retain import embedding_utils
from ..search.lifecycle_filter import ALL_WITH_STATUS
from ..source_facts import select_source_facts_within_budget
from . import lifecycle as claim_lifecycle
from .prompts import (
    build_consolidation_input,
    build_consolidation_system_prompt,
)
from .triage import MAX_CANDIDATES, TriageResult, triage_facts, triage_model

if TYPE_CHECKING:
    # The engine's connection abstraction, which is what every caller passes; the annotations
    # below named asyncpg's concrete type, which predates it.
    from ...api.http import RequestContext
    from ..adaptive.settings import AdaptiveSettings
    from ..db.base import DatabaseConnection
    from ..memories.base import StoredMemory
    from ..memory_engine import MemoryEngine
    from ..response_models import ConsolidationStrategiesPreview, MemoryFact, RecallResult

logger = logging.getLogger(__name__)


#: How many of one response's actions are prepared at once (see `_process_memory_batch`). Bounds the
#: embedder calls and, with semantic dedup on, the nearest-observation probes (each takes a pooled
#: connection) and their LLM verdicts one response puts in flight. The liveness preflight is one read
#: for the whole response, done before any action is prepared.
_PREPARE_CONCURRENCY = 8


async def _gather_or_cancel(coros: list[Any]) -> list[Any]:
    """``asyncio.gather`` that leaves no task running behind it.

    Plain ``asyncio.gather`` re-raises the first exception immediately but does
    NOT cancel its siblings — they keep running detached. In consolidation that
    is actively harmful: the failure propagates out of ``run_consolidation_job``
    to the worker, which marks the operation failed and re-queues it with a 5s
    base backoff, while the orphaned tag groups are still calling the LLM,
    stamping ``mark_consolidated`` and committing write-groups. The per-scope
    ``scope_locks`` are local to one dispatch, so nothing serialises an orphan
    against the retry, and the "batches within a group run serially" invariant
    that keeps two consolidators out of the same observation scope is broken
    exactly when it matters.

    So: cancel the outstanding tasks and await them before propagating. A
    cancelled batch's writes stay invisible (its witness row is never
    committed) and are resolved by the recovery sweep, which is the same state
    a crash would leave.

    Deliberately not ``asyncio.TaskGroup``: it wraps failures in an
    ``ExceptionGroup``, and the worker's ``_is_non_retryable_task_error`` does
    ``isinstance`` checks on the raised exception — a wrapped
    ``IntegrityConstraintViolationError`` would be misclassified as retryable
    and retried forever. This helper re-raises the original exception unchanged.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        # Await the cancellations before propagating: returning while they are
        # still unwinding would reintroduce the very overlap this prevents.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _norm_obs_text(text: str) -> str:
    """Whitespace-normalised observation text for exact-duplicate matching.

    Collapses runs of whitespace only; case is preserved. The reconciliation guard
    drops a CREATE on the premise that an exact-text match loses no information — but
    case-folding would also drop a create differing only in case (e.g. "TLS" vs "tls"),
    which *does* lose information, so we match case-sensitively.
    """
    return " ".join((text or "").split()).strip()


def _duplicate_create_target(
    create_text: str,
    shown_obs_by_text: "dict[str, MemoryFact]",
    update_texts: set[str],
) -> str | None:
    """Return a human label for what ``create_text`` duplicates, or None if novel.

    A CREATE is a duplicate when its normalised text matches an observation that was
    already shown to the LLM, or the text of an UPDATE issued in the same response
    (the model occasionally UPDATEs the twin to text X and also CREATEs X). Exact-text
    match means no information is lost by dropping the CREATE.
    """
    norm = _norm_obs_text(create_text)
    matched = shown_obs_by_text.get(norm)
    if matched is not None:
        return f"shown observation {str(matched.id)[:8]}"
    if norm in update_texts:
        return "an UPDATE in this response"
    return None


# Top-K existing observations probed (by the new observation's own embedding) when
# semantic dedup is enabled. Small: we only need the nearest few candidates.
_DEDUP_TOP_K = 5


class _DedupDecision(BaseModel):
    """Focused 1-by-1 verdict for whether a new observation duplicates an existing one."""

    action: Literal["merge", "keep"] = "keep"
    text: str = ""  # the synthesized merged observation (when action == "merge")
    reason: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"merge", "keep"}:
                return normalized

        logger.warning("Invalid consolidation dedup action %r; defaulting to keep", value)
        return "keep"


def _dedup_decision_from_response(raw: Any) -> _DedupDecision:
    try:
        if isinstance(raw, _DedupDecision):
            return raw
        if isinstance(raw, str):
            return _DedupDecision.model_validate_json(raw)
        return _DedupDecision.model_validate(raw)
    except ValueError as exc:
        logger.warning("Invalid consolidation dedup response %r; defaulting to keep: %s", raw, exc)
        return _DedupDecision(action="keep", reason="invalid structured response")


_DEDUP_PROMPT = """You reconcile long-term memory observations. A NEW observation is about to be \
stored, and it is highly similar to an EXISTING one:

[NEW] {new}
[EXISTING] {existing}

Respond with ONLY one valid JSON object matching one of these shapes:

For duplicate facts:
{{"action": "merge", "text": "...", "reason": "..."}}

For distinct facts:
{{"action": "keep", "text": "", "reason": "..."}}

Do NOT use key=value lines, markdown fences, or any text outside the JSON object.

If they assert the SAME fact (wording aside), set "action" to "merge" and provide "text": a \
single observation that preserves EVERY detail from both. If they differ in ANY important detail \
— a number/quantity, a named entity or language, a negation, or a condition — set "action" to \
"keep" and "text" to an empty string."""


def _dedup_active(config: Any) -> bool:
    """Whether create/update semantic dedup runs for this consolidation.

    Enabled when the resolved threshold is < 1.0, EXCEPT on Oracle: the merge path uses
    Postgres-only SQL (``unnest``/``array_agg``, ``UPDATE ... FROM``), so on Oracle dedup is
    skipped — it behaves exactly as it did before this feature, regardless of the configured
    threshold. This is why the feature can ship enabled-by-default without breaking Oracle.
    """
    if config is None or config.consolidation_dedup_threshold >= 1.0:
        return False
    return get_config().database_backend != "oracle"


@dataclass(frozen=True)
class _TemporalBounds:
    """The temporal columns an observation inherits from the facts behind it.

    Merging two observations (or an observation and a fresh set of source facts) must widen
    these, never replace them: ``event_date``/``occurred_start`` keep the earliest known value
    and ``occurred_end``/``mentioned_at`` the latest, with a missing value on either side
    ignored. That is exactly the ``_aggregate_source_fields`` rule, and the Python mirror of the
    ``LEAST``/``GREATEST`` the SQL paths apply.

    The SQL spelling differs by reach, deliberately. The dedup folds only ever run on PostgreSQL
    (``_dedup_active`` disables dedup on Oracle) and use the plain
    ``LEAST(col, COALESCE(x, col))``, which is enough there because PostgreSQL ignores NULL
    arguments. ``_apply_update_action`` also runs on Oracle, where LEAST/GREATEST return NULL
    if any argument is NULL, so it wraps the whole expression in one more COALESCE — see the
    comment there.
    """

    event_date: "datetime | None" = None
    occurred_start: "datetime | None" = None
    occurred_end: "datetime | None" = None
    mentioned_at: "datetime | None" = None

    @classmethod
    def of(cls, row: "StoredMemory | _SourceAggregation") -> "_TemporalBounds":
        """The bounds carried by a stored memory or by an aggregation over source facts.

        Deliberately not a recall ``MemoryFact``: that model has no ``event_date`` at all and
        keeps the rest as ISO strings, so it has to be read field by field where it is used.
        """
        return cls(
            event_date=row.event_date,
            occurred_start=row.occurred_start,
            occurred_end=row.occurred_end,
            mentioned_at=row.mentioned_at,
        )

    def merged_with(self, other: "_TemporalBounds") -> "_TemporalBounds":
        return _TemporalBounds(
            event_date=_merge_min(self.event_date, other.event_date),
            occurred_start=_merge_min(self.occurred_start, other.occurred_start),
            occurred_end=_merge_max(self.occurred_end, other.occurred_end),
            mentioned_at=_merge_max(self.mentioned_at, other.mentioned_at),
        )


@dataclass
class _DedupOutcome:
    """Result of probing one observation against its in-scope neighbours.

    ``best_id`` is the nearest observation at/above the threshold (None if none),
    ``merged_text`` is the LLM-synthesized union text (set only when ``should_merge``).
    """

    best_id: str | None
    merged_text: str
    should_merge: bool
    # The twin's text at probe time. Guards the fold against a concurrent survivor
    # rewrite during the connection-free LLM window (set on the two non-None returns).
    best_text: str = ""


async def _dedup_adjudicate(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    dedup_llm_config: Any,
    anchor_text: str,
    anchor_emb_str: str | None,
    tags: list[str] | None,
    exclude_id: str | None,
    exclude_non_active: bool = False,
) -> _DedupOutcome:
    """Probe one observation's embedding against in-scope observations and adjudicate a merge.

    Anchored on the observation text — the correct obs<->obs comparison, unlike consolidation
    recall which is anchored on the raw fact. Returns the nearest observation at/above
    ``consolidation_dedup_threshold`` and, when found, the LLM's focused 1-by-1 merge-or-keep
    verdict (scope ``consolidation_dedup``): the LLM reads both texts, so a word-level difference
    (number / negation / entity) is respected. ``exclude_id`` skips the anchor observation itself
    (used by the UPDATE path, where the anchor row already exists and would self-match at 1.0).
    ``anchor_emb_str`` reuses an already-computed embedding (the UPDATE path just embedded it);
    pass None to embed ``anchor_text`` here (the CREATE path).

    The embedder and the LLM both run with NO connection held; only the semantic+BM25 probe
    briefly borrows a short-lived connection.

    ``exclude_non_active`` (lifecycle mode) leaves superseded and refuted observations out of the
    probe inside the store's SQL, before its limit, so retired near-twins can never crowd the live
    twin out of the top-K nor be folded into.
    """
    from ..memories import get_memories

    threshold = config.consolidation_dedup_threshold
    if anchor_emb_str is None:
        embs = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [anchor_text])
        if not embs:
            return _DedupOutcome(best_id=None, merged_text="", should_merge=False)
        anchor_emb_str = str(embs[0])
    tags_match = "all_strict" if tags else "any"
    # Dedup only needs the dense/keyword arms over observations — no graph, no temporal window.
    grouped = await get_memories().recall_unified(
        conn=pool,
        bank_id=bank_id,
        fact_types=["observation"],
        query_embedding=anchor_emb_str,
        query_text=anchor_text,
        limit=_DEDUP_TOP_K,
        tags=tags,
        tags_match=tags_match,
        enable_graph=False,
        temporal_window=None,
        **({"exclude_non_active": True} if exclude_non_active else {}),
    )
    results = grouped["observation"].semantic
    best_id: str | None = None
    best_text = ""
    best_sim = threshold  # only candidates at/above the threshold are considered
    for r in results:
        rid = str(r.id)
        if exclude_id is not None and rid == exclude_id:
            continue  # never match the anchor observation against itself
        sim = r.similarity or 0.0
        if sim >= best_sim:
            best_id, best_text, best_sim = rid, r.text, sim

    if best_id is None:
        return _DedupOutcome(best_id=None, merged_text="", should_merge=False)

    dedup_call = await dedup_llm_config.call(
        messages=[{"role": "user", "content": _DEDUP_PROMPT.format(new=anchor_text, existing=best_text)}],
        response_format=_DedupDecision,
        temperature=config.llm_temperature_consolidation,
        scope="consolidation_dedup",
        strict_schema=get_config().llm_strict_schema_consolidation,
    )
    decision = _dedup_decision_from_response(dedup_call.content)
    if decision.action != "merge":
        return _DedupOutcome(best_id=best_id, merged_text="", should_merge=False, best_text=best_text)
    merged_text = (sanitize_llm_output(decision.text) or "").strip() or best_text
    return _DedupOutcome(best_id=best_id, merged_text=merged_text, should_merge=True, best_text=best_text)


async def _apply_dedup_create_fold(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    outcome: _DedupOutcome,
    create_source_ids: list[uuid.UUID],
    source_bounds: _TemporalBounds,
) -> str | None:
    """Fold a CREATE the adjudicator called a duplicate into its existing twin.

    Runs on the caller's connection, inside the caller's transaction: this is one of the
    writes derived from a single consolidation LLM response, and all of them commit or roll
    back together (#3876). The slow half — the embed, the semantic probe and the
    merge-or-keep adjudication that produced ``outcome`` — already ran connection-free in
    the prepare phase (:func:`_dedup_adjudicate`).

    Returns the twin's id (the caller then skips the CREATE), or None when the fold did not
    happen and the observation must be inserted after all.

    ``source_bounds`` are the dates the skipped CREATE would have been stamped with. They are
    folded into the twin too: this path bypasses the CREATE writer, so without them the twin
    would cite dated source facts while reporting the dates of its original sources only (#3477).
    """
    if not outcome.should_merge or outcome.best_id is None:
        return None

    # Re-check liveness inside the write transaction; CREATE performed the slow embed/LLM
    # work off-connection, so sources may have been deleted since the decision was made.
    live_source_ids = await _filter_live_source_memories(conn, bank_id, create_source_ids)
    if not live_source_ids:
        return None
    # Fold the new source facts into the twin and persist the merged text, gated on the twin's
    # probe-time text so a concurrent survivor rewrite during the connection-free LLM window
    # can't be clobbered.
    folded = await get_memories().fold_sources_into_observation(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        observation_id=outcome.best_id,
        expected_text=outcome.best_text,
        merged_text=outcome.merged_text,
        source_memory_ids=live_source_ids,
        bounds=source_bounds,
        embeddings=memory_engine.embeddings,
    )
    if not folded:
        # The twin vanished (or was rewritten) during the connection-free LLM window.
        # Don't skip the CREATE: returning None lets the caller insert the observation
        # so nothing is lost.
        logger.debug(
            "[CONSOLIDATION] dedup-merge target %s vanished before fold; proceeding with CREATE",
            outcome.best_id[:8],
        )
        return None
    return outcome.best_id


async def _apply_dedup_update_fold(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    config: Any,
    outcome: _DedupOutcome,
    updated_id: str,
    updated_text: str,
    claim_state: "claim_lifecycle.ApplyState | None" = None,
) -> bool:
    """Fold a just-rewritten observation into the twin the adjudicator matched it to.

    An UPDATE rewrites an observation's text and re-embeds it, so its vector can drift to
    within threshold of a DIFFERENT existing observation. The create-time guard never sees
    this (it only runs on CREATE), so without this the two persist as a near-duplicate pair —
    the measured residual-duplicate source. On "merge", fold the just-updated observation's
    sources into the twin, persist the merged text, and DELETE the updated row. Unlike the
    CREATE path the row already exists, so reconciliation is a fold-and-delete, not a skip.

    Like :func:`_apply_dedup_create_fold` this runs on the caller's connection inside the
    caller's transaction (#3876); ``outcome`` comes from the connection-free prepare phase.
    Returns True when the updated row was folded away.

    ``claim_state`` (lifecycle mode): fold only into a twin that is still live under the batch's
    claim lock, and retire the updated row as merged into it instead of deleting it.
    """
    if not outcome.should_merge or outcome.best_id is None:
        return False
    if claim_state is not None and not await _merge_precheck(conn, claim_state, updated_id, outcome.best_id):
        return False

    # Lock order is sources-before-observation, the same as the create fold and the normal
    # write paths (_apply_create_observation / _apply_update_action); the store keeps it.
    folded = await get_memories().fold_observation_into_twin(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        observation_id=updated_id,
        observation_text=updated_text,
        twin_id=outcome.best_id,
        twin_text=outcome.best_text,
        merged_text=outcome.merged_text,
        embeddings=memory_engine.embeddings,
    )
    if not folded:
        # Nothing live to fold, or the twin / updated row vanished during the LLM window —
        # keep the updated row as a distinct observation instead of deleting it unfolded.
        return False
    if claim_state is not None:
        await _merge_retire(conn, claim_state, updated_id, outcome.best_id)
        return True
    await _execute_delete_action(conn, bank_id, updated_id)
    logger.info(
        "[CONSOLIDATION] dedup-merged updated observation %s into %s (cosine>=%.2f)",
        updated_id[:8],
        outcome.best_id[:8],
        config.consolidation_dedup_threshold,
    )
    return True


@dataclass
class _BatchDeltas:
    """Per-LLM-batch deltas, merged into the job's running stats after dispatch.

    Returned by value rather than mutated into the outer ``stats`` /
    ``consolidated_tags`` so parallel batches cannot race on those shared
    structures (the merge happens once, serially, after dispatch completes).
    """

    stats: dict[str, int]
    tags: set[str]
    cancelled: bool


@dataclass
class _LifecycleRun:
    """Adaptive memory's claim lifecycle across one consolidation job (lifecycle or observe mode).

    ``counts`` tallies what the job's batches did (transitions whose CAS won, validation drops by
    reason, history attaches, shadow records); it is reported under ``stats["lifecycle"]``.
    Parallel batches update it between awaits only, so the single event loop keeps it consistent.
    """

    settings: "AdaptiveSettings"
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def lifecycle(self) -> bool:
        return self.settings.effective_mode.lifecycle

    @property
    def observe(self) -> bool:
        return self.settings.effective_mode is AdaptiveMode.OBSERVE


@dataclass
class _BatchLifecycle:
    """One LLM batch's view of the job's lifecycle run, plus what it hands back to the job:
    the tags of the observations it retired (mental models scoped to them must refresh)."""

    run: _LifecycleRun
    retired_tags: set[str] = field(default_factory=set)


def _parse_observation_scopes(memory: dict[str, Any]) -> Any:
    """Parse the per-memory ``observation_scopes`` value.

    The value arrives already decoded when read through the memories store (its
    reader coerces the JSONB column) or as raw JSON text from a driver without a
    JSONB codec. A scalar mode such as ``"per_tag"`` decodes to a bare string that
    is not itself valid JSON, so a blind ``json.loads`` would raise on it — try to
    parse, but treat an unparseable string as an already-decoded scalar.
    """
    raw = memory.get("observation_scopes")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _resolve_obs_tags_list(memory: dict[str, Any]) -> list[list[str]] | None:
    """Resolve a memory's ``observation_scopes`` spec into concrete scope tags.

    Returns ``None`` for the default ``combined``-mode single pass (caller uses
    the memory's own tags). Returns a list[list[str]] when the memory requested
    multi-pass scoping (``per_tag``, ``all_combinations``, ``shared``, or an
    explicit list).

    ``shared`` resolves to ``[[]]`` — a single pass over the empty (untagged)
    scope. The created observation carries no tags and recall/dedup match it with
    ``tags_match="any"``, so every memory consolidates into one shared observation
    regardless of its own tags. Use it to deduplicate across volatile per-call
    provenance tags (e.g. per-session ids) without dropping those tags from the
    source facts.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [[t] for t in tags] if tags else None
    if parsed == "all_combinations":
        if not tags:
            return None
        return [list(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [[]]
    if parsed == "combined" or parsed is None:
        return None
    return parsed  # explicit list[list[str]]


def _resolve_write_scopes(memory: dict[str, Any]) -> list[frozenset[str]]:
    """Return the observation scopes a memory will write to, as frozensets.

    Used by the parallel dispatcher to acquire one lock per scope before
    processing a tag group, so that two groups whose write-scope sets overlap
    serialise on the overlapping scopes rather than racing on the same
    observation row. The mapping mirrors ``_resolve_obs_tags_list`` exactly:

    - ``combined`` / ``None``    -> ``[frozenset(memory.tags)]``
    - ``per_tag``                -> ``[frozenset({t}) for t in memory.tags]``
    - ``all_combinations``       -> one frozenset per nonempty subset of tags
    - ``shared``                 -> ``[frozenset()]`` (the single untagged scope)
    - explicit ``list[list[str]]`` -> one frozenset per declared scope

    Empty-tag memories collapse to a single ``frozenset()`` in all modes so they
    still take exactly one lock and serialise against other untagged work.
    """
    parsed = _parse_observation_scopes(memory)
    tags = list(memory.get("tags") or [])

    if parsed == "per_tag":
        return [frozenset([t]) for t in tags] if tags else [frozenset()]
    if parsed == "all_combinations":
        if not tags:
            return [frozenset()]
        return [frozenset(c) for r in range(1, len(tags) + 1) for c in combinations(tags, r)]
    if parsed == "shared":
        return [frozenset()]
    if parsed == "combined" or parsed is None:
        return [frozenset(tags)]
    # Explicit list[list[str]]. An *empty* list resolves to no passes at all, and
    # the pass loop (``if obs_tags_list:``) then falls back to the combined
    # single pass over the memory's own tags — so report that scope here too,
    # or the group takes no lock for the scope it actually writes.
    return [frozenset(s) for s in parsed] or [frozenset(tags)]


def write_scopes_changed(
    old_tags: list[str] | None, old_scopes: Any, new_tags: list[str] | None, new_scopes: Any
) -> bool:
    """Whether relabelling a memory moves it to a different set of observation scopes.

    Its observations need rebuilding only then: under ``shared`` or an explicit scope list
    the tags take no part in where observations go, so a tag change alone moves nothing
    (#5593). Compares the exact scopes :func:`_resolve_write_scopes` resolves before and
    after, so this and the consolidator can never disagree — a mode change that resolves to
    the same scopes (``combined`` over ``[a]`` -> ``[["a"]]``) rebuilds nothing either.
    """
    return set(_resolve_write_scopes({"tags": old_tags, "observation_scopes": old_scopes})) != set(
        _resolve_write_scopes({"tags": new_tags, "observation_scopes": new_scopes})
    )


def _batch_scope_signature(memory: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """The exact set of observation scopes consolidating this memory will write.

    Derived from the pass loop rather than from the grouping key, so it can be
    used to *check* the key: a truthy ``_resolve_obs_tags_list`` yields one pass
    per resolved scope (each written with that ``obs_tags_override``), and a
    falsy one — ``None`` from ``combined``, or a degenerate empty explicit list —
    yields the single combined pass whose tags come from the memory's own tag set.

    Two memories may share an LLM call only if their signatures are equal, because
    the batch resolves its scope once from ``sub_batch[0]`` and applies it to all.
    """
    obs_tags_list = _resolve_obs_tags_list(memory)
    if not obs_tags_list:
        return (tuple(sorted(memory.get("tags") or [])),)
    return tuple(sorted(tuple(sorted(scope)) for scope in obs_tags_list))


def _consolidation_batch_key(memory: dict[str, Any]) -> tuple[str, ...]:
    """Return the key that decides which memories may share an LLM batch.

    The real security requirement is "memories targeting different observation
    scopes must never share an LLM call" — every branch below keys on the
    memory's *resolved* scope(s), never on raw tags, so two memories with the
    same native tags but different ``observation_scopes`` modes can never
    collide into the same group:

    - default ``combined`` (``resolved is None``), and the degenerate empty
      resolution (an explicit ``[]``, which the pass loop below treats as
      combined because ``if obs_tags_list:`` is falsy): the memory's target
      scope *is* its own tag set, so it keys on those tags.
    - a single alternate scope (``shared``, an explicit one-scope list, or
      ``per_tag`` with exactly one tag): keys on that resolved scope instead,
      so it batches with any other memory naming the identical scope
      regardless of native tags — that is the whole point of requesting it.
    - fan-out to *multiple* scopes (``per_tag`` with more than one tag,
      ``all_combinations``, or a multi-scope explicit list): writes several
      observations from one LLM call, so it keys on the full resolved
      scope-list, not native tags — two fan-out memories only share a batch
      when every one of their target scopes matches exactly. A distinct
      leading marker per branch keeps e.g. a ``combined`` memory tagged
      ``["a","b"]`` out of the same key as a ``per_tag`` memory tagged
      ``["a","b"]``, even though both would otherwise sort to ``("a","b")``.
    """
    resolved = _resolve_obs_tags_list(memory)
    if not resolved:
        # ``None`` (combined) and ``[]`` (an explicit empty scope list) both fall
        # through to the combined pass downstream — ``if obs_tags_list:`` is
        # falsy for each — so they must key the same way. Testing ``is None``
        # alone sent ``[]`` down the fan-out branch, where every such memory
        # keyed to the tag-free ``("fanout",)`` and pooled with memories of
        # unrelated tags; ``obs_tags_override`` then being ``None``, the whole
        # group took ``memories[0]``'s tags and leaked across scopes.
        return ("combined", *sorted(memory.get("tags") or []))
    if len(resolved) == 1:
        return ("scope", *sorted(resolved[0]))
    return ("fanout", *sorted("\x1f".join(sorted(scope)) for scope in resolved))


def _scope_sort_key(scope: frozenset[str]) -> tuple[str, ...]:
    """Total ordering on scope frozensets for deadlock-free lock acquisition.

    Every parallel group acquires its scope locks in this same order, so two
    groups that share any subset of scopes cannot acquire them in opposite
    orders and deadlock.
    """
    return tuple(sorted(scope))


async def _filter_live_source_memories(
    conn: "DatabaseConnection",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
) -> list[uuid.UUID]:
    """Return only the source memory ids that still exist in the bank.

    The SQL store takes a ``FOR SHARE`` lock on the surviving rows so a concurrent
    delete can't remove one between this check and the observation write that
    follows — without it, a source deleted in that window would leave an orphan
    observation until the next sweep, because the delete path's stale-observation
    sweep only catches observations that already exist when it runs. (Oracle has no
    ``FOR SHARE``; the SQL rewriter promotes it to ``FOR UPDATE`` — more
    conservative, still correct.) A store that keeps memories outside SQL has its
    own concurrency model, so it answers with an unlocked existence check.
    """
    if not source_memory_ids:
        return []
    live = await get_memories().lock_live_memory_ids(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=source_memory_ids
    )
    return [mid for mid in source_memory_ids if str(mid) in live]


async def _sources_changed_since_read(
    conn: "DatabaseConnection",
    bank_id: str,
    memories: list[dict[str, Any]],
) -> list[str]:
    """Ids of the batch's source facts edited since the batch read them (#4831).

    The LLM decided on the facts as they were read; a fact edited meanwhile (a retag,
    a curation) was already requeued by that edit, and its observations dropped. Writing
    this response would rebuild them from the stale copy — under the old tags — and the
    ``consolidated_at`` stamp would then undo the requeue. ``updated_at`` is the signal:
    every edit stamps it and consolidation's own bookkeeping never does (META_UPDATED_AT).

    Takes ``FOR SHARE`` on the rows, so an edit cannot land between this check and the
    writes in the same transaction. A deleted fact is not "changed" — the per-action
    liveness checks handle that. A store that keeps memories outside SQL reports no
    ``updated_at`` on its reads, so it is not checked.
    """
    read_at: dict[str, datetime] = {str(m["id"]): m["updated_at"] for m in memories if m.get("updated_at") is not None}
    if not read_at:
        return []
    return await get_memories().memories_changed_since(conn=conn, fq_table=fq_table, bank_id=bank_id, read_at=read_at)


async def _live_source_ids(pool: DatabaseBackend, bank_id: str, unit_ids: list[Any]) -> set[str]:
    """Which of ``unit_ids`` still exist, as strings — the preflight before embedding.

    One read for a whole set, so a response's actions are checked together rather than one read
    each. It only decides whether the (slow) embed is worth spending: the authoritative,
    FOR SHARE liveness check still runs inside the write transaction. Outside a transaction the
    share lock this reads with ends with the statement, so nothing is held.
    """
    if not unit_ids:
        return set()
    async with acquire_with_retry(pool) as conn:
        live = await get_memories().lock_live_memory_ids(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids
        )
    return {str(u) for u in live}


def _unique_source_ids(v: str | list[str]) -> list[str]:
    """Drop repeated ids, keeping order. A looping model can repeat one id thousands of
    times, and every copy would be stored and re-sent to the next prompt (#4799)."""
    if isinstance(v, str):
        return [v]
    return list(dict.fromkeys(v))


class _CreateAction(BaseModel):
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    # One-sentence justification from the LLM (why CREATE vs UPDATE). Diagnostic
    # only — surfaced in the consolidation trace to explain duplicate creates.
    reason: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        return _unique_source_ids(v)


class _UpdateAction(BaseModel):
    observation_id: str  # UUID of the existing observation to update
    # The observation's text as it stands, copied before ``text`` is written. Not used by the
    # code: field order is generation order, and a small model that copies the old text first
    # edits it, where one writing ``text`` cold tends to emit just the new fact and erase the
    # rest of the observation (eval ``noisy-service-catalogue-mission``). Optional, so a
    # response without it still validates.
    current_text: str = ""
    text: str
    source_fact_ids: list[str]  # memory UUIDs from the NEW FACTS list
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)

    @field_validator("text", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        return sanitize_llm_output(v) or ""

    @field_validator("source_fact_ids", mode="before")
    @classmethod
    def ensure_list(cls, v: str | list[str]) -> list[str]:
        return _unique_source_ids(v)


class _DeleteAction(BaseModel):
    """One DELETE from an LLM response.

    ``observation_id`` stays required — a delete naming no target has no defensible
    fallback, and guessing one would remove the wrong observation. But rejecting the
    entry rejects the ENTIRE ``_ConsolidationBatchResponse``, taking the batch's
    perfectly good creates and updates with it (#4152), so a near-miss is worth
    absorbing rather than paying a bisected re-run for: models that copy the
    observation's own field name emit ``id``, which is unambiguous here because a
    delete entry has exactly one identifier. ``populate_by_name`` keeps the
    canonical name working for in-process construction, and the generated JSON
    schema still advertises ``observation_id`` alone (pydantic emits the first
    of the ``AliasChoices``), so what a grammar-constrained provider is told to
    emit does not change — this only widens what a free-form one gets away with.
    """

    model_config = ConfigDict(populate_by_name=True)

    observation_id: str = Field(validation_alias=AliasChoices("observation_id", "id"))
    # The observation that now carries this one's content, when the delete merges it into
    # another: its evidence moves there. A delete without it retires a belief.
    merged_into: str | None = None
    reason: str = ""  # LLM's one-sentence justification (diagnostic only)


# Lifecycle-mode actions (adaptive memory). Every label is optional and lenient: a
# malformed entry is dropped by its ``mode="before"`` normalizer rather than rejecting the response
# (#4152), and whether a named id is one the model was shown is checked after miscopied ids are
# repaired (``_resolve_miscopied_ids``), in ``lifecycle.plan_lifecycle``. Only ``_build_response_model
# (lifecycle=True)`` advertises them; everything else reads them with ``getattr`` defaults, so base,
# triage and MockLLM instances keep working.
_SUPERSEDES_DESCRIPTION = "Ids of retirable existing observations this one replaces because their state changed."
_REFUTES_DESCRIPTION = "Ids of retirable existing observations this one shows were wrong."
_REFINES_DESCRIPTION = "Ids of existing observations this one adds detail to; they stay true."
_CONFLICTS_DESCRIPTION = "Ids of existing observations this one contradicts when neither clearly wins."
_SPLIT_FROM_DESCRIPTION = "Id of the observation two or more creates split into separate facets."
_KEEPS_DESCRIPTION = "Source fact ids of a replaced observation that stay true and carry over to the successor."
_RELATION_DESCRIPTION = (
    "supersedes: the observation's state changed; refutes: it was wrong; historical: an older fact "
    "restating a superseded or refuted observation. Null for an ordinary update."
)
_RETIRES_DESCRIPTION = (
    "Source fact ids of the target that are no longer true (empty: every one this response does not cite)."
)


class _LifecycleCreateAction(_CreateAction):
    supersedes: list[str] = Field(default_factory=list, description=_SUPERSEDES_DESCRIPTION)
    refutes: list[str] = Field(default_factory=list, description=_REFUTES_DESCRIPTION)
    refines: list[str] = Field(default_factory=list, description=_REFINES_DESCRIPTION)
    conflicts_with: list[str] = Field(default_factory=list, description=_CONFLICTS_DESCRIPTION)
    split_from: str | None = Field(default=None, description=_SPLIT_FROM_DESCRIPTION)
    keeps_source_ids: list[str] = Field(default_factory=list, description=_KEEPS_DESCRIPTION)

    @field_validator("supersedes", "refutes", "refines", "conflicts_with", "keeps_source_ids", mode="before")
    @classmethod
    def _lenient_ids(cls, v: Any) -> list[str]:
        return claim_lifecycle.clean_ids(v)

    @field_validator("split_from", mode="before")
    @classmethod
    def _lenient_id(cls, v: Any) -> str | None:
        return claim_lifecycle.clean_id(v)


class _LifecycleUpdateAction(_UpdateAction):
    relation: Literal["supersedes", "refutes", "historical"] | None = Field(
        default=None, description=_RELATION_DESCRIPTION
    )
    retires_source_ids: list[str] = Field(default_factory=list, description=_RETIRES_DESCRIPTION)
    keeps_source_ids: list[str] = Field(default_factory=list, description=_KEEPS_DESCRIPTION)

    @field_validator("relation", mode="before")
    @classmethod
    def _lenient_relation(cls, v: Any) -> str | None:
        return claim_lifecycle.clean_relation(v)

    @field_validator("retires_source_ids", "keeps_source_ids", mode="before")
    @classmethod
    def _lenient_ids(cls, v: Any) -> list[str]:
        return claim_lifecycle.clean_ids(v)


@dataclass
class _PreparedDelete:
    observation_id: str
    #: The delete is a merge: its evidence must move, not vanish.
    is_merge: bool
    #: Merge target, already checked against the recall pool. None for a plain delete, or for
    #: a merge whose target the model could not name (an observation it creates in the same
    #: response has no id yet): that evidence is re-queued instead.
    merged_into: str | None


@dataclass
class _MergedAway:
    gone: StoredMemory
    target_id: str | None


class _ConsolidationBatchResponse(BaseModel):
    creates: list[_CreateAction] = []
    updates: list[_UpdateAction] = []
    deletes: list[_DeleteAction] = []


@dataclass
class _PreparedUpdate:
    """One UPDATE from an LLM response, with every slow step already done.

    Consolidation applies all of a response's writes in a single transaction (#3876), and a
    transaction must never be held open across an embedder or LLM call. So each action is
    first *prepared* connection-free — the source facts resolved and security-checked, the
    new text embedded, the semantic-dedup verdict adjudicated — and the resulting value
    object carries everything the write needs.
    """

    update: _UpdateAction
    #: Pre-update snapshot of the observation, from the batch's recall.
    model: "MemoryFact"
    source_mems: list[dict[str, Any]]
    source_memory_ids: list[uuid.UUID]
    source_fact_tags: list[str]
    source_bounds: _TemporalBounds
    embedding_str: str | None
    #: Fold target for the rewritten observation, when create/update dedup is enabled and
    #: the re-embed drifted it into a near-twin.
    dedup: _DedupOutcome | None = None


@dataclass
class _PreparedCreate:
    """One CREATE from an LLM response, with every slow step already done.

    See :class:`_PreparedUpdate`. ``dedup`` here is a fold into an existing near-twin,
    which replaces the insert rather than following it.
    """

    text: str
    source_mems: list[dict[str, Any]]
    source_memory_ids: list[uuid.UUID]
    source_fact_tags: list[str]
    agg: "_SourceAggregation"
    embedding_str: str | None
    dedup: _DedupOutcome | None = None
    #: The inserted observation's id, set by the write (None until then, or when it was skipped).
    observation_id: str | None = None
    #: Lifecycle mode: the claim transitions this create carries (it is a successor, a split child,
    #: a refinement or a conflicting claim). None for an ordinary create.
    lifecycle_plan: "claim_lifecycle.SuccessorPlan | None" = None


@dataclass
class _BatchLLMResult:
    creates: list[_CreateAction] = field(default_factory=list)
    updates: list[_UpdateAction] = field(default_factory=list)
    deletes: list[_DeleteAction] = field(default_factory=list)
    obs_count: int = 0
    prompt_chars: int = 0
    failed: bool = False
    #: How many attempts inside this batch call raised. Non-zero even when a later
    #: attempt succeeded, so the run summary can report calls that were retried out
    #: of existence — `failed` alone hides them (#4151).
    failed_attempts: int = 0


@dataclass
class _SourceAggregation:
    """Fields inherited by an observation from its source memories."""

    event_date: datetime | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    tags: list[str]


def _aggregate_source_fields(source_mems: list[dict[str, Any]], tags: list[str] | None = None) -> _SourceAggregation:
    """Compute the observation fields inherited from a set of source memories.

    Temporal aggregation rules:
    - ``event_date``    — earliest across sources (min)
    - ``occurred_start`` — earliest across sources (min)
    - ``occurred_end``   — latest across sources (max)
    - ``mentioned_at``   — latest across sources (max)

    Fields remain ``None`` when no source memory carries that information, so
    observations are never stamped with an artificial timestamp.

    ``tags`` defaults to those of the first source memory when not explicitly
    provided (all memories in a consolidation batch share the same tag set).
    """
    effective_tags = tags if tags is not None else (source_mems[0].get("tags") or [] if source_mems else [])
    return _SourceAggregation(
        event_date=_min_date(m.get("event_date") for m in source_mems),
        occurred_start=_min_date(m.get("occurred_start") for m in source_mems),
        occurred_end=_max_date(m.get("occurred_end") for m in source_mems),
        mentioned_at=_max_date(m.get("mentioned_at") for m in source_mems),
        tags=effective_tags,
    )


async def _count_observations_for_scope(
    conn: "DatabaseConnection",
    bank_id: str,
    tags: list[str],
) -> int:
    """Count existing observations matching the given tag scope.

    Returns the count of observations whose tags contain all specified tags.
    Observations with no tags are not counted (the limit does not apply to them).
    """
    return await get_memories().count_observations_with_tags(conn=conn, fq_table=fq_table, bank_id=bank_id, tags=tags)


@dataclass(frozen=True)
class _ScopeLimitRule:
    """One ``observation_scope_limits`` rule: a scope pattern -> an observation cap.

    DEPRECATED — superseded by :class:`_ConsolidationStrategy`, which carries the
    mission too. Still honoured, but consulted only after the strategies.

    ``globs`` is a tuple of fnmatch tag-globs describing one consolidation scope.
    A concrete scope (the set of ``fact_tags`` for a consolidation pass) matches
    under *exact cover*: every tag is matched by some glob AND every glob matches
    some tag. So ``["shared"]`` matches the scope ``{shared}`` but not
    ``{run_1, shared}``, and ``["run_*", "shared"]`` matches ``{run_1, shared}``
    but not ``{shared}``.

    ``limit`` is the cap applied to matching scopes (-1 = unlimited, 0 = no new
    observations, >0 = hard cap), mirroring ``max_observations_per_scope``.
    """

    globs: tuple[str, ...]
    limit: int


def _parse_scope_limit_rules(raw: Any) -> list[_ScopeLimitRule]:
    """Parse the raw ``observation_scope_limits`` config into ordered rules.

    The config round-trips as JSON through env and the bank-config API, so this
    is defensive: malformed entries are skipped rather than raising, and list
    order is preserved (first match wins in :func:`_effective_scope_limit`).
    """
    if not isinstance(raw, list):
        return []
    rules: list[_ScopeLimitRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        scope = entry.get("scope")
        limit = entry.get("limit")
        if not isinstance(scope, list) or not scope:
            continue
        if not all(isinstance(g, str) and g for g in scope):
            continue
        # bool is an int subclass — reject True/False masquerading as a limit.
        if not isinstance(limit, int) or isinstance(limit, bool):
            continue
        rules.append(_ScopeLimitRule(globs=tuple(scope), limit=limit))
    return rules


def _scope_matches_globs(globs: tuple[str, ...], tags: list[str]) -> bool:
    """Exact-cover match between a scope pattern and a concrete tag set.

    True iff every tag is covered by at least one glob AND every glob covers at
    least one tag (no uncovered tags, no vacuous globs). Untagged scopes never
    match, so a scope limit never applies to untagged observations (consistent
    with the ``and fact_tags`` guard at the call site). Matching is
    case-sensitive (``fnmatchcase``) for deterministic cross-platform behaviour.
    """
    tagset = set(tags)
    if not tagset:
        return False
    if not all(any(fnmatchcase(t, g) for g in globs) for t in tagset):
        return False
    if not all(any(fnmatchcase(t, g) for t in tagset) for g in globs):
        return False
    return True


# Settings a consolidation strategy may override, in addition to the scopes it
# claims. Kept to what actually varies per audience: the brief, how many
# observations the scope may hold, and how much source evidence each consolidation
# call is shown. Everything else stays bank-wide.
_STRATEGY_INT_SETTINGS = (
    "max_observations_per_scope",
    "consolidation_source_facts_max_tokens",
    "consolidation_source_facts_max_tokens_per_observation",
)


def _scope_contains_globs(globs: tuple[str, ...], tags: list[str]) -> bool:
    """Containment match: every glob matches some tag; extra tags are allowed.

    ``("company:*", "team:*")`` matches ``{company:acme, team:exec}`` and
    ``{user:dana, team:exec, company:acme}``, but not ``{company:acme}``. The
    untagged scope never matches, as with :func:`_scope_matches_globs`.
    """
    return bool(tags) and all(any(fnmatchcase(t, g) for t in tags) for g in globs)


# How one scope pattern of a strategy matches a consolidation scope. Named after
# the `tags_match` values recall already uses, so the vocabulary is the same:
#   "all"   — the scope has all of the pattern's tags; other tags are allowed.
#   "exact" — the scope has exactly the pattern's tags and nothing else.
# "any" is deliberately absent: a strategy's patterns are already alternatives
# (any one matching claims the scope), so "any of these tags" is written as one
# pattern per tag.
_STRATEGY_TAGS_MATCH = {"all": _scope_contains_globs, "exact": _scope_matches_globs}
_DEFAULT_STRATEGY_TAGS_MATCH = "all"


@dataclass(frozen=True)
class _ScopePattern:
    """One alternative in a strategy's ``scopes``: tag-globs plus how they match.

    The mode is per pattern, not per strategy, so one strategy can mix them —
    "exactly ``company:*``" OR "``team:*``, other tags allowed". A first version
    had a single ``tags_match`` for the whole strategy, which forced a second
    strategy (with duplicated settings) to express that.

    ``"all"`` is the default because the common case is a scope retained with all
    of a memory's tags together (``observation_scopes`` default ``combined``) —
    ``{user:dana, team:exec, company:acme}`` — which a "company and team" pattern
    should claim. The deprecated ``observation_scope_limits`` stays exact-only.
    """

    tags: tuple[str, ...]
    tags_match: str = _DEFAULT_STRATEGY_TAGS_MATCH

    def matches(self, fact_tags: list[str]) -> bool:
        return _STRATEGY_TAGS_MATCH[self.tags_match](self.tags, fact_tags)


def _parse_scope_pattern(raw: Any) -> _ScopePattern | None:
    """``{"tags": [...], "tags_match": "all" | "exact"}`` -> pattern, or None.

    Malformed patterns are dropped (see :func:`_parse_consolidation_strategies`).
    An unknown mode falls back to the default rather than dropping the pattern —
    same "never take consolidation down" rule as the rest.
    """
    if not isinstance(raw, dict):
        return None
    tags = raw.get("tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(g, str) and g for g in tags):
        return None
    tags_match = raw.get("tags_match")
    if tags_match not in _STRATEGY_TAGS_MATCH:
        tags_match = _DEFAULT_STRATEGY_TAGS_MATCH
    return _ScopePattern(tags=tuple(tags), tags_match=tags_match)


@dataclass(frozen=True)
class _ConsolidationStrategy:
    """One ``consolidation_strategies`` entry: the scopes it claims -> settings.

    ``scopes`` are alternatives: a concrete scope (the ``fact_tags`` of one
    consolidation pass) is claimed when *any* pattern matches it, each under its
    own ``tags_match`` (see :class:`_ScopePattern`). Listing several is what lets
    one strategy serve several scopes without being written out per scope.

    Each setting is optional; ``None`` means "this strategy does not override
    it" and the bank-wide value (the "Default" strategy in the control plane)
    applies.
    """

    scopes: tuple[_ScopePattern, ...]
    observations_mission: str | None = None
    max_observations_per_scope: int | None = None
    consolidation_source_facts_max_tokens: int | None = None
    consolidation_source_facts_max_tokens_per_observation: int | None = None

    def claims(self, fact_tags: list[str]) -> bool:
        return any(pattern.matches(fact_tags) for pattern in self.scopes)


def _parse_consolidation_strategies(raw: Any) -> list[_ConsolidationStrategy]:
    """Parse the raw ``consolidation_strategies`` config into ordered strategies.

    Defensive for the same reason as :func:`_parse_scope_limit_rules`: the config
    round-trips as JSON through env and the bank-config API, so a malformed entry
    (or a malformed setting or pattern within an otherwise valid entry) is dropped
    rather than raised. An entry naming no usable scope, or ending up overriding
    nothing, is dropped entirely. List order is preserved: it is the priority
    order — see :func:`_strategy_for_scope`.
    """
    if not isinstance(raw, list):
        return []
    strategies: list[_ConsolidationStrategy] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        raw_scopes = entry.get("scopes")
        if not isinstance(raw_scopes, list):
            continue
        scopes = tuple(p for p in (_parse_scope_pattern(scope) for scope in raw_scopes) if p is not None)
        if not scopes:
            continue
        mission = entry.get("observations_mission")
        if not isinstance(mission, str) or not mission.strip():
            mission = None
        ints: dict[str, int] = {}
        for name in _STRATEGY_INT_SETTINGS:
            value = entry.get(name)
            # bool is an int subclass — reject True/False masquerading as a number.
            if isinstance(value, int) and not isinstance(value, bool):
                ints[name] = value
        if mission is None and not ints:
            continue
        strategies.append(_ConsolidationStrategy(scopes=scopes, observations_mission=mission, **ints))
    return strategies


def _strategies_for(config: Any) -> list[_ConsolidationStrategy]:
    return _parse_consolidation_strategies(getattr(config, "consolidation_strategies", None))


def _strategy_for_scope(config: Any, fact_tags: list[str]) -> _ConsolidationStrategy | None:
    """The one consolidation strategy that applies to a scope, if any.

    **The first strategy in list order that claims the scope wins, whole.** When
    two strategies both claim a scope (``["company:*"]`` and
    ``["company:acme"]``, say), only the earlier one applies — its settings, and
    for anything it leaves unset, the bank-wide value. A later strategy never
    fills in the earlier one's gaps.

    That is a deliberate change from the first version, which resolved each
    setting separately (mission from the first strategy that set one, cap from
    the first that set one). That let two strategies silently blend on one scope,
    so nobody could tell which strategy a scope was actually using; with one
    winner, the control plane can show it.

    A strategy that sets nothing is dropped at parse time, so it never claims a
    scope and never hides a later one.
    """
    return next((s for s in _strategies_for(config) if s.claims(fact_tags)), None)


def _effective_scope_limit(config: Any, fact_tags: list[str]) -> int:
    """Resolve the observation cap for one concrete consolidation scope.

    The winning strategy's cap (see :func:`_strategy_for_scope`) if it sets one;
    otherwise the deprecated ``observation_scope_limits``, same matching; otherwise
    the bank-wide ``max_observations_per_scope``. Wildcards live only here,
    matched against the already-resolved concrete tags — the SQL count stays
    exact and indexed.
    """
    if config is None:
        return -1
    strategy = _strategy_for_scope(config, fact_tags)
    if strategy is not None and strategy.max_observations_per_scope is not None:
        return strategy.max_observations_per_scope
    for limit_rule in _parse_scope_limit_rules(config.observation_scope_limits):
        if _scope_matches_globs(limit_rule.globs, fact_tags):
            return limit_rule.limit
    return config.max_observations_per_scope


def _config_for_scope(config: Any, fact_tags: list[str]) -> Any:
    """The bank config as one consolidation scope sees it.

    The winning strategy's settings (see :func:`_strategy_for_scope` — first
    claiming strategy wins, whole) over the bank-wide ones. The cap also honours
    the deprecated ``observation_scope_limits`` (see :func:`_effective_scope_limit`).

    Returning a whole config — rather than one resolver per setting, as the first
    version did for the mission and cap — is what lets every downstream reader
    (the related-observation recall that applies the source-facts token limits,
    the prompt that carries the mission) pick up the scope's values without being
    told about strategies. Same idea as ``apply_strategy`` for retain strategies.
    Shallow copy + setattr rather than ``dataclasses.replace`` so it also works on
    the lightweight config stand-ins tests pass in.

    Safe because each resolved scope gets its own LLM call — see the pass loop's
    ``obs_tags_override`` — so one scope's settings never reach another's call.
    """
    if config is None:
        return None
    scoped = copy.copy(config)
    strategy = _strategy_for_scope(config, fact_tags)
    if strategy is not None:
        for name in (
            "observations_mission",
            "consolidation_source_facts_max_tokens",
            "consolidation_source_facts_max_tokens_per_observation",
        ):
            value = getattr(strategy, name)
            if value is not None:
                setattr(scoped, name, value)
    # The cap resolves separately: it also has to consult the deprecated
    # observation_scope_limits.
    scoped.max_observations_per_scope = _effective_scope_limit(config, fact_tags)
    return scoped


def _build_response_model(
    max_creates: int | None = None,
    *,
    fact_ids: list[str] | None = None,
    supports_max_items: bool = True,
    lifecycle: bool = False,
    target_ids: Sequence[str] = (),
    source_ids: Sequence[str] = (),
) -> type[_ConsolidationBatchResponse]:
    """Build a response model, optionally constraining creates via JSON schema.

    Some structured-output backends (notably Bedrock Converse) reject the JSON
    Schema ``maxItems`` keyword emitted by Pydantic's list ``max_length``. Operators
    can disable the schema hint for those backends; the prompt capacity note and
    post-response truncation still enforce the observation cap.

    ``fact_ids`` pins ``source_fact_ids`` to the batch's fact ids (``enum``, plus
    ``minItems: 1`` where array-length keywords are accepted). Small models miscopy
    UUIDs — truncating them or pasting an observation id — and an action citing no
    batch fact is discarded, so a grammar-constrained provider must not be able to
    emit one (#5273). Schema only: validation stays ``list[str]``, so a provider that
    ignores the hint cannot fail the batch.

    ``lifecycle`` (adaptive memory, lifecycle mode) builds the lifecycle subclasses instead, with
    the label lists pinned to ``target_ids`` (the pool's retirable observations) and
    ``source_ids`` (the source fact ids shown with them); every other mode gets the models above,
    byte for byte.
    """
    cap = max(max_creates, 0) if supports_max_items and max_creates is not None and max_creates >= 0 else None
    if lifecycle:
        return _build_lifecycle_response_model(
            cap,
            fact_ids=fact_ids,
            supports_max_items=supports_max_items,
            target_ids=target_ids,
            source_ids=source_ids,
        )
    if cap is None and not fact_ids:
        return _ConsolidationBatchResponse

    ids_schema = _fact_ids_schema(fact_ids, supports_max_items)

    class _Create(_CreateAction):
        source_fact_ids: list[str] = Field(json_schema_extra=ids_schema)

    class _Update(_UpdateAction):
        source_fact_ids: list[str] = Field(json_schema_extra=ids_schema)

    class _ConstrainedConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_Create] = Field(default=[], max_length=cap)
        updates: list[_Update] = []

    return _ConstrainedConsolidationBatchResponse


def _fact_ids_schema(fact_ids: list[str] | None, supports_max_items: bool) -> dict[str, Any]:
    ids_schema: dict[str, Any] = {}
    if fact_ids:
        ids_schema["items"] = {"type": "string", "enum": list(fact_ids)}
        if supports_max_items:
            ids_schema["minItems"] = 1
    return ids_schema


def _pinned_ids_schema(ids: Sequence[str], supports_max_items: bool) -> dict[str, Any]:
    """An id-list field's schema hint: ``enum``-pinned to ``ids``, or empty-only when there are none.

    Schema only, like ``source_fact_ids``: validation stays a lenient ``list[str]``.
    """
    if ids:
        return {"items": {"type": "string", "enum": list(ids)}}
    return {"maxItems": 0} if supports_max_items else {}


def _pin_nullable_id(ids: Sequence[str]) -> Callable[[dict[str, Any]], None]:
    """``json_schema_extra`` for ``str | None``: put the ``enum`` on the string branch of the ``anyOf``
    (on the field itself it would forbid null)."""

    def pin(schema: dict[str, Any]) -> None:
        if not ids:
            return
        for branch in schema.get("anyOf") or []:
            if branch.get("type") == "string":
                branch["enum"] = list(ids)

    return pin


#: ``$defs`` names of the id enums the lifecycle schema spells once and references from every field.
_BATCH_FACT_ID_DEF = "BatchFactId"
_RETIRABLE_OBSERVATION_ID_DEF = "RetirableObservationId"
_SHOWN_SOURCE_ID_DEF = "ShownSourceFactId"


def _hoist_id_enums(
    schema: dict[str, Any], enums: Mapping[str, Sequence[str]], *, ref_template: str = DEFAULT_REF_TEMPLATE
) -> dict[str, Any]:
    """Spell each pinned id list once under ``$defs`` and ``$ref`` it wherever a field pins it.

    The lifecycle labels pin the pool's retirable ids on five fields and the shown source ids on
    three. Inline, every id is repeated on each of them, so the schema (sent as input on every call)
    grows with the pool several times over. The constraint itself is unchanged, and every provider
    path already resolves ``$defs`` references (the create and update actions are references too).
    """
    names = {json.dumps(list(ids)): name for name, ids in enums.items() if ids}
    used: dict[str, list[Any]] = {}

    def hoist(node: Any) -> Any:
        if isinstance(node, dict):
            if set(node) == {"type", "enum"} and node["type"] == "string":
                name = names.get(json.dumps(node["enum"]))
                if name is not None:
                    used[name] = node["enum"]
                    return {"$ref": ref_template.format(model=name)}
            return {key: hoist(value) for key, value in node.items()}
        if isinstance(node, list):
            return [hoist(item) for item in node]
        return node

    hoisted = hoist(schema)
    if used:
        defs = hoisted.setdefault("$defs", {})
        for name, ids in used.items():
            defs[name] = {"type": "string", "enum": list(ids)}
    return hoisted


def _build_lifecycle_response_model(
    cap: int | None,
    *,
    fact_ids: list[str] | None,
    supports_max_items: bool,
    target_ids: Sequence[str],
    source_ids: Sequence[str],
) -> type[_ConsolidationBatchResponse]:
    """The lifecycle-mode response model: the base fields plus the lifecycle labels.

    Target lists are ``enum``-pinned to the pool's retirable observation ids and the source-id lists
    to the source fact ids the pool showed with them, so a grammar-constrained provider can only
    name what it was shown. Each id list is spelled once in the JSON schema (``_hoist_id_enums``).
    A subclass per batch, like the base constrained model.
    """
    ids_schema = _fact_ids_schema(fact_ids, supports_max_items)
    targets = _pinned_ids_schema(target_ids, supports_max_items)
    sources = _pinned_ids_schema(source_ids, supports_max_items)

    class _Create(_LifecycleCreateAction):
        source_fact_ids: list[str] = Field(json_schema_extra=ids_schema)
        supersedes: list[str] = Field(
            default_factory=list, description=_SUPERSEDES_DESCRIPTION, json_schema_extra=targets
        )
        refutes: list[str] = Field(default_factory=list, description=_REFUTES_DESCRIPTION, json_schema_extra=targets)
        refines: list[str] = Field(default_factory=list, description=_REFINES_DESCRIPTION, json_schema_extra=targets)
        conflicts_with: list[str] = Field(
            default_factory=list, description=_CONFLICTS_DESCRIPTION, json_schema_extra=targets
        )
        split_from: str | None = Field(
            default=None, description=_SPLIT_FROM_DESCRIPTION, json_schema_extra=_pin_nullable_id(target_ids)
        )
        keeps_source_ids: list[str] = Field(
            default_factory=list, description=_KEEPS_DESCRIPTION, json_schema_extra=sources
        )

    class _Update(_LifecycleUpdateAction):
        source_fact_ids: list[str] = Field(json_schema_extra=ids_schema)
        retires_source_ids: list[str] = Field(
            default_factory=list, description=_RETIRES_DESCRIPTION, json_schema_extra=sources
        )
        keeps_source_ids: list[str] = Field(
            default_factory=list, description=_KEEPS_DESCRIPTION, json_schema_extra=sources
        )

    shared_enums = {
        _BATCH_FACT_ID_DEF: list(fact_ids or []),
        _RETIRABLE_OBSERVATION_ID_DEF: list(target_ids),
        _SHOWN_SOURCE_ID_DEF: list(source_ids),
    }

    class _LifecycleConsolidationBatchResponse(_ConsolidationBatchResponse):
        creates: list[_Create] = Field(default=[], max_length=cap)
        updates: list[_Update] = []

        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = DEFAULT_REF_TEMPLATE,
            schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
            mode: JsonSchemaMode = "validation",
            **kwargs: Any,
        ) -> dict[str, Any]:
            schema = super().model_json_schema(by_alias, ref_template, schema_generator, mode, **kwargs)
            return _hoist_id_enums(schema, shared_enums, ref_template=ref_template)

    return _LifecycleConsolidationBatchResponse


class ConsolidationPerfLog:
    """Performance logging for consolidation operations."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.start_time = time.perf_counter()
        self.lines: list[str] = []
        self.timings: dict[str, float] = {}
        self.timing_counts: dict[str, int] = {}
        self.llm_calls: int = 0
        self.total_obs_in_context: int = 0
        self.total_prompt_chars: int = 0
        self.llm_batch_failures: int = 0
        self.unresolved_actions: int = 0

    def log(self, message: str) -> None:
        """Add a log line."""
        self.lines.append(message)

    def record_timing(self, key: str, duration: float) -> None:
        """Record a timing measurement.

        Tracks both total seconds and call count so the summary can
        distinguish one slow call from many fast calls in aggregate.
        """
        self.timings[key] = self.timings.get(key, 0.0) + duration
        self.timing_counts[key] = self.timing_counts.get(key, 0) + 1

    def record_llm_call(self, obs_count: int, prompt_chars: int) -> None:
        """Record stats for a single LLM call."""
        self.llm_calls += 1
        self.total_obs_in_context += obs_count
        self.total_prompt_chars += prompt_chars

    def record_llm_batch_failures(self, count: int) -> None:
        """Record LLM batch attempts that raised, whether or not a retry rescued them."""
        self.llm_batch_failures += count

    def merge_from(self, other: "ConsolidationPerfLog") -> None:
        """Merge a per-batch perf log into this (job-level) one.

        Used by the parallel dispatcher: each in-flight batch records into its
        own ``ConsolidationPerfLog`` so the per-batch log line shows only that
        batch's timings (no cross-batch interleaving). After the batch finishes
        we fold the local counters into the job-level perf, which then drives
        the final ``flush()`` summary.

        ``lines`` is intentionally NOT merged — log lines are emitted directly
        in ``logger.info`` calls by the dispatcher; the perf object's ``lines``
        buffer is only used by the top-level job summary.
        """
        for key, value in other.timings.items():
            self.timings[key] = self.timings.get(key, 0.0) + value
        for key, count in other.timing_counts.items():
            self.timing_counts[key] = self.timing_counts.get(key, 0) + count
        self.llm_calls += other.llm_calls
        self.total_obs_in_context += other.total_obs_in_context
        self.total_prompt_chars += other.total_prompt_chars
        self.llm_batch_failures += other.llm_batch_failures
        self.unresolved_actions += other.unresolved_actions

    def flush(self) -> None:
        """Flush all log lines to the logger."""
        total_time = time.perf_counter() - self.start_time
        header = f"\n{'=' * 60}\nCONSOLIDATION for bank {self.bank_id}"
        footer = f"{'=' * 60}\nCONSOLIDATION COMPLETE: {total_time:.3f}s total\n{'=' * 60}"

        log_output = header + "\n" + "\n".join(self.lines) + "\n" + footer
        logger.info(log_output)


def _log_unresolved_action(kind: str, source_fact_ids: list[str], text: str, perf: ConsolidationPerfLog | None) -> None:
    """Report a create/update the model wrote but whose ``source_fact_ids`` match no batch fact.

    The action cannot be applied — there is no fact to attach it to — but it is not the
    model declining the facts either, so it must not vanish into ``no_durable_knowledge``
    (#5273). The schema's ``enum`` prevents this on grammar-constrained providers; this is
    what is left for the ones that ignore it.
    """
    logger.warning(
        "[CONSOLIDATION] discarded observation %s: none of its source_fact_ids %s is a fact in this batch; text=%r",
        kind.upper(),
        source_fact_ids,
        text[:200],
    )
    if perf:
        perf.unresolved_actions += 1


def _as_dt(v: "datetime | str | None") -> "datetime | None":
    """Coerce an ISO string to a datetime. Recall results can carry timestamps as strings while
    the store's addressed reads hand back datetimes, so normalise before comparing."""
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def _merge_min(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``LEAST(a, COALESCE(b, a))`` in Python: the earlier of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else min(a, b)


def _merge_max(a: "datetime | str | None", b: "datetime | str | None") -> "datetime | None":
    """SQL ``GREATEST(a, COALESCE(b, a))`` in Python: the later of two times, ignoring None."""
    a, b = _as_dt(a), _as_dt(b)
    return a if b is None else b if a is None else max(a, b)


#: How many rounds' worth of candidates the fair fetch looks at before choosing one round.
#: Whatever the window misses, the next round sees, so this trades a bigger read for fairness
#: rather than for correctness.
#: ponytail: fixed factor, make it configurable if a bank's groups are wider than 5 rounds.
_FAIR_FETCH_OVERFETCH = 5


def _fair_group_slice(memories: list[StoredMemory], limit: int, quota: int) -> list[StoredMemory]:
    """Oldest-first, but at most ``quota`` facts per consolidation group.

    ``memories`` must already be sorted oldest-first: groups are then visited in the order
    of their oldest fact, and the round fills from the front. Keying is
    ``_consolidation_batch_key``, the same key the dispatcher groups by, so a round that
    holds N keys gives the dispatcher N groups to run in parallel.
    """
    taken: list[StoredMemory] = []
    seen: defaultdict[tuple[str, ...], int] = defaultdict(int)
    for m in memories:
        key = _consolidation_batch_key({"tags": list(m.tags or []), "observation_scopes": m.observation_scopes})
        if seen[key] >= quota:
            continue
        seen[key] += 1
        taken.append(m)
        if len(taken) >= limit:
            break
    return taken


async def _fetch_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    observation_scopes: list[list[str]] | None,
    llm_parallelism: int,
    with_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Unconsolidated candidate facts, read through the memories store.

    The store owns the memories, so this must ask it rather than query ``memory_units``
    directly — otherwise a store that keeps its rows elsewhere yields nothing and
    consolidation silently produces no observations. Returns the same row-dict shape the
    consolidation loop consumes. Mirrors the job's scope filter: with scopes, OR each
    "tags ⊇ scope" and merge oldest-first; without, one unscoped read.

    With ``llm_parallelism > 1`` the round is picked *fairly* across consolidation groups
    instead of strictly oldest-first. A strict oldest-first round holds only the group that
    owns the oldest facts, and the dispatcher parallelises across groups — one group in the
    round means one LLM call at a time, whatever the parallelism, and a big group's backlog
    starves every other group until it drains (#4823). So read a window of
    ``_FAIR_FETCH_OVERFETCH`` rounds and take at most ``ceil(limit / llm_parallelism)`` facts
    per group, leaving enough distinct groups in the round to fill the parallel slots. Facts
    are still consumed oldest-first *within* a group, which is the ordering consolidation
    actually depends on.

    ``with_provenance`` (adaptive memory in rank/lifecycle) adds what reinforcement keys a fact by:
    ``document_id``, ``metadata`` and ``document_id_minted``. The first two come with the read; the
    minted marks are one batched read of the round's documents. Nothing reaches the prompt.
    """
    store = get_memories()
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    fair = llm_parallelism > 1
    read_limit = limit * _FAIR_FETCH_OVERFETCH if fair else limit
    by_id: dict[str, Any] = {}
    for scope in scopes:
        for m in await store.find_unconsolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=read_limit, scope_tags=scope
        ):
            by_id.setdefault(m.unit_id, m)
    oldest_first = sorted(by_id.values(), key=lambda m: (m.created_at is None, m.created_at))
    ordered = (
        _fair_group_slice(oldest_first, limit, math.ceil(limit / llm_parallelism)) if fair else oldest_first[:limit]
    )
    rows = [
        {
            "id": uuid.UUID(m.unit_id),
            "text": m.text,
            "fact_type": m.fact_type,
            "occurred_start": m.occurred_start,
            "occurred_end": m.occurred_end,
            "event_date": m.event_date,
            "tags": list(m.tags or []),
            "mentioned_at": m.mentioned_at,
            "observation_scopes": m.observation_scopes,
            # Read-time version: the write re-checks it (see _sources_changed_since_read).
            "updated_at": m.updated_at,
            # The prompt shows only its scope keys (see _scope_label).
            "metadata": m.metadata or {},
        }
        for m in ordered
    ]
    if with_provenance and rows:
        minted = await store.minted_document_ids(
            conn, bank_id=bank_id, document_ids=[m.document_id for m in ordered if m.document_id]
        )
        for row, m in zip(rows, ordered):
            row["document_id"] = m.document_id
            row["document_id_minted"] = m.document_id is not None and str(m.document_id) in minted
    return rows


#: Cap on the store-side count of unconsolidated facts. Used only for the "is there work?"
#: gate and progress reporting, so a floor at this size is harmless on a huge backlog.
_COUNT_LIMIT = 100_000


async def _count_unconsolidated_rows(
    conn,
    bank_id: str,
    fact_types: list[str],
    observation_scopes: list[list[str]] | None,
) -> int:
    """Count of unconsolidated candidate facts, from the store (bounded by ``_COUNT_LIMIT``).

    Asks the store for a *count* rather than fetching the rows and taking ``len`` — on the SQL
    store that is one bounded ``COUNT(*)`` instead of shipping up to ``_COUNT_LIMIT`` full memory
    rows across the wire on every job start / progress tick.
    """
    scopes: list[list[str] | None] = list(observation_scopes) if observation_scopes else [None]
    return await get_memories().count_unconsolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, scopes=scopes, limit=_COUNT_LIMIT
    )


def _as_op_uuid(operation_id: str | uuid.UUID) -> uuid.UUID:
    return uuid.UUID(operation_id) if isinstance(operation_id, str) else operation_id


async def _persist_pending_refresh_tags(conn, operation_id: str, new_tags: list[str]) -> None:
    """Union ``new_tags`` into the consolidation op's durable ``pending_refresh_tags``.

    Called inside each batch's witness transaction, so the tags of an
    already-consolidated batch are durable the instant that batch is — a mid-round
    worker crash no longer loses them. On retry the op re-reads ``task_payload`` and the
    final round still refreshes those models (#3411); without this, a crash after batch 1
    committed but before the round finished would drop batch 1's tags, because the retry
    skips its now-consolidated rows and never re-collects them. ``SELECT ... FOR UPDATE``
    serialises the concurrent batches of one op so their unions don't clobber each other.
    """
    op_uuid = _as_op_uuid(operation_id)
    row = await conn.fetchrow(
        f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1 FOR UPDATE",
        op_uuid,
    )
    if row is None:
        return
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    existing = set(payload.get("pending_refresh_tags") or [])
    merged = existing | set(new_tags)
    if merged == existing:
        return
    payload["pending_refresh_tags"] = sorted(merged)
    await conn.execute(
        f"UPDATE {fq_table('async_operations')} SET task_payload = $1::jsonb, updated_at = now() "
        f"WHERE operation_id = $2",
        json.dumps(payload),
        op_uuid,
    )


async def _read_pending_refresh_tags(pool, operation_id: str) -> set[str]:
    """Read the op's durably-accumulated ``pending_refresh_tags`` (crash-safe source of
    truth for the final-round flush)."""
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(
            f"SELECT task_payload FROM {fq_table('async_operations')} WHERE operation_id = $1",
            _as_op_uuid(operation_id),
        )
    if row is None:
        return set()
    payload = row["task_payload"]
    payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    return set(payload.get("pending_refresh_tags") or [])


async def run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run consolidation job for a bank.

    This is called after retain operations to consolidate new memories into mental models.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        operation_id: Optional operation ID for tracking
        observation_scopes: Optional list of tag scopes. When provided, only
            unconsolidated memories whose tags contain all tags in at least one
            scope are processed.
        pending_refresh_tags: Tags of memories consolidated by earlier rounds of this
            round-limited chain, carried through the re-queue so the final round can
            refresh every affected mental model exactly once (#3411).

    Returns:
        Dict with consolidation results
    """
    # Resolve bank-specific config with hierarchical overrides
    config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # Build a configured LLM wrapper that applies per-bank settings (e.g. safety settings)
    # to every call without leaking across operations.
    llm_config = memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation")

    # Bind the operation trace context for the whole run so the create/update DB
    # sites (deep inside _process_memory_batch) can accumulate the observations
    # this consolidation produced and the source memories it consumed onto the
    # trace — flushed onto every trace row on exit by attach_memory_ids.
    trace_ctx = trace_context_of(llm_config)
    trace_token = set_trace_context(trace_ctx) if trace_ctx is not None else None
    try:
        return await _run_consolidation_job(
            memory_engine,
            bank_id,
            request_context,
            config,
            llm_config,
            operation_id,
            observation_scopes,
            pending_refresh_tags,
        )
    finally:
        if trace_token is not None:
            reset_trace_context(trace_token)
            # Fire-and-forget: patched on a background task, off the consolidation
            # critical path.
            memory_engine._llm_recorder.attach_memory_ids(trace_ctx)


async def _run_consolidation_job(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    config: Any,
    llm_config: Any,
    operation_id: str | None = None,
    observation_scopes: list[list[str]] | None = None,
    pending_refresh_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Core consolidation flow. See ``run_consolidation_job`` for the public entrypoint."""
    perf = ConsolidationPerfLog(bank_id)
    max_memories_per_batch = config.consolidation_batch_size
    max_memories_per_round = config.consolidation_max_memories_per_round
    llm_batch_size = max(1, config.consolidation_llm_batch_size)

    # Check if consolidation is enabled
    if not config.enable_observations:
        logger.debug(f"Consolidation disabled for bank {bank_id}")
        return {"status": "disabled", "bank_id": bank_id}

    pool = memory_engine._backend

    # Adaptive memory: reinforcement settings while the bank's mode writes state (rank/lifecycle)
    # and its reinforcement switch is on; None otherwise, and the batches run exactly as before.
    adaptive_settings = write_settings(config, bank_id=bank_id)
    if adaptive_settings is not None and not reinforcement_allowed(adaptive_settings):
        adaptive_settings = None
    # Claim lifecycle: lifecycle mode labels and applies transitions, observe mode records the
    # retirements a plain delete would have been. None otherwise: off and rank run as before.
    lifecycle_settings = mode_settings(config, bank_id=bank_id)
    lifecycle_run = (
        _LifecycleRun(lifecycle_settings)
        if lifecycle_settings is not None
        and (lifecycle_settings.effective_mode.lifecycle or lifecycle_settings.effective_mode is AdaptiveMode.OBSERVE)
        else None
    )

    # Get bank profile
    async with acquire_with_retry(pool) as conn:
        t0 = time.perf_counter()
        bank_row = await conn.fetchrow(
            f"""
            SELECT bank_id, name
            FROM {fq_table("banks")}
            WHERE bank_id = $1
            """,
            bank_id,
        )

        if not bank_row:
            logger.warning(f"Bank {bank_id} not found for consolidation")
            return {"status": "bank_not_found", "bank_id": bank_id}

        perf.record_timing("fetch_bank", time.perf_counter() - t0)

        # Count total unconsolidated memories for progress logging — through the store.
        total_count = await _count_unconsolidated_rows(conn, bank_id, ["experience", "world"], observation_scopes)

    if total_count == 0:
        logger.debug(f"No new memories to consolidate for bank {bank_id}")
        return {"status": "no_new_memories", "bank_id": bank_id, "memories_processed": 0}

    logger.info(f"[CONSOLIDATION] bank={bank_id} total_unconsolidated={total_count}")
    perf.log(f"[1] Found {total_count} pending memories to consolidate")

    # Initial durable progress snapshot so an operator polling the operation status
    # API sees the job has started and how much work it found, before the first batch
    # of LLM work completes (which can take minutes on a dense bank). Uses the same
    # "consolidating" stage as the per-batch heartbeat so the operator sees a single
    # phase advancing 0/N -> N/N rather than an opaque "scanning" -> "processing" hop.
    set_stage("consolidation.consolidating")
    await memory_engine._write_operation_progress(operation_id, stage="consolidating", processed=0, total=total_count)

    async def _count_unconsolidated() -> int:
        """Re-count memories still pending consolidation in this job's scope.

        ``total_count`` is a point-in-time estimate from job start; memories retained
        while consolidation runs get picked up by later fetches, so processed can pass
        it. When that happens we re-count to report a real total (processed + remaining)
        instead of pinning the bar at 100%."""
        async with acquire_with_retry(pool) as count_conn:
            return await _count_unconsolidated_rows(count_conn, bank_id, ["experience", "world"], observation_scopes)

    async def _progress_total(processed: int) -> int:
        # Cheap path: while we're still within the start-of-job estimate it's exact, so
        # no extra query. Only re-count once the estimate is exhausted (≈the final batch
        # normally, or repeatedly only if memories keep arriving mid-run).
        if processed < total_count:
            return total_count
        return processed + await _count_unconsolidated()

    # Process each memory with individual commits for crash recovery
    stats: dict[str, int] = {
        "memories_processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "actions_executed": 0,
        "skipped": 0,
        "memories_failed": 0,
        # LLM batch attempts that raised, including those a retry or the adaptive bisection
        # later rescued. `memories_failed` counts only facts left stuck, so it reads 0 for
        # a run that discarded every response it got (#4151, #4152). One batch call can
        # contribute several attempts, so this is not bounded by the batch count.
        "llm_batch_failures": 0,
        # Creates/updates the model wrote whose source_fact_ids matched no fact in their
        # batch, so they were discarded unapplied (#5273).
        "unresolved_actions": 0,
    }

    # Track all unique tags from consolidated memories for mental model refresh filtering
    consolidated_tags: set[str] = set()

    round_limit_enabled = max_memories_per_round > 0
    round_remaining = max_memories_per_round if round_limit_enabled else float("inf")
    hit_round_limit = False

    llm_batch_num = 0
    # Cumulative counters across the whole job, shared by the per-batch log and the
    # durable progress snapshot so both report processed/total (and observation
    # tallies) under parallelism. Mutable container so the inner closure can update
    # without a `nonlocal`.
    cumulative_progress = {
        "processed": 0,
        "observations_created": 0,
        "observations_updated": 0,
        "observations_merged": 0,
        "observations_deleted": 0,
        "memories_failed": 0,
    }
    while True:
        # Cap fetch size by remaining round budget
        fetch_limit = (
            min(max_memories_per_batch, int(round_remaining)) if round_limit_enabled else max_memories_per_batch
        )

        # Fetch next batch of unconsolidated memories — through the store, so a store that
        # keeps its rows outside Postgres is read too.
        async with acquire_with_retry(pool) as conn:
            t0 = time.perf_counter()
            memories = await _fetch_unconsolidated_rows(
                conn,
                bank_id,
                ["experience", "world"],
                fetch_limit,
                observation_scopes,
                max(1, config.consolidation_llm_parallelism),
                # Lifecycle mode reads the facts' metadata too (scope fingerprints, source trust).
                with_provenance=adaptive_settings is not None
                or (lifecycle_run is not None and lifecycle_run.lifecycle),
            )
            perf.record_timing("fetch_memories", time.perf_counter() - t0)

        if not memories:
            break  # No more unconsolidated memories

        # Group memories by target observation scope before batching — security
        # requirement: memories targeting different scopes must never share an
        # LLM call. See _consolidation_batch_key for why that's scope, not raw
        # tags: a memory requesting observation_scopes="shared" (or another
        # single-scope override) targets a scope that can differ from its own
        # tags, and must only batch with peers naming that same scope (#3924).
        tag_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for m in memories:
            tag_key = _consolidation_batch_key(m)
            tag_groups.setdefault(tag_key, []).append(dict(m))

        # Split each tag group into LLM batches respecting llm_batch_size, keeping
        # the group boundary intact so the dispatcher can parallelise across
        # distinct groups while running each group's batches serially.
        grouped_batches: list[list[list[dict[str, Any]]]] = []
        for group in tag_groups.values():
            grouped_batches.append([group[i : i + llm_batch_size] for i in range(0, len(group), llm_batch_size)])

        # Compute each group's union write-scope set. Used below to acquire
        # per-scope locks: any two groups whose write-scope sets share a scope S
        # will serialise on the lock for S, leaving truly disjoint groups to run
        # concurrently. We union over every memory because per-memory
        # observation_scopes can differ within a group.
        group_scopes: list[list[frozenset[str]]] = []
        for batches in grouped_batches:
            scopes: set[frozenset[str]] = set()
            for batch in batches:
                for memory in batch:
                    scopes.update(_resolve_write_scopes(memory))
            group_scopes.append(sorted(scopes, key=_scope_sort_key))

        async def _process_one_llm_batch(llm_batch_local: list[dict[str, Any]], batch_num_local: int) -> _BatchDeltas:
            """Process one LLM batch independently. Returns local deltas + cancelled flag.

            Each batch records timings/llm-call counters into its OWN
            ``ConsolidationPerfLog`` so the per-batch log line reflects only
            this batch's work — not interleaved timings from concurrent batches
            sharing the global ``perf``. The local perf is merged into the
            job-level ``perf`` once at the end so the final summary still totals
            everything.
            """
            llm_batch_start = time.perf_counter()
            batch_perf = ConsolidationPerfLog(bank_id)

            local_tags: set[str] = set()
            for memory in llm_batch_local:
                memory_tags = memory.get("tags") or []
                if memory_tags:
                    local_tags.update(memory_tags)
            batch_lifecycle = _BatchLifecycle(lifecycle_run) if lifecycle_run is not None else None

            # Adaptive splitting: on LLM failure, halve the sub-batch and retry,
            # down to batch_size=1. Only if a single-memory batch still fails is
            # the memory marked with consolidation_failed_at.
            all_results: list[dict[str, Any]] = []
            all_deleted = 0
            succeeded_ids: list[Any] = []
            failed_ids: list[Any] = []

            pending: list[list[dict[str, Any]]] = [llm_batch_local]
            # How many extra looks each fact has had (see below): at most two.
            reasked: dict[str, int] = {}
            while pending:
                sub_batch = pending.pop(0)

                # Defence in depth. Everything below writes the sub-batch at ONE
                # resolved scope, read off ``sub_batch[0]``. If a memory with a
                # different scope ever reaches this batch — a regression in
                # ``_consolidation_batch_key``, or in how groups are split into
                # batches — that single read stamps one memory's scope onto
                # another's observation: an untagged, globally recallable
                # observation built from a tagged fact, or a dropped override
                # (#3953). So verify the invariant against the scopes the pass
                # loop will actually write, independently of the grouping key,
                # and split instead of trusting it. The cost in the case that
                # must never happen is one extra LLM call.
                if len(sub_batch) > 1:
                    by_scope: dict[tuple[tuple[str, ...], ...], list[dict[str, Any]]] = {}
                    for m in sub_batch:
                        by_scope.setdefault(_batch_scope_signature(m), []).append(m)
                    if len(by_scope) > 1:
                        logger.error(
                            f"[CONSOLIDATION] bank={bank_id} sub-batch of {len(sub_batch)} memories mixes"
                            f" {len(by_scope)} observation scopes {sorted(by_scope)} — splitting it."
                            " LLM batches must be scope-homogeneous; this is a grouping bug."
                        )
                        pending[0:0] = list(by_scope.values())
                        continue

                # No connection is held across the batch: recall, the main LLM call, the
                # per-action embeds, and dedup adjudication all run connection-free. Only then
                # does the sub-batch take one connection and write everything that LLM response
                # decided — observations and consolidated_at stamps alike — in one transaction.
                obs_tags_list = _resolve_obs_tags_list(sub_batch[0]) if sub_batch else None

                sub_deleted: int = 0
                sub_llm_failed = False
                sub_ids = [m["id"] for m in sub_batch]
                if obs_tags_list:
                    sub_results: list[dict[str, Any]] = []
                    for pass_index, obs_tags in enumerate(obs_tags_list):
                        # A memory consolidated at several tag scopes gets one LLM call per
                        # scope; the ``consolidated_at`` stamp belongs to the last of them, so
                        # an earlier scope's write and the stamp never commit apart (#3876).
                        is_final_pass = pass_index == len(obs_tags_list) - 1
                        pass_results, pass_deleted, pass_failed = await _process_memory_batch(
                            # Optional on the caller only because the engine clears its backend on close.
                            pool=cast("DatabaseBackend", pool),
                            memory_engine=memory_engine,
                            llm_config=llm_config,
                            bank_id=bank_id,
                            memories=sub_batch,
                            request_context=request_context,
                            perf=batch_perf,
                            config=config,
                            obs_tags_override=obs_tags,
                            mark_consolidated_ids=sub_ids if is_final_pass else None,
                            adaptive_settings=adaptive_settings,
                            lifecycle=batch_lifecycle,
                        )
                        sub_deleted += pass_deleted
                        if pass_failed:
                            # Stop the remaining scopes: the sub-batch is going to be bisected
                            # and re-run in full, and every write this pass made is already
                            # committed. Running the rest would only add writes to discard.
                            sub_llm_failed = True
                            break
                        if not sub_results:
                            sub_results = pass_results
                        else:
                            for i, (existing, new) in enumerate(zip(sub_results, pass_results)):
                                if existing.get("action") == "skipped" and new.get("action") != "skipped":
                                    sub_results[i] = new
                                elif existing.get("action") != "skipped" and new.get("action") != "skipped":
                                    existing_created = existing.get(
                                        "created", 1 if existing.get("action") == "created" else 0
                                    )
                                    existing_updated = existing.get(
                                        "updated", 1 if existing.get("action") == "updated" else 0
                                    )
                                    new_created = new.get("created", 1 if new.get("action") == "created" else 0)
                                    new_updated = new.get("updated", 1 if new.get("action") == "updated" else 0)
                                    total = existing_created + existing_updated + new_created + new_updated
                                    sub_results[i] = {
                                        "action": "multiple",
                                        "created": existing_created + new_created,
                                        "updated": existing_updated + new_updated,
                                        "merged": 0,
                                        "total_actions": total,
                                    }
                else:
                    sub_results, sub_deleted, sub_llm_failed = await _process_memory_batch(
                        # Optional on the caller only because the engine clears its backend on close.
                        pool=cast("DatabaseBackend", pool),
                        memory_engine=memory_engine,
                        llm_config=llm_config,
                        bank_id=bank_id,
                        memories=sub_batch,
                        request_context=request_context,
                        perf=batch_perf,
                        config=config,
                        mark_consolidated_ids=sub_ids,
                        adaptive_settings=adaptive_settings,
                        lifecycle=batch_lifecycle,
                    )

                all_deleted += sub_deleted

                if sub_llm_failed and len(sub_batch) > 1:
                    mid = len(sub_batch) // 2
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for sub-batch of {len(sub_batch)},"
                        f" splitting into {mid}/{len(sub_batch) - mid}"
                    )
                    pending[0:0] = [sub_batch[:mid], sub_batch[mid:]]
                elif sub_llm_failed:
                    failed_ids.append(sub_batch[0]["id"])
                    all_results.append({"action": "failed"})
                    logger.warning(
                        f"[CONSOLIDATION] bank={bank_id} LLM failed for single memory"
                        f" {sub_batch[0]['id']}, marking consolidation_failed_at"
                    )
                else:
                    # A fact the response never acted on is stamped consolidated all the same, and
                    # would never be shown to the model again. In a batch that is as likely an
                    # oversight as a judgement (a small model handles seven of eight facts and
                    # forgets the eighth), so the facts it left are asked again together, in a
                    # smaller batch; one that batch forgets too gets a last call of its own.
                    # Skipped while alone is the model's answer (chatter) and is never re-asked.
                    ignored = (
                        [
                            m
                            for m, r in zip(sub_batch, sub_results)
                            if r.get("reason") == "no_durable_knowledge" and reasked.get(str(m["id"]), 0) < 2
                        ]
                        if len(sub_batch) > 1
                        else []
                    )
                    ignored_ids = {str(m["id"]) for m in ignored}
                    for m, r in zip(sub_batch, sub_results):
                        if str(m["id"]) not in ignored_ids:
                            succeeded_ids.append(m["id"])
                            all_results.append(r)
                    if ignored:
                        logger.info(
                            f"[CONSOLIDATION] bank={bank_id} response left {len(ignored)}/{len(sub_batch)}"
                            " facts unaddressed; re-asking them"
                        )
                        first = [m for m in ignored if str(m["id"]) not in reasked]
                        again = [m for m in ignored if str(m["id"]) in reasked]
                        for fid in ignored_ids:
                            reasked[fid] = reasked.get(fid, 0) + 1
                        pending[0:0] = ([first] if first else []) + [[m] for m in again]

            # The successful sub-batches stamped their own ``consolidated_at`` inside the
            # transaction that wrote their observations (#3876) — a stamp and the writes it
            # accounts for must never commit apart. What is left here is the failed ids, which
            # have no writes to share a transaction with, and the refresh tags. Marking goes
            # through the store so the flag lands wherever the source facts live. ONE short
            # transaction, with no LLM work inside it: the sub-batch loop above must not hold a
            # Postgres connection across its LLM calls.
            async with acquire_with_retry(pool) as conn:
                store = get_memories()
                now = datetime.now(timezone.utc)
                if failed_ids:
                    await store.mark_consolidated(
                        conn=conn,
                        fq_table=fq_table,
                        bank_id=bank_id,
                        unit_ids=[str(mem_id) for mem_id in failed_ids],
                        when=now,
                        failed=True,
                    )
                async with conn.transaction():
                    # Persist this batch's mental-model refresh tags atomically with the
                    # witness, so they share the batch's fate: durable iff the batch is
                    # (#3411). Only the succeeded source facts — the ones just marked
                    # consolidated — contribute a tag. (Store-owned: no witness, so this is a
                    # plain best-effort write; a missed tag only defers a mental-model refresh.)
                    # A retired observation's tags count too: a mental model grounded on it must refresh.
                    retired_tags = batch_lifecycle.retired_tags if batch_lifecycle is not None else set()
                    if operation_id and succeeded_ids:
                        succeeded_set = {str(mem_id) for mem_id in succeeded_ids}
                        batch_tags = sorted(
                            {t for m in llm_batch_local if str(m["id"]) in succeeded_set for t in (m.get("tags") or [])}
                            | retired_tags
                        )
                        if batch_tags:
                            await _persist_pending_refresh_tags(conn, operation_id, batch_tags)

            cancelled_local = False
            if operation_id and not await memory_engine._check_op_alive(operation_id):
                logger.info(f"[CONSOLIDATION] bank={bank_id} operation {operation_id} cancelled, stopping early")
                cancelled_local = True

            # Per-batch local stats; merged into outer state once, serially,
            # after dispatch completes.
            local_stats: dict[str, int] = {
                "memories_processed": 0,
                "observations_created": 0,
                "observations_updated": 0,
                "observations_merged": 0,
                "observations_deleted": all_deleted,
                "actions_executed": 0,
                "skipped": 0,
                "memories_failed": 0,
            }
            for result in all_results:
                local_stats["memories_processed"] += 1
                action = result.get("action")
                if action == "created":
                    local_stats["observations_created"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "updated":
                    local_stats["observations_updated"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "merged":
                    local_stats["observations_merged"] += 1
                    local_stats["actions_executed"] += 1
                elif action == "multiple":
                    local_stats["observations_created"] += result.get("created", 0)
                    local_stats["observations_updated"] += result.get("updated", 0)
                    local_stats["observations_merged"] += result.get("merged", 0)
                    local_stats["actions_executed"] += result.get("total_actions", 0)
                elif action == "skipped":
                    local_stats["skipped"] += 1
                elif action == "failed":
                    local_stats["memories_failed"] += 1

            # Maintain the cumulative-progress indicator under parallelism:
            # increment shared counters and snapshot under the same statements so
            # the snapshot includes this batch. No await between the reads and
            # writes, so single-threaded asyncio gives us atomicity for free —
            # no lock needed.
            cumulative_progress["processed"] += local_stats["memories_processed"]
            cumulative_progress["observations_created"] += local_stats["observations_created"]
            cumulative_progress["observations_updated"] += local_stats["observations_updated"]
            cumulative_progress["observations_merged"] += local_stats["observations_merged"]
            cumulative_progress["observations_deleted"] += local_stats["observations_deleted"]
            cumulative_progress["memories_failed"] += local_stats["memories_failed"]
            cum_processed = cumulative_progress["processed"]
            cum_snapshot = dict(cumulative_progress)

            # Per-batch log uses batch_perf so timings/llm-calls/tokens reflect
            # only this batch's own work, even when other batches are running
            # concurrently under parallelism > 1. ``processed=`` is the
            # cumulative count across all batches that have finished so far in
            # this job (monotonic, may be reported out of strict batch-number
            # order under parallelism).
            llm_batch_time = time.perf_counter() - llm_batch_start
            timing_parts = [
                f"{key}={batch_perf.timings[key]:.3f}s"
                for key in ("recall", "llm", "embedding", "db_write")
                if key in batch_perf.timings
            ]
            input_tokens = int(batch_perf.total_prompt_chars / 4)
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} llm_batch #{batch_num_local}"
                f" ({len(llm_batch_local)} memories, {batch_perf.llm_calls} llm calls)"
                f" | processed={cum_processed}/{total_count}"
                f" | {', '.join(timing_parts)}"
                f" | created={local_stats['observations_created']}"
                f" updated={local_stats['observations_updated']}"
                f" skipped={local_stats['skipped']}"
                + (f" failed={local_stats['memories_failed']}" if local_stats["memories_failed"] else "")
                + f" | input_tokens=~{input_tokens}"
                f" | avg={llm_batch_time / max(1, len(llm_batch_local)):.3f}s/memory"
            )

            # Durable progress snapshot per LLM batch — this is the heartbeat an
            # operator polls. The whole fetched batch is processed inside one outer
            # round, so a round-boundary write would sit at the pre-round count for
            # the entire (often minutes-long) LLM phase; writing here advances
            # processed/total as each batch commits. set_stage mirrors it for the
            # live worker log.
            set_stage(f"consolidation.llm_batch.{batch_num_local}")
            await memory_engine._write_operation_progress(
                operation_id,
                stage="consolidating",
                processed=cum_processed,
                total=await _progress_total(cum_processed),
                detail={
                    "observations_created": cum_snapshot["observations_created"],
                    "observations_updated": cum_snapshot["observations_updated"],
                    "observations_merged": cum_snapshot["observations_merged"],
                    "observations_deleted": cum_snapshot["observations_deleted"],
                    "memories_failed": cum_snapshot["memories_failed"],
                },
            )

            # Fold batch counters into the job-level perf so the final summary
            # (perf.flush) totals every batch correctly. Safe without a lock —
            # ConsolidationPerfLog.merge_from is a series of += on Python ints
            # and floats with no intervening awaits, so single-threaded asyncio
            # gives us atomicity.
            perf.merge_from(batch_perf)

            if batch_lifecycle is not None:
                local_tags.update(batch_lifecycle.retired_tags)
            return _BatchDeltas(stats=local_stats, tags=local_tags, cancelled=cancelled_local)

        # Number every batch up front so log line numbering is deterministic
        # regardless of dispatch order under parallelism. Each group keeps its own
        # (batch, number) list so it can be processed as one serial unit.
        numbered_groups: list[list[tuple[list[dict[str, Any]], int]]] = []
        for batches in grouped_batches:
            numbered: list[tuple[list[dict[str, Any]], int]] = []
            for b in batches:
                llm_batch_num += 1
                numbered.append((b, llm_batch_num))
            numbered_groups.append(numbered)

        async def _process_tag_group(
            group_batches: list[tuple[list[dict[str, Any]], int]],
        ) -> list[_BatchDeltas]:
            # Batches within a group share a tag set and observation scope, so
            # they MUST run serially. Stop early if the op was cancelled mid-group.
            deltas: list[_BatchDeltas] = []
            for b, n in group_batches:
                d = await _process_one_llm_batch(b, n)
                deltas.append(d)
                if d.cancelled:
                    break
            return deltas

        llm_parallelism = max(1, config.consolidation_llm_parallelism)

        if llm_parallelism > 1 and len(numbered_groups) > 1:
            sem = asyncio.Semaphore(llm_parallelism)
            # Per-scope async locks shared across all parallel groups in this
            # fetch iteration. Each group acquires locks for every scope it will
            # write to, in _scope_sort_key order (deadlock-free). Groups with
            # disjoint scope sets never contend; any overlap serialises on the
            # overlapping scopes — covering combined / per_tag / all_combinations
            # / explicit-list modes uniformly without operator opt-in.
            scope_locks: defaultdict[frozenset[str], asyncio.Lock] = defaultdict(asyncio.Lock)

            async def _run_group(
                group_batches: list[tuple[list[dict[str, Any]], int]],
                scopes: list[frozenset[str]],
            ) -> list[_BatchDeltas]:
                async with sem:
                    async with AsyncExitStack() as stack:
                        for s in scopes:
                            await stack.enter_async_context(scope_locks[s])
                        return await _process_tag_group(group_batches)

            group_results = await _gather_or_cancel([_run_group(g, s) for g, s in zip(numbered_groups, group_scopes)])
            batch_results: list[_BatchDeltas] = [d for gd in group_results for d in gd]
            any_cancelled = any(d.cancelled for d in batch_results)
        else:
            batch_results = []
            any_cancelled = False
            for g in numbered_groups:
                group_deltas = await _process_tag_group(g)
                batch_results.extend(group_deltas)
                if any(d.cancelled for d in group_deltas):
                    any_cancelled = True
                    break

        # Merge per-batch deltas into outer state — serial, post-dispatch, so
        # concurrent batches cannot race on the shared counters / tag set.
        for d in batch_results:
            for k, v in d.stats.items():
                stats[k] = stats.get(k, 0) + v
            consolidated_tags.update(d.tags)

        if any_cancelled:
            return {"status": "cancelled", "bank_id": bank_id, **stats}

        # Update round budget after processing this DB fetch batch
        if round_limit_enabled:
            round_remaining -= len(memories)
            if round_remaining <= 0:
                hit_round_limit = True
                break

    # Re-submit consolidation if we hit the round limit and there's likely more work.
    # Any failure here must propagate: swallowing it (the prior behavior) leaves the
    # bank with backlog and no queued work — silently stuck — because the outer op
    # gets marked completed in the success path. Letting the exception bubble up to
    # execute_task's retry handler means the op is retried with backoff; on retry the
    # consolidator skips already-consolidated rows via the consolidated_at filter and
    # picks up the remainder. Issue #1842.
    # The affected-tag union for the whole round-limited chain. Refresh fires once, when
    # the backlog has fully drained (the final round), not once per round — a model's
    # memories can straddle rounds, and gating on the final round alone (the prior
    # behaviour) dropped every model consolidated earlier because the final round's tags
    # no longer named them (#3411). The union is durable: each batch writes its tags into
    # the op's ``task_payload`` in its own transaction (crash-safe), and the re-queue threads
    # the accumulated set forward to the next round. Prefer that durable
    # value; fall back to the in-memory union when there is no backing op (a direct
    # ``run_consolidation_job`` call, e.g. in tests).
    all_refresh_tags = set(pending_refresh_tags or []) | consolidated_tags
    if operation_id:
        all_refresh_tags |= await _read_pending_refresh_tags(pool, operation_id)

    if hit_round_limit:
        remaining = total_count - stats["memories_processed"]
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} hit round limit of {max_memories_per_round} memories,"
            f" ~{remaining} remaining. Re-queuing consolidation."
        )
        await memory_engine.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            observation_scopes=observation_scopes,
            pending_refresh_tags=sorted(all_refresh_tags) or None,
        )

    # Build summary
    perf.log(
        f"[3] Results: {stats['memories_processed']} memories -> "
        f"{stats['actions_executed']} actions "
        f"({stats['observations_created']} created, "
        f"{stats['observations_updated']} updated, "
        f"{stats['observations_merged']} merged, "
        f"{stats['skipped']} skipped)"
    )

    # Add timing breakdown. Each phase is recorded once per call, so the count
    # disambiguates a single slow call from many fast calls — important for
    # operators triaging "the recall phase took 15s" log lines, where the
    # total is the sum of many serial sub-calls rather than one slow query.
    def _fmt(key: str) -> str:
        total = perf.timings[key]
        count = perf.timing_counts.get(key, 0)
        if count > 1:
            avg_ms = total * 1000.0 / count
            return f"{key}={total:.3f}s ({count} calls, avg={avg_ms:.0f}ms)"
        return f"{key}={total:.3f}s"

    timing_parts = []
    for key in ("recall", "llm", "embedding", "db_write"):
        if key in perf.timings:
            timing_parts.append(_fmt(key))

    if perf.llm_calls > 0:
        timing_parts.append(f"avg_obs={perf.total_obs_in_context / perf.llm_calls:.1f}")
        timing_parts.append(f"avg_prompt_tokens=~{perf.total_prompt_chars / perf.llm_calls / 4:.0f}")

    if timing_parts:
        perf.log(f"[4] Timing breakdown: {', '.join(timing_parts)}")

    # A run whose LLM calls kept failing schema validation looks exactly like a clean one
    # from the counters above: adaptive bisection rescues the facts, so nothing is left
    # carrying `consolidation_failed_at`, while everything those responses would have done
    # -- notably their deletes -- was thrown away (#4151, #4152). Say so, loudly, and only
    # when it happened, so a healthy summary is unchanged.
    stats["llm_batch_failures"] = perf.llm_batch_failures
    if perf.llm_batch_failures:
        # Attempts, not batches: one batch call can burn up to `consolidation_max_attempts`
        # of them, so this can exceed the batch count above rather than being a share of it.
        perf.log(
            f"[5] WARNING: {perf.llm_batch_failures} LLM batch attempt(s) failed and their responses were "
            f"discarded (creates, updates AND deletes alike). Facts the retry/bisection path rescued are NOT "
            f"reflected in failed_consolidation; see hindsight.consolidation.batch_failures for the "
            f"per-class breakdown."
        )
    stats["unresolved_actions"] = perf.unresolved_actions
    if lifecycle_run is not None and lifecycle_run.counts:
        # Only when the bank runs the claim lifecycle and something happened: an off-mode result
        # carries no new key.
        lifecycle_stats = {f"lifecycle_{name}": count for name, count in sorted(lifecycle_run.counts.items())}
        stats.update(lifecycle_stats)
        perf.log(f"[7] Lifecycle: {lifecycle_stats}")
    if perf.unresolved_actions:
        perf.log(
            f"[6] WARNING: {perf.unresolved_actions} observation create/update action(s) cited no fact id from "
            f"their batch and were discarded; their facts are counted as skipped."
        )

    # Trigger mental-model refreshes once, when the chain has fully drained. On a
    # round-limited round we skip and carry the affected tags forward (above); the
    # final round flushes the accumulated union, so a model whose memories were
    # consolidated in ANY round is refreshed exactly once — deduplicated, not dropped
    # (#3411). Each model is still refreshed at most once per drain: a strict tagged
    # model appears once in the trigger's candidate query regardless of how many rounds
    # its tag spanned.
    if hit_round_limit:
        stats["mental_models_refreshed"] = 0
        logger.info(
            f"[CONSOLIDATION] bank={bank_id} deferring mental model refresh to the final round "
            f"(round limit hit; carrying {len(all_refresh_tags)} tags forward)"
        )
    else:
        set_stage("consolidation.refreshing_mental_models")
        await memory_engine._write_operation_progress(
            operation_id,
            stage="refreshing_mental_models",
            processed=stats["memories_processed"],
            total=await _progress_total(stats["memories_processed"]),
        )
        # SECURITY: Only refresh mental models whose scope covers what was consolidated
        mental_models_refreshed = await _trigger_mental_model_refreshes(
            memory_engine=memory_engine,
            bank_id=bank_id,
            request_context=request_context,
            consolidated_tags=sorted(all_refresh_tags) or None,
            perf=perf,
        )
        stats["mental_models_refreshed"] = mental_models_refreshed

    perf.flush()

    return {"status": "completed", "bank_id": bank_id, **stats}


# SQL predicate: "an untagged write can make this mental model stale".
#
# A model's scope is NOT its ``tags`` column — it is whatever
# ``_mental_model_stale_scope`` resolves from it and the trigger. Two cases reach
# untagged memories:
#   - no tags at all             -> no tag constraint, every bank memory is in scope
#   - trigger.tag_groups         -> overrides the tags column entirely, and a group can
#                                   select untagged rows on purpose (``not``, an empty
#                                   ``exact`` leaf)
# Gating on the tags column alone starved both, plus tagged "any"/"all" models (#3053).
# Those tagged models are excluded again since #4857: their refresh still *reads*
# untagged memories, but staleness counts only writes carrying their tags, so an
# untagged-only consolidation can never make them stale and asking would be wasted.
_MM_SCOPE_REACHES_UNTAGGED = "((tags IS NULL OR tags = '{}') OR trigger ? 'tag_groups')"


async def _trigger_mental_model_refreshes(
    memory_engine: "MemoryEngine",
    bank_id: str,
    request_context: "RequestContext",
    consolidated_tags: list[str] | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> int:
    """
    Trigger refreshes for mental models with refresh_after_consolidation=true.

    SECURITY: Only triggers refresh for mental models whose refresh scope can contain
    what this consolidation touched, preventing unnecessary refreshes across security
    boundaries.

    Args:
        memory_engine: MemoryEngine instance
        bank_id: Bank identifier
        request_context: Request context for authentication
        consolidated_tags: Tags of the memories that were consolidated. None means only
            untagged memories were consolidated (or nothing was), so only models whose
            scope reaches untagged memories are candidates.
        perf: Performance logging

    Returns:
        Number of mental models scheduled for refresh
    """
    pool = memory_engine._backend

    # Find mental models with refresh_after_consolidation=true that are actually stale.
    # The tag predicate on the SELECT is a cheap prefilter that skips models this
    # consolidation cannot have affected; compute_mental_model_is_stale then verifies
    # against the model's *resolved* scope that new memories really were ingested since
    # its last refresh.
    async with acquire_with_retry(pool) as conn:
        if consolidated_tags:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, last_memory_seen_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND (
                    (tags IS NOT NULL AND tags != '{{}}' AND tags && $2::varchar[])
                    OR {_MM_SCOPE_REACHES_UNTAGGED}
                  )
                """,
                bank_id,
                consolidated_tags,
            )
        else:
            candidates = await conn.fetch(
                f"""
                SELECT id, name, tags, last_refreshed_at, last_memory_seen_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1
                  AND (trigger->>'refresh_after_consolidation')::boolean = true
                  AND {_MM_SCOPE_REACHES_UNTAGGED}
                """,
                bank_id,
            )

        rows = []
        for candidate in candidates:
            if await memory_engine.compute_mental_model_is_stale(conn, bank_id, candidate):
                rows.append(candidate)

    if not rows:
        return 0

    if perf:
        if consolidated_tags:
            perf.log(
                f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true "
                f"(filtered by tags: {consolidated_tags})"
            )
        else:
            perf.log(f"[5] Triggering refresh for {len(rows)} mental models with refresh_after_consolidation=true")

    # Submit refresh tasks for each mental model
    refreshed_count = 0
    for row in rows:
        mental_model_id = row["id"]
        try:
            # skip_if_in_flight: a consolidation chain fires this every round and
            # overlapping consolidations can run on the same bank, so a model still
            # pending/processing a refresh must not be enqueued a second time (#3411).
            result = await memory_engine.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
                skip_if_in_flight=True,
                automatic=True,
            )
            if result.get("paused"):
                logger.info(
                    f"[CONSOLIDATION] Skipped refresh for mental model {mental_model_id}: its last refresh failed"
                )
                continue
            refreshed_count += 1
            logger.info(
                f"[CONSOLIDATION] Triggered refresh for mental model {mental_model_id} "
                f"(name: {row['name']}) in bank {bank_id}"
            )
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to trigger refresh for mental model {mental_model_id}: {e}")

    return refreshed_count


async def _process_memory_batch(
    pool: DatabaseBackend,
    memory_engine: "MemoryEngine",
    llm_config: Any,
    bank_id: str,
    memories: list[dict[str, Any]],
    request_context: "RequestContext",
    perf: ConsolidationPerfLog | None = None,
    config: Any = None,
    obs_tags_override: list[str] | None = None,
    mark_consolidated_ids: list[Any] | None = None,
    adaptive_settings: "AdaptiveSettings | None" = None,
    lifecycle: _BatchLifecycle | None = None,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Process a batch of memories in a single LLM call.

    Steps:
    1. Parallel recalls — one per fact (read-only; safe to parallelise)
    2. Union of retrieved observations across the batch (deduped by id)
    3. Single LLM call with all N facts + unioned observations
    4. Prepare every action connection-free, then apply them in ONE transaction
    5. Returns one result dict per memory, in the same order as `memories`

    Per-fact security: action execution validates each learning_id against the
    observations that were recalled specifically for that fact, so cross-tag
    updates cannot occur.

    Args:
        obs_tags_override: When set, use these tags for observation recall and
            create/update instead of the memory's own tags. This enables multi-pass
            consolidation where a single memory can contribute to observations
            scoped at different tag levels (e.g., user-level vs session-level).
        mark_consolidated_ids: Source facts to stamp ``consolidated_at`` inside this
            batch's write transaction, so the stamps share the fate of the observations
            derived from the same LLM response (#3876). None on a pass that is not the
            final one for these memories — the caller stamps once, on the last pass.
        adaptive_settings: Set when the bank reinforces (rank/lifecycle, reinforcement on):
            the observations this batch creates, updates and folds into are reinforced by
            their new evidence, in a savepoint of the batch's transaction. The facts then
            carry their provenance (``_fetch_unconsolidated_rows(with_provenance=True)``).
        lifecycle: Set when the bank runs the claim lifecycle. In lifecycle mode the pool shows
            retired observations labelled, the prompt and schema are the lifecycle variants, and the
            response's labels become claim transitions inside the batch's transaction (see
            ``lifecycle``); in observe mode a plain delete also records the retirements it would have
            been (shadow rows). The retired observations' tags are handed back on it.
    """
    # Map the source memories this batch consumes onto the consolidation trace.
    record_source_memory_ids([str(m["id"]) for m in memories])

    # Determine effective tag scope for observations.
    # When obs_tags_override is set, use it; otherwise use the memory's own tags.
    if obs_tags_override is not None:
        fact_tags = obs_tags_override
    else:
        # All memories in the batch share the same tag set (enforced by batching)
        fact_tags = memories[0].get("tags") or [] if memories else []

    # Everything below — the related-observation recall, the cap, the prompt — reads
    # the config as this scope sees it, so a consolidation strategy claiming the
    # scope applies to the whole pass. Resolved before the recall because the recall
    # applies the source-facts token limits a strategy may override.
    config = _config_for_scope(config, fact_tags)
    lifecycle_mode = lifecycle is not None and lifecycle.run.lifecycle
    scope_tags = frozenset(fact_tags)
    # Passed only in lifecycle mode, so every other mode calls the recall and dedup seams exactly as before.
    lifecycle_kwargs: dict[str, Any] = {"lifecycle": True} if lifecycle_mode else {}
    dedup_kwargs: dict[str, Any] = {"exclude_non_active": True} if lifecycle_mode else {}

    # 1. Parallel recalls — one per fact
    # When obs_tags_override is set, use it as the observation scope for all facts.
    t0 = time.perf_counter()
    observation_scope_tags = obs_tags_override if obs_tags_override is not None else None
    recall_tasks = [
        _find_related_observations(
            memory_engine=memory_engine,
            bank_id=bank_id,
            query=m["text"],
            request_context=request_context,
            tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
            config=config,
            **lifecycle_kwargs,
        )
        for m in memories
    ]
    # A failed recall must fail the batch rather than degrade to "no related
    # observations": proceeding with an empty candidate set would hide an
    # existing twin from the LLM and turn an UPDATE into a duplicate CREATE.
    # The batch's memories stay unconsolidated and are picked up on retry.
    per_fact_recalls = await _gather_or_cancel(recall_tasks)
    lifecycle_pool: dict[str, claim_lifecycle.PoolEntry] | None = None
    if lifecycle_mode:

        def live_recall(index: int):
            m = memories[index]
            return _find_related_observations(
                memory_engine=memory_engine,
                bank_id=bank_id,
                query=m["text"],
                request_context=request_context,
                tags=observation_scope_tags if observation_scope_tags is not None else (m.get("tags") or []),
                config=config,
                lifecycle=True,
                live_only=True,
            )

        shaped = await _shape_lifecycle_pools(pool, bank_id, per_fact_recalls, config, scope_tags, live_recall)
        per_fact_recalls, lifecycle_pool = shaped.recalls, shaped.entries
    if perf:
        perf.record_timing("recall", time.perf_counter() - t0)

    # Union all observations (deduped by id)
    seen_ids: set[str] = set()
    union_observations: list["MemoryFact"] = []
    union_source_facts: dict[str, "MemoryFact"] = {}
    for recall_result in per_fact_recalls:
        for obs in recall_result.results:
            obs_id = str(obs.id)
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                union_observations.append(obs)
        if recall_result.source_facts:
            union_source_facts.update(recall_result.source_facts)

    # 2b. Compute remaining observation slots for this scope (if limit configured).
    # The cap is resolved per-scope: an observation_scope_limits rule may override
    # the bank-wide max_observations_per_scope for scopes matching its tag pattern.
    max_obs = _effective_scope_limit(config, fact_tags)
    remaining_observation_slots: int | None = None
    if max_obs >= 0 and fact_tags:
        # max_obs == 0 means "no new observations": there are no slots regardless
        # of the current count, so skip the count query for that case.
        current_count = 0
        if max_obs > 0:
            async with acquire_with_retry(pool) as count_conn:
                current_count = (
                    # Lifecycle mode: a retired observation no longer holds a slot.
                    await get_memories().count_active_observations(
                        count_conn, bank_id=bank_id, tags=fact_tags, tags_match="all_strict"
                    )
                    if lifecycle_mode
                    else await _count_observations_for_scope(count_conn, bank_id, fact_tags)
                )
        remaining_observation_slots = max(max_obs - current_count, 0)
        if remaining_observation_slots == 0:
            logger.info(
                f"[CONSOLIDATION] bank={bank_id} scope={fact_tags} at observation limit "
                f"({current_count}/{max_obs}), only updates/deletes allowed"
            )

    # 2c. Triage: restatements and chatter are settled by a typed judgment, not the LLM.
    triage = TriageResult()
    if (decision_model := triage_model(config)) is not None:
        triage_memories, triage_candidates = list(memories), [list(r.results) for r in per_fact_recalls]
        if lifecycle_pool is not None:
            # A fact whose candidates include a retired observation may be old news about it, which
            # only the lifecycle prompt can label: triage must not settle it as new or a restatement.
            pairs = [
                (m, c)
                for m, c in zip(triage_memories, triage_candidates)
                if not any(lifecycle_pool[str(o.id)].retired for o in c[:MAX_CANDIDATES] if str(o.id) in lifecycle_pool)
            ]
            triage_memories, triage_candidates = [m for m, _ in pairs], [c for _, c in pairs]
        if triage_memories:
            async with acquire_with_retry(pool) as conn:
                entities_by_fact = await get_memories().entities_for_units(
                    conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(m["id"]) for m in triage_memories]
                )
            triage = await triage_facts(
                triage_memories,
                triage_candidates,
                decision_model,
                entities_by_fact=entities_by_fact,
                mission=(config.observations_mission or "").strip(),
                may_create=remaining_observation_slots is None,
            )
    llm_memories = [m for m in memories if not triage.settled(str(m["id"]))]

    # 3. Single LLM call — skipped when triage settled every fact.
    t0 = time.perf_counter()
    llm_result = (
        await _consolidate_batch_with_llm(
            llm_config=llm_config,
            memories=llm_memories,
            union_observations=union_observations,
            union_source_facts=union_source_facts,
            config=config,
            remaining_observation_slots=remaining_observation_slots,
            max_observations_per_scope=max_obs,
            lifecycle_pool=lifecycle_pool,
        )
        if llm_memories
        else _BatchLLMResult()
    )
    if perf and llm_memories:
        perf.record_timing("llm", time.perf_counter() - t0)
        perf.record_llm_call(llm_result.obs_count, llm_result.prompt_chars)
        perf.record_llm_batch_failures(llm_result.failed_attempts)
    lifecycle_plan: claim_lifecycle.LifecyclePlan | None = None
    lifecycle_drops: claim_lifecycle.Drops | None = None
    if lifecycle is not None and lifecycle_pool is not None:
        # Ids first: the labels and the triage redirect below must see the observations they name.
        _resolve_miscopied_ids(
            llm_result,
            {str(obs.id) for obs in union_observations},
            {s for e in lifecycle_pool.values() for s in e.source_ids},
        )
        _apply_triage(llm_result, triage, union_observations, retiring=_retirements_named(llm_result, lifecycle_pool))
        lifecycle_drops = claim_lifecycle.Drops(bank_id=bank_id)
        lifecycle_plan = claim_lifecycle.plan_lifecycle(
            llm_result,
            claim_lifecycle.PlanInputs(
                pool=lifecycle_pool,
                facts={str(m["id"]): m for m in memories},
                scope_tags=scope_tags,
                settings=lifecycle.run.settings,
                now=adaptive_clock.utcnow(),
                target_sources=await _lifecycle_target_sources(pool, bank_id, llm_result, lifecycle_pool),
            ),
            lifecycle_drops,
            new_create=lambda text, fact_ids, reason: _CreateAction(text=text, source_fact_ids=fact_ids, reason=reason),
        )
        _cap_lifecycle_creates(llm_result, lifecycle_plan, remaining_observation_slots, max_obs)
    else:
        _apply_triage(llm_result, triage, union_observations)

    # 4. Prepare every action connection-free, then apply them all in ONE transaction.
    #
    # Before #3876 each action wrote on its own connection the moment it was decided:
    # deletes first (to free observation slots), then updates, then creates. A batch whose
    # DELETE succeeded and whose replacement CREATE then failed — a raising embedder, a
    # cancelled sibling, a create silently dropped because the model cited a fact id that
    # was not in this batch — left the observation deleted and nothing in its place, while
    # the source facts were still stamped ``consolidated_at`` and so were never
    # re-consolidated. The knowledge was gone with the operation still reporting
    # ``completed``. The writes derived from one LLM response are now all-or-nothing, and
    # the ``consolidated_at`` stamps for the facts that response consumed commit with them.
    #
    # The transaction must therefore contain no LLM or embedder call: each action is first
    # prepared (source facts resolved and security-checked, text embedded, semantic-dedup
    # verdict adjudicated) with no connection held, and only then written.
    per_memory_created: set[str] = set()
    per_memory_updated: set[str] = set()

    mem_by_id = {str(m["id"]): m for m in memories}

    # Semantic dedup: when enabled, an observation that is >= the threshold cosine to a DIFFERENT
    # existing observation is reconciled by a focused 1-by-1 LLM merge (anchored on the observation
    # text, not the source fact). It runs on both CREATE (a near-dup emitted despite the twin being
    # in context — weak-model failure mode) and UPDATE (a rewrite+re-embed that drifts an existing
    # observation into a twin — the create-time guard can't see this). The trace operation/scope is
    # "consolidation_dedup" (routes through the consolidation concurrency bucket via llm_wrapper's
    # "consolidation" prefix; recorded distinctly in llm_requests).
    dedup_enabled = _dedup_active(config)
    dedup_llm_config = (
        memory_engine._consolidation_llm_config.with_config(config, bank_id=bank_id, operation="consolidation_dedup")
        if dedup_enabled
        else None
    )

    _resolve_miscopied_ids(llm_result, {str(obs.id) for obs in union_observations})

    # --- Prepare: deletes -----------------------------------------------------
    # Deletes carry no slow work; validating them here keeps the whole decision set in
    # one place. They are still applied first inside the transaction, so an observation
    # slot freed by a delete is available to a create in the same response.
    prepared_deletes: list[_PreparedDelete] = []
    pool_ids = {str(obs.id) for obs in union_observations}
    # A response can both UPDATE and DELETE one observation (seen when a fact supersedes it:
    # the model rewrites it to the new state and also deletes "the old one"). Deletes apply
    # first, so the update then found no row and was skipped while its source facts were
    # still stamped consolidated — the new state was lost for good. The update carries the
    # newer knowledge, so it wins.
    updated_ids = {u.observation_id for u in llm_result.updates}
    for delete in llm_result.deletes:
        if delete.observation_id in updated_ids:
            logger.warning(
                f"Batch consolidation: dropped delete of observation {delete.observation_id} — "
                "the same response updates it"
            )
            continue
        # Security: the observation must be present in the unioned recall
        if delete.observation_id not in pool_ids:
            logger.debug(
                f"Batch consolidation: rejected delete — observation {delete.observation_id} not in unioned recall"
            )
            continue
        # A merge target must be one the model was shown, like any other target it names.
        target = delete.merged_into if delete.merged_into in pool_ids - {delete.observation_id} else None
        prepared_deletes.append(
            _PreparedDelete(observation_id=delete.observation_id, is_merge=bool(delete.merged_into), merged_into=target)
        )

    # --- Prepare: updates -----------------------------------------------------
    # Each action is prepared on its own: a liveness preflight, an embedding and, with semantic
    # dedup on, a nearest-observation probe and possibly a focused LLM verdict. None of it depends
    # on another action's preparation — the probes read the state the batch started from, which is
    # also why two near-twin CREATEs in one response can both land (see below) — so the actions are
    # prepared concurrently. Done one after another, a response with many actions paid every
    # round trip and every dedup verdict in sequence. Order is preserved: the lists below keep the
    # response's order, which the apply step relies on.
    prepare_slots = asyncio.Semaphore(_PREPARE_CONCURRENCY)

    # Every action's sources are facts of this batch, so ONE liveness read answers the preflight
    # for all of them, instead of one read per action.
    live_before_prepare = await _live_source_ids(pool, bank_id, [m["id"] for m in memories])

    async def _bounded(coro: Any) -> Any:
        async with prepare_slots:
            return await coro

    async def _prepare_update(update: _UpdateAction) -> _PreparedUpdate | None:
        source_mems = [mem_by_id[fid] for fid in update.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            _log_unresolved_action("update", update.source_fact_ids, update.text, perf)
            return None
        # Security: the observation must be one the model was shown, the pooled recall already
        # confined to this bank and scope, exactly as for a delete. Previously it also had to be
        # recalled for one of the cited facts, but the model reads the whole pool: attaching a
        # fact to an observation a sibling fact surfaced is legitimate, and rejecting it dropped
        # the fact's evidence silently (it is stamped consolidated all the same).
        model = next((m for m in union_observations if str(m.id) == update.observation_id), None)
        if model is None:
            logger.warning(
                f"Batch consolidation: rejected update — observation {update.observation_id} not in the recall pool"
            )
            return None
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        source_memory_ids = [m["id"] for m in source_mems]
        # Preflight: if every source memory is already gone, skip BEFORE the slow embed. The write
        # itself re-checks liveness under FOR SHARE.
        if not any(str(mid) in live_before_prepare for mid in source_memory_ids):
            logger.debug(
                f"Update skipped: all {len(source_memory_ids)} source memories for observation "
                f"{update.observation_id} were deleted before embedding"
            )
            return None
        embedding_str = await _embed_observation_text(memory_engine, update.text, perf)
        prepared = _PreparedUpdate(
            update=update,
            model=model,
            source_mems=source_mems,
            source_memory_ids=source_memory_ids,
            source_fact_tags=agg.tags,
            source_bounds=_TemporalBounds.of(agg),
            embedding_str=embedding_str,
        )
        # Reconcile the rewritten observation against its neighbours: the re-embed may have
        # drifted it into a near-twin of another existing observation (the residual-duplicate
        # source). Adjudicated here, folded inside the transaction below.
        if dedup_enabled and embedding_str is not None:
            prepared.dedup = await _dedup_adjudicate(
                pool,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                update.text,
                embedding_str,
                agg.tags,
                exclude_id=update.observation_id,
                **dedup_kwargs,
            )
        return prepared

    prepared_updates: list[_PreparedUpdate] = [
        p for p in await _gather_or_cancel([_bounded(_prepare_update(u)) for u in llm_result.updates]) if p is not None
    ]

    # --- Prepare: creates -----------------------------------------------------
    # Deterministic dedup guard: map the observations the LLM was SHOWN by their
    # normalised text. The model intermittently emits a CREATE whose text is identical
    # to an observation already in its context (over-aggregation / incoherence — it even
    # UPDATEs the twin and creates a sibling). When that happens we drop the duplicate
    # CREATE instead of inserting a redundant row. No extra LLM/embedding cost — the
    # match is exact text against the in-memory set.
    shown_obs_by_text = {_norm_obs_text(o.text): o for o in union_observations}
    # Also collapse a CREATE that reproduces the text of an UPDATE issued in the SAME
    # response (the model occasionally UPDATEs the twin to text X and also CREATEs X).
    update_texts = {_norm_obs_text(u.text) for u in llm_result.updates if u.text}

    async def _prepare_create(create: _CreateAction) -> _PreparedCreate | None:
        source_mems = [mem_by_id[fid] for fid in create.source_fact_ids if fid in mem_by_id]
        if not source_mems:
            _log_unresolved_action("create", create.source_fact_ids, create.text, perf)
            return None
        agg = _aggregate_source_fields(source_mems, tags=fact_tags)
        create_source_ids = [m["id"] for m in source_mems]
        plan = lifecycle_plan.for_create(create) if lifecycle_plan is not None else None
        labelled = plan is not None and plan.labelled
        if plan is not None and lifecycle_pool is not None:
            create_source_ids = _lifecycle_create_sources(plan, create_source_ids)

        # Reconcile against observations shown to the LLM: an exact-text match means
        # this CREATE reproduces verbatim an observation the model already had in context.
        # Since that observation already carries this exact text, drop the duplicate CREATE
        # — no row is inserted, nothing is lost. We deliberately do NOT also UPDATE the twin
        # here: the LLM frequently UPDATEd it earlier in this same batch, and a second update
        # would run off the pre-LLM snapshot and clobber that change (see _dedupe_updates).
        # A labelled create (a successor, a split facet) is a new claim by construction: it is
        # never collapsed into an existing observation, by exact text or by semantic dedup.
        duplicate_of = None if labelled else _duplicate_create_target(create.text, shown_obs_by_text, update_texts)
        if duplicate_of is not None:
            logger.warning(
                "[CONSOLIDATION] dropped duplicate observation CREATE — verbatim match of %s; llm_reason=%r",
                duplicate_of,
                create.reason or "(none given)",
            )
            return None

        if not any(str(mid) in live_before_prepare for mid in create_source_ids):
            logger.debug(f"Create skipped: all {len(create_source_ids)} source memories were deleted before embedding")
            return None
        embedding_str = await _embed_observation_text(memory_engine, create.text, perf)
        prepared_create = _PreparedCreate(
            text=create.text,
            source_mems=source_mems,
            source_memory_ids=create_source_ids,
            source_fact_tags=agg.tags,
            agg=agg,
            embedding_str=embedding_str,
            lifecycle_plan=plan,
        )
        # Semantic near-duplicate reconciliation: merge this CREATE into an existing
        # near-identical observation (LLM-adjudicated, 1-by-1) instead of inserting a dup.
        # The probe reads the state the batch started from, so two near-twin CREATEs in ONE
        # response can both land; the next round's probe sees them and folds them, and the
        # exact-text guard above already covers the common case.
        if dedup_enabled and not labelled:
            prepared_create.dedup = await _dedup_adjudicate(
                pool,
                memory_engine,
                bank_id,
                config,
                dedup_llm_config,
                create.text,
                embedding_str,
                agg.tags,
                exclude_id=None,
                **dedup_kwargs,
            )
        return prepared_create

    prepared_creates: list[_PreparedCreate] = [
        p for p in await _gather_or_cancel([_bounded(_prepare_create(c)) for c in llm_result.creates]) if p is not None
    ]

    # --- Prepare: reinforcement evidence --------------------------------------
    # The survivors' existing sources and their provenance, read before the transaction like the
    # liveness preflight above, so the write holds its locks no longer than it did.
    evidence: ObservationEvidence | None = None
    if adaptive_settings is not None and (prepared_updates or prepared_creates or prepared_deletes):
        evidence = await _prepare_observation_evidence(
            pool, bank_id, adaptive_settings, memories, prepared_updates, prepared_creates, prepared_deletes
        )

    # --- Apply: one transaction for everything this LLM response decided ------
    deleted_count = 0
    # A failed LLM call yields no actions, and its memories must NOT be stamped: the caller
    # bisects and retries them, and a stamp would exclude them from pending consolidation
    # for good. With neither writes nor stamps there is nothing to open a transaction for.
    stamp_ids = list(mark_consolidated_ids or []) if not llm_result.failed else []
    lifecycle_only_writes = lifecycle_plan is not None and bool(
        lifecycle_plan.historical or lifecycle_plan.plain_retires
    )
    observe_shadow = lifecycle is not None and lifecycle.run.observe
    claim_state: claim_lifecycle.ApplyState | None = None
    merged_retentions: list[tuple[str, str]] = []
    if prepared_deletes or prepared_updates or prepared_creates or stamp_ids or lifecycle_only_writes:
        # One store batch for everything this response decided. The SQL transaction already makes
        # these writes all-or-nothing for a store that keeps its rows in SQL; a store whose every
        # write is its own atomic call needs the batch to say the same thing — otherwise it applies
        # them one by one, and a failure part-way leaves the deletes applied and the creates not.
        # Committed INSIDE the transaction, so a batch that fails to apply rolls the SQL side back
        # with it.
        write_batch = await get_memories().begin_write_batch(bank_id=bank_id)
        try:
            async with acquire_with_retry(pool) as conn:
                async with conn.transaction():
                    if lifecycle_plan is not None and lifecycle_drops is not None:
                        # Lifecycle mode: the observations' rows, then ONE sorted pass over every
                        # existing claim this response may touch, before anything else is locked —
                        # the order the sweep, the delete paths and PATCH take them in.
                        claim_state = claim_lifecycle.ApplyState(
                            bank_id=bank_id,
                            store=get_memories(),
                            fq_table=fq_table,
                            scope_tags=scope_tags,
                            now=adaptive_clock.utcnow(),
                            drops=lifecycle_drops,
                        )
                        await _lock_lifecycle_claims(
                            conn, claim_state, lifecycle_plan, prepared_deletes, prepared_updates, prepared_creates
                        )
                    changed_ids = await _sources_changed_since_read(conn, bank_id, memories)
                    if changed_ids:
                        # Drop the whole response, stamps included: the facts stay pending and the
                        # job's next fetch re-reads them as they are now (#4831).
                        logger.info(
                            f"[CONSOLIDATION] bank={bank_id} discarding batch of {len(memories)}: "
                            f"{len(changed_ids)} source fact(s) edited since read, e.g. {changed_ids[0]}"
                        )
                        prepared_deletes, prepared_updates, prepared_creates, stamp_ids = [], [], [], []
                        lifecycle_plan = claim_lifecycle.LifecyclePlan() if lifecycle_plan is not None else None

                    # Each action below reads its sources (liveness, entities) and, for an update,
                    # the observation it rewrites — one read at a time. Naming them all first lets
                    # a store whose reads are round trips fetch them together.
                    await write_batch.prefetch(
                        list(
                            dict.fromkeys(
                                [str(mid) for p in prepared_updates for mid in p.source_memory_ids]
                                + [p.update.observation_id for p in prepared_updates]
                                + [str(mid) for p in prepared_creates for mid in p.source_memory_ids]
                            )
                        )
                    )

                    if observe_shadow and lifecycle is not None:
                        # Before the deletes: what lifecycle mode would have retired instead.
                        await _record_shadow_retirements(
                            conn, memory_engine, bank_id, lifecycle.run, prepared_deletes, memories, union_observations
                        )

                    merged_away: list[_MergedAway] = []
                    for prepared_delete in prepared_deletes:
                        if prepared_delete.is_merge:
                            # Read before deleting: its sources and dates move to the survivor.
                            gone = await get_memories().get_memories(
                                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[prepared_delete.observation_id]
                            )
                            if gone:
                                merged_away.append(_MergedAway(gone=gone[0], target_id=prepared_delete.merged_into))
                        if claim_state is not None:
                            # Lifecycle mode keeps the row: a merge retires it once its evidence moved.
                            continue
                        await _execute_delete_action(
                            conn=conn, bank_id=bank_id, observation_id=prepared_delete.observation_id
                        )
                        deleted_count += 1

                    for prepared in prepared_updates:
                        updated_emb_str = await _apply_update_action(
                            conn=conn,
                            memory_engine=memory_engine,
                            bank_id=bank_id,
                            prepared=prepared,
                            perf=perf,
                        )
                        if updated_emb_str is None:
                            # Skipped inside the write (sources or the observation vanished).
                            continue
                        for m in prepared.source_mems:
                            per_memory_updated.add(str(m["id"]))
                        folded_into_twin = False
                        if prepared.dedup is not None:
                            folded_into_twin = await _apply_dedup_update_fold(
                                conn,
                                memory_engine,
                                bank_id,
                                config,
                                prepared.dedup,
                                prepared.update.observation_id,
                                prepared.update.text,
                                claim_state=claim_state,
                            )
                            if folded_into_twin and claim_state is not None and prepared.dedup.best_id:
                                merged_retentions.append((prepared.dedup.best_id, prepared.update.observation_id))
                        if evidence is not None:
                            if folded_into_twin and prepared.dedup is not None and prepared.dedup.best_id:
                                # The rewritten observation is gone; its twin now carries every source.
                                evidence.folded(
                                    prepared.dedup.best_id,
                                    [*(prepared.model.source_fact_ids or []), *prepared.source_memory_ids],
                                )
                            else:
                                evidence.attached(
                                    prepared.update.observation_id,
                                    prepared.source_memory_ids,
                                    unchanged_text=_norm_obs_text(prepared.update.text)
                                    == _norm_obs_text(prepared.model.text),
                                    previous_mentioned_at=prepared.model.mentioned_at,
                                )

                    for prepared_create in prepared_creates:
                        if prepared_create.dedup is not None and (
                            claim_state is None
                            or prepared_create.dedup.best_id is None
                            or claim_state.live(prepared_create.dedup.best_id)
                        ):
                            merged_into = await _apply_dedup_create_fold(
                                conn,
                                memory_engine,
                                bank_id,
                                prepared_create.dedup,
                                prepared_create.source_memory_ids,
                                _TemporalBounds.of(prepared_create.agg),
                            )
                            if merged_into is not None:
                                logger.info(
                                    "[CONSOLIDATION] dedup-merged observation CREATE into %s (cosine>=%.2f)",
                                    merged_into[:8],
                                    config.consolidation_dedup_threshold,
                                )
                                for m in prepared_create.source_mems:
                                    per_memory_created.add(str(m["id"]))
                                if evidence is not None:
                                    evidence.folded(merged_into, prepared_create.source_memory_ids)
                                continue

                        action = await _apply_create_action(
                            conn=conn,
                            memory_engine=memory_engine,
                            bank_id=bank_id,
                            prepared=prepared_create,
                            perf=perf,
                        )
                        # Count a memory as created only when an observation was actually written (the
                        # source-liveness recheck inside the write can skip it).
                        if action == "created":
                            for m in prepared_create.source_mems:
                                per_memory_created.add(str(m["id"]))
                            if evidence is not None and prepared_create.observation_id is not None:
                                evidence.created(prepared_create.observation_id, prepared_create.source_memory_ids)

                    # After the response's own updates, so the survivor is extended as rewritten.
                    for merged in merged_away:
                        if claim_state is not None and not await _merge_precheck(
                            conn, claim_state, merged.gone.unit_id, merged.target_id
                        ):
                            continue
                        folded = await _fold_deleted_evidence(
                            conn,
                            memory_engine,
                            bank_id,
                            merged.gone,
                            merged.target_id,
                            requeue_on_miss=claim_state is None,
                        )
                        if folded and claim_state is not None and merged.target_id is not None:
                            if await _merge_retire(conn, claim_state, merged.gone.unit_id, merged.target_id):
                                merged_retentions.append((merged.target_id, merged.gone.unit_id))
                        if folded and evidence is not None and merged.target_id is not None:
                            evidence.folded(merged.target_id, merged.gone.source_memory_ids)

                    if claim_state is not None and lifecycle_plan is not None:
                        attached = await _apply_lifecycle_transitions(
                            conn, claim_state, lifecycle_plan, prepared_creates, memories
                        )
                        per_memory_updated.update(attached)
                        created_ids = [p.observation_id for p in prepared_creates if p.observation_id]
                        if created_ids:
                            # A successor rebuilt under a new id (its old one deleted by an edit's sweep)
                            # takes over the retired claims its sources carry the evidence of.
                            await get_memories().relink_rebuilt_successors(
                                conn,
                                bank_id=bank_id,
                                new_observation_ids=created_ids,
                                now=claim_state.now,
                                locked_claim_ids=claim_state.locked,
                            )

                    # After every write it accounts for, before the stamps: a failure rolls back its
                    # own savepoint and nothing else.
                    if evidence is not None:
                        await _record_observation_reinforcement(conn, bank_id, evidence)
                    if merged_retentions and lifecycle is not None:
                        await _merge_retentions(conn, bank_id, lifecycle.run.settings, merged_retentions)

                    # The facts this response consumed are marked consolidated in the SAME transaction
                    # as the observations that now carry them. Stamping them separately is what let a
                    # half-applied batch orphan its sources forever: the pending-consolidation predicate
                    # excludes a stamped fact, so nothing would ever rebuild what the batch failed to
                    # write (#3876).
                    if stamp_ids:
                        await get_memories().mark_consolidated(
                            conn=conn,
                            fq_table=fq_table,
                            bank_id=bank_id,
                            unit_ids=[str(mem_id) for mem_id in stamp_ids],
                            when=datetime.now(timezone.utc),
                            failed=False,
                        )
                    await write_batch.commit()
        except BaseException:
            await write_batch.abort()
            raise

    if lifecycle is not None:
        if claim_state is not None:
            lifecycle.retired_tags.update(claim_state.retired_tags)
            lifecycle.run.counts.update(claim_state.transitions)
        if lifecycle_drops is not None:
            lifecycle.run.counts.update({f"dropped_{reason}": n for reason, n in lifecycle_drops.counts.items()})

    # Build per-memory result dicts for the stats tracker in the outer loop
    old_news = lifecycle_plan.acted_on if lifecycle_plan is not None else set()
    results: list[dict[str, Any]] = []
    for m in memories:
        mid = str(m["id"])
        created = mid in per_memory_created
        updated = mid in per_memory_updated
        if created and updated:
            results.append({"action": "multiple", "created": 1, "updated": 1, "merged": 0, "total_actions": 2})
        elif created:
            results.append({"action": "created"})
        elif updated:
            results.append({"action": "updated"})
        elif mid in triage.chatter:
            results.append({"action": "skipped", "reason": "triage_chatter"})
        elif mid in old_news:
            # Old news about a retired observation: acted on, never re-asked.
            results.append({"action": "skipped", "reason": "lifecycle_old_news"})
        else:
            results.append({"action": "skipped", "reason": "no_durable_knowledge"})

    return results, deleted_count, llm_result.failed


def _min_date(dates: "Any") -> "datetime | None":
    """Return the minimum non-None datetime from an iterable."""
    return min((d for d in dates if d is not None), default=None)


def _max_date(dates: "Any") -> "datetime | None":
    """Return the maximum non-None datetime from an iterable."""
    return max((d for d in dates if d is not None), default=None)


@dataclass(frozen=True)
class _ObservationHistorySnapshot:
    """Pre-update state of an observation, persisted as the ``content`` JSON blob
    of one observation_history row.

    Temporal fields are the ISO strings carried on MemoryFact; new_source_memory_ids
    are the ids added by the update.
    """

    previous_text: str | None
    previous_tags: list[str]
    previous_occurred_start: str | None
    previous_occurred_end: str | None
    previous_mentioned_at: str | None
    new_source_memory_ids: list[str]


async def _append_observation_history(
    conn: "DatabaseConnection",
    bank_id: str,
    observation_id: str,
    snapshot: _ObservationHistorySnapshot,
    max_entries: int,
) -> None:
    """Insert one pre-update snapshot into ``observation_history``, then delete the
    oldest rows beyond ``max_entries`` for this observation.

    The snapshot is stored as a single JSONB ``content`` blob (per-row, so it stays
    small). Bounding by row count keeps a frequently-reinforced observation's
    history from growing without bound.
    """
    obs_uuid = uuid.UUID(observation_id)
    try:
        await conn.execute(
            f"""
        INSERT INTO {fq_table("observation_history")} (observation_id, bank_id, content, changed_at)
        VALUES ($1, $2, $3::jsonb, now())
        """,
            obs_uuid,
            bank_id,
            json.dumps(asdict(snapshot)),
        )
    except asyncpg.exceptions.ForeignKeyViolationError:
        logger.warning(
            f"FK violation writing observation_history for {observation_id}: "
            "observation was removed before history could be written (race with parallel consolidation). Skipping."
        )
        return
    if max_entries and max_entries > 0:
        await conn.execute(
            f"""
            DELETE FROM {fq_table("observation_history")}
            WHERE observation_id = $1
              AND id NOT IN (
                  SELECT id FROM {fq_table("observation_history")}
                  WHERE observation_id = $1
                  ORDER BY changed_at DESC, id DESC
                  LIMIT $2
              )
            """,
            obs_uuid,
            max_entries,
        )


async def _apply_update_action(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    prepared: _PreparedUpdate,
    perf: ConsolidationPerfLog | None = None,
) -> str | None:
    """Write one prepared UPDATE: rewrite the observation and re-attach its sources.

    Extends source_memory_ids with all contributing memories, widens the observation's temporal
    bounds by ``prepared.source_bounds`` (see :class:`_TemporalBounds`), and merges tags.

    Runs entirely on the caller's connection inside the caller's transaction. Everything slow —
    the liveness preflight and the embedding — already happened in the prepare phase, so this
    write commits or rolls back together with every other write derived from the same LLM
    response, including the ``consolidated_at`` stamps for its source facts (#3876).

    Returns the observation's embedding (pgvector literal) so the caller can run UPDATE-path
    dedup without re-embedding, or None when the update was skipped.
    """
    from ...config import get_config

    model = prepared.model
    observation_id = prepared.update.observation_id
    new_text = prepared.update.text
    source_memory_ids = prepared.source_memory_ids
    source_fact_tags = prepared.source_fact_tags
    source_bounds = prepared.source_bounds
    embedding_str = prepared.embedding_str

    config = get_config()
    store = get_memories()

    # FOR SHARE liveness + the write share the batch's transaction, so a concurrent
    # delete cannot remove a source row between the check and the UPDATE.
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(
            f"Update skipped: all {len(source_memory_ids)} source memories for observation "
            f"{observation_id} were deleted concurrently"
        )
        return None
    live_ids = live_source_memory_ids

    # Recall predates the LLM/embedding work, so the snapshot's tags may be stale (#4831):
    # merging them would drop tags added since and bring back tags removed since. Merge
    # into the observation's current tags, held until the caller commits.
    current_tags = await store.lock_observation_tags(
        conn=conn, fq_table=fq_table, bank_id=bank_id, observation_id=observation_id
    )
    if current_tags is None:
        logger.debug(f"Update skipped: observation {observation_id} no longer exists")
        return None

    history_entry = _ObservationHistorySnapshot(
        previous_text=model.text,
        previous_tags=current_tags,
        previous_occurred_start=model.occurred_start,
        previous_occurred_end=model.occurred_end,
        previous_mentioned_at=model.mentioned_at,
        new_source_memory_ids=[str(mid) for mid in live_ids],
    )

    # Stored ids are strings, fresh ones are UUIDs: normalise before dropping repeats.
    merged = dict.fromkeys(str(s) for s in [*(model.source_fact_ids or []), *live_ids])
    source_ids = [uuid.UUID(s) for s in merged]

    # SECURITY: Merge source fact's tags into existing observation tags so all contributors can see it
    existing_tags = set(current_tags)
    source_tags = set(source_fact_tags or [])
    merged_tags = list(existing_tags | source_tags)

    t0 = time.perf_counter()
    rewritten = await store.rewrite_observation(
        conn=conn,
        fq_table=fq_table,
        bank_id=bank_id,
        observation_id=observation_id,
        text=new_text,
        embedding=embedding_str,
        source_memory_ids=source_ids,
        tags=merged_tags,
        bounds=source_bounds,
        previous=model,
    )
    # The source-liveness checks above guard the *source* memories; the
    # observation row itself can still be invalidated/deleted
    # concurrently, matching 0 rows. Bail out BEFORE the observation_history
    # INSERT below — that INSERT carries an observation_id FK onto memory_units,
    # so appending history for a now-missing row raises ForeignKeyViolationError,
    # a non-retryable integrity failure that would fail the whole consolidation
    # op for a row that simply no longer exists.
    if not rewritten:
        logger.debug(
            f"Update skipped: observation {observation_id} no longer exists "
            "(deleted/invalidated concurrently); not appending history"
        )
        return None

    # Record the pre-update snapshot in the dedicated observation_history table
    # (one row per change), then trim to the configured cap. History lived in a
    # single unbounded JSONB column before; an often-reinforced observation grew
    # it until it crossed Postgres's 256MB jsonb limit and got stuck.
    if config.enable_observation_history:
        await _append_observation_history(
            conn, bank_id, observation_id, history_entry, config.observation_history_max_entries
        )

    # Sync observation_sources junction table (Oracle only — PG uses native array ops).
    if memory_engine._backend.ops.uses_observation_sources_table:
        obs_uuid = uuid.UUID(observation_id)
        await conn.execute(
            f"DELETE FROM {fq_table('observation_sources')} WHERE observation_id = $1",
            obs_uuid,
        )
        if source_ids:
            await conn.executemany(
                f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
                [(obs_uuid, sid) for sid in dict.fromkeys(source_ids)],
            )

    if perf:
        perf.record_timing("db_write", time.perf_counter() - t0)

    # Map the updated observation onto the consolidation trace as a produced memory.
    record_created_memory_ids([observation_id])
    logger.debug(f"Updated observation {observation_id} from {len(source_memory_ids)} source memories")
    return embedding_str


async def _apply_create_action(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    prepared: "_PreparedCreate",
    perf: ConsolidationPerfLog | None = None,
) -> str:
    """
    Create a new observation from one or more source memories.

    Tags are inherited from the source facts (determined algorithmically, not by LLM)
    to maintain visibility scope. Returns the write action ("created" or "skipped").

    Runs on the caller's connection inside the caller's transaction (#3876).
    """
    created = await _apply_create_observation(
        conn=conn,
        memory_engine=memory_engine,
        bank_id=bank_id,
        source_memory_ids=prepared.source_memory_ids,
        observation_text=prepared.text,
        embedding_str=prepared.embedding_str,
        tags=prepared.source_fact_tags or [],
        event_date=prepared.agg.event_date,
        occurred_start=prepared.agg.occurred_start,
        occurred_end=prepared.agg.occurred_end,
        mentioned_at=prepared.agg.mentioned_at,
        perf=perf,
    )
    # Map the new observation onto the consolidation trace as a produced memory.
    new_id = created.get("observation_id")
    if new_id:
        prepared.observation_id = str(new_id)
        record_created_memory_ids([new_id])
    logger.debug(f"Created observation from {len(prepared.source_memory_ids)} source memories")
    return created["action"]


def _apply_triage(
    llm_result: _BatchLLMResult,
    triage: TriageResult,
    union_observations: list["MemoryFact"],
    retiring: Sequence[str] = (),
) -> None:
    """Write the triage verdicts as actions: a new fact is a CREATE of its own sentence, and a
    restatement is evidence for its observation — an UPDATE that keeps the text as it is.

    Folded into the LLM's own update of that observation when there is one, since two
    updates of one observation in a response overwrite each other.

    ``retiring`` (lifecycle mode): observations the response supersedes or refutes. A restatement
    of one is old-state evidence; it must not land on the successor, which has no id in this
    response either, so it is dropped like a retired one's and the fact is asked again next pass.
    """
    text_by_id = {str(o.id): o.text for o in union_observations}
    id_by_text = {_norm_obs_text(o.text): str(o.id) for o in union_observations}
    # An update wins over a delete of the same observation, so evidence for one the LLM deleted
    # (retired, or merged away) on another fact's word would bring it back. A merged one's
    # evidence goes to the survivor; a retired one's goes nowhere, like the LLM's own facts.
    deleted: dict[str, str | None] = {d.observation_id: d.merged_into for d in llm_result.deletes}
    deleted.update(dict.fromkeys(retiring))
    restates = dict(triage.restates)
    for fact_id, sentence in triage.new.items():
        # Recorded verbatim before, an exact copy would be dropped as a duplicate create and the
        # fact left backing nothing: it is evidence for the observation that already says it.
        if (twin := id_by_text.get(_norm_obs_text(sentence))) is not None:
            restates[fact_id] = twin
            continue
        llm_result.creates.append(
            _CreateAction(text=sentence, source_fact_ids=[fact_id], reason="triage: a new facet, recorded as stated")
        )
    for fact_id, observation_id in restates.items():
        if observation_id in deleted:
            if (observation_id := deleted[observation_id]) is None:
                continue
        existing = next((u for u in llm_result.updates if u.observation_id == observation_id), None)
        if existing is not None:
            existing.source_fact_ids = [*existing.source_fact_ids, fact_id]
        elif observation_id in text_by_id:
            llm_result.updates.append(
                _UpdateAction(
                    observation_id=observation_id,
                    text=text_by_id[observation_id],
                    source_fact_ids=[fact_id],
                    reason="triage: restates this observation",
                )
            )


def _resolve_pool_id(raw: str, pool_ids: set[str]) -> str:
    """``raw`` as a pool id, tolerating a miscopied tail.

    Small models copy the head of a UUID and invent the rest (``d0a3effa-1795-4c48-…`` came
    back as ``d0a3effa-1795-4ef0-…``). An exact match wins; otherwise the one pool id sharing
    the first two groups is meant: 48 bits, so two candidates never collide by chance in a
    pool of tens. Anything else is returned unchanged and rejected downstream.
    """
    if raw in pool_ids:
        return raw
    matches = [pid for pid in pool_ids if len(raw) >= 13 and pid.startswith(raw[:13])]
    return matches[0] if len(matches) == 1 else raw


#: Lifecycle label fields naming pool observations, and those naming the pool's source facts.
_LIFECYCLE_TARGET_FIELDS = ("supersedes", "refutes", "refines", "conflicts_with")
_LIFECYCLE_SOURCE_FIELDS = ("retires_source_ids", "keeps_source_ids")


def _resolve_miscopied_ids(llm_result: _BatchLLMResult, pool_ids: set[str], source_ids: set[str] | None = None) -> None:
    """Repair miscopied ids in place: every observation id the response names against the pool,
    and (lifecycle mode) every source fact id it names against the source ids the pool showed."""
    for update in llm_result.updates:
        update.observation_id = _resolve_pool_id(update.observation_id, pool_ids)
    for delete in llm_result.deletes:
        delete.observation_id = _resolve_pool_id(delete.observation_id, pool_ids)
        if delete.merged_into is not None:
            delete.merged_into = _resolve_pool_id(delete.merged_into, pool_ids)
    sources = source_ids or set()
    for action in [*llm_result.creates, *llm_result.updates]:
        for name in _LIFECYCLE_TARGET_FIELDS:
            if ids := getattr(action, name, None):
                setattr(action, name, [_resolve_pool_id(i, pool_ids) for i in ids])
        if split_from := getattr(action, "split_from", None):
            action.split_from = _resolve_pool_id(split_from, pool_ids)
        for name in _LIFECYCLE_SOURCE_FIELDS:
            if ids := getattr(action, name, None):
                setattr(action, name, [_resolve_pool_id(i, sources) for i in ids])


async def _execute_delete_action(
    conn: "DatabaseConnection",
    bank_id: str,
    observation_id: str,
) -> None:
    """Delete a superseded or contradicted observation."""
    await get_memories().delete_observation(
        conn=conn, fq_table=fq_table, bank_id=bank_id, observation_id=observation_id
    )
    # History lives in Postgres regardless of where the observation itself does, and no
    # longer cascades from memory_units (that FK was dropped so it could be recorded for
    # observations kept outside SQL). Drop it explicitly so a deleted observation's
    # snapshots don't accumulate forever.
    await conn.execute(
        f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1 AND observation_id = $2",
        bank_id,
        uuid.UUID(observation_id),
    )
    logger.debug(f"Deleted observation {observation_id}")


async def _fold_deleted_evidence(
    conn: "DatabaseConnection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    gone: StoredMemory,
    target_id: str | None,
    requeue_on_miss: bool = True,
) -> bool:
    """Move a merged-away observation's evidence onto the observation that absorbed it.

    Small models merge observations by rewriting one and deleting the others. An UPDATE can
    only cite this batch's facts, so the deleted ones' evidence (stamped consolidated long
    ago) was cited by nothing and never looked at again (eval ``noisy-personal-contacts``).
    Re-queueing every deleted observation's facts was tried first: it also brought back the
    false beliefs a delete retires (eval ``correction-retires-false-belief``), so only a
    delete the model marks as a merge (``merged_into``) moves its evidence. Without a usable
    target (the model merged into an observation it creates in the same response, which has no
    id yet) or with the survivor gone, the evidence is re-queued and the next pass attaches it.

    Returns True when the evidence was folded into ``target_id``.

    ``requeue_on_miss`` False (lifecycle mode): a missed fold leaves the merged observation in place
    and active, still carrying its evidence, so there is nothing to requeue.
    """
    live = await _filter_live_source_memories(conn, bank_id, [uuid.UUID(str(sid)) for sid in gone.source_memory_ids])
    if not live:
        return False
    store = get_memories()
    if target_id is not None:
        target = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[target_id])
        if target and target[0].fact_type == "observation":
            folded = await store.fold_sources_into_observation(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                observation_id=target_id,
                expected_text=target[0].text,
                merged_text=target[0].text,
                source_memory_ids=live,
                bounds=_TemporalBounds.of(gone),
                embeddings=memory_engine.embeddings,
            )
            if folded:
                return True
    if requeue_on_miss:
        await store.requeue_units(conn, bank_id=bank_id, unit_ids=[str(sid) for sid in live], reason="fold_miss")
    return False


async def _prepare_observation_evidence(
    pool: DatabaseBackend,
    bank_id: str,
    settings: "AdaptiveSettings",
    memories: list[dict[str, Any]],
    prepared_updates: list[_PreparedUpdate],
    prepared_creates: list[_PreparedCreate],
    prepared_deletes: list[_PreparedDelete],
) -> ObservationEvidence | None:
    """The evidence a batch's writes may reinforce its observations with, read before they run.

    The batch's own facts carry their provenance (``with_provenance``). An observation the batch
    may add sources to — an UPDATE target, a dedup twin, a ``merged_into`` survivor — also needs the
    evidence keys of the sources it already has, so a re-statement from a session it has already
    heard counts nothing: one read of those observations, one of their sources, one of the sources'
    documents. None (logged, counted) when that read fails: the batch then simply reinforces nothing.
    """
    now = adaptive_clock.utcnow()
    facts: dict[str, FactEvidence] = {
        str(m["id"]): fact_evidence(
            memory_id=str(m["id"]),
            document_id=m.get("document_id"),
            document_id_minted=bool(m.get("document_id_minted")),
            metadata=m.get("metadata"),
            mentioned_at=m.get("mentioned_at"),
            occurred_start=m.get("occurred_start"),
            event_date=m.get("event_date"),
            now=now,
        )
        for m in memories
    }
    observation_ids = list(
        dict.fromkeys(
            [p.update.observation_id for p in prepared_updates]
            + [
                p.dedup.best_id
                for p in [*prepared_updates, *prepared_creates]
                if p.dedup is not None and p.dedup.should_merge and p.dedup.best_id
            ]
            + [d.merged_into for d in prepared_deletes if d.is_merge and d.merged_into]
            # A merged-away observation's sources move to its survivor, so they need keys too.
            + [d.observation_id for d in prepared_deletes if d.is_merge and d.merged_into]
        )
    )
    prior_sources: dict[str, list[str]] = {}
    prior_mentioned_at: dict[str, datetime | None] = {}
    if observation_ids:
        store = get_memories()
        try:
            async with acquire_with_retry(pool) as conn:
                for obs in await store.get_memories(
                    conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=observation_ids
                ):
                    prior_sources[obs.unit_id] = [str(s) for s in obs.source_memory_ids]
                    prior_mentioned_at[obs.unit_id] = obs.mentioned_at
                others = list(dict.fromkeys(s for srcs in prior_sources.values() for s in srcs if s not in facts))
                if others:
                    sources = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=others)
                    minted = await store.minted_document_ids(
                        conn, bank_id=bank_id, document_ids=[f.document_id for f in sources if f.document_id]
                    )
                    for f in sources:
                        facts[f.unit_id] = fact_evidence(
                            memory_id=f.unit_id,
                            document_id=f.document_id,
                            document_id_minted=f.document_id is not None and str(f.document_id) in minted,
                            metadata=f.metadata,
                            mentioned_at=f.mentioned_at,
                            occurred_start=f.occurred_start,
                            event_date=f.event_date,
                            now=now,
                        )
        except Exception as exc:
            logger.warning(f"[CONSOLIDATION] bank={bank_id} reinforcement evidence read failed, skipping it: {exc!r}")
            get_metrics_collector().record_adaptive_reinforcement_dropped(
                producer="consolidation", reason="error", bank_id=bank_id
            )
            return None
    return ObservationEvidence(
        settings, facts=facts, prior_sources=prior_sources, prior_mentioned_at=prior_mentioned_at
    )


async def _record_observation_reinforcement(conn, bank_id: str, evidence: ObservationEvidence) -> None:
    """Write a batch's observation reinforcement in a savepoint of the batch's transaction.

    A failure rolls back the savepoint (and nothing the batch wrote), is logged and counted, and
    never fails the batch: reinforcement must not cost a valid consolidation. ``background``: a
    memory another transaction holds (being deleted) is skipped rather than waited for.
    """
    call = evidence.call()
    if call is None:
        return
    try:
        async with conn.transaction():
            await get_memories().record_reinforcement(
                conn,
                bank_id=bank_id,
                events=call.events,
                settings=evidence.settings,
                now=adaptive_clock.utcnow(),
                background=True,
                anchors=call.anchors,
                initial_memory_ids=call.initial_memory_ids,
            )
    except Exception as exc:
        logger.warning(f"[CONSOLIDATION] bank={bank_id} reinforcement failed, batch kept: {exc!r}")
        get_metrics_collector().record_adaptive_reinforcement_dropped(
            producer="consolidation", reason="error", bank_id=bank_id
        )


# --------------------------------------------------------------------------- lifecycle mode
#
# The claim lifecycle's half of a batch (the labels are planned in ``lifecycle``). Every
# helper below runs only when the bank's effective mode is lifecycle (or, for the shadow, observe),
# so an off or rank bank never reaches one.


def _retirements_named(llm_result: _BatchLLMResult, pool: Mapping[str, claim_lifecycle.PoolEntry]) -> list[str]:
    """Pool observations the response supersedes or refutes (before validation: what triage must
    not land a restatement on)."""
    named = [
        target
        for create in llm_result.creates
        for name in ("supersedes", "refutes")
        for target in claim_lifecycle.clean_ids(getattr(create, name, None))
    ]
    named += [u.observation_id for u in llm_result.updates if getattr(u, "relation", None) in ("supersedes", "refutes")]
    return [t for t in dict.fromkeys(named) if t in pool]


async def _lifecycle_target_sources(
    pool: DatabaseBackend,
    bank_id: str,
    llm_result: _BatchLLMResult,
    lifecycle_pool: Mapping[str, claim_lifecycle.PoolEntry],
) -> dict[str, list[StoredMemory]]:
    """The source facts of every observation the response's labels name, as stored: their metadata
    gives a target its scope fingerprint and its side of a conflict its trust. One read."""
    targets = claim_lifecycle.lifecycle_target_ids(llm_result, lifecycle_pool)
    wanted = list(dict.fromkeys(s for t in targets for s in lifecycle_pool[t].source_ids))
    if not wanted:
        return {}
    async with acquire_with_retry(pool) as conn:
        rows = {
            m.unit_id: m
            for m in await get_memories().get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=wanted)
        }
    return {t: [rows[s] for s in lifecycle_pool[t].source_ids if s in rows] for t in targets}


def _cap_lifecycle_creates(
    llm_result: _BatchLLMResult,
    plan: claim_lifecycle.LifecyclePlan,
    remaining_observation_slots: int | None,
    max_observations_per_scope: int,
) -> None:
    """The observation cap in lifecycle mode, applied after the labels: a retirement in the same
    response frees its slot. Labelled creates are never dropped (a successor replaces what it
    retires); ordinary creates keep what is left, in order."""
    if remaining_observation_slots is None or remaining_observation_slots < 0:
        return
    freed = (
        len(plan.retire_target_ids())
        + len({p.split_from for p in plan.successors.values() if p.split_from})
        + len(plan.plain_retires)
        + sum(1 for d in llm_result.deletes if d.merged_into)
    )
    labelled = [c for c in llm_result.creates if (p := plan.for_create(c)) is not None and p.labelled]
    budget = max(0, remaining_observation_slots + freed - len(labelled))
    kept: list[_CreateAction] = []
    plain_kept = 0
    for create in llm_result.creates:
        if (p := plan.for_create(create)) is not None and p.labelled:
            kept.append(create)
        elif plain_kept < budget:
            kept.append(create)
            plain_kept += 1
    if len(kept) < len(llm_result.creates):
        logger.info(
            f"[CONSOLIDATION] Truncating {len(llm_result.creates)} creates to {len(kept)} "
            f"(max_observations_per_scope={max_observations_per_scope}, {freed} slot(s) freed by retirements)"
        )
    llm_result.creates = kept


def _lifecycle_create_sources(plan: claim_lifecycle.SuccessorPlan, cited_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    """A labelled create's sources: its cited facts and the replaced observation's sources it keeps.

    Never the replaced (or split) observation's other sources: those are what the retirement passes
    its status down to. A split facet records the parent's sources as claim evidence instead
    (``_apply_split``), so the over-general raw facts do not stay current through the facets."""
    ids = list(cited_ids)
    seen = {str(i) for i in ids}
    for sid in plan.keeps:
        if sid not in seen:
            seen.add(sid)
            ids.append(uuid.UUID(sid))
    return ids


async def _lock_lifecycle_claims(
    conn: "DatabaseConnection",
    state: claim_lifecycle.ApplyState,
    plan: claim_lifecycle.LifecyclePlan,
    prepared_deletes: list[_PreparedDelete],
    prepared_updates: list[_PreparedUpdate],
    prepared_creates: list[_PreparedCreate],
) -> None:
    """Lock everything the batch's claim transitions touch: the observations' rows, then — in the
    one sorted ``lock_claims`` pass — every existing claim (targets, kept and cited facts,
    inheritance candidates, merge pairs, dedup twins, and the retired claims a create may re-link
    as their rebuilt successor).

    Observation rows first because that is the order every other writer of these claims takes
    them in: the stale-observation sweep (and the delete paths and the PATCH lifecycle API on a raw
    fact behind it) row-locks the observations, then locks claims. Taking claims first here would
    let a PATCH of one of a target's sources and this batch wait on each other.
    """
    observations: list[str] = [*plan.retire_target_ids(), *plan.plain_retires, *(a.target_id for a in plan.historical)]
    facts: list[str] = []
    for prepared in prepared_creates:
        lp = prepared.lifecycle_plan
        if lp is not None:
            observations += [
                *lp.refines,
                *(c.target_id for c in lp.conflicts),
                *([lp.split_from] if lp.split_from else []),
            ]
            facts += [*lp.cited, *lp.keeps]
        if prepared.dedup is not None and prepared.dedup.should_merge and prepared.dedup.best_id:
            observations.append(prepared.dedup.best_id)
    for prepared in prepared_updates:
        if prepared.dedup is not None and prepared.dedup.should_merge and prepared.dedup.best_id:
            observations += [prepared.dedup.best_id, prepared.update.observation_id]
    for delete in prepared_deletes:
        if delete.is_merge:
            observations += [delete.observation_id, *([delete.merged_into] if delete.merged_into else [])]
    for attach in plan.historical:
        facts += list(attach.fact_ids)
    observations = sorted(set(observations))
    for observation_id in observations:
        await state.store.lock_observation_tags(
            conn=conn, fq_table=state.fq_table, bank_id=state.bank_id, observation_id=observation_id
        )
    # Raw inheritance candidates: the sources of whatever may be retired (with a successor, or split).
    conflict_targets = [c.target_id for p in prepared_creates if p.lifecycle_plan for c in p.lifecycle_plan.conflicts]
    split_parents = [
        p.lifecycle_plan.split_from for p in prepared_creates if p.lifecycle_plan and p.lifecycle_plan.split_from
    ]
    inheriting = list(dict.fromkeys([*plan.retire_target_ids(), *conflict_targets, *split_parents]))
    if inheriting:
        state.sources = await state.store.observation_sources(conn, bank_id=state.bank_id, observation_ids=inheriting)
        facts += [s for sources in state.sources.values() for s in sources]
    # The predecessors ``relink_rebuilt_successors`` may point at a create: in this pass, so the
    # re-link after the creates takes no claim lock of its own on rows another writer can hold.
    created_sources = sorted({str(s) for p in prepared_creates for s in p.source_memory_ids})
    relinkable = (
        await state.store.relinkable_claims(conn, bank_id=state.bank_id, evidence_ids=created_sources)
        if created_sources
        else set()
    )
    await claim_lifecycle.lock(conn, state, [*observations, *facts, *relinkable])


async def _merge_precheck(
    conn: "DatabaseConnection", state: claim_lifecycle.ApplyState, gone_id: str, survivor_id: str | None
) -> bool:
    """Lifecycle MERGE: fold only into a live survivor, and only an observation that is live and in
    this scope — it is retired as merged, not deleted, once its evidence has moved."""
    if survivor_id is None or not state.live(survivor_id):
        logger.info(f"[CONSOLIDATION][lifecycle] merge of {gone_id[:8]} skipped: survivor not live")
        return False
    if not state.live(gone_id):
        state.drops.drop("cas_lost", f"merge of {gone_id}")
        return False
    return await claim_lifecycle.in_scope(conn, state, gone_id)


async def _merge_retire(
    conn: "DatabaseConnection", state: claim_lifecycle.ApplyState, gone_id: str, survivor_id: str
) -> bool:
    """Retire a folded-away observation as merged into its survivor (a rewording, not an earlier state)."""
    won = await claim_lifecycle.retire(
        conn,
        state,
        target_id=gone_id,
        status=TruthStatus.SUPERSEDED,
        retired_via="merges",
        successor_id=survivor_id,
        valid_to=None,
        reason="consolidation:merge",
        relation="merges",
        evidence_ids=[],
        prechecked=True,
    )
    if won:
        state.count("merge")
    return won


async def _merge_retentions(
    conn: "DatabaseConnection", bank_id: str, settings: "AdaptiveSettings", pairs: list[tuple[str, str]]
) -> None:
    """The survivor of each merge keeps the larger stability of the pair, in a savepoint: a failure
    is logged and never costs the batch."""
    for survivor_id, merged_id in pairs:
        try:
            async with conn.transaction():
                await get_memories().merge_retention(
                    conn,
                    bank_id=bank_id,
                    survivor_id=survivor_id,
                    merged_id=merged_id,
                    settings=settings,
                    now=adaptive_clock.utcnow(),
                )
        except Exception as exc:
            logger.warning(
                f"[CONSOLIDATION] bank={bank_id} merge retention for {survivor_id[:8]} failed, kept: {exc!r}"
            )


async def _apply_lifecycle_transitions(
    conn: "DatabaseConnection",
    state: claim_lifecycle.ApplyState,
    plan: claim_lifecycle.LifecyclePlan,
    prepared_creates: list[_PreparedCreate],
    memories: list[dict[str, Any]],
) -> set[str]:
    """Write the batch's claim transitions, after the observations they name exist.

    Successors (and the facts asserting the new state) get their ``valid_from``; each retirement
    is a CAS on the target, then its relation, its evidence and the raw inheritance; splits retire
    their parent once every facet exists; conflicts mark both sides uncertain unless the trust rule
    picks a winner; old facts join a retired observation as history; a plain delete retires its
    target without a successor. Returns the facts attached as history.
    """
    store = state.store
    lifecycle_creates = [p for p in prepared_creates if p.lifecycle_plan is not None]
    await claim_lifecycle.lock_new(conn, state, [p.observation_id for p in lifecycle_creates if p.observation_id])
    split_children: defaultdict[str, list[_PreparedCreate]] = defaultdict(list)
    for prepared in lifecycle_creates:
        lp = prepared.lifecycle_plan
        successor = prepared.observation_id
        if lp is None:
            continue
        if successor is None:
            state.drops.drop("successor_not_written", f"{len(lp.retires)} retirement(s) of a create that was skipped")
            continue
        await claim_lifecycle.assert_current(
            conn, state, memory_id=successor, valid_from=lp.t_star, fingerprint=lp.fingerprint
        )
        for fact_id in lp.cited:
            await claim_lifecycle.assert_current(conn, state, memory_id=fact_id, valid_from=lp.t_star)
        for target in lp.retires:
            await _retire_with_successor(conn, state, plan, target, successor, lp)
        for refined in lp.refines:
            await store.insert_relation(
                conn,
                bank_id=state.bank_id,
                relation_type="refines",
                source_id=successor,
                target_id=refined,
                origin=claim_lifecycle.ORIGIN,
                reason="consolidation:refines",
                confidence=None,
                evidence={"fact_ids": lp.cited},
                now=state.now,
            )
        for conflict in lp.conflicts:
            await _apply_conflict(conn, state, plan, conflict, successor, lp)
        if lp.split_from is not None:
            split_children[lp.split_from].append(prepared)

    for parent, children in split_children.items():
        await _apply_split(conn, state, parent, children)

    attached: set[str] = set()
    for attach in plan.historical:
        live = await _filter_live_source_memories(conn, state.bank_id, [uuid.UUID(f) for f in attach.fact_ids])
        retired = await claim_lifecycle.attach_historical(conn, state, attach, live)
        if live:
            attached.update(str(f) for f in live)
            state.transitions["historical_attached"] += len(live)
            state.transitions["historical_retired"] += len(retired)

    if plan.plain_retires:
        batch_ids = [str(m["id"]) for m in memories]
        earliest = claim_lifecycle.change_instant(memories, state.now)
        for target_id in plan.plain_retires:
            claim = state.claims.get(target_id)
            valid_to = max(earliest, claim.valid_from) if claim is not None and claim.valid_from else earliest
            won = await claim_lifecycle.retire(
                conn,
                state,
                target_id=target_id,
                status=TruthStatus.SUPERSEDED,
                retired_via="supersedes",
                successor_id=None,
                valid_to=valid_to,
                reason="retire_without_successor",
                relation=None,
                evidence_ids=batch_ids,
            )
            if won:
                state.count("supersession")
                state.transitions["retire_without_successor"] += 1
                logger.info(f"[CONSOLIDATION][lifecycle] retire_without_successor: {target_id[:8]}")
    return attached


async def _retire_with_successor(
    conn: "DatabaseConnection",
    state: claim_lifecycle.ApplyState,
    plan: claim_lifecycle.LifecyclePlan,
    target: claim_lifecycle.RetireTarget,
    successor: str,
    lp: claim_lifecycle.SuccessorPlan,
) -> None:
    won = await claim_lifecycle.retire(
        conn,
        state,
        target_id=target.target_id,
        status=target.status,
        retired_via=target.relation,
        successor_id=successor,
        valid_to=lp.t_star,
        reason=f"consolidation:{target.relation}",
        relation=target.relation,
        evidence_ids=lp.cited,
    )
    if not won:
        return
    state.count("supersession" if target.status is TruthStatus.SUPERSEDED else "refutation")
    inherited = await claim_lifecycle.inherit(
        conn,
        state,
        target=target,
        successor_id=successor,
        valid_to=lp.t_star,
        evidence_ids=lp.cited,
        keeps=lp.keeps,
        recited=plan.recited,
    )
    state.transitions["inherited"] += len(inherited)


async def _apply_conflict(
    conn: "DatabaseConnection",
    state: claim_lifecycle.ApplyState,
    plan: claim_lifecycle.LifecyclePlan,
    conflict: claim_lifecycle.ConflictPlan,
    created_id: str,
    lp: claim_lifecycle.SuccessorPlan,
) -> None:
    """One ``conflicts_with``: a ``conflicts`` relation, then both sides uncertain — or, when the
    trust rule (or policy ``latest``) picks a winner, a supersession of the losing side."""
    if not state.live(conflict.target_id) or not await claim_lifecycle.in_scope(conn, state, conflict.target_id):
        return
    outcome = conflict.outcome
    await state.store.insert_relation(
        conn,
        bank_id=state.bank_id,
        relation_type="conflicts",
        source_id=created_id,
        target_id=conflict.target_id,
        origin=claim_lifecycle.ORIGIN,
        reason=f"consolidation:conflict:{outcome.winner or 'unresolved'}",
        confidence=outcome.new_confidence,
        evidence={"fact_ids": lp.cited},
        now=state.now,
    )
    if outcome.winner == "new":
        await _retire_with_successor(
            conn,
            state,
            plan,
            claim_lifecycle.RetireTarget(
                target_id=conflict.target_id,
                status=TruthStatus.SUPERSEDED,
                relation="supersedes",
                explicit_sources=None,
            ),
            created_id,
            lp,
        )
        return
    if outcome.winner == "existing":
        # The new claim is born superseded by the observation that won; its own facts go with it.
        won = await claim_lifecycle.retire(
            conn,
            state,
            target_id=created_id,
            status=TruthStatus.SUPERSEDED,
            retired_via="supersedes",
            successor_id=conflict.target_id,
            valid_to=lp.t_star,
            reason="consolidation:conflict_lost",
            relation="supersedes",
            evidence_ids=lp.cited,
            prechecked=True,
        )
        if won:
            state.count("supersession")
            inherited = await claim_lifecycle.inherit(
                conn,
                state,
                target=claim_lifecycle.RetireTarget(
                    target_id=created_id,
                    status=TruthStatus.SUPERSEDED,
                    relation="supersedes",
                    explicit_sources=tuple(lp.cited),
                ),
                successor_id=conflict.target_id,
                valid_to=lp.t_star,
                evidence_ids=lp.cited,
                keeps=(),
                recited=set(),
            )
            state.transitions["inherited"] += len(inherited)
        return
    await claim_lifecycle.mark_uncertain(
        conn, state, memory_id=conflict.target_id, confidence=outcome.existing_confidence, valid_from=None
    )
    await claim_lifecycle.mark_uncertain(
        conn, state, memory_id=created_id, confidence=outcome.new_confidence, valid_from=lp.t_star
    )


async def _apply_split(
    conn: "DatabaseConnection", state: claim_lifecycle.ApplyState, parent: str, children: list[_PreparedCreate]
) -> None:
    """Retire a split parent once its facets exist.

    The parent is superseded with no successor pointer (there are several facets) and one
    ``splits`` relation per facet. Its raw sources inherit the retirement (``retired_via='splits'``,
    no successor) under the usual exclusions, where the facets themselves do not count as
    observations still standing on them: the over-general statement becomes history and current
    recall answers from the facets. Each facet keeps the parent's sources as claim evidence, not as
    its own sources.
    """
    written = [c for c in children if c.observation_id is not None]
    if len(written) < 2:
        state.drops.drop("split_incomplete", f"split of {parent}: {len(written)} facet(s) written")
        return
    plans = [c.lifecycle_plan for c in written if c.lifecycle_plan is not None]
    valid_to = min(p.t_star for p in plans)
    cited = list(dict.fromkeys(f for p in plans for f in p.cited))
    won = await claim_lifecycle.retire(
        conn,
        state,
        target_id=parent,
        status=TruthStatus.SUPERSEDED,
        retired_via="splits",
        successor_id=None,
        valid_to=valid_to,
        reason="consolidation:split",
        relation=None,
        evidence_ids=cited,
    )
    if not won:
        return
    state.count("split")
    facets = [str(c.observation_id) for c in written]
    parent_sources = state.sources.get(parent, [])
    if parent_sources:
        inherited = await state.store.inherit_from_observation(
            conn,
            bank_id=state.bank_id,
            observation_id=parent,
            to_status=TruthStatus.SUPERSEDED,
            successor_id=None,
            valid_to=valid_to,
            evidence_ids=cited,
            candidate_source_ids=parent_sources,
            reason="consolidation:split",
            retired_via="splits",
            ignored_citer_ids=facets,
            now=state.now,
        )
        state.transitions["inherited"] += len(inherited)
    for child in written:
        if parent_sources:
            await state.store.add_claim_evidence(
                conn,
                bank_id=state.bank_id,
                memory_id=str(child.observation_id),
                evidence_ids=parent_sources,
                now=state.now,
            )
        await state.store.insert_relation(
            conn,
            bank_id=state.bank_id,
            relation_type="splits",
            source_id=child.observation_id,
            target_id=parent,
            origin=claim_lifecycle.ORIGIN,
            reason="consolidation:split",
            confidence=None,
            evidence={"fact_ids": child.lifecycle_plan.cited if child.lifecycle_plan else []},
            now=state.now,
        )


async def _record_shadow_retirements(
    conn: "DatabaseConnection",
    memory_engine: "MemoryEngine",
    bank_id: str,
    run: _LifecycleRun,
    prepared_deletes: list[_PreparedDelete],
    memories: list[dict[str, Any]],
    union_observations: list["MemoryFact"],
) -> None:
    """Observe mode: for each plain delete, record what lifecycle mode would have retired instead.

    The delete still happens (observe changes nothing visible), so the observation's own record
    would cascade away with it; the record goes on the raw sources the retirement would have passed
    its status down to (same exclusions as the store's inheritance), with the deciding observation
    in the snapshot. A savepoint: a failed diagnostic write never costs the batch.
    """
    store = get_memories()
    text_by_id = {str(o.id): o.text for o in union_observations}
    batch_ids = [str(m["id"]) for m in memories]
    for delete in prepared_deletes:
        if delete.is_merge:
            continue
        try:
            async with conn.transaction():
                would = await claim_lifecycle.would_inherit(
                    conn,
                    store,
                    fq_table,
                    memory_engine._backend.ops,
                    bank_id=bank_id,
                    observation_id=delete.observation_id,
                )
                snapshot = {
                    "decided_by": "consolidation_delete",
                    "observation_id": delete.observation_id,
                    "observation_text": text_by_id.get(delete.observation_id),
                    "retired_via": "inherited",
                    "batch_fact_ids": batch_ids,
                }
                for fact_id in would:
                    if await store.record_shadow_retirement(
                        conn,
                        bank_id=bank_id,
                        memory_id=fact_id,
                        would_status=TruthStatus.SUPERSEDED,
                        deciding_snapshot=snapshot,
                        evidence_ids=batch_ids,
                        now=adaptive_clock.utcnow(),
                    ):
                        run.counts["shadow_retirements"] += 1
        except Exception as exc:
            logger.warning(
                f"[CONSOLIDATION] bank={bank_id} shadow retirement for {delete.observation_id[:8]} failed: {exc!r}"
            )


async def _embed_observation_text(
    memory_engine: "MemoryEngine",
    text: str,
    perf: ConsolidationPerfLog | None = None,
) -> str | None:
    """Embed one observation text into a pgvector literal, with no connection held.

    Every caller is in the prepare phase of a batch (#3876): the embedder is the slowest
    thing consolidation does per action, and it must finish before the batch's single write
    transaction opens.
    """
    t0 = time.perf_counter()
    embeddings = await embedding_utils.generate_embeddings_batch(memory_engine.embeddings, [text])
    if perf:
        perf.record_timing("embedding", time.perf_counter() - t0)
    return str(embeddings[0]) if embeddings else None


async def _find_related_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    tags: list[str] | None = None,
    config: Any = None,
    lifecycle: bool = False,
    live_only: bool = False,
) -> "RecallResult":
    """
    Find observations related to the given query using optimized recall.

    SECURITY: Filters by tags using all_strict matching to prevent cross-tenant/cross-user
    information leakage. Observations are only consolidated within the same tag scope.

    Uses max_tokens to naturally limit observations (no artificial count limit).
    Includes source memories with dates for LLM context.

    Args:
        tags: Optional tags to filter observations (uses all_strict matching for security)
        lifecycle: Adaptive memory in lifecycle mode: recall through the internal
            ``all-with-status`` view, over-fetched (see ``_shape_lifecycle_pools``).
        live_only: With ``lifecycle``: the live observations only, under the real budgets —
            the current view, with the retired ones dropped inside the store's arms before
            their limits, so they cannot take a live observation's place.

    Returns:
        List of related observations with their tags, source memories, and dates
    """
    # Use recall to find related observations with token budget
    # max_tokens naturally limits how many observations are returned
    from ...tracing import get_tracer, is_tracing_enabled

    # The consolidation pass hands in the config already resolved for its scope, so
    # a consolidation strategy's source-facts token limits reach this recall.
    # Resolving the bank config here instead (as this function used to always do)
    # would silently ignore them.
    if config is None:
        config = await memory_engine._config_resolver.resolve_full_config(bank_id, request_context)

    # SECURITY: Use all_strict matching if tags provided to prevent cross-scope consolidation
    tags_match = "all_strict" if tags else "any"

    # Create span for recall operation within consolidation
    tracer = get_tracer()
    if is_tracing_enabled():
        recall_span = tracer.start_span("hindsight.consolidation_recall")
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])  # Truncate for brevity
        recall_span.set_attribute("hindsight.fact_type", "observation")
    else:
        recall_span = None

    # Resolve budget: consolidation doesn't need deep recall, default to LOW to reduce memory fan-out
    recall_budget = Budget(config.consolidation_recall_budget)

    # Lifecycle mode sees retired observations too, labelled (the internal ``all-with-status``
    # view). They must not spend the pool's token budget, so the recall over-fetches and
    # ``_shape_lifecycle_pools`` applies the real budgets to the live observations alone.
    max_tokens = config.consolidation_max_tokens
    max_source_facts_tokens = config.consolidation_source_facts_max_tokens
    lifecycle_kwargs: dict[str, Any] = {}
    if lifecycle and live_only:
        lifecycle_kwargs.update(lifecycle_view="current", _exclude_retired=True)
    elif lifecycle:
        if max_tokens > 0:
            max_tokens *= _LIFECYCLE_POOL_OVERFETCH
        if max_source_facts_tokens > 0:
            max_source_facts_tokens *= _LIFECYCLE_POOL_OVERFETCH
        lifecycle_kwargs["lifecycle_view"] = ALL_WITH_STATUS

    try:
        recall_result = await memory_engine.recall_async(
            bank_id=bank_id,
            query=query,
            budget=recall_budget,
            max_tokens=max_tokens,  # Token budget for observations (configurable)
            fact_type=["observation"],  # Only retrieve observations
            request_context=request_context,
            tags=tags,  # Filter by source memory's tags
            tags_match=tags_match,  # Use strict matching for security
            include_source_facts=True,  # Embed source facts so we avoid a separate DB fetch
            max_source_facts_tokens=max_source_facts_tokens,
            max_source_facts_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            # Round-robin interleave fusion (no cross-encoder): consolidation is looking
            # for an existing near-identical observation to merge into. Both the
            # cross-encoder (semantic #1 -> reranked #37) and RRF (semantic #1 -> outside
            # the 512-token budget) were measured to bury that twin; interleave guarantees
            # each retrieval arm's top hits a slot, so the semantic-#1 twin is always shown
            # to the LLM, which then UPDATEs instead of creating a duplicate.
            reranking="interleave",
            _quiet=True,  # Suppress logging
            **lifecycle_kwargs,
        )
    finally:
        if recall_span:
            recall_span.end()

    return recall_result


#: How many times the pool's token budgets a lifecycle-mode recall asks for. Retired observations
#: are shown but must not spend the budget, which recall itself cannot tell apart, so it hands back
#: more and ``_shape_lifecycle_pools`` spends the real budget on the live ones. A fact whose
#: over-fetch holds a retired observation has its live ones recalled again without the retired ones
#: (``live_only``), so retired near-twins can never crowd a live one out.
_LIFECYCLE_POOL_OVERFETCH = 3

#: Retired observations with no successor (refuted or retired explicitly, a split parent, one whose
#: successor was deleted) shown per fact, after the live ones and at no budget cost: without one, a
#: fact restating a refuted claim would find nothing to attach to and become a current claim again.
_LIFECYCLE_UNLINKED_HISTORY_PER_FACT = 1


@dataclass
class _LifecyclePool:
    """A batch's per-fact recalls, shaped for lifecycle mode, and the claim state of every entry."""

    recalls: list["RecallResult"]
    entries: dict[str, claim_lifecycle.PoolEntry]


async def _shape_lifecycle_pools(
    pool: DatabaseBackend,
    bank_id: str,
    recalls: list["RecallResult"],
    config: Any,
    scope_tags: frozenset[str],
    live_recall: Callable[[int], Any],
) -> _LifecyclePool:
    """The lifecycle-mode pool: live observations first, within the budget; retired ones labelled.

    One lock-free claim read for every recalled observation. Per recall: the active and uncertain
    observations, in recall order, take the observation budget exactly as recall's own tail would.
    When the over-fetched recall holds a retired observation at all, the retired ones may have taken
    a live one's place in it, so the live ones come instead from ``live_recall(i)`` (the fact's
    recall with the retired observations dropped inside the store's arms, under the real budgets).
    Retired observations are shown after the live ones and cost nothing: one whose successor is shown
    (at most one per successor: one link of each supersession chain), and up to
    ``_LIFECYCLE_UNLINKED_HISTORY_PER_FACT`` with no successor at all (a refuted claim, a split parent,
    one whose successor was deleted), so a fact restating it can be attached to it as history. The
    source facts are those of the live observations, under the real source budget; retired ones are
    shown without their sources.
    """
    ids = list(dict.fromkeys(str(o.id) for r in recalls for o in r.results))
    store = get_memories()
    claims: dict[str, ClaimState] = {}
    if ids:
        async with acquire_with_retry(pool) as conn:
            claims = await store.claims_for(conn, bank_id=bank_id, memory_ids=ids)

    def retired(observation_id: str) -> bool:
        claim = claims.get(observation_id)
        return claim is not None and claim.truth_status in RETIRED_STATUSES

    crowded = [i for i, recall in enumerate(recalls) if any(retired(str(o.id)) for o in recall.results)]
    live_recalls: dict[int, "RecallResult"] = dict(
        zip(crowded, await _gather_or_cancel([live_recall(i) for i in crowded]))
    )
    unread = list(dict.fromkeys(str(o.id) for r in live_recalls.values() for o in r.results if str(o.id) not in claims))
    if unread:
        async with acquire_with_retry(pool) as conn:
            claims.update(await store.claims_for(conn, bank_id=bank_id, memory_ids=unread))

    shaped: list["RecallResult"] = []
    entries: dict[str, claim_lifecycle.PoolEntry] = {}
    for index, recall in enumerate(recalls):
        live_source = live_recalls.get(index, recall)
        live = [o for o in live_source.results if not retired(str(o.id))]
        selection = select_facts_within_budget(
            fact_ids_ordered=[str(o.id) for o in live],
            text_by_id={str(o.id): o.text for o in live},
            max_tokens=config.consolidation_max_tokens,
            count_tokens=count_tokens,
        )
        admitted_ids = set(selection.ids)
        admitted = [o for o in live if str(o.id) in admitted_ids]
        history: list["MemoryFact"] = []
        successors_shown: set[str] = set()
        unlinked = 0
        for o in recall.results:
            claim = claims.get(str(o.id))
            if not retired(str(o.id)) or claim is None:
                continue
            successor = claim.superseded_by_id
            if successor is None:
                if claim.retired_via != "merges" and unlinked < _LIFECYCLE_UNLINKED_HISTORY_PER_FACT:
                    unlinked += 1
                    history.append(o)
            elif successor in admitted_ids and successor not in successors_shown:
                successors_shown.add(successor)
                history.append(o)
        source_facts = live_source.source_facts or {}
        by_observation = {str(o.id): list(dict.fromkeys(o.source_fact_ids or [])) for o in admitted}
        source_selection = select_source_facts_within_budget(
            source_ids_ordered=list(dict.fromkeys(s for sids in by_observation.values() for s in sids)),
            source_fact_ids_by_obs=by_observation,
            text_by_id={sid: sf.text for sid, sf in source_facts.items()},
            max_total_tokens=config.consolidation_source_facts_max_tokens,
            max_tokens_per_observation=config.consolidation_source_facts_max_tokens_per_observation,
            count_tokens=count_tokens,
        )
        shaped.append(
            recall.model_copy(
                update={
                    "results": [*admitted, *history],
                    "source_facts": {sid: source_facts[sid] for sid in source_selection.ids if sid in source_facts},
                }
            )
        )
        for o in [*admitted, *history]:
            entries.setdefault(str(o.id), claim_lifecycle.pool_entry(o, claims.get(str(o.id)), scope_tags))
    return _LifecyclePool(recalls=shaped, entries=entries)


#: Retain-metadata keys that say *where* a fact holds rather than what it says. The same
#: setting in two environments (reminders at 06:00 on staging, at 09:00 on dev) is two true
#: facts; with no environment in the prompt the second reads as a state change and the model
#: rewrites the first in place. ``repository`` is the same for code: the coding-agents
#: integration stamps it on everything it retains, and a service is often named only that
#: way, so without it two services' values of one setting read as one setting changing.
#: Shown as each fact's ``scope``, which the SCOPES STAY APART processing rule keeps apart.
#: Only scope keys reach the prompt (these, and the server's lifecycle scope keys, see
#: :func:`_prompt_scope_keys`): the rest of a fact's metadata is mostly provenance (source,
#: session, harness, branch) that would spend tokens on every fact line and invite splits on
#: what is not a scope.
_SCOPE_METADATA_KEYS = ("environment", "service", "module", "repository")
#: A scope value is an identifier; cap it so a pasted blob in metadata cannot bloat every line.
_SCOPE_VALUE_MAX_CHARS = 120


def _prompt_scope_keys() -> tuple[str, ...]:
    """The keys the prompt renders as a fact's scope: :data:`_SCOPE_METADATA_KEYS`, then any
    further key in ``HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS``.

    Lifecycle validation drops a label whose claim and target differ on one of the server's scope
    keys, so the model is shown every key it compares; a key it cannot see would get its labels
    dropped for a difference it was never told about. Every mode renders them all: the setting is
    server-level, and a scope the model can see keeps two values apart without lifecycle too.
    """
    return tuple(dict.fromkeys((*_SCOPE_METADATA_KEYS, *lifecycle_scope_keys(get_config()))))


def _scope_label(metadata: dict[str, Any] | None, scope_keys: Sequence[str]) -> str | None:
    """``key=value, ...`` over the fact's non-empty ``scope_keys``, in that order, or None when it
    names no scope.

    Whitespace is collapsed so a value cannot break the one-fact-per-line input format.
    """
    if not metadata:
        return None
    parts: list[str] = []
    for key in scope_keys:
        value = metadata.get(key)
        words = str(value).split() if value is not None else []
        if words:
            parts.append(f"{key}={' '.join(words)[:_SCOPE_VALUE_MAX_CHARS]}")
    return ", ".join(parts) or None


def _build_observations_for_llm(
    observations: "list[MemoryFact]",
    source_facts: "dict[str, MemoryFact]",
    lifecycle_pool: Mapping[str, claim_lifecycle.PoolEntry] | None = None,
    *,
    scope_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Serialize MemoryFact observations into dicts for the consolidation LLM prompt.

    ``scope_keys`` are the metadata keys a source fact's ``scope`` shows; None reads them with
    :func:`_prompt_scope_keys`.

    ``lifecycle_pool`` (lifecycle mode) adds each entry's ``status``, ``successor`` and
    ``retirable``, the ids of a retirable entry's source facts (what ``retires_source_ids`` /
    ``keeps_source_ids`` name), and shows a retired entry without its sources.
    """
    if scope_keys is None:
        scope_keys = _prompt_scope_keys()
    obs_list = []
    for obs in observations:
        # Rows written before #4799 may repeat an id thousands of times; show each source once.
        unique_ids = list(dict.fromkeys(obs.source_fact_ids or []))
        obs_data: dict[str, Any] = {
            "id": obs.id,
            "text": obs.text,
            "proof_count": len(unique_ids) or 1,
        }
        entry = lifecycle_pool.get(str(obs.id)) if lifecycle_pool is not None else None
        if entry is not None:
            obs_data["status"] = entry.status.value
            if entry.successor is not None:
                obs_data["successor"] = entry.successor
            obs_data["retirable"] = entry.retirable
        if obs.occurred_start:
            obs_data["occurred_start"] = obs.occurred_start
        if obs.occurred_end:
            obs_data["occurred_end"] = obs.occurred_end
        if obs.mentioned_at:
            obs_data["mentioned_at"] = obs.mentioned_at
        source_memories = []
        for sid in unique_ids if entry is None or not entry.retired else []:
            sf = source_facts.get(sid)
            if sf is None:
                continue
            sf_data: dict[str, Any] = (
                {"id": sid, "text": sf.text} if entry is not None and entry.retirable else {"text": sf.text}
            )
            if sf.context:
                sf_data["context"] = truncate_context_for_prompt(sf.context)
            if scope := _scope_label(sf.metadata, scope_keys):
                sf_data["scope"] = scope
            if sf.occurred_start:
                sf_data["occurred_start"] = sf.occurred_start
            if sf.occurred_end:
                sf_data["occurred_end"] = sf.occurred_end
            if sf.mentioned_at:
                sf_data["mentioned_at"] = sf.mentioned_at
            source_memories.append(sf_data)
        if source_memories:
            obs_data["source_memories"] = source_memories
        obs_list.append(obs_data)
    return obs_list


def _dedupe_updates(updates: list[_UpdateAction], *, batch_label: str) -> list[_UpdateAction]:
    """Collapse `updates` that target the same `observation_id`.

    LLMs occasionally emit several update entries for one observation in a
    single response (one per facet drawn from the same fact). Without
    deduplication the downstream loop would issue separate DB writes for each
    and the last write would silently overwrite the earlier ones. We keep the
    last text (the LLM's most recent attempt) and union all contributing
    `source_fact_ids`, then warn so the misbehavior is visible in logs.
    """
    if len(updates) < 2:
        return list(updates)

    by_id: dict[str, _UpdateAction] = {}
    collisions = 0
    for upd in updates:
        existing = by_id.get(upd.observation_id)
        if existing is None:
            by_id[upd.observation_id] = upd
            continue
        collisions += 1
        merged_ids = list(dict.fromkeys([*existing.source_fact_ids, *upd.source_fact_ids]))
        if isinstance(upd, _LifecycleUpdateAction) or isinstance(existing, _LifecycleUpdateAction):
            # Lifecycle labels survive the collapse: the last relation the model gave, and every
            # source id either entry named.
            by_id[upd.observation_id] = _LifecycleUpdateAction(
                text=upd.text,
                observation_id=upd.observation_id,
                source_fact_ids=merged_ids,
                reason=upd.reason,
                relation=getattr(upd, "relation", None) or getattr(existing, "relation", None),
                retires_source_ids=[
                    *getattr(existing, "retires_source_ids", []),
                    *getattr(upd, "retires_source_ids", []),
                ],
                keeps_source_ids=[*getattr(existing, "keeps_source_ids", []), *getattr(upd, "keeps_source_ids", [])],
            )
            continue
        by_id[upd.observation_id] = _UpdateAction(
            text=upd.text,
            observation_id=upd.observation_id,
            source_fact_ids=merged_ids,
        )

    if collisions:
        logger.warning(
            f"[CONSOLIDATION] {batch_label}: LLM emitted {collisions} duplicate update(s) targeting "
            f"the same observation_id ({len(updates)} updates -> {len(by_id)} after dedup). "
            "Kept the last text and unioned source_fact_ids."
        )

    return list(by_id.values())


# Backoff for the OUTER batch retry ladder (the provider runs its own, independently).
# Deliberately short: it only has to ride out a blip, and the caller's adaptive
# bisection is the real recovery path for anything longer-lived.
_OUTER_RETRY_INITIAL_BACKOFF = 1.0
_OUTER_RETRY_MAX_BACKOFF = 8.0


class _BatchFailureClass(StrEnum):
    """How the batch retry loop must treat an exception from the LLM call."""

    PROPAGATE = "propagate"
    """Not a batch failure at all — re-raise so the caller's handler sees it."""

    FAIL_FAST = "fail_fast"
    """A re-send of the identical payload cannot help; fail the batch immediately."""

    RETRY = "retry"
    """Transport-shaped; an unchanged re-send may well succeed."""


def _classify_batch_failure(exc: Exception) -> _BatchFailureClass:
    """Decide how ``_consolidate_batch_with_llm`` should react to ``exc``.

    Before #3684 the loop caught bare ``Exception`` and retried everything on one
    ladder, on top of the provider's own. Three problems, in descending severity:

    1. ``ProviderRateLimitResetError`` is a *control signal*, not a failure: the
       provider told us when quota reopens, and ``execute_task`` turns it into a
       ``DeferOperation`` that reschedules the job for exactly then. Catching it
       here meant the defer never fired — the batch was reported failed, adaptive
       bisection re-hit the same quota wall on every sub-batch, and each memory
       ended up stamped ``consolidation_failed_at``. That flag is the exclusion
       predicate for pending consolidation and is cleared only by an explicit
       bank-wide reset, so a *transient* quota exhaustion permanently orphaned
       those facts. ``fact_extraction`` re-raises it for the same reason.
    2. A 401/403 is a permanent server misconfiguration. Same shape as (1): every
       memory in the bank would be marked failed because a key was wrong.
    3. Malformed or schema-invalid output is input-shaped. Consolidation pins
       ``llm_temperature_consolidation`` (0.0 by default) and this loop rebuilds a
       byte-identical payload, so a re-send asks a greedy decoder the same question
       and gets the same answer — the reporter measured twelve failures at the same
       character offset. Retrying still costs a full generation, and for a JSON
       parse error that is on top of the four the provider already burned.

    ``FAIL_FAST`` is not "give up": the batch is still reported failed, so the
    caller's adaptive bisection halves it and tries again. That path *does* vary
    the input, which is what an input-shaped failure needs. It is the identical
    re-send at the same batch size that has nothing to offer.
    """
    if isinstance(exc, ProviderRateLimitResetError):
        return _BatchFailureClass.PROPAGATE
    # Duck-typed rather than importing a provider SDK's error class: every SDK we
    # front (openai, anthropic) exposes the HTTP status this way.
    if getattr(exc, "status_code", None) in (401, 403):
        return _BatchFailureClass.PROPAGATE
    if isinstance(exc, json.JSONDecodeError | ValidationError | OutputTooLongError):
        return _BatchFailureClass.FAIL_FAST
    # Providers that surface an empty/unusable body flag their own retryability.
    if getattr(exc, "retryable", None) is False:
        return _BatchFailureClass.FAIL_FAST
    return _BatchFailureClass.RETRY


async def _consolidate_batch_with_llm(
    llm_config: Any,
    memories: list[dict[str, Any]],
    union_observations: "list[MemoryFact]",
    union_source_facts: "dict[str, MemoryFact]",
    config: Any,
    remaining_observation_slots: int | None = None,
    max_observations_per_scope: int = -1,
    lifecycle_pool: Mapping[str, claim_lifecycle.PoolEntry] | None = None,
) -> _BatchLLMResult:
    """Single LLM call for a batch of facts against a pooled set of observations.

    ``lifecycle_pool`` (adaptive memory, lifecycle mode) switches to the lifecycle prompt variant and
    response schema, renders the pool's claim state, and leaves the observation cap to the caller,
    because a retirement in the same response frees a slot (``_cap_lifecycle_creates``).
    """
    if config is None:
        raise ValueError("config is required for _consolidate_batch_with_llm")
    scope_keys = _prompt_scope_keys()
    if union_observations:
        obs_list = _build_observations_for_llm(
            union_observations, union_source_facts, lifecycle_pool, scope_keys=scope_keys
        )
        # One observation per line, no indentation: the pool is nearly all of a batch's
        # uncached input, and `indent=2` spent a fifth of it on whitespace.
        observations_text = "[\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in obs_list) + "\n]"
    else:
        observations_text = "[]"

    def _fact_line(m: dict[str, Any]) -> str:
        text = f"[{m['id']}] {m['text']}"
        temporal_parts = []
        if m.get("occurred_start"):
            temporal_parts.append(f"occurred_start={m['occurred_start']}")
        if m.get("occurred_end"):
            temporal_parts.append(f"occurred_end={m['occurred_end']}")
        if m.get("mentioned_at"):
            temporal_parts.append(f"mentioned_at={m['mentioned_at']}")
        if temporal_parts:
            text += f" ({', '.join(temporal_parts)})"
        if scope := _scope_label(m.get("metadata"), scope_keys):
            text += f" [scope: {scope}]"
        return text

    facts_lines = "\n".join(_fact_line(m) for m in memories)

    # Build capacity note for the prompt when observation limit is configured
    observation_capacity_note: str | None = None
    if remaining_observation_slots is not None and max_observations_per_scope >= 0:
        if remaining_observation_slots == 0:
            observation_capacity_note = (
                f"OBSERVATION LIMIT REACHED ({max_observations_per_scope}/{max_observations_per_scope}). "
                "Only UPDATE or DELETE existing observations. Do NOT create new ones — "
                "merge new knowledge into existing observations via UPDATE."
            )
        elif remaining_observation_slots <= len(memories):
            observation_capacity_note = (
                f"This scope has {remaining_observation_slots} observation slot(s) remaining "
                f"(out of {max_observations_per_scope}). Prefer UPDATE over CREATE when possible."
            )

    # Split the prompt: a bank-agnostic system instruction (rules + input format +
    # decision guide + output format) that is byte-identical across batches AND
    # across banks, and a per-batch user message (mission + capacity note + facts +
    # existing observations). The split lets the system prefix be served from a
    # single Gemini context cache shared by every bank — the bank mission, capacity
    # note, and response_schema (all bank/batch-variable) are kept OUT of the
    # cached prefix so one cache serves all and it never busts within a run.
    lifecycle = lifecycle_pool is not None
    system_prompt = build_consolidation_system_prompt(
        llm_output_language=config.llm_output_language if config is not None else None,
        lifecycle=lifecycle,
    )
    user_content = build_consolidation_input(
        facts_text=facts_lines,
        observations_text=observations_text,
        observations_mission=config.observations_mission,
        observation_capacity_note=observation_capacity_note,
    )

    # Opt into context caching of the stable system prefix. response_schema is
    # NOT part of the cache key: it varies per batch (max_creates) but is not part
    # of the cached prefix, so keying on it would needlessly bust the cache.
    prompt_cache = PromptCachePrefix(system_instruction=system_prompt)

    # Constrain the schema: the observation cap on creates, and source_fact_ids to this batch's ids
    # (lifecycle mode: and the label lists to the retirable observations and their shown source ids)
    retirable = [e for e in lifecycle_pool.values() if e.retirable] if lifecycle_pool is not None else []
    response_model = _build_response_model(
        max_creates=remaining_observation_slots,
        fact_ids=[str(m["id"]) for m in memories],
        supports_max_items=config.llm_supports_max_items,
        lifecycle=lifecycle,
        target_ids=[e.id for e in retirable],
        source_ids=list(dict.fromkeys(s for e in retirable for s in e.source_ids if s in union_source_facts)),
    )

    max_attempts = config.consolidation_max_attempts
    inner_max_retries = config.consolidation_llm_max_retries
    last_exc: Exception | None = None
    attempts_made = 0
    failed_attempts = 0
    # Pre-compute a stable identifier set for the batch so failure logs name the
    # exact memories whose consolidation is failing — without this, an opaque
    # "LLM batch call failed" line gives operators no way to find the offending
    # input until adaptive bisection narrows the batch down to a single memory.
    memory_ids = [str(m.get("id")) for m in memories]
    if len(memory_ids) <= 5:
        ids_label = ", ".join(memory_ids)
    else:
        ids_label = f"{', '.join(memory_ids[:3])}, ... +{len(memory_ids) - 3} more"
    batch_label = f"{len(memory_ids)} memories [{ids_label}]"
    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            call_kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": response_model,
                "temperature": config.llm_temperature_consolidation,
                "scope": "consolidation",
                # Resolved per operation (HINDSIGHT_API_LLM_STRICT_SCHEMA_CONSOLIDATION, falling
                # back to the global flag) so an operator can grammar-enforce consolidation's
                # structured output -- which narrows the raw-JSON failure mode behind #2668 --
                # without forcing strict schema on operations whose model can't satisfy it.
                "strict_schema": config.llm_strict_schema_consolidation,
                "prompt_cache": prompt_cache,
            }
            # Only request an explicit output budget when configured. Left unset by default the key is
            # omitted, so each provider keeps its implicit default (backwards compatible). Operators on
            # providers with a low hidden cap (notably Bedrock imported models, which truncate structured
            # consolidation JSON) set HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS to fix it.
            if config.consolidation_max_completion_tokens is not None:
                call_kwargs["max_completion_tokens"] = config.consolidation_max_completion_tokens
            if inner_max_retries is not None:
                call_kwargs["max_retries"] = inner_max_retries
            batch_call = await llm_config.call(**call_kwargs)
            response: _ConsolidationBatchResponse = batch_call.content
            # A model recording a new fact sometimes names that fact as the UPDATE target. There is
            # no observation to update, and dropping it loses the fact: it is a create.
            fact_ids = {str(m["id"]) for m in memories}
            misaimed = [u for u in response.updates if u.observation_id in fact_ids]
            creates = response.creates + [
                _CreateAction(text=u.text, source_fact_ids=u.source_fact_ids, reason=u.reason) for u in misaimed
            ]
            # Defensive truncation: some LLM providers may not enforce JSON schema max_length. In
            # lifecycle mode the caller caps after the labels are planned (a retirement frees a slot).
            if not lifecycle and remaining_observation_slots is not None and remaining_observation_slots >= 0:
                if len(creates) > remaining_observation_slots:
                    logger.info(
                        f"[CONSOLIDATION] Truncating {len(creates)} creates to {remaining_observation_slots} "
                        f"(max_observations_per_scope={max_observations_per_scope})"
                    )
                    creates = creates[:remaining_observation_slots]
            updates = _dedupe_updates(
                [u for u in response.updates if u.observation_id not in fact_ids], batch_label=batch_label
            )
            return _BatchLLMResult(
                creates=creates,
                updates=updates,
                deletes=response.deletes,
                obs_count=len(union_observations),
                prompt_chars=len(system_prompt) + len(user_content),
                failed_attempts=failed_attempts,
            )
        except Exception as exc:
            failure_class = _classify_batch_failure(exc)
            # Count every failed call, including the ones adaptive bisection goes on to
            # rescue. `failed_consolidation` cannot show those — it is a gauge over rows
            # still carrying `consolidation_failed_at` when the run ends — so without this
            # a run that burned N schema-invalid calls and dropped every delete they
            # carried reads exactly like a clean one (#4151, #4152). Exception path only.
            get_metrics_collector().record_consolidation_batch_failure(
                failure_class=str(failure_class), error_type=type(exc).__name__
            )
            failed_attempts += 1
            if failure_class is _BatchFailureClass.PROPAGATE:
                logger.warning(
                    f"[CONSOLIDATION] LLM batch call for {batch_label} raised a non-batch failure "
                    f"({type(exc).__name__}); propagating to the task handler rather than marking "
                    f"the memories failed: {exc}"
                )
                raise
            last_exc = exc
            if failure_class is _BatchFailureClass.FAIL_FAST:
                logger.warning(
                    f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for "
                    f"{batch_label} with a non-retryable {type(exc).__name__}; not re-sending the "
                    f"identical payload: {exc}"
                )
                break
            logger.warning(
                f"[CONSOLIDATION] LLM batch call failed (attempt {attempt}/{max_attempts}) for {batch_label}: {exc}"
            )
            # Backoff on the outer ladder too. Without it a rate limit or a transient
            # overload was re-sent immediately, three times, defeating the point of the
            # provider's own backoff. Skipped after the final attempt — nothing follows it.
            if attempt < max_attempts:
                await asyncio.sleep(min(_OUTER_RETRY_INITIAL_BACKOFF * (2 ** (attempt - 1)), _OUTER_RETRY_MAX_BACKOFF))

    logger.error(
        f"[CONSOLIDATION] LLM batch call failed after {attempts_made}/{max_attempts} attempt(s) for "
        f"{batch_label}, skipping batch (the caller will bisect it). Last error: {last_exc}"
    )
    return _BatchLLMResult(
        obs_count=len(union_observations),
        prompt_chars=len(system_prompt) + len(user_content),
        failed=True,
        failed_attempts=failed_attempts,
    )


async def _apply_create_observation(
    conn,
    memory_engine: "MemoryEngine",
    bank_id: str,
    source_memory_ids: list[uuid.UUID],
    observation_text: str,
    embedding_str: str | None,
    tags: list[str] | None = None,
    event_date: datetime | None = None,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    mentioned_at: datetime | None = None,
    perf: ConsolidationPerfLog | None = None,
) -> dict[str, Any]:
    """Insert one observation from pre-embedded text.

    Runs on the caller's connection inside the caller's transaction, so it commits together
    with every other write derived from the same LLM response (#3876). The embedding is
    computed by the caller's prepare phase — a slow embedder must never pin a pooled
    connection, let alone one holding an open transaction.
    """
    now = datetime.now(timezone.utc)
    obs_event_date = event_date or now
    obs_occurred_start = occurred_start
    obs_occurred_end = occurred_end
    obs_mentioned_at = mentioned_at or now
    obs_tags = tags or []
    observation_id = uuid.uuid4()

    # Write the observation. A SQL store keeps it as a `memory_units` row (with the
    # search_vector the configured backend needs); a store that owns its rows takes it
    # as a normal Observation-type memory carrying all of its own state.
    store = get_memories()
    # FOR SHARE liveness + INSERT share the batch's transaction, so a concurrent
    # delete cannot orphan the new observation between the check and the insert.
    live_source_memory_ids = await _filter_live_source_memories(conn, bank_id, source_memory_ids)
    if not live_source_memory_ids:
        logger.debug(f"Create skipped: all {len(source_memory_ids)} source memories were deleted concurrently")
        return {"action": "skipped", "reason": "sources_deleted"}
    # Each source once: the store stores len() of this list as proof_count. It used to
    # write a literal 1, so a multi-source observation was undercounted until updated (#4955).
    source_memory_ids = list(dict.fromkeys(live_source_memory_ids))

    t0 = time.perf_counter()
    created_id = await store.insert_observation(
        conn=conn,
        ops=memory_engine._backend.ops,
        fq_table=fq_table,
        bank_id=bank_id,
        observation_id=observation_id,
        text=observation_text,
        embedding=embedding_str,
        source_memory_ids=source_memory_ids,
        tags=obs_tags,
        event_date=obs_event_date,
        occurred_start=obs_occurred_start,
        occurred_end=obs_occurred_end,
        mentioned_at=obs_mentioned_at,
        created_at=now,
    )

    if perf:
        perf.record_timing("db_write", time.perf_counter() - t0)

    logger.debug(f"Created observation {observation_id} from {len(source_memory_ids)} memories (tags: {obs_tags})")

    return {"action": "created", "observation_id": str(created_id), "tags": obs_tags}


def preview_consolidation_strategies(
    raw_strategies: list[Any],
    scopes: list[tuple[list[str], int]],
    *,
    sample_limit: int,
    complete: bool,
) -> "ConsolidationStrategiesPreview":
    """Which of ``scopes`` (``(tags, observation_count)``, most populous first) each
    strategy would apply to — using the same parsing, matching and
    first-strategy-wins rule consolidation uses, so the control plane never
    re-implements them.

    Aligned by position with ``raw_strategies``: a strategy the server would drop
    (no usable rule, or nothing set) keeps its slot, marked inactive, because the
    editor shows the list as typed. Each strategy is parsed on its own for that
    reason — parsing the list at once would shift every index after a dropped one.
    Pure: no I/O, so the endpoint's cost is the one scope query plus this loop.
    """
    from ..response_models import (
        ConsolidationStrategiesPreview,
        DefaultScopesPreview,
        StrategyPreview,
        StrategyRulePreview,
        StrategyScopePreview,
    )

    parsed = [(_parse_consolidation_strategies([entry]) or [None])[0] for entry in raw_strategies]

    # Match every rule against every scope exactly once, then derive both the
    # per-rule counts and each scope's winner from those hit sets. Matching twice
    # (once for winners via Strategy.claims, once per rule) doubled the work, and
    # this runs on the event loop as the user types. A strategy claims a scope when
    # any of its *valid* rules hits it — the same thing Strategy.claims computes.
    rule_patterns: list[list[_ScopePattern | None]] = []
    rule_hits: list[list[list[int]]] = []
    for entry in raw_strategies:
        raw_rules = entry.get("scopes") if isinstance(entry, dict) else None
        patterns = [_parse_scope_pattern(raw_rule) for raw_rule in (raw_rules if isinstance(raw_rules, list) else [])]
        rule_patterns.append(patterns)
        rule_hits.append(
            [
                [k for k, (tags, _) in enumerate(scopes) if pattern.matches(tags)] if pattern is not None else []
                for pattern in patterns
            ]
        )

    winners: list[int | None] = [None] * len(scopes)
    for i, strategy in enumerate(parsed):
        if strategy is None:
            continue
        for hits in rule_hits[i]:
            for k in hits:
                if winners[k] is None:
                    winners[k] = i
    # A scope is won by the *earliest* claiming strategy; iterating strategies in
    # order and never overwriting gives exactly that.

    def sample(indices: list[int]) -> list[StrategyScopePreview]:
        return [
            StrategyScopePreview(tags=scopes[k][0], count=scopes[k][1], handled_by=winners[k])
            for k in indices[:sample_limit]
        ]

    strategies: list[StrategyPreview] = []
    for i in range(len(raw_strategies)):
        rules: list[StrategyRulePreview] = []
        for pattern, hits in zip(rule_patterns[i], rule_hits[i]):
            rules.append(
                StrategyRulePreview(
                    match_count=len(hits),
                    taken_count=sum(1 for k in hits if winners[k] != i),
                    observation_count=sum(scopes[k][1] for k in hits),
                    samples=sample(hits),
                )
            )
        strategies.append(
            StrategyPreview(
                active=parsed[i] is not None,
                claimed_count=sum(1 for winner in winners if winner == i),
                rules=rules,
            )
        )

    unclaimed = [k for k, winner in enumerate(winners) if winner is None]
    return ConsolidationStrategiesPreview(
        strategies=strategies,
        default=DefaultScopesPreview(
            match_count=len(unclaimed),
            observation_count=sum(scopes[k][1] for k in unclaimed),
            samples=sample(unclaimed),
        ),
        scopes_scanned=len(scopes),
        complete=complete,
    )
