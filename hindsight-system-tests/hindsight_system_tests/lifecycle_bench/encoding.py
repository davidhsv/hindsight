"""From a scenario's abstract decision to the consolidation reply the stub sends.

Three layers, so the same decision can be replayed against any arm:

1. **Capabilities.** The request's own JSON schema says whether the server speaks the
   lifecycle contract (``supersedes`` / ``refutes`` on creates, ``relation`` on updates).
   An arm is never told which encoding to expect; it advertises it.
2. **Needle form.** A decision is encoded as a ``NeedleReply``: the reply's action structure
   with every fact and observation named by a substring of its *text*, never by a runtime
   UUID. Author-written decisions produce one; a frozen real-model fixture *is* one. That is
   what makes the payload source pluggable.
3. **Resolution.** At call time the needles are resolved against the batch facts and the
   observation pool the request actually showed, through the same parsing the
   ``payloads`` helpers use. A needle that resolves to nothing degrades the action the way a
   model that could not see the target would act, and the degradation is recorded — never
   silently dropped.

Base encodings (``--encoding``):

- ``base-update`` — a state change is an in-place UPDATE whose text folds the earlier state
  in, as the base prompt's rules 4 and 10 instruct ("runs on Java 25; upgraded from 21").
- ``base-delete`` — a state change is CREATE the new state + DELETE the old observation, the
  shape ``test_23_contradiction`` scripts.

Lifecycle encoding (only when the schema advertises it), following the normalised shape the
design review fixed: SUPERSEDE / REFUTE either as ``updates[].relation`` (paired with
``base-update``) or as ``creates[].supersedes`` / ``refutes`` (paired with ``base-delete``),
citing only this batch's facts, with no delete sent for the same target; an out-of-order old
fact is ``updates[].relation = "historical"`` onto the retired observation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..payloads import fact_ids_in
from ..rulebook import ChatRequest
from .model import (
    Conflict,
    Create,
    Decision,
    Historical,
    Ignore,
    Merge,
    Reinforce,
    Split,
    Supersede,
)

BaseEncoding = Literal["base-update", "base-delete"]
LifecycleForm = Literal["update", "create"]
Relation = Literal["supersedes", "refutes", "historical"]

#: The marker the openai-compatible provider puts in front of the schema when it is not
#: grammar-enforced (``llm_strict_schema_consolidation`` is off by default).
_SCHEMA_IN_PROMPT = "You must respond with valid JSON matching this schema:\n"

#: The lifecycle label fields; any one of them on the schema means "lifecycle".
LIFECYCLE_CREATE_FIELDS = frozenset({"supersedes", "refutes", "conflicts_with", "split_from", "refines"})
LIFECYCLE_UPDATE_FIELDS = frozenset({"relation", "retires_source_ids"})


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaCapabilities:
    """What the consolidation response schema of one request allows."""

    found: bool
    """Whether a schema was found at all (response_format, or embedded in the prompt)."""
    create_fields: frozenset[str] = frozenset()
    update_fields: frozenset[str] = frozenset()

    @property
    def lifecycle(self) -> bool:
        return bool(self.create_fields & LIFECYCLE_CREATE_FIELDS or self.update_fields & LIFECYCLE_UPDATE_FIELDS)

    def form(self, preferred: LifecycleForm) -> LifecycleForm:
        """The lifecycle form to send: ``preferred`` when the schema has it, else the other."""
        has_update = "relation" in self.update_fields
        has_create = "supersedes" in self.create_fields
        if preferred == "update" and not has_update and has_create:
            return "create"
        if preferred == "create" and not has_create and has_update:
            return "update"
        return preferred

    @classmethod
    def from_request(cls, request: ChatRequest) -> SchemaCapabilities:
        schema = _request_schema(request)
        if schema is None:
            return cls(found=False)
        return cls(
            found=True,
            create_fields=_item_properties(schema, "creates"),
            update_fields=_item_properties(schema, "updates"),
        )


def _request_schema(request: ChatRequest) -> dict[str, Any] | None:
    response_format = request.body.get("response_format")
    if isinstance(response_format, dict):
        schema = (response_format.get("json_schema") or {}).get("schema")
        if isinstance(schema, dict):
            return schema
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str) and _SCHEMA_IN_PROMPT in content:
            embedded = content[content.index(_SCHEMA_IN_PROMPT) + len(_SCHEMA_IN_PROMPT) :]
            try:
                parsed, _end = json.JSONDecoder().raw_decode(embedded.lstrip())
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _item_properties(schema: dict[str, Any], array: str) -> frozenset[str]:
    """Property names of the items of top-level array ``array``, following ``$ref``s."""
    definitions = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    def resolve(node: Any) -> dict[str, Any]:
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 10:
            node = definitions.get(node["$ref"].rsplit("/", 1)[-1], {})
            seen += 1
        if isinstance(node, dict) and "anyOf" in node:
            # Optional[list[...]] renders as anyOf [array, null]; take the array branch.
            node = next((resolve(branch) for branch in node["anyOf"] if resolve(branch).get("type") == "array"), {})
        return node if isinstance(node, dict) else {}

    array_schema = resolve(schema.get("properties", {}).get(array, {}))
    items = resolve(array_schema.get("items", {}))
    return frozenset(items.get("properties", {}))


# ---------------------------------------------------------------------------
# Needle form (also the frozen-fixture format)
# ---------------------------------------------------------------------------


class NeedleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    facts: list[str]
    supersedes: list[str] = Field(default_factory=list)
    refutes: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    split_from: str | None = None
    foreign_targets: bool = False
    reason: str = "memory_lifecycle benchmark"


class NeedleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    text: str = ""
    """The observation's new text; empty keeps the text the pool showed (a restatement)."""
    facts: list[str]
    relation: Relation | None = None
    foreign_target: bool = False
    if_missing: Literal["create", "drop"] = "create"
    """What a model that could not see the target does instead: write ``fallback_text`` (else
    ``text``) as a new observation, or — a restatement of old news — nothing."""
    fallback_text: str = ""
    reason: str = "memory_lifecycle benchmark"


class NeedleDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    merged_into: str | None = None
    foreign_target: bool = False
    reason: str = "memory_lifecycle benchmark"


class NeedleReply(BaseModel):
    """A consolidation reply with every id replaced by a text needle."""

    model_config = ConfigDict(extra="forbid")

    creates: list[NeedleCreate] = Field(default_factory=list)
    updates: list[NeedleUpdate] = Field(default_factory=list)
    deletes: list[NeedleDelete] = Field(default_factory=list)

    def extend(self, other: NeedleReply) -> None:
        self.creates.extend(other.creates)
        self.updates.extend(other.updates)
        self.deletes.extend(other.deletes)


def encode(
    decision: Decision,
    *,
    encoding: BaseEncoding,
    lifecycle: bool,
    form: LifecycleForm,
) -> NeedleReply:
    """One abstract decision in the needle form of the arm's encoding."""
    if isinstance(decision, Create):
        return NeedleReply(creates=[NeedleCreate(text=decision.text, facts=list(decision.facts))])
    if isinstance(decision, Ignore):
        return NeedleReply()
    if isinstance(decision, Reinforce):
        return NeedleReply(
            updates=[
                NeedleUpdate(target=decision.target, facts=list(decision.facts), fallback_text=decision.fallback_text)
            ]
        )
    if isinstance(decision, Supersede):
        return _encode_supersede(decision, encoding=encoding, lifecycle=lifecycle, form=form)
    if isinstance(decision, Conflict):
        if lifecycle:
            return NeedleReply(
                creates=[NeedleCreate(text=decision.text, facts=list(decision.facts), conflicts_with=[decision.target])]
            )
        if encoding == "base-update":
            return NeedleReply(
                updates=[
                    NeedleUpdate(
                        target=decision.target,
                        text=decision.folded_text,
                        facts=list(decision.facts),
                        fallback_text=decision.text,
                    )
                ]
            )
        # base-delete has no conflict concept either; without a winner it deletes nothing.
        return NeedleReply(creates=[NeedleCreate(text=decision.text, facts=list(decision.facts))])
    if isinstance(decision, Split):
        children = [NeedleCreate(text=child.text, facts=list(child.facts)) for child in decision.children]
        if lifecycle:
            for child in children:
                child.split_from = decision.target
            return NeedleReply(creates=children)
        if encoding == "base-update":
            facts = [needle for child in decision.children for needle in child.facts]
            return NeedleReply(updates=[NeedleUpdate(target=decision.target, text=decision.folded_text, facts=facts)])
        return NeedleReply(creates=children, deletes=[NeedleDelete(target=decision.target)])
    if isinstance(decision, Merge):
        # MERGE exists in the base contract (``deletes[].merged_into``), so every arm sends it.
        return NeedleReply(
            updates=[NeedleUpdate(target=decision.survivor, text=decision.text, facts=list(decision.facts))],
            deletes=[NeedleDelete(target=decision.merged, merged_into=decision.survivor)],
        )
    if isinstance(decision, Historical):
        if lifecycle:
            return NeedleReply(
                updates=[
                    NeedleUpdate(
                        target=decision.lifecycle_target,
                        facts=list(decision.facts),
                        relation="historical",
                        if_missing="drop",
                    )
                ]
            )
        return NeedleReply(
            updates=[NeedleUpdate(target=decision.base_target, text=decision.base_text, facts=list(decision.facts))]
        )
    raise TypeError(f"unknown decision {decision!r}")


def _encode_supersede(
    decision: Supersede, *, encoding: BaseEncoding, lifecycle: bool, form: LifecycleForm
) -> NeedleReply:
    facts = list(decision.facts)
    foreign = decision.foreign_target
    if lifecycle and form == "update":
        return NeedleReply(
            updates=[
                NeedleUpdate(
                    target=decision.target,
                    text=decision.text,
                    facts=facts,
                    relation=decision.relation,
                    foreign_target=foreign,
                    fallback_text=decision.text,
                )
            ]
        )
    if lifecycle:
        targets = [decision.target]
        create = NeedleCreate(
            text=decision.text,
            facts=facts,
            supersedes=targets if decision.relation == "supersedes" else [],
            refutes=targets if decision.relation == "refutes" else [],
            foreign_targets=foreign,
        )
        return NeedleReply(creates=[create])
    if encoding == "base-update":
        return NeedleReply(
            updates=[
                NeedleUpdate(
                    target=decision.target,
                    text=decision.folded_text,
                    facts=facts,
                    foreign_target=foreign,
                    fallback_text=decision.text,
                )
            ]
        )
    return NeedleReply(
        creates=[NeedleCreate(text=decision.text, facts=facts)],
        deletes=[NeedleDelete(target=decision.target, foreign_target=foreign)],
    )


# ---------------------------------------------------------------------------
# What the request showed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchFact:
    id: str
    text: str


class PoolEntry(BaseModel):
    """One existing observation as the consolidation prompt rendered it."""

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str
    status: str | None = None

    @property
    def active(self) -> bool:
        return self.status in (None, "active", "uncertain")


@dataclass(frozen=True)
class BatchView:
    """The facts a consolidation call asked about and the observations it showed."""

    facts: tuple[BatchFact, ...]
    pool: tuple[PoolEntry, ...]

    @classmethod
    def from_request(cls, request: ChatRequest) -> BatchView:
        text = request.user_text
        # The same slicing as ``payloads.batch_fact_ids``: the batch section only, not the
        # worked examples the system prompt also writes as ``[uuid] text``.
        section = text[text.rfind("### New facts") : text.rfind("### Existing observations")]
        ids = set(fact_ids_in(section))
        facts = tuple(
            BatchFact(id=line[1:37], text=line[38:].strip())
            for line in section.splitlines()
            if line.startswith("[") and line[1:37] in ids
        )
        shown = text[text.rfind("### Existing observations") + len("### Existing observations") :].strip()
        pool: tuple[PoolEntry, ...] = ()
        if shown.startswith("["):
            entries, _end = json.JSONDecoder().raw_decode(shown)
            pool = tuple(PoolEntry.model_validate(entry) for entry in entries)
        return cls(facts=facts, pool=pool)

    def fact_ids(self, needle: str) -> list[str]:
        """Every batch fact whose text contains ``needle`` — all of them, since a claim repeated
        within one session arrives as several identical facts."""
        return [fact.id for fact in self.facts if needle in fact.text]

    def matches(self, needles: Iterable[str]) -> bool:
        return any(self.fact_ids(needle) for needle in needles)

    def observation(self, needle: str, *, prefer_retired: bool = False) -> PoolEntry | None:
        candidates = [entry for entry in self.pool if needle in entry.text]
        if not candidates:
            return None
        preferred = [entry for entry in candidates if entry.active != prefer_retired]
        return (preferred or candidates)[0]


# ---------------------------------------------------------------------------
# Resolution to the wire
# ---------------------------------------------------------------------------


class WireCreate(BaseModel):
    text: str
    source_fact_ids: list[str]
    reason: str
    supersedes: list[str] | None = None
    refutes: list[str] | None = None
    conflicts_with: list[str] | None = None
    split_from: str | None = None


class WireUpdate(BaseModel):
    observation_id: str
    text: str
    source_fact_ids: list[str]
    reason: str
    relation: Relation | None = None


class WireDelete(BaseModel):
    observation_id: str
    reason: str
    merged_into: str | None = None


class WireReply(BaseModel):
    creates: list[WireCreate] = Field(default_factory=list)
    updates: list[WireUpdate] = Field(default_factory=list)
    deletes: list[WireDelete] = Field(default_factory=list)

    def to_json(self) -> str:
        # exclude_none: a base arm must receive exactly the base keys, never a null lifecycle field.
        return self.model_dump_json(exclude_none=True)


@dataclass
class Resolution:
    reply: WireReply
    degradations: list[str] = field(default_factory=list)
    """Actions the payload could not resolve against what the call showed."""
    notes: list[str] = field(default_factory=list)
    """Adjustments a sensible model would make anyway (not failures)."""


def resolve(needles: NeedleReply, batch: BatchView, foreign: Mapping[str, str]) -> Resolution:
    """Turn a needle reply into ids this request can accept.

    ``foreign`` maps observation text → id for the whole bank (read through the public API
    before the step), for error-injection actions that deliberately name an observation the
    call did not show.
    """
    out = Resolution(reply=WireReply())

    def facts_for(needles_: Sequence[str], what: str) -> list[str]:
        ids: list[str] = []
        for needle in needles_:
            found = batch.fact_ids(needle)
            if not found:
                out.degradations.append(f"{what}: fact needle {needle!r} not in batch")
            ids.extend(i for i in found if i not in ids)
        return ids

    def observation_id(needle: str, *, foreign_ok: bool, what: str, prefer_retired: bool = False) -> str | None:
        if foreign_ok:
            found = next((oid for text, oid in foreign.items() if needle in text), None)
            if found is None:
                out.degradations.append(f"{what}: foreign target {needle!r} not in bank")
            return found
        entry = batch.observation(needle, prefer_retired=prefer_retired)
        if entry is None:
            out.degradations.append(f"{what}: target {needle!r} not in pool")
            return None
        return entry.id

    for create in needles.creates:
        cited = facts_for(create.facts, "create")
        if not cited:
            out.degradations.append(f"create {create.text!r} dropped: no cited fact in batch")
            continue
        plain = not (create.supersedes or create.refutes or create.conflicts_with or create.split_from)
        twin = next((entry for entry in batch.pool if entry.active and entry.text == create.text), None)
        if plain and twin is not None:
            # The server splits one step's facts across LLM batches of
            # ``consolidation_llm_batch_size``; by the second batch the observation the first one
            # created is in the pool, and a model attaches the rest to it (rule 1) instead of
            # creating its exact twin — which the server would discard as unaddressed facts.
            out.reply.updates.append(
                WireUpdate(observation_id=twin.id, text=twin.text, source_fact_ids=cited, reason=create.reason)
            )
            out.notes.append(f"create {create.text!r} attached to its existing twin")
            continue

        def targets(names: list[str]) -> list[str] | None:
            ids = [
                oid
                for oid in (observation_id(n, foreign_ok=create.foreign_targets, what="create") for n in names)
                if oid is not None
            ]
            return ids or None

        split_from = (
            observation_id(create.split_from, foreign_ok=False, what="split") if create.split_from is not None else None
        )
        out.reply.creates.append(
            WireCreate(
                text=create.text,
                source_fact_ids=cited,
                reason=create.reason,
                supersedes=targets(create.supersedes),
                refutes=targets(create.refutes),
                conflicts_with=targets(create.conflicts_with),
                split_from=split_from,
            )
        )

    for update in needles.updates:
        cited = facts_for(update.facts, "update")
        if not cited:
            out.degradations.append(f"update of {update.target!r} dropped: no cited fact in batch")
            continue
        target = observation_id(
            update.target,
            foreign_ok=update.foreign_target,
            what="update",
            prefer_retired=update.relation == "historical",
        )
        if target is None:
            fallback = update.fallback_text or update.text
            if update.if_missing == "create" and fallback:
                out.reply.creates.append(WireCreate(text=fallback, source_fact_ids=cited, reason=update.reason))
                out.degradations.append(f"update of {update.target!r} became a create")
            continue
        text = update.text or next((entry.text for entry in batch.pool if entry.id == target), "")
        out.reply.updates.append(
            WireUpdate(
                observation_id=target, text=text, source_fact_ids=cited, reason=update.reason, relation=update.relation
            )
        )

    for delete in needles.deletes:
        target = observation_id(delete.target, foreign_ok=delete.foreign_target, what="delete")
        if target is None:
            continue
        merged_into = None
        if delete.merged_into is not None:
            merged_into = observation_id(delete.merged_into, foreign_ok=False, what="merge")
            if merged_into is None:
                continue
        out.reply.deletes.append(WireDelete(observation_id=target, merged_into=merged_into, reason=delete.reason))

    return out


# ---------------------------------------------------------------------------
# Frozen real-model fixtures
# ---------------------------------------------------------------------------


class CapturedCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    refutes: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    split_from: str | None = None
    reason: str = ""


class CapturedUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observation_id: str
    text: str = ""
    source_fact_ids: list[str] = Field(default_factory=list)
    relation: Relation | None = None
    reason: str = ""


class CapturedDelete(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observation_id: str
    merged_into: str | None = None
    reason: str = ""


class CapturedReply(BaseModel):
    """A real model's consolidation reply as the server received it (ids, not needles)."""

    model_config = ConfigDict(extra="ignore")

    creates: list[CapturedCreate] = Field(default_factory=list)
    updates: list[CapturedUpdate] = Field(default_factory=list)
    deletes: list[CapturedDelete] = Field(default_factory=list)


def needleize(raw: CapturedReply, batch: BatchView) -> NeedleReply:
    """A captured real-model reply in needle form, keyed by the full texts the request showed
    — the shape a frozen fixture stores, replayable on any later run whatever ids it mints."""
    fact_text = {fact.id: fact.text for fact in batch.facts}
    observation_text = {entry.id: entry.text for entry in batch.pool}

    def obs(oid: str) -> str:
        return observation_text.get(oid, oid)

    return NeedleReply(
        creates=[
            NeedleCreate(
                text=create.text,
                facts=[fact_text.get(fid, fid) for fid in create.source_fact_ids],
                supersedes=[obs(oid) for oid in create.supersedes],
                refutes=[obs(oid) for oid in create.refutes],
                conflicts_with=[obs(oid) for oid in create.conflicts_with],
                split_from=None if create.split_from is None else obs(create.split_from),
                reason=create.reason,
            )
            for create in raw.creates
        ],
        updates=[
            NeedleUpdate(
                target=obs(update.observation_id),
                text=update.text,
                facts=[fact_text.get(fid, fid) for fid in update.source_fact_ids],
                relation=update.relation,
                reason=update.reason,
            )
            for update in raw.updates
        ],
        deletes=[
            NeedleDelete(
                target=obs(delete.observation_id),
                merged_into=None if delete.merged_into is None else obs(delete.merged_into),
                reason=delete.reason,
            )
            for delete in raw.deletes
        ],
    )


@dataclass(frozen=True, order=True)
class ActionKey:
    """One action's structure: what the modal over captured replies compares."""

    kind: str
    target: str
    relation: str
    facts: tuple[str, ...]


def action_structure(reply: NeedleReply) -> tuple[ActionKey, ...]:
    """The structure the modal of K captured replies is taken over: per action its type, its
    target observation needle, its relation label and its cited fact needles — wording of the
    observation text deliberately excluded, so two replies that agree on what to do but
    phrase it differently count as the same decision."""
    keys: list[ActionKey] = []
    for create in reply.creates:
        if create.supersedes or create.refutes:
            relation = "supersedes" if create.supersedes else "refutes"
            target = (create.supersedes or create.refutes)[0]
        elif create.conflicts_with:
            relation, target = "conflicts_with", create.conflicts_with[0]
        elif create.split_from is not None:
            relation, target = "split_from", create.split_from
        else:
            relation, target = "", ""
        keys.append(ActionKey("create", target, relation, tuple(sorted(create.facts))))
    for update in reply.updates:
        keys.append(ActionKey("update", update.target, update.relation or "", tuple(sorted(update.facts))))
    for delete in reply.deletes:
        keys.append(ActionKey("delete", delete.target, "merged_into" if delete.merged_into else "", ()))
    return tuple(sorted(keys))


def modal_reply(replies: Sequence[NeedleReply]) -> NeedleReply:
    """The reply whose action structure occurs most often among ``replies``; ties go to the
    structure seen first, and the first reply with the modal structure is returned."""
    if not replies:
        raise ValueError("modal_reply needs at least one reply")
    structures = [action_structure(reply) for reply in replies]
    counts = Counter(structures)
    best = max(counts.values())
    modal = next(structure for structure in structures if counts[structure] == best)
    return replies[structures.index(modal)]
