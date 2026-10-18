"""The payloads the stub returns, as models rather than hand-built dicts.

Two reasons these are typed. The extraction envelope has six required fields, most
of which are "N/A" in any given test, so writing them inline would bury the one or
two a test actually cares about. And a mistyped key in a raw dict does not fail
here — it fails several layers away as "all N facts returned by the LLM were
unusable", which is a long walk back to a typo.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from .rulebook import ChatRequest

FactType = Literal["world", "assistant"]


class Fact(BaseModel):
    """One extracted fact, mirroring the server's ``ExtractedFact``."""

    what: str
    when: str = "N/A"
    where: str = "N/A"
    who: str = "N/A"
    why: str = "N/A"
    fact_kind: Literal["event", "conversation"] = "conversation"
    """Whether this is a datable occurrence. Load-bearing: the extraction
    pipeline only keeps ``occurred_start``/``occurred_end`` for ``event`` facts
    and discards them for a ``conversation``, so a date without this set is
    silently dropped."""

    fact_type: FactType = "world"
    """The server's own distinction, not the docs' user-facing one: ``world`` covers
    objective facts (including the user's preferences and corrections), ``assistant``
    covers what the agent itself did."""

    entities: list[str] = Field(default_factory=list)
    occurred_start: str | None = None
    occurred_end: str | None = None
    from_attachments: list[int] | None = None
    """1-based numbers of the chunk's attachments this fact was read off — the edge
    that makes a fact (and what it is consolidated into) carry its attachments."""
    labels: dict[str, str | None] | None = None
    """The bank's entity-label values for this fact (``{"kind": "rule"}``), keyed by label
    group. Only meaningful when the bank configures ``entity_labels``; a group with
    ``tag: true`` turns its value into a ``key:value`` tag on the fact."""


class ExtractedFacts(BaseModel):
    """The ``FactExtractionResponse`` envelope."""

    facts: list[Fact] = Field(default_factory=list)


#: The lifecycle-mode label fields (adaptive memory, ``lifecycle``). Left out of the reply when
#: unset, so a story that never labels anything sends exactly the bytes it always did.
_LIFECYCLE_KEYS = frozenset(
    {
        "supersedes",
        "refutes",
        "refines",
        "conflicts_with",
        "split_from",
        "keeps_source_ids",
        "relation",
        "retires_source_ids",
    }
)


class _OmitsUnsetLabels(BaseModel):
    @model_serializer(mode="wrap")
    def _omit_unset_labels(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        return {k: v for k, v in data.items() if k not in _LIFECYCLE_KEYS or v not in (None, [])}


class Observation(_OmitsUnsetLabels):
    """One entry for ``Consolidation.creates``.

    The lifecycle labels (lifecycle mode only) name observations the prompt showed by id:
    ``supersedes`` / ``refutes`` retire them, ``refines`` / ``conflicts_with`` relate to them,
    two creates naming one ``split_from`` split it, ``keeps_source_ids`` carries named source
    facts of a replaced observation over to this one.
    """

    text: str
    source_fact_ids: list[str]
    reason: str = "system test"
    supersedes: list[str] = Field(default_factory=list)
    refutes: list[str] = Field(default_factory=list)
    refines: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    split_from: str | None = None
    keeps_source_ids: list[str] = Field(default_factory=list)


class ObservationUpdate(_OmitsUnsetLabels):
    """One entry for ``Consolidation.updates``. In lifecycle mode ``relation`` labels it: a
    ``supersedes`` / ``refutes`` update keeps the old observation as history and stores ``text``
    as its successor; ``historical`` attaches old facts to a retired observation."""

    text: str
    observation_id: str
    source_fact_ids: list[str]
    reason: str = "system test"
    relation: Literal["supersedes", "refutes", "historical"] | None = None
    retires_source_ids: list[str] = Field(default_factory=list)
    keeps_source_ids: list[str] = Field(default_factory=list)


class ObservationDelete(BaseModel):
    observation_id: str
    #: The observation that absorbed this one's content; its evidence moves there.
    merged_into: str | None = None
    reason: str = "system test"


class Consolidation(BaseModel):
    """The three arrays consolidation must always return.

    All three default to empty, which is the honest answer for a story that is not
    about consolidation: the facts arrived and none warranted an observation. A test
    still has to say so — it is a declaration, not a silent default.
    """

    creates: list[Observation] = Field(default_factory=list)
    updates: list[ObservationUpdate] = Field(default_factory=list)
    deletes: list[ObservationDelete] = Field(default_factory=list)


def fact(what: str, **fields: object) -> Fact:
    """``Fact(what=...)`` with the required-but-usually-irrelevant fields defaulted."""
    return Fact(what=what, **fields)  # type: ignore[arg-type]


def extracted(*facts: Fact) -> ExtractedFacts:
    return ExtractedFacts(facts=list(facts))


def consolidation(
    *,
    creates: list[Observation] | None = None,
    updates: list[ObservationUpdate] | None = None,
    deletes: list[ObservationDelete] | None = None,
) -> Consolidation:
    return Consolidation(creates=creates or [], updates=updates or [], deletes=deletes or [])


FACT_ID_IN_PROMPT = re.compile(r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]")


def fact_ids_in(prompt: str) -> list[str]:
    """The fact ids the consolidation prompt is asking about.

    Consolidation must cite the facts it drew on by id, and those ids are minted
    by the server during the retain that just happened — a test cannot write them
    down in advance. They arrive in the prompt as ``[uuid] fact text``, so a rule
    reads them back out of the request it is answering.
    """
    return FACT_ID_IN_PROMPT.findall(prompt)


def batch_fact_ids(request: "ChatRequest") -> list[str]:
    """The ids of the facts this consolidation call is about — the batch, not the
    worked examples the system prompt also writes as ``[uuid] text``."""
    text = request.user_text
    return fact_ids_in(text[text.rfind("### New facts") : text.rfind("### Existing observations")])


def batch_fact_id(request: "ChatRequest", needle: str) -> str | None:
    """The id of the batch fact whose text contains ``needle``, if it is in this batch."""
    text = request.user_text
    facts = text[text.rfind("### New facts") : text.rfind("### Existing observations")]
    return next((line[1:37] for line in facts.splitlines() if line.startswith("[") and needle in line), None)


def shown_observations(request: "ChatRequest") -> dict[str, str]:
    """``{text: id}`` of the existing observations this call was shown."""
    text = request.user_text
    shown = text[text.rfind("### Existing observations") + len("### Existing observations") :].strip()
    if not shown.startswith("["):
        return {}
    return {o["text"]: o["id"] for o in json.loads(shown[: shown.rindex("]") + 1])}


def shown_observation_entries(request: "ChatRequest") -> list[dict[str, object]]:
    """The existing observations this call was shown, as the prompt renders them (in lifecycle
    mode with ``status``, ``successor``, ``retirable`` and their source facts' ``id``)."""
    text = request.user_text
    shown = text[text.rfind("### Existing observations") + len("### Existing observations") :].strip()
    if not shown.startswith("["):
        return []
    return json.loads(shown[: shown.rindex("]") + 1])


def _shown_id(request: "ChatRequest", needle: str) -> str:
    """The id of the one shown observation whose text contains ``needle``."""
    matches = [str(o["id"]) for o in shown_observation_entries(request) if needle in str(o["text"])]
    if len(matches) != 1:
        raise AssertionError(f"expected one shown observation containing {needle!r}, found {len(matches)}")
    return matches[0]


def supersedes(text: str, *, replacing: str, citing: str | None = None) -> Callable[["ChatRequest"], Consolidation]:
    """A lifecycle-mode reply: the shown observation containing ``replacing`` changed state, and a
    labelled UPDATE records ``text`` as its successor, citing the batch fact containing ``citing``
    (every batch fact when None). The old observation stays as history.
    """
    return _labelled_update(text, replacing=replacing, citing=citing, relation="supersedes")


def refutes(text: str, *, replacing: str, citing: str | None = None) -> Callable[["ChatRequest"], Consolidation]:
    """Like :func:`supersedes`, for a shown observation the facts show was wrong."""
    return _labelled_update(text, replacing=replacing, citing=citing, relation="refutes")


def historical(*, restating: str, citing: str | None = None) -> Callable[["ChatRequest"], Consolidation]:
    """A lifecycle-mode reply: the batch fact(s) restate the retired shown observation containing
    ``restating`` — old news, attached to it as history with its text unchanged."""

    def build(request: ChatRequest) -> Consolidation:
        target = next(o for o in shown_observation_entries(request) if restating in str(o["text"]))
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text=str(target["text"]),
                    observation_id=str(target["id"]),
                    source_fact_ids=_cited(request, citing),
                    relation="historical",
                )
            ]
        )

    return build


def _cited(request: "ChatRequest", citing: str | None) -> list[str]:
    if citing is None:
        return batch_fact_ids(request)
    fact_id = batch_fact_id(request, citing)
    if fact_id is None:
        raise AssertionError(f"no batch fact contains {citing!r}")
    return [fact_id]


def _labelled_update(
    text: str, *, replacing: str, citing: str | None, relation: Literal["supersedes", "refutes"]
) -> Callable[["ChatRequest"], Consolidation]:
    def build(request: ChatRequest) -> Consolidation:
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text=text,
                    observation_id=_shown_id(request, replacing),
                    source_fact_ids=_cited(request, citing),
                    relation=relation,
                )
            ]
        )

    return build


def observes(text: str) -> Callable[["ChatRequest"], Consolidation]:
    """A consolidation reply drawing one observation from every fact in the prompt.

    Pair with ``llm.on_step("consolidate").answers_with(...)``. The ids have to be
    read out of the request because the server minted them during the retain that
    triggered this call.
    """

    def build(request: ChatRequest) -> Consolidation:
        return Consolidation(
            creates=[Observation(text=text, source_fact_ids=fact_ids_in(request.all_text), reason="system test")]
        )

    return build
