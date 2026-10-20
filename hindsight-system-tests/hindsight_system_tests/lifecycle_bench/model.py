"""The scenario vocabulary: what a bank is told, what the stub decides, what a query must see.

A scenario is data, not code. It names the documents a bank receives (each one labelled
memory, so a raw fact's label is its ``document_id``), the consolidation decision the
scripted model takes for each step — in abstract terms, so one decision can be written in
whichever wire encoding the arm under test advertises (see ``encoding.py``) — and the
labelled queries asked once every step has settled.

Time is expressed in days from ``DAY_ZERO`` so a scenario reads as a timeline; every
timestamp a bank or a query sees is derived from it, which is what keeps runs comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

DAY_ZERO = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)

QueryKind = Literal["current", "historical", "as_of"]
TruthStatus = Literal["active", "uncertain", "superseded", "refuted"]


def at_day(days: float) -> datetime:
    return DAY_ZERO + timedelta(days=days)


@dataclass(frozen=True)
class Item:
    """One retained document, which is one labelled memory."""

    doc: str
    """The ``document_id``, and the label every query refers to this memory by."""
    content: str
    """What is retained. Also the needle the extraction rule matches on, so it must be unique
    within its step."""
    facts: tuple[str, ...]
    """What the scripted extraction returns for this document."""
    day: float
    session: str
    """``metadata.session_id``: the evidence identity independent reinforcement keys on."""
    minute: int = 0
    """Minutes past the day's 09:00, so same-day items still have an order."""
    tags: tuple[str, ...] = ()
    source: str | None = None
    """``metadata.source``, the provenance a source-trust map would key on."""
    observation_scopes: str | None = None
    entities: tuple[str, ...] = ()

    @property
    def timestamp(self) -> datetime:
        return at_day(self.day) + timedelta(minutes=self.minute)


@dataclass(frozen=True)
class Create:
    """A new observation over the batch facts matching ``facts``."""

    text: str
    facts: tuple[str, ...]


@dataclass(frozen=True)
class Reinforce:
    """Batch facts restate an existing observation: attach them, keep its text."""

    target: str
    """A substring of the existing observation's text."""
    facts: tuple[str, ...]
    fallback_text: str
    """The observation to create instead when the target is not in the pool the model saw."""


@dataclass(frozen=True)
class Supersede:
    """The batch facts end the target's state (``supersedes``) or show it was never true
    (``refutes``)."""

    target: str
    text: str
    facts: tuple[str, ...]
    folded_text: str
    """The in-place rewrite the base prompt's rules 4 and 10 ask for: the new state, with the
    one it replaces folded into the same sentence."""
    relation: Literal["supersedes", "refutes"] = "supersedes"
    foreign_target: bool = False
    """Error injection: name the target by its id even though it is outside the pool the call
    showed (another scope), the way a model that miscopies an id from elsewhere would."""


@dataclass(frozen=True)
class Conflict:
    """The batch facts contradict the target and nothing says which side is right."""

    target: str
    text: str
    facts: tuple[str, ...]
    folded_text: str
    """The base in-place form: one observation stating both sides."""


@dataclass(frozen=True)
class Split:
    """A generalization stops holding as stated; it becomes two narrower observations."""

    target: str
    children: tuple[Create, ...]
    folded_text: str


@dataclass(frozen=True)
class Merge:
    """Two observations turn out to say the same thing: keep ``survivor``, fold ``merged``."""

    survivor: str
    merged: str
    facts: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Historical:
    """An old fact arrives after the state it describes was already replaced.

    The base prompt (rule 10) keeps the current value and folds the fact in as its earlier
    state; the lifecycle contract attaches it to the retired observation instead.
    """

    base_target: str
    base_text: str
    lifecycle_target: str
    facts: tuple[str, ...]


@dataclass(frozen=True)
class Ignore:
    """Batch facts that deliberately produce no observation."""

    facts: tuple[str, ...]


Decision = Create | Reinforce | Supersede | Conflict | Split | Merge | Historical | Ignore


def decision_facts(decision: Decision) -> tuple[str, ...]:
    """Every fact needle a decision cites — what decides whether it applies to a batch."""
    if isinstance(decision, Split):
        return tuple(needle for child in decision.children for needle in child.facts)
    return decision.facts


@dataclass(frozen=True)
class Step:
    """One retain call, then whatever consolidation it triggers, settled before the next."""

    items: tuple[Item, ...]
    decisions: tuple[Decision, ...]


@dataclass(frozen=True)
class ConflictSide:
    """One side of an unresolved conflict: its documents, and the value it asserts."""

    docs: tuple[str, ...]
    pattern: str | None = None


@dataclass(frozen=True)
class Query:
    """One labelled recall. Label fields hold document ids or scenario group names."""

    id: str
    text: str
    kind: QueryKind
    day: float
    """The ``query_timestamp``: the moment the question is asked."""
    required_active: tuple[str, ...] = ()
    forbidden_stale: tuple[str, ...] = ()
    forbidden_refuted: tuple[str, ...] = ()
    allowed_historical: tuple[str, ...] = ()
    current_context: tuple[str, ...] = ()
    """Current-state documents on a historical query. Not scored there; they tell the
    observation classifier which citations are the successor's."""
    current_pattern: str | None = None
    stale_pattern: str | None = None
    refuted_pattern: str | None = None
    as_of: float | None = None
    tags: tuple[str, ...] = ()
    tags_match: str = "any"
    scope_tags: tuple[str, ...] = ()
    """When set, the query is a scope query: results must sit inside this scope."""
    conflict_sides: tuple[ConflictSide, ...] = ()
    spaced: tuple[str, ...] = ()
    """Independent-reinforcement queries: the memory confirmed across spaced sessions …"""
    contrast: tuple[str, ...] = ()
    """… and the one repeated inside a single session, under an equal cue."""


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    covers: tuple[str, ...]
    """The lifecycle situations / error injections this bank exercises."""
    steps: tuple[Step, ...]
    queries: tuple[Query, ...]
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Label → documents, for a claim stated by several documents (any member counts)."""
    expected_states: dict[str, TruthStatus] = field(default_factory=dict)
    """Document → the claim state a perfect lifecycle would hold for its raw fact. Scored only
    where the server reports state (``lifecycle`` on GET memory)."""
    dedup_merges: tuple[tuple[str, str], ...] = ()
    """Pairs of observation needles the dedup adjudicator calls the same fact."""

    def docs_of(self, label: str) -> tuple[str, ...]:
        return self.groups.get(label, (label,))

    @property
    def items(self) -> tuple[Item, ...]:
        return tuple(item for step in self.steps for item in step.items)
