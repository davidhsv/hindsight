"""How a recalled memory is labelled for one query, decided from recall output alone.

Every arm is scored by the same rule, so nothing here may depend on server internals: a
raw fact is labelled by its ``document_id``; an observation by the documents of the source
facts it cites plus its text. The observation rule was fixed before any arm was run.

Observation classification for a query (``classify``), in this order:

1. **refuted** — every relevant cited document is a refuted-state document, or the text
   matches the query's refuted-value pattern and not its current-value pattern.
2. **stale** — every relevant cited document is a stale-state document (forbidden_stale or
   allowed_historical), or the text matches only the stale-value pattern.
3. **current** — it cites at least one current-state document (required_active or
   current_context), its text matches the current-value pattern (or the query has none), and
   it matches neither the stale- nor the refuted-value pattern.
4. **mixed** — anything else that cites a relevant document or matches a value pattern. The
   typical case is the base prompt's rule-4/10 in-place update, "runs on Java 25; it was
   upgraded from Java 21": it states the current value and the one it replaced together.
   Metrics count mixed as current for recall and precision and report its share separately
   (``mixed_rate``), never silently.
5. **other** — unrelated to the query's labels.

"Relevant cited documents" are the cited documents that appear in any of the query's label
sets; an observation that also absorbed an unrelated fact is judged on the related ones.
A raw fact is current / stale / refuted when its document is in that set, else other.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .model import Query, Scenario

ItemClass = Literal["current", "stale", "refuted", "mixed", "other"]


@dataclass(frozen=True)
class RankedItem:
    """One recall result, reduced to what scoring reads."""

    rank: int
    memory_id: str
    type: str
    text: str
    docs: tuple[str, ...]
    """A raw fact's own document; an observation's cited documents (sorted, deduplicated)."""
    tags: tuple[str, ...]
    score: float | None = None

    @property
    def is_observation(self) -> bool:
        return self.type == "observation"


@dataclass(frozen=True)
class QueryLabels:
    """A query with its labels expanded to document ids — what the classifier and the metrics
    read. ``doc_tags`` is the scenario's document → tags map, for scope membership."""

    query: Query
    groups: Mapping[str, tuple[str, ...]]
    doc_tags: Mapping[str, frozenset[str]]

    @classmethod
    def for_query(cls, scenario: Scenario, query: Query) -> QueryLabels:
        return cls(
            query=query,
            groups=scenario.groups,
            doc_tags={item.doc: frozenset(item.tags) for item in scenario.items},
        )

    def docs_of(self, label: str) -> frozenset[str]:
        return frozenset(self.groups.get(label, (label,)))

    def _expand(self, labels: Sequence[str]) -> frozenset[str]:
        return frozenset(doc for label in labels for doc in self.docs_of(label))

    @property
    def current_docs(self) -> frozenset[str]:
        return self._expand(self.query.required_active) | self._expand(self.query.current_context)

    @property
    def stale_docs(self) -> frozenset[str]:
        return self._expand(self.query.forbidden_stale) | self._expand(self.query.allowed_historical)

    @property
    def refuted_docs(self) -> frozenset[str]:
        return self._expand(self.query.forbidden_refuted)

    def in_scope(self, doc: str) -> bool:
        return frozenset(self.query.scope_tags) <= self.doc_tags.get(doc, frozenset())


def _matches(pattern: str | None, text: str) -> bool:
    return pattern is not None and re.search(pattern, text, re.IGNORECASE) is not None


def classify(item: RankedItem, labels: QueryLabels) -> ItemClass:
    """The item's class for this query; see the module docstring for the rule."""
    current, stale, refuted = labels.current_docs, labels.stale_docs, labels.refuted_docs
    if not item.is_observation:
        doc = item.docs[0] if item.docs else None
        if doc in current:
            return "current"
        if doc in stale:
            return "stale"
        if doc in refuted:
            return "refuted"
        return "other"

    query = labels.query
    cited = frozenset(item.docs) & (current | stale | refuted)
    says_current = _matches(query.current_pattern, item.text)
    says_stale = _matches(query.stale_pattern, item.text)
    says_refuted = _matches(query.refuted_pattern, item.text)

    if (cited and cited <= refuted) or (says_refuted and not says_current):
        return "refuted"
    if (cited and cited <= stale) or (says_stale and not says_current):
        return "stale"
    if cited & current and (says_current or query.current_pattern is None) and not says_stale and not says_refuted:
        return "current"
    if cited or says_current or says_stale or says_refuted:
        return "mixed"
    return "other"


def carries(item: RankedItem, docs: frozenset[str], classes: frozenset[ItemClass], labels: QueryLabels) -> bool:
    """Whether ``item`` presents one of ``docs`` in one of ``classes``.

    A raw fact carries its own document. An observation carries a cited document only when its
    class says it presents that state — "runs on Java 25" cites the upgrade fact but does not
    present the Java 21 state that fact also mentions.
    """
    if not docs & frozenset(item.docs):
        return False
    return not item.is_observation or classify(item, labels) in classes
