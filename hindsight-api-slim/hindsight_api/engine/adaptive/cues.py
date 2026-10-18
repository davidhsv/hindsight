"""Historical cues: phrases that make an external recall ask about the past (adaptive memory lifecycle).

A lifecycle-mode recall shows the bank's default view, normally ``current``: superseded and refuted
claims are left out. A question about how things USED to be ("what did we use before the
migration?") needs the history view instead, and a caller that does not pass ``lifecycle_view``
cannot say so. With the bank flag ``adaptive_memory_historical_cues`` on, an external recall
(HTTP or MCP, never an internal one) is checked against the phrase patterns below and, on a match,
switches to the historical view, or to ``as_of`` when the query says "before <date>" and the
temporal analyzer resolved that date.

Pure and deliberately narrow: fixed phrase patterns, no model call, and only for short queries.
Consolidation recalls with whole fact texts as queries, and a long prompt from an agent is far
more likely to contain "previously" incidentally than to ask about history, so anything over
:data:`MAX_CUE_QUERY_TOKENS` words never fires.
"""

import re
from dataclasses import dataclass
from datetime import datetime

#: Queries longer than this (whitespace-separated words) never fire a cue.
MAX_CUE_QUERY_TOKENS = 30

# Name -> pattern, matched case-insensitively against the query. The name is what the trace and
# the ``historical_cue_fired{cue}`` metric report, so the set is a bounded label. Order decides
# which name is reported when several match.
_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("used_to", re.compile(r"\bused\s+to\b")),
    ("what_did_we_use", re.compile(r"\bwhat\s+did\s+we\s+use\b")),
    # "what was X before", "what database did we run before": up to three words between "what"
    # and the verb, so the noun the question is about may sit there.
    ("what_before", re.compile(r"\bwhat\b(?:\s+\w+){0,3}?\s+(?:was|were|did)\b.*\bbefore\b")),
    ("before_change", re.compile(r"\bbefore\s+the\s+(?:upgrade|migration|switch|change)\b")),
    ("previously", re.compile(r"\bprevious(?:ly)?\b")),
    ("formerly", re.compile(r"\bformerly\b")),
    ("originally", re.compile(r"\boriginally\b")),
    ("earlier_version", re.compile(r"\bearlier\s+version\b")),
    ("history_of", re.compile(r"\bhistory\s+of\b")),
    ("changed_from", re.compile(r"\bchanged\s+from\b")),
)

_BEFORE = re.compile(r"\bbefore\b")

#: The cue a query fires with no phrase from the list but "before" and a date the temporal analyzer
#: resolved ("which runtime ran before 2024-03-01?"): the claims as they stood at that date.
BEFORE_DATE_CUE = "before_date"

#: Every cue name a trace or metric can carry.
CUE_NAMES: tuple[str, ...] = (*(name for name, _ in _CUE_PATTERNS), BEFORE_DATE_CUE)


@dataclass(frozen=True)
class HistoricalCue:
    """A cue that fired: its name, and the instant to view the claims at when the query named one."""

    cue: str
    #: Set when the query says "before <date>" and the temporal analyzer resolved the date: the
    #: start of that date, so claims are shown as they stood just before it. None means "show the
    #: historical view" (every superseded claim, annotated).
    as_of: datetime | None = None


def detect_historical_cue(query: str, temporal_constraint_start: datetime | None = None) -> HistoricalCue | None:
    """The historical cue in ``query``, or None.

    ``temporal_constraint_start`` is the start of the date range the recall's temporal analyzer
    resolved from the query (None when it resolved none). It matters only for a query that says
    "before": "what did we use before 2024-03-01?" views the claims as of that date (and so does a
    query with no listed phrase, as :data:`BEFORE_DATE_CUE`); "what did we use before the
    migration?" names no date and gets the historical view.
    """
    if len(query.split()) > MAX_CUE_QUERY_TOKENS:
        return None
    text = query.casefold()
    says_before = _BEFORE.search(text) is not None
    as_of = temporal_constraint_start if says_before else None
    for name, pattern in _CUE_PATTERNS:
        if pattern.search(text):
            return HistoricalCue(cue=name, as_of=as_of)
    if as_of is not None:
        return HistoricalCue(cue=BEFORE_DATE_CUE, as_of=as_of)
    return None
