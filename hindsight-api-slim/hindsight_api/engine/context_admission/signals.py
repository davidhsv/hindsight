"""Memory-reference signals: phrases that say the task is about something already shared.

"What did we decide about retries?" or "like last time" asks for a memory by its nature, so an
automatic-context recall can afford a lower bar for it. Pure and deliberately narrow: fixed phrase
patterns, no model call. A bare "remember" is not one ("remember to run the tests" is an
instruction), so the pattern needs a question or a pronoun next to it.
"""

import re

# Name -> pattern, matched case-insensitively. The name is what the trace reports. Order decides
# which name is reported when several match.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "remember",
        re.compile(
            r"\b(?:do|did|can|could|would)\s+you\s+remember\b"
            r"|\bremember\s+(?:when|what|how|why|which|where|who|the\s+time|that\s+time|we|our|us)\b"
        ),
    ),
    ("last_time", re.compile(r"\b(?:last|previous|that)\s+time\b")),
    ("previously", re.compile(r"\bpreviously\b")),
    (
        "what_did_we_decide",
        re.compile(r"\bwhat\s+(?:did|have)\s+we\s+(?:decide|decided|agree|agreed|choose|chose|settle|settled)\b"),
    ),
    ("we_discussed", re.compile(r"\bwe\s+(?:discussed|talked\s+about|agreed\s+on|decided\s+on|went\s+over)\b")),
    ("as_mentioned", re.compile(r"\bas\s+(?:we|i|you)\s+(?:said|mentioned|discussed|noted)\b")),
)


def detect_reference_signal(query: str) -> str | None:
    """The name of the first memory-reference phrase in ``query``, or None."""
    text = query.casefold()
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            return name
    return None
