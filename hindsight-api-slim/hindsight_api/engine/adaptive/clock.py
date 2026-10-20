"""The adaptive-memory write clock.

Every timestamp an adaptive-memory writer stores (event times, ``processed_at``, ``updated_at``,
``valid_to``, ``below_threshold_since``) comes from here and is bound as a SQL parameter, never
from the database's ``now()``: a DB-side clock cannot be moved by a test, and Oracle's
``SYSTIMESTAMP`` would be a second, unsynchronised one.

Callers reach it as ``clock.utcnow()`` (module attribute, not a ``from`` import) so a test can
``monkeypatch.setattr(clock, "utcnow", ...)`` and move every writer at once.

Recall scoring does NOT use this: it is anchored on ``_recall_scoring_now(question_date)`` so a
caller-supplied ``query_timestamp`` moves recency and retrievability together.
"""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current time, tz-aware UTC."""
    return datetime.now(UTC)
