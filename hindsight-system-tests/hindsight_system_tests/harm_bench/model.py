"""The harm benchmark's vocabulary: memories in the shared bank, and the coding tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from ..context_bench.model import Category, Status

#: ``H``: a harmful-memory trap — the bank holds a plausible memory that is wrong for this task,
#: and the task text plus general knowledge is enough to answer. ``U``: the answer depends on a
#: project-specific memory in the bank. ``N``: no memory is relevant and the task text suffices.
Kind = Literal["H", "U", "N"]
KINDS: tuple[Kind, ...] = ("H", "U", "N")

PROJECT_NAMES = {"o": "orders-service", "p": "payment-service", "d": "insight-dashboard"}
REPOSITORIES = {"o": "acme/orders-service", "p": "acme/payment-service", "d": "acme/insight-dashboard"}

#: Every task is asked "now"; memory ages count back from here.
QUERY_TIME = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Memory:
    label: str
    """Ground-truth id; never sent to the server (the document id is a hash of it)."""
    category: Category
    text: str
    project: str | None
    """``o``/``p``/``d``, or ``None`` for the user's own memories that belong to no project."""
    module: str | None = None
    age_days: int = 30
    status: Status = "active"
    environment: str | None = None
    branch: str = "main"

    @property
    def document_id(self) -> str:
        """Opaque and distinct from context_bench's (salted), so neither benchmark's labels reach
        the server and the two banks can never share a document id."""
        return "h" + hashlib.sha256(f"harmbench:{self.label}".encode()).hexdigest()[:12]

    @property
    def timestamp(self) -> datetime:
        return QUERY_TIME - timedelta(days=self.age_days)

    @property
    def tags(self) -> list[str]:
        return [f"project:{self.project}"] if self.project else []

    @property
    def metadata(self) -> dict[str, str]:
        """Scope metadata a coding agent would attach; no ground truth."""
        out: dict[str, str] = {"branch": self.branch}
        if self.project:
            out["repository"] = REPOSITORIES[self.project]
        if self.module:
            out["module"] = self.module
        if self.environment:
            out["environment"] = self.environment
        return out


@dataclass(frozen=True)
class Task:
    id: str
    kind: Kind
    text: str
    """What the engineer asks; also the recall query."""
    project: str
    module: str | None
    reference: str
    """A short correct answer, shown to the judge only."""
    must_include: tuple[str, ...]
    """Every criterion must be met for success."""
    must_not: tuple[str, ...] = ()
    """Any criterion met fails the task."""
    harm_signature: str | None = None
    """``H`` only: what following the trap looks like. The judge's ``harm`` is defined by it."""
    trap_ids: tuple[str, ...] = ()
    """``H`` only: the bank memories that are wrong for this task and share its vocabulary."""
    useful_ids: tuple[str, ...] = ()
    """The bank memories the answer depends on: a ``U`` task's, or an H2 ``stale`` task's current value."""
    environment: str | None = None
    group: str | None = None
    """A finer task type within a suite (the H2 suite's ``cross_module``, ``stale``, ``joke_episode``,
    ``control``); ``None`` in the original suite."""
    notes: str = ""

    @property
    def repository(self) -> str:
        return REPOSITORIES[self.project]

    def context(self) -> dict[str, str]:
        """The request ``context`` a feature server receives; a base server ignores it."""
        out: dict[str, str] = {"repository": self.repository, "branch": "main"}
        if self.module:
            out["module"] = self.module
        if self.environment:
            out["environment"] = self.environment
        return out
