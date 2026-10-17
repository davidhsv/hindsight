"""The scenario vocabulary: memories, queries, and the scope they carry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

Category = Literal[
    "durable_fact",
    "architectural_decision",
    "procedure",
    "preference",
    "historical_event",
    "debugging_episode",
    "casual_context",
    "humor",
    "speculation",
    "failed_attempt",
]

#: Belief state of a memory. ``superseded``: a newer memory replaced it; ``refuted``: a later
#: memory showed it was wrong. Both are irrelevant to a present-state task and are what the
#: stale/refuted admission rates count.
Status = Literal["active", "superseded", "refuted"]

Purpose = Literal["auto_context", "explicit_search", "historical_search"]

QueryCategory = Literal[
    "none_relevant",
    "all_candidates_irrelevant",
    "cross_project",
    "wrong_subsystem",
    "humor_casual",
    "stale_superseded",
    "exact_historical_cue",
    "useful_old",
    "multi_memory",
    "coding_session",
]

QUERY_CATEGORIES: tuple[QueryCategory, ...] = (
    "none_relevant",
    "all_candidates_irrelevant",
    "cross_project",
    "wrong_subsystem",
    "humor_casual",
    "stale_superseded",
    "exact_historical_cue",
    "useful_old",
    "multi_memory",
    "coding_session",
)

PROJECT_NAMES = {"a": "Payments API", "b": "FlowKeeper", "c": "audit-api"}
REPOSITORIES = {"a": "acme/payments-api", "b": "acme/flowkeeper", "c": "acme/audit-api"}

MAIN_BANK = "main"
SESSION_BANK = "session"

#: Every query is asked "now"; memory ages count back from here.
QUERY_TIME = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


@dataclass(frozen=True)
class Memory:
    label: str
    """Ground-truth id; never sent to the server (the document id is a hash of it)."""
    category: Category
    text: str
    project: str | None
    """``a``/``b``/``c``, or ``None`` for a memory that belongs to no project (the user's own)."""
    module: str | None = None
    age_days: int = 30
    status: Status = "active"
    environment: str | None = None
    branch: str = "main"
    bank: str = MAIN_BANK

    @property
    def document_id(self) -> str:
        """Opaque, so neither the label nor the category reaches the server."""
        return "m" + hashlib.sha256(self.label.encode()).hexdigest()[:12]

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
class Query:
    id: str
    category: QueryCategory
    text: str
    project: str | None
    """The task's project scope; ``None`` for a task outside any project."""
    relevant: tuple[str, ...] = ()
    """Memories that should be injected; empty means the right answer is to inject nothing."""
    acceptable: tuple[str, ...] = ()
    """Related memories whose injection is neither credited nor counted as pollution."""
    hard_negatives: tuple[str, ...] = ()
    """Irrelevant memories that share vocabulary or tech with the task — the deceptive ones."""
    module: str | None = None
    environment: str | None = None
    purpose: Purpose = "auto_context"
    bank: str = MAIN_BANK
    notes: str = ""

    @property
    def expected_abstain(self) -> bool:
        return not self.relevant

    @property
    def repository(self) -> str | None:
        return REPOSITORIES.get(self.project) if self.project else None

    @property
    def present_state(self) -> bool:
        """Whether a superseded or refuted memory is wrong for this task (it is not for history)."""
        return self.purpose != "historical_search"

    def context(self) -> dict[str, str]:
        """The A49 request ``context`` a feature server receives; a base server ignores it."""
        out: dict[str, str] = {"branch": "main"}
        if self.repository:
            out["repository"] = self.repository
        if self.module:
            out["module"] = self.module
        if self.environment:
            out["environment"] = self.environment
        return out
