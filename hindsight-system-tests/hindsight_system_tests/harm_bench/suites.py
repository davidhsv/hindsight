"""The task suites a run can choose, each with the bank it is answered against.

``original`` is the pre-registered suite (24 tasks, 152 memories). ``h2`` is exploratory (see
:mod:`.h2`): its own 16 tasks against the original bank plus the H2 memories, so adding it never
changes the original suite's bank or results.
"""

from __future__ import annotations

from dataclasses import dataclass

from .h2 import H2_MEMORIES, H2_TASKS, validate_h2
from .model import Memory, Task
from .scenario import MEMORIES, TASKS, validate


@dataclass(frozen=True)
class Suite:
    name: str
    memories: tuple[Memory, ...]
    tasks: tuple[Task, ...]
    exploratory: str | None = None


SUITES: dict[str, Suite] = {
    "original": Suite("original", MEMORIES, TASKS),
    "h2": Suite(
        "h2",
        MEMORIES + H2_MEMORIES,
        H2_TASKS,
        exploratory="exploratory, designed after the base result was seen",
    ),
}

#: Every task and memory of every suite, by id; ids are unique across suites (validated).
TASK_BY_ID: dict[str, Task] = {task.id: task for suite in SUITES.values() for task in suite.tasks}
LABEL_BY_DOCUMENT: dict[str, str] = {
    memory.document_id: memory.label for suite in SUITES.values() for memory in suite.memories
}


def validate_all() -> list[str]:
    return validate() + validate_h2()
