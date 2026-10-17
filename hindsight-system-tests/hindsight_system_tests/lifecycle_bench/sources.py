"""Where a consolidation reply comes from: the scenario author, or a frozen real model.

Both produce a ``NeedleReply`` for one consolidation call; ``encoding.resolve`` turns either
into ids. Author-written payloads are mechanism tests of the server against a model that
does exactly what the scenario says. A frozen fixture replaces them with what a real model
actually answered under each arm's real prompt (captured K times, modal kept — see
``encoding.modal_reply``), keyed by statement text so it replays on any run.

Fixture layout: ``<dir>/<scenario_id>.json`` holding a ``FrozenFixture``. A call with no
matching entry is an error, never a fall-back to authored payloads: a run that mixes the two
would report numbers nobody could attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .encoding import BaseEncoding, BatchView, LifecycleForm, NeedleReply, SchemaCapabilities, encode
from .model import Scenario, decision_facts

_TEMPORAL_SUFFIX = re.compile(r"\s*\((?:occurred_start|occurred_end|mentioned_at)=[^)]*\)\s*$")


def statement(rendered_fact: str) -> str:
    """A batch fact's text without the temporal fields the prompt appends to it."""
    return _TEMPORAL_SUFFIX.sub("", rendered_fact)


class PayloadError(Exception):
    """A consolidation call the payload source cannot answer honestly."""


class PayloadSource(Protocol):
    name: str

    def consolidation(
        self, scenario: Scenario, step_index: int, batch: BatchView, caps: SchemaCapabilities
    ) -> NeedleReply: ...


@dataclass(frozen=True)
class AuthoredSource:
    """The scenario's own decisions, in the arm's encoding."""

    encoding: BaseEncoding
    lifecycle_form: LifecycleForm
    name: str = "authored"

    def consolidation(
        self, scenario: Scenario, step_index: int, batch: BatchView, caps: SchemaCapabilities
    ) -> NeedleReply:
        step = scenario.steps[step_index]
        needles = [needle for decision in step.decisions for needle in decision_facts(decision)]
        uncovered = [fact.text for fact in batch.facts if not any(needle in fact.text for needle in needles)]
        if uncovered:
            raise PayloadError(f"{scenario.id} step {step_index}: no decision covers batch facts {uncovered}")
        reply = NeedleReply()
        form = caps.form(self.lifecycle_form)
        for decision in step.decisions:
            if batch.matches(decision_facts(decision)):
                reply.extend(encode(decision, encoding=self.encoding, lifecycle=caps.lifecycle, form=form))
        return reply


class FrozenEntry(BaseModel):
    step: int
    batch_facts: list[str]
    """Statements of the batch facts the call carried, which is what identifies the call."""
    reply: NeedleReply
    captured: int = 1
    """How many real-model replies the modal was taken over."""


class FrozenFixture(BaseModel):
    scenario: str
    arm: str | None = None
    model: str | None = None
    entries: list[FrozenEntry] = Field(default_factory=list)


@dataclass
class FrozenSource:
    """Replay frozen real-model replies from a fixture directory."""

    directory: Path
    name: str = "frozen"

    def consolidation(
        self, scenario: Scenario, step_index: int, batch: BatchView, caps: SchemaCapabilities
    ) -> NeedleReply:
        path = self.directory / f"{scenario.id}.json"
        if not path.is_file():
            raise PayloadError(f"no frozen fixture {path}")
        fixture = FrozenFixture.model_validate_json(path.read_text())
        key = sorted(statement(fact.text) for fact in batch.facts)
        for entry in fixture.entries:
            if entry.step == step_index and sorted(entry.batch_facts) == key:
                return entry.reply
        raise PayloadError(f"{path}: no entry for step {step_index} with batch {key}")
