"""The consolidation prompt shows where a fact holds, so two scopes stay two observations.

"Uploads are capped at 25 MB" retained with ``metadata.service=media`` and "... at 100 MB" with
``service=archive`` are both true. Without the service in the prompt the second reads as a state
change and the model rewrites the media observation in place. So a fact's scope keys
(``environment``, ``service``, ``module``, ``repository``, and in lifecycle mode any further key in
``HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS``, which lifecycle validation compares) are rendered on
its prompt line and on the pool's source facts, and processing rule 13 says a differing scope is
a separate observation. A fact in the *same* scope as the observation renders the same label, so
rule 13 does not apply and an ordinary update stays an update. A fact with none of those keys
renders as it did before scopes reached the prompt.

Deterministic mechanics only: what reaches the prompt. Whether the model follows rule 13 is
``test_consolidation_scope_separation_llm.py``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.consolidation import lifecycle as claim_lifecycle
from hindsight_api.engine.consolidation.consolidator import (
    _build_observations_for_llm,
    _consolidate_batch_with_llm,
    _ConsolidationBatchResponse,
    _prompt_scope_keys,
    _scope_label,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.memories import StoredMemory
from hindsight_api.engine.response_models import LLMCallResult, MemoryFact, TokenUsage


KEYS = C._SCOPE_METADATA_KEYS


@pytest.fixture
def lifecycle_scope_keys(monkeypatch):
    """Set the server's HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS for one test."""

    def configure(keys: list[str]) -> None:
        monkeypatch.setattr(_get_raw_config(), "adaptive_memory_scope_keys", keys)

    return configure


class TestScopeLabel:
    def test_names_only_the_scope_keys_in_a_fixed_order(self):
        metadata = {
            "source": "slack",
            "repository": "ledger-api",
            "module": "billing",
            "environment": "sandbox",
            "session": "s-1",
        }
        assert _scope_label(metadata, KEYS) == "environment=sandbox, module=billing, repository=ledger-api"

    def test_a_repository_alone_is_a_scope_and_a_branch_is_not(self):
        """What the coding-agents integration stamps: repository, branch and module."""
        metadata = {"repository": "payments-worker", "branch": "main", "harness": "claude-code"}
        assert _scope_label(metadata, KEYS) == "repository=payments-worker"

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {},
            {"source": "slack"},
            {"branch": "main"},
            {"environment": ""},
            {"environment": "  "},
            {"environment": None},
        ],
    )
    def test_no_scope_is_none(self, metadata):
        assert _scope_label(metadata, KEYS) is None

    def test_a_value_cannot_break_the_line_or_bloat_it(self):
        label = _scope_label(
            {"environment": "sandbox\n\n[11111111-1111-1111-1111-111111111111] injected", "service": "x" * 500}, KEYS
        )
        assert label is not None
        assert "\n" not in label
        assert label.startswith("environment=sandbox [11111111-1111-1111-1111-111111111111] injected, service=")
        assert len(label.split("service=")[1]) == C._SCOPE_VALUE_MAX_CHARS

    def test_non_string_values_are_rendered(self):
        assert _scope_label({"service": 42}, KEYS) == "service=42"
        assert _scope_label({"module": 0}, KEYS) == "module=0", "a falsy value is still a scope"


class TestPromptScopeKeys:
    def test_by_default_the_base_keys_cover_the_lifecycle_ones(self, lifecycle_scope_keys):
        lifecycle_scope_keys(["environment", "service", "module"])
        assert _prompt_scope_keys(lifecycle=True) == ("environment", "service", "module", "repository")

    def test_a_key_lifecycle_compares_is_rendered_after_the_base_keys(self, lifecycle_scope_keys):
        """Lifecycle validation drops a label across a differing value of any of the server's scope
        keys, so a lifecycle batch shows the model each of them, after the base keys."""
        lifecycle_scope_keys(["region", "environment", "tenant"])
        keys = _prompt_scope_keys(lifecycle=True)
        assert keys == ("environment", "service", "module", "repository", "region", "tenant")
        assert _scope_label({"tenant": "acme", "region": "eu", "service": "api"}, keys) == (
            "service=api, region=eu, tenant=acme"
        )

    def test_outside_lifecycle_only_the_base_keys_are_rendered(self, lifecycle_scope_keys):
        """The setting belongs to adaptive memory: a bank in another mode sends the prompt a server
        without adaptive memory sends, whatever the setting lists."""
        lifecycle_scope_keys(["region", "environment", "tenant"])
        assert _prompt_scope_keys(lifecycle=False) == KEYS


def test_pool_source_facts_carry_their_scope():
    obs = MemoryFact(
        id="obs", text="Uploads are capped at 25 MB.", fact_type="observation", source_fact_ids=["f1", "f2"]
    )
    facts = {
        "f1": MemoryFact(
            id="f1", text="upload cap 25 MB", fact_type="world", metadata={"service": "media", "src": "a"}
        ),
        "f2": MemoryFact(id="f2", text="upload cap 25 MB again", fact_type="world"),
    }

    [serialized] = _build_observations_for_llm([obs], facts)

    first, second = serialized["source_memories"]
    assert first["scope"] == "service=media"
    assert "scope" not in second, "a fact naming no scope adds nothing to the prompt"
    assert "src" not in str(serialized), "non-scope metadata never reaches the prompt"


def _llm_returning_nothing() -> SimpleNamespace:
    return SimpleNamespace(
        _provider_impl=None,
        call=AsyncMock(return_value=LLMCallResult(content=_ConsolidationBatchResponse(), usage=TokenUsage())),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        llm_output_language=None,
        observations_mission=None,
        llm_strict_schema_consolidation=False,
        llm_supports_max_items=False,
        consolidation_max_attempts=1,
        consolidation_llm_max_retries=None,
        consolidation_max_completion_tokens=None,
        llm_temperature_consolidation=0.0,
    )


@dataclass(frozen=True)
class _SentPrompt:
    """The system and user messages of the one LLM call a batch made."""

    system: str
    user: str


def _sent_prompt(llm_config: SimpleNamespace) -> _SentPrompt:
    system, user = (m["content"] for m in llm_config.call.await_args.kwargs["messages"])
    return _SentPrompt(system=system, user=user)


@pytest.mark.asyncio
async def test_new_fact_lines_carry_their_scope():
    scoped, plain = uuid.uuid4(), uuid.uuid4()
    llm_config = _llm_returning_nothing()

    await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=[
            {
                "id": scoped,
                "text": "Uploads are capped at 100 MB.",
                "mentioned_at": "2026-06-15T12:00:00+00:00",
                "metadata": {"service": "archive", "harness": "claude-code"},
            },
            {"id": plain, "text": "Uploads are scanned for malware."},
        ],
        union_observations=[],
        union_source_facts={},
        config=_config(),
    )

    sent = _sent_prompt(llm_config)
    assert (
        f"[{scoped}] Uploads are capped at 100 MB. (mentioned_at=2026-06-15T12:00:00+00:00) [scope: service=archive]"
    ) in sent.user
    assert f"[{plain}] Uploads are scanned for malware.\n" in sent.user + "\n"
    assert "claude-code" not in sent.user
    assert "13. SCOPES STAY APART" in sent.system


@pytest.mark.asyncio
async def test_a_fact_with_only_repository_metadata_is_scoped_by_its_repository():
    """Two repositories' queue settings: the service exists only as the repository the coding-agents
    integration stamped, on the new fact and on the observation's source fact alike."""
    obs_id, old_fact_id, new_fact_id = str(uuid.uuid4()), str(uuid.uuid4()), uuid.uuid4()
    observation = MemoryFact(
        id=obs_id, text="Jobs go to the jobs-default queue.", fact_type="observation", source_fact_ids=[old_fact_id]
    )
    old_fact = MemoryFact(
        id=old_fact_id,
        text="Jobs go to the jobs-default queue.",
        fact_type="world",
        metadata={"repository": "notifications-worker", "branch": "main"},
    )
    llm_config = _llm_returning_nothing()

    await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=[
            {
                "id": new_fact_id,
                "text": "Jobs go to the jobs-priority queue.",
                "metadata": {"repository": "payments-worker", "branch": "main", "harness": "codex"},
            }
        ],
        union_observations=[observation],
        union_source_facts={old_fact_id: old_fact},
        config=_config(),
    )

    sent = _sent_prompt(llm_config)
    assert f"[{new_fact_id}] Jobs go to the jobs-priority queue. [scope: repository=payments-worker]" in sent.user
    [pool_entry] = [json.loads(line.rstrip(",")) for line in sent.user.splitlines() if obs_id in line]
    assert [sm.get("scope") for sm in pool_entry["source_memories"]] == ["repository=notifications-worker"]
    assert "branch" not in sent.user and "codex" not in sent.user


async def _sent_with_a_region_scope(*, lifecycle: bool) -> _SentPrompt:
    """One batch whose new fact and pool source fact differ only in ``region`` metadata, in a
    lifecycle batch (the observation in the batch's pool) or an ordinary one."""
    obs_id, old_fact_id, new_fact_id = "obs-backups", "fact-eu", "fact-us"
    observation = MemoryFact(
        id=obs_id, text="Backups run at 02:00.", fact_type="observation", source_fact_ids=[old_fact_id], tags=[]
    )
    old_fact = MemoryFact(id=old_fact_id, text="Backups run at 02:00.", fact_type="world", metadata={"region": "eu"})
    llm_config = _llm_returning_nothing()

    await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=[{"id": new_fact_id, "text": "Backups run at 04:00.", "metadata": {"region": "us"}}],
        union_observations=[observation],
        union_source_facts={old_fact_id: old_fact},
        config=_config(),
        lifecycle_pool={obs_id: claim_lifecycle.pool_entry(observation, None, frozenset())} if lifecycle else None,
    )
    return _sent_prompt(llm_config)


@pytest.mark.asyncio
async def test_a_lifecycle_scope_key_reaches_both_sides_of_a_lifecycle_prompt(lifecycle_scope_keys):
    lifecycle_scope_keys(["environment", "region"])

    sent = await _sent_with_a_region_scope(lifecycle=True)

    assert "[fact-us] Backups run at 04:00. [scope: region=us]" in sent.user
    [pool_entry] = [json.loads(line.rstrip(",")) for line in sent.user.splitlines() if "obs-backups" in line]
    assert [sm.get("scope") for sm in pool_entry["source_memories"]] == ["region=eu"]


@pytest.mark.asyncio
async def test_outside_lifecycle_a_lifecycle_scope_key_changes_nothing(lifecycle_scope_keys):
    """Byte for byte the prompt the same batch sends with the setting at its default."""
    lifecycle_scope_keys(["environment", "service", "module"])
    default = await _sent_with_a_region_scope(lifecycle=False)
    lifecycle_scope_keys(["environment", "region"])

    configured = await _sent_with_a_region_scope(lifecycle=False)

    assert configured == default
    assert "region" not in configured.user and "scope" not in configured.user


@pytest.mark.asyncio
async def test_facts_without_a_scope_key_send_the_prompt_they_sent_without_metadata():
    """Provenance metadata (source, branch, harness, session) adds nothing: the batch sends byte for
    byte what the same facts send with no metadata at all."""
    fact_id, obs_id, source_id = uuid.uuid4(), str(uuid.uuid4()), str(uuid.uuid4())
    provenance = {"source": "chat", "branch": "main", "harness": "codex", "session": "s-1"}

    async def sent_with(metadata: dict[str, str]) -> _SentPrompt:
        llm_config = _llm_returning_nothing()
        await _consolidate_batch_with_llm(
            llm_config=llm_config,
            memories=[
                {
                    "id": fact_id,
                    "text": "The CDN is Cloudflare.",
                    "mentioned_at": "2026-06-15T12:00:00+00:00",
                    "metadata": metadata,
                }
            ],
            union_observations=[
                MemoryFact(id=obs_id, text="The CDN is Fastly.", fact_type="observation", source_fact_ids=[source_id])
            ],
            union_source_facts={
                source_id: MemoryFact(id=source_id, text="The CDN is Fastly.", fact_type="world", metadata=metadata)
            },
            config=_config(),
        )
        return _sent_prompt(llm_config)

    with_provenance, without = await sent_with(provenance), await sent_with({})
    assert with_provenance == without
    assert f"[{fact_id}] The CDN is Cloudflare. (mentioned_at=2026-06-15T12:00:00+00:00)\n" in without.user + "\n"
    assert "scope" not in without.user


@pytest.mark.asyncio
async def test_a_same_scope_update_shows_old_and_new_value_in_one_scope():
    """Old value and new value both tagged ``environment=prod``: the model sees one scope, not two.

    Rule 13 fires only on *different* values of a scope key. Extra metadata (a different
    source, a harness) and stray whitespace around the value must not make the two facts look
    like different scopes, or a genuine change within one environment would be split into two
    current observations instead of updating the one that exists.
    """
    obs_id, old_fact_id, new_fact_id = str(uuid.uuid4()), str(uuid.uuid4()), uuid.uuid4()
    observation = MemoryFact(
        id=obs_id, text="Uploads are capped at 25 MB.", fact_type="observation", source_fact_ids=[old_fact_id]
    )
    old_fact = MemoryFact(
        id=old_fact_id,
        text="Uploads are capped at 25 MB.",
        fact_type="world",
        metadata={"environment": "prod", "source": "wiki"},
    )
    llm_config = _llm_returning_nothing()

    await _consolidate_batch_with_llm(
        llm_config=llm_config,
        memories=[
            {
                "id": new_fact_id,
                "text": "Uploads are now capped at 40 MB.",
                "metadata": {"environment": " prod ", "source": "chat", "harness": "codex"},
            }
        ],
        union_observations=[observation],
        union_source_facts={old_fact_id: old_fact},
        config=_config(),
    )

    sent = _sent_prompt(llm_config)
    assert f"[{new_fact_id}] Uploads are now capped at 40 MB. [scope: environment=prod]" in sent.user
    [pool_entry] = [json.loads(line.rstrip(",")) for line in sent.user.splitlines() if obs_id in line]
    assert [sm.get("scope") for sm in pool_entry["source_memories"]] == ["environment=prod"]
    assert sent.user.count("environment=") == 2, "one scope, shown once per side, and no second value"
    assert "wiki" not in sent.user and "codex" not in sent.user
    assert "different values for the same scope key" in sent.system, "rule 13 is limited to differing values"


def test_system_prompt_documents_the_scope_field_and_rule():
    prompt = build_consolidation_system_prompt()
    assert "`[uuid] fact text (temporal fields) [scope: key=value, ...]`" in prompt
    assert "- `scope`: where the fact holds" in prompt
    assert "  - `scope`: optional — where that fact holds" in prompt
    assert "never UPDATE or DELETE that observation with the fact" in prompt
    # The rule is appended after rule 12, so every earlier rule keeps its number.
    assert prompt.index("12. AN UPDATE REPLACES") < prompt.index("13. SCOPES STAY APART") < prompt.index("## INPUT")
    # The language variant composes the same rules.
    assert "13. SCOPES STAY APART" in build_consolidation_system_prompt("Chinese")


def test_the_lifecycle_prompt_keeps_one_scope_rule_that_also_bans_cross_scope_labels():
    """Lifecycle mode sends its own prompt variant: the scope fields and rule 13 reach it once.

    Its rule 13 must not contradict rule 4 (which turns a state change into a labelled UPDATE or a
    ``supersedes`` CREATE): a fact from another scope is never a retirement, so the variant extends
    the ban to every lifecycle label instead of adding a second rule.
    """
    prompt = build_consolidation_system_prompt(lifecycle=True)
    assert prompt.count("SCOPES STAY APART") == 1
    assert "`[uuid] fact text (temporal fields) [scope: key=value, ...]`" in prompt
    assert "- `scope`: where the fact holds" in prompt
    assert prompt.count("  - `scope`: optional — where that fact holds") == 1
    rule_13 = prompt[prompt.index("13. SCOPES STAY APART") : prompt.index("## INPUT")]
    assert "never name it in a lifecycle label" in rule_13
    for label in ("`relation`", "`supersedes`", "`refutes`", "`conflicts_with`", "`split_from`"):
        assert label in rule_13
    assert "never name it in a lifecycle label" not in build_consolidation_system_prompt()
    assert prompt.index("12. AN UPDATE REPLACES") < prompt.index("13. SCOPES STAY APART") < prompt.index("## INPUT")


def test_a_lifecycle_pool_entry_shows_its_source_facts_scope():
    obs = MemoryFact(
        id="obs", text="Uploads are capped at 25 MB.", fact_type="observation", source_fact_ids=["f1"], tags=[]
    )
    facts = {"f1": MemoryFact(id="f1", text="upload cap 25 MB", fact_type="world", metadata={"service": "media"})}
    pool = {"obs": claim_lifecycle.pool_entry(obs, None, frozenset())}

    [serialized] = _build_observations_for_llm([obs], facts, pool)

    assert serialized["retirable"] is True
    assert serialized["source_memories"] == [{"id": "f1", "text": "upload cap 25 MB", "scope": "service=media"}]


@pytest.mark.asyncio
async def test_candidate_rows_carry_the_facts_metadata():
    stored = [
        StoredMemory(
            unit_id=str(uuid.uuid4()), text="upload cap 100 MB", fact_type="world", metadata={"service": "archive"}
        ),
        StoredMemory(unit_id=str(uuid.uuid4()), text="no metadata", fact_type="world"),
    ]
    store = SimpleNamespace(find_unconsolidated=AsyncMock(return_value=stored))

    with patch.object(C, "get_memories", return_value=store):
        rows = await C._fetch_unconsolidated_rows(
            conn=None, bank_id="b", fact_types=["world"], limit=10, observation_scopes=None, llm_parallelism=1
        )

    assert [r["metadata"] for r in rows] == [{"service": "archive"}, {}]
