"""Configuration surface of adaptive memory (retention + truth lifecycle).

Covers both doors a value comes through: the environment (``HindsightConfig.from_env``
and ``validate``, strict, reported by env name) and the bank-config API (per-field
validators in ``config_resolver``, reported by field name), plus the read side, where
a stored value that is no longer valid must resolve to something safe rather than
wedge the bank.
"""

import dataclasses
import math
import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.config import (
    ADAPTIVE_MEMORY_MODES,
    DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS,
    DEFAULT_ADAPTIVE_MEMORY_TYPE_STABILITY,
    ConfigFieldAccessError,
    HindsightConfig,
    StaticConfigProxy,
)
from hindsight_api.config_resolver import ConfigResolver, _coerce_stored_bank_overrides

from .test_hierarchical_config import FakeBankConfigBackend

_HIERARCHICAL = {
    "adaptive_memory_mode": "off",
    "adaptive_memory_score_alpha": 0.1,
    "adaptive_memory_score_floor": 0.3,
    "adaptive_memory_reinforcement_enabled": True,
    "adaptive_memory_base_stability_days": 30.0,
    "adaptive_memory_max_stability_days": 3650.0,
    "adaptive_memory_default_view": "current",
    "adaptive_memory_historical_cues": False,
}
_STATIC = {
    "adaptive_memory_recall_reinforcement_enabled": False,
    "adaptive_memory_reflect_reinforcement_enabled": True,
    "adaptive_memory_reinforcement_cooldown_hours": 6.0,
    "adaptive_memory_reinforcement_gain": 1.0,
    "adaptive_memory_source_weights": {
        "explicit_confirmation": 1.0,
        "independent_retain_confirmation": 0.8,
        "successful_reflect_use": 0.6,
        "observation_new_evidence": 0.5,
        "final_external_recall": 0.2,
    },
    "adaptive_memory_type_stability": {"experience": 0.5, "world": 1.0, "observation": 2.0},
    "adaptive_memory_source_trust": {},
    "adaptive_memory_conflict_policy": "evidence",
    "adaptive_memory_archive_enabled": False,
    "adaptive_memory_archive_threshold": 0.05,
    "adaptive_memory_archive_grace_days": 30.0,
    "adaptive_memory_archive_recall_min_score": 0.8,
    "adaptive_memory_archive_protect_importance": 0.8,
    "adaptive_memory_prune_enabled": False,
    "adaptive_memory_prune_after_days": 180.0,
    "adaptive_memory_maintenance_interval_seconds": 3600,
    "adaptive_memory_scope_keys": ["environment", "service", "module"],
}


def _env_name(field_name: str) -> str:
    return f"HINDSIGHT_API_{field_name.upper()}"


@pytest.fixture
def clean_env(monkeypatch):
    """No adaptive-memory env var set, and a provider that needs no credentials."""
    for field_name in (*_HIERARCHICAL, *_STATIC):
        monkeypatch.delenv(_env_name(field_name), raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    return monkeypatch


def _bank() -> str:
    return f"adaptive-config-{uuid.uuid4().hex[:12]}"


# --- Field inventory ---------------------------------------------------------------


def test_field_split_matches_the_contract():
    """Exactly the eight per-bank fields are configurable; the rest are server-level."""
    adaptive = {f.name for f in dataclasses.fields(HindsightConfig) if f.name.startswith("adaptive_memory_")}
    assert adaptive == set(_HIERARCHICAL) | set(_STATIC)
    configurable = HindsightConfig.get_configurable_fields()
    assert set(_HIERARCHICAL) <= configurable
    assert not set(_STATIC) & configurable
    assert set(_STATIC) <= HindsightConfig.get_static_fields()


def test_every_field_has_a_matching_env_constant():
    """The env var name is derivable from the field, so docs and code cannot drift."""
    import hindsight_api.config as config_module

    for field_name in (*_HIERARCHICAL, *_STATIC):
        constant = f"ENV_{field_name.upper()}"
        assert getattr(config_module, constant) == _env_name(field_name)


def test_mode_is_typed_str_for_the_bank_config_type_contract():
    """A Literal annotation would leave the field with no runtime type contract."""
    annotations = {f.name: f.type for f in dataclasses.fields(HindsightConfig)}
    assert annotations["adaptive_memory_mode"] is str
    assert annotations["adaptive_memory_default_view"] is str


def test_static_proxy_hides_per_bank_fields_and_exposes_server_level_ones(clean_env):
    proxy = StaticConfigProxy(HindsightConfig.from_env())
    with pytest.raises(ConfigFieldAccessError):
        proxy.adaptive_memory_mode
    assert proxy.adaptive_memory_archive_enabled is False
    assert proxy.adaptive_memory_source_weights == _STATIC["adaptive_memory_source_weights"]


def test_modes_match_the_engine_enum():
    """config.py keeps its own tuple so it need not import the engine; keep them equal."""
    models = pytest.importorskip("hindsight_api.engine.adaptive.models")
    assert tuple(mode.value for mode in models.AdaptiveMode) == ADAPTIVE_MEMORY_MODES


# --- Environment: defaults and reads -----------------------------------------------


def test_defaults(clean_env):
    config = HindsightConfig.from_env()
    for field_name, expected in {**_HIERARCHICAL, **_STATIC}.items():
        assert getattr(config, field_name) == expected, field_name


def test_default_maps_are_copies_not_the_module_constants(clean_env):
    """Mutating one config's map must not rewrite the documented defaults."""
    config = HindsightConfig.from_env()
    assert config.adaptive_memory_source_weights is not DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS
    assert config.adaptive_memory_type_stability is not DEFAULT_ADAPTIVE_MEMORY_TYPE_STABILITY


def test_scalar_values_read_from_env(clean_env):
    values = {
        "adaptive_memory_mode": ("RANK", "rank"),
        "adaptive_memory_score_alpha": ("0.2", 0.2),
        "adaptive_memory_score_floor": ("0.5", 0.5),
        "adaptive_memory_reinforcement_enabled": ("false", False),
        "adaptive_memory_base_stability_days": ("14", 14.0),
        "adaptive_memory_max_stability_days": ("365", 365.0),
        "adaptive_memory_default_view": (" Historical ", "historical"),
        "adaptive_memory_historical_cues": ("1", True),
        "adaptive_memory_recall_reinforcement_enabled": ("true", True),
        "adaptive_memory_reflect_reinforcement_enabled": ("false", False),
        "adaptive_memory_reinforcement_cooldown_hours": ("0", 0.0),
        "adaptive_memory_reinforcement_gain": ("2.5", 2.5),
        "adaptive_memory_conflict_policy": ("latest", "latest"),
        "adaptive_memory_archive_enabled": ("true", True),
        "adaptive_memory_archive_threshold": ("0.1", 0.1),
        "adaptive_memory_archive_grace_days": ("7", 7.0),
        "adaptive_memory_archive_recall_min_score": ("0.9", 0.9),
        "adaptive_memory_archive_protect_importance": ("1", 1.0),
        "adaptive_memory_prune_enabled": ("true", True),
        "adaptive_memory_prune_after_days": ("365", 365.0),
    }
    for field_name, (raw, _expected) in values.items():
        clean_env.setenv(_env_name(field_name), raw)

    config = HindsightConfig.from_env()

    for field_name, (_raw, expected) in values.items():
        assert getattr(config, field_name) == expected, field_name


@pytest.mark.parametrize("mode", ADAPTIVE_MEMORY_MODES)
def test_every_mode_is_accepted(clean_env, mode):
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_MODE", mode)
    assert HindsightConfig.from_env().adaptive_memory_mode == mode


def test_empty_choice_env_means_default(clean_env):
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_MODE", "")
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_DEFAULT_VIEW", "")
    config = HindsightConfig.from_env()
    assert config.adaptive_memory_mode == "off"
    assert config.adaptive_memory_default_view == "current"


def test_json_maps_merge_over_the_defaults(clean_env):
    """Retuning one weight keeps the documented value of every other one."""
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_SOURCE_WEIGHTS", '{"final_external_recall": 0.05}')
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_TYPE_STABILITY", '{"world": 3}')
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_SOURCE_TRUST", '{"crm": 0.9, "slack": 0.4}')

    config = HindsightConfig.from_env()

    assert config.adaptive_memory_source_weights == {
        **_STATIC["adaptive_memory_source_weights"],
        "final_external_recall": 0.05,
    }
    assert config.adaptive_memory_type_stability == {"experience": 0.5, "world": 3.0, "observation": 2.0}
    assert isinstance(config.adaptive_memory_type_stability["world"], float)
    assert config.adaptive_memory_source_trust == {"crm": 0.9, "slack": 0.4}


# --- Environment: invalid values are rejected ---------------------------------------


@pytest.mark.parametrize(
    ("field_name", "raw"),
    [
        ("adaptive_memory_mode", "ranked"),
        ("adaptive_memory_mode", "true"),
        ("adaptive_memory_default_view", "past"),
        ("adaptive_memory_conflict_policy", "oldest"),
        ("adaptive_memory_score_alpha", "1.5"),
        ("adaptive_memory_score_alpha", "-0.1"),
        ("adaptive_memory_score_alpha", "nan"),
        ("adaptive_memory_score_floor", "2"),
        ("adaptive_memory_score_floor", "abc"),
        ("adaptive_memory_base_stability_days", "0"),
        ("adaptive_memory_base_stability_days", "inf"),
        ("adaptive_memory_max_stability_days", "0"),
        ("adaptive_memory_max_stability_days", "10"),  # below the default base of 30
        ("adaptive_memory_reinforcement_cooldown_hours", "-1"),
        ("adaptive_memory_reinforcement_gain", "nan"),
        ("adaptive_memory_archive_threshold", "1.01"),
        ("adaptive_memory_archive_recall_min_score", "5"),
        ("adaptive_memory_archive_protect_importance", "1.2"),
        ("adaptive_memory_archive_grace_days", "-3"),
        ("adaptive_memory_prune_after_days", "-1"),
        ("adaptive_memory_reinforcement_enabled", "yes"),
        ("adaptive_memory_archive_enabled", "on"),
        ("adaptive_memory_source_weights", "not json"),
        ("adaptive_memory_source_weights", "[1, 2]"),
        ("adaptive_memory_source_weights", '{"explicit_confirmation": "high"}'),
        ("adaptive_memory_source_weights", '{"explicit_confirmation": true}'),
        ("adaptive_memory_source_weights", '{"explicit_confirmation": -1}'),
        ("adaptive_memory_source_weights", '{"explict_confirmation": 1.0}'),
        ("adaptive_memory_type_stability", '{"opinion": 1.0}'),
        ("adaptive_memory_type_stability", '{"world": 0}'),
        ("adaptive_memory_source_trust", '{"crm": NaN}'),
        ("adaptive_memory_source_trust", '{"": 0.5}'),
    ],
)
def test_invalid_env_values_fail_startup(clean_env, field_name, raw):
    """Strict on env: a typo must fail start-up, never silently mean a different arm."""
    env_name = _env_name(field_name)
    clean_env.setenv(env_name, raw)
    with pytest.raises(ValueError, match=env_name):
        HindsightConfig.from_env()


def test_max_stability_equal_to_base_is_allowed(clean_env):
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_BASE_STABILITY_DAYS", "90")
    clean_env.setenv("HINDSIGHT_API_ADAPTIVE_MEMORY_MAX_STABILITY_DAYS", "90")
    config = HindsightConfig.from_env()
    assert config.adaptive_memory_base_stability_days == config.adaptive_memory_max_stability_days == 90.0


# --- Bank config API: validation -----------------------------------------------------


@pytest.mark.asyncio
async def test_bank_override_is_validated_and_normalized():
    bank_id = _bank()
    resolver = ConfigResolver(backend=FakeBankConfigBackend())

    validated = await resolver.validate_bank_config_updates(
        bank_id,
        {
            "adaptive_memory_mode": " Observe ",
            "HINDSIGHT_API_ADAPTIVE_MEMORY_DEFAULT_VIEW": "HISTORICAL",
            "adaptive_memory_score_alpha": 1,
            "adaptive_memory_score_floor": 0,
            "adaptive_memory_reinforcement_enabled": False,
            "adaptive_memory_base_stability_days": 7.5,
            "adaptive_memory_historical_cues": True,
        },
    )

    assert validated.updates == {
        "adaptive_memory_mode": "observe",
        "adaptive_memory_default_view": "historical",
        "adaptive_memory_score_alpha": 1,
        "adaptive_memory_score_floor": 0,
        "adaptive_memory_reinforcement_enabled": False,
        "adaptive_memory_base_stability_days": 7.5,
        "adaptive_memory_historical_cues": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"adaptive_memory_mode": "turbo"}, "adaptive_memory_mode must be one of"),
        ({"adaptive_memory_mode": 1}, "adaptive_memory_mode must be a string"),
        ({"adaptive_memory_default_view": "past"}, "adaptive_memory_default_view must be one of"),
        ({"adaptive_memory_score_alpha": 1.5}, "adaptive_memory_score_alpha"),
        ({"adaptive_memory_score_alpha": -0.01}, "adaptive_memory_score_alpha"),
        ({"adaptive_memory_score_alpha": math.nan}, "adaptive_memory_score_alpha must be a finite number"),
        ({"adaptive_memory_score_floor": True}, "adaptive_memory_score_floor must be a number"),
        ({"adaptive_memory_base_stability_days": 0}, "adaptive_memory_base_stability_days must be > 0"),
        ({"adaptive_memory_max_stability_days": math.inf}, "adaptive_memory_max_stability_days"),
        ({"adaptive_memory_reinforcement_enabled": "yes"}, "adaptive_memory_reinforcement_enabled must be a boolean"),
        (
            {"adaptive_memory_base_stability_days": 100, "adaptive_memory_max_stability_days": 50},
            "must be >= adaptive_memory_base_stability_days",
        ),
        # One-sided updates are checked against the server value of the other side.
        ({"adaptive_memory_max_stability_days": 10}, "must be >= adaptive_memory_base_stability_days"),
        ({"adaptive_memory_base_stability_days": 4000}, "must be >= adaptive_memory_base_stability_days"),
    ],
)
async def test_invalid_bank_override_is_rejected(updates, match):
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    with pytest.raises(ValueError, match=match):
        await resolver.validate_bank_config_updates(_bank(), updates)


@pytest.mark.asyncio
async def test_static_adaptive_fields_cannot_be_overridden_per_bank():
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    for field_name in ("adaptive_memory_archive_enabled", "adaptive_memory_source_weights"):
        with pytest.raises(ValueError, match="static"):
            await resolver.validate_bank_config_updates(_bank(), {field_name: True})


@pytest.mark.asyncio
async def test_stability_cap_is_checked_against_the_stored_base():
    """The cross-field rule holds against what the bank already stores, not just the update."""
    bank_id = _bank()
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    await resolver.update_bank_config(bank_id, {"adaptive_memory_base_stability_days": 120})

    with pytest.raises(ValueError, match="must be >= adaptive_memory_base_stability_days"):
        await resolver.update_bank_config(bank_id, {"adaptive_memory_max_stability_days": 60})

    # Clearing the stored base (null = server default 30) makes the same cap valid.
    await resolver.update_bank_config(
        bank_id, {"adaptive_memory_base_stability_days": None, "adaptive_memory_max_stability_days": 60}
    )
    config = await resolver.resolve_full_config(bank_id)
    assert config.adaptive_memory_base_stability_days == 30.0
    assert config.adaptive_memory_max_stability_days == 60


@pytest.mark.asyncio
async def test_bank_override_resolves_and_clears():
    bank_id = _bank()
    resolver = ConfigResolver(backend=FakeBankConfigBackend())

    await resolver.update_bank_config(bank_id, {"adaptive_memory_mode": "Lifecycle", "adaptive_memory_score_alpha": 0})
    config = await resolver.resolve_full_config(bank_id)
    assert config.adaptive_memory_mode == "lifecycle"
    assert config.adaptive_memory_score_alpha == 0

    await resolver.update_bank_config(bank_id, {"adaptive_memory_mode": None, "adaptive_memory_score_alpha": None})
    config = await resolver.resolve_full_config(bank_id)
    assert config.adaptive_memory_mode == "off"
    assert config.adaptive_memory_score_alpha == 0.1


# --- Bank config API: stored values that are no longer valid -------------------------


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"adaptive_memory_mode": "turbo"}, {"adaptive_memory_mode": "off"}),
        ({"adaptive_memory_mode": " RANK "}, {"adaptive_memory_mode": "rank"}),
        ({"adaptive_memory_mode": 3}, {"adaptive_memory_mode": "off"}),
        ({"adaptive_memory_mode": {"mode": "rank"}}, {"adaptive_memory_mode": "off"}),
        ({"adaptive_memory_default_view": "past"}, {"adaptive_memory_default_view": "current"}),
        ({"adaptive_memory_mode": "observe"}, {"adaptive_memory_mode": "observe"}),
    ],
)
def test_unknown_stored_choice_is_coerced_to_the_safe_value(stored, expected):
    """An unknown stored mode means the feature is off, never partly on."""
    assert _coerce_stored_bank_overrides("bank", stored) == expected


@pytest.mark.asyncio
async def test_unknown_stored_mode_resolves_to_off():
    """A value written before validation existed (or by hand) must not wedge recall."""
    backend = FakeBankConfigBackend()
    backend.config = {"adaptive_memory_mode": "aggressive", "adaptive_memory_score_alpha": 0.2}
    resolver = ConfigResolver(backend=backend)

    config = await resolver.resolve_full_config(_bank(), cached=False)

    assert config.adaptive_memory_mode == "off"
    assert config.adaptive_memory_score_alpha == 0.2


# --- Bank config API over HTTP -----------------------------------------------------------


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_per_bank_override_through_the_config_api(api_client, memory, request_context):
    """PATCH sets one bank's mode; a second bank keeps the server default."""
    bank_a, bank_b = _bank(), _bank()
    try:
        for bank_id in (bank_a, bank_b):
            resp = await api_client.put(f"/v1/default/banks/{bank_id}", json={})
            assert resp.status_code == 200, resp.text

        resp = await api_client.patch(
            f"/v1/default/banks/{bank_a}/config",
            json={
                "updates": {
                    "adaptive_memory_mode": "Rank",
                    "adaptive_memory_score_alpha": 0.2,
                    "adaptive_memory_max_stability_days": 400,
                }
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["overrides"] == {
            "adaptive_memory_mode": "rank",
            "adaptive_memory_score_alpha": 0.2,
            "adaptive_memory_max_stability_days": 400,
        }
        assert body["config"]["adaptive_memory_mode"] == "rank"
        assert body["config"]["adaptive_memory_base_stability_days"] == 30.0
        # Server-level settings are not part of the per-bank surface.
        assert "adaptive_memory_source_weights" not in body["config"]

        resp = await api_client.get(f"/v1/default/banks/{bank_b}/config")
        assert resp.status_code == 200, resp.text
        assert resp.json()["config"]["adaptive_memory_mode"] == "off"
        assert resp.json()["overrides"] == {}

        for bad in (
            {"adaptive_memory_mode": "turbo"},
            {"adaptive_memory_score_floor": 1.5},
            {"adaptive_memory_base_stability_days": 500},  # above the stored cap of 400
            {"adaptive_memory_archive_enabled": True},  # static
        ):
            resp = await api_client.patch(f"/v1/default/banks/{bank_a}/config", json={"updates": bad})
            assert resp.status_code == 400, (bad, resp.text)

        resp = await api_client.get(f"/v1/default/banks/{bank_a}/config")
        assert resp.json()["config"]["adaptive_memory_mode"] == "rank"
        assert resp.json()["config"]["adaptive_memory_base_stability_days"] == 30.0
    finally:
        for bank_id in (bank_a, bank_b):
            await memory.delete_bank(bank_id, request_context=request_context)
