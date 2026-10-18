"""Configuration surface of context admission: env (strict, by env name), the bank-config API
(per-field validators, by field name) and the read side (a stored value that is no longer valid
resolves to something safe)."""

import dataclasses
import math
import uuid

import pytest

from hindsight_api.config import (
    CONTEXT_ADMISSION_MODES,
    CONTEXT_ADMISSION_SCORERS,
    DEFAULT_CONTEXT_ADMISSION_WEIGHTS,
    ConfigFieldAccessError,
    HindsightConfig,
    StaticConfigProxy,
)
from hindsight_api.config_resolver import ConfigResolver, _coerce_stored_bank_overrides
from hindsight_api.engine.context_admission import AdmissionMode, ContextAdmissionSettings, ScorerKind

from .test_hierarchical_config import FakeBankConfigBackend

_HIERARCHICAL = {
    "context_admission_mode": "off",
    "context_admission_threshold_auto": 0.6,
    "context_admission_threshold_explicit": 0.35,
    "context_admission_reference_bonus": 0.15,
    "context_admission_candidates": 30,
    "context_admission_max_memories": 10,
    "context_admission_max_tokens": 2000,
    "context_admission_mmr_lambda": 0.7,
    "context_admission_dup_similarity": 0.92,
}
_STATIC = {
    "context_admission_scorer": "deterministic",
    "context_admission_weights": DEFAULT_CONTEXT_ADMISSION_WEIGHTS,
    "context_admission_scope_keys": ["repository", "module", "environment", "branch", "service"],
    "context_admission_hard_scope_keys": ["repository"],
    "context_admission_uncertain_margin": 0.1,
    "context_admission_jev_weight": 0.5,
}
_DECISION_MODEL = (
    "context_admission_decision_model_provider",
    "context_admission_decision_model_api_key",
    "context_admission_decision_model_model",
    "context_admission_decision_model_base_url",
)


def _env_name(field_name: str) -> str:
    return f"HINDSIGHT_API_{field_name.upper()}"


@pytest.fixture
def clean_env(monkeypatch):
    for field_name in (*_HIERARCHICAL, *_STATIC, *_DECISION_MODEL):
        monkeypatch.delenv(_env_name(field_name), raising=False)
    for name in ("PROVIDER", "API_KEY", "MODEL", "BASE_URL"):
        monkeypatch.delenv(f"HINDSIGHT_API_DECISION_MODEL_{name}", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    return monkeypatch


def _bank() -> str:
    return f"ctx-adm-config-{uuid.uuid4().hex[:12]}"


def test_exactly_the_per_bank_fields_are_configurable():
    fields = {f.name for f in dataclasses.fields(HindsightConfig) if f.name.startswith("context_admission_")}
    assert fields == set(_HIERARCHICAL) | set(_STATIC) | set(_DECISION_MODEL)
    configurable = HindsightConfig.get_configurable_fields()
    assert set(_HIERARCHICAL) <= configurable
    assert not (set(_STATIC) | set(_DECISION_MODEL)) & configurable


def test_every_field_has_a_matching_env_constant():
    import hindsight_api.config as config_module

    for field_name in (*_HIERARCHICAL, *_STATIC):
        assert getattr(config_module, f"ENV_{field_name.upper()}") == _env_name(field_name)


def test_modes_and_scorers_match_the_engine_enums():
    assert tuple(m.value for m in AdmissionMode) == CONTEXT_ADMISSION_MODES
    assert tuple(s.value for s in ScorerKind) == CONTEXT_ADMISSION_SCORERS


def test_defaults(clean_env):
    config = HindsightConfig.from_env()
    for field_name, expected in {**_HIERARCHICAL, **_STATIC}.items():
        assert getattr(config, field_name) == expected, field_name
    assert config.decision_model_config("context_admission") is None


def test_static_proxy_hides_per_bank_fields(clean_env):
    proxy = StaticConfigProxy(HindsightConfig.from_env())
    with pytest.raises(ConfigFieldAccessError):
        proxy.context_admission_mode
    assert proxy.context_admission_scorer == "deterministic"


def test_env_values_are_read_and_weights_merge_over_the_defaults(clean_env):
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_MODE", " Filter ")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_SCORER", "HYBRID")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_CANDIDATES", "12")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_THRESHOLD_AUTO", "0.7")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_WEIGHTS", '{"bias": -1.5, "scope_mismatch.workspace": -2}')
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_SCOPE_KEYS", "repository,workspace")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_HARD_SCOPE_KEYS", "repository, workspace")
    config = HindsightConfig.from_env()
    config.validate()
    assert config.context_admission_mode == "filter"
    assert config.context_admission_scorer == "hybrid"
    assert config.context_admission_candidates == 12
    assert config.context_admission_threshold_auto == 0.7
    assert config.context_admission_weights["bias"] == -1.5
    assert config.context_admission_weights["scope_mismatch.workspace"] == -2.0
    # Narrowing the scope keys keeps the defaults' own weight keys valid.
    assert (
        config.context_admission_weights["scope_match.module"]
        == DEFAULT_CONTEXT_ADMISSION_WEIGHTS["scope_match.module"]
    )
    assert config.context_admission_hard_scope_keys == ["repository", "workspace"]


def test_a_hard_scope_key_outside_the_soft_keys_is_still_compared(clean_env):
    """The hard keys are compared with the soft ones, so a weight may name one, and the resolved
    settings compare it even when the soft keys leave it out."""
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_SCOPE_KEYS", "module,branch")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_HARD_SCOPE_KEYS", "tenant")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_WEIGHTS", '{"scope_mismatch.tenant": -2}')
    config = HindsightConfig.from_env()
    config.validate()
    settings = ContextAdmissionSettings.resolve({"context_admission_mode": "strict"}, config)
    assert settings.scope_keys == ("module", "branch")
    assert settings.compared_scope_keys == ("module", "branch", "tenant")
    assert settings.weights["scope_mismatch.tenant"] == -2.0


@pytest.mark.parametrize(
    ("field_name", "raw"),
    [
        ("context_admission_mode", "enforce"),
        ("context_admission_scorer", "llm"),
        ("context_admission_threshold_auto", "1.5"),
        ("context_admission_threshold_explicit", "nan"),
        ("context_admission_reference_bonus", "-0.1"),
        ("context_admission_candidates", "0"),
        ("context_admission_max_memories", "two"),
        ("context_admission_max_tokens", "-5"),
        ("context_admission_mmr_lambda", "1.2"),
        ("context_admission_dup_similarity", "inf"),
        ("context_admission_uncertain_margin", "2"),
        ("context_admission_jev_weight", "1.01"),
        ("context_admission_weights", "not json"),
        ("context_admission_weights", '{"reranker": "high"}'),
        ("context_admission_weights", '{"rerankr": 1.0}'),
        ("context_admission_weights", '{"scope_match.workspace": 1.0}'),
        ("context_admission_weights", '{"fact_type.opinion": 1.0}'),
        ("context_admission_weights", '{"bias": NaN}'),
    ],
)
def test_invalid_env_values_fail_startup(clean_env, field_name, raw):
    env_name = _env_name(field_name)
    clean_env.setenv(env_name, raw)
    with pytest.raises(ValueError, match=env_name):
        HindsightConfig.from_env().validate()


def test_decision_model_override_for_context_admission(clean_env):
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER", "typesafe")
    clean_env.setenv("HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_API_KEY", "test-key")
    config = HindsightConfig.from_env()
    setup = config.decision_model_config("context_admission")
    assert setup is not None and setup.provider == "typesafe" and setup.api_key == "test-key"
    # The other operations stay without one.
    assert config.decision_model_config("recall") is None


@pytest.mark.asyncio
async def test_bank_override_is_validated_and_normalized():
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    validated = await resolver.validate_bank_config_updates(
        _bank(),
        {
            "context_admission_mode": " STRICT ",
            "context_admission_threshold_auto": 0.8,
            "context_admission_candidates": 20,
            "context_admission_max_tokens": 500,
        },
    )
    assert validated.updates == {
        "context_admission_mode": "strict",
        "context_admission_threshold_auto": 0.8,
        "context_admission_candidates": 20,
        "context_admission_max_tokens": 500,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"context_admission_mode": "enforce"}, "context_admission_mode must be one of"),
        ({"context_admission_threshold_auto": 1.5}, "context_admission_threshold_auto"),
        ({"context_admission_threshold_explicit": math.nan}, "context_admission_threshold_explicit"),
        ({"context_admission_candidates": 0}, "context_admission_candidates must be an integer >= 1"),
        ({"context_admission_dup_similarity": -0.1}, "context_admission_dup_similarity"),
    ],
)
async def test_invalid_bank_override_is_rejected(updates, match):
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    with pytest.raises(ValueError, match=match):
        await resolver.validate_bank_config_updates(_bank(), updates)


@pytest.mark.asyncio
async def test_static_fields_cannot_be_overridden_per_bank():
    resolver = ConfigResolver(backend=FakeBankConfigBackend())
    for field_name, value in (("context_admission_scorer", "jev"), ("context_admission_hard_scope_keys", [])):
        with pytest.raises(ValueError, match="static"):
            await resolver.validate_bank_config_updates(_bank(), {field_name: value})


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"context_admission_mode": "enforce"}, {"context_admission_mode": "off"}),
        ({"context_admission_mode": " Observe "}, {"context_admission_mode": "observe"}),
    ],
)
def test_unknown_stored_mode_is_coerced_to_off(stored, expected):
    assert _coerce_stored_bank_overrides("bank", stored) == expected


def test_settings_resolve_leniently(clean_env):
    """A stored or tenant value out of range falls back to its default instead of failing recall."""
    config = HindsightConfig.from_env()
    settings = ContextAdmissionSettings.resolve(
        {
            "context_admission_mode": "Filter",
            "context_admission_threshold_auto": 7,
            "context_admission_candidates": True,
            "context_admission_max_memories": 3,
            "context_admission_mmr_lambda": math.inf,
        },
        config,
    )
    assert settings.mode is AdmissionMode.FILTER
    assert settings.threshold_auto == _HIERARCHICAL["context_admission_threshold_auto"]
    assert settings.candidates == _HIERARCHICAL["context_admission_candidates"]
    assert settings.max_memories == 3
    assert settings.mmr_lambda == _HIERARCHICAL["context_admission_mmr_lambda"]
    assert settings.scorer is ScorerKind.DETERMINISTIC
    assert settings.hard_scope_keys == ("repository",)
    assert ContextAdmissionSettings.resolve({"context_admission_mode": "turbo"}, config).mode is AdmissionMode.OFF
