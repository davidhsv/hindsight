"""Resolved adaptive-memory settings for one operation on one bank."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...config import (
    ADAPTIVE_MEMORY_CONFLICT_POLICIES,
    ADAPTIVE_MEMORY_VIEWS,
    DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_ENABLED,
    DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_GRACE_DAYS,
    DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_PROTECT_IMPORTANCE,
    DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_RECALL_MIN_SCORE,
    DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_THRESHOLD,
    DEFAULT_ADAPTIVE_MEMORY_BASE_STABILITY_DAYS,
    DEFAULT_ADAPTIVE_MEMORY_CONFLICT_POLICY,
    DEFAULT_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL,
    DEFAULT_ADAPTIVE_MEMORY_DEFAULT_VIEW,
    DEFAULT_ADAPTIVE_MEMORY_HISTORICAL_CUES,
    DEFAULT_ADAPTIVE_MEMORY_MAX_STABILITY_DAYS,
    DEFAULT_ADAPTIVE_MEMORY_PRUNE_AFTER_DAYS,
    DEFAULT_ADAPTIVE_MEMORY_PRUNE_ENABLED,
    DEFAULT_ADAPTIVE_MEMORY_RECALL_REINFORCEMENT_ENABLED,
    DEFAULT_ADAPTIVE_MEMORY_REFLECT_REINFORCEMENT_ENABLED,
    DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_COOLDOWN_HOURS,
    DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_ENABLED,
    DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_GAIN,
    DEFAULT_ADAPTIVE_MEMORY_SCOPE_KEYS,
    DEFAULT_ADAPTIVE_MEMORY_SCORE_ALPHA,
    DEFAULT_ADAPTIVE_MEMORY_SCORE_FLOOR,
    DEFAULT_ADAPTIVE_MEMORY_SOURCE_TRUST,
    DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS,
    DEFAULT_ADAPTIVE_MEMORY_TYPE_STABILITY,
    HindsightConfig,
    StaticConfigProxy,
)
from .models import AdaptiveMode


def _mode(value: Any) -> AdaptiveMode:
    """A stored mode, read leniently: anything unrecognisable is off, never an error.

    Env values are validated strictly at startup and bank writes by the config API, so an
    unknown value here can only be a legacy row or a tenant override. Recall must not fail on
    it, and treating it as "off" is the one reading that cannot change a payload.
    """
    if isinstance(value, AdaptiveMode):
        return value
    if not isinstance(value, str):
        return AdaptiveMode.OFF
    try:
        return AdaptiveMode(value.strip().lower())
    except ValueError:
        return AdaptiveMode.OFF


def _finite(value: Any, default: float) -> float:
    # bool is an int; a stored True is a type error, not 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _unit_interval(value: Any, default: float) -> float:
    return min(1.0, max(0.0, _finite(value, default)))


def _positive(value: Any, default: float) -> float:
    number = _finite(value, default)
    return number if number > 0 else default


def _non_negative(value: Any, default: float) -> float:
    number = _finite(value, default)
    return number if number >= 0 else default


def _flag(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _choice(value: Any, choices: tuple[str, ...], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in choices:
        return value.strip().lower()
    return default


def _keys(value: Any, defaults: tuple[str, ...]) -> tuple[str, ...]:
    """A list-of-names setting: its non-empty strings, stripped and deduplicated in order; the
    defaults when it holds none (``None`` is a config built by field introspection in tests)."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return defaults
    keys = tuple(dict.fromkeys(k.strip() for k in value if isinstance(k, str) and k.strip()))
    return keys or defaults


def lifecycle_scope_keys(static: HindsightConfig | StaticConfigProxy) -> tuple[str, ...]:
    """The server's ``HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS``: the metadata keys a claim's scope
    fingerprint reads in lifecycle consolidation. Consolidation's prompt renders them too, so both
    read them through this one function."""
    return _keys(getattr(static, "adaptive_memory_scope_keys", None), DEFAULT_ADAPTIVE_MEMORY_SCOPE_KEYS)


def _weights(value: Any, defaults: Mapping[str, float]) -> Mapping[str, float]:
    """A JSON-map setting merged over its defaults, read-only.

    ``None`` (a config built by field introspection in tests) means the defaults. Non-finite or
    negative entries are dropped back to the default rather than trusted.
    """
    merged = dict(defaults)
    if isinstance(value, Mapping):
        for key, raw in value.items():
            number = _finite(raw, math.nan)
            if isinstance(key, str) and math.isfinite(number) and number >= 0:
                merged[key] = number
    return MappingProxyType(merged)


@dataclass(frozen=True)
class AdaptiveSettings:
    """Every adaptive-memory knob, resolved for one bank, plus the mode recall actually runs.

    ``mode`` is what the bank asked for; ``effective_mode`` is what runs — ``off`` when the
    memories store cannot hold adaptive state for the bank, whatever was asked. Built per
    operation by :meth:`resolve` from values the caller already holds; nothing is cached.

    The per-bank values are clamped defensively on read (alpha and floor to [0, 1], max
    stability to at least base): the config API validates writes, but tenant overrides and
    legacy rows are never validated and a bad one must not produce a factor outside its bound.
    Without the server's ``adaptive_memory_decay_experimental`` the alpha is 0 whatever the bank
    or tenant holds, so retention never moves a score.
    """

    mode: AdaptiveMode
    effective_mode: AdaptiveMode
    # Per bank (hierarchical).
    score_alpha: float
    score_floor: float
    reinforcement_enabled: bool
    base_stability_days: float
    max_stability_days: float
    default_view: str
    historical_cues: bool
    # Server level (static).
    recall_reinforcement_enabled: bool
    reflect_reinforcement_enabled: bool
    reinforcement_cooldown_hours: float
    reinforcement_gain: float
    source_weights: Mapping[str, float]
    type_stability: Mapping[str, float]
    source_trust: Mapping[str, float]
    conflict_policy: str
    archive_enabled: bool
    archive_threshold: float
    archive_grace_days: float
    archive_recall_min_score: float
    archive_protect_importance: float
    prune_enabled: bool
    prune_after_days: float
    #: Metadata keys whose values make up a claim's scope fingerprint (lifecycle consolidation).
    scope_keys: tuple[str, ...] = DEFAULT_ADAPTIVE_MEMORY_SCOPE_KEYS

    @classmethod
    def resolve(
        cls,
        bank_config: Mapping[str, Any],
        static: HindsightConfig | StaticConfigProxy,
        *,
        store_supports: bool,
    ) -> "AdaptiveSettings":
        """Resolve from a bank's resolved config dict and the static (server) config.

        ``bank_config`` is the dict recall already fetched (``get_bank_config``); a missing key
        reads as its default, a missing or unknown mode as off. ``store_supports`` is
        ``MemoriesExtension.supports_adaptive_state_for(bank_id)``.
        """
        mode = _mode(bank_config.get("adaptive_memory_mode"))
        decay_allowed = _flag(
            getattr(static, "adaptive_memory_decay_experimental", None), DEFAULT_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL
        )
        base = _positive(
            bank_config.get("adaptive_memory_base_stability_days"), DEFAULT_ADAPTIVE_MEMORY_BASE_STABILITY_DAYS
        )
        maximum = max(
            base,
            _positive(
                bank_config.get("adaptive_memory_max_stability_days"), DEFAULT_ADAPTIVE_MEMORY_MAX_STABILITY_DAYS
            ),
        )
        return cls(
            mode=mode,
            effective_mode=mode if store_supports else AdaptiveMode.OFF,
            score_alpha=(
                _unit_interval(bank_config.get("adaptive_memory_score_alpha"), DEFAULT_ADAPTIVE_MEMORY_SCORE_ALPHA)
                if decay_allowed
                else 0.0
            ),
            score_floor=_unit_interval(
                bank_config.get("adaptive_memory_score_floor"), DEFAULT_ADAPTIVE_MEMORY_SCORE_FLOOR
            ),
            reinforcement_enabled=_flag(
                bank_config.get("adaptive_memory_reinforcement_enabled"), DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_ENABLED
            ),
            base_stability_days=base,
            max_stability_days=maximum,
            default_view=_choice(
                bank_config.get("adaptive_memory_default_view"),
                ADAPTIVE_MEMORY_VIEWS,
                DEFAULT_ADAPTIVE_MEMORY_DEFAULT_VIEW,
            ),
            historical_cues=_flag(
                bank_config.get("adaptive_memory_historical_cues"), DEFAULT_ADAPTIVE_MEMORY_HISTORICAL_CUES
            ),
            recall_reinforcement_enabled=_flag(
                static.adaptive_memory_recall_reinforcement_enabled,
                DEFAULT_ADAPTIVE_MEMORY_RECALL_REINFORCEMENT_ENABLED,
            ),
            reflect_reinforcement_enabled=_flag(
                static.adaptive_memory_reflect_reinforcement_enabled,
                DEFAULT_ADAPTIVE_MEMORY_REFLECT_REINFORCEMENT_ENABLED,
            ),
            reinforcement_cooldown_hours=_non_negative(
                static.adaptive_memory_reinforcement_cooldown_hours,
                DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_COOLDOWN_HOURS,
            ),
            reinforcement_gain=_non_negative(
                static.adaptive_memory_reinforcement_gain, DEFAULT_ADAPTIVE_MEMORY_REINFORCEMENT_GAIN
            ),
            source_weights=_weights(static.adaptive_memory_source_weights, DEFAULT_ADAPTIVE_MEMORY_SOURCE_WEIGHTS),
            type_stability=_weights(static.adaptive_memory_type_stability, DEFAULT_ADAPTIVE_MEMORY_TYPE_STABILITY),
            source_trust=_weights(static.adaptive_memory_source_trust, DEFAULT_ADAPTIVE_MEMORY_SOURCE_TRUST),
            conflict_policy=_choice(
                static.adaptive_memory_conflict_policy,
                ADAPTIVE_MEMORY_CONFLICT_POLICIES,
                DEFAULT_ADAPTIVE_MEMORY_CONFLICT_POLICY,
            ),
            archive_enabled=_flag(static.adaptive_memory_archive_enabled, DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_ENABLED),
            archive_threshold=_unit_interval(
                static.adaptive_memory_archive_threshold, DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_THRESHOLD
            ),
            archive_grace_days=_non_negative(
                static.adaptive_memory_archive_grace_days, DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_GRACE_DAYS
            ),
            archive_recall_min_score=_unit_interval(
                static.adaptive_memory_archive_recall_min_score, DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_RECALL_MIN_SCORE
            ),
            archive_protect_importance=_unit_interval(
                static.adaptive_memory_archive_protect_importance, DEFAULT_ADAPTIVE_MEMORY_ARCHIVE_PROTECT_IMPORTANCE
            ),
            prune_enabled=_flag(static.adaptive_memory_prune_enabled, DEFAULT_ADAPTIVE_MEMORY_PRUNE_ENABLED),
            prune_after_days=_non_negative(
                static.adaptive_memory_prune_after_days, DEFAULT_ADAPTIVE_MEMORY_PRUNE_AFTER_DAYS
            ),
            scope_keys=lifecycle_scope_keys(static),
        )
