"""Resolved context-admission settings for one recall on one bank."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...config import (
    DEFAULT_CONTEXT_ADMISSION_CANDIDATES,
    DEFAULT_CONTEXT_ADMISSION_DUP_SIMILARITY,
    DEFAULT_CONTEXT_ADMISSION_HARD_SCOPE_KEYS,
    DEFAULT_CONTEXT_ADMISSION_JEV_WEIGHT,
    DEFAULT_CONTEXT_ADMISSION_MAX_MEMORIES,
    DEFAULT_CONTEXT_ADMISSION_MAX_TOKENS,
    DEFAULT_CONTEXT_ADMISSION_MMR_LAMBDA,
    DEFAULT_CONTEXT_ADMISSION_REFERENCE_BONUS,
    DEFAULT_CONTEXT_ADMISSION_SCOPE_KEYS,
    DEFAULT_CONTEXT_ADMISSION_THRESHOLD_AUTO,
    DEFAULT_CONTEXT_ADMISSION_THRESHOLD_EXPLICIT,
    DEFAULT_CONTEXT_ADMISSION_UNCERTAIN_MARGIN,
    DEFAULT_CONTEXT_ADMISSION_WEIGHTS,
)
from .models import AdmissionMode, Purpose, ScorerKind


def mode_of(value: Any) -> AdmissionMode:
    """A stored mode, read leniently: anything unrecognisable is off, never an error.

    Env values are validated at startup and bank writes by the config API; an unknown value here
    is a legacy row or a tenant override, and "off" is the one reading that cannot change a payload.
    """
    if not isinstance(value, str):
        return AdmissionMode.OFF
    try:
        return AdmissionMode(value.strip().lower())
    except ValueError:
        return AdmissionMode.OFF


def _number(value: Any, default: float, *, low: float, high: float = math.inf) -> float:
    # bool is an int; a stored True is a type error, not 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value) if low <= value <= high else default


def _count(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else default


def _keys(value: Any, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return defaults
    keys = tuple(dict.fromkeys(k.strip() for k in value if isinstance(k, str) and k.strip()))
    return keys or defaults


def _weights(value: Any) -> Mapping[str, float]:
    merged = dict(DEFAULT_CONTEXT_ADMISSION_WEIGHTS)
    if isinstance(value, Mapping):
        for key, raw in value.items():
            number = _number(raw, math.nan, low=-math.inf)
            if isinstance(key, str) and math.isfinite(number):
                merged[key] = number
    return MappingProxyType(merged)


def _scorer(value: Any) -> ScorerKind:
    try:
        return ScorerKind(str(value).strip().lower())
    except ValueError:
        return ScorerKind.DETERMINISTIC


@dataclass(frozen=True)
class ContextAdmissionSettings:
    """Every context-admission knob, resolved for one bank. Field names drop the prefix."""

    mode: AdmissionMode
    threshold_auto: float
    threshold_explicit: float
    reference_bonus: float
    candidates: int
    max_memories: int
    max_tokens: int
    mmr_lambda: float
    dup_similarity: float
    scorer: ScorerKind
    weights: Mapping[str, float]
    scope_keys: tuple[str, ...]
    hard_scope_keys: tuple[str, ...]
    uncertain_margin: float
    jev_weight: float

    @classmethod
    def resolve(cls, bank_config: Mapping[str, Any], static_config: Any) -> "ContextAdmissionSettings":
        """Per-bank fields from ``bank_config`` (a resolved bank config dict), server-level ones
        from ``static_config``; anything missing or out of range takes its default."""
        return cls(
            mode=mode_of(bank_config.get("context_admission_mode")),
            threshold_auto=_number(
                bank_config.get("context_admission_threshold_auto"),
                DEFAULT_CONTEXT_ADMISSION_THRESHOLD_AUTO,
                low=0,
                high=1,
            ),
            threshold_explicit=_number(
                bank_config.get("context_admission_threshold_explicit"),
                DEFAULT_CONTEXT_ADMISSION_THRESHOLD_EXPLICIT,
                low=0,
                high=1,
            ),
            reference_bonus=_number(
                bank_config.get("context_admission_reference_bonus"),
                DEFAULT_CONTEXT_ADMISSION_REFERENCE_BONUS,
                low=0,
                high=1,
            ),
            candidates=_count(bank_config.get("context_admission_candidates"), DEFAULT_CONTEXT_ADMISSION_CANDIDATES),
            max_memories=_count(
                bank_config.get("context_admission_max_memories"), DEFAULT_CONTEXT_ADMISSION_MAX_MEMORIES
            ),
            max_tokens=_count(bank_config.get("context_admission_max_tokens"), DEFAULT_CONTEXT_ADMISSION_MAX_TOKENS),
            mmr_lambda=_number(
                bank_config.get("context_admission_mmr_lambda"), DEFAULT_CONTEXT_ADMISSION_MMR_LAMBDA, low=0, high=1
            ),
            dup_similarity=_number(
                bank_config.get("context_admission_dup_similarity"),
                DEFAULT_CONTEXT_ADMISSION_DUP_SIMILARITY,
                low=0,
                high=1,
            ),
            scorer=_scorer(getattr(static_config, "context_admission_scorer", None)),
            weights=_weights(getattr(static_config, "context_admission_weights", None)),
            scope_keys=_keys(
                getattr(static_config, "context_admission_scope_keys", None), DEFAULT_CONTEXT_ADMISSION_SCOPE_KEYS
            ),
            hard_scope_keys=_keys(
                getattr(static_config, "context_admission_hard_scope_keys", None),
                DEFAULT_CONTEXT_ADMISSION_HARD_SCOPE_KEYS,
            ),
            uncertain_margin=_number(
                getattr(static_config, "context_admission_uncertain_margin", None),
                DEFAULT_CONTEXT_ADMISSION_UNCERTAIN_MARGIN,
                low=0,
                high=1,
            ),
            jev_weight=_number(
                getattr(static_config, "context_admission_jev_weight", None),
                DEFAULT_CONTEXT_ADMISSION_JEV_WEIGHT,
                low=0,
                high=1,
            ),
        )

    @property
    def compared_scope_keys(self) -> tuple[str, ...]:
        """Every key the request is compared with memories on: the soft-scope keys, then any
        hard-scope key that is not one of them (it gates without weighing in the score)."""
        return tuple(dict.fromkeys((*self.scope_keys, *self.hard_scope_keys)))

    def threshold(self, purpose: Purpose, *, reference_signal: bool) -> float:
        """The admission threshold for ``purpose``; a memory-reference phrase lowers the auto one."""
        if purpose is Purpose.AUTO_CONTEXT:
            base = self.threshold_auto - (self.reference_bonus if reference_signal else 0.0)
            return min(1.0, max(0.0, base))
        return self.threshold_explicit
