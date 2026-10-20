"""Dev-only fitting and selection of the pre-registered context-admission arms.

Arms (all on the feature server, adaptive memory in lifecycle mode, request ``context`` and
``purpose`` sent):

- ``E-det``: deterministic scorer, filter mode;
- ``E-jev``: decision-model (Jev) scorer, filter mode;
- ``F``: hybrid scorer, strict mode (hard scope, Jev, lifecycle, redundancy, budget).

One *record* run (hybrid scorer, strict, a trace per recall) captures every dev candidate's
features and the decision model's probability. From it, offline:

1. the deterministic weights are fit by logistic regression on candidate features against the
   dev relevance labels (relevant 1, irrelevant 0, acceptable left out);
2. each arm is replayed over the 0.30-0.90 grid of auto and explicit thresholds, for each weight
   candidate (the fit, its class-balanced and prior-shifted forms, the server's hand-set defaults);
3. the pre-registered rule (:func:`.admission.select`) picks the setting.

Only dev rows are ever read: :func:`dev_rows` refuses a run that holds a holdout row of the variant.
:func:`verify_live` replays a live run's own recalls and compares them with what the server did.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .admission import (
    RECALL_FLOOR,
    THRESHOLD_GRID,
    GateSettings,
    LogisticFit,
    ReplayCheck,
    SettingResult,
    admitted_results,
    candidate_scores,
    decision_model_probabilities,
    fit_logistic,
    replay,
    select,
    similarity_function,
    verify_replay,
)
from .metrics import Admitted, aggregate, approx_tokens, percentile
from .model import Query
from .report import INJECTION_CAP, T2_SECTION_FILE, Config, admitted_labels
from .runner import AdmissionRow, QueryRow, RunRecord, load_record
from .scenarios import HOLDOUT, MEMORY_BY_LABEL, QUERIES

QUERY_BY_ID = {query.id: query for query in QUERIES}
DEV_QUERIES = [query for query in QUERIES if query.id not in HOLDOUT]

#: The server's hand-set starting weights (``DEFAULT_CONTEXT_ADMISSION_WEIGHTS`` in the api's
#: config.py). Every admission run passes a complete weight map, starting from this one, so
#: the record states the weights it ran with; the replay's recomputation of every recorded score
#: part fails loudly if the server's arithmetic or weights differ from these.
ENGINE_DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": -3.0,
    "reranker": 4.0,
    "semantic": 2.0,
    "bm25": 0.3,
    "arm_agreement": 0.4,
    "rank": 0.6,
    "retrievability": 0.4,
    "importance": 0.4,
    "uncertain": -0.4,
    "token_cost": -0.1,
    "scope_match.repository": 0.8,
    "scope_match.module": 0.6,
    "scope_match.environment": 0.3,
    "scope_match.branch": 0.1,
    "scope_match.service": 0.3,
    "scope_mismatch.repository": -3.0,
    "scope_mismatch.module": -1.2,
    "scope_mismatch.environment": -0.8,
    "scope_mismatch.branch": -0.2,
    "scope_mismatch.service": -0.8,
    "fact_type.world": 0.0,
    "fact_type.experience": 0.0,
    "fact_type.observation": 0.0,
}

ENV = "HINDSIGHT_API_CONTEXT_ADMISSION_"
#: Server-level (static) settings every admission run states explicitly, at the server's defaults.
STATIC_DEFAULTS: dict[str, str] = {
    "HINDSIGHT_API_ADAPTIVE_MEMORY_MODE": "lifecycle",
    "HINDSIGHT_API_ADAPTIVE_MEMORY_RECALL_REINFORCEMENT_ENABLED": "false",
    f"{ENV}JEV_WEIGHT": "0.5",
    f"{ENV}UNCERTAIN_MARGIN": "0.1",
    f"{ENV}HARD_SCOPE_KEYS": "repository",
    f"{ENV}SCOPE_KEYS": "repository,module,environment,branch,service",
}


@dataclass(frozen=True)
class Arm:
    name: str
    scorer: str
    mode: str
    #: Whether the deterministic weights are part of the arm's setting (chosen on dev).
    weighted: bool


ARMS: tuple[Arm, ...] = (
    Arm("E-det", "deterministic", "filter", True),
    Arm("E-jev", "jev", "filter", False),
    Arm("F", "hybrid", "strict", True),
)

SELECTION_RULE = (
    "Per arm, on the 85 dev queries only: minimise context_pollution_rate subject to useful_memory_recall >= 0.85, "
    "ties broken by higher abstention_accuracy; if no setting reaches recall 0.85, maximise useful_memory_recall - "
    "context_pollution_rate. Remaining ties: higher useful_memory_recall, then higher auto threshold, then higher "
    "explicit threshold. Candidates: auto and explicit thresholds on the 0.30-0.90 grid (step 0.05) and, for E-det "
    "and F, the deterministic weights (the dev logistic fit; the same fit with class-balanced sample weights; the "
    "unweighted fit with its intercept shifted to a 50% prior; or the server's hand-set defaults). Metrics at the "
    "injection cap of 10."
)

#: Which parts of SELECTION_RULE the pre-registration fixed before the dev fit, and which it
#: does not cover; the report prints the three groups under the rule. Endpoints decided by the last
#: two groups are reported as exploratory (pre-registration amendment of 2026-10-18 01:45 BRT).
SELECTION_RULE_PREREGISTERED = (
    "on the 85 dev queries only, minimise context_pollution_rate subject to useful_memory_recall >= 0.85, ties "
    "broken by higher abstention_accuracy; if no setting reaches recall 0.85, maximise useful_memory_recall - "
    "context_pollution_rate; thresholds on the 0.30-0.90 range; for E-det and F, deterministic weights fit by "
    "logistic regression on dev candidate features against dev relevance labels (pass 1's candidates: the dev "
    "logistic fit and the server's hand-set defaults)."
)
SELECTION_RULE_ADDED_AFTER_DEV_RESULTS = (
    "the class-balanced fit (pass 2) and the prior-shifted intercept (pass 3) as weight candidates, each added "
    "after a dev sweep was seen."
)
SELECTION_RULE_NOT_IN_PREREGISTRATION = (
    "the remaining tie-breaks (higher useful_memory_recall, then the higher auto threshold, then the higher "
    "explicit threshold); choosing the auto and explicit thresholds jointly; the 0.05 grid step; the injection cap "
    "of 10. When during the dev passes these were written is not recorded; all of them precede the holdout."
)

#: Recorded in chosen.json and the report: what was decided after a dev result was seen.
DEVIATIONS = (
    "Pass 1 (unweighted logistic fit and the hand-set defaults as weight candidates; chosen.json sha256 "
    "7eb77db5bcff1c8488f4bf11115ac75f827a51b8ab58db8d9199a7188bc9f3d5, superseded): the unweighted fit, calibrated "
    "to the 3.1% base rate of relevant candidates, cannot reach useful recall 0.85 anywhere in the pre-registered "
    "0.30-0.90 grid (best 0.763 at 0.30), so E-det fell back to the hand-set defaults at the grid edge.",
    "Pass 2 added the class-balanced fit (sample weights n / (2 n_class)) as a candidate (chosen.json sha256 "
    "1a8b76dcee4a2b01636b7933c896257e5b22421d5605e13816f25c678e1ae676, superseded). Its dev sweep was seen.",
    "Pass 3 (this file) added the other textbook balanced-prior form, the unweighted fit with its intercept shifted "
    "by log(negatives / positives) (same ranking as the fit, read at a 50% prior), before it was scored; no further "
    "candidate is added. All passes used dev data only; no holdout query was run or read.",
    "The deterministic weights of E-jev (used only when the decision model fails) are E-det's chosen weights; the "
    "selection rule does not weigh them.",
)


def server_env_for(scorer: str, weights: Mapping[str, float]) -> dict[str, str]:
    """The server env (no secrets) an arm runs with; the decision model's key comes from the
    ``jev-admission`` profile."""
    env = dict(STATIC_DEFAULTS)
    env[f"{ENV}SCORER"] = scorer
    env[f"{ENV}WEIGHTS"] = json.dumps(dict(weights), sort_keys=True)
    if scorer != "deterministic":
        env[f"{ENV}DECISION_MODEL_PROVIDER"] = "typesafe"
    return env


def variant_for(mode: str, threshold_auto: float, threshold_explicit: float) -> str:
    return f"adm={mode}+ta={threshold_auto:.2f}+te={threshold_explicit:.2f}"


# --------------------------------------------------------------------------- reading runs


class HoldoutRowError(Exception):
    """A run handed to the dev tooling holds a holdout row; nothing of it is read."""


def dev_rows(record: RunRecord, variant: str) -> dict[str, QueryRow]:
    """The variant's dev rows by query id. Refuses a run with any holdout row of the variant
    (before reading any of its content) and any failed recall."""
    rows = [row for row in record.rows if row.variant == variant]
    if not rows:
        raise ValueError(f"{record.label} has no rows for variant {variant!r}")
    held = sorted(row.query_id for row in rows if row.query_id in HOLDOUT or row.split != "dev")
    if held:
        raise HoldoutRowError(f"{record.label}/{variant} holds {len(held)} holdout rows; the dev tooling reads none")
    failed = [row.query_id for row in rows if row.status != 200 or row.admission is None]
    if failed:
        raise ValueError(f"{record.label}/{variant}: failed or unrecorded recalls {failed}")
    return {row.query_id: row for row in rows}


def settings_of(record: RunRecord, variant: str) -> GateSettings:
    """What the server applied to a recorded variant: static settings from the run's server env
    (which every admission run states in full), per-bank ones from the config the server resolved."""
    env = record.server_env
    missing = [key for key in (*STATIC_DEFAULTS, f"{ENV}SCORER", f"{ENV}WEIGHTS") if key not in env]
    if missing:
        raise ValueError(f"{record.label}: server env does not state {missing}")
    weights = json.loads(env[f"{ENV}WEIGHTS"])
    unknown = sorted(set(ENGINE_DEFAULT_WEIGHTS) - set(weights))
    if unknown:
        raise ValueError(f"{record.label}: the weight map leaves {unknown} to the server's defaults")
    bank = record.admission_settings.get(variant)
    if bank is None:
        raise ValueError(f"{record.label}: no admission settings recorded for {variant!r}")
    return GateSettings(
        scorer=env[f"{ENV}SCORER"],
        mode=bank["context_admission_mode"],
        weights=weights,
        threshold_auto=float(bank["context_admission_threshold_auto"]),
        threshold_explicit=float(bank["context_admission_threshold_explicit"]),
        reference_bonus=float(bank["context_admission_reference_bonus"]),
        uncertain_margin=float(env[f"{ENV}UNCERTAIN_MARGIN"]),
        jev_weight=float(env[f"{ENV}JEV_WEIGHT"]),
        mmr_lambda=float(bank["context_admission_mmr_lambda"]),
        dup_similarity=float(bank["context_admission_dup_similarity"]),
        max_memories=int(bank["context_admission_max_memories"]),
        max_tokens=int(bank["context_admission_max_tokens"]),
        hard_scope_keys=tuple(k.strip() for k in env[f"{ENV}HARD_SCOPE_KEYS"].split(",") if k.strip()),
    )


def load_embeddings(record_path: Path, record: RunRecord) -> dict[str, list[float]]:
    if record.embeddings_file is None:
        return {}
    return json.loads((record_path.parent / record.embeddings_file).read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- scoring a setting


def _tokens_of(label: str) -> int:
    memory = MEMORY_BY_LABEL.get(label)
    return approx_tokens(memory.text) if memory else 0


def metrics_of(admitted: Sequence[Admitted], bank_sizes: Mapping[str, int]) -> dict[str, float | None]:
    return aggregate(admitted, memories=MEMORY_BY_LABEL, bank_sizes=bank_sizes, tokens_of=_tokens_of)


_CAP = Config("t2", "t2", "t2", None, None, INJECTION_CAP)


def capped(results: Sequence[Any]) -> tuple[str, ...]:
    """What an agent would inject: the report's tie canonicalisation, then the cap."""
    row = QueryRow(variant="t2", query_id="-", split="dev", latency_ms=0.0, status=200, results=list(results))
    return admitted_labels(row, _CAP)


@dataclass
class Prepared:
    """One dev recall, ready to replay: the recorded row, its decision-model probabilities and
    the similarity function over its candidates."""

    query: Query
    row: QueryRow
    admission: AdmissionRow
    probabilities: list[float | None]
    similarity: Callable[[str, str], float]


def prepare(
    record_path: Path, variant: str, *, embeddings: Mapping[str, Sequence[float]] | None = None
) -> tuple[RunRecord, GateSettings, list[Prepared]]:
    """``embeddings`` replace the run's own side file (a run that queried another run's banks)."""
    record = load_record(record_path)
    rows = dev_rows(record, variant)
    source = settings_of(record, variant)
    if embeddings is None:
        embeddings = load_embeddings(record_path, record)
    prepared: list[Prepared] = []
    for query in DEV_QUERIES:
        row = rows.get(query.id)
        if row is None:
            raise ValueError(f"{record.label}/{variant} has no row for dev query {query.id}")
        admission = row.admission
        assert admission is not None  # dev_rows refuses rows without one
        prepared.append(
            Prepared(
                query=query,
                row=row,
                admission=admission,
                probabilities=decision_model_probabilities(admission, source),
                similarity=similarity_function([d.label for d in admission.decisions], embeddings),
            )
        )
    return record, source, prepared


def evaluate(
    prepared: Sequence[Prepared],
    base: GateSettings,
    grid: Sequence[tuple[float, float]],
    bank_sizes: Mapping[str, int],
) -> dict[tuple[float, float], dict[str, float | None]]:
    """Dev metrics of ``base`` at every (auto, explicit) threshold pair of ``grid``."""
    scores = [candidate_scores(p.admission, base, p.probabilities) for p in prepared]
    out: dict[tuple[float, float], dict[str, float | None]] = {}
    for auto, explicit in grid:
        settings = base.with_thresholds(auto, explicit)
        admitted = [
            Admitted(
                p.query,
                capped(admitted_results(replay(p.admission, settings, scores=s, similarity=p.similarity))),
            )
            for p, s in zip(prepared, scores)
        ]
        out[(auto, explicit)] = metrics_of(admitted, bank_sizes)
    return out


def fit_samples(prepared: Sequence[Prepared]) -> tuple[list[tuple[dict[str, float], int]], dict[str, int]]:
    """(features, relevant?) per judged dev candidate; acceptable memories are left out."""
    samples: list[tuple[dict[str, float], int]] = []
    counts = {"relevant": 0, "irrelevant": 0, "acceptable_skipped": 0, "unlabelled": 0}
    for p in prepared:
        for decision in p.admission.decisions:
            if decision.label.startswith("?"):
                counts["unlabelled"] += 1
            if decision.label in p.query.acceptable:
                counts["acceptable_skipped"] += 1
                continue
            relevant = decision.label in p.query.relevant
            counts["relevant" if relevant else "irrelevant"] += 1
            samples.append((dict(decision.features), int(relevant)))
    return samples, counts


def fitted_weights(fit: LogisticFit) -> dict[str, float]:
    """The complete weight map the server gets: the fit's weights (constant features at 0, folded
    into the bias); features the dev data never set keep the server's defaults."""
    out = dict(ENGINE_DEFAULT_WEIGHTS)
    out.update(fit.weights)
    return out


# --------------------------------------------------------------------------- the dev fit


def _arm_settings(arm: Arm, source: GateSettings, weights: Mapping[str, float]) -> GateSettings:
    return GateSettings(
        scorer=arm.scorer,
        mode=arm.mode,
        weights=weights,
        threshold_auto=source.threshold_auto,
        threshold_explicit=source.threshold_explicit,
        reference_bonus=source.reference_bonus,
        uncertain_margin=source.uncertain_margin,
        jev_weight=source.jev_weight,
        mmr_lambda=source.mmr_lambda,
        dup_similarity=source.dup_similarity,
        max_memories=source.max_memories,
        max_tokens=source.max_tokens,
        hard_scope_keys=source.hard_scope_keys,
    )


GRID_2D: tuple[tuple[float, float], ...] = tuple((a, e) for a in THRESHOLD_GRID for e in THRESHOLD_GRID)


def _metric_row(m: Mapping[str, float | None]) -> dict[str, float | None]:
    keys = (
        "context_precision",
        "useful_memory_recall",
        "context_pollution_rate",
        "polluted_queries_rate",
        "abstention_accuracy",
        "false_abstention_rate",
        "cross_scope_leakage@10",
        "stale_admission_rate",
        "refuted_admission_rate",
        "hard_negative_admission_rate",
        "token_overhead",
        "mean_admitted",
    )
    return {key: m.get(key) for key in keys}


def run_dev_fit(record_path: Path, variant: str, out_dir: Path, *, l2: float = 1.0) -> dict[str, Any]:
    """Fit, sweep and select every arm on the dev split; write ``chosen.json`` (+ its sha256)
    and the per-arm sweeps under ``out_dir``. Returns the chosen document."""
    record, source, prepared = prepare(record_path, variant)
    samples, counts = fit_samples(prepared)
    fit = fit_logistic(samples, l2=l2)
    balanced = fit_logistic(samples, l2=l2, class_balanced=True)
    weight_options = {
        "logistic_fit": fitted_weights(fit),
        "logistic_fit_balanced": fitted_weights(balanced),
        "logistic_fit_prior_shift": {**fitted_weights(fit), "bias": fit.prior_shifted()["bias"]},
        "engine_default": dict(ENGINE_DEFAULT_WEIGHTS),
    }

    sweeps: dict[str, dict[str, Any]] = {}
    chosen_arms: dict[str, Any] = {}
    det_choice: str | None = None
    for arm in ARMS:
        options = (
            weight_options
            if arm.weighted
            else {det_choice or "logistic_fit": weight_options[det_choice or "logistic_fit"]}
        )
        results: list[SettingResult] = []
        grids: dict[str, dict[tuple[float, float], dict[str, float | None]]] = {}
        for name, weights in options.items():
            grid = evaluate(prepared, _arm_settings(arm, source, weights), GRID_2D, record.bank_sizes)
            grids[name] = grid
            results += [SettingResult(name, a, e, m) for (a, e), m in grid.items()]
        best = select(results)
        if arm.name == "E-det":
            det_choice = best.weights_name
        weights = options[best.weights_name]
        grid = grids[best.weights_name]
        sweeps[arm.name] = {
            "chosen": {"weights": best.weights_name, "auto": best.threshold_auto, "explicit": best.threshold_explicit},
            "feasible_settings": sum(
                1 for r in results if (r.metrics.get("useful_memory_recall") or 0.0) >= RECALL_FLOOR
            ),
            "settings": [
                {
                    "weights": r.weights_name,
                    "auto": r.threshold_auto,
                    "explicit": r.threshold_explicit,
                    **_metric_row(r.metrics),
                }
                for r in results
            ],
            "curve_shared": [{"threshold": t, **_metric_row(grids[best.weights_name][(t, t)])} for t in THRESHOLD_GRID],
            "curve_auto": [{"threshold": t, **_metric_row(grid[(t, best.threshold_explicit)])} for t in THRESHOLD_GRID],
            "curve_explicit": [{"threshold": t, **_metric_row(grid[(best.threshold_auto, t)])} for t in THRESHOLD_GRID],
            "per_weights_best": {
                name: _setting_summary(select([r for r in results if r.weights_name == name])) for name in options
            },
        }
        chosen_arms[arm.name] = {
            "scorer": arm.scorer,
            "mode": arm.mode,
            "weights_source": best.weights_name,
            "thresholds": {"auto": best.threshold_auto, "explicit": best.threshold_explicit},
            "variant": variant_for(arm.mode, best.threshold_auto, best.threshold_explicit),
            "weights": weights,
            "server_env": server_env_for(arm.scorer, weights),
            "bank_config": {
                "context_admission_mode": arm.mode,
                "context_admission_threshold_auto": best.threshold_auto,
                "context_admission_threshold_explicit": best.threshold_explicit,
                **{
                    key: record.admission_settings[variant][key]
                    for key in (
                        "context_admission_reference_bonus",
                        "context_admission_candidates",
                        "context_admission_max_memories",
                        "context_admission_max_tokens",
                        "context_admission_mmr_lambda",
                        "context_admission_dup_similarity",
                    )
                },
            },
            "request": {"send_context": True, "purpose": "the query's declared purpose", "variant_base": "plain"},
            "decision_model": None if arm.scorer == "deterministic" else "typesafe (profile jev-admission)",
            "dev": _metric_row(best.metrics),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev").mkdir(exist_ok=True)
    for name, sweep in sweeps.items():
        (out_dir / "dev" / f"sweep-{name}.json").write_text(json.dumps(sweep, indent=1))
    fit_doc = {
        "l2": fit.l2,
        "samples": fit.samples,
        "positives": fit.positives,
        "counts": counts,
        "fitted_features": list(fit.fitted),
        "constant_features_folded_into_bias": list(fit.constant),
        "absent_features_kept_at_default": list(fit.absent),
        "fits": {
            name: {
                "class_balanced": f.class_balanced,
                "iterations": f.iterations,
                "penalized_log_loss_per_sample": f.log_loss,
                "weights": f.weights,
            }
            for name, f in (("logistic_fit", fit), ("logistic_fit_balanced", balanced))
        },
        "prior_shift_bias": fit.prior_shifted()["bias"],
    }
    (out_dir / "dev" / "fit.json").write_text(json.dumps(fit_doc, indent=1))
    chosen = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "dev_queries": len(prepared),
        "selection_rule": SELECTION_RULE,
        "deviations": list(DEVIATIONS),
        "threshold_grid": list(THRESHOLD_GRID),
        "injection_cap": INJECTION_CAP,
        "record": {
            "label": record.label,
            "variant": variant,
            "file_sha256": file_sha256(record_path),
            "api_dir": record.api_dir,
            "api_sha": record.api_sha,
            "api_dirty": record.api_dirty,
            "server_env": record.server_env,
        },
        "logistic_fit": fit_doc,
        "arms": chosen_arms,
    }
    path = out_dir / "chosen.json"
    path.write_text(json.dumps(chosen, indent=1, sort_keys=False) + "\n")
    (out_dir / "chosen.sha256").write_text(f"{file_sha256(path)}  chosen.json\n")
    return chosen


def _setting_summary(result: SettingResult) -> dict[str, Any]:
    return {
        "auto": result.threshold_auto,
        "explicit": result.threshold_explicit,
        **_metric_row(result.metrics),
    }


# --------------------------------------------------------------------------- live verification


def verify_live(
    live_path: Path,
    variant: str,
    *,
    record_path: Path | None = None,
    record_variant: str | None = None,
) -> dict[str, Any]:
    """Replay a live run's dev recalls (a) from their own traces under the settings the server
    applied, and (b) when ``record_path`` is given, from the record run's traces, i.e. across server
    processes; each compared with the live decisions and results. (b) must be exact for the
    deterministic scorer; under a decision-model scorer the record run's answers are another
    sample, so (b) measures sampling variation instead. Also returns the live run's dev metrics at
    the injection cap.

    A live run that queried the record run's banks (``--reuse``) and saved no embeddings of its
    own is replayed with the record run's, after checking that both name the same schema and banks.
    """
    embeddings: Mapping[str, Sequence[float]] | None = None
    live_record = load_record(live_path)
    if live_record.embeddings_file is None and record_path is not None:
        record = load_record(record_path)
        if (record.database_schema, record.banks) != (live_record.database_schema, live_record.banks):
            raise ValueError(f"{live_record.label} did not query {record.label}'s banks; its embeddings do not apply")
        embeddings = load_embeddings(record_path, record)
    live_record, live_settings, live = prepare(live_path, variant, embeddings=embeddings)
    own = ReplayCheck()
    for p in live:
        scores = candidate_scores(p.admission, live_settings, p.probabilities)
        decisions = replay(p.admission, live_settings, scores=scores, similarity=p.similarity)
        verify_replay(p.row.results, p.admission, decisions, own, query_id=p.query.id)
    out: dict[str, Any] = {
        "live": {"label": live_record.label, "variant": variant, "file_sha256": file_sha256(live_path)},
        "settings": {
            "scorer": live_settings.scorer,
            "mode": live_settings.mode,
            "auto": live_settings.threshold_auto,
            "explicit": live_settings.threshold_explicit,
        },
        "own_trace": _check_doc(own),
        "jev_errors": sum(1 for p in live if p.admission.jev_error),
        "decision_model_unanswered": sum(
            1 for p in live for prob in p.probabilities if prob is None and live_settings.scorer != "deterministic"
        ),
        "dev_live": _metric_row(
            metrics_of([Admitted(p.query, capped(p.row.results)) for p in live], live_record.bank_sizes)
        ),
        # Whole-recall latency on the (loaded) bench machine; informational only.
        "recall_latency_ms": {
            "p50": percentile([p.row.latency_ms for p in live], 50),
            "p95": percentile([p.row.latency_ms for p in live], 95),
        },
    }
    if record_path is not None and record_variant is not None:
        record, _source, recorded = prepare(record_path, record_variant)
        cross = ReplayCheck()
        by_query = {p.query.id: p for p in live}
        replayed: list[Admitted] = []
        deltas: list[float] = []
        for p in recorded:
            target = by_query[p.query.id]
            scores = candidate_scores(p.admission, live_settings, p.probabilities)
            decisions = replay(p.admission, live_settings, scores=scores, similarity=p.similarity)
            verify_replay(target.row.results, target.admission, decisions, cross, query_id=p.query.id)
            replayed.append(Admitted(p.query, capped(admitted_results(decisions))))
            if [d.label for d in p.admission.decisions] == [d.label for d in target.admission.decisions]:
                deltas += [
                    abs(a - b) for a, b in zip(p.probabilities, target.probabilities) if a is not None and b is not None
                ]
        doc: dict[str, Any] = {"label": record.label, "variant": record_variant, **_check_doc(cross)}
        if live_settings.scorer != "deterministic":
            # The record run's decision-model answers are another sample: this measures how much a
            # fresh sample moves the decisions and the metrics, not replay exactness.
            doc["exactness_expected"] = False
            doc["dev_replayed_on_record_sample"] = _metric_row(metrics_of(replayed, live_record.bank_sizes))
            doc["probability_abs_diff"] = {
                "candidates": len(deltas),
                "mean": sum(deltas) / len(deltas) if deltas else None,
                "share_over_0.1": sum(1 for d in deltas if d > 0.1) / len(deltas) if deltas else None,
                "max": max(deltas) if deltas else None,
            }
        else:
            doc["exactness_expected"] = True
        out["from_record"] = doc
    return out


def _check_doc(check: ReplayCheck) -> dict[str, Any]:
    return {
        "queries": check.queries,
        "exact": check.exact,
        "candidates": check.candidates,
        "all_exact": check.all_exact,
        "queries_with_redundant": check.redundant_queries,
        "queries_with_budget": check.budget_queries,
        "queries_with_wrong_scope": check.wrong_scope_queries,
        "reasons": dict(sorted(check.reasons.items())),
        "mismatches": check.mismatches,
    }


# --------------------------------------------------------------------------- the report section

SECTION_FILE = T2_SECTION_FILE


def _fmt(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


_CURVE_COLUMNS = (
    ("context_precision", "prec"),
    ("useful_memory_recall", "recall"),
    ("context_pollution_rate", "pollution"),
    ("polluted_queries_rate", "polluted_q"),
    ("abstention_accuracy", "abstain_acc"),
    ("false_abstention_rate", "false_abstain"),
    ("cross_scope_leakage@10", "leak@10"),
    ("stale_admission_rate", "stale"),
    ("token_overhead", "tokens/q"),
    ("mean_admitted", "adm/q"),
)


def _curve_table(first: str, rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    head = [first, *(short for _, short in _CURVE_COLUMNS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [_fmt(row[key], 2)]
        for name, _ in _CURVE_COLUMNS:
            cells.append(_fmt(row.get(name), 1 if name in ("token_overhead", "mean_admitted") else 3))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _baseline_dev(out_dir: Path) -> Mapping[str, Any] | None:
    """The tags + Jev baseline's dev metrics from the baseline report's summary, when present."""
    summary = out_dir.parent / "summary.json"
    if not summary.is_file():
        return None
    data = json.loads(summary.read_text())
    key = data.get("selected", {}).get("base-jev:tags")
    return data.get("arms", {}).get(key, {}).get("dev") if key else None


def write_section(out_dir: Path, *, verifications: Sequence[Mapping[str, Any]] = ()) -> str:
    """Render the ``T2 dev`` report section from ``chosen.json``, the sweeps and any live
    verifications, write it to ``out_dir / SECTION_FILE`` and return it."""
    chosen = json.loads((out_dir / "chosen.json").read_text())
    sha = (out_dir / "chosen.sha256").read_text().split()[0]
    lines = ["## T2 dev (context admission arms, dev split only)", ""]
    record = chosen["record"]
    lines.append(
        f"Record run `{record['label']}` (variant `{record['variant']}`, file sha256 `{record['file_sha256'][:16]}…`), "
        f"api `{record['api_dir']}` @ {record['api_sha']} (dirty={record['api_dirty']}); "
        f"{chosen['dev_queries']} dev queries, injection cap {chosen['injection_cap']}. No holdout query was run or "
        "read for any admission arm."
    )
    lines.append("")
    lines.append(f"**chosen.json sha256: `{sha}`** (written before any holdout run).")
    lines.append("")
    lines.append(f"Selection rule as applied (chosen.json `selection_rule`): {chosen['selection_rule']}")
    lines.append("")
    lines += [
        "Of that rule:",
        "",
        f"- **Pre-registered** (2026-10-17 23:50 amendment): {SELECTION_RULE_PREREGISTERED}",
        f"- **Added on dev after a dev result was seen:** {SELECTION_RULE_ADDED_AFTER_DEV_RESULTS}",
        f"- **Not in the pre-registration:** {SELECTION_RULE_NOT_IN_PREREGISTRATION}",
        "",
        "The 2026-10-18 01:45 BRT amendment to PREREGISTRATION.md records the last two groups. Under the "
        "pre-registration's rule for later changes, the holdout endpoints they decide are reported as exploratory: "
        "every E-det and F endpoint (weights from passes 2 and 3), the harm benchmark's `gated` condition (arm F), "
        "and E-jev's endpoints that its explicit-threshold tie-break moves (pollution, precision, polluted queries, "
        "tokens). See the T2 holdout section.",
        "",
    ]
    fit = chosen["logistic_fit"]
    seen = {
        *fit["fitted_features"],
        *fit["constant_features_folded_into_bias"],
        *fit["absent_features_kept_at_default"],
    }
    unseen = sorted(k for k in ENGINE_DEFAULT_WEIGHTS if k not in seen and k != "bias")
    lines.append(
        f"Logistic fits (L2 {fit['l2']} on standardized features, intercept unpenalized, Newton): {fit['samples']} "
        f"judged dev candidates ({fit['positives']} relevant; {fit['counts']['acceptable_skipped']} acceptable left "
        f"out). Fitted: {', '.join(fit['fitted_features'])}. Constant on dev (weight 0, folded into the bias): "
        f"{', '.join(fit['constant_features_folded_into_bias']) or 'none'}. Never non-zero on dev, so not "
        f"identifiable (server default kept): "
        f"{', '.join([*fit['absent_features_kept_at_default'], *unseen]) or 'none'}. Some fitted signs run against "
        "intuition (e.g. arm_agreement and scope_match.repository negative, token_cost positive): they describe this "
        "scenario's dev candidates, where most hard negatives share the task's repository; the holdout tests whether "
        "they transfer."
    )
    lines.append("")
    names = list(fit["fits"])
    lines += ["| feature | " + " | ".join(names) + " | server default |", "|---|" + "---|" * (len(names) + 1)]
    features = sorted({k for f in fit["fits"].values() for k in f["weights"]})
    for feature in features:
        cells = [f"{fit['fits'][n]['weights'].get(feature, 0.0):+.4f}" for n in names]
        lines.append(f"| {feature} | " + " | ".join(cells) + f" | {ENGINE_DEFAULT_WEIGHTS.get(feature, 0.0):+.2f} |")
    lines.append("")
    if "prior_shift_bias" in fit:
        lines.append(
            f"`logistic_fit_prior_shift` is `logistic_fit` with bias {fit['prior_shift_bias']:+.4f} "
            "(+ log(negatives / positives)), every other weight unchanged."
        )
        lines.append("")
    if chosen.get("deviations"):
        lines.append("Deviations (decided on dev, recorded before any holdout run):")
        lines.append("")
        lines += [f"- {item}" for item in chosen["deviations"]]
        lines.append("")
    lines += ["### Chosen settings and dev metrics (offline replay of the record run)", ""]
    head = ["arm", "scorer", "mode", "weights", "auto", "explicit", *(short for _, short in _CURVE_COLUMNS)]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, arm in chosen["arms"].items():
        cells = [
            name,
            arm["scorer"],
            arm["mode"],
            arm["weights_source"],
            _fmt(arm["thresholds"]["auto"], 2),
            _fmt(arm["thresholds"]["explicit"], 2),
        ]
        for key, _ in _CURVE_COLUMNS:
            cells.append(_fmt(arm["dev"].get(key), 1 if key in ("token_overhead", "mean_admitted") else 3))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    baseline = _baseline_dev(out_dir)
    if baseline is not None:
        lines.append(
            f"For reference, the pre-registered comparator C'+D (tags + Jev pruning, base server) on the same dev "
            f"queries: precision {_fmt(baseline.get('context_precision'))}, recall "
            f"{_fmt(baseline.get('useful_memory_recall'))}, pollution {_fmt(baseline.get('context_pollution_rate'))}, "
            f"polluted queries {_fmt(baseline.get('polluted_queries_rate'))}, abstention accuracy "
            f"{_fmt(baseline.get('abstention_accuracy'))}."
        )
        lines.append("")
    for name in chosen["arms"]:
        sweep = json.loads((out_dir / "dev" / f"sweep-{name}.json").read_text())
        pick = sweep["chosen"]
        weighted = next(arm.weighted for arm in ARMS if arm.name == name)
        lines += [
            f"### {name}: dev threshold sweeps (weights `{pick['weights']}`)"
            if weighted
            else f"### {name}: dev threshold sweeps (decision-model scores; fallback weights `{pick['weights']}`)",
            "",
        ]
        lines.append(
            f"{sweep['feasible_settings']} of {len(sweep['settings'])} grid settings reach recall ≥ {RECALL_FLOOR}. "
            "Best setting per weight candidate: "
            + "; ".join(
                f"`{w}` auto {_fmt(s['auto'], 2)} / explicit {_fmt(s['explicit'], 2)} → recall "
                f"{_fmt(s['useful_memory_recall'])}, pollution {_fmt(s['context_pollution_rate'])}, abstention "
                f"{_fmt(s['abstention_accuracy'])}"
                for w, s in sweep["per_weights_best"].items()
            )
            + "."
        )
        lines.append("")
        lines.append("Shared threshold (auto = explicit = t):")
        lines.append("")
        lines += _curve_table("t", sweep["curve_shared"], "threshold")
        lines.append("")
        candidates = sorted({r["weights"] for r in sweep["settings"]})
        if len(candidates) > 1:
            lines.append("Shared threshold per weight candidate (recall / pollution / abstention accuracy):")
            lines.append("")
            lines += ["| t | " + " | ".join(candidates) + " |", "|---|" + "---|" * len(candidates)]
            shared = {(r["weights"], r["auto"]): r for r in sweep["settings"] if r["auto"] == r["explicit"]}
            for t in THRESHOLD_GRID:
                cells = []
                for name in candidates:
                    r = shared[(name, t)]
                    cells.append(
                        f"{_fmt(r['useful_memory_recall'])} / {_fmt(r['context_pollution_rate'])} / "
                        f"{_fmt(r['abstention_accuracy'])}"
                    )
                lines.append(f"| {_fmt(t, 2)} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.append(f"Auto threshold, explicit fixed at {_fmt(pick['explicit'], 2)}:")
        lines.append("")
        lines += _curve_table("auto", sweep["curve_auto"], "threshold")
        lines.append("")
        lines.append(f"Explicit threshold, auto fixed at {_fmt(pick['auto'], 2)}:")
        lines.append("")
        lines += _curve_table("explicit", sweep["curve_explicit"], "threshold")
        lines.append("")
    if verifications:
        lines += ["### Replay exactness and live dev runs at the chosen settings", ""]
        lines.append(
            "Each live run is replayed from its own trace under the settings the server applied (per-candidate admitted "
            "flag, reason and duplicate count, and the returned list); deterministic runs are also replayed from the "
            "record run's trace (another server process)."
        )
        lines.append("")
        head = [
            "live run / variant",
            "own trace exact",
            "from record exact",
            "q w/ redundant",
            "q w/ budget",
            "q w/ wrong_scope",
            "jev errors",
            "recall p50/p95 ms",
            *(short for _, short in _CURVE_COLUMNS),
        ]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for v in verifications:
            own = v["own_trace"]
            cross = v.get("from_record")
            cells = [
                f"`{v['live']['label']}` `{v['live']['variant']}`",
                f"{own['exact']}/{own['queries']}",
                (
                    f"{cross['exact']}/{cross['queries']}"
                    + ("" if cross.get("exactness_expected", True) else " (other Jev sample)")
                )
                if cross
                else "–",
                str(own["queries_with_redundant"]),
                str(own["queries_with_budget"]),
                str(own["queries_with_wrong_scope"]),
                str(v["jev_errors"]),
                f"{_fmt(v.get('recall_latency_ms', {}).get('p50'), 0)} / {_fmt(v.get('recall_latency_ms', {}).get('p95'), 0)}",
            ]
            for key, _ in _CURVE_COLUMNS:
                cells.append(_fmt(v["dev_live"].get(key), 1 if key in ("token_overhead", "mean_admitted") else 3))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        sampled = [
            v for v in verifications if v.get("from_record") and not v["from_record"].get("exactness_expected", True)
        ]
        if sampled:
            lines.append(
                "Decision-model sampling: the live Jev arms' settings replayed on the record run's Jev answers (a "
                "different sample of the same requests) against the live run's own sample."
            )
            lines.append("")
            head = [
                "live run / variant",
                "sample",
                "mean abs Δp",
                "share abs Δp > 0.1",
                "max abs Δp",
                "prec",
                "recall",
                "pollution",
                "abstain_acc",
            ]
            lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for v in sampled:
                cross = v["from_record"]
                diff = cross["probability_abs_diff"]
                for sample, m in (("live", v["dev_live"]), ("record", cross["dev_replayed_on_record_sample"])):
                    lines.append(
                        f"| `{v['live']['label']}` `{v['live']['variant']}` | {sample} | {_fmt(diff['mean'])} | "
                        f"{_fmt(diff['share_over_0.1'])} | {_fmt(diff['max'])} | {_fmt(m['context_precision'])} | "
                        f"{_fmt(m['useful_memory_recall'])} | {_fmt(m['context_pollution_rate'])} | "
                        f"{_fmt(m['abstention_accuracy'])} |"
                    )
            lines.append("")
        for v in verifications:
            for name, check in (("own trace", v["own_trace"]), ("from record", v.get("from_record"))):
                if check and check.get("exactness_expected", True) is False:
                    continue
                if check and check["mismatches"]:
                    lines.append(
                        f"- `{v['live']['label']}` {name}: mismatches in {', '.join(m['query'] for m in check['mismatches'])}"
                    )
        lines.append("")
    lines += ["### Running an arm at its chosen setting", ""]
    lines.append(
        "Every static setting is in the arm's `server_env` in chosen.json (adaptive memory lifecycle, scorer, the "
        "complete weight map, Jev weight, uncertain margin, scope keys); the bank-level ones are set by the arm's "
        "`variant` (`adm=<mode>+ta=<auto>+te=<explicit>`, patched before the queries; the rest at their recorded "
        "defaults). Jev/hybrid arms add `--profile jev-admission` (TypeSafe key from the environment, through the "
        "usage-recording proxy). Requests carry the query's `context` and declared `purpose` (`--send-context`)."
    )
    lines.append("")
    lines += [
        "```python",
        "# from hindsight-system-tests/: uv run python - (one arm per server; FEAT = the feature api dir)",
        "import json",
        "from hindsight_system_tests.context_bench.__main__ import main",
        "arm = json.load(open('<out>/t2/chosen.json'))['arms']['F']  # or 'E-det', 'E-jev'",
        "argv = ['run', '--label', 't2-F-<split>', '--api-dir', FEAT, '--send-context', '--split', '<split>',",
        "        '--variants', arm['variant'], '--out', '<out>/t2']",
        "argv += [f'--server-env={k}={v}' for k, v in arm['server_env'].items()]",
        "if arm['scorer'] != 'deterministic':",
        "    argv += ['--profile', 'jev-admission']  # needs TYPESAFE_API_KEY in the environment",
        "main(argv)",
        "```",
        "",
        "The dev tooling here (`admission-fit`, `admission-verify`) refuses any run holding a holdout row.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / SECTION_FILE).write_text(text)
    return text
