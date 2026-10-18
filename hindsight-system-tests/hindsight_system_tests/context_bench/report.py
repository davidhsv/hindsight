"""Score every run file in ``<out>/runs`` and write per-arm JSON, ``summary.json`` and REPORT.md.

An *arm configuration* is (run, variant family, threshold, cap): the rows of one server
variant, optionally passed through a client-side score floor, cut to ``cap`` memories.
Families come from the variant names, so a new run (a feature server, a new mode) shows up
without touching this file:

- ``<run>:plain`` / ``<run>:tags`` — ungated (``tags`` = hard project scope);
- ``<run>:final`` / ``<run>:reranker`` / ``<run>:tags+final`` — server-side ``min_scores``
  sweeps, one configuration per threshold;
- ``<run>:plain+client_semantic`` / ``<run>:tags+client_semantic`` — the similarity
  threshold (arm B), applied here to ``scores.semantic`` of the ungated rows;
- ``<run>:plain+client_final`` / ``<run>:tags+client_final`` — arm C's final-score floor
  extended past 0.9, applied client-side (see ``CLIENT_SWEEPS``).

The default cap is ``INJECTION_CAP`` (10): what an agent would inject. Thresholds are picked
per family on the dev split by F0.5 (precision-favoured, as auto_context is), ties broken by
lower pollution; the holdout split is then scored once at that threshold.
"""

from __future__ import annotations

import inspect
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import metrics as metric_fns
from .metrics import Admitted, aggregate, approx_tokens, percentile
from .model import QUERY_CATEGORIES, Query
from .runner import SWEEP, QueryRow, ResultRow, RunRecord, load_record, typesafe_summary
from .scenarios import HOLDOUT, MEMORY_BY_LABEL, QUERIES

INJECTION_CAP = 10
#: A run labelled ``<run>-repeat`` re-asks some of ``<run>``'s queries; it is scored only as
#: agreement with ``<run>`` (run-to-run stability of a network model), never as an arm.
REPEAT_SUFFIX = "-repeat"
TIE_TOLERANCE = 1e-6
#: Where the admission arms' runs and dev fit live, apart from the baseline runs.
T2_DIR = "t2"
T2_SECTION_FILE = "dev-section.md"
T2_PREFIX = "## T2 "
T2_HEADING = f"{T2_PREFIX}dev"
T2_HOLDOUT_HEADING = f"{T2_PREFIX}holdout"


def insert_t2_section(report_path: Path, section: str, *, heading: str = T2_HEADING) -> None:
    """Put a T2 section (``heading``: the dev one by default) into an existing REPORT.md without
    re-scoring the baseline runs: it replaces a previous section under that heading, up to the next
    T2 section or the appendix (so regenerating one T2 section keeps the others), or goes before the
    appendix (where a regeneration puts it)."""
    text = report_path.read_text()
    body = section.rstrip() + "\n\n"
    appendix = text.find("## Arms not run")
    if appendix < 0:
        raise ValueError(f"{report_path} has no '## Arms not run' section to place the T2 section before")
    start = text.find(heading)
    if start >= 0:
        following = text.find(f"\n{T2_PREFIX}", start + len(heading))
        end = following + 1 if 0 <= following < appendix else appendix
        text = text[:start] + body + text[end:]
    else:
        text = text[:appendix] + body + text[appendix:]
    report_path.write_text(text)


#: Client-side floors over the ungated rows. ``semantic`` is arm B: the threshold sweep plus two
#: midpoints, because bge-small cosines of relevant memories sit in a narrow 0.7-0.9 band.
#: ``final`` extends arm C past the sweep's range: the server sweep peaks at its 0.9 edge, and the
#: server's ``min_scores.final`` equals this client-side floor on every swept threshold
#: (``server_vs_client_min_scores`` in summary.json), so the extension costs no server run.
CLIENT_SWEEPS = {"semantic": (*SWEEP, 0.75, 0.85), "final": (0.95, 0.97, 0.99)}

DISPLAY = {
    "base-local:plain": "A no gate",
    "base-local:plain+client_semantic": "B similarity threshold (client, scores.semantic)",
    "base-local:final": "C min_scores.final",
    "base-local:plain+client_final": "C min_scores.final, extended ≥0.95 (client-side)",
    "base-local:reranker": "C min_scores.reranker",
    "base-local:tags": "C' hard scope (tags)",
    "base-local:tags+client_semantic": "C'+B",
    "base-local:tags+final": "C'+C (tags + min_scores.final)",
    "base-local:tags+client_final": "C'+C, extended ≥0.95 (client-side)",
    "base-jev:plain": "D Jev (TypeSafe, prune_candidates)",
    "base-jev:tags": "C'+D (tags + Jev)",
}
ORDER = list(DISPLAY)

QUERY_BY_ID = {query.id: query for query in QUERIES}


@dataclass(frozen=True)
class Config:
    run: str
    family: str
    variant: str
    threshold: float | None
    client_field: str | None
    cap: int | None

    @property
    def key(self) -> str:
        parts = [self.family]
        if self.threshold is not None:
            parts.append(f"{self.threshold:.2f}")
        parts.append(f"cap{self.cap if self.cap is not None else 'all'}")
        return re.sub(r"[^A-Za-z0-9_.+-]+", "_", "@".join(parts))


def canonical(results: Sequence[ResultRow]) -> list[ResultRow]:
    """Server order, with runs of equal final scores (within ``TIE_TOLERANCE``) ordered by
    label: the server's order inside such a run is not a ranking, and rows must reproduce."""
    out: list[ResultRow] = []
    group: list[ResultRow] = []
    for result in results:
        previous = group[-1] if group else None
        if (
            previous is not None
            and result.final is not None
            and previous.final is not None
            and abs(previous.final - result.final) <= TIE_TOLERANCE
        ):
            group.append(result)
            continue
        out.extend(sorted(group, key=lambda row: row.label))
        group = [result]
    out.extend(sorted(group, key=lambda row: row.label))
    return out


def admitted_labels(row: QueryRow, config: Config) -> tuple[str, ...]:
    results = canonical(row.results)
    if config.client_field is not None and config.threshold is not None:
        results = [
            result
            for result in results
            if (value := getattr(result, config.client_field)) is not None and value >= config.threshold
        ]
    labels = [result.label for result in results]
    return tuple(labels if config.cap is None else labels[: config.cap])


@dataclass(frozen=True)
class _Family:
    """The arm a run variant belongs to, and the gate threshold it was run at (None: ungated)."""

    name: str
    threshold: float | None


def _family_of(variant: str) -> _Family | None:
    if "#" in variant:
        return None  # a repeat: determinism only
    match = re.fullmatch(r"(?:(tags)\+)?(final|reranker|semantic|keyword)@([0-9.]+)", variant)
    if match:
        prefix = f"{match.group(1)}+" if match.group(1) else ""
        return _Family(f"{prefix}{match.group(2)}", float(match.group(3)))
    return _Family(variant, None)


def configs_for(record: RunRecord) -> list[Config]:
    out: list[Config] = []
    variants = dict.fromkeys(row.variant for row in record.rows)
    # Arm B is a similarity gate over the shipping pipeline; over a TypeSafe run it would be a
    # different arm (D+B), so only local-reranker runs get it.
    has_semantic = record.server_env.get("HINDSIGHT_API_RERANKER_PROVIDER") != "typesafe" and any(
        result.semantic is not None for row in record.rows for result in row.results
    )
    for variant in variants:
        family = _family_of(variant)
        if family is None:
            continue
        threshold = family.threshold
        family_key = f"{record.label}:{family.name}"
        if threshold is None:
            for cap in (5, INJECTION_CAP, None):
                out.append(Config(record.label, family_key, variant, None, None, cap))
            if has_semantic and variant in ("plain", "tags"):
                for field, thresholds in CLIENT_SWEEPS.items():
                    for t in thresholds:
                        out.append(
                            Config(record.label, f"{family_key}+client_{field}", variant, t, field, INJECTION_CAP)
                        )
        else:
            out.append(Config(record.label, family_key, variant, threshold, None, INJECTION_CAP))
    return out


class Scorer:
    def __init__(self, records: Mapping[str, RunRecord]) -> None:
        self.records = records
        self.rows: dict[tuple[str, str], dict[str, QueryRow]] = defaultdict(dict)
        self.unlabelled: dict[str, str] = {}
        for record in records.values():
            self.unlabelled.update(record.unlabelled_texts)
            for row in record.rows:
                self.rows[(record.label, row.variant)][row.query_id] = row
        sizes = next(iter(records.values())).bank_sizes
        self.bank_sizes = sizes

    def tokens_of(self, label: str) -> int:
        memory = MEMORY_BY_LABEL.get(label)
        return approx_tokens(memory.text if memory else self.unlabelled.get(label, ""))

    def admitted(self, config: Config, queries: Sequence[Query]) -> list[Admitted]:
        rows = self.rows[(config.run, config.variant)]
        return [Admitted(query, admitted_labels(rows[query.id], config)) for query in queries if query.id in rows]

    def metrics(self, config: Config, queries: Sequence[Query]) -> dict[str, float | None]:
        return aggregate(
            self.admitted(config, queries),
            memories=MEMORY_BY_LABEL,
            bank_sizes=self.bank_sizes,
            tokens_of=self.tokens_of,
        )

    def latency(self, config: Config) -> dict[str, float | None]:
        values = [row.latency_ms for row in self.rows[(config.run, config.variant)].values() if row.status == 200]
        return {"p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95), "n": float(len(values))}


DEV = [query for query in QUERIES if query.id not in HOLDOUT]
HOLD = [query for query in QUERIES if query.id in HOLDOUT]


@dataclass(frozen=True, order=True)
class _Selection:
    """How a configuration ranks on dev: F0.5 first, then less pollution, then the higher threshold."""

    f05: float
    neg_pollution: float
    threshold: float


def _select(scorer: Scorer, configs: Sequence[Config]) -> Config:
    def key(config: Config) -> _Selection:
        metrics = scorer.metrics(config, DEV)
        return _Selection(metrics["f0.5"] or 0.0, -(metrics["context_pollution_rate"] or 0.0), config.threshold or 0.0)

    return max(configs, key=key)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


TABLE_METRICS = (
    ("context_precision", "prec"),
    ("useful_memory_recall", "recall"),
    ("context_pollution_rate", "pollution"),
    ("polluted_queries_rate", "polluted_q"),
    ("abstention_accuracy", "abstain_acc"),
    ("false_abstention_rate", "false_abstain"),
    ("hard_negative_admission_rate", "hard_neg_adm"),
    ("cross_scope_leakage@10", "leak@10"),
    ("stale_admission_rate", "stale"),
    ("refuted_admission_rate", "refuted"),
    ("false_positive_admission_rate", "FPR"),
    ("token_overhead", "tokens/q"),
    ("mean_admitted", "adm/q"),
    ("f0.5", "F0.5"),
)


def _row(label: str, metrics: Mapping[str, float | None], extra: Sequence[str] = ()) -> str:
    cells = [label, *extra]
    for key, _ in TABLE_METRICS:
        digits = 1 if key in ("token_overhead", "mean_admitted") else 3
        cells.append(_fmt(metrics.get(key), digits))
    return "| " + " | ".join(cells) + " |"


def _header(first: Sequence[str]) -> list[str]:
    names = [*first, *(short for _, short in TABLE_METRICS)]
    return ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]


def _family_label(family: str) -> str:
    return DISPLAY.get(family, family)


@dataclass(frozen=True, order=True)
class _FamilyOrder:
    """Report order: the arms in ``ORDER`` first, in that order, then any other by name."""

    rank: int
    family: str


def _family_sort(family: str) -> _FamilyOrder:
    return _FamilyOrder(ORDER.index(family) if family in ORDER else len(ORDER), family)


def determinism(scorer: Scorer, run: str, first: str, second: str) -> dict[str, Any] | None:
    a = scorer.rows.get((run, first))
    b = scorer.rows.get((run, second))
    if not a or not b:
        return None
    return _compare(a, b)


def _compare(a: Mapping[str, QueryRow], b: Mapping[str, QueryRow]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    raw_same = canon_same = top10_same = sets_same = top1_same = 0
    jaccards: list[float] = []
    max_delta = 0.0
    differing: list[str] = []
    for qid in common:
        ra, rb = a[qid].results, b[qid].results
        la, lb = [r.label for r in ra], [r.label for r in rb]
        ca, cb = [r.label for r in canonical(ra)], [r.label for r in canonical(rb)]
        raw_same += la == lb
        canon_same += ca == cb
        top10_same += ca[:INJECTION_CAP] == cb[:INJECTION_CAP]
        sets_same += set(la) == set(lb)
        top1_same += ca[:1] == cb[:1]
        union = set(ca[:INJECTION_CAP]) | set(cb[:INJECTION_CAP])
        jaccards.append(len(set(ca[:INJECTION_CAP]) & set(cb[:INJECTION_CAP])) / len(union) if union else 1.0)
        if ca != cb:
            differing.append(qid)
        scores_b = {r.label: r.final for r in rb}
        for r in ra:
            other = scores_b.get(r.label)
            if r.final is not None and other is not None:
                max_delta = max(max_delta, abs(r.final - other))
    n = len(common)
    return {
        "queries": n,
        "identical_raw_order": raw_same,
        "identical_canonical_order": canon_same,
        "identical_top10": top10_same,
        "identical_result_sets": sets_same,
        "identical_top1": top1_same,
        "mean_top10_jaccard": sum(jaccards) / n if n else None,
        "max_abs_final_delta": max_delta,
        "differing_queries": differing,
    }


def write_report(out_dir: Path) -> None:
    run_files = sorted(path for path in (out_dir / "runs").glob("*.json") if not path.name.endswith(".partial.json"))
    if not run_files:
        raise SystemExit(f"no run files in {out_dir / 'runs'}")
    records = {record.label: record for record in map(load_record, run_files)}
    scorer = Scorer(records)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    families: dict[str, list[Config]] = defaultdict(list)
    for record in records.values():
        if record.label.endswith(REPEAT_SUFFIX):
            continue  # a repeat only measures agreement with the run it repeats
        for config in configs_for(record):
            families[config.family].append(config)

    summary: dict[str, Any] = {"arms": {}, "selected": {}}
    for family, configs in families.items():
        for config in configs:
            payload = {
                "family": family,
                "display": _family_label(family),
                "run": config.run,
                "variant": config.variant,
                "threshold": config.threshold,
                "client_field": config.client_field,
                "cap": config.cap,
                "all": scorer.metrics(config, QUERIES),
                "dev": scorer.metrics(config, DEV),
                "holdout": scorer.metrics(config, HOLD),
                "per_category_dev": {
                    category: scorer.metrics(config, [q for q in DEV if q.category == category])
                    for category in QUERY_CATEGORIES
                },
                "per_category_all": {
                    category: scorer.metrics(config, [q for q in QUERIES if q.category == category])
                    for category in QUERY_CATEGORIES
                },
                "latency": scorer.latency(config),
            }
            (metrics_dir / f"{config.key}.json").write_text(json.dumps(payload, indent=1))
            summary["arms"][config.key] = {k: payload[k] for k in ("family", "threshold", "cap", "dev", "holdout")}

    selected: dict[str, Config] = {}
    for family, configs in families.items():
        capped = [c for c in configs if c.cap == INJECTION_CAP]
        if capped:
            selected[family] = _select(scorer, capped)
            summary["selected"][family] = selected[family].key

    lines = _markdown(scorer, records, families, selected)
    determinism_local = determinism(scorer, "base-local", "plain", "plain#2")
    repeats: dict[str, dict[str, Any]] = {}
    for (run, variant), rows in scorer.rows.items():
        base = run.removesuffix(REPEAT_SUFFIX)
        if run.endswith(REPEAT_SUFFIX) and (base, variant) in scorer.rows:
            repeats[f"{base}/{variant}"] = _compare(scorer.rows[(base, variant)], rows)
    summary["determinism_arm_a"] = determinism_local
    summary["repeat_agreement"] = repeats
    summary["typesafe"] = {
        label: typesafe_summary(r.typesafe_calls) for label, r in records.items() if r.typesafe_calls
    }
    summary["server_vs_client_min_scores"] = _min_scores_equivalence(scorer)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    lines += _determinism_section(
        determinism_local, repeats, summary["typesafe"], summary["server_vs_client_min_scores"]
    )
    t2_section = out_dir / T2_DIR / T2_SECTION_FILE
    if t2_section.is_file():
        # The admission arms' dev fit, written by `context_bench admission-section`; kept across
        # regenerations. Their runs live under t2/ so they are never scored as baseline arms here.
        lines += [t2_section.read_text().rstrip(), ""]
    lines += _appendix()
    findings = out_dir / "findings.md"
    if findings.is_file():
        # Hand-written analysis, kept across regenerations; placed after the title.
        lines[2:2] = [findings.read_text().rstrip(), ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


NOT_RUN = (
    "D' decision-model reranker (`HINDSIGHT_API_DECISION_MODEL_PROVIDER=typesafe`): not run. It orders the pool by one "
    "Noul question per candidate and hands back positions (`engine/search/decision_reranker.py`); it never prunes, so it "
    "cannot abstain or shrink the injected set, and it would spend ~300 TypeSafe questions per recall.",
    "Generative judge and the lifecycle/admission arms (B-F): need the feature server; see the commands below.",
)


COMMANDS = [
    "BASE=/path/to/base-worktree/hindsight-api-slim; FEAT=/path/to/feature-worktree/hindsight-api-slim",
    "# baselines A, B, C, C', C'+C (local embedder + cross-encoder, CPU)",
    "uv run python -m hindsight_system_tests.context_bench run --label base-local --api-dir $BASE --variants local-baselines",
    "# D and C'+D (TypeSafe reranker, prune_candidates=true); key from the environment, never logged",
    "set -a; source ~/.config/hindsight-adaptive/secrets.env; set +a",
    "uv run python -m hindsight_system_tests.context_bench run --label base-jev --profile jev --pg0 hindsight-ctxbench-jev:15531 \\",
    "    --api-dir $BASE --variants plain,tags",
    "uv run python -m hindsight_system_tests.context_bench run --label base-jev-repeat --profile jev --pg0 hindsight-ctxbench-jev:15531 \\",
    "    --api-dir $BASE --variants plain --queries <holdout ids> --reuse <out>/runs/base-jev.json",
    "# a feature-server arm: request context + purpose in every request, the admission mode as server env",
    "uv run python -m hindsight_system_tests.context_bench run --label feat-observe --api-dir $FEAT --send-context \\",
    "    --variants plain,tags --server-env HINDSIGHT_API_CONTEXT_ADMISSION_MODE=observe",
    "uv run python -m hindsight_system_tests.context_bench run --label feat-filter --api-dir $FEAT --send-context \\",
    "    --variants plain,tags,final@0.50 --server-env HINDSIGHT_API_CONTEXT_ADMISSION_MODE=filter",
    "# re-score every run in the out dir (client-side gates and thresholds need no server)",
    "uv run python -m hindsight_system_tests.context_bench report",
]


def _appendix() -> list[str]:
    lines = ["## Arms not run", ""]
    lines += [f"- {item}" for item in NOT_RUN]
    lines += [
        "",
        "## Running more arms",
        "",
        "From `hindsight-system-tests/` (each run gets its own schema; pass `--pg0` to use another instance):",
        "",
        "```bash",
    ]
    lines += COMMANDS
    lines += ["```"]
    lines += ["", "## Metric definitions (from `context_bench/metrics.py`)", ""]
    lines.append(inspect.getdoc(metric_fns) or "")
    lines.append("")
    for name in (
        "context_precision",
        "useful_memory_recall",
        "context_pollution_rate",
        "polluted_queries_rate",
        "abstention_accuracy",
        "false_abstention_rate",
        "false_positive_admission_rate",
        "false_negative_admission_rate",
        "hard_negative_admission_rate",
        "cross_scope_leakage",
        "stale_admission_rate",
        "refuted_admission_rate",
        "token_overhead",
        "f_beta",
    ):
        doc = " ".join((inspect.getdoc(getattr(metric_fns, name)) or "").split())
        lines.append(f"- `{name}`: {doc}")
    lines.append(f"- tokens: `approx_tokens` — {' '.join((inspect.getdoc(approx_tokens) or '').split())}")
    lines.append("")
    return lines


def _min_scores_equivalence(scorer: Scorer) -> dict[str, float] | None:
    """Whether server-side ``min_scores.final@t`` equals a client-side final-score floor over
    the ungated rows (top 10): a check that the server filter is what its docs say."""
    if ("base-local", "plain") not in scorer.rows:
        return None
    out: dict[str, float] = {}
    for t in SWEEP:
        variant = f"final@{t:.2f}"
        server = scorer.rows.get(("base-local", variant))
        if not server:
            continue
        client = Config("base-local", "x", "plain", t, "final", INJECTION_CAP)
        server_cfg = Config("base-local", "x", variant, None, None, INJECTION_CAP)
        plain = scorer.rows[("base-local", "plain")]
        same = sum(
            admitted_labels(plain[qid], client) == admitted_labels(server[qid], server_cfg)
            for qid in server
            if qid in plain
        )
        out[variant] = same / len(server)
    return out


def _markdown(
    scorer: Scorer,
    records: Mapping[str, RunRecord],
    families: Mapping[str, Sequence[Config]],
    selected: Mapping[str, Config],
) -> list[str]:
    from collections import Counter

    lines = ["# context_pollution benchmark — baseline arms", ""]
    lines.append(
        f"Scenario: {sum(1 for m in MEMORY_BY_LABEL.values() if m.bank == 'main')} memories in the mixed bank, "
        f"{sum(1 for m in MEMORY_BY_LABEL.values() if m.bank == 'session')} in the coding-session bank; "
        f"{len(QUERIES)} queries ({len(DEV)} dev / {len(HOLD)} holdout), "
        f"{sum(q.expected_abstain for q in QUERIES)} with nothing relevant. Injection cap {INJECTION_CAP}."
    )
    lines.append("")
    counts = Counter(q.category for q in QUERIES)
    hold_counts = Counter(q.category for q in HOLD)
    lines += ["| category | queries | holdout |", "|---|---|---|"]
    lines += [f"| {c} | {counts[c]} | {hold_counts[c]} |" for c in QUERY_CATEGORIES]
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for record in records.values():
        lines.append(
            f"- `{record.label}`: api `{record.api_dir}` @ {record.api_sha} (dirty={record.api_dirty}), "
            f"schema `{record.database_schema}`, {len(record.rows)} recalls, {record.duration_s}s, "
            f"reranker={record.server_env.get('HINDSIGHT_API_RERANKER_PROVIDER')}, "
            f"prune={record.server_env.get('HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES', 'n/a')}"
        )
    lines.append("")

    ordered = sorted(selected, key=_family_sort)
    lines += ["## Best configuration per arm (selected on dev by F0.5), cap 10", "", "### Dev split", ""]
    lines += _header(["arm", "threshold", "p50 ms", "p95 ms"])
    for family in ordered:
        config = selected[family]
        lat = scorer.latency(config)
        lines.append(
            _row(
                _family_label(family),
                scorer.metrics(config, DEV),
                [_fmt(config.threshold, 2), _fmt(lat["p50_ms"], 0), _fmt(lat["p95_ms"], 0)],
            )
        )
    lines += ["", "### Holdout split (scored once, at the dev-selected threshold)", ""]
    lines += _header(["arm", "threshold"])
    for family in ordered:
        config = selected[family]
        lines.append(_row(_family_label(family), scorer.metrics(config, HOLD), [_fmt(config.threshold, 2)]))
    lines.append("")

    lines += ["## Arm A cutoffs (no gate), all queries", ""]
    lines += _header(["cutoff"])
    for config in families.get("base-local:plain", []):
        lines.append(_row(f"top-{config.cap or 'all'}", scorer.metrics(config, QUERIES)))
    lines.append("")

    lines += ["## Threshold sweeps, dev split, cap 10", ""]
    for family in sorted(families, key=_family_sort):
        sweep = sorted((c for c in families[family] if c.threshold is not None), key=lambda c: c.threshold or 0)
        if not sweep:
            continue
        lines += [f"### {_family_label(family)}", ""]
        lines += _header(["threshold", "p50 ms"])
        for config in sweep:
            lines.append(
                _row(
                    _fmt(config.threshold, 2), scorer.metrics(config, DEV), [_fmt(scorer.latency(config)["p50_ms"], 0)]
                )
            )
        lines.append("")

    lines += ["## Per category (dev split, selected configurations)", ""]
    for key, short in (
        ("context_precision", "precision"),
        ("useful_memory_recall", "recall"),
        ("context_pollution_rate", "pollution"),
        ("abstention_accuracy", "abstention accuracy"),
    ):
        lines += [f"### {short}", ""]
        lines.append("| arm | " + " | ".join(QUERY_CATEGORIES) + " |")
        lines.append("|---|" + "---|" * len(QUERY_CATEGORIES))
        for family in ordered:
            config = selected[family]
            cells = [
                _fmt(scorer.metrics(config, [q for q in DEV if q.category == c]).get(key)) for c in QUERY_CATEGORIES
            ]
            lines.append(f"| {_family_label(family)} | " + " | ".join(cells) + " |")
        lines.append("")

    target = [q for q in QUERIES if q.category == "all_candidates_irrelevant"]
    lines += ["## all_candidates_irrelevant, all splits", ""]
    lines += [
        "| arm | threshold | returned [] | mean admitted | tokens/q | most common top-1 |",
        "|---|---|---|---|---|---|",
    ]
    for family in ordered:
        config = selected[family]
        rows = scorer.admitted(config, target)
        tops = Counter(row.labels[0] for row in rows if row.labels)
        empty = sum(1 for row in rows if not row.labels)
        tokens = sum(sum(scorer.tokens_of(label) for label in row.labels) for row in rows) / max(1, len(rows))
        mean = sum(len(row.labels) for row in rows) / max(1, len(rows))
        common = ", ".join(f"{label}×{n}" for label, n in tops.most_common(2)) or "–"
        lines.append(
            f"| {_family_label(family)} | {_fmt(config.threshold, 2)} | {empty}/{len(rows)} | {mean:.1f} | {tokens:.0f} | {common} |"
        )
    lines.append("")
    lines += _examples(scorer, selected)
    return lines


EXAMPLE_QUERIES = ("h01", "h02", "h05", "w01", "w02", "x01", "e01", "z04", "z03", "s13", "cs1", "e02", "n06", "m01")


def _examples(scorer: Scorer, selected: Mapping[str, Config]) -> list[str]:
    families = [
        f for f in ("base-local:final", "base-local:tags+final", "base-jev:plain", "base-jev:tags") if f in selected
    ]
    plain = scorer.rows.get(("base-local", "plain"), {})
    lines = ["## Examples (query, candidate, scores, decision, expected)", ""]
    lines.append(
        "Candidates: arm A's top 3 plus anything a gated arm admitted. Scores are arm A's "
        "(semantic cosine / final). Decision columns: ✓ admitted, · not admitted. "
        "Expected: R relevant, A acceptable, H hard negative, x other irrelevant."
    )
    lines.append("")
    for qid in EXAMPLE_QUERIES:
        query = QUERY_BY_ID.get(qid)
        if query is None or qid not in plain:
            continue
        decisions = {
            f: set(admitted_labels(scorer.rows[(selected[f].run, selected[f].variant)][qid], selected[f]))
            for f in families
            if qid in scorer.rows[(selected[f].run, selected[f].variant)]
        }
        a_top = [r.label for r in canonical(plain[qid].results)[:3]]
        candidates = list(dict.fromkeys([*a_top, *(label for f in families for label in sorted(decisions.get(f, ())))]))
        scores = {r.label: r for r in plain[qid].results}
        lines.append(
            f"**{qid}** [{query.category}, {query.purpose}] “{query.text}” — expected {list(query.relevant) or '[] (abstain)'}"
        )
        lines.append("")
        head = (
            ["candidate", "sem", "final", "A@10"]
            + [f"{_family_label(f).split(' ')[0]}@{_fmt(selected[f].threshold, 2)}" for f in families]
            + ["expected"]
        )
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        a_admitted = set(admitted_labels(plain[qid], Config("base-local", "a", "plain", None, None, INJECTION_CAP)))
        for label in candidates:
            memory = MEMORY_BY_LABEL.get(label)
            text = (memory.text if memory else scorer.unlabelled.get(label, ""))[:70]
            result = scores.get(label)
            expected = (
                "R"
                if label in query.relevant
                else "A"
                if label in query.acceptable
                else "H"
                if label in query.hard_negatives
                else "x"
            )
            cells = [
                f"`{label}` {text}…",
                _fmt(result.semantic if result else None, 2),
                _fmt(result.final if result else None, 3),
                "✓" if label in a_admitted else "·",
                *("✓" if label in decisions.get(f, set()) else "·" for f in families),
                expected,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _determinism_section(
    local: Mapping[str, Any] | None,
    repeats: Mapping[str, Mapping[str, Any]],
    typesafe: Mapping[str, Any],
    equivalence: Mapping[str, float] | None,
) -> list[str]:
    lines = ["## Determinism", ""]
    if local:
        lines.append(
            f"- Arm A run twice in one server process (`plain` vs `plain#2`, {local['queries']} queries): "
            f"identical raw server order {local['identical_raw_order']}, identical after tie canonicalisation "
            f"{local['identical_canonical_order']}, identical top-10 {local['identical_top10']}, "
            f"max |Δ final| {local['max_abs_final_delta']:.2e}; differing: {local['differing_queries'] or 'none'}."
        )
    for name, agreement in repeats.items():
        n = agreement["queries"]
        lines.append(
            f"- `{name}` run twice (separate server processes, {n} queries): identical admitted list (cap 10) "
            f"{agreement['identical_top10']}/{n}, identical top-1 {agreement['identical_top1']}/{n}, mean Jaccard of "
            f"admitted sets {_fmt(agreement['mean_top10_jaccard'])}; differing: {agreement['differing_queries'] or 'none'}."
        )
    if equivalence:
        lines.append(
            "- Server `min_scores.final@t` vs the same floor applied client-side to arm A's rows "
            "(share of queries with an identical top-10): " + ", ".join(f"{k} {v:.3f}" for k, v in equivalence.items())
        )
    for label, usage in typesafe.items():
        lines.append(
            f"- TypeSafe usage in `{label}`: {usage['calls']} calls {usage['by_question']}, non-200 {usage['non_200']}, "
            f"retried {usage['retried_calls']}, reported usage {usage['usage_totals']}, "
            f"call latency p50 {_fmt(usage['latency_ms_p50'], 0)} ms / p95 {_fmt(usage['latency_ms_p95'], 0)} ms."
        )
    lines.append("")
    return lines
