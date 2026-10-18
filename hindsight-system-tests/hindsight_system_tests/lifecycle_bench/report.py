"""Markdown from run JSON: one run's report, a cross-run table, and a row-level diff.

Every number printed here is read back from a run file's rows and aggregates; nothing is
recomputed from a server, so a report can be regenerated from the JSON alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .metrics import K_VALUES
from .runner import MetricValue, QueryRow, RunRecord

#: Report columns: metric name → short header.
KEY_METRICS: dict[str, str] = {
    "current_precision": "prec",
    "current_recall": "rec",
    "stale_contamination": "stale",
    "stale_contamination_raw": "stale raw",
    "stale_contamination_obs": "stale obs",
    "mixed_rate": "mixed",
    "refuted_leakage": "refuted",
    "historical_recall": "hist",
    "scope_accuracy": "scope",
    "supersession_accuracy": "supers",
    "uncertainty_accuracy": "uncert",
    "independent_reinforcement_accuracy": "indep",
}

_LOWER_IS_BETTER = {"stale_contamination", "stale_contamination_raw", "stale_contamination_obs", "refuted_leakage"}


def _cell(value: MetricValue | None, *, with_n: bool = False) -> str:
    if value is None or value.mean is None:
        return "N/A"
    return f"{value.mean:.3f} ({value.n})" if with_n else f"{value.mean:.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_run(record: RunRecord) -> str:
    out = [f"# memory_lifecycle — {record.label}", ""]
    meta = [
        ("arm", record.arm),
        ("api dir", f"`{record.api_dir}`"),
        ("api sha", f"`{record.api_sha}`" + (" (dirty)" if record.api_dirty else "")),
        ("harness sha", f"`{record.harness_sha}`" + (" (dirty)" if record.harness_dirty else "")),
        ("encoding", f"{record.encoding} (lifecycle form when advertised: {record.lifecycle_form})"),
        ("payload source", record.payload_source),
        ("embedder / reranker", record.embedder),
        ("bank config", ", ".join(f"{k}={v}" for k, v in record.bank_config.items()) or "—"),
        ("recall types", ", ".join(record.recall_types) if record.recall_types else "default (all)"),
        (
            "server env",
            ", ".join(
                f"{k}={v}" for k, v in sorted(record.server_env.items()) if not k.startswith(("HF_", "TOKENIZERS"))
            )
            or "—",
        ),
        ("scenarios / queries", f"{len(record.scenarios)} / {len(record.rows)}"),
        ("holdout scenarios", ", ".join(record.holdout)),
        ("started / duration", f"{record.started_at} / {record.duration_s}s"),
        ("invalid scenarios", ", ".join(record.invalid_scenarios) or "none"),
    ]
    out += [_table(["", ""], [[k, v] for k, v in meta]), ""]

    out += [
        "## Metrics @5 by split",
        "",
        "Cells are mean (n queries the metric applied to); N/A = applied to none.",
        "",
    ]
    out.append(
        _table(
            ["split", *KEY_METRICS.values()],
            [
                [split, *(_cell(record.metrics.get(split, {}).get(f"{m}@5"), with_n=True) for m in KEY_METRICS)]
                for split in ("all", "dev", "holdout")
            ],
        )
    )
    out += ["", "## Every k (all scenarios)", ""]
    out.append(
        _table(
            ["k", *KEY_METRICS.values()],
            [[str(k), *(_cell(record.metrics.get("all", {}).get(f"{m}@{k}")) for m in KEY_METRICS)] for k in K_VALUES],
        )
    )
    out += ["", "## Per scenario @5", ""]
    out.append(
        _table(
            ["scenario", "split", *KEY_METRICS.values()],
            [
                [
                    s.id,
                    s.split,
                    *(_cell(record.per_scenario.get(s.id, {}).get(f"{m}@5")) for m in KEY_METRICS),
                ]
                for s in record.scenarios
            ],
        )
    )
    out += ["", "## Claim-state conformance (E-only; GET memory `lifecycle`)", ""]
    out.append(
        _table(
            ["split", "conformance", "retirement recall", "false retirement"],
            [
                [
                    split,
                    *(
                        _cell(record.state_conformance.get(f"{split}:{m}"), with_n=True)
                        for m in ("conformance", "retirement_recall", "false_retirement")
                    ),
                ]
                for split in ("all", "dev", "holdout")
            ],
        )
    )
    out += ["", "## Consolidation calls", ""]
    out.append(
        _table(
            ["scenario", "calls", "lifecycle schema", "degradations", "dedup (merge/keep)", "observations by status"],
            [
                [
                    s.id,
                    str(len(s.calls)),
                    str(sum(c.lifecycle_schema for c in s.calls)),
                    str(sum(len(c.degradations) for c in s.calls)),
                    f"{sum(d.action == 'merge' for d in s.dedup)}/{sum(d.action == 'keep' for d in s.dedup)}",
                    ", ".join(f"{k}={v}" for k, v in sorted((s.state.observation_statuses if s.state else {}).items())),
                ]
                for s in record.scenarios
            ],
        )
    )
    degradations = [(s.id, c.step, d) for s in record.scenarios for c in s.calls for d in c.degradations]
    if degradations:
        out += ["", "Degradations (an action the payload could not resolve against what the call showed):", ""]
        out += [f"- `{sid}` step {step}: {text}" for sid, step, text in degradations]
    errors = [(s.id, e) for s in record.scenarios for e in s.errors]
    if errors:
        out += ["", "## Errors", ""]
        out += [f"- `{sid}`: {e}" for sid, e in errors]
    return "\n".join(out) + "\n"


def render_compare(records: Sequence[RunRecord], *, split: str = "all", k: int = 5) -> str:
    """One row per run, the key metrics at ``k`` on ``split``. Lower is better for the
    contamination / leakage columns, higher for the rest."""
    header = ["run", "arm", "encoding", "embedder", *(f"{h}@{k}" for h in KEY_METRICS.values())]
    rows = [
        [
            r.label,
            r.arm,
            r.encoding
            + (f" +{','.join(f'{a}={b}' for a, b in r.bank_config.items())}" if r.bank_config else "")
            + (f" types={','.join(r.recall_types)}" if r.recall_types else ""),
            "lexical" if r.embedder.startswith("lexical") else "local",
            *(_cell(r.metrics.get(split, {}).get(f"{m}@{k}")) for m in KEY_METRICS),
        ]
        for r in records
    ]
    lower = ", ".join(KEY_METRICS[m] for m in KEY_METRICS if m in _LOWER_IS_BETTER)
    return f"Split `{split}`, k={k}. Lower is better for: {lower}.\n\n" + _table(header, rows) + "\n"


@dataclass
class RowDiff:
    """Row-level comparison of two runs: are the ranked labels identical query by query?"""

    compared: int = 0
    identical: int = 0
    mismatches: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    max_score_delta: float = 0.0

    @property
    def deterministic(self) -> bool:
        return not self.mismatches and not self.missing


def _signature(row: QueryRow) -> list[tuple[str, tuple[str, ...], str, str]]:
    # Memory ids are minted per run, so a row is compared on what it says, not which row it is.
    return [(r.type, tuple(r.docs), r.cls, r.text) for r in row.results]


def diff_rows(a: RunRecord, b: RunRecord) -> RowDiff:
    result = RowDiff()
    b_rows = {(row.scenario, row.query_id): row for row in b.rows}
    for row in a.rows:
        key = (row.scenario, row.query_id)
        other = b_rows.pop(key, None)
        if other is None:
            result.missing.append(f"{key[0]}/{key[1]} only in {a.label}")
            continue
        result.compared += 1
        if _signature(row) == _signature(other):
            result.identical += 1
            for x, y in zip(row.results, other.results):
                if x.score is not None and y.score is not None:
                    result.max_score_delta = max(result.max_score_delta, abs(x.score - y.score))
        else:
            first = next(
                (i for i, (x, y) in enumerate(zip(_signature(row), _signature(other))) if x != y),
                min(len(row.results), len(other.results)),
            )
            result.mismatches.append(f"{key[0]}/{key[1]}: first difference at rank {first + 1}")
    result.missing += [f"{s}/{q} only in {b.label}" for s, q in b_rows]
    return result


def render_diff(a: RunRecord, b: RunRecord) -> str:
    diff = diff_rows(a, b)
    lines = [
        f"Row diff `{a.label}` vs `{b.label}`: {diff.identical}/{diff.compared} queries identical "
        f"(type, labels, class and text at every rank); max |Δ final score| on identical rows "
        f"{diff.max_score_delta:.2e}.",
    ]
    lines += [f"- {m}" for m in diff.mismatches + diff.missing]
    return "\n".join(lines) + "\n"
