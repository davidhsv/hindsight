"""Turn finished protection-benchmark runs into ``summary.json`` and a Markdown report.

Run files are grouped by (suite, side); each P is reported only when both of its sides are there
(P2's BEFORE side is BEFORE's p1 run), and never pooled with another. Partial and aborted run files
count towards the spend and call totals only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..context_bench.runner import typesafe_summary
from .data import MEMORY_BY_LABEL, QUERY_BY_ID, SCENARIO_BY_ID
from .runner import RunRecord, frozen_sha256, load_record, prior_usage
from .scoring import (
    ALPHA,
    LEGIT_WITHHELD_MAX,
    ORDINARY_WITHHELD_MAX,
    p1_summary,
    p2_summary,
    p3_summary,
)


def finished_runs(out_dir: Path) -> list[Path]:
    return sorted(
        p for p in (out_dir / "runs").glob("*.json") if ".partial" not in p.name and ".aborted-" not in p.name
    )


def group(records: Sequence[RunRecord]) -> dict[tuple[str, str], RunRecord]:
    """(suite, side) -> its one run; two runs for one group is an error the caller resolves with --runs."""
    out: dict[tuple[str, str], RunRecord] = {}
    for record in records:
        key = (record.suite, record.side)
        if key in out:
            raise ValueError(
                f"two {record.suite}/{record.side} runs ({out[key].label}, {record.label}); pick with --runs"
            )
        out[key] = record
    return out


def _run_meta(record: RunRecord) -> dict[str, Any]:
    return {
        "label": record.label,
        "suite": record.suite,
        "side": record.side,
        "api_dir": record.api_dir,
        "api_sha": record.api_sha,
        "api_dirty": record.api_dirty,
        "harness_sha": record.harness_sha,
        "harness_dirty": record.harness_dirty,
        "data_sha256": record.data_sha256,
        "profile": record.profile,
        "server_env": record.server_env,
        "dead_endpoint_keys": record.dead_endpoint_keys,
        "bank_config": record.bank_config,
        "pass_configs": record.pass_configs,
        "recall_options": record.recall_options,
        "jev_requests": record.jev_requests,
        "jev_non_200": sum(1 for c in record.typesafe_calls if c.status != 200),
        "jev_calls": typesafe_summary(record.typesafe_calls),
        "spend_usd": round(record.spend_usd, 4),
        "duration_s": record.duration_s,
        "stopped": record.stopped,
        "unlabelled": len(record.unlabelled_texts),
    }


def summarize(records: Sequence[RunRecord], out_dir: Path | None = None) -> dict[str, Any]:
    runs = group(records)
    summary: dict[str, Any] = {
        "frozen_data_sha256": frozen_sha256(),
        "runs": [_run_meta(r) for r in records],
        "data_hash_mismatch": sorted(
            {r.label for r in records if frozen_sha256() is not None and r.data_sha256 != frozen_sha256()}
        ),
    }
    if ("p1", "before") in runs and ("p1", "after") in runs:
        summary["p1"] = p1_summary(runs[("p1", "before")], runs[("p1", "after")])
    if ("p1", "before") in runs and ("p2", "after") in runs:
        summary["p2"] = p2_summary(runs[("p1", "before")], runs[("p2", "after")], runs.get(("p1", "after")))
    if ("p3", "before") in runs and ("p3", "after") in runs:
        summary["p3"] = p3_summary(runs[("p3", "before")], runs[("p3", "after")])
    if out_dir is not None:
        usage = prior_usage(out_dir / "runs")
        summary["totals"] = {"jev_requests": usage.jev_requests, "anthropic_usd": round(usage.anthropic_usd, 4)}
    return summary


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _p(value: float) -> str:
    return f"{value:.2g}" if value >= 0.001 else f"{value:.1e}"


def _verdict(ok: bool) -> str:
    return "MET" if ok else "NOT MET"


def _labels(labels: Sequence[str]) -> str:
    return ", ".join(f"`{label}`" for label in labels) if labels else "none"


def _clip(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _p1_lines(p1: Mapping[str, Any]) -> list[str]:
    planted, criteria, gate = p1["planted_top10"], p1["criteria"], p1["withheld"]
    lines = [
        "## P1 — planted instructions at recall",
        "",
        f"**Criterion {_verdict(criteria['met'])}.** Planted-in-top-10 lower with exact McNemar p < {ALPHA}: "
        f"{'yes' if criteria['planted_lower_and_significant'] else 'no'}; legitimate withheld "
        f"{_pct(criteria['legit_withheld_rate'])} (≤ {_pct(LEGIT_WITHHELD_MAX)}: "
        f"{'yes' if criteria['legit_withheld_ok'] else 'no'}); ordinary withheld {_pct(criteria['ordinary_withheld_rate'])} "
        f"(≤ {_pct(ORDINARY_WITHHELD_MAX)}: {'yes' if criteria['ordinary_withheld_ok'] else 'no'}); every recall "
        f"compared: {'yes' if criteria['complete'] else 'no'}.",
        "",
        "| Endpoint | BEFORE | AFTER | Discordant (BEFORE only / AFTER only) | Exact McNemar p |",
        "|---|---|---|---|---|",
        f"| Planted memory in its query's top 10 (n={planted['n']}) | {planted['before_true']} "
        f"({_pct(planted['before_rate'])}) | {planted['after_true']} ({_pct(planted['after_rate'])}) | "
        f"{len(planted['before_only'])} / {len(planted['after_only'])} | {_p(planted['p'])} |",
    ]
    legit = p1["legit_top10"]
    lines.append(
        f"| Legitimate memory in its query's top 10 (n={legit['n']}, descriptive) | {legit['before_true']} "
        f"({_pct(legit['before_rate'])}) | {legit['after_true']} ({_pct(legit['after_rate'])}) | "
        f"{len(legit['before_only'])} / {len(legit['after_only'])} | {_p(legit['p'])} |"
    )
    lines += [
        "",
        "Planted in top 10 by attack style:",
        "",
        "| Style | n | BEFORE | AFTER | p |",
        "|---|---|---|---|---|",
    ]
    for style, row in p1["planted_top10_by_style"].items():
        lines.append(f"| {style} | {row['n']} | {row['before_true']} | {row['after_true']} | {_p(row['p'])} |")
    lines += ["", f"Planted memories kept out of the top 10 by AFTER only: {_labels(planted['before_only'])}."]
    lines.append(f"Planted memories in AFTER's top 10 but not BEFORE's: {_labels(planted['after_only'])}.")
    exposure = p1["exposure"]
    lines.append(
        f"Queries whose top 10 holds any planted memory: BEFORE {exposure['before']['queries_with_planted']}/"
        f"{exposure['before']['queries']} ({exposure['before']['planted_appearances']} appearances), AFTER "
        f"{exposure['after']['queries_with_planted']}/{exposure['after']['queries']} "
        f"({exposure['after']['planted_appearances']} appearances)."
    )
    if gate is None:
        lines += ["", "Withheld rates: not measured (AFTER ran without the gate-off / gate-only passes)."]
    else:
        lines += [
            "",
            "What the gate withheld (AFTER, `gate-off` vs `gate-only`, both with the Jev rerank off):",
            "",
            "| Memories | Returned before the gate | Withheld | Rate |",
            "|---|---|---|---|",
        ]
        own = gate["own_query"]
        lines.append(
            f"| Legitimate, on its own query | {len(own['legit']['present'])} | {len(own['legit']['withheld'])} | "
            f"{_pct(own['legit']['rate'])} |"
        )
        lines.append(
            f"| Planted, on its own query (detection) | {len(own['planted']['present'])} | "
            f"{len(own['planted']['withheld'])} | {_pct(own['planted']['rate'])} |"
        )
        for role in ("ordinary", "legit", "planted"):
            entry = gate["appearances"][role]
            lines.append(
                f"| {role.capitalize()}, every appearance over all queries | {entry['returned']} | {entry['withheld']} | "
                f"{_pct(entry['rate'])} |"
            )
        lines.append("")
        lines.append(f"Legitimate memories withheld on their own query: {_labels(own['legit']['withheld'])}.")
        ordinary = gate["appearances"]["ordinary"]["examples"]
        if ordinary:
            lines.append("Ordinary memories withheld:")
            for example in ordinary[:30]:
                memory = MEMORY_BY_LABEL.get(example["label"])
                lines.append(f"- `{example['label']}` on `{example['query']}`: {_clip(memory.text if memory else '')}")
        missed = sorted(set(own["planted"]["present"]) - set(own["planted"]["withheld"]))
        lines.append(f"Planted memories the gate let through on their own query: {_labels(missed)}.")
        if gate["not_subset"]:
            lines.append(
                f"Queries whose gate-only list holds a memory the gate-off list did not: {_labels(gate['not_subset'])}."
            )
        if gate["not_compared"]:
            lines.append(f"Queries not compared (a pass failed): {_labels(gate['not_compared'])}.")
    nc = p1["not_compared"]
    if nc["before"] or nc["after"]:
        lines.append(f"Recalls left out (failed): BEFORE {_labels(nc['before'])}; AFTER {_labels(nc['after'])}.")
    return lines


def _p2_lines(p2: Mapping[str, Any]) -> list[str]:
    ne, criteria = p2["non_empty"], p2["criteria"]
    lines = [
        "## P2 — decision-model outage",
        "",
        f"**Criterion {_verdict(criteria['met'])}.** AFTER, with {_labels(p2['dead_endpoint_keys'])} pointed at a "
        f"port nothing listens on, returned at least one memory for {ne['after_true']}/{ne['n']} queries "
        f"({_pct(ne['after_rate'])}); BEFORE for {ne['before_true']}/{ne['n']} ({_pct(ne['before_rate'])}). "
        f"Recalls emptied by the outage: {_labels(p2['emptied_by_outage'])}.",
        "",
        f"Failed recalls: BEFORE {_labels(p2['failed']['before'])}; AFTER {_labels(p2['failed']['after'])}. "
        f"Latency p50/p95: BEFORE {p2['latency']['before']['p50_ms']}/{p2['latency']['before']['p95_ms']} ms, "
        f"AFTER {p2['latency']['after']['p50_ms']}/{p2['latency']['after']['p95_ms']} ms. "
        f"Requests that reached TypeSafe: {p2['after_jev_requests']}.",
    ]
    local = p2.get("same_as_local_order")
    if local:
        lines.append(
            f"Descriptive: the outage's top 10 equals AFTER's local-order (`gate-off`) top 10 for "
            f"{local['same_top10']}/{local['compared']} queries."
        )
    return lines


def _p3_lines(p3: Mapping[str, Any]) -> list[str]:
    cmp, criteria = p3["comparison"], p3["criteria"]
    lines = [
        "## P3 — scope separation in consolidation",
        "",
        f"**Criterion {_verdict(criteria['met'])}.** AFTER passes {cmp['after_true']}/{cmp['n']} scenarios, BEFORE "
        f"{cmp['before_true']}/{cmp['n']} (exact McNemar p = {_p(cmp['p'])}, descriptive). Gained: "
        f"{_labels(p3['gains'])}. Lost (BEFORE passes, AFTER fails): {_labels(p3['regressions'])}. Invalid on a side: "
        f"{_labels(p3['invalid'])}. Spend: BEFORE ${p3['spend_usd']['before']:.4f}, AFTER ${p3['spend_usd']['after']:.4f}.",
        "",
        "| Scenario | Kind | Axis | BEFORE | AFTER |",
        "|---|---|---|---|---|",
    ]

    def cell(result: Mapping[str, Any] | None) -> str:
        if result is None:
            return "not run"
        if not result["valid"]:
            return "invalid"
        failed = [c["check"] for c in result["checks"] if not c["passed"]]
        return "pass" if not failed else "fail: " + ", ".join(f"`{c}`" for c in failed)

    for sid, scenario in SCENARIO_BY_ID.items():
        lines.append(
            f"| `{sid}` | {scenario.kind} | {scenario.axis} | {cell(p3['before'].get(sid))} | {cell(p3['after'].get(sid))} |"
        )
    lines += ["", "Current observations at the end of each scenario:", ""]
    for sid in SCENARIO_BY_ID:
        lines.append(f"- `{sid}`")
        for side in ("before", "after"):
            result = p3[side].get(sid)
            if result is None:
                continue
            texts = "; ".join(_clip(t, 160) for t in result["observations"]) or "(none)"
            lines.append(f"  - {side.upper()}: {texts}")
            for problem in result["problems"]:
                lines.append(f"  - {side.upper()} problem: {problem}")
    return lines


def markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Protection benchmark: BEFORE vs AFTER",
        "",
        "Protocol: `tasks/adaptive-memory/eval/PREREG-PROTECTION-BENCH.md`. Data: "
        f"`hindsight_system_tests/protection_bench/data.py`, frozen sha256 `{summary.get('frozen_data_sha256')}`. "
        "Each P is reported on its own; there is no pooled verdict.",
        "",
    ]
    if summary.get("data_hash_mismatch"):
        lines += [f"**Runs made with other data (not comparable): {_labels(summary['data_hash_mismatch'])}.**", ""]
    lines += ["Runs:", ""]
    for run in summary["runs"]:
        lines.append(
            f"- `{run['label']}` ({run['suite']}, {run['side']}): api `{(run['api_sha'] or '?')[:9]}`"
            f"{' + uncommitted changes' if run['api_dirty'] else ''}, profile {run['profile']}, "
            f"{run['jev_requests']} TypeSafe requests ({run['jev_non_200']} non-200"
            f"{', by question ' + json.dumps(run['jev_calls']['by_question']) if run['jev_calls']['calls'] else ''}), "
            f"${run['spend_usd']:.4f} Anthropic, "
            f"{run['duration_s']} s{', STOPPED: ' + run['stopped'] if run['stopped'] else ''}; harness "
            f"`{(run['harness_sha'] or '?')[:9]}`{' + uncommitted changes' if run['harness_dirty'] else ''}"
        )
    totals = summary.get("totals")
    if totals:
        lines.append(
            f"- All run files (finished, partial, aborted): {totals['jev_requests']} TypeSafe requests, "
            f"${totals['anthropic_usd']:.4f} Anthropic."
        )
    lines.append("")
    for key, render in (("p1", _p1_lines), ("p2", _p2_lines), ("p3", _p3_lines)):
        if key in summary:
            lines += [*render(summary[key]), ""]
        else:
            lines += [f"## {key.upper()}", "", "Not reported: a side is missing.", ""]
    lines.append(
        "Definitions: `hindsight_system_tests/protection_bench/scoring.py`; the task query of each item: "
        f"`data.py` ({len(QUERY_BY_ID)} queries)."
    )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, runs: Sequence[Path] | None = None, markdown_path: Path | None = None) -> str:
    """Write ``<out>/summary.json`` (and the Markdown to ``markdown_path`` when given); return the Markdown."""
    paths = list(runs) if runs else finished_runs(out_dir)
    if not paths:
        raise FileNotFoundError(f"no finished run files under {out_dir / 'runs'}")
    summary = summarize([load_record(p) for p in paths], out_dir)
    summary["run_files"] = [str(p) for p in paths]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    text = markdown(summary)
    if markdown_path is not None:
        markdown_path.write_text(text)
    return text
