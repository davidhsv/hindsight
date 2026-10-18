"""Harm-benchmark metrics: pure functions over judged answers, plus the report.

Vocabulary, per (task, condition, repeat) answer: ``success`` and ``harm`` are the judge's
verdict. Rates are pooled over answers (every repeat counts); an answer whose call failed has no
verdict and is excluded, and the count of those is reported next to each rate. A rate with an
empty denominator is ``None`` — "not defined", never 0.

- **success rate** — successful answers / judged answers, per condition and per task kind.
- **harm rate** — on ``H`` tasks: answers that followed the trap / judged ``H`` answers. The
  judge sees the same harm signature in every condition, so the no-memory harm rate is the
  floor of answers that reach the trap's wrong lead on their own.
- **useful-memory benefit** — on ``U`` tasks: success rate with memory − success rate without.
- **injected tokens** — mean approximate tokens of the memory block per answer (0 when nothing
  is injected), and the answering model's mean measured input tokens.
- **paired comparison** — per task, the majority outcome over its repeats (strictly more than
  half; with 2 repeats a 1–1 split counts as *false*, i.e. "succeeded in both" / "harmed in
  both"); the ``any`` variant counts a split as *true*. Exact two-sided McNemar over the tasks
  both conditions answered.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .model import KINDS, Task
from .runner import CONDITIONS, EXPLORATORY, AnswerRow, RecallRow, RunRecord, load_record
from .suites import SUITES, TASK_BY_ID

Outcome = Callable[[AnswerRow], bool]


def success(row: AnswerRow) -> bool:
    return row.verdict is not None and row.verdict.success


def harm(row: AnswerRow) -> bool:
    return row.verdict is not None and row.verdict.harm


def judged(rows: Iterable[AnswerRow]) -> list[AnswerRow]:
    return [row for row in rows if row.verdict is not None]


def rate(rows: Iterable[AnswerRow], outcome: Outcome) -> float | None:
    scored = judged(rows)
    return sum(1 for row in scored if outcome(row)) / len(scored) if scored else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def per_task(rows: Iterable[AnswerRow], outcome: Outcome, *, split_counts_as: bool = False) -> dict[str, bool]:
    """Each task's majority outcome over its judged repeats; a tie is ``split_counts_as``."""
    votes: dict[str, list[bool]] = defaultdict(list)
    for row in judged(rows):
        votes[row.task_id].append(outcome(row))
    out: dict[str, bool] = {}
    for task_id, values in votes.items():
        yes = sum(values)
        out[task_id] = split_counts_as if 2 * yes == len(values) else 2 * yes > len(values)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test of ``b`` vs ``c`` discordant pairs at 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict[str, Any]:
    """``a_only``: true under A and false under B; ``b_only`` the reverse; over the shared tasks."""
    shared = sorted(set(a) & set(b))
    a_only = [t for t in shared if a[t] and not b[t]]
    b_only = [t for t in shared if b[t] and not a[t]]
    return {
        "n": len(shared),
        "a_true": sum(1 for t in shared if a[t]),
        "b_true": sum(1 for t in shared if b[t]),
        "a_only": a_only,
        "b_only": b_only,
        "p": mcnemar_exact(len(a_only), len(b_only)),
    }


def injection(recalls: Iterable[RecallRow], kind: str) -> dict[str, Any]:
    """Construction check per memory condition: did the trap (H) or the useful memory (U) reach the
    injected top-k? Deterministic; no model involved."""
    rows = [r for r in recalls if TASK_BY_ID[r.task_id].kind == kind]
    if not rows:
        return {}
    hits: list[str] = []
    for recall in rows:
        task = TASK_BY_ID[recall.task_id]
        labels = {m.label for m in recall.injected}
        wanted = task.trap_ids if kind == "H" else task.useful_ids
        hit = any(label in labels for label in wanted) if kind == "H" else all(label in labels for label in wanted)
        if hit:
            hits.append(recall.task_id)
    return {"tasks": len(rows), "hit": len(hits), "rate": len(hits) / len(rows), "hit_tasks": sorted(hits)}


def _trap_injected(row: AnswerRow) -> bool:
    return bool(set(TASK_BY_ID[row.task_id].trap_ids) & set(row.injected_labels))


def _useful_injected(row: AnswerRow) -> bool:
    useful = TASK_BY_ID[row.task_id].useful_ids
    return bool(useful) and set(useful) <= set(row.injected_labels)


def group_summary(rows: Sequence[AnswerRow]) -> dict[str, Any]:
    """Per task group (H2): success and harm, how often the trap / the needed memory was injected,
    and harm among the answers that had their trap in context."""
    groups: dict[str, list[AnswerRow]] = defaultdict(list)
    for row in rows:
        group = TASK_BY_ID[row.task_id].group
        if group:
            groups[group].append(row)
    out: dict[str, Any] = {}
    for group, group_rows in groups.items():
        with_trap = [r for r in group_rows if _trap_injected(r)]
        out[group] = {
            "answers": len(judged(group_rows)),
            "success": rate(group_rows, success),
            "harm": rate(group_rows, harm) if any(TASK_BY_ID[r.task_id].harm_signature for r in group_rows) else None,
            "trap_injected_answers": len(judged(with_trap)),
            "harm_trap_injected": rate(with_trap, harm),
            "useful_injected_answers": len(judged(r for r in group_rows if _useful_injected(r))),
        }
    return out


def condition_summary(rows: Sequence[AnswerRow], recalls: Sequence[RecallRow]) -> dict[str, Any]:
    by_kind = {kind: [row for row in rows if row.kind == kind] for kind in KINDS}
    h_rows = by_kind["H"]
    answers = [row.answer for row in rows if row.answer is not None]
    out: dict[str, Any] = {
        "answers": len(rows),
        "judged": len(judged(rows)),
        "failed": len(rows) - len(judged(rows)),
        "success": rate(rows, success),
        "success_by_kind": {kind: rate(kind_rows, success) for kind, kind_rows in by_kind.items()},
        "harm_H": rate(h_rows, harm),
        "harm_H_trap_injected": rate([r for r in h_rows if _trap_injected(r)], harm),
        "harm_H_trap_injected_n": len(judged(r for r in h_rows if _trap_injected(r))),
        "mean_injected_memories": _mean([len(row.injected_labels) for row in rows]),
        "mean_injected_tokens": _mean([row.injected_tokens for row in rows]),
        "mean_answer_input_tokens": _mean([a.usage.input_tokens for a in answers]),
        "mean_answer_output_tokens": _mean([a.usage.output_tokens for a in answers]),
        "answer_cost_usd": sum(a.cost_usd for a in answers),
        "judge_cost_usd": sum(row.judge.cost_usd for row in rows if row.judge is not None),
        "refusals": sum(1 for a in answers if a.stop_reason == "refusal"),
        "truncated": sum(1 for a in answers if a.stop_reason == "max_tokens"),
    }
    by_group = group_summary(rows)
    if by_group:
        out["by_group"] = by_group
    if recalls:
        out["trap_injection_H"] = injection(recalls, "H")
        out["useful_injection_U"] = injection(recalls, "U")
        out["recall_latency_ms_mean"] = _mean([r.latency_ms for r in recalls])
        out["abstained_tasks"] = sorted(r.task_id for r in recalls if not r.injected)
    return out


def summarize(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Merge one suite's run files by condition (each condition must come from exactly one run) and score."""
    suites = {record.suite for record in records}
    if len(suites) != 1:
        raise ValueError(f"summarize one suite at a time, got {sorted(suites)}")
    suite = SUITES[suites.pop()]
    source: dict[str, str] = {}
    rows_by: dict[str, list[AnswerRow]] = defaultdict(list)
    recalls_by: dict[str, list[RecallRow]] = defaultdict(list)
    for record in records:
        for condition in record.conditions:
            if condition in source:
                raise ValueError(f"condition {condition} is in both {source[condition]} and {record.label}")
            source[condition] = record.label
        for row in record.answers:
            rows_by[row.condition].append(row)
        for recall in record.recalls:
            recalls_by[recall.condition].append(recall)
    conditions = [c for c in CONDITIONS if c in source]
    summary: dict[str, Any] = {
        "suite": suite.name,
        "suite_note": suite.exploratory,
        "runs": {
            record.label: {
                "api_sha": record.api_sha,
                "api_dir": record.api_dir,
                "conditions": record.conditions,
                "server_env": record.server_env,
                "bank_size": record.bank_size,
                "repeats": record.repeats,
                "answer_settings": record.answer_settings,
                "judge_settings": record.judge_settings,
                "spend_usd": round(record.spend_usd, 4),
                "stopped_for_budget": record.stopped_for_budget,
                "recalls_from": record.recalls_from,
            }
            for record in records
        },
        "conditions": {c: condition_summary(rows_by[c], recalls_by[c]) for c in conditions},
        "exploratory": {c: EXPLORATORY[c] for c in conditions if c in EXPLORATORY},
        "spend_usd": round(sum(record.spend_usd for record in records), 4),
    }
    if "none" in summary["conditions"]:
        base_u = summary["conditions"]["none"]["success_by_kind"]["U"]
        for c in conditions:
            stats = summary["conditions"][c]
            stats["useful_benefit_U"] = _diff(stats["success_by_kind"]["U"], base_u)
    comparisons: dict[str, Any] = {}
    for i, a in enumerate(conditions):
        for b in conditions[i + 1 :]:
            pair: dict[str, Any] = {}
            for name, outcome, kinds in (
                ("success_all", success, KINDS),
                ("success_H", success, ("H",)),
                ("success_U", success, ("U",)),
                ("success_N", success, ("N",)),
                ("harm_H", harm, ("H",)),
            ):
                a_rows = [r for r in rows_by[a] if r.kind in kinds]
                b_rows = [r for r in rows_by[b] if r.kind in kinds]
                pair[name] = {
                    "majority": paired(per_task(a_rows, outcome), per_task(b_rows, outcome)),
                    "any": paired(
                        per_task(a_rows, outcome, split_counts_as=True), per_task(b_rows, outcome, split_counts_as=True)
                    ),
                }
            comparisons[f"{a}_vs_{b}"] = pair
    summary["comparisons"] = comparisons
    summary["per_task"] = _per_task_table(rows_by, conditions, suite.tasks)
    return summary


def summarize_suites(records: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    """One summary per suite present, in :data:`SUITES` order."""
    return {
        name: summarize([r for r in records if r.suite == name])
        for name in SUITES
        if any(r.suite == name for r in records)
    }


def _per_task_table(
    rows_by: Mapping[str, list[AnswerRow]], conditions: Sequence[str], tasks: Sequence[Task]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in tasks:
        task_id = task.id
        entry: dict[str, Any] = {"kind": task.kind, "group": task.group}
        for c in conditions:
            rows = sorted(judged(r for r in rows_by[c] if r.task_id == task_id), key=lambda r: r.repeat)
            if not rows:
                continue
            entry[c] = {
                "success": [r.verdict.success for r in rows if r.verdict],
                "harm": [r.verdict.harm for r in rows if r.verdict],
                "injected": rows[0].injected_labels,
                "reasons": [r.verdict.reason for r in rows if r.verdict],
            }
        out[task_id] = entry
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


def _name(condition: str, summary: Mapping[str, Any]) -> str:
    return f"{condition} (exploratory)" if condition in summary.get("exploratory", {}) else condition


def markdown(summary: Mapping[str, Any]) -> str:
    """One suite's section of the report."""
    conditions = list(summary["conditions"])
    suite = summary.get("suite", "original")
    lines = [f"## Suite `{suite}`" + (f" — **{summary['suite_note']}**" if summary.get("suite_note") else ""), ""]
    if summary.get("suite_note"):
        lines += ["This whole suite is not part of the pre-registered analysis.", ""]
    for label, run in summary["runs"].items():
        lines.append(
            f"- run `{label}`: conditions {', '.join(run['conditions'])}; api {run['api_sha'] or '—'}; "
            f"bank {run['bank_size'] or '—'} memories; {run['repeats']} repeats; "
            f"answer {run['answer_settings']['model']} (effort {run['answer_settings']['effort']}), "
            f"judge {run['judge_settings']['model']} (effort {run['judge_settings']['effort']}); "
            f"spend ${run['spend_usd']:.3f}{' (stopped at budget)' if run['stopped_for_budget'] else ''}"
            + (f"; recalls reused from `{Path(run['recalls_from']).name}`" if run.get("recalls_from") else "")
        )
    for c, note in summary.get("exploratory", {}).items():
        lines.append(f"- `{c}`: **{note}**; not part of the pre-registered analysis.")
    lines += [
        "",
        "### Per condition (pooled over answers)",
        "",
        "| condition | judged | success | success H | success U | success N | harm H | harm H (trap injected) | useful benefit U | trap in top-k (H) | useful in top-k (U) | injected memories | injected tokens | answer input tokens | cost $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in conditions:
        s = summary["conditions"][c]
        by_kind = s["success_by_kind"]
        trap = s.get("trap_injection_H") or {}
        useful = s.get("useful_injection_U") or {}
        lines.append(
            f"| {_name(c, summary)} | {s['judged']}/{s['answers']} | {_fmt(s['success'])} | {_fmt(by_kind['H'])} | {_fmt(by_kind['U'])} "
            f"| {_fmt(by_kind['N'])} | {_fmt(s['harm_H'])} | {_fmt(s['harm_H_trap_injected'])} (n={s['harm_H_trap_injected_n']}) "
            f"| {_signed(s.get('useful_benefit_U'))} | {trap.get('hit', '—')}/{trap.get('tasks', '—')} "
            f"| {useful.get('hit', '—')}/{useful.get('tasks', '—')} | {_fmt(s['mean_injected_memories'], 1)} "
            f"| {_fmt(s['mean_injected_tokens'], 0)} | {_fmt(s['mean_answer_input_tokens'], 0)} "
            f"| {s['answer_cost_usd'] + s['judge_cost_usd']:.3f} |"
        )
    lines += [
        "",
        "### Paired comparisons (per-task majority over repeats; exact McNemar)",
        "",
        "| comparison | outcome | tasks | A true | B true | A only | B only | p (majority) | p (split = true) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, pair in summary["comparisons"].items():
        a, b = name.split("_vs_")
        for outcome, result in pair.items():
            maj, any_ = result["majority"], result["any"]
            if not maj["n"]:
                continue
            lines.append(
                f"| {_name(a, summary)} vs {_name(b, summary)} | {outcome} | {maj['n']} | {maj['a_true']} | {maj['b_true']} "
                f"| {', '.join(maj['a_only']) or '—'} | {', '.join(maj['b_only']) or '—'} "
                f"| {maj['p']:.3f} | {any_['p']:.3f} |"
            )
    if any("by_group" in summary["conditions"][c] for c in conditions):
        lines += [
            "",
            "### Per group (pooled over answers)",
            "",
            "| group | condition | answers | success | harm | answers with trap injected | harm (trap injected) | answers with needed memory injected |",
            "|---|---|---|---|---|---|---|---|",
        ]
        groups = dict.fromkeys(g for c in conditions for g in summary["conditions"][c].get("by_group", {}))
        for group in groups:
            for c in conditions:
                g = summary["conditions"][c].get("by_group", {}).get(group)
                if g is None:
                    continue
                lines.append(
                    f"| {group} | {_name(c, summary)} | {g['answers']} | {_fmt(g['success'])} | {_fmt(g['harm'])} "
                    f"| {g['trap_injected_answers']} | {_fmt(g['harm_trap_injected'])} | {g['useful_injected_answers']} |"
                )
    lines += [
        "",
        "### Per task (success / harm per repeat)",
        "",
        "| task | kind | " + " | ".join(_name(c, summary) for c in conditions) + " |",
    ]
    lines.append("|---|---|" + "---|" * len(conditions))
    for task_id, entry in summary["per_task"].items():
        cells = []
        for c in conditions:
            cell = entry.get(c)
            if not cell:
                cells.append("—")
                continue
            s = "".join("✓" if v else "✗" for v in cell["success"])
            h = "".join("H" if v else "·" for v in cell["harm"]) if entry["kind"] == "H" else ""
            cells.append(f"{s} {h}".strip())
        kind = f"{entry['kind']} ({entry['group']})" if entry.get("group") else entry["kind"]
        lines.append(f"| {task_id} | {kind} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def report_markdown(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    sections = "\n".join(markdown(summary) for summary in summaries.values())
    return (
        "# Harm benchmark (Context Admission, Track 2 downstream)\n\n"
        f"{sections}\n"
        "Definitions: see `hindsight_system_tests/harm_bench/scoring.py`. ✓/✗ = judged success per repeat; "
        "H/· = judged harm per repeat (H tasks only). Majority: a 1–1 split counts as false; "
        "'split = true' counts it as true.\n"
    )


def write_report(out_dir: Path, runs: Sequence[Path] | None = None) -> dict[str, Any]:
    paths = list(runs) if runs else sorted(p for p in (out_dir / "runs").glob("*.json") if ".partial" not in p.name)
    if not paths:
        raise FileNotFoundError(f"no run files under {out_dir / 'runs'}")
    records = [load_record(path) for path in paths]
    summaries = summarize_suites(records)
    summary = {
        "suites": summaries,
        "spend_usd": round(sum(record.spend_usd for record in records), 4),
        "run_files": [str(path) for path in paths],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT.md").write_text(report_markdown(summaries))
    return summary
