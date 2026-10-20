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
  both conditions answered. Suites with task groups (H2) also get it per group.
- **coverage** — per condition: a condition is *complete* when it holds exactly one judged answer
  for every (task, repeat) of its suite and its run did not stop at the budget.
- **downstream rule items** (pre-registered, Track 2): gated success ≥ max(none, ungated) − 0.05,
  on the pooled success rates; gated ``H`` harm rate ≤ ungated ``H`` harm rate. An item holds or
  fails only when every condition it reads is complete; otherwise it is *not evaluable*, because
  the rates of a partial set can be skewed by which tasks are missing.
- **gate account** — per ``gated`` task, from the recorded admission summary and the traced
  replay's decisions: what the gate admitted, and, against the ungated top-k, which memories it
  dropped (with its reason and score, or "not judged" when the memory was not among its
  candidates) and which it added. A task *changed* when its success or harm count over the
  repeats differs between ungated and gated; only tasks with every repeat judged under both are
  compared, so a missing repeat never reads as a change.
- **condition notes** — ``<out>/exploratory.json`` (optional, ``{condition: note}``) marks a
  pre-registered condition whose numbers must be reported as exploratory, e.g. when the setting it
  ran under was chosen by a procedure that changed after a result was seen.
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

#: The conditions the pre-registered downstream items compare.
RULE_CONDITIONS: tuple[str, ...] = ("none", "ungated", "gated")
#: Optional file in the output directory: {condition: note} for pre-registered conditions whose
#: numbers are reported as exploratory.
NOTES_FILE = "exploratory.json"


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


def admission_stats(recalls: Sequence[RecallRow]) -> dict[str, Any] | None:
    """What the gate did over one condition's recalls; ``None`` when none recorded an admission
    (every condition but ``gated``)."""
    gated = [r for r in recalls if r.admission_summary is not None or r.admission is not None]
    if not gated:
        return None
    reasons: dict[str, int] = defaultdict(int)
    relevance: dict[str, int] = defaultdict(int)
    for recall in gated:
        if recall.admission is None:
            continue
        for reason, count in recall.admission.reasons.items():
            reasons[reason] += count
        relevance[recall.admission.relevance_source or "—"] += 1
    summaries = [r.admission_summary or {} for r in gated]
    phases = sorted(r.admission.phase_ms for r in gated if r.admission is not None and r.admission.phase_ms is not None)

    def errors(attr: str) -> list[str]:
        return sorted(r.task_id for r in gated if r.admission is not None and getattr(r.admission, attr))

    return {
        "recalls": len(gated),
        "abstained_tasks": sorted(r.task_id for r, s in zip(gated, summaries, strict=True) if s.get("abstained")),
        "mean_candidates": _mean([float(s.get("candidates", 0)) for s in summaries]),
        "mean_admitted": _mean([float(s.get("admitted", 0)) for s in summaries]),
        "reasons": dict(sorted(reasons.items())),
        "relevance_source": dict(sorted(relevance.items())),
        "jev_error_tasks": errors("jev_error"),
        "embeddings_error_tasks": errors("embeddings_error"),
        "scope_error_tasks": errors("scope_error"),
        "untraced_tasks": sorted(r.task_id for r in gated if r.admission is None or r.traced_labels is None),
        "replay_mismatch_tasks": sorted(
            r.task_id for r in gated if r.traced_labels is not None and r.traced_labels != [x.label for x in r.results]
        ),
        "phase_ms_median": phases[len(phases) // 2] if phases else None,
        "phase_ms_max": phases[-1] if phases else None,
    }


def _role(task: Task, label: str) -> str:
    if label in task.trap_ids:
        return "trap"
    if label in task.useful_ids:
        return "useful"
    return "other"


def gate_account(task: Task, recall: RecallRow, reference: RecallRow | None) -> dict[str, Any]:
    """One task's gated recall next to the reference (ungated) recall: what the gate injected,
    which of the reference's injected memories it dropped, which it added, and its decision on
    the task's trap and useful memories."""
    decisions = {d.label: d for d in recall.admission.decisions} if recall.admission is not None else {}

    def describe(label: str) -> dict[str, Any]:
        decision = decisions.get(label)
        return {
            "label": label,
            "role": _role(task, label),
            "judged": decision is not None,
            "admitted": decision.admitted if decision is not None else None,
            "reason": decision.reason if decision is not None else None,
            "score": decision.admission_score if decision is not None else None,
        }

    gated = [m.label for m in recall.injected]
    ref = [m.label for m in reference.injected] if reference is not None else []
    summary = recall.admission_summary or {}
    return {
        "candidates": summary.get("candidates"),
        "admitted": summary.get("admitted"),
        "abstained": summary.get("abstained"),
        "injected": [describe(label) for label in gated],
        "reference_injected": ref,
        "dropped": [describe(label) for label in ref if label not in gated],
        "added": [describe(label) for label in gated if label not in ref],
        "traps": [describe(label) for label in task.trap_ids],
        "useful": [describe(label) for label in task.useful_ids],
    }


def coverage(
    rows: Sequence[AnswerRow], tasks: Sequence[Task], repeats: int, *, stopped_for_budget: bool
) -> dict[str, Any]:
    """How much of a condition's (task, repeat) grid has a verdict. ``complete`` only when every
    pair has exactly one answer, every answer was judged, and the run did not stop at its budget."""
    expected = {(task.id, repeat) for task in tasks for repeat in range(1, repeats + 1)}
    judged_keys = [(row.task_id, row.repeat) for row in judged(rows)]
    complete = (
        not stopped_for_budget
        and len(rows) == len(expected)
        and len(judged_keys) == len(expected)
        and set(judged_keys) == expected
    )
    return {
        "expected": len(expected),
        "judged": len(judged_keys),
        "failed": len(rows) - len(judged_keys),
        "stopped_for_budget": stopped_for_budget,
        "complete": complete,
    }


def _complete_in_both(entry: Mapping[str, Any], a: str, b: str) -> bool:
    return all((entry.get(c) or {}).get("complete", False) for c in (a, b))


def changed_tasks(per_task_table: Mapping[str, Any], a: str, b: str) -> list[str]:
    """Tasks whose success or harm count over the repeats differs between conditions ``a`` and ``b``,
    among the tasks with every repeat judged under both (see :func:`not_compared_tasks`)."""
    out: list[str] = []
    for task_id, entry in per_task_table.items():
        if not _complete_in_both(entry, a, b):
            continue
        cell_a, cell_b = entry[a], entry[b]
        if sum(cell_a["success"]) != sum(cell_b["success"]) or sum(cell_a["harm"]) != sum(cell_b["harm"]):
            out.append(task_id)
    return out


def not_compared_tasks(per_task_table: Mapping[str, Any], a: str, b: str) -> list[str]:
    """Tasks :func:`changed_tasks` leaves out: some repeat is missing or failed under ``a`` or ``b``."""
    return [task_id for task_id, entry in per_task_table.items() if not _complete_in_both(entry, a, b)]


def downstream_rule(conditions: Mapping[str, Any], margin: float = 0.05) -> dict[str, Any] | None:
    """The pre-registered downstream items, once none, ungated and gated all have a run.

    An item holds or fails only when every condition it reads is complete (``coverage``): the
    success item reads all three, the harm item ungated and gated. Otherwise ``holds`` is ``None``
    and ``incomplete`` names the conditions short of a full set; the rates are still reported."""
    if not set(RULE_CONDITIONS) <= set(conditions):
        return None

    def incomplete(names: Sequence[str]) -> list[str]:
        return [c for c in names if not (conditions[c].get("coverage") or {}).get("complete", False)]

    s_none, s_ung, s_gate = (conditions[c]["success"] for c in RULE_CONDITIONS)
    h_ung, h_gate = conditions["ungated"]["harm_H"], conditions["gated"]["harm_H"]
    eps = 1e-9  # rates are small fractions; keep 0.95 >= 1.0 - 0.05 from failing on float error
    success_item: dict[str, Any] = {
        "none": s_none,
        "ungated": s_ung,
        "gated": s_gate,
        "bar": None,
        "holds": None,
        "incomplete": incomplete(RULE_CONDITIONS),
    }
    if not success_item["incomplete"] and None not in (s_none, s_ung, s_gate):
        success_item["bar"] = max(s_none, s_ung) - margin
        success_item["holds"] = s_gate >= success_item["bar"] - eps
    harm_item: dict[str, Any] = {
        "ungated": h_ung,
        "gated": h_gate,
        "holds": None,
        "incomplete": incomplete(("ungated", "gated")),
    }
    if not harm_item["incomplete"] and h_ung is not None and h_gate is not None:
        harm_item["holds"] = h_gate <= h_ung + eps
    return {
        "margin": margin,
        "coverage": {c: conditions[c].get("coverage") for c in RULE_CONDITIONS},
        "success": success_item,
        "harm": harm_item,
    }


def group_comparisons(rows_by: Mapping[str, Sequence[AnswerRow]], conditions: Sequence[str]) -> dict[str, Any]:
    """Paired comparisons within each task group (H2), for every pair of conditions."""
    groups = list(dict.fromkeys(TASK_BY_ID[r.task_id].group for c in conditions for r in rows_by[c]))
    out: dict[str, Any] = {}
    for group in groups:
        if group is None:
            continue
        has_harm = any(
            TASK_BY_ID[r.task_id].harm_signature
            for c in conditions
            for r in rows_by[c]
            if TASK_BY_ID[r.task_id].group == group
        )
        outcomes: list[tuple[str, Outcome]] = [("success", success)] + ([("harm", harm)] if has_harm else [])
        pairs: dict[str, Any] = {}
        for i, a in enumerate(conditions):
            for b in conditions[i + 1 :]:
                a_rows = [r for r in rows_by[a] if TASK_BY_ID[r.task_id].group == group]
                b_rows = [r for r in rows_by[b] if TASK_BY_ID[r.task_id].group == group]
                pairs[f"{a}_vs_{b}"] = {
                    name: {
                        "majority": paired(per_task(a_rows, outcome), per_task(b_rows, outcome)),
                        "any": paired(
                            per_task(a_rows, outcome, split_counts_as=True),
                            per_task(b_rows, outcome, split_counts_as=True),
                        ),
                    }
                    for name, outcome in outcomes
                }
        out[group] = pairs
    return out


def _error_counts(rows: Iterable[AnswerRow]) -> dict[str, int]:
    """Failed answers by error message (cut at 200 chars), so a report says why rows are missing."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.verdict is None:
            counts[" ".join((row.error or "no verdict").split())[:200]] += 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


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
        "errors": _error_counts(rows),
    }
    by_group = group_summary(rows)
    if by_group:
        out["by_group"] = by_group
    if recalls:
        out["trap_injection_H"] = injection(recalls, "H")
        out["useful_injection_U"] = injection(recalls, "U")
        out["recall_latency_ms_mean"] = _mean([r.latency_ms for r in recalls])
        out["abstained_tasks"] = sorted(r.task_id for r in recalls if not r.injected)
        admission = admission_stats(recalls)
        if admission is not None:
            out["admission"] = admission
    return out


def summarize(records: Sequence[RunRecord], notes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Merge one suite's run files by condition (each condition must come from exactly one run) and score.
    ``notes`` marks pre-registered conditions that are reported as exploratory (see ``NOTES_FILE``)."""
    suites = {record.suite for record in records}
    if len(suites) != 1:
        raise ValueError(f"summarize one suite at a time, got {sorted(suites)}")
    suite = SUITES[suites.pop()]
    source: dict[str, str] = {}
    by_label = {record.label: record for record in records}
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
                "harness_sha": record.harness_sha,
                "harness_dirty": record.harness_dirty,
                "admission_settings": record.admission_settings,
                "typesafe_calls": len(record.typesafe_calls),
                "typesafe_non_200": sum(1 for call in record.typesafe_calls if call.status != 200),
            }
            for record in records
        },
        "conditions": {c: condition_summary(rows_by[c], recalls_by[c]) for c in conditions},
        "exploratory": {c: EXPLORATORY[c] for c in conditions if c in EXPLORATORY},
        "reported_exploratory": {c: note for c, note in (notes or {}).items() if c in conditions},
        "spend_usd": round(sum(record.spend_usd for record in records), 4),
    }
    for c in conditions:
        run = by_label[source[c]]
        summary["conditions"][c]["coverage"] = coverage(
            rows_by[c], suite.tasks, run.repeats, stopped_for_budget=run.stopped_for_budget
        )
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
    by_group = group_comparisons(rows_by, conditions)
    if by_group:
        summary["group_comparisons"] = by_group
    summary["per_task"] = _per_task_table(rows_by, {c: by_label[source[c]].repeats for c in conditions}, suite.tasks)
    rule = downstream_rule(summary["conditions"])
    if rule is not None:
        summary["downstream_rule"] = rule
    gated_recalls = {r.task_id: r for r in recalls_by.get("gated", [])}
    if any(r.admission_summary is not None or r.admission is not None for r in gated_recalls.values()):
        ungated_recalls = {r.task_id: r for r in recalls_by.get("ungated", [])}
        summary["gate_accounts"] = {
            task.id: gate_account(task, gated_recalls[task.id], ungated_recalls.get(task.id))
            for task in suite.tasks
            if task.id in gated_recalls
        }
        summary["changed_vs_ungated"] = changed_tasks(summary["per_task"], "ungated", "gated")
        summary["not_compared_vs_ungated"] = not_compared_tasks(summary["per_task"], "ungated", "gated")
    return summary


def summarize_suites(records: Sequence[RunRecord], notes: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """One summary per suite present, in :data:`SUITES` order."""
    return {
        name: summarize([r for r in records if r.suite == name], notes)
        for name in SUITES
        if any(r.suite == name for r in records)
    }


def _per_task_table(
    rows_by: Mapping[str, list[AnswerRow]], repeats: Mapping[str, int], tasks: Sequence[Task]
) -> dict[str, Any]:
    """Per task and condition (``repeats`` maps each condition to its run's repeat count): the
    judged outcomes in repeat order, and ``complete`` when repeats 1..n each have one judged answer."""
    out: dict[str, Any] = {}
    for task in tasks:
        task_id = task.id
        entry: dict[str, Any] = {"kind": task.kind, "group": task.group}
        for c, n in repeats.items():
            answered = [r for r in rows_by[c] if r.task_id == task_id]
            rows = sorted(judged(answered), key=lambda r: r.repeat)
            if not rows:
                continue
            entry[c] = {
                "success": [r.verdict.success for r in rows if r.verdict],
                "harm": [r.verdict.harm for r in rows if r.verdict],
                "injected": rows[0].injected_labels,
                "reasons": [r.verdict.reason for r in rows if r.verdict],
                "complete": len(answered) == n and [r.repeat for r in rows] == list(range(1, n + 1)),
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
    exploratory = condition in summary.get("exploratory", {}) or condition in summary.get("reported_exploratory", {})
    return f"{condition} (exploratory)" if exploratory else condition


def _coverage_note(c: str, cov: Mapping[str, Any] | None) -> str:
    if not cov:
        return f"{c}: coverage unknown"
    note = f"{c} {cov['judged']}/{cov['expected']} judged"
    if cov["failed"]:
        note += f", {cov['failed']} failed"
    if cov["stopped_for_budget"]:
        note += ", stopped at budget"
    return note


def _item_result(item: Mapping[str, Any], coverage_by: Mapping[str, Any]) -> str:
    if item["holds"] is not None:
        return "**holds**" if item["holds"] else "**fails**"
    if item["incomplete"]:
        return f"not evaluable ({'; '.join(_coverage_note(c, coverage_by.get(c)) for c in item['incomplete'])})"
    return "not evaluable (no judged answers)"


def _failure_lines(summary: Mapping[str, Any]) -> list[str]:
    """One line per condition whose answers are missing: failed calls (by error) or not answered yet."""
    lines: list[str] = []
    for c, s in summary["conditions"].items():
        if s["answers"] == 0:
            lines.append(f"- `{c}`: recalls recorded, no answer made yet (rates are undefined).")
        elif s["failed"]:
            errors = "; ".join(f"{count}× {message}" for message, count in s.get("errors", {}).items())
            lines.append(
                f"- `{c}`: {s['failed']} of {s['answers']} answers failed and are excluded from every rate: {errors}"
            )
    return ["", *lines] if lines else []


def _decision(item: Mapping[str, Any]) -> str:
    if not item["judged"]:
        return f"`{item['label']}` not judged"
    verdict = "admitted" if item["admitted"] else item["reason"]
    return f"`{item['label']}` {verdict} {item['score']:.2f}"


def _tagged_decision(item: Mapping[str, Any]) -> str:
    """:func:`_decision`, plus the memory's role when it is the task's trap or needed memory."""
    return _decision(item) + ("" if item["role"] == "other" else f" [{item['role']}]")


def _outcome_cell(cell: Mapping[str, Any] | None, kind: str) -> str:
    if not cell:
        return "—"
    s = "".join("✓" if v else "✗" for v in cell["success"])
    h = "".join("H" if v else "·" for v in cell["harm"]) if kind == "H" else ""
    return f"{s} {h}".strip()


def _clip(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rule_lines(summary: Mapping[str, Any]) -> list[str]:
    rule = summary.get("downstream_rule")
    if rule is None:
        return []
    noted = [c for c in RULE_CONDITIONS if c in summary.get("reported_exploratory", {})]
    if summary.get("suite_note"):
        status = "exploratory: suite not pre-registered"
    elif noted:
        status = f"pre-registered items; exploratory because {', '.join(f'`{c}`' for c in noted)} is, see above"
    else:
        status = "pre-registered"
    s, h = rule["success"], rule["harm"]
    return [
        "",
        f"### Downstream rule items ({status})",
        "",
        "| item | none | ungated | gated | bar | result |",
        "|---|---|---|---|---|---|",
        f"| gated success ≥ max(none, ungated) − {rule['margin']:.2f} | {_fmt(s['none'])} | {_fmt(s['ungated'])} "
        f"| {_fmt(s['gated'])} | {_fmt(s['bar'])} | {_item_result(s, rule['coverage'])} |",
        f"| gated harm (H) ≤ ungated harm (H) | — | {_fmt(h['ungated'])} | {_fmt(h['gated'])} | {_fmt(h['ungated'])} "
        f"| {_item_result(h, rule['coverage'])} |",
        "",
        "An item holds or fails only when every condition it reads has one judged answer for each of its "
        "tasks × repeats and its run did not stop at the budget; otherwise it is not evaluable.",
    ]


def _group_comparison_lines(summary: Mapping[str, Any]) -> list[str]:
    groups = summary.get("group_comparisons")
    if not groups:
        return []
    lines = [
        "",
        "### Paired comparisons per group (per-task majority over repeats; exact McNemar)",
        "",
        "With n tasks in a group the smallest attainable two-sided p is 2/2^n (0.125 for 4 tasks), so these "
        "rows describe direction only.",
        "",
        "| group | comparison | outcome | tasks | A true | B true | A only | B only | p (majority) | p (split = true) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, pairs in groups.items():
        for name, outcomes in pairs.items():
            a, b = name.split("_vs_")
            for outcome, result in outcomes.items():
                maj, any_ = result["majority"], result["any"]
                if not maj["n"]:
                    continue
                lines.append(
                    f"| {group} | {_name(a, summary)} vs {_name(b, summary)} | {outcome} | {maj['n']} | {maj['a_true']} "
                    f"| {maj['b_true']} | {', '.join(maj['a_only']) or '—'} | {', '.join(maj['b_only']) or '—'} "
                    f"| {maj['p']:.3f} | {any_['p']:.3f} |"
                )
    return lines


def _gate_lines(summary: Mapping[str, Any]) -> list[str]:
    accounts = summary.get("gate_accounts")
    if not accounts:
        return []
    stats = summary["conditions"]["gated"].get("admission") or {}
    replayed = stats.get("recalls", 0) - len(stats.get("untraced_tasks", []))
    lines = [
        "",
        "### What the gate did (`gated`)",
        "",
        f"- {stats.get('recalls', 0)} gated recalls; abstained (nothing injected) on "
        f"{len(stats.get('abstained_tasks', []))}: {', '.join(stats.get('abstained_tasks', [])) or '—'}.",
        f"- Mean candidates judged {_fmt(stats.get('mean_candidates'), 1)}, mean admitted {_fmt(stats.get('mean_admitted'), 2)}; "
        f"rejections by reason (traced replay): {', '.join(f'{k} {v}' for k, v in stats.get('reasons', {}).items()) or '—'}; "
        f"relevance source: {', '.join(f'{k} {v}' for k, v in stats.get('relevance_source', {}).items()) or '—'}.",
        f"- Decision-model (Jev) failures, which fall back to the deterministic score: "
        f"{', '.join(stats.get('jev_error_tasks', [])) or 'none'}; embedding-read failures: "
        f"{', '.join(stats.get('embeddings_error_tasks', [])) or 'none'}; scope-read failures: "
        f"{', '.join(stats.get('scope_error_tasks', [])) or 'none'}.",
        f"- The traced replay reproduced the injected result list on {replayed - len(stats.get('replay_mismatch_tasks', []))}"
        f"/{replayed} recalls (mismatch: {', '.join(stats.get('replay_mismatch_tasks', [])) or 'none'}); admission phase "
        f"median {_fmt(stats.get('phase_ms_median'), 1)} ms, max {_fmt(stats.get('phase_ms_max'), 1)} ms (loaded machine).",
        "",
        "Scores are the gate's admission score (strict mode, hybrid scorer); `not judged` means the memory was not "
        "among the candidates the gate scored. The ungated column is the base server's top-10.",
        "",
        "| task | kind | ungated | gated | gated injected | trap: gate decision | needed memory: gate decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for task_id, account in accounts.items():
        entry = summary["per_task"][task_id]
        kind = f"{entry['kind']} ({entry['group']})" if entry.get("group") else entry["kind"]
        injected = ", ".join(_decision(i).replace(" admitted", "") for i in account["injected"]) or "— (abstained)"
        lines.append(
            f"| {task_id} | {kind} | {_outcome_cell(entry.get('ungated'), entry['kind'])} "
            f"| {_outcome_cell(entry.get('gated'), entry['kind'])} | {injected} "
            f"| {'; '.join(_decision(i) for i in account['traps']) or '—'} "
            f"| {'; '.join(_decision(i) for i in account['useful']) or '—'} |"
        )
    changed = summary.get("changed_vs_ungated", [])
    if not summary["conditions"]["gated"]["judged"]:
        return [
            *lines,
            "",
            "#### Tasks where gated differs from ungated",
            "",
            "Not evaluable: no gated answer was judged.",
        ]
    lines += ["", f"#### Tasks where gated differs from ungated ({len(changed)})", ""]
    skipped = summary.get("not_compared_vs_ungated", [])
    if skipped:
        lines += [
            f"Compared over {len(summary['per_task']) - len(skipped)} of {len(summary['per_task'])} tasks (every "
            f"repeat judged under both conditions); not compared ({len(skipped)}): {', '.join(skipped)}.",
            "",
        ]
    if not changed:
        lines.append("None: every compared task has the same success and harm counts under both conditions.")
    for task_id in changed:
        entry, account = summary["per_task"][task_id], accounts[task_id]
        task = TASK_BY_ID[task_id]
        lines += [
            f"**{task_id}** ({entry['kind']}{', ' + entry['group'] if entry.get('group') else ''}): ungated "
            f"{_outcome_cell(entry.get('ungated'), entry['kind'])} → gated {_outcome_cell(entry.get('gated'), entry['kind'])}",
            "",
            f"- traps: {', '.join(f'`{t}`' for t in task.trap_ids) or '—'}; needed: "
            f"{', '.join(f'`{u}`' for u in task.useful_ids) or '—'}",
            f"- ungated injected: {', '.join(f'`{label}`' for label in account['reference_injected']) or '—'}",
            f"- gate kept: {', '.join(_decision(i) for i in account['injected']) or '— (abstained)'}",
            f"- gate dropped (of the ungated top-k): {'; '.join(_tagged_decision(i) for i in account['dropped']) or '—'}",
            f"- gate added (not in the ungated top-k): {'; '.join(_decision(i) for i in account['added']) or '—'}",
        ]
        for condition in ("ungated", "gated"):
            cell = entry.get(condition) or {}
            for repeat, reason in enumerate(cell.get("reasons", []), start=1):
                lines.append(f"- judge, {condition} r{repeat}: {_clip(reason)}")
        lines.append("")
    return lines


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
            + (
                f"; harness {run['harness_sha'][:9]}{' + uncommitted changes' if run.get('harness_dirty') else ''}"
                if run.get("harness_sha")
                else ""
            )
            + (
                f"; {run['typesafe_calls']} TypeSafe calls ({run['typesafe_non_200']} non-200)"
                if run.get("typesafe_calls")
                else ""
            )
        )
        if run.get("admission_settings"):
            settings = ", ".join(
                f"{k.removeprefix('context_admission_')}={v}" for k, v in run["admission_settings"].items()
            )
            lines.append(f"  - resolved bank config: {settings}")
        scorer_env = {
            k.removeprefix("HINDSIGHT_API_CONTEXT_ADMISSION_").lower(): v
            for k, v in (run.get("server_env") or {}).items()
            if k.startswith("HINDSIGHT_API_CONTEXT_ADMISSION_")
            and not k.endswith(("_API_KEY", "_BASE_URL", "_WEIGHTS"))
            and v != "<redacted>"
        }
        if scorer_env:
            lines.append(f"  - admission server env: {', '.join(f'{k}={v}' for k, v in sorted(scorer_env.items()))}")
    for c, note in summary.get("exploratory", {}).items():
        lines.append(f"- `{c}`: **{note}**; not part of the pre-registered analysis.")
    for c, note in summary.get("reported_exploratory", {}).items():
        lines.append(f"- `{c}`: **{note}**; every number for it is reported as exploratory.")
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
    lines += _failure_lines(summary)
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
    lines += _rule_lines(summary)
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
    lines += _group_comparison_lines(summary)
    lines += [
        "",
        "### Per task (success / harm per repeat)",
        "",
        "| task | kind | " + " | ".join(_name(c, summary) for c in conditions) + " |",
    ]
    lines.append("|---|---|" + "---|" * len(conditions))
    for task_id, entry in summary["per_task"].items():
        cells = [_outcome_cell(entry.get(c), entry["kind"]) for c in conditions]
        kind = f"{entry['kind']} ({entry['group']})" if entry.get("group") else entry["kind"]
        lines.append(f"| {task_id} | {kind} | " + " | ".join(cells) + " |")
    lines += _gate_lines(summary)
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


def load_notes(out_dir: Path) -> dict[str, str]:
    """``out_dir / NOTES_FILE`` as {condition: note}; empty when the file is absent."""
    path = out_dir / NOTES_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(isinstance(v, str) and v.strip() for v in data.values()):
        raise ValueError(f"{path}: expected an object of condition -> non-empty note")
    unknown = sorted(set(data) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"{path}: unknown condition(s) {unknown}; choose from {CONDITIONS}")
    return dict(data)


def write_report(out_dir: Path, runs: Sequence[Path] | None = None) -> dict[str, Any]:
    paths = list(runs) if runs else sorted(p for p in (out_dir / "runs").glob("*.json") if ".partial" not in p.name)
    if not paths:
        raise FileNotFoundError(f"no run files under {out_dir / 'runs'}")
    records = [load_record(path) for path in paths]
    summaries = summarize_suites(records, load_notes(out_dir))
    summary = {
        "suites": summaries,
        "spend_usd": round(sum(record.spend_usd for record in records), 4),
        "run_files": [str(path) for path in paths],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "REPORT.md").write_text(report_markdown(summaries))
    return summary
