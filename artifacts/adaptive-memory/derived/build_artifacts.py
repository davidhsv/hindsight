"""Build artifacts/adaptive-memory/*.json from the scored eval files (no LLM calls, no servers).

Every number is copied from a scored source file, or recomputed from that file's per-question rows
with the same rule its scorer uses (exact McNemar, Clopper-Pearson). Each block names its source.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/davidvieira/code/hindsight")
T = ROOT / "tasks/adaptive-memory"
E = T / "eval"
OUT = ROOT / "artifacts/adaptive-memory"
SCRATCH = Path("/private/tmp/claude-502/-Users-davidvieira-code-hindsight/823f87da-11e6-43e4-a7a6-b3e4cbaba9e3/scratchpad")
AMB_OUT = SCRATCH / "amb-checkout/outputs"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


# ----------------------------------------------------------------------------- statistics

def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def _cdf(x: int, n: int, p: float) -> float:
    return min(1.0, sum(math.exp(_log_binom_pmf(k, n, p)) for k in range(0, x + 1)))


def _bisect(f, lo=0.0, hi=1.0, it=200) -> float:
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Two-sided exact interval."""
    if n == 0:
        return None
    lower = 0.0 if x == 0 else _bisect(lambda p: 1 - _cdf(x - 1, n, p) >= alpha / 2)
    upper = 1.0 if x == n else _bisect(lambda p: _cdf(x, n, p) <= alpha / 2)
    return [round(lower, 4), round(upper, 4)]


def one_sided_upper(x: int, n: int, alpha: float = 0.05) -> float:
    return round(1.0 if x == n else _bisect(lambda p: _cdf(x, n, p) <= alpha), 4)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * ps[k]))
        out[k] = running
    return out


def prop(x: int, n: int, **extra) -> dict:
    d = {"value": round(x / n, 4) if n else None, "x": x, "n": n, "ci95_clopper_pearson": clopper_pearson(x, n)}
    d.update(extra)
    return d


# ----------------------------------------------------------------------------- memory_lifecycle

LC_KEYS = [
    "current_precision", "current_recall", "stale_contamination", "stale_contamination_raw",
    "stale_contamination_obs", "mixed_rate", "refuted_leakage", "historical_recall", "scope_accuracy",
    "supersession_accuracy", "uncertainty_accuracy", "independent_reinforcement_accuracy",
]


def lifecycle_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {
        "label": d["label"], "arm": d["arm"], "encoding": d["encoding"], "api_sha": d["api_sha"],
        "api_dirty": d["api_dirty"], "embedder": d["embedder"], "payload_source": d["payload_source"],
        "server_env_adaptive": {k: v for k, v in d["server_env"].items() if "ADAPTIVE" in k},
        "bank_config": d["bank_config"], "recall_types": d["recall_types"], "started_at_utc": d["started_at"],
        "source": rel(path), "metrics": {},
        "note": "each value is the mean over the split's applicable queries of a per-query score (n = queries); no CI computed",
    }
    for split in ("all", "holdout"):
        out["metrics"][split] = {}
        for k in LC_KEYS:
            for kk in (1, 5):
                v = d["metrics"][split].get(f"{k}@{kk}")
                if v is not None:
                    out["metrics"][split][f"{k}@{kk}"] = {"mean": None if v["mean"] is None else round(v["mean"], 4), "n": v["n"]}
    sc = d.get("state_conformance") or {}
    out["state_conformance"] = {k: {"mean": None if v["mean"] is None else round(v["mean"], 4), "n": v["n"]} for k, v in sc.items()}
    return out


LB = E / "lifecycle-bench"


def lc(name: str) -> dict:
    return lifecycle_run(LB / name)


def row_equiv(path: Path) -> dict:
    return {"text": path.read_text().strip(), "source": rel(path)}


# ----------------------------------------------------------------------------- AMB retrieval endpoints

def endpoints(path: Path, arm: str, ref: str) -> dict:
    tab = json.loads(path.read_text())[arm]
    out = {}
    for e, t in sorted(tab.items()):
        x = round(t["rate"] * t["n"])
        entry = prop(x, t["n"])
        if arm != ref:
            entry.update({"reference_arm": ref, "reference_value": round(t["ref_rate"], 4), "ref_only": t["ref_only"],
                          "arm_only": t["arm_only"], "p_exact_mcnemar": t["p_mcnemar"]})
        entry["source"] = rel(path)
        out[e] = entry
    return out


NOOBS = E / "arms/score-noobs/endpoints.json"
OBS = E / "arms/score-obs/endpoints.json"
OBS_REF_E = OUT / "derived/score-obs-ref-Eobs/endpoints.json"

# ----------------------------------------------------------------------------- judged LongMemEval

@functools.lru_cache(maxsize=None)
def load_lme(name: str) -> dict[str, dict]:
    d = json.loads((AMB_OUT / "longmemeval" / name / "rag/s.json").read_text())
    return {r["query_id"]: r for r in d["results"]}


def top10_agreement(a: str, b: str) -> dict:
    """Questions whose top-10 recalled (document, type, mentioned_at) sequence is identical in runs a and b.
    Import assigns new memory ids per server, so ids are left out of the signature."""
    ra, rb = load_lme(a), load_lme(b)
    common = set(ra) & set(rb)

    def sig(r: dict) -> list:
        return [tuple(x[1:]) for x in r["meta"]["retrieved"][:10]]

    return {"runs": [a, b], "n": len(common), "top10_identical": sum(sig(ra[q]) == sig(rb[q]) for q in common),
            "full_context_identical": sum(ra[q]["context"] == rb[q]["context"] for q in common)}


def judged_table(j: dict) -> dict:
    """Compact per-arm view of a judged() result, so every arm file shows all arms."""
    out = {}
    for a, e in j["arms"].items():
        row = {"per_repeat": [r["value"] for r in e["per_repeat"]], "both_correct": e["both_correct"]["value"],
               "both_correct_x": e["both_correct"]["x"], "at_least_one": e["at_least_one"]["value"]}
        for key in ("both_correct", "at_least_one"):
            if "ref_only" in e[key]:
                row[f"{key}_vs_{e[key]['reference_arm']}"] = {"ref_only": e[key]["ref_only"], "arm_only": e[key]["arm_only"], "p_exact_mcnemar": e[key]["p_exact_mcnemar"]}
        for cat in ("knowledge-update", "temporal-reasoning"):
            if f"both_correct_{cat}" in e:
                row[f"both_correct_{cat}"] = e[f"both_correct_{cat}"]["value"]
        out[a] = row
    return out


def qtype(r: dict) -> str:
    for v in (r.get("category_axes") or {}).values():
        if v and v[0] in ("knowledge-update", "temporal-reasoning"):
            return v[0]
    return str((r.get("meta") or {}).get("question_type", "?"))


def judged(arms: dict[str, list[str]], ref: str, source_md: Path) -> dict:
    runs = {a: [load_lme(n) for n in ns] for a, ns in arms.items()}
    common = set.intersection(*(set(r) for rs in runs.values() for r in rs))
    types = {q: qtype(runs[ref][0][q]) for q in common}
    out = {"n_questions": len(common), "source_scored": rel(source_md),
           "source_rows": {a: [str(AMB_OUT / "longmemeval" / n / "rag/s.json") for n in ns] for a, ns in arms.items()},
           "rule": "primary = correct in both repeats; sensitivity = correct in at least one (PREREGISTRATION amendment 2026-10-17 20:34); exact McNemar vs reference",
           "arms": {}}
    outcome = {}
    for a, rs in runs.items():
        outcome[a] = ({q: all(bool(r[q]["correct"]) for r in rs) for q in common},
                      {q: any(bool(r[q]["correct"]) for r in rs) for q in common})
    for a, rs in runs.items():
        both, anyc = outcome[a]
        entry = {"runs": arms[a],
                 "per_repeat": [prop(sum(bool(r[q]["correct"]) for q in common), len(common)) for r in rs],
                 "both_correct": prop(sum(both.values()), len(common)),
                 "at_least_one": prop(sum(anyc.values()), len(common))}
        for cat in ("knowledge-update", "temporal-reasoning"):
            qs = [q for q in common if types[q] == cat]
            if qs:
                entry[f"both_correct_{cat}"] = prop(sum(both[q] for q in qs), len(qs))
        if a != ref:
            for key, idx in (("both_correct", 0), ("at_least_one", 1)):
                rq, aq = outcome[ref][idx], outcome[a][idx]
                b = sum(rq[q] and not aq[q] for q in common)
                c = sum(aq[q] and not rq[q] for q in common)
                entry[key].update({"reference_arm": ref, "ref_only": b, "arm_only": c, "p_exact_mcnemar": round(mcnemar(b, c), 4)})
        out["arms"][a] = entry
    return out


# ----------------------------------------------------------------------------- LoCoMo

RANK = re.compile(r"first gold rank=(\d+)")


def load_locomo(name: str) -> dict[str, int]:
    d = json.loads((AMB_OUT / "locomo" / name / "retrieval/locomo10.json").read_text())
    out = {}
    for r in d["results"]:
        m = RANK.search(r.get("judge_reason") or "")
        out[r["query_id"]] = int(m.group(1)) if m else 10**9
    return out


def locomo(arms: dict[str, str], ref: str) -> dict:
    ranks = {a: load_locomo(n) for a, n in arms.items()}
    common = set.intersection(*(set(r) for r in ranks.values()))
    out = {"n_queries": len(common), "source_scored": rel(E / "judged/locomo-endpoints.md"),
           "source_rows": {a: str(AMB_OUT / "locomo" / n / "retrieval/locomo10.json") for a, n in arms.items()},
           "endpoint": "first gold rank <= k (AMB retrieval mode, AMB default budget)", "arms": {}}
    for a in arms:
        entry = {"run": arms[a]}
        for k in (5, 10, 20):
            hit = {q: ranks[a][q] <= k for q in common}
            e = prop(sum(hit.values()), len(common))
            if a != ref:
                rh = {q: ranks[ref][q] <= k for q in common}
                b = sum(rh[q] and not hit[q] for q in common)
                c = sum(hit[q] and not rh[q] for q in common)
                e.update({"reference_arm": ref, "ref_only": b, "arm_only": c, "p_exact_mcnemar": float(f"{mcnemar(b, c):.3g}")})
            entry[f"gold@{k}"] = e
        entry["any_gold"] = prop(sum(ranks[a][q] < 10**9 for q in common), len(common))
        out["arms"][a] = entry
    return out


# ----------------------------------------------------------------------------- shared metadata

SHAS = {
    "base": "5331da2f4db369528bb122075341a7de9c3df6df",
    "head": "554284233f4fdb38d1e065c0d90b94ed9deb38a4",
    "P1": "dafd09a64", "P2": "3816d0e4b", "P3": "b491b2bb9", "P4": "d65f17b05", "P5": "728b265c9",
    "T2": "504ca72e9", "T2_harm_bench": "2f4c7daae",
}
JUDGED_E_PROVENANCE = {
    "runs": "judged-E-r1 and judged-E-r2 are fresh re-runs (2026-10-18 07:07-07:55 BRT) on srv-e: P5 worktree 728b265c9, ADAPTIVE_MEMORY_MODE=lifecycle, frozen arm-A archives, command eval/judged/rerun_E.sh",
    "unused_partial": "the pre-outage E-r1 (189/211 judged answers, exit 1 on the 2026-10-18 01:09:41 BRT credit error) and E-r2 (exit 1, nothing saved) are used for no number; the partial AMB output is kept at <scratchpad>/judged-E-r1-partial-credit/rag/s.json, and its usage rows are the first 379 lines (00:47-01:09 BRT) of eval/judged/usage-judged-E-r1.jsonl",
    "exit_codes": rel(E / "judged/exit-codes"),
    "spend_usd": {"E-r1": 1.358, "E-r2": 1.358, "method": "eval/scripts/amb_cost.py over the usage rows after 06:00 BRT (AMB-side answer + judge)"},
}

# ----------------------------------------------------------------------------- performance (no LLM; eval/perf)

PERF = E / "perf/performance.json"
PERF_REPORT = E / "perf/REPORT.md"
T95_3DF = 3.182446305284263  # two-sided 95% Student t quantile, 3 degrees of freedom (4 rounds)
RUN_LEVEL_COMPARISONS = ("B_vs_A", "C_vs_B", "D_vs_B", "E_vs_B")
ADMISSION_ARMS = ("E+observe", "E+filter-det", "E+strict-det", "E+strict-hybrid")
PERF_RAW = E / "perf/raw"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_level(perf: dict) -> dict:
    """Between-run view of each overhead: a paired t interval over the 4 rounds' per-round p95 (and p50)
    overheads, and DESIGN.md 11.4's form, median of per-run p95 vs median of per-run p95."""
    out: dict = {"method": "per round r: overhead_r = p_arm_r / p_ref_r - 1; mean +/- t(0.975, 3) * sd / sqrt(4); "
                           "median_of_run_p95 = median_r p95_arm_r / median_r p95_ref_r - 1. Recomputed here from the per-round "
                           "quantiles in performance.json; it reproduces the run-level t95 intervals quoted in eval/perf/REPORT.md"}
    for scen in ("recall_noobs", "recall_obs"):
        out[scen] = {}
        for cmp_ in RUN_LEVEL_COMPARISONS:
            arm, ref = cmp_.split("_vs_")
            out[scen][cmp_] = {}
            for shape in ("seq", "conc"):
                ra = perf[scen]["arms"][arm][shape]["per_round"]
                rr = perf[scen]["arms"][ref][shape]["per_round"]
                rounds = sorted(ra)
                cell = {}
                for q in ("p95", "p50"):
                    v = [ra[r][f"{q}_ms"] / rr[r][f"{q}_ms"] - 1 for r in rounds]
                    m, h = statistics.mean(v), T95_3DF * statistics.stdev(v) / math.sqrt(len(v))
                    cell[f"{q}_per_round"] = [round(x, 4) for x in v]
                    cell[f"{q}_run_level_mean"] = round(m, 4)
                    cell[f"{q}_run_level_t95"] = [round(m - h, 4), round(m + h, 4)]
                cell["median_of_run_p95"] = round(statistics.median(ra[r]["p95_ms"] for r in rounds)
                                                  / statistics.median(rr[r]["p95_ms"] for r in rounds) - 1, 4)
                cell["pooled_p95"] = perf[scen]["comparisons"][cmp_][shape]["pooled"]["p95"]["overhead"]
                cell["pooled_p95_cluster_ci95"] = perf[scen]["comparisons"][cmp_][shape]["pooled"]["p95"]["ci95"]
                out[scen][cmp_][shape] = cell
    # The admission arms ran in the noobs block only; their reference is E-ext (lifecycle recall made the way the
    # HTTP endpoint makes it, admission off). Same paired-t form as above.
    out["recall_noobs_admission_vs_E_ext"] = {}
    for arm in ADMISSION_ARMS:
        cmp_ = f"{arm}_vs_E-ext"
        out["recall_noobs_admission_vs_E_ext"][cmp_] = {}
        for shape in ("seq", "conc"):
            ra = perf["recall_noobs"]["arms"][arm][shape]["per_round"]
            rr = perf["recall_noobs"]["arms"]["E-ext"][shape]["per_round"]
            rounds = sorted(ra)
            v = [ra[r]["p95_ms"] / rr[r]["p95_ms"] - 1 for r in rounds]
            m, h = statistics.mean(v), T95_3DF * statistics.stdev(v) / math.sqrt(len(v))
            pooled = perf["recall_noobs"]["comparisons"][cmp_][shape]["pooled"]["p95"]
            out["recall_noobs_admission_vs_E_ext"][cmp_][shape] = {
                "p95_per_round": [round(x, 4) for x in v],
                "p95_run_level_mean": round(m, 4),
                # 5 places, so the one-decimal percentages quoted from it are not double-rounded
                "p95_run_level_t95": [round(m - h, 5), round(m + h, 5)],
                "pooled_p95": pooled["overhead"], "pooled_p95_delta_ms": pooled["delta_ms"],
                "pooled_p95_cluster_ci95": pooled["ci95"],
                "pooled_p95_cluster_ci95_note": "the analysis plan's within-run cluster bootstrap; anti-conservative (62% A/A false positives, eval/perf/REPORT.md), so read the run-level interval",
            }
    return out


def harness_pct(values: list[float], p: float) -> float:
    """The perf harness's percentile, sorted[min(int(p/100*n), n-1)] (eval/perf/scripts/analyze.py)."""
    v = sorted(values)
    return v[min(int(p / 100 * len(v)), len(v) - 1)]


def admission_phase_per_round() -> dict:
    """Per-round context_admission phase p95 (ms) of the in-process admission arms, from eval/perf/raw/recall/noobs."""
    out: dict = {"method": "harness percentile of the context_admission phase per run (16 warm-ups discarded by the driver); "
                           "the pooled p95 and its within-run bootstrap CI are in admission_inprocess"}
    for arm in ADMISSION_ARMS:
        stem = arm.replace("+", "_")
        out[arm] = {}
        for shape in ("seq", "conc"):
            per = []
            for f in sorted((PERF_RAW / "recall/noobs").glob(f"r*-{stem}.json")):
                s = json.loads(f.read_text())["samples"]
                per.append(round(harness_pct([x["phases"].get("context_admission", 0.0) * 1000 for x in s if x["phase"] == shape], 95), 1))
            out[arm][shape] = {"p95_ms_per_round": per, "range_ms": [min(per), max(per)]}
    return out


def hybrid_inprocess_decision_model_use() -> dict:
    """The in-process E+strict-hybrid arm used the real TypeSafe decision model on every recall, outside the
    proxy that counted the context_bench runs' calls; count what the raw samples show."""
    runs = sorted((PERF_RAW / "recall/noobs").glob("r*-E_strict-hybrid.json"))
    smoke = PERF_RAW / "smoke/noobs/r0-E_strict-hybrid.json"
    n = slow = err = abst = 0
    providers = set()
    for f in runs:
        d = json.loads(f.read_text())
        providers.add(d["env"].get("HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER"))
        for s in d["samples"]:
            n += 1
            slow += s["phases"].get("context_admission", 0.0) > 0.1
            err += bool(s["admission"].get("jev_error"))
            abst += bool(s["admission"].get("abstained"))
    smoke_n = len(json.loads(smoke.read_text())["samples"]) if smoke.exists() else 0
    return {
        "decision_model_provider_in_run_env": sorted(p for p in providers if p),
        "analysed_recalls": n, "analysed_recalls_admission_phase_over_100ms": slow, "jev_error": err, "abstained": abst,
        "warmup_recalls": 16 * len(runs), "smoke_recalls_recorded": smoke_n,
        "requests_counted": "no: run_arm.py passes the real TYPESAFE_API_KEY with no base-URL override, so these requests bypassed the proxy that counted the context_bench runs' 510 calls",
        "reading": "at least one decision-model request per gated recall (one per recall per the jev/hybrid scorer's design), inferred from the run env, the code and the measured >100 ms admission phase on every analysed recall; so about 2,500 uncounted requests",
    }


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def ci_pct(ci: list[float]) -> str:
    return f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"


def recall_cell(rl: dict, scen: str, cmp_: str, shape: str) -> str:
    c = rl[scen][cmp_][shape]
    return (f"pooled p95 {pct(c['pooled_p95'])} (cluster CI {ci_pct(c['pooled_p95_cluster_ci95'])}); run-level t95 "
            f"{ci_pct(c['p95_run_level_t95'])}; median of per-run p95 {pct(c['median_of_run_p95'])}")


def verified_verdicts(perf: dict, rl: dict) -> dict:
    adm = perf["criteria"]["admission_det_p95_le_150ms"]
    return {
        "source": rel(PERF_REPORT) + " (verdicts rewritten after an independent read-only verification that recomputed every pooled quantile, overhead and CI from eval/perf/raw/ with a different bootstrap seed)",
        "supersedes": "the `criteria` block below as written by analyze.py: its 'holds' flags are point-estimate readings, and its 'holds_with_ci' flags rest on the analysis plan's cluster bootstrap, which the verification found anti-conservative. eval/perf/REPORT.md's closing 'What holds' list still carries that earlier wording; its criteria table and verification paragraph carry the verified verdicts",
        "why": "the cluster bootstrap resamples within runs and ignores between-run variance; in an A/A check (one arm's 4 runs split into two pseudo-arms, 60 comparisons) its 95% CI excluded 0 in 62% of p95 cases against a nominal 5% (eval/perf/REPORT.md; the A/A check was not re-run here). Implausible 'significant' results follow from it (D faster than B at noobs conc p95; B vs A conc p95 +8.8% [+1.0, +19.3])",
        "lifecycle_E_vs_B_p95_lt_10pct": {
            "verdict": "point estimate under the bar on the primary (noobs) scenario; not resolved once run-to-run variance is counted. With observations, the concurrent point estimate is over the bar and also unresolved at run level",
            "noobs_seq": recall_cell(rl, "recall_noobs", "E_vs_B", "seq"),
            "noobs_conc": recall_cell(rl, "recall_noobs", "E_vs_B", "conc"),
            "obs_seq": recall_cell(rl, "recall_obs", "E_vs_B", "seq"),
            "obs_conc": recall_cell(rl, "recall_obs", "E_vs_B", "conc"),
            "robust": f"obs conc p50 run-level {pct(rl['recall_obs']['E_vs_B']['conc']['p50_run_level_mean'])} {ci_pct(rl['recall_obs']['E_vs_B']['conc']['p50_run_level_t95'])}; noobs seq p50 pooled +8.6% (+6.3 ms)",
        },
        "rank_D_vs_B_p95_lt_5pct": {
            "verdict": "point estimate under the bar on the primary (noobs) scenario; not resolved; fragile to round 1 (without round 1, exploratory: noobs conc +8.1%). With observations, the concurrent point estimate is over the bar, unresolved at run level",
            "noobs_seq": recall_cell(rl, "recall_noobs", "D_vs_B", "seq"),
            "noobs_conc": recall_cell(rl, "recall_noobs", "D_vs_B", "conc"),
            "obs_seq": recall_cell(rl, "recall_obs", "D_vs_B", "seq"),
            "obs_conc": recall_cell(rl, "recall_obs", "D_vs_B", "conc"),
        },
        "admission_det_p95_le_150ms": {
            "verdict": "holds for sequential traffic (about 50x margin); under c=8 in-process, observe reaches 273.9 ms p95 (fails) and strict 68.6 ms",
            "primary_context_bench_http_255_recalls_per_mode": adm["primary_context_bench_http"],
            "secondary_inprocess": adm["secondary_inprocess"],
            "secondary_inprocess_phase_p95_per_round": admission_phase_per_round(),
            "observe_recall_p95_overhead_c8": "E+observe vs E-ext concurrent recall p95: pooled point estimate "
                                              f"{pct(rl['recall_noobs_admission_vs_E_ext']['E+observe_vs_E-ext']['conc']['pooled_p95'])} "
                                              f"(+{rl['recall_noobs_admission_vs_E_ext']['E+observe_vs_E-ext']['conc']['pooled_p95_delta_ms']:.0f} ms), not resolved at run level: t95 "
                                              f"{ci_pct(rl['recall_noobs_admission_vs_E_ext']['E+observe_vs_E-ext']['conc']['p95_run_level_t95'])}, per round "
                                              + ", ".join(pct(x) for x in rl['recall_noobs_admission_vs_E_ext']['E+observe_vs_E-ext']['conc']['p95_per_round'])
                                              + "; the admission phase p95 itself (273.9 ms pooled) is stable across rounds",
            "jev_and_hybrid_paths": "one TypeSafe call per recall: phase p50 / p95 344.9 / 432.5 ms (hybrid, F strict) and 347.7 / 432.1 ms (jev, E-jev filter); not under the 150 ms item, which names the deterministic path",
        },
        "p95_is_gc_dominated": "about 12% of sequential recalls in every arm, A included, carry an extra ~190 ms; the sequential p95 (about 255-270 ms everywhere) measures that stall, not the per-recall work (gc_probe below: 47 gen-2 collections over 100 ms in a window the length of the sequential phase vs 47 sequential recalls over 150 ms, a count match, not a per-sample alignment)",
        "median_cost": "noobs sequential p50: D +9.1% (+6.6 ms), E +8.6% (+6.3 ms); concurrent p50 +11.4% / +14.2%",
        "undeclared_deviation": "DESIGN.md 11.4's gate (median of >= 5 runs at --scale large) was replaced by 4 pooled medium rounds; under the median-of-per-run-p95 form, noobs E conc is +10.7% and obs D seq +6.2%, both over their bars (run_level_recomputation)",
        "machine": "Apple M1 Pro, 10 cores; runs waited while the 1-minute load was > 8 (at most 10 min); background load about 3-4 cores; 3 noobs round-4 runs started above the ceiling after the maximum wait; not an idle machine",
    }


def decision_rule_application(rl: dict) -> dict:
    return {
        "track1_item4_lifecycle_p95_lt_10pct": {
            "mechanical": "holds: raw/analysis_plan.json (file mtime 2026-10-18T05:13:48Z, before the first analysed run's machine-state check at 05:13:53Z and after tiny smoke runs and one medium pilot of arm E, both excluded; its own written_at field says 05:20Z, which is not its write time) makes the noobs scenario on both shapes the decision cell and reads 'holds' as a point estimate under the bar; E vs B noobs p95 "
                          f"{pct(rl['recall_noobs']['E_vs_B']['seq']['pooled_p95'])} seq, {pct(rl['recall_noobs']['E_vs_B']['conc']['pooled_p95'])} conc",
            "author_judgment": "not established: unresolved at run level on the primary scenario; over the bar at c=8 with observations (the recommended configuration); over the bar in DESIGN 11.4's median-of-per-run form (noobs conc +10.7%); and the p95 mostly measures GC pauses",
        },
        "track1_item_c_rank_p95_lt_5pct": {
            "mechanical": f"holds: D vs B noobs p95 {pct(rl['recall_noobs']['D_vs_B']['seq']['pooled_p95'])} seq, {pct(rl['recall_noobs']['D_vs_B']['conc']['pooled_p95'])} conc",
            "author_judgment": "not established (run-level t95 straddles the bar; without round 1 noobs conc +8.1%, exploratory; obs conc +38.9%; obs seq median-of-per-run +6.2%). Moot: items a and b already fail or are non-confirmatory, so opt-in rank is not met either way",
        },
        "track2_deterministic_admission_p95_le_150ms": {
            "mechanical": "holds: phase p95 3.18 / 1.17 / 3.11 ms (observe / filter / strict) over 255 sequential HTTP recalls each",
            "author_judgment": "holds for sequential traffic only (observe 273.9 ms p95 in-process at c=8); moot for the Track 2 verdict, which the quality items already fail",
        },
    }


def meta(arm: str, description: str, extra: dict | None = None) -> dict:
    d = {"arm": arm, "description": description, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
         "base_sha": SHAS["base"], "feature_head_sha": SHAS["head"], "phase_commits": {k: v for k, v in SHAS.items() if k not in ("base", "head")},
         "models": {"server_extraction_consolidation": "claude-haiku-5-5", "amb_answer": "claude-haiku-5-5", "judge": "claude-sonnet-5-5", "decision_model": "TypeSafe jev-latest"},
         "embedder": "local bge-small-en-v1.5 (CPU)", "reranker": "local ms-marco-MiniLM-L-6-v2 cross-encoder (CPU)",
         "frozen_parameters": {"adaptive_memory_score_alpha": 0.1, "score_floor": 0.3, "base_stability_days": 30, "max_stability_days": 3650,
                               "type_multipliers": {"experience": 0.5, "world": 1.0, "observation": 2.0}, "cooldown_hours": 6, "gain": 1.0,
                               "recall_reinforcement": False, "conflict_policy": "evidence", "historical_cues": False,
                               "source": rel(E / "PREREGISTRATION.md")}}
    if extra:
        d.update(extra)
    return d


def write(name: str, obj: dict) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    judged_noobs = judged({"A": ["judged-A-r1", "judged-A-r2"], "B": ["judged-B-r1", "judged-B-r2"], "D": ["judged-D-r1", "judged-D-r2"],
                           "E": ["judged-E-r1", "judged-E-r2"]},
                          "B", E / "judged/judged-noobs.md")
    judged_noobs["all_arms"] = judged_table(judged_noobs)
    judged_e_vs_d = judged({"D": ["judged-D-r1", "judged-D-r2"], "E": ["judged-E-r1", "judged-E-r2"]}, "D", E / "judged/judged-noobs.md")
    judged_e_recall_check = {
        "what": "top-10 recalled (document, type, mentioned_at) sequence per question, ids excluded (import assigns new ids per server)",
        "E_vs_D": [top10_agreement("judged-D-r1", "judged-E-r1"), top10_agreement("judged-D-r2", "judged-E-r2")],
        "D_repeat_vs_repeat": top10_agreement("judged-D-r1", "judged-D-r2"),
        "E_repeat_vs_repeat": top10_agreement("judged-E-r1", "judged-E-r2"),
        "B_vs_E": top10_agreement("judged-B-r1", "judged-E-r1"),
        "B_vs_D": top10_agreement("judged-B-r1", "judged-D-r1"),
        "reading": "E's top-10 matches D's as closely as D's two repeats match each other, as expected without observations (no claims); E differs from B like D does (the retention factor). Full contexts beyond the top 10 differ in most questions even between two repeats of one arm, so the context tail is not deterministic",
    }
    judged_obs = judged({"B-obs": ["judged-Bobs-r1", "judged-Bobs-r2"], "E-obs": ["judged-Eobs-r1", "judged-Eobs-r2"]},
                        "B-obs", E / "judged/judged-obs.md")
    judged_obs["all_arms"] = judged_table(judged_obs)
    perf = json.loads(PERF.read_text())
    perf_rl = run_level(perf)
    perf_verified = verified_verdicts(perf, perf_rl)
    perf_rule = decision_rule_application(perf_rl)

    def perf_arm(scen_arm: str) -> dict:
        cmp_ = f"{scen_arm}_vs_B"
        return {
            "source": rel(PERF), "report": rel(PERF_REPORT),
            "recall_noobs_vs_B": perf["recall_noobs"]["comparisons"][cmp_],
            "recall_obs_vs_B": perf["recall_obs"]["comparisons"][cmp_],
            "run_level": {"recall_noobs": perf_rl["recall_noobs"][cmp_], "recall_obs": perf_rl["recall_obs"][cmp_]},
            "recall_pooled_ms": {"noobs": {s: perf["recall_noobs"]["arms"][scen_arm][s]["pooled"] for s in ("seq", "conc")},
                                 "obs": {s: perf["recall_obs"]["arms"][scen_arm][s]["pooled"] for s in ("seq", "conc")}},
            "official_suites_medium": perf["official_suites"][scen_arm],
            "obs_runs_consolidation_s": perf["recall_obs"]["arms"][scen_arm]["consolidation_s"],
            "storage_after_100_observations": {"rows": perf["recall_obs"]["arms"][scen_arm]["table_rows"], "bytes_round1": perf["recall_obs"]["arms"][scen_arm]["table_bytes_round1"]},
        }
    loco = locomo({"A": "locomo-A", "B": "locomo-B", "D": "locomo-D", "E": "locomo-E", "E-nodecay": "locomo-E-nodecay"}, "B")

    noobs_primary_p = json.loads(NOOBS.read_text())["D"]["knowledge-update:latest_first"]["p_mcnemar"]
    obs_primary_p = json.loads(OBS.read_text())["E-obs"]["knowledge-update:latest_first"]["p_mcnemar"]
    holm_adj = holm({"P1-life": obs_primary_p, "P1-rank": noobs_primary_p})

    amb_common = {
        "dataset": "LongMemEval-s cleaned (AMB loader), knowledge-update 72 + temporal-reasoning 127 non-abstention questions; frozen arm-A banks imported into every arm",
        "endpoint_definitions": rel(E / "arms/score_endpoints.py"),
    }

    # ---------------- baseline (A)
    write("baseline.json", {
        **meta("A", "base SHA 5331da2f4, unmodified Hindsight (arm A)"),
        "memory_lifecycle": {
            "base-delete": lc("arm-A-base-delete.json"), "base-update": lc("arm-A-base-update.json"),
            "existing_feature_baselines": {
                "recall_prefer_observations=true": lc("arm-A-base-delete-prefer-observations.json"),
                "types=['observation']": lc("arm-A-base-delete-types-observation.json"),
                "lexical stub embedder (smoke only)": lc("arm-A-base-delete-lexical.json"),
            },
            "determinism": row_equiv(LB / "determinism.txt"),
            "report": rel(LB / "arm-A-report.md"),
        },
        "amb_longmemeval_retrieval_noobs": {**amb_common, "reference_arm": "B", "endpoints": endpoints(NOOBS, "A", "B")},
        "amb_longmemeval_judged_noobs": {"budget": "AMB_RECALL_MAX_TOKENS=4096, AMB_RECALL_MAX_CHUNK_TOKENS=2048", **{k: v for k, v in judged_noobs.items() if k != "arms"}, "arm": judged_noobs["arms"]["A"]},
        "locomo_retrieval": {k: v for k, v in loco.items() if k != "arms"} | {"arm": loco["arms"]["A"]},
        "context_bench_and_harm_bench": "base-server baselines of Track 2 are in context-admission.json (arms C'+D tags+Jev, no gate, min_scores, similarity; harm bench none/ungated)",
        "performance": {"source": rel(PERF), "pg0_instance": "hindsight-perf-a (PostgreSQL 18.1.0)",
                        "recall_pooled_ms": {scen: {sh: perf[f"recall_{scen}"]["arms"]["A"][sh]["pooled"] for sh in ("seq", "conc")} for scen in ("noobs", "obs")},
                        "official_suites_medium": perf["official_suites"]["A"],
                        "fk_delete_base_schema": {k: perf["fk_delete"]["base"][k] for k in ("n", "rows", "sql_median_total_s", "engine_median_s")},
                        "harness_patch": "raw/base-harness-now-kwarg.patch (the 3-line now-kwarg fix from 60c1cc87f; the patched file is byte-identical to the feature's)"},
    })

    # ---------------- feature off (B)
    write("feature-off.json", {
        **meta("B", "feature branch, adaptive_memory_mode=off (and context_admission_mode=off)"),
        "equivalence_R37": {
            "golden_off_mode_gate": {"result": "byte-identical to the base 5331da2f4 transcript at every phase gate (P1-P5, T2, A68/R59 audit)", "test": "hindsight-system-tests/tests/test_99_golden_off_mode.py", "source": rel(T / "STATE.md")},
            "memory_lifecycle_rows_A_vs_off_P1": row_equiv(LB / "p1/compare-A-vs-off-rows.txt"),
            "memory_lifecycle_rows_A_vs_off_P3": row_equiv(LB / "p3/rows-A-vs-off.txt"),
            "amb_longmemeval": "A vs B: 0 discordant on every endpoint (see baseline.json amb_longmemeval_retrieval_noobs)",
            "locomo": "A vs B: 0/0 at k=5/10/20 (see baseline.json locomo_retrieval)",
            "known_off_mode_regression": "bulk deletes are slower in every mode, off included (the 10 new FKs). 50,000 units on pg0, SQL delete, median of 3: empty side tables 0.245 -> 1.115 s (4.55x, +17.4 us/row); base side rows (1.5M links, 150k unit_entities) 2.590 -> 3.975 s (1.53x, +27.7 us/row); engine delete_bank 2.917 -> 3.741 s (1.28x). P1's earlier 0.418 -> 1.867 s (4.47x) is reproduced in ratio. Not measured: --scale large, Oracle. See performance.json fk_delete",
            "recall_latency_B_vs_A": {"noobs": perf_rl["recall_noobs"]["B_vs_A"], "obs": perf_rl["recall_obs"]["B_vs_A"],
                                      "reading": "within the noise floor of base; the pooled conc p95 +8.8% [+1.0, +19.3] (noobs) is one of the implausible cluster-bootstrap results the verification flags (run-level t95 [-27.5, +28.1])",
                                      "source": rel(PERF)},
        },
        "memory_lifecycle": {"P1 base-delete": lc("p1/arm-off-base-delete.json"), "P3 base-delete": lc("p3/p3-off-base-delete.json"), "P3 base-update": lc("p3/p3-off-base-update.json")},
        "amb_longmemeval_retrieval_noobs": {**amb_common, "reference_arm": "B", "endpoints": endpoints(NOOBS, "B", "B")},
        "amb_longmemeval_retrieval_obs": {
            "dataset": "40 KU questions (eval/arms/obs-ku-qids.txt, sha256 7ba3a529...), each bank consolidated once with the base prompt (mode off) on P3 code, Haiku 5.5",
            "reference_arm": "B-obs",
            "B-obs": endpoints(OBS, "B-obs", "B-obs"),
            "P-obs (off + recall_prefer_observations=true)": endpoints(OBS, "P-obs", "B-obs"),
            "T-obs (off + types=['observation'])": endpoints(OBS, "T-obs", "B-obs"),
        },
        "amb_longmemeval_judged_noobs": {**{k: v for k, v in judged_noobs.items() if k != "arms"}, "arm": judged_noobs["arms"]["B"]},
        "amb_longmemeval_judged_obs": {**{k: v for k, v in judged_obs.items() if k != "arms"}, "arm": judged_obs["arms"]["B-obs"]},
        "locomo_retrieval": {k: v for k, v in loco.items() if k != "arms"} | {"arm": loco["arms"]["B"]},
        "classification_eval_arm_B": {
            "P3_K3": {"judged_pass": prop(73, 81), "source": rel(E / "classify/REPORT.md"), "failures": "environment-divergence 3/3 (rewrote 50 -> 5), conflict-same-day-gateway 2/3, conflict-same-day-date 1/3, employer-supersedes 1/3, cascade-to-group 1/3"},
            "P5_K1": {"judged_pass": prop(23, 27), "source": rel(E / "classify/raw-p5/score.md")},
            "tokens": {"P3": {"calls": 504, "input_tokens": 5633246, "output_tokens": 296821, "server_usd": 0.71}, "P5": {"calls": 169, "input_tokens": 1864911, "output_tokens": 97029, "server_usd": 0.235}},
        },
    })

    # ---------------- observe (C)
    write("observe.json", {
        **meta("C", "feature branch, adaptive_memory_mode=observe: compute retention/would-change, change nothing visible"),
        "memory_lifecycle": {"P1 base-delete": lc("p1/arm-observe-base-delete.json"), "note": "observe == off on every metric (compare p1/compare.md)", "compare": rel(LB / "p1/compare.md")},
        "amb_longmemeval_retrieval_noobs": {**amb_common, "reference_arm": "B", "endpoints": endpoints(NOOBS, "C", "B")},
        "judged": "not run judged (PREREGISTRATION amendment 2026-10-17 20:34: retrieval identical to B by construction, 0 discordant measured)",
        "tests": {
            "api": ["tests/test_adaptive_recall_integration.py::test_off_and_observe_return_the_same_recall",
                    "tests/test_context_admission_recall.py::test_observe_changes_nothing_visible_and_reports_the_shadow"],
            "system": ["hindsight-system-tests/tests/test_103_observe_mode.py (observe bank answers exactly like the off bank; consolidation delete leaves a shadow, no claim)"],
            "source": rel(E / "R59-CHECKLIST.md"),
        },
        "limitation": "observe does not run lifecycle classification (would change the prompt or need an extra LLM call); it shadows base-schema deletes only (DESIGN.md section 2, 13)",
        "performance": {**perf_arm("C"), "note": "observe computes R, the adaptive factor and the would-change top-k on every recall; round 1 was slow for C (+40% sequential p50) as well as B, cause unknown; without round 1 (exploratory) C's sequential p50 overhead is +10.2% vs D's +12.1% (eval/perf/REPORT.md)"},
    })

    # ---------------- rank (D)
    write("rank.json", {
        **meta("D", "feature branch, adaptive_memory_mode=rank: retention factor (alpha 0.1, floor 0.3) + reinforcement; no claim influence"),
        "memory_lifecycle": {"P1 base-delete": lc("p1/arm-rank-base-delete.json"), "P3 base-delete": lc("p3/p3-rank-base-delete.json"), "P3 base-update": lc("p3/p3-rank-base-update.json")},
        "amb_longmemeval_retrieval_noobs": {**amb_common, "reference_arm": "B", "endpoints": endpoints(NOOBS, "D", "B"),
                                            "primary_P1_rank": {"endpoint": "knowledge-update:latest_first", "p_exact_mcnemar": noobs_primary_p, "p_holm": holm_adj["P1-rank"], "significant": holm_adj["P1-rank"] < 0.05}},
        "decision_rule_branch_b": {
            "rule": "memory_lifecycle stale@1 improves with no endpoint regression beyond the margin (PREREGISTRATION.md)",
            "base_delete": "stale@1 0.222 -> 0.167 (all), 0.286 -> 0.143 (holdout)",
            "base_update": "stale@1 0.111 -> 0.111 (all), 0.143 -> 0.143 (holdout); every k=1 metric identical to off",
            "provenance": "the stale@1 criterion was written at ~09:45 after the P1 rank result on memory_lifecycle was known (p1/arm-rank-base-delete.json 05:46; STATE.md P1 entry; PREREGISTRATION header); it replaced DESIGN.md 11.5's stale_contamination@5 criterion, which D does not meet (0.233 = 0.233 all, 0.200 = 0.200 holdout)",
            "status": "non-confirmatory; branch b cannot establish opt-in rank whatever the perf item shows",
        },
        "amb_longmemeval_retrieval_obs": {"reference_arm": "B-obs", "D-obs (rank on base-consolidated banks)": endpoints(OBS, "D-obs", "B-obs")},
        "amb_longmemeval_judged_noobs": {**{k: v for k, v in judged_noobs.items() if k != "arms"}, "arm": judged_noobs["arms"]["D"]},
        "locomo_retrieval": {k: v for k, v in loco.items() if k != "arms"} | {"arm": loco["arms"]["D"],
                             "per_category_note": "regression concentrated in temporal (11 worse / 0 better) and open-domain (15/3); worse-at-5 queries have older gold items (source: STATE.md 2026-10-17 23:10)"},
        "performance": {**perf_arm("D"), "verified_verdict": perf_verified["rank_D_vs_B_p95_lt_5pct"], "decision_rule_item_c": perf_rule["track1_item_c_rank_p95_lt_5pct"]},
    })

    # ---------------- lifecycle (E)
    cls_p3 = json.loads((E / "classify/summary.json").read_text())
    write("lifecycle.json", {
        **meta("E", "feature branch, adaptive_memory_mode=lifecycle (rank + claim lifecycle, lifecycle consolidation prompt, current view by default)"),
        "memory_lifecycle": {
            "P3 base-delete": lc("p3/p3-lifecycle-base-delete.json"), "P3 base-update": lc("p3/p3-lifecycle-base-update.json"),
            "P5 base-delete (final split semantics; tree dirty at run time)": lc("p5/p5-lifecycle-base-delete.json"),
            "P5 base-update": lc("p5/p5-lifecycle-base-update.json"),
            "T2 base-delete (clean 2f4c7daae, admission off)": lc("t2/t2-lifecycle-base-delete.json"),
            "rows_P3_vs_P5": row_equiv(LB / "p5/rows-p3-vs-p5-base-delete.txt"),
            "rows_P5_vs_T2": row_equiv(LB / "t2/rows-p5-vs-t2-base-delete.txt"),
            "caveat": "author-scripted consolidation decisions (stub LLM): a conformance and robustness suite, not evidence about a real model (DESIGN.md 11.1)",
        },
        "amb_longmemeval_retrieval_noobs": {**amb_common, "reference_arm": "B", "endpoints": endpoints(NOOBS, "E", "B"),
                                            "note": "no observations means no claim state, so E == D on every endpoint"},
        "amb_longmemeval_retrieval_obs": {
            "dataset": "40 KU questions; each bank consolidated once with the lifecycle prompt (mode lifecycle) on P3 code, Haiku 5.5",
            "reference_arm": "B-obs",
            "E-obs (lifecycle, current view)": endpoints(OBS, "E-obs", "B-obs"),
            "E-obs-hist (lifecycle, historical view)": endpoints(OBS, "E-obs-hist", "B-obs"),
            "primary_P1_life": {"endpoint": "knowledge-update:latest_first", "p_exact_mcnemar": obs_primary_p, "p_holm": holm_adj["P1-life"], "significant": holm_adj["P1-life"] < 0.05,
                                "observation_source_repair": {
                                    "what": "shared_import_eval.py recorded empty source lists for every recalled observation; eval/arms/resolve_observations.py (mtime 2026-10-17 20:15 BRT) re-resolved them post-hoc against the still-running servers (resolve-cb.log: 20907 observation rows, 0 unresolved), keeping the originals in obs-cons-{base,life}/rows-unresolved/",
                                    "pre_specified_rule": "eval/amb-pilot.md (h), 04:17 BRT: observation -> the set of sessions of its source_memory_ids; score_endpoints.py docstring (09:44 BRT) scores an observation through its resolved source documents",
                                    "interim_look": "eval/arms/score-obs-interim/ (17:09 BRT, n=21, on the unrepaired rows): latest_first B-obs 0.952, E-obs 1.000, T-obs 0.000 on every endpoint",
                                    "on_rows_as_recorded": "score_endpoints.py on rows-unresolved (40 questions): latest_first B-obs 0.850, E-obs 0.900 (0/2, p=0.5), D-obs 0.875, P-obs 0.525, T-obs 0.000; not significant",
                                    "reading": "the pre-specified rule is applied, so P1-life holds mechanically, but its significance depends entirely on the post-hoc repair",
                                }},
            "exploratory_attribution_rescoring": {
                **json.loads((OUT / "derived/latest-first-attribution-exploratory.json").read_text()),
                "source": rel(OUT / "derived/latest-first-attribution-exploratory.json"),
                "reading": "post-hoc, not pre-registered. Measured: in 28/40 questions B-obs's top-1 is an observation citing both evidence sessions, which the as-scored tie rule counts as not latest-first; attributing an observation to its newest source session gives B-obs 0.925 vs E-obs 0.875 (2/0), raw facts only 0.850 vs 0.900 (0/2). Hypothesis (not measured): these are observations the base prompt updated in place; check: the consolidation /llm-requests trace or observation history for those 28 banks",
            },
        },
        "amb_longmemeval_judged_noobs": {**{k: v for k, v in judged_noobs.items() if k != "arms"}, "arm": judged_noobs["arms"]["E"],
                                         "provenance": JUDGED_E_PROVENANCE, "recall_check": judged_e_recall_check,
                                         "exploratory_E_vs_D": {"status": "exploratory: not a pre-registered comparison (the judged rule compares with B)", "both_correct": judged_e_vs_d["arms"]["E"]["both_correct"], "at_least_one": judged_e_vs_d["arms"]["E"]["at_least_one"]}},
        "amb_longmemeval_judged_obs": {**{k: v for k, v in judged_obs.items() if k != "arms"}, "arm": judged_obs["arms"]["E-obs"],
                                       "recall_check": "judged E-obs top-10 (type, document, mentioned_at) identical to the endpoint E-obs arm in 40/40 (r1) and 39/40 (r2); source eval/judged/judged-obs.md and STATE.md 2026-10-18 00:20"},
        "locomo_retrieval": {k: v for k, v in loco.items() if k != "arms"} | {"arm": loco["arms"]["E"]},
        "classification_eval": {
            "P3_K3": {
                "runs": 162, "scenarios": 27, "repeats": 3, "server_commit": "b491b2bb9",
                "false_retirement_controls_scenario_run": {"x": 0, "n": 51, "one_sided_95_upper": cls_p3["E_false_retirement_controls"]["one_sided_95_upper"], "ci95": cls_p3["E_false_retirement_controls"]["ci95"]},
                "false_retirement_controls_per_scenario": {"x": 0, "n": 17, "one_sided_95_upper": cls_p3["E_false_retirement_by_scenario"]["one_sided_95_upper"]},
                "control_runs_with_wrong_uncertain_label": {"x": 1, "n": 51, "one_sided_95_upper": cls_p3["E_control_runs_with_uncertain_protected"]["one_sided_95_upper"]},
                "supersede_recall": prop(54, 57), "supersede_precision": prop(60, 65),
                "refute_recall_strict": prop(1, 3), "refute_recall_either_label": prop(3, 3), "refute_precision": prop(2, 2),
                "conflict_recall": prop(6, 6), "conflict_precision": prop(6, 7),
                "raw_fact_leak": prop(0, 67), "judged_pass": prop(79, 81),
                "tokens": {"calls": 505, "input_tokens": 11352628, "output_tokens": 327071, "server_usd": 1.29, "input_ratio_vs_B": 2.0},
                "source": rel(E / "classify/REPORT.md"), "summary": rel(E / "classify/summary.json"),
            },
            "P5_K1": {
                "runs": 54, "server_code": "P5 working tree (schema $defs fix)",
                "false_retirement_controls": {"x": 0, "n": 17, "one_sided_95_upper": one_sided_upper(0, 17)},
                "supersede_recall": prop(18, 19), "supersede_precision": prop(20, 21), "judged_pass": prop(25, 27),
                "tokens": {"calls": 169, "input_tokens": 2834592, "output_tokens": 108371, "server_usd": 0.335, "input_ratio_vs_B": round(2834592 / 1864911, 3)},
                "source": rel(E / "classify/raw-p5/score.md"),
            },
            "pooled_P3_P5_false_retirement_controls": {"x": 0, "n": 68, "one_sided_95_upper": one_sided_upper(0, 68),
                                                       "label": "computed for this report; pooling two code versions is the author's judgment, not pre-registered"},
        },
        "hs_llm_core_judge_tests": {"result": "4/4 passed, single run (P3 worktree, server claude-haiku-5-5, judge claude-sonnet-5-5)", "source": rel(T / "STATE.md")},
        "performance": {**perf_arm("E"), "phases_note": "feature phases p50 / p95 ms (noobs): adaptive_state_load 1.7 / 2.4 seq, 42.5 / 302 conc; lifecycle_claim_load 2.5 / 3.5 seq, 44.7 / 304 conc (eval/perf/REPORT.md); the perf banks hold no retired claims",
                        "verified_verdict": perf_verified["lifecycle_E_vs_B_p95_lt_10pct"], "decision_rule_item_4": perf_rule["track1_item4_lifecycle_p95_lt_10pct"]},
    })

    # ---------------- ablations
    write("ablations.json", {
        **meta("ablations", "one arm per optional piece where one ran; 'not run' where none did"),
        "retention_decay_alpha": {
            "rule_status": "decay (alpha) is a frozen parameter, not one of the pre-registered optional pieces; applying the optional-piece keep rule to it is the author's extension. Pre-registered decay ablations: memory_lifecycle E alpha=0 and AMB E-obs-nodecay. LoCoMo E-nodecay is exploratory (added after the LoCoMo regression was seen, STATE.md 2026-10-17 23:10)",
            "memory_lifecycle_lifecycle_vs_nodecay": {"lifecycle": lc("p3/p3-lifecycle-base-delete.json")["metrics"], "lifecycle alpha=0": lc("p3/p3-lifecycle-nodecay-base-delete.json")["metrics"],
                                                      "note": "identical at k=5 on every metric; at k=1 decay gives supers@1 0.944 vs 0.889 and prec@1 0.948 vs 0.931 (one query each)"},
            "amb_noobs_rank_vs_off": "D vs B (no observations: the retention factor is the only active component): see rank.json; latest_first 0.819 -> 0.861 (0/3, p=0.25)",
            "amb_obs_E_vs_E_nodecay": {"reference_arm": "E-obs", "E-obs-nodecay": endpoints(OBS_REF_E, "E-obs-nodecay", "E-obs"),
                                       "derived_by": "python3 -I tasks/adaptive-memory/eval/arms/score_endpoints.py --runs obs-cons-base obs-cons-life --reference E-obs (rows unchanged; the B-obs-reference rerun reproduces eval/arms/score-obs/endpoints.json exactly)"},
            "amb_obs_vs_B_obs": {"reference_arm": "B-obs", "E-obs-nodecay": endpoints(OBS, "E-obs-nodecay", "B-obs")},
            "locomo": {"status": "exploratory: arm decided after the LoCoMo D/E regression was seen", "E-nodecay vs B": loco["arms"]["E-nodecay"], "E vs B": loco["arms"]["E"], "source_scored": rel(E / "judged/locomo-endpoints.md")},
        },
        "lifecycle_consolidation_vs_recall_filter": {"status": "exploratory: E-obs-off is not in PREREGISTRATION.md (defined in eval/arms/arms-cons-life.json at 14:16 BRT, before any obs-on data)",
                                                      "reference_arm": "B-obs", "E-obs-off (lifecycle-consolidated banks read in mode off)": endpoints(OBS, "E-obs-off", "B-obs"),
                                                      "vs_E_obs": endpoints(OBS_REF_E, "E-obs-off", "E-obs")},
        "historical_view_default": {"memory_lifecycle": lc("p3/p3-lifecycle-histview-base-delete.json"), "note": "default view historical: stale@5 0.411 all / 0.400 holdout, refuted@5 0, hist@5 1.000; the stale rise is by design (history is shown)"},
        "historical_cues": {"memory_lifecycle": lc("p3/p3-lifecycle-cues-base-delete.json"), "note": "identical to lifecycle: the bench sends explicit views, so the detector never decides; no false-positive/negative rates measured"},
        "existing_feature_baselines": {
            "memory_lifecycle": "baseline.json existing_feature_baselines (prefer_observations: stale@5 0.233 -> 0.352; types=observation: stale 0 but hist@5 0.125)",
            "amb_obs": {"P-obs": endpoints(OBS, "P-obs", "B-obs"), "T-obs": endpoints(OBS, "T-obs", "B-obs")},
        },
        "not_run": {
            "R55_last_write_wins (conflict_policy=latest)": "8 of 40 banks consolidated before the Anthropic credit ran out (4 failed, 28 not run); none used; not resumed by user decision (STATE.md 2026-10-18 02:03)",
            "R55_recency_decay_only_baseline": "not run; no reason recorded. Hypothesis: on banks without reinforcement events D reduces to a decay-only arm; check: count memory_reinforcement_events rows on a D bank",
            "archive / prune": "no benchmark arm; system story tests/test_101_retention_archive.py and api tests only",
            "recall_reinforcement": "no benchmark arm (default off)",
            "observe_shadow": "no benchmark arm; system story tests/test_103_observe_mode.py",
            "D_reinforcement_off": "pre-registered memory_lifecycle ablation; not run; no reason recorded",
            "C_observe_base_update": "pre-registered memory_lifecycle encoding for arm C; only p1/arm-observe-base-delete.json exists; no reason recorded",
        },
        "harm_bench_ungated_nopreamble": "see context-admission.json harm_bench.original (exploratory; preamble hypothesis refuted)",
    })

    # ---------------- performance
    write("performance.json", {
        **meta("performance", "R39/R40/A52 performance: a mirror of tasks/adaptive-memory/eval/perf/performance.json (written by eval/perf/scripts/analyze.py) plus the verified verdicts of eval/perf/REPORT.md, the run-level recomputation and the pre-perf measurements"),
        "mirror_of": rel(PERF), "mirror_of_sha256": sha256(PERF), "report": rel(PERF_REPORT), "report_sha256": sha256(PERF_REPORT),
        "verified_verdicts": perf_verified,
        "decision_rule_application": perf_rule,
        "run_level_recomputation": perf_rl,
        "analysis_plan_timing": "analysis_plan.written_at below is the plan's own text (05:20Z) and is not its write time: the file's mtime is 2026-10-18T05:13:48Z, "
                                "5 s before the first analysed run (recall/noobs/r1-A) began its machine-state check (05:13:53Z; cleared to start 05:14:33Z after a 40 s load wait) "
                                "and after the smoke run (05:09:34Z) and the medium pilot of arm E (05:10:29Z), both excluded (raw/machine_state.jsonl)",
        "criteria_note": "`criteria` is analyze.py's output as written, before the independent verification; read verified_verdicts instead",
        **perf,
        "fk_bulk_delete_cost_P1": {
            "measured_P1_pg0": {"rows": 50000, "base_s": 0.418, "feature_s_empty_side_tables": 1.867, "feature_s_cascade_fks_only": 0.988, "per_row_us": 29, "modes": "every mode including off"},
            "source": rel(T / "PLAN.md") + " (P6 bullet) and STATE.md P1 entry",
            "reproduced_by": "fk_delete above: 0.245 -> 1.115 s (4.55x) vs P1's 4.47x",
        },
        "not_measured": {
            "fk_delete_scale_large": "no script on file plans it (fk_delete_bench.py runs 50,000 units); +17-35 s per 1M-unit bank delete is a projection from the measured +17-33 us per deleted unit, not a measurement",
            "fk_delete_oracle": "not run: no Oracle on this machine (starting the 23ai container OOM-kills the user's MySQL container); fk_delete_bench.py has no Oracle path",
            "design_11_4_gate_form": "median of >= 5 runs at --scale large was not run; 4 pooled medium rounds replaced it without a declared deviation",
            "storage_growth_real_supersession": "claim_state, evidence, relations and shadow stayed empty in every perf run because the mock LLM retires nothing",
        },
        "r39_coverage": {
            "recall_p50_p95_p99": "measured: recall_noobs and recall_obs (pooled and per round, 10 / 5 arms, 4 rounds, 400 sequential + 200 concurrent recalls per run)",
            "retain_p50_p95_p99": "per-request percentiles not measured: the perf-test retain suite reports whole-batch duration and items/s (official_suites, 3 rounds; recall_obs arms retain_sync_s, 4 rounds)",
            "consolidation_p50_p95_p99": "per-request percentiles not measured: whole-round duration and memories/s for one round of 100 memories, mock LLM (official_suites; recall_obs arms consolidation_s)",
            "storage_growth": "measured for retain and for consolidation without retirements: table rows and bytes after retain and after consolidating 100 observations (recall_obs arms table_rows, table_bytes_round1); growth from real supersession not measured",
            "write_qps": "retain throughput only: items/s in official_suites (medians A 123.1, B 119.8, C 120.8, D 119.5, E 119.3)",
            "db_statements_per_recall": "not measured: neither eval/perf/scripts nor hindsight-dev/benchmarks/perf reads pg_stat_statements",
            "wal_bytes": "not measured: neither eval/perf/scripts nor hindsight-dev/benchmarks/perf collects it",
            "llm_calls_and_tokens": "measured: see llm_calls_and_tokens below",
        },
        "measured_under_load_indicative_only": {
            "admission_phase_ms_holdout": {a: json.loads((E / "context-bench/t2/holdout/holdout-primary.json").read_text())["arms"][a]["latency"] for a in ("E-det", "E-jev", "F")},
            "machine_load": "uptime load averages 53 / 68 / 101 at 2026-10-18 00:46 BRT (context-bench REPORT)",
            "mmr_redundancy_pass": {"candidates": 30, "ms_384d": 0.57, "ms_1536d": 1.56, "source": rel(T / "impl/T2-NOTES.md")},
            "t2_sanity_admission_phase_ms": "6-7 ms including the embedding read, one recall, 5 candidates (impl/T2-NOTES.md)",
            "harm_bench_gated_admission_phase_ms": {"original_median": 338.1, "original_max": 396.3, "h2_median": 353.6, "h2_max": 443.7, "source": rel(E / "harm-bench/REPORT.md")},
        },
        "llm_calls_and_tokens": {
            "recall": "0 generative LLM calls in every mode (tests/test_context_admission_invariants.py::test_no_recall_calls_a_generative_model_in_any_mode); Jev/hybrid admission makes 1 TypeSafe decision-model request per gated recall (quiet machine: phase p50 / p95 about 345 / 432 ms; holdout F under load 352 / 418 ms)",
            "perf_runs": {
                "anthropic": "no call",
                "typesafe_proxy_counted": "510 calls in the context_bench runs (perf-ctxb-hybrid 255 + perf-ctxb-jev 255), 0 errors (eval/perf/REPORT.md, analyze.py typesafe_calls)",
                "typesafe_uncounted_inprocess_hybrid_arm": hybrid_inprocess_decision_model_use(),
                "total_reading": "about 3,000 decision-model requests in all: 510 counted plus about 2,500 uncounted from the in-process hybrid arm; eval/perf/REPORT.md's '510 TypeSafe calls' leaves the in-process arm out",
            },
            "consolidation_lifecycle_vs_off": {
                "classification_P3": {"input_ratio": 2.0, "calls": "505 vs 504"},
                "classification_P5": {"input_ratio": round(2834592 / 1864911, 3), "calls": "169 vs 169", "per_call_input": "16.8k vs 11.0k"},
                "amb_obs_consolidation_P3": {"usd_per_bank_lifecycle": 0.62, "usd_per_bank_base": 0.40, "calls_per_bank": "~143 vs ~140", "source": rel(T / "STATE.md") + " 2026-10-17 15:35"},
                "schema_tokens_after_P5_hoist_local_tokenizer": {"pool_10_25_30": {"off": 1229, "lifecycle_before": 7075, "lifecycle_after": 2964}, "pool_10_50_120": {"off": 1229, "lifecycle_before": 16841, "lifecycle_after": 5810}, "source": rel(T / "impl/P5-NOTES.md")},
            },
        },
        "retain_timing_note": "one api-slim test (test_streaming_retain_with_the_budget_disabled_matches_the_budgeted_run) timed out at 300.43 s in the -n 8 full suite; alone 27.7 s branch vs 26.6 s base (n=4 each, spread overlaps) (impl/P5-NOTES.md)",
    })

    # ---------------- context admission (Track 2)
    hp = json.loads((E / "context-bench/t2/holdout/holdout-primary.json").read_text())
    chosen = json.loads((E / "context-bench/t2/chosen.json").read_text())
    cb = json.loads((E / "context-bench/summary.json").read_text())
    harm = json.loads((E / "harm-bench/summary.json").read_text())
    baselines = {fam: {"arm": arm, "dev": cb["arms"][arm]["dev"], "holdout": cb["arms"][arm]["holdout"]} for fam, arm in cb["selected"].items()}

    def harm_suite(name: str) -> dict:
        s = harm["suites"][name]
        return {"conditions": {c: {k: v for k, v in d.items() if k not in ("errors",)} for c, d in s["conditions"].items()},
                "comparisons": s["comparisons"], "downstream_rule": s["downstream_rule"],
                "exploratory": s.get("exploratory"), "reported_exploratory": s.get("reported_exploratory"), "suite_note": s.get("suite_note"),
                "changed_vs_ungated": s.get("changed_vs_ungated"), "spend_usd": s.get("spend_usd")}

    rule_items = []
    for arm in ("E-det", "E-jev", "F"):
        m = hp["arms"][arm]["metrics"]
        drop = round(hp["baseline"]["metrics"]["context_pollution_rate"] - m["context_pollution_rate"], 4)
        rule_items.append({
            "arm": arm,
            "pollution_le_0.20": {"value": round(m["context_pollution_rate"], 4), "holds": m["context_pollution_rate"] <= 0.20},
            "abstention_ge_0.80": {"value": m["abstention_accuracy"], "holds": m["abstention_accuracy"] >= 0.80},
            "useful_recall_ge_0.85": {"value": round(m["useful_memory_recall"], 4), "holds": m["useful_memory_recall"] >= 0.85},
            "opt_in_filter_pollution_drop_ge_0.15": {"value": drop, "holds": drop >= 0.15},
            "opt_in_filter_recall_ge_0.80": {"value": round(m["useful_memory_recall"], 4), "holds": m["useful_memory_recall"] >= 0.80},
            "endpoint_status": "exploratory (weights from dev passes 2-3)" if arm != "E-jev" else "pollution/precision/tokens exploratory (tie-break); recall/abstention/leakage confirmatory",
        })

    write("context-admission.json", {
        **meta("track2", "Context Admission Layer (T2 504ca72e9; evaluated on frozen worktree 2f4c7daae)"),
        "context_bench": {
            "scenario": "303 + 9 memories, 131 queries (85 dev / 46 holdout), 24 expect abstention (8 in holdout); injection cap 10",
            "baselines_base_server": baselines,
            "baselines_source": rel(E / "context-bench/summary.json"),
            "t2_dev_selection": {"chosen_sha256": "02b02eecca7408489dc2b45dfbb51b84b24beb1a463f33707256e64f5a510029", "selection_rule": chosen["selection_rule"], "deviations": chosen["deviations"],
                                 "arms": {a: {k: v for k, v in chosen["arms"][a].items() if k in ("scorer", "mode", "weights_source", "thresholds", "variant", "dev")} for a in chosen["arms"]},
                                 "source": rel(E / "context-bench/t2/chosen.json")},
            "t2_holdout": {"baseline_tags_plus_jev": hp["baseline"],
                           "arms": {a: {k: hp["arms"][a][k] for k in ("run", "metrics", "dev_metrics_at_choice", "intervals", "mcnemar_polluted_vs_baseline", "calibration", "latency", "replay_own_trace", "decision_model", "per_category")} for a in hp["arms"]},
                           "source": rel(E / "context-bench/t2/holdout/holdout-primary.json"),
                           "sweep_informational": rel(E / "context-bench/t2/holdout/holdout-sweep.json"),
                           "endpoint_status": "PREREGISTRATION amendment 2026-10-18 01:45: every E-det and F endpoint exploratory; E-jev pollution, precision, polluted queries, McNemar, tokens, admitted/q exploratory; E-jev recall, abstention, leakage, stale admission and calibration confirmatory"},
            "decision_rule_items": rule_items,
            "e_jev_dev_tied_settings_holdout_pollution": [0.307, 0.295, 0.276, 0.263],
            "lifecycle_off_admission_t2": row_equiv(LB / "t2/rows-p5-vs-t2-base-delete.txt"),
        },
        "harm_bench": {
            "original": harm_suite("original"), "h2": harm_suite("h2"),
            "models": "answer claude-haiku-5-5 (effort medium), judge claude-sonnet-5-5 (effort high)",
            "gated_setting": "arm F at its dev-chosen setting (hybrid, strict, auto 0.80 / explicit 0.85); exploratory (eval/harm-bench/exploratory.json)",
            "spend_usd_total": harm["spend_usd"], "source": rel(E / "harm-bench/summary.json"), "report": rel(E / "harm-bench/REPORT.md"),
        },
        "a68_checklist": rel(E / "A68-CHECKLIST.md"),
        "admission_latency_quiet_machine": {
            "source": rel(PERF), "report": rel(PERF_REPORT),
            "verified_verdict": perf_verified["admission_det_p95_le_150ms"],
            "decision_rule_item": perf_rule["track2_deterministic_admission_p95_le_150ms"],
            "context_bench_http": {run: {"api_sha": r["api_sha"], "api_dirty": r["api_dirty"], "typesafe_calls": r["typesafe_calls"], "variants": r["variants"]}
                                   for run, r in perf["admission_context_bench"].items()},
            "inprocess_perf_bank": perf["admission_inprocess"],
            "inprocess_phase_p95_per_round": perf_verified["admission_det_p95_le_150ms"]["secondary_inprocess_phase_p95_per_round"],
            "recall_overhead_vs_E_ext_noobs": {a: perf["recall_noobs"]["comparisons"][f"{a}_vs_E-ext"] for a in ADMISSION_ARMS},
            "recall_overhead_vs_E_ext_noobs_run_level": perf_rl["recall_noobs_admission_vs_E_ext"],
            "recall_overhead_reading": perf_verified["admission_det_p95_le_150ms"]["observe_recall_p95_overhead_c8"],
            "note": "E-ext = lifecycle recall as the HTTP endpoint does it (_external=True, purpose auto_context) with admission off; filter and strict return 5-6 results instead of 300, which is why they are faster than E-ext; the in-process hybrid arm abstained on 1761 of 2400 recalls because the perf bank has no scope metadata",
        },
    })

    print("wrote", sorted(p.name for p in OUT.glob("*.json")))
    print("holm", holm_adj)
    for a, e in judged_noobs["arms"].items():
        print("judged", a, e["both_correct"]["value"], e["at_least_one"]["value"], [r["value"] for r in e["per_repeat"]], e["both_correct"].get("ref_only"), e["both_correct"].get("arm_only"))
    for a, e in judged_obs["arms"].items():
        print("judged-obs", a, e["both_correct"]["value"], e["at_least_one"]["value"], [r["value"] for r in e["per_repeat"]])
    print("judged E vs D (exploratory)", judged_e_vs_d["arms"]["E"]["both_correct"], judged_e_vs_d["arms"]["E"]["at_least_one"])
    print("judged E recall check", [(c["top10_identical"], c["full_context_identical"]) for c in judged_e_recall_check["E_vs_D"]])
    print("perf verdicts", json.dumps(perf_rule, indent=1))
    for a, e in loco["arms"].items():
        print("locomo", a, e["gold@5"]["value"], e["gold@5"].get("ref_only"), e["gold@5"].get("arm_only"), e["gold@10"]["value"], e["gold@20"]["value"])


if __name__ == "__main__":
    main()
