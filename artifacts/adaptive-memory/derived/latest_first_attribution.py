"""Exploratory, post-hoc: how much of KU obs-on latest_first depends on how an observation is attributed to sessions.

Variants per arm (same rows as eval/arms/score-obs, same tie rule: ties and absences are False):
- as_scored: an observation represents every session among its sources (score_endpoints.py rule)
- raw_only: observations ignored, raw facts only
- obs_newest_source: an observation represents only its newest-dated source session
Also: share of questions whose top-1 result is an observation citing both evidence sessions.
"""
import json, sys
from math import comb
from pathlib import Path

arms_dir = Path(sys.argv[1]); ds = sys.argv[2]; out = Path(sys.argv[3])
items = {d["question_id"]: d for d in json.load(open(ds))}

def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)

def sess(qid, d):
    p = f"{qid}_"
    return d[len(p):] if d.startswith(p) else d

res = {}
for run in ("obs-cons-base", "obs-cons-life"):
    for f in sorted((arms_dir / run / "rows").glob("*.json")):
        r = json.loads(f.read_text()); qid = r["question_id"]; it = items[qid]
        dates = dict(zip(it["haystack_session_ids"], it["haystack_dates"]))
        ans = sorted(it["answer_session_ids"], key=lambda s: dates.get(s, ""))
        earlier, latest = ans[0], ans[-1]
        for arm, rows in r["arms"].items():
            out_v = {}
            for variant in ("as_scored", "raw_only", "obs_newest_source"):
                first = {}
                for rank, row in enumerate(rows, 1):
                    if row["type"] == "observation":
                        if variant == "raw_only":
                            continue
                        srcs = {sess(qid, d) for d in (row.get("source_documents") or [])}
                        if variant == "obs_newest_source" and srcs:
                            srcs = {max(srcs, key=lambda s: dates.get(s, ""))}
                    else:
                        srcs = {sess(qid, row["document_id"])} if row.get("document_id") else set()
                    for s in srcs:
                        first.setdefault(s, rank)
                inf = 10**9
                out_v[variant] = first.get(latest, inf) < first.get(earlier, inf)
            top = rows[0] if rows else {}
            top_srcs = {sess(qid, d) for d in (top.get("source_documents") or [])}
            out_v["top1_obs_cites_both"] = top.get("type") == "observation" and {earlier, latest} <= top_srcs
            res.setdefault(arm, {})[qid] = out_v

summary = {"label": "EXPLORATORY, post-hoc re-scoring (not pre-registered); same rows as eval/arms/score-obs", "arms": {}}
ref = res["B-obs"]
for arm, qs in sorted(res.items()):
    summary["arms"][arm] = {}
    for v in ("as_scored", "raw_only", "obs_newest_source", "top1_obs_cites_both"):
        x = sum(q[v] for q in qs.values()); n = len(qs)
        e = {"x": x, "n": n, "rate": round(x / n, 4)}
        if arm != "B-obs" and v != "top1_obs_cites_both":
            b = sum(ref[k][v] and not qs[k][v] for k in qs); c = sum(qs[k][v] and not ref[k][v] for k in qs)
            e.update({"vs_B_obs_ref_only": b, "vs_B_obs_arm_only": c, "p_exact_mcnemar": mcn(b, c)})
        summary["arms"][arm][v] = e
out.write_text(json.dumps(summary, indent=1) + "\n")
for arm, d in summary["arms"].items():
    print(arm, {v: (e["rate"], e.get("vs_B_obs_ref_only"), e.get("vs_B_obs_arm_only"), e.get("p_exact_mcnemar")) for v, e in d.items()})
