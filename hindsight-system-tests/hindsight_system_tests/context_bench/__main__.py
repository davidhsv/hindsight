"""CLI: ``python -m hindsight_system_tests.context_bench {run,report,validate}``.

Examples (from ``hindsight-system-tests/``)::

    # Baseline arms A/B/C/C'/C'+C against a base checkout, on a pg0 instance of its own.
    uv run python -m hindsight_system_tests.context_bench run --label base-local \\
        --api-dir /path/to/base/hindsight-api-slim --variants local-baselines

    # Jev (TypeSafe reranker, prune_candidates=true) on the same banks; needs TYPESAFE_API_KEY.
    uv run python -m hindsight_system_tests.context_bench run --label base-jev --profile jev \\
        --api-dir /path/to/base/hindsight-api-slim --variants jev --reuse <out>/runs/base-local.json

    # A feature server: send the request context + purpose and set its mode.
    uv run python -m hindsight_system_tests.context_bench run --label feat-filter \\
        --api-dir /path/to/feature/hindsight-api-slim --variants plain,tags --send-context \\
        --server-env HINDSIGHT_API_CONTEXT_ADMISSION_MODE=filter

    uv run python -m hindsight_system_tests.context_bench report

    # Context-admission arms (dev split only): a record run, the offline fit, live checks.
    uv run python -m hindsight_system_tests.context_bench run --label t2-record --profile jev-admission \\
        --api-dir /path/to/feature/hindsight-api-slim --send-context --split dev --dump-embeddings \\
        --variants 'adm=strict+ta=0.60+te=0.35' --out <out>/t2 --server-env ...
    uv run python -m hindsight_system_tests.context_bench admission-fit \\
        --record <out>/t2/runs/t2-record.json --variant 'adm=strict+ta=0.60+te=0.35'
    uv run python -m hindsight_system_tests.context_bench admission-verify --live <out>/t2/runs/<live>.json \\
        --variant <variant> [--record <record> --record-variant <variant>]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .runner import DEFAULT_PG0, RunConfig, run, sweep_variants
from .scenarios import validate

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "tasks" / "adaptive-memory" / "eval" / "context-bench"

PRESETS = {
    "local-baselines": [
        "plain",
        "tags",
        *sweep_variants("plain", "final"),
        *sweep_variants("plain", "reranker"),
        *sweep_variants("tags", "final"),
        # Last, so the repeat is separated from the first pass by the whole run.
        "plain#2",
    ],
    "jev": ["plain", "tags", "plain#2"],
}

PROFILES = ("local", "jev", "jev-admission")


def _variants(spec: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in spec.split(","):
        out.extend(PRESETS.get(part.strip(), [part.strip()]))
    return tuple(dict.fromkeys(out))


def _profile_env(profile: str) -> dict[str, str]:
    """``jev``: TypeSafe as recall's reranker (pruning). ``jev-admission``: TypeSafe as the
    context-admission decision model (the Jev scorer); recall keeps the local reranker. Both go
    through the usage-recording proxy."""
    if profile == "local":
        return {}
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit(f"--profile {profile} needs TYPESAFE_API_KEY in the environment")
    if profile == "jev-admission":
        return {
            "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER": "typesafe",
            "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_API_KEY": key,
        }
    return {
        "HINDSIGHT_API_RERANKER_PROVIDER": "typesafe",
        "HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY": key,
        "HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES": "true",
    }


def _query_ids(queries: str | None, split: str | None) -> tuple[str, ...] | None:
    from .scenarios import QUERIES, split_of

    ids = tuple(queries.split(",")) if queries else None
    if split is None:
        return ids
    chosen = tuple(q.id for q in QUERIES if split_of(q.id) == split and (ids is None or q.id in ids))
    if ids is not None and len(chosen) != len(ids):
        sys.exit(f"--queries names ids outside the {split} split")
    return chosen


def _admission(args: argparse.Namespace) -> int:
    import json

    from .admission_dev import run_dev_fit, verify_live, write_section
    from .report import insert_t2_section

    out: Path = args.out
    if args.command == "admission-fit":
        chosen = run_dev_fit(args.record, args.variant, out, l2=args.l2)
        for name, arm in chosen["arms"].items():
            dev = arm["dev"]
            print(
                f"{name}: {arm['weights_source']} auto {arm['thresholds']['auto']:.2f} explicit "
                f"{arm['thresholds']['explicit']:.2f} -> recall {dev['useful_memory_recall']}, pollution "
                f"{dev['context_pollution_rate']}, abstention {dev['abstention_accuracy']}"
            )
        print((out / "chosen.sha256").read_text().strip())
        return 0
    if args.command == "admission-verify":
        result = verify_live(args.live, args.variant, record_path=args.record, record_variant=args.record_variant)
        target = out / "verify" / f"{result['live']['label']}__{_slug(args.variant)}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=1))
        print(
            json.dumps(
                {k: result[k] for k in ("own_trace", "dev_live")} | {"from_record": result.get("from_record")}, indent=1
            )
        )
        cross = result.get("from_record") or {}
        cross_ok = cross.get("all_exact", True) or not cross.get("exactness_expected", True)
        return 0 if result["own_trace"]["all_exact"] and cross_ok else 1
    verifications = [json.loads(path.read_text()) for path in sorted((out / "verify").glob("*.json"))]
    section = write_section(out, verifications=verifications)
    insert_t2_section(args.report, section)
    return 0


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run variants against one server build")
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--api-dir", type=Path, required=True, help="a hindsight-api-slim checkout (synced venv)")
    run_p.add_argument("--variants", default="local-baselines", help=f"comma list; presets: {', '.join(PRESETS)}")
    run_p.add_argument("--profile", choices=PROFILES, default="local")
    run_p.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE")
    run_p.add_argument("--send-context", action="store_true", help="send the request context and purpose fields")
    run_p.add_argument("--max-tokens", type=int, default=4096)
    run_p.add_argument("--pg0", default=DEFAULT_PG0, help="pg0 instance:port of this benchmark's own")
    run_p.add_argument("--reuse", type=Path, help="a run file whose schema and banks to query instead of ingesting")
    run_p.add_argument("--queries", help="comma list of query ids (default: all)")
    run_p.add_argument("--split", choices=("dev", "holdout"), help="only this split's queries")
    run_p.add_argument("--trace", action="store_true", help="ask every recall for a trace")
    run_p.add_argument(
        "--dump-embeddings", action="store_true", help="save the banks' stored embeddings (offline admission replay)"
    )
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    report_p = sub.add_parser("report", help="score every run file and write tables + REPORT.md")
    report_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    sub.add_parser("validate", help="check the scenario's integrity")

    t2_out = DEFAULT_OUT / "t2"
    fit_p = sub.add_parser("admission-fit", help="dev-only: fit, sweep and select the admission arms")
    fit_p.add_argument("--record", type=Path, required=True, help="a record run (admission variant with traces)")
    fit_p.add_argument("--variant", required=True)
    fit_p.add_argument("--l2", type=float, default=1.0)
    fit_p.add_argument("--out", type=Path, default=t2_out)

    verify_p = sub.add_parser("admission-verify", help="dev-only: replay a live admission run against itself")
    verify_p.add_argument("--live", type=Path, required=True)
    verify_p.add_argument("--variant", required=True)
    verify_p.add_argument("--record", type=Path, help="also replay this record run under the live settings")
    verify_p.add_argument("--record-variant")
    verify_p.add_argument("--out", type=Path, default=t2_out)

    section_p = sub.add_parser("admission-section", help="render the T2 dev section into REPORT.md")
    section_p.add_argument("--out", type=Path, default=t2_out)
    section_p.add_argument("--report", type=Path, default=DEFAULT_OUT / "REPORT.md")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "validate":
        problems = validate()
        for problem in problems:
            print(problem)
        return 1 if problems else 0

    if args.command == "report":
        from .report import write_report

        write_report(args.out)
        return 0

    if args.command in ("admission-fit", "admission-verify", "admission-section"):
        return _admission(args)

    server_env = _profile_env(args.profile)
    for item in args.server_env:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--server-env expects KEY=VALUE, got {item!r}")
        server_env[key] = value
    record = run(
        RunConfig(
            label=args.label,
            api_dir=args.api_dir.expanduser().resolve(),
            variants=_variants(args.variants),
            out_dir=args.out,
            server_env=server_env,
            typesafe_proxy=args.profile in ("jev", "jev-admission"),
            send_context=args.send_context,
            max_tokens=args.max_tokens,
            pg0=args.pg0,
            reuse=args.reuse,
            query_ids=_query_ids(args.queries, args.split),
            trace=args.trace,
            dump_embeddings=args.dump_embeddings,
        )
    )
    failed = sum(1 for row in record.rows if row.status != 200)
    print(f"{record.label}: {len(record.rows)} recalls ({failed} failed) in {record.duration_s}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
