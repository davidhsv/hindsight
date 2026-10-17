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

PROFILES = ("local", "jev")


def _variants(spec: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in spec.split(","):
        out.extend(PRESETS.get(part.strip(), [part.strip()]))
    return tuple(dict.fromkeys(out))


def _profile_env(profile: str) -> dict[str, str]:
    if profile == "local":
        return {}
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("--profile jev needs TYPESAFE_API_KEY in the environment")
    return {
        "HINDSIGHT_API_RERANKER_PROVIDER": "typesafe",
        "HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY": key,
        "HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES": "true",
    }


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
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    report_p = sub.add_parser("report", help="score every run file and write tables + REPORT.md")
    report_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    sub.add_parser("validate", help="check the scenario's integrity")

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
            typesafe_proxy=args.profile == "jev",
            send_context=args.send_context,
            max_tokens=args.max_tokens,
            pg0=args.pg0,
            reuse=args.reuse,
            query_ids=tuple(args.queries.split(",")) if args.queries else None,
        )
    )
    failed = sum(1 for row in record.rows if row.status != 200)
    print(f"{record.label}: {len(record.rows)} recalls ({failed} failed) in {record.duration_s}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
