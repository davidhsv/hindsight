"""CLI: ``python -m hindsight_system_tests.protection_bench {run,report,validate}``.

From ``hindsight-system-tests/``, with the keys in the environment (``set -a; source <secrets>``)::

    # P1 BEFORE: plain Hindsight, defaults (no key needed).
    uv run python -m hindsight_system_tests.protection_bench run --suite p1 --side before \\
        --label p1-before --api-dir $BASE/hindsight-api-slim

    # P1 AFTER: the personal profile plus the injection gate and the Jev rerank (TYPESAFE_API_KEY);
    # served, gate-off and gate-only passes.
    uv run python -m hindsight_system_tests.protection_bench run --suite p1 --side after \\
        --label p1-after --api-dir $FEAT/hindsight-api-slim --profile personal

    # P2 AFTER: the same server with every TypeSafe base URL pointed at a dead port.
    uv run python -m hindsight_system_tests.protection_bench run --suite p2 --side after \\
        --label p2-after-dead --api-dir $FEAT/hindsight-api-slim --profile personal --dead-endpoint

    # P3, both sides (ANTHROPIC_API_KEY; consolidation by claude-haiku-5-5, K=1).
    uv run python -m hindsight_system_tests.protection_bench run --suite p3 --side before \\
        --label p3-before --api-dir $BASE/hindsight-api-slim
    uv run python -m hindsight_system_tests.protection_bench run --suite p3 --side after \\
        --label p3-after --api-dir $FEAT/hindsight-api-slim --profile personal

    uv run python -m hindsight_system_tests.protection_bench report --markdown /path/to/REPORT.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .data import validate
from .runner import (
    DEFAULT_ANTHROPIC_CAP_USD,
    DEFAULT_JEV_CAP,
    DEFAULT_OUT,
    PASSES,
    SIDES,
    SUITES,
    BenchmarkError,
    RunConfig,
    data_sha256,
    failed_recalls,
    frozen_sha256,
    run,
)


def _parse_env(parser: argparse.ArgumentParser, items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="protection_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one suite against one server build")
    run_p.add_argument("--suite", choices=SUITES, required=True)
    run_p.add_argument("--side", choices=SIDES, required=True, help="before: plain Hindsight; after: the branch")
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--api-dir", type=Path, required=True, help="a hindsight-api-slim checkout with a synced venv")
    run_p.add_argument(
        "--profile",
        choices=("none", "personal"),
        default="none",
        help="personal: the documented personal profile + injection gate + Jev rerank (key from TYPESAFE_API_KEY)",
    )
    run_p.add_argument("--set", dest="server_env", action="append", default=[], metavar="KEY=VALUE")
    run_p.add_argument("--passes", help=f"p1/p2: comma list of {', '.join(PASSES)} (default by suite and side)")
    run_p.add_argument(
        "--dead-endpoint", action="store_true", help="p2: point every TypeSafe base URL at a port nothing listens on"
    )
    run_p.add_argument("--pg0-name", help="pg0 instance of this run (default hindsight-protbench-<suite>-<side>)")
    run_p.add_argument("--queries", help="p1/p2: comma list of query ids (a smoke subset; the protocol runs all)")
    run_p.add_argument("--scenarios", help="p3: comma list of scenario ids (default all)")
    run_p.add_argument("--jev-cap", type=int, default=DEFAULT_JEV_CAP, help="TypeSafe requests over every run file")
    run_p.add_argument(
        "--anthropic-cap-usd", type=float, default=DEFAULT_ANTHROPIC_CAP_USD, help="P3 spend over every run file"
    )
    run_p.add_argument("--resume", action="store_true", help="p3: continue <label>.partial.json")
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    report_p = sub.add_parser("report", help="score the finished runs: summary.json, Markdown on stdout")
    report_p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    report_p.add_argument("--runs", help="comma list of run files (default: every finished run under <out>/runs)")
    report_p.add_argument("--markdown", type=Path, help="also write the Markdown to this file")

    sub.add_parser("validate", help="check the data's integrity and its frozen hash")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "validate":
        problems = validate()
        frozen = frozen_sha256()
        if frozen != data_sha256():
            problems.append(f"data.py hashes {data_sha256()}, the frozen hash is {frozen}")
        for problem in problems:
            print(problem)
        if not problems:
            print(f"data ok, sha256 {data_sha256()} (frozen)")
        return 1 if problems else 0

    if args.command == "report":
        from .report import write_report

        runs = [Path(p) for p in args.runs.split(",")] if args.runs else None
        print(write_report(args.out, runs, args.markdown))
        return 0

    config = RunConfig(
        suite=args.suite,
        side=args.side,
        label=args.label,
        api_dir=args.api_dir.expanduser().resolve(),
        out_dir=args.out,
        profile=args.profile,
        server_env=_parse_env(parser, args.server_env),
        passes=tuple(p.strip() for p in args.passes.split(",") if p.strip()) if args.passes else None,
        dead_endpoint=args.dead_endpoint,
        pg0_name=args.pg0_name,
        query_ids=tuple(args.queries.split(",")) if args.queries else None,
        scenario_ids=tuple(args.scenarios.split(",")) if args.scenarios else None,
        jev_cap=args.jev_cap,
        anthropic_cap_usd=args.anthropic_cap_usd,
        resume=args.resume,
    )
    try:
        record = run(config)
    except (ValueError, BenchmarkError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    failed = failed_recalls(record)
    errored = [row.scenario_id for row in record.scenarios if row.error]
    print(
        f"{record.label}: {len(record.recalls)} recalls ({len(failed)} failed), {len(record.scenarios)} scenarios "
        f"({len(errored)} errored), {record.jev_requests} TypeSafe requests, ${record.spend_usd:.4f}"
        f"{' — STOPPED: ' + record.stopped if record.stopped else ''}"
    )
    return 1 if failed or errored or record.stopped else 0


if __name__ == "__main__":
    sys.exit(main())
