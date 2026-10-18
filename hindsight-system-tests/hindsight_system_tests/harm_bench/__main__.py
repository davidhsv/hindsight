"""CLI: ``python -m hindsight_system_tests.harm_bench {run,report,validate}``.

Examples (from ``hindsight-system-tests/``, with ANTHROPIC_API_KEY in the environment)::

    # No memory + ungated top-10 against a base checkout, on a pg0 instance of its own.
    uv run python -m hindsight_system_tests.harm_bench run --label base \\
        --conditions none,ungated --api-dir /path/to/base/hindsight-api-slim

    # The gated condition against a feature checkout (strict admission is the default for gated).
    uv run python -m hindsight_system_tests.harm_bench run --label feat-gated \\
        --conditions gated --api-dir /path/to/feature/hindsight-api-slim \\
        --set HINDSIGHT_API_CONTEXT_ADMISSION_MODE=strict

    # Gated with the Jev (TypeSafe) decision model, TYPESAFE_API_KEY from the environment, its
    # requests counted by the usage-recording proxy; scorer, weights and thresholds via --set.
    uv run python -m hindsight_system_tests.harm_bench run --label feat-gated --profile jev-admission \\
        --conditions gated --api-dir /path/to/feature/hindsight-api-slim \\
        --set HINDSIGHT_API_CONTEXT_ADMISSION_SCORER=hybrid --set ...

    # Exploratory: the ungated recalls of run "base" injected without the integration's preamble.
    uv run python -m hindsight_system_tests.harm_bench run --label base-nopreamble \\
        --conditions ungated-nopreamble --reuse-recalls <out>/runs/base.json --budget-usd 1

    # Exploratory H2 suite (under-specified traps + controls) on its own bank (original + H2 memories).
    uv run python -m hindsight_system_tests.harm_bench run --label base-h2 --suite h2 \\
        --conditions none,ungated --api-dir /path/to/base/hindsight-api-slim --budget-usd 1.5

    # Build the bank and record the recalls only (no model calls, no spend).
    uv run python -m hindsight_system_tests.harm_bench run --label probe --conditions ungated \\
        --api-dir /path/to/base/hindsight-api-slim --recall-only

    uv run python -m hindsight_system_tests.harm_bench report
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

from .runner import (
    DEFAULT_ANSWER,
    DEFAULT_BUDGET_USD,
    DEFAULT_JUDGE,
    DEFAULT_PG0,
    ENV_ADMISSION_MODE,
    GATED_MODE,
    RunConfig,
    check_conditions,
    run,
)
from .suites import SUITES, validate_all

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "tasks" / "adaptive-memory" / "eval" / "harm-bench"

PROFILES = ("local", "jev-admission")


def _profile_env(profile: str) -> dict[str, str]:
    """``jev-admission``: TypeSafe as the context-admission decision model (the Jev scorer), its
    key read from ``TYPESAFE_API_KEY`` so it never appears on a command line; recall keeps the
    local reranker. The same profile context_bench's admission arms ran under."""
    if profile == "local":
        return {}
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit(f"--profile {profile} needs TYPESAFE_API_KEY in the environment")
    return {
        "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_PROVIDER": "typesafe",
        "HINDSIGHT_API_CONTEXT_ADMISSION_DECISION_MODEL_API_KEY": key,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harm_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run conditions against one server build")
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--suite", choices=list(SUITES), default="original", help="h2 is exploratory (see h2.py)")
    run_p.add_argument(
        "--conditions", default="none,ungated", help="comma list of none, ungated, ungated-nopreamble, gated"
    )
    run_p.add_argument("--api-dir", type=Path, help="a hindsight-api-slim checkout (synced venv); memory conditions")
    run_p.add_argument(
        "--set",
        "--server-env",
        dest="server_env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"server env; gated defaults {ENV_ADMISSION_MODE}={GATED_MODE}",
    )
    run_p.add_argument(
        "--profile", choices=PROFILES, default="local", help="jev-admission: TypeSafe decision model via the proxy"
    )
    run_p.add_argument("--repeats", type=int, default=2)
    run_p.add_argument("--top-k", type=int, default=10)
    run_p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    run_p.add_argument("--concurrency", type=int, default=6)
    run_p.add_argument("--answer-effort", default=DEFAULT_ANSWER.effort)
    run_p.add_argument("--judge-effort", default=DEFAULT_JUDGE.effort)
    run_p.add_argument("--pg0", default=DEFAULT_PG0, help="pg0 instance:port of this benchmark's own")
    run_p.add_argument("--tasks", help="comma list of task ids (default: all 24)")
    run_p.add_argument("--resume", type=Path, help="a partial run file to continue")
    run_p.add_argument(
        "--reuse-recalls",
        type=Path,
        help="inject the recalls recorded in this run file instead of starting a server (same bank and top-k)",
    )
    run_p.add_argument("--recall-only", action="store_true", help="record recalls, make no model call")
    run_p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    report_p = sub.add_parser("report", help="score run files and write summary.json + REPORT.md")
    report_p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    report_p.add_argument("--runs", help="comma list of run files (default: every run under <out>/runs)")

    sub.add_parser("validate", help="check the scenario's integrity")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "validate":
        problems = validate_all()
        for problem in problems:
            print(problem)
        return 1 if problems else 0

    if args.command == "report":
        from .scoring import write_report

        runs = [Path(p) for p in args.runs.split(",")] if args.runs else None
        summary = write_report(args.out, runs)
        print(f"spend ${summary['spend_usd']:.3f}; wrote {args.out / 'REPORT.md'}")
        return 0

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    server_env: dict[str, str] = _profile_env(args.profile)
    for item in args.server_env:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        server_env[key] = value
    if "gated" in conditions:
        server_env.setdefault(ENV_ADMISSION_MODE, GATED_MODE)
    try:
        check_conditions(conditions, server_env)
    except ValueError as e:
        parser.error(str(e))
    if any(c != "none" for c in conditions) and args.api_dir is None and args.reuse_recalls is None:
        parser.error("memory conditions need --api-dir (or --reuse-recalls)")

    record = run(
        RunConfig(
            label=args.label,
            suite=args.suite,
            conditions=conditions,  # type: ignore[arg-type]
            out_dir=args.out,
            api_dir=args.api_dir.expanduser().resolve() if args.api_dir else None,
            server_env=server_env,
            repeats=args.repeats,
            top_k=args.top_k,
            pg0=args.pg0,
            budget_usd=args.budget_usd,
            concurrency=args.concurrency,
            answer=dataclasses.replace(DEFAULT_ANSWER, effort=args.answer_effort or None),
            judge=dataclasses.replace(DEFAULT_JUDGE, effort=args.judge_effort or None),
            task_ids=tuple(args.tasks.split(",")) if args.tasks else None,
            resume=args.resume,
            reuse_recalls=args.reuse_recalls,
            recall_only=args.recall_only,
            typesafe_proxy=args.profile == "jev-admission",
        )
    )
    failed = sum(1 for row in record.answers if row.verdict is None)
    print(
        f"{record.label}: {len(record.recalls)} recalls, {len(record.answers)} answers ({failed} failed), "
        f"${record.spend_usd:.3f}{' — stopped at budget' if record.stopped_for_budget else ''}"
    )
    return 1 if failed or record.stopped_for_budget else 0


if __name__ == "__main__":
    sys.exit(main())
