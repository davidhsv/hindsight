"""``python -m hindsight_system_tests.lifecycle_bench {run,report,compare}``.

    # arm A, real local models, both base encodings
    uv run python -m hindsight_system_tests.lifecycle_bench run --arm A \\
        --api-dir ../hindsight-base/hindsight-api-slim --encoding base-delete --out runs/arm-A-base-delete.json

    # a feature arm is an env set plus an api dir
    uv run python -m hindsight_system_tests.lifecycle_bench run --arm E \\
        --set ADAPTIVE_MEMORY_MODE=lifecycle --encoding base-delete --out runs/arm-E-base-delete.json

    uv run python -m hindsight_system_tests.lifecycle_bench report runs/arm-A-base-delete.json
    uv run python -m hindsight_system_tests.lifecycle_bench compare runs/*.json --split holdout
    uv run python -m hindsight_system_tests.lifecycle_bench compare runs/a.json runs/b.json --rows
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import diff_rows, render_compare, render_diff, render_run
from .runner import DEFAULT_PG0_INSTANCE, DEFAULT_PG0_PORT, RunConfig, default_api_dir, load_record, run, write_record
from .scenarios import validate


@dataclass(frozen=True)
class Setting:
    key: str
    value: str


def _key_value(raw: str) -> Setting:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return Setting(key=key.strip(), value=value)


def _env_name(key: str) -> str:
    """``ADAPTIVE_MEMORY_MODE`` → ``HINDSIGHT_API_ADAPTIVE_MEMORY_MODE``; a name that already
    carries a prefix (or names another tool, ``HF_*``) is kept as given."""
    if key.startswith(("HINDSIGHT_", "HF_", "UV_", "TRANSFORMERS_", "TOKENIZERS_")):
        return key
    return f"HINDSIGHT_API_{key.upper()}"


def _config_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hindsight_system_tests.lifecycle_bench")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="run every scenario against one arm and write the run JSON")
    run_cmd.add_argument("--arm", required=True, help="arm label recorded in the run (A, B, C, ...)")
    run_cmd.add_argument("--label", help="run label (default: arm-<arm>-<encoding>[-lexical])")
    run_cmd.add_argument("--api-dir", type=Path, default=default_api_dir(), help="hindsight-api-slim checkout to run")
    run_cmd.add_argument("--encoding", choices=("base-update", "base-delete"), default="base-delete")
    run_cmd.add_argument(
        "--lifecycle-form",
        choices=("auto", "update", "create"),
        default="auto",
        help="lifecycle shape when the schema advertises it; auto pairs base-update→update, base-delete→create",
    )
    run_cmd.add_argument("--set", dest="env", type=_key_value, action="append", default=[], metavar="KEY=VALUE")
    run_cmd.add_argument(
        "--bank-config", type=_key_value, action="append", default=[], metavar="KEY=VALUE", help="per-bank config"
    )
    run_cmd.add_argument("--types", help="comma-separated recall types override, e.g. observation")
    run_cmd.add_argument("--lexical", action="store_true", help="stub lexical embedder/reranker (no-torch smoke)")
    run_cmd.add_argument("--fixtures", type=Path, help="frozen real-model consolidation fixtures directory")
    run_cmd.add_argument("--scenarios", help="comma-separated scenario ids (default: all)")
    run_cmd.add_argument("--pg0-instance", default=DEFAULT_PG0_INSTANCE)
    run_cmd.add_argument("--pg0-port", type=int, default=DEFAULT_PG0_PORT)
    run_cmd.add_argument("--out", type=Path, required=True, help="run JSON path (a .md report is written beside it)")
    run_cmd.add_argument("--keep-banks", action="store_true")
    run_cmd.add_argument("--keep-going", action="store_true", help="record an invalid scenario and continue")

    report_cmd = commands.add_parser("report", help="markdown report of one or more runs")
    report_cmd.add_argument("runs", nargs="+", type=Path)
    report_cmd.add_argument("--out", type=Path)

    compare_cmd = commands.add_parser("compare", help="key metrics across runs, or a row diff of two runs")
    compare_cmd.add_argument("runs", nargs="+", type=Path)
    compare_cmd.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    compare_cmd.add_argument("--k", type=int, choices=(1, 3, 5, 10), default=5)
    compare_cmd.add_argument("--rows", action="store_true", help="diff the per-query rows of exactly two runs")
    compare_cmd.add_argument("--out", type=Path)
    return parser


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        sys.stdout.write(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # One line per HTTP request drowns the per-scenario progress lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "run":
        problems = validate()
        if problems:
            sys.stderr.write("scenario catalogue is invalid:\n" + "\n".join(f"  {p}" for p in problems) + "\n")
            return 2
        form = args.lifecycle_form
        if form == "auto":
            form = "update" if args.encoding == "base-update" else "create"
        label = args.label or f"arm-{args.arm}-{args.encoding}" + ("-lexical" if args.lexical else "")
        config = RunConfig(
            label=label,
            arm=args.arm,
            api_dir=args.api_dir.expanduser().resolve(),
            encoding=args.encoding,
            lifecycle_form=form,
            lexical=args.lexical,
            server_env={_env_name(s.key): s.value for s in args.env},
            bank_config={s.key: _config_value(s.value) for s in args.bank_config},
            recall_types=tuple(t.strip() for t in args.types.split(",")) if args.types else None,
            fixtures=args.fixtures,
            scenario_ids=tuple(s.strip() for s in args.scenarios.split(",")) if args.scenarios else None,
            pg0_instance=args.pg0_instance,
            pg0_port=args.pg0_port,
            log_dir=args.out.parent / "logs" / label,
            keep_banks=args.keep_banks,
            keep_going=args.keep_going,
        )
        record = run(config)
        write_record(record, args.out)
        args.out.with_suffix(".md").write_text(render_run(record))
        sys.stdout.write(render_compare([record]))
        return 1 if record.invalid_scenarios else 0

    records = [load_record(path) for path in args.runs]
    if args.command == "report":
        _emit("\n".join(render_run(record) for record in records), args.out)
        return 0

    if args.rows:
        if len(records) != 2:
            sys.stderr.write("--rows compares exactly two runs\n")
            return 2
        _emit(render_diff(records[0], records[1]), args.out)
        return 0 if diff_rows(records[0], records[1]).deterministic else 1
    _emit(render_compare(records, split=args.split, k=args.k), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
