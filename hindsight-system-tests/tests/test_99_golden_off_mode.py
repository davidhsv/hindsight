"""With the adaptive-memory feature off, the server says exactly what the baseline said.

Off is the default, so off is what every existing deployment gets on upgrade, and its
promise is byte-identical behaviour: the same recall payloads and traces, the same
consolidation prompt and JSON schema, the same MCP tool list, the same transfer
archive. No other story checks that, because no other story looks at all of it at
once. This one replays the golden scenario (``hindsight_system_tests.golden``) against
the server under test and compares the canonical transcript with one captured from the
base commit.

Skipped unless ``HINDSIGHT_SYSTEST_GOLDEN_DIR`` names the golden transcript to compare
against. Capture one from a base-commit worktree with::

    uv run python -m hindsight_system_tests.golden capture --out <dir> --api-dir <worktree>/hindsight-api-slim

then point this story at it::

    HINDSIGHT_SYSTEST_GOLDEN_DIR=<dir> uv run pytest tests/test_99_golden_off_mode.py

Run it on its own. The stub is shared by every server in the session, and a straggling
call from another story's server would land in this story's recorded LLM requests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from hindsight_system_tests import golden, start_hindsight_server

GOLDEN_DIR = os.environ.get("HINDSIGHT_SYSTEST_GOLDEN_DIR")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not GOLDEN_DIR, reason="HINDSIGHT_SYSTEST_GOLDEN_DIR names no golden transcript"),
]

#: Enough of the comparison to act on; the full report is written beside the transcript.
_REPORT_LIMIT = 20_000


@pytest.fixture(scope="module")
def golden_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """A server of this story's own, on a schema nothing else has written to.

    Not the session ``hindsight_server``: a capture is only deterministic on a fresh
    schema (see ``golden.fresh_schema``), and the session server's schema holds every
    other story's leftovers. Its worker claims only this schema's work.
    """
    log_path: Path = tmp_path_factory.mktemp("golden-server") / "server.log"
    server = start_hindsight_server(
        stub_url=stub_server.url, log_path=log_path, extra_env=golden.server_env(golden.fresh_schema())
    )
    yield server
    server.stop()


async def test_feature_off_reproduces_the_baseline(golden_server, stubs, tmp_path):
    expected_dir = Path(GOLDEN_DIR or "").expanduser().resolve()
    expected = golden.load(expected_dir)

    actual = golden.canonicalize(await golden.capture(golden_server.url, stubs))
    actual_dir = tmp_path / "actual"
    golden.write(actual_dir, actual, manifest={"server_log": str(golden_server.log_path)})

    differences = golden.compare(expected, actual)
    report = golden.report(str(expected_dir), str(actual_dir), differences)
    (tmp_path / "comparison.txt").write_text(report + "\n")

    hint = ""
    if tied := golden.ties(actual):
        hint = "\n\nThis capture has retrieval ties, so part of the difference may be tie order:\n  " + "\n  ".join(
            tied
        )
    assert not differences, f"{report[:_REPORT_LIMIT]}{hint}\n\nFull comparison: {tmp_path / 'comparison.txt'}"
