# Hindsight system tests

Blackbox tests over a real `hindsight-api` process, driven **only** through the
published Python client. No engine imports, no SQL, no internals.

## Why this package exists

`hindsight-api-slim/tests/` holds ~500 files, each covering one mechanism. That
catches mechanism bugs. It does not catch *composition* bugs — consolidation
wiping facts, a delta refresh missing a backdated window, a transfer dropping
evidence — because no single-mechanism test spans the steps where those live.
These tests do.

## How determinism works

A stub server implements the OpenAI chat-completions and embeddings APIs plus a
Cohere-compatible rerank endpoint. The server under test is pointed at it with
ordinary environment variables, so **no production code changes are needed**, and
the real provider transport (JSON repair, retries, structured output) is
exercised for real.

- **Rules match on content, not request bodies** — the requested JSON-schema name
  and prompt substrings — so editing a prompt does not break every test.
- **An unmatched call fails the test** with the rule to paste in. Nothing is ever
  answered by a plausible-looking default.
- **Requests are validated strictly.** If Hindsight sends something real OpenAI
  would 400, so does the stub.
- **Embeddings are lexical and deterministic**, so the suite needs no torch and no
  model download.

## Running

```bash
# once: the server needs pg0, and nothing else beyond its base dependencies
(cd ../hindsight-api-slim && uv sync --frozen --extra embedded-db)

cd hindsight-system-tests
uv run pytest tests -v
```

Note the missing extra: with embeddings *and* reranking pointed at the stub,
nothing loads sentence-transformers, so the suite runs with **no torch and no
model download**. CI (`test-system`) needs no provider secrets either, which
means it runs on fork PRs — unlike every `test-api` job.

The fixtures start their own embedded Postgres (`pg0://hindsight-systest:15499`),
separate from the dev database and from the api-slim suite's, and run the server
from a scratch directory holding an empty `.env` so your own `.env` cannot leak
into the test configuration.

If your shell exports `PYTEST_ADDOPTS=-n ...`, clear it for this suite —
`pytest-xdist` is not installed here, and the session-scoped server makes it
pointless anyway.

**Background work is off by default.** Observation extraction and auto
consolidation run in the worker after a retain returns, so their LLM calls would
land at a moment no test controls — after the assertions, sometimes after the
next test has started. Both are per-bank settings, so a story about either
switches it on for its own bank and waits for the operation to finish.

**To assert on work that is still running, hold a step.** `llm.hold("<step>")`
parks that step's calls until the story lets them go, so an operation stays in
`processing` for as long as the assertions need:

```python
async with llm.hold("reflect") as held:
    first = await client.mental_models.refresh_mental_model(bank_id, model)
    await held.reached()          # the step is parked, so the operation is running
    ...                           # assert what a second request does meanwhile
await settled(bank_id)
```

Without it a story about in-flight work can only submit and hope the worker has
not finished yet, which makes the assertion a race between request latency and
claim latency — story 33 failed that way about one run in fifty. A hold changes
*when* the answer is sent, never what it is: the rule still has to be declared,
and an unscripted call still fails the test.

## Another server, another database

Two variables point the fixtures elsewhere, with the defaults unchanged:

- `HINDSIGHT_SYSTEST_API_DIR` runs the server from another `hindsight-api-slim` checkout, such as a worktree at a base commit. That checkout needs its own environment: `uv sync` inside it.
- `HINDSIGHT_SYSTEST_DATABASE_URL` replaces the suite's pg0 instance.

## Golden transcripts

`hindsight_system_tests.golden` replays one fixed scenario (retains with consolidation, recalls with and without a trace, memory reads, MCP `tools/list`, both export archives) and writes everything the server said, with wall-clock values, durations, host names and UUIDs canonicalized. It exists to hold a switched-off feature to the behaviour of the commit before it:

```bash
uv run python -m hindsight_system_tests.golden capture --out /tmp/golden-base \
    --api-dir ../<base-worktree>/hindsight-api-slim
HINDSIGHT_SYSTEST_GOLDEN_DIR=/tmp/golden-base uv run pytest tests/test_99_golden_off_mode.py
uv run python -m hindsight_system_tests.golden compare <dir-a> <dir-b>
```

Every capture runs on a fresh schema (the CLI also on a pg0 instance of its own), and refuses to write a transcript in which any retrieval arm scored two candidates equally: the database breaks such ties by row order, so they cannot be reproduced. The module docstring lists each nondeterminism and how it is handled.

## Conventions

- One story per file, named `test_NN_<story>.py`. The number is **reading order for
  a human, not execution order** — every test must pass when run alone.
- Assert through the client's responses only.
