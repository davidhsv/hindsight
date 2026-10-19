"""``protection_bench``: does the branch protect better than plain Hindsight, on new data?

Pre-registered in ``tasks/adaptive-memory/eval/PREREG-PROTECTION-BENCH.md``; BEFORE is plain
Hindsight with its defaults, AFTER the branch with the documented personal profile plus the
recall injection gate and the Jev rerank. Three suites, each scored on its own:

- ``p1`` — planted instructions at recall: a coding-project bank (45 planted, 48 legitimate
  instruction-like, 233 ordinary memories; chunks-mode retain, no LLM) and one natural task query
  per planted and legitimate memory, recalled as the coding-agents integration recalls (top 10);
- ``p2`` — a decision-model outage: AFTER with its TypeSafe base URL pointed at a dead port,
  the same queries, and whether any recall comes back empty;
- ``p3`` — scope separation in consolidation: 11 scenarios retained in waves with metadata,
  consolidated by ``claude-haiku-5-5`` (K=1), graded by deterministic checks on the observations.

The data (:mod:`.data`) was written blind to the gate's wording and thresholds and to the scope
rule, and hashed before any run. Run ``python -m hindsight_system_tests.protection_bench --help``.
"""
