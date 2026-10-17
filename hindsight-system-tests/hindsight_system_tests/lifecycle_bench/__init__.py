"""memory_lifecycle: a deterministic benchmark of stale, refuted and historical memory.

A conformance and robustness suite, not evidence of benefit: each scenario
scripts what a bank is told and what the consolidation model decides, then asks labelled
queries and scores the recall output with pure metrics (``metrics.py``). Every arm — the
unmodified base server included — is driven and scored identically through the public API.

- ``model.py`` / ``scenarios.py``: the vocabulary and the catalogue (the lifecycle situations + error injection).
- ``encoding.py`` / ``sources.py``: abstract decision → the arm's wire encoding; authored or
  frozen real-model payloads.
- ``labels.py`` / ``metrics.py``: the pre-registered labelling rule and metric definitions.
- ``runner.py`` / ``report.py`` / ``__main__.py``: execution, run JSON, markdown, CLI.
"""
