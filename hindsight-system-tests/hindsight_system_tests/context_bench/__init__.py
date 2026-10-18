"""``context_pollution``: does recall inject memories an agent should not see?

A deterministic benchmark for the Context Admission Layer experiment. One mixed bank
holds three projects' memories — durable facts, decisions, procedures, preferences,
history, debugging episodes, casual talk, humor, speculation, failed attempts and
superseded facts — and a separate small bank holds one long simulated coding session.
Every memory is retained in ``chunks`` mode, so each document becomes exactly one memory
with known text and no LLM is involved; a memory is labelled by its document id.

Labelled queries ask the server for context the way a coding agent would, and the
metrics in :mod:`.metrics` score what each arm would have injected. Arms are either a
server configuration (env), a request variant (tags, ``min_scores``, future ``context`` /
``purpose`` fields) or a client-side gate applied to recorded results; see :mod:`.arms`.

Run ``python -m hindsight_system_tests.context_bench --help``.
"""
