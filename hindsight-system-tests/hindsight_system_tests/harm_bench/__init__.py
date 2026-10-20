"""``harm_bench``: does injected memory make a coding agent answer worse — or better?

The downstream half of the Context Admission experiment (A38–A40, A61–A62; pre-registered as
Track 2's downstream/harm benchmark). 24 coding tasks, 8 of each kind:

- ``H`` — harmful-memory traps: the bank holds a plausible memory that is wrong for the task and
  shares its vocabulary (an old VPN-timeout episode, a superseded config value, another
  project's procedure, a joke); the task text plus general knowledge is enough to answer.
- ``U`` — useful-memory tasks: the answer depends on a project-specific memory in the bank.
- ``N`` — neutral: nothing relevant is in the bank and the task text suffices.

Each task is answered by ``claude-haiku-5-5`` under each condition — no memory, the ungated
top-10 recall, or the gated recall (``purpose=auto_context`` + request context on a server in
strict admission mode) — and judged by ``claude-sonnet-5-5`` against the task's rubric for
``success`` and ``harm`` (the answer followed the trap). The bank is built like
context_bench's: chunks-mode retain, one memory per document, no extraction LLM, scope
metadata on every memory, and the real local embedder and cross-encoder.

Run ``python -m hindsight_system_tests.harm_bench --help``.
"""
