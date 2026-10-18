# Adaptive Memory Lifecycle and Context Admission: final report

Branch `feat/adaptive-memory-lifecycle`, base `5331da2f4` (main, 2026-10-16), head `554284233`. Work and measurements ran 2026-10-17 to 2026-10-18. All times are BRT (UTC−3) unless they carry `+00:00` or `Z`.

**Status: nothing is pending.** The two items the previous draft marked PENDING are measured: judged LongMemEval for arm E without observations (re-run from scratch 07:07–07:55 after the credit outage, §5.4) and performance (perf workflow `wf_8c21ccb4-85c`, 02:14–06:48, §6). The partial pre-outage E-r1 is used for no number. Everything that will not be measured is listed under "Not run".

Not run:
- by user decision, the R55 last-write-wins baseline (`conflict_policy=latest`): 8 of its 40 banks were consolidated before the credit ran out, and none was used;
- the R55 recency-decay-only baseline, which REQUIREMENTS R55 asks for; no reason recorded;
- the pre-registered memory_lifecycle ablation "D reinforcement off"; no reason recorded;
- the pre-registered memory_lifecycle base-update encoding for arm C (observe); only `p1/arm-observe-base-delete.json` exists; no reason recorded;
- the FK delete cost at `--scale large` (no script on file plans it) and on Oracle (no Oracle on this machine; `fk_delete_bench.py` has no Oracle path);
- DESIGN §11.4's form of the perf gate (median of at least 5 runs at `--scale large`); the perf workflow ran 4 pooled medium rounds instead, a deviation it did not declare (§6);
- per-request retain and consolidation p50/p95/p99 (the perf-test suites report whole-batch time and throughput only), statements per recall (`pg_stat_statements`) and WAL bytes (no script collects either), and storage growth from real supersession (the perf runs' mock LLM retires nothing);
- by user decision ("essential only"), any further LLM-backed run after judged E, including a tight-budget judged run (§11 item 1).

Every number below comes from a file under `tasks/adaptive-memory/` or from this directory's JSON files. Each JSON block names its source. Where a number was recomputed here (Clopper-Pearson intervals, McNemar against a different reference, re-scorings, run-level perf intervals, spend), the script and its output are in `derived/` or the command is named. A **measured** claim was run in this session and its output is on file. A **hypothesis** is inferred from code, docs or reasoning, and names the check that would settle it. An **exploratory** endpoint or arm was not pre-registered, or was decided after data had been seen. Upstream numbers in §9 are the upstream authors' measurements, not reproduced here.

Files: `summary.md` (this report), `baseline.json` (A), `feature-off.json` (B), `observe.json` (C), `rank.json` (D), `lifecycle.json` (E), `ablations.json`, `performance.json` (a mirror of `eval/perf/performance.json` plus the verified verdicts and the run-level recomputation), `context-admission.json` (Track 2), and `derived/`, which holds the builder script, the re-scorings, the AMB patch and `scratchpad-inputs.sha256` (hashes of the inputs that live only in the temporary scratchpad, §13).

## 1. TL;DR

### Track 1: Adaptive Memory Lifecycle. Verdict: partly successful

**What works (measured).**
- **Feature off reproduces base recall, prompts and transcripts exactly, but deletes are slower.** The golden off-mode transcript gate was byte-identical at every phase, 74/74 memory_lifecycle rows match, there are 0 discordant questions on LongMemEval and on LoCoMo, and off-mode recall latency is within the noise floor of base. One off-mode regression is measured: bulk deletes are slower in every mode, off included, because of the 10 new FKs. For 50,000 units on pg0 with realistic side rows (1.5 M links, 150 k entity mentions) the SQL delete goes from 2.59 to 3.97 s (+27.7 µs per row) and the engine's `delete_bank` from 2.92 to 3.74 s; with empty side tables, 0.245 → 1.115 s. Large scale and Oracle are not run.
- **On the stub-scripted conformance suite, lifecycle mode removes stale and false memories without losing history.** The consolidation decisions there are author-written, not a real model's. On the memory_lifecycle holdout, stale contamination@5 falls from 0.200 to 0.000 and refuted leakage@5 from 0.200 to 0.000. Historical recall@5 stays at 1.000.
- **With a real model, no protected memory was wrongly retired, though some labels were wrong.** With Haiku 5.5 doing consolidation, a protected memory was truth-retired in 0 of 51 control runs (one-sided 95% upper bound 5.7%; at the cluster unit of 17 control scenarios the bound is 16.2%). 95% of the expected supersessions were labelled. Wrong labels: a protected memory marked `uncertain` in 1/51 control runs (upper bound 9.0%), a conflict declared between two different services, strict refute recall 1/3, and "four years at Google" left current.
- **The pre-registered primary endpoint is nominally significant, but it does not measure better ranking (see the attribution bullet below), and its significance needs a post-hoc repair of observation sources.** On LongMemEval knowledge-update with observations on, latest_first rose from 0.225 to 0.825 (0/24 discordant, Holm p = 2.4e-7). The driver had recorded empty source lists for every observation. An interim scoring of 21 questions on those rows exists from 17:09, and the sources were re-resolved at 20:15 under the mapping rule fixed in advance (`amb-pilot.md` (h)). On the rows as recorded, the result is 0.850 vs 0.900 (0/2, p = 0.5), not significant (§5.3).

**What does not work, or was not shown.**
- **Answers did not get better.** With observations, B-obs answers 0.950 both-correct and E-obs 0.925 (2/1, p = 1). Without observations, E answers 0.905 and B 0.891 (3/6, p = 0.51); E's top-10 recall equals D's in 211/211 and 209/211 questions, as expected when there are no claims. Upstream measured the same null for its own recall-time stale-value filter, which it then removed (#5632, §9).
- **The primary gain is an attribution artifact, not better ranking.** In 28/40 questions B-obs's top result is an observation citing both the earlier and the latest session (measured), and the tie rule counts that as a miss. A hand-read sample of those base observations (measured, 6 questions) shows they state the update with dates, for example "as of June 18 … about 6 months; as of September 2 … about 9 months". Attributing each observation to its newest source gives B-obs 0.925 vs E-obs 0.875 (2/0, p = 0.5); raw facts only gives 0.850 vs 0.900 (0/2, p = 0.5). The re-scorings are exploratory; the text check is a small sample.
- **Retention decay has no statistically significant benefit on any benchmark, and its only significant effect is a LoCoMo regression.** Its gains are point estimates only, for example +4.2 points on P1-rank (0/3, p = 0.25). On LoCoMo, rank mode's gold@5 falls from 0.902 to 0.886 (32/8, p = 1.8e-4). The regression disappears entirely with alpha = 0 (an exploratory arm, added after the regression was seen).
- **It costs more.** Lifecycle consolidation uses 52% more input tokens and, on the perf harness's mock LLM, 10–17% more time. Median recall latency rises about 9% sequentially (E +6.3 ms, D +6.6 ms) and 11–17% under 8 concurrent recalls (measured, §6).
- **The recall p95 bars are not resolved.** On the decision cell the perf analysis plan fixed before any analysed perf run (no observations; only tiny smoke runs and one medium pilot of arm E, both excluded, had been seen), the p95 point estimates are under the bars: lifecycle +1.1% sequential and −3.0% concurrent against < 10%, rank +1.6% and −7.2% against < 5%. Counting run-to-run variance, none of these four is resolved (lifecycle concurrent run-level 95% interval −13.9% to +29.9%). With observations and 8 concurrent recalls the point estimates are over the bars, lifecycle +22.3% and rank +38.9%, also unresolved at run level. The one cell resolved under its bar is lifecycle with observations, sequential: +0.9%, run-level −6.6% to +7.9%. The sequential p95 mostly measures CPython garbage-collection pauses present in every arm (§6).
- **Mistakes cost more (conformance suite).** A wrong supersession now also hides the still-true raw facts: recall@5 falls from 0.983 to 0.951 under error injection.

**The pre-registered rule, applied mechanically (section 8).**
- **Opt-in lifecycle: all four items hold as written,** so the rule's mechanical outcome is "opt-in lifecycle". Item 2 (the primary endpoint) holds only through the post-hoc source repair. Item 4 (recall p95 < 10%) holds only as a point estimate on the decision cell the perf analysis plan fixed before any analysed perf run (no observations, both shapes; the plan came after tiny smoke runs and one medium pilot of arm E, both excluded).
- **Opt-in rank: not met.** Item a fails. Branch b does not hold on the pre-registered base-update encoding (stale@1 unchanged) and is non-confirmatory on base-delete, because its stale@1 criterion was written after its own data were known. Item c (p95 < 5%) holds as a point estimate, which cannot rescue it.

**Outside the rule.** Decay alpha is a frozen parameter, not a pre-registered optional piece. Applying the optional-piece keep rule to it is the author's extension, and under that extension the outcome depends on how the margin is read (§8.3).

**Recommended default (author's judgment): "experimental", the rule's residual category, not "opt-in lifecycle" and not "safe opt-in".** Default `off` for every bank; never on by default. A bank may enable lifecycle mode as an experiment, with observations on, alpha = 0 and modest recall concurrency. Three readings drive this, each labelled in §8.1:
- the primary endpoint's large gain is an attribution artifact on base's dated in-place observations (§5.3), so item 2 is not counted;
- the pre-registered fallback, "safe opt-in, benefit unproven on LongMemEval", holds only if its margin covers the named non-inferiority endpoints; read literally ("any primary or secondary deterministic endpoint"), it fails on KU-obs earlier@5 (−12.5 points) and memory_lifecycle current recall@5 (−3.2 points). The previous draft applied the lenient reading without flagging it;
- item 4 is unresolved at run level and over the bar in the recommended configuration under concurrency.

The alpha = 0 part rests partly on an exploratory arm (LoCoMo E-nodecay) and agrees with upstream's experience of recency boosts (§9).

### Track 2: Context Admission Layer. Verdict: not successful against its pre-registered rule

**No arm meets either bar on the holdout.** The only confirmatory quality failure is E-jev's useful recall of 0.848 < 0.85. Every E-det and F number is exploratory, because their weights come from dev passes added after dev results were seen, and so are E-jev's pollution numbers (a tie-break decided them).
- **Pollution:** the best is 0.240 (arm F, exploratory), against a bar of at most 0.20. E-jev reaches 0.263 (exploratory).
- **Useful recall:** 0.761 to 0.848 (E-det 0.761, F 0.804, E-jev 0.848), against a bar of at least 0.85.
- **Drop vs the tags + Jev baseline:** at most 0.120 (F, exploratory), against the 0.15 the opt-in filter needs.
- **E-det made things worse (exploratory):** pollution 0.486 vs the baseline's 0.360, at lower recall (0.761).

**What the gate does well (all exploratory, arm F).**
- Polluted queries fall from 24 to 10 of 46: 15 were polluted only under the baseline and 1 only under F (p = 0.0005).
- It abstains correctly on 8 of 8 queries that should get nothing.
- On the pre-registered harm suite it injects 91% fewer tokens with no loss of task success.

**What it gets wrong.**
- It drops useful memories: recall falls from 0.978 to 0.804 (F, exploratory) or 0.848 (E-jev, confirmatory).
- It sometimes admits a stale memory alone, without the current one.
- The default deterministic scorer admits the requirements' own joke example (score 0.862 against the 0.6 threshold).
- On the exploratory H2 harm suite the gate made outcomes worse: success fell from 0.719 to 0.562 and harm rose from 0.292 to 0.417 (2 discordant tasks on each endpoint, all against the gate, p = 0.5). The gated arm often kept a single wrong memory. Hypothesis: a lone wrong memory misleads the model more than the same memory among the full top-10; no arm isolates this (check in §5.8).

**Latency (measured, §6).** The pre-registered latency item holds: the deterministic admission phase has a p95 of 1.2–3.2 ms over 255 sequential recalls per mode, against ≤ 150 ms. Under 8 concurrent in-process recalls, observe's admission phase reaches 274 ms p95 (258–287 ms per round, stable), and recall p95 rises by a pooled +22% (+158 ms), a point estimate that is not resolved at run level (per round −0.7% to +36.2%). The Jev and hybrid paths cost about 345 ms p50 and 432 ms p95 per recall (one decision-model call). The latency item cannot change the verdict, because the quality items already fail.

**Recommended default (author's judgment).** Observe-only, experimental. Not for automatic injection. Observe costs little for sequential recalls but adds tail latency under concurrency, and with the jev or hybrid scorer it adds about 345 ms per recall and sends memory text to the decision-model provider (§8.5).

## 2. Architecture and changes

### 2.1 Three independent axes

| | what enters context (admission, Track 2) | what is believed (claim state) | what is remembered (retention) |
|---|---|---|---|
| question | useful for this task now, within budget? | current state within its scope? | how readily it comes back without a strong cue |
| state | per-request decision, never persisted as memory state | `active`, `uncertain`, `superseded`, `refuted` plus a validity interval | stability S, last_reinforced_at, importance, exemption |
| changed by | nothing persistent | consolidation lifecycle labels, explicit PATCH | idempotent reinforcement events only |
| read by recall as | a gate after ranking | a filter or annotation (views) | a penalty-only factor |

No code path derives one axis from another. New tests pin this: `test_adaptive_axis_independence.py`.

### 2.2 Storage

There is one migration, `a7d3e91c5f42_add_adaptive_memory_tables.py`, off head `d4f7a2c9e1b5`. It has PG and Oracle slots, needs no backfill, and every memory without state is initialized lazily. All tables are store-owned (`STORE_TABLES`), take part in backup and transfer, and get their own copy in each tenant schema.

| table | holds | integrity |
|---|---|---|
| `memory_retention_state` | importance, stability_days, last_reinforced_at, reinforcement/access counts, exemption, retention_state (active/archived), below_threshold_since, archived_at | PK memory_id → memory_units ON DELETE CASCADE (deferred on PG) |
| `memory_claim_state` | truth_status, confidence, valid_from/valid_to, retired_via, superseded_by_id, inherited_from_id, reason | CASCADE on the memory, SET NULL on successor and inheritance pointers |
| `memory_claim_evidence` | the raw facts that justify a non-active claim; raw facts have stable ids, observations do not | CASCADE both ways; a claim reverts only when all its evidence is gone |
| `memory_lifecycle_relations` | supersedes / refutes / refines / splits / merges / conflicts, with origin, reason, confidence, evidence and endpoint snapshots | endpoints SET NULL; snapshots keep an auditable tombstone |
| `memory_reinforcement_events` | source_type, source_key, event_time, weight, independence_weight, applied_gain, outcome | UNIQUE (memory_id, source_type, source_key), which makes reinforcement idempotent |
| `memory_claim_shadow` | observe mode's would-retire records; never read for filtering | CASCADE |
| `invalidated_memory_units.adaptive_state` | a JSON snapshot, so invalidate → restore keeps the learned state | — |

### 2.3 Lifecycle states, relations and modes

- **States.** `truth_status` is one of `active`, `uncertain`, `superseded` or `refuted`. `retired_via` is one of `supersedes`, `refutes`, `merges`, `splits`, `explicit` or `inherited`.
  - Superseded means "true for an interval". It is shown by the historical view and by `as_of`.
  - Refuted means "never true". It is shown by no view, only by `GET` on the memory and the relation trail.
- **Labels come from the existing consolidation call; no extra LLM call.** Lifecycle mode sends a lifecycle variant of the system prompt with these fields:
  - `updates[].relation` (`supersedes` / `refutes`): the model still emits the in-place update the prompt and mission ask for, and the label turns it into a lifecycle change;
  - `creates[].supersedes | refutes | conflicts_with | refines | split_from`;
  - `deletes[].merged_into`.

  Off, observe and rank keep the byte-identical base prompt and schema (asserted).
- **Validation is deterministic.** A target must have been shown to the model, be active or uncertain, carry the same scope tags and pass a recency guard. The write is a compare-and-swap (CAS) on the claim row, and an invalid action is logged and dropped.
- **Raw facts inherit the status.** Raw facts that no other live observation cites inherit a retired status (`retired_via=inherited`), so a stale raw fact stops answering as current. A split parent's raw facts inherit `retired_via=splits` with no successor (P5 fix).
- **Conflicts.** `conflicts_with` marks both sides `uncertain`. A side wins only on a source-trust differential, so with the default (empty) trust map, repetition never settles a conflict.
- **Explicit API.** `PATCH /memories/{id}/lifecycle` and `POST /memories/{id}/confirm`; `GET /memories/{id}` gains a `lifecycle` key.
- **Mental models.** A page that cites a retired memory gets a full refresh (`mode_fallback_reason=retired_grounding`).
- **Deletes, invalidation, transfer and re-ingest carry the state.** A delete reverts only claims that lost all their evidence. Invalidate → restore keeps stability and count. Transfer carries claims, retention, relations and evidence, but not the event ledger. A re-ingest carries claims forward by (document, normalized-text hash).

Modes (per bank, hierarchical, all default `off`):

| `adaptive_memory_mode` | effect |
|---|---|
| `off` | no state load, no writes, no prompt/schema/response change (golden-gated) |
| `observe` | computes R, the adaptive factor and the would-change top-k; shadows base-schema deletes; changes nothing visible |
| `rank` | retention factor plus reinforcement; no claim influence, no prompt change |
| `lifecycle` | rank plus the lifecycle prompt/schema, claim filtering, views, `as_of`, mental-model refresh |

| `context_admission_mode` | effect |
|---|---|
| `off` | one dict lookup; no new key, phase or metric |
| `observe` | simulates `strict` and records `would_drop`, `would_abstain`, `would_change_top_k`; payload byte-identical |
| `filter` | admission threshold only; repository is a soft −3.0 logit, not a gate |
| `strict` | hard scope (repository) + threshold + currentness + MMR redundancy + count/token budget |

### 2.4 Recall integration

- **Retention factor.** `R = exp(−max(0, now − last_reinforced_at) / S)`. Then `retention_factor = floor + (1 − floor)·R`, `adaptive_factor = (1 − α) + α·retention_factor`, and `final = score · adaptive_factor`.
  - Defaults are α = 0.1 and floor = 0.3, so the factor is penalty-only and at worst 0.93.
  - It is applied after combined scoring and the post-rerank boost.
  - State for the post-trim ids is loaded in one batch, overlapped with the cross-encoder.
  - The decision-model and interleave branches get no factor.
- **Reinforcement sources and weights:** explicit confirmation 1.0, independent retain confirmation 0.8, successful reflect use 0.6, observation gaining new evidence 0.5, final external recall 0.2. The last one is off by default.
  - **Gain:** `g·w_src·w_indep·(1 − R_before) / √(1 + count)`, with a 6 h cooldown.
  - **Independence:** evidence keys follow source_event_id > session_id > document_id > conversation_id > tool_execution_id > memory id.
- **Lifecycle views.** In lifecycle mode, claims are loaded for the whole candidate set before the trim:
  - `current` drops superseded and refuted memories;
  - `historical` keeps superseded ones, annotated with valid_to and successor;
  - `as_of` keeps the memories valid at that instant.

  `query_timestamp` anchors ranking only, never truth. If the claim load fails, the recall fails closed with a 503.
- **No new synchronous generative LLM call on recall in any mode.** A test asserts this. (The jev and hybrid admission scorers make one decision-model request per gated recall; that is a scoring call, not a generative one.)

### 2.5 Admission gate (Track 2)

- **Where it runs.** Only on external recalls (consolidation and reflect recalls are never gated). It runs after combined scoring, archive suppression and the `min_scores` floors, over the top `candidates` (30). It runs before prefer-observations dedup, truncation and the token budget.
- **Scorers.**
  - `deterministic`: a logistic score over reranker, semantic, BM25, arm agreement, rank, scope match/mismatch, retrievability, importance, uncertainty, token cost and fact type.
  - `jev`: one TypeSafe decision-model request per recall; each candidate is sent as `{question, memory, memory_metadata}`.
  - `hybrid`: a 0.5 blend of the two.
  - If the reranker scores by rank, the semantic similarity takes its place in the score.
- **Currentness.** Refuted is rejected. Superseded is rejected unless the recall is historical. Uncertain needs a 0.1 margin above the threshold.
- **Purpose thresholds:** `auto_context` 0.6, minus a 0.15 bonus when the query refers to shared history; `explicit_search` 0.35; `historical_search` uses the history view.
- **Scope and context.** Soft-scope keys (module, branch, environment, service) score but never filter. Request context is `{repository, branch, module, environment, service, task_intent, workspace}`. An observation is scoped by its source facts' metadata.
- **Abstention is a first-class answer.** In filter and strict, the response carries a `context_admission {abstained, candidates, admitted}` summary.
- **No popularity loop.** While a bank gates, recall reinforcement is forced off.

### 2.6 File-level changes (`git diff --stat 5331da2f4..HEAD`: 298 files, +76,542 / −1,665)

| area | files | lines | main content |
|---|---:|---|---|
| api-slim `engine/adaptive/` | 11 | +2,071 | retention math, scoring, claims, evidence keys, producers, settings, clock, cues, views |
| api-slim `engine/context_admission/` | 7 | +1,238 | models, scoring, selection (MMR, budget), signals, pipeline, settings |
| api-slim `engine/memories/` (store) | 17 | +6,054 / −191 | `pg/adaptive.py`, `adaptive_claims.py`, `adaptive_archive.py`, `adaptive_maintenance.py`, `adaptive_transfer.py`; delete-repair, re-ingest carry-forward |
| api-slim `engine/consolidation/` | 3 | +2,599 / −48 | `lifecycle.py` (prompt variant, labels, validation, application), consolidator changes |
| api-slim `engine/db/` | 3 | +1,381 / −1 | per-dialect `DataAccessOps` methods (PG + Oracle) |
| api-slim engine core, search, reflect, retain, transfer | 26 | +3,423 / −160 | `memory_engine.py`, `search/lifecycle_filter.py`, reflect `include_history`, retain importance/exemption, transfer carriage |
| api-slim `api/http.py`, `config.py`, `config_resolver.py`, `metrics.py`, `mcp_tools.py`, admin, extensions | 7 | +1,832 / −6 | endpoints, 72 bank-config fields, 25+ env vars, OTel instruments |
| api-slim migration | 1 | +540 | `a7d3e91c5f42` |
| api-slim tests | 35 | +16,463 / −12 | `test_adaptive_*`, `test_context_admission*` |
| system tests | 54 | +20,638 / −6 | `lifecycle_bench`, `context_bench` and `harm_bench` packages; stories `test_86`–`test_104` |
| system evals | 6 | +1,316 / −37 | `test_10_lifecycle_classification.py`, `lifecycle_eval.py`, Anthropic judge backend |
| clients (Python, TS, Go, Rust build) | 60 | +12,363 / −762 | mostly generated models; wrapper methods and tests |
| control plane | 21 | +957 / −11 | confirm/lifecycle routes, lifecycle badge and section, i18n |
| docs + generated docs skill | 26 | +4,902 / −318 | new `developer/adaptive-memory.mdx`; configuration, recall, memories, monitoring |
| integrations (coding-agents) | 11 | +539 / −107 | repository identity, branch and module stamped at retain and sent at recall |
| CLI, dev, embed, api-slim `pyproject.toml`, `.env.example`, `uv.lock` | 10 | +226 / −6 | struct literals, env template |

## 3. Behaviour examples (from tests and docs, not invented)

**Version upgrade.** From `hindsight-system-tests/tests/test_87_java_upgrade.py`, which runs Java 17 → 21 on the stub LLM. The docs tell the same story as Java 21 → 25.

| request | returns |
|---|---|
| current recall, "Which Java version does the billing service run?" | only the new observation and fact |
| `lifecycle_view="historical"` | the new pair plus the old pair, annotated `superseded`, with `valid_to` = day 30 and `superseded_by` = the new observation |
| `as_of` day 15 / day 45 | exactly the old pair / exactly the new pair |
| `query_timestamp` day 15 (no `as_of`) | the new pair only: the ranking clock is not the truth clock |
| `GET` the old observation | text unchanged, `retired_via=supersedes`, evidence = the new fact, a `supersedes` relation with origin `consolidation` |

Two related tests:
- Deleting the successor's document makes the old version current again (`test_91`).
- A mental model refreshed after the supersession was shown Java 25 and never Java 21 (`test_95`).

**False claim refuted.** From `test_88_false_claim_refuted.py` and the docs. "The checkout service caches sessions in Redis" is later corrected by "never used Redis; Postgres". Consolidation labels it `refutes`.
- The Redis observation and its fact answer in no view: current, historical or any `as_of`.
- `GET` keeps the claim, the reason, the correcting memory and both texts.
- Re-ingesting the README does not bring the refuted claim back.

**Same-session repetition vs spaced confirmation.** From `test_86_spaced_reinforcement.py` and the docs.
- One claim is stated in sessions s1, s2 and s3 (days 0, 30 and 120). The other is stated ten times in session r1 on day 0, so it has the larger proof count (10 vs 3).
- The spaced claim gets `reinforcement_count` 2, with keys `session:s1..s3`. The repeated claim gets at most 1, with key `session:r1`.
- At day 200 the spaced claim outranks the repeated one, with higher retrievability and adaptive factor in the trace.
- An explicit confirmation retried with the same key counts once.

The docs work the same pattern through the formula:

| pattern (world fact, S₀ = 30 d) | stability | R on day 200 |
|---|---|---|
| ten confirmations on day 0 | 30 d (all inside the cooldown) | ≈ 0.001 |
| one confirmation on day 30, one on day 120, different sources | 49 d, then 78 d | ≈ 0.36 |

**Admission: a precise cue vs a strong but irrelevant memory.** From `test_context_admission_invariants.py`, using real cross-encoder scores, strict mode and auto_context.
- The query is "What exact workaround did we use for FIN-1602?". The workaround memory has R = 0.15 and is the only memory returned.
- An on-call memory with importance 1.0, S = 1000 d and R = 0.99 is rejected as `below_threshold`.

**Admission: the joke (failures included).**
- The requirements' own pair is the Kubernetes-gods joke vs "How is the payments API deployed to production?". The default deterministic scorer gives the joke **0.862** against the 0.6 auto threshold and **admits it**: reranker 0.815, semantic 0.694, BM25 0. That is a strict xfail (Defect 1, `A68-CHECKLIST.md`).
- A query that asks for the joke ("What is the running team joke…") admits it, and the gate never deletes it.

Holdout examples from the benchmark (`eval/context-bench/REPORT.md` and `eval/harm-bench/REPORT.md`):

| case | arm | what happened | expected |
|---|---|---|---|
| h01 "How is the Payments API deployed?" | F | admits `a.deploy.ssh`, the outdated SSH-deploy hard negative | only the Argo CD memory |
| w08 (wrong subsystem) | E-det, E-jev, F | all three admit `c.vpn.timeout`, which the baseline did not | not admitted |
| s07 (stale on-call) | all three arms | admit `a.oncall.old` and miss `a.oncall.new` | the current one |
| e10 (exact historical cue) | F | abstains; misses `c.cert` | admit `c.cert` |
| n12 (none relevant) | E-jev | admits `bg.secrets.c` | nothing |
| x19 (cross project) | F | abstains; misses `bg.logging.c` | admit it |
| harm H01–H08 (traps) | gated | abstains; the trap scores 0.18–0.73, below 0.80 | correct |
| harm U01–U08 (useful) | gated | the needed memory is admitted at 0.95–0.97 | correct |
| harm H2 M04 (cross module) | gated | admits only the capture-module procedure (0.85); the answer applies it to refunds (harm 2/2) | flag it as the wrong module |
| harm H2 J01–J04 (old episodes) | gated | admits the debugging episode (0.90–0.93) and rejects the jokes (0.51–0.54); harm 8/8 answers | reject the episode |

## 4. Test results

| gate (commit) | api-slim | system tests | golden off-mode gate | other |
|---|---|---|---|---|
| base `5331da2f4` | full suite: 12,107 passed, 8 failed, 1 error, 199 skipped; the 9 are pre-existing timeouts and missing credentials | 385 passed, 1 skipped | captured | — |
| P1 `dafd09a64` | 69 files: 1,900 passed, 114 skipped, 0 failed | — | identical | lint and ty clean; 5/5 mutants killed; Oracle 23ai live: 8 passed (4 new migration tests, `test_oracle_migration_idempotent`, 3 delete-path engine tests) |
| P2 `3816d0e4b` | 87 files: 2,698 passed, 4 skipped, 1 xfailed, 0 failed | 31 passed (targeted) | pass | Python client 247 passed; client coverage 63/63 |
| P3 `b491b2bb9` | 186 files: 5,757 passed, 118 skipped, 1 xfailed, 1 failed (a timing flake that passed 7/7 on rerun) | 484 passed, 1 skipped | pass | 10/10 mutants killed; `hs_llm_core` lifecycle judge tests 4/4 (single run) |
| P4 `d65f17b05` | full suite: 12,770 passed, 9 failed, 1 error, 200 skipped, 5 xfailed (7 baseline entries + 2 parallel-interference flakes that pass at `-n0`); after the review fixes, 138 files: 2,748 passed | 488 passed, 1 skipped | pass | — |
| P5 `728b265c9` | full suite: 12,781 passed, 8 failed, 1 error, 200 skipped, 5 xfailed; 8 baseline entries + 1 timeout at the 300 s edge (alone: 27.7 s branch vs 26.6 s base) | 499 passed, 1 skipped | pass | control plane 346 passed; TS client 116 passed; docs build success |
| T2 `504ca72e9` / `2f4c7daae` | 60 files: 1,500 passed, 4 skipped | 537 passed, 1 skipped | pass | CLI `cargo test` 167 passed (pinned deps on rustc 1.87); coding-agents 1,396 passed |
| A68/R59 audit (`5f523c2e7`) | adaptive + admission files: 667 passed, 1 xfailed (the A6 joke reproduction) | `test_104` 4 passed | pass | context_bench 44 and harm_bench 46 unit tests (STATE 2026-10-18 01:45) |

Sources: `STATE.md`, `impl/P1..P5-NOTES.md`, `impl/T2-NOTES.md`, `eval/A68-CHECKLIST.md` and `eval/R59-CHECKLIST.md`.

Not run:
- Oracle suites after P1. P1 ran its migration and delete-path tests live on Oracle 23ai Free (8 passed) and then removed the container, because starting it had OOM-killed the user's MySQL container. A full Oracle engine run in P1 OOM-killed the Oracle instance itself after about 8 minutes (30 passed, then 25 failed after the crash, not attributable). Every P2–T2 Oracle path is checked statically only, and the two `-m oracle` live tests written in P4 have never run;
- the `hs_llm_mat` matrix;
- the mental-model judge test;
- the live-server client suites;
- the evaluation arms and measurements listed under "Not run" at the top.

## 5. Benchmark results across arms

Arms:
- A: base;
- B: fork, mode off;
- C: observe;
- D: rank;
- E: lifecycle.

Paired statistics are exact two-sided McNemar tests on per-question outcomes. Cells show discordant pairs as `reference-only / arm-only`. Brackets are 95% Clopper-Pearson intervals unless stated.

### 5.1 memory_lifecycle (deterministic, stub LLM with author-scripted decisions; 18 scenarios, 74 queries)

**This is a conformance suite, not evidence about a real model.** Its consolidation decisions are written by the author (DESIGN §11.1). Each cell is the mean of a per-query score. The n's are the queries a metric applies to: all split current 58, stale 18, refuted 4, history 16; holdout current 18, stale 7, refuted 4, history 10. Both encodings are pre-registered for every arm; arm C was run on base-delete only.

k = 5 (the pre-registered secondary endpoints):

| arm (code) | encoding | split | prec@5 | rec@5 | stale@5 | refuted@5 | hist@5 | supers@5 |
|---|---|---|---|---|---|---|---|---|
| A (base) | base-delete | all | 0.621 | 0.983 | 0.233 | 0.200 | 1.000 | 0.000 |
| B off (P3) | base-delete | all | 0.621 | 0.983 | 0.233 | 0.200 | 1.000 | 0.000 |
| C observe (P1) | base-delete | all | 0.621 | 0.983 | 0.233 | 0.200 | 1.000 | 0.000 |
| D rank (P3) | base-delete | all | 0.621 | 0.983 | 0.233 | 0.200 | 1.000 | 0.000 |
| E lifecycle (P3) | base-delete | all | 0.676 | 0.951 | 0.056 | 0.000 | 1.000 | 0.833 |
| E lifecycle (P5; reproduced 74/74 on clean `2f4c7daae`) | base-delete | all | 0.683 | 0.951 | 0.000 | 0.000 | 1.000 | 1.000 |
| A = B = D | base-update | all | 0.611 | 0.983 | 0.247 | 0.250 | 1.000 | 0.000 |
| E (P3 / P5) | base-update | all | identical to base-delete | | | | | |
| A = B = C = D | base-delete | holdout | 0.444 | 1.000 | 0.200 | 0.200 | 1.000 | 0.000 |
| A = B = D | base-update | holdout | 0.444 | 1.000 | 0.200 | 0.250 | 1.000 | 0.000 |
| E (P3 = P5, both encodings) | — | holdout | 0.528 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |

k = 1 (not a pre-registered endpoint, except stale@1, which the rank rule names):

| arm | encoding | split | prec@1 | rec@1 | stale@1 | refuted@1 | hist@1 | supers@1 |
|---|---|---|---|---|---|---|---|---|
| A = B = C | base-delete | all | 0.914 | 0.782 | 0.222 | 0.250 | 0.188 | 0.778 |
| D rank | base-delete | all | 0.931 | 0.799 | 0.167 | 0.250 | 0.188 | 0.833 |
| E (P3) | base-delete | all | 0.948 | 0.816 | 0.056 | 0.000 | 0.375 | 0.944 |
| E (P5) | base-delete | all | 0.966 | 0.825 | 0.000 | 0.000 | 0.375 | 0.944 |
| A = B = D | base-update | all | 0.948 | 0.859 | 0.111 | 0.250 | 0.812 | 0.889 |
| E (P3) | base-update | all | 0.948 | 0.816 | 0.056 | 0.000 | 0.375 | 0.944 |
| E (P5) | base-update | all | 0.966 | 0.825 | 0.000 | 0.000 | 0.375 | 0.944 |
| A = B = C | base-delete | holdout | 0.833 | 0.833 | 0.286 | 0.250 | 0.000 | 0.714 |
| D rank | base-delete | holdout | 0.889 | 0.889 | 0.143 | 0.250 | 0.000 | 0.857 |
| E (P3 = P5) | base-delete | holdout | 0.944 | 0.944 | 0.000 | 0.000 | 0.400 | 1.000 |
| A = B = D | base-update | holdout | 0.889 | 0.889 | 0.143 | 0.250 | 0.800 | 0.857 |
| E (P3 = P5) | base-update | holdout | 0.944 | 0.944 | 0.000 | 0.000 | 0.400 | 1.000 |

What these tables show:
- **Off equals base.** Off vs A is 74/74 rows identical, at P1 and again at P3.
- **The encodings barely differ at k = 5 but differ a lot at k = 1.** At k = 5, base-update A gives stale@5 0.247 and refuted@5 0.250, and E is identical to base-delete. At k = 1, base-update A keeps the old version's history at the top (hist@1 0.812), and against it lifecycle halves historical recall@1: 0.812 → 0.375 (all) and 0.800 → 0.400 (holdout). Current recall@1 also falls on the all split (0.859 → 0.816 at P3, 0.825 at P5) while it rises on the holdout (0.889 → 0.944). Stale@1 falls (0.111 → 0.056 at P3, 0.000 at P5). Under base-delete, lifecycle raises hist@1 instead (0.188 → 0.375).
- **Rank's stale@1 gain exists only on base-delete.** On base-update every k = 1 metric of D equals off (stale@1 0.111 = 0.111 all, 0.143 = 0.143 holdout).
- **Lifecycle state conformance** (P5, all): conformance 0.941 (n = 17), retirement recall 1.000 (n = 10), false retirement 0.000 (n = 15).
- **The residual stale contamination is fixed at P5.** P3's remaining stale@5 of 0.056 was entirely the split_generalization scenario, and the P5 split fix brings it to 0.000. The P5 rows differ from P3 only on `jf-q1..q4`.
- **Recall drops under error injection.** rec@5 falls from 0.983 to 0.951 because of the `supersede_still_true` injection: a wrong supersession now also hides still-true raw facts. That is the error-amplification limitation.
- **Pre-registered ablations that ran** (P3, base-delete):
  - lifecycle alpha = 0: identical at k = 5 on every metric; at k = 1 it gives prec@1 0.931 vs 0.948 and supers@1 0.889 vs 0.944 (all), and supers@1 0.857 vs 1.000 (holdout);
  - lifecycle + historical cues: identical to lifecycle on every metric (the bench sends explicit views, so the detector never decides);
  - lifecycle with `lifecycle_view=historical` as the default: stale@5 0.411 (all) / 0.400 (holdout), refuted@5 0.000, hist@5 1.000. The stale rise is by design, because history is shown.
- **Pre-registered but not run:** D reinforcement-off, and C on base-update. No reason is recorded.
- **The existing features do not fix it** (A, base-delete, k = 5).
  - `recall_prefer_observations=true` makes stale@5 worse (0.233 to 0.352) and refuted@5 worse (0.200 to 0.333).
  - `types=['observation']` brings stale@5 to 0 but drops hist@5 from 1.000 to 0.125.

### 5.2 AMB LongMemEval retrieval, observations off (frozen arm-A banks; KU n = 72, TR n = 127; reference B)

| endpoint | A | B | C | D | E |
|---|---|---|---|---|---|
| KU latest_first (**P1-rank**, primary) | 0.819 (0/0) | 0.819 [0.711, 0.900] | 0.819 (0/0) | 0.861 [0.759, 0.931] (0/3, p = 0.25) | 0.861 (0/3, p = 0.25) |
| KU latest@5 / @20 | 1.000 / 1.000 | 1.000 / 1.000 | same | same | same |
| KU earlier@5 | 0.986 | 0.986 | 0.986 | 0.972 (1/0, p = 1) | 0.972 (1/0, p = 1) |
| KU earlier@20 (non-inferiority) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| TR gold@5 | 0.976 | 0.976 | 0.976 | 0.984 (0/1, p = 1) | 0.984 (0/1, p = 1) |
| TR gold@20 (non-inferiority) | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| TR all_gold@5 / @20 (non-inferiority at @20) | 0.701 / 0.913 | 0.701 / 0.913 | same | same | same |

- **A == B on every endpoint, with 0 discordant questions** (R37). C == B.
- **E == D**, because without observations there are no claims.
- **P1-rank is not significant:** Holm-adjusted p = 0.25.
- **Ceiling.** latest@5 is 1.000 in every arm, and AMB returns up to 300 results, so ranking changes rarely reach the context.

### 5.3 AMB LongMemEval retrieval, observations on (40 KU questions; reference B-obs)

Each bank was consolidated once per condition on P3 code with Haiku 5.5, either with the base prompt (B-obs, D-obs, P-obs, T-obs) or the lifecycle prompt (E-obs*).

| endpoint | B-obs | D-obs | P-obs | T-obs | E-obs | E-obs-hist | E-obs-nodecay | E-obs-off (exploratory) |
|---|---|---|---|---|---|---|---|---|
| latest_first (**P1-life**, primary) | 0.225 [0.108, 0.385] | 0.175 (2/0, p = 0.5) | 0.150 (3/0, p = 0.25) | 0.150 (3/0, p = 0.25) | **0.825** [0.672, 0.927] (0/24, p = 1.19e-7) | 0.800 (1/24, p = 1.55e-6) | 0.825 (0/24) | 0.800 (1/24) |
| latest@5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 (1/0) | 0.975 (1/0) |
| earlier@5 | 1.000 | 0.975 (1/0) | 1.000 | 1.000 | 0.875 (5/0, p = 0.0625) | 0.925 (3/0) | 0.925 (3/0) | 0.975 (1/0) |
| latest@20 / earlier@20 | 1.000 / 1.000 | same | same | same | same | same | same | same |

Arm definitions:
- **P-obs** = off with `recall_prefer_observations=true`.
- **T-obs** = off with `types=['observation']`.
- **E-obs-off** = the lifecycle-consolidated banks read in off mode. It is **exploratory**: it is not in the pre-registration (whose cons-life arms are E-obs, E-obs-hist, E-obs-nodecay and E-obs-latest), and it was defined in `eval/arms/arms-cons-life.json` at 14:16, before any observations-on data.

**Observation sources were repaired post-hoc, and P1-life's significance depends on it (measured).**
- `shared_import_eval.py` recorded empty source lists for every recalled observation, because `GET /memories/{id}`'s `source_memories` entries carry no document_id.
- An interim scoring of 21 questions on those rows exists (`eval/arms/score-obs-interim/`, 17:09): latest_first B-obs 0.952 and E-obs 1.000, and T-obs 0.000 on every endpoint (observations mapped to no session).
- At 20:15 `eval/arms/resolve_observations.py` re-resolved every observation against the still-running servers (`resolve-cb.log`: 20,907 observation rows, 0 without a source document), keeping the originals in `obs-cons-{base,life}/rows-unresolved/`. The final scoring (`score-obs/`, 20:18) came after it.
- The repair implements the rule fixed in advance: `eval/amb-pilot.md` (h), on file by 04:17, maps "observation -> the set of sessions of its `source_memory_ids`", and the `score_endpoints.py` docstring (09:44) scores an observation through its resolved source documents. So the pre-specified rule is what was applied.
- On the rows as recorded (`score_endpoints.py` re-run on `rows-unresolved/` for this report), latest_first is B-obs 0.850, E-obs 0.900 (0/2, p = 0.5), D-obs 0.875, P-obs 0.525 and T-obs 0.000, the same as the "raw facts only" variant below. Without the repair, P1-life is not significant.

What the table shows:
- **P1-life is significant after the source repair:** Holm-adjusted p = 2.4e-7.
- **Exploratory: most of the gain comes from the lifecycle consolidation, not the recall filter.** E-obs-off reaches 0.800, against 0.825 for E-obs.
- **History leaves the current top 5 by design.** earlier@5 falls from 1.000 to 0.875 for E-obs (5/0, p = 0.0625, not significant).

**Exploratory, post-hoc re-scoring.** Same rows, same tie rule; `derived/latest-first-attribution-exploratory.json`, script `derived/latest_first_attribution.py`.

| latest_first variant | B-obs | D-obs | E-obs | E-obs-off | P-obs | T-obs |
|---|---|---|---|---|---|---|
| as pre-registered: an observation stands for every session it cites | 0.225 | 0.175 | 0.825 | 0.800 | 0.150 | 0.150 |
| observation attributed to its newest source session | 0.925 | 0.900 (1/0) | 0.875 (2/0, p = 0.5) | 0.850 (3/0) | 0.900 | 0.900 |
| raw facts only | 0.850 | 0.875 (0/1) | 0.900 (0/2, p = 0.5) | 0.850 (0/0) | 0.525 | 0.000 |
| share of questions whose top-1 is an observation citing both evidence sessions | 0.700 | 0.725 | 0.050 | 0.050 | 0.750 | 0.750 |

Measured: in 28 of 40 questions, B-obs's top result is an observation citing both evidence sessions, and the pre-registered rule counts that tie as "not latest-first". Hypothesis: these are observations the base prompt updated in place, rather than observations created citing both facts; the rows record no create/update action, so they cannot tell the two apart. Check: the consolidation `/llm-requests` trace, or each top-1 observation's creation and update history compared with its source facts' times, for those 28 banks.

When observations are attributed to their newest source, or ignored, the large gain disappears: E-obs vs B-obs is 0.875 vs 0.925 and 0.900 vs 0.850, neither significant. So P1-life measures mostly a change in provenance: lifecycle creates a successor observation citing only the new evidence. It does not measure better ranking of the current value.

Whether B-obs's top observation's text states the latest value is a text question this re-scoring does not answer by itself. **Measured, small sample (6 questions, read by hand from the exported `observations.json` of the cons-base archives; the archive keeps no observation ids, so these are the observations whose sources include both evidence sessions, not provably the top-1):** the base prompt writes the update into one observation with dates, and the current value is stated explicitly. Examples: "As of June 18, 2023, they had used it for about 6 months; as of September 2, 2023, about 9 months" (gold: 9 months); "leads a team of 5 engineers … Earlier (May 2023) the team was reported as 4" (gold: 4 then 5); "completed 4 painting projects … (as of August 16) … As of October 9, completed their 5th" (gold: 5). This is consistent with B-obs's 0.950 judged accuracy. So for base Hindsight's in-place observations the pre-registered latest_first rule scores a correct, dated update as a miss: the endpoint does not measure what it was meant to measure on this arm.

### 5.4 Judged answer accuracy (AMB, recall budget 4096/2048, answer Haiku 5.5, judge Sonnet 5.5, 2 unseeded repeats)

Primary per the 20:34 amendment: correct in both repeats. Sensitivity: correct in at least one. Abstention items are included.

| arm | per repeat | both correct [95% CI] | vs B | ≥ 1 correct | vs B | KU both (n = 78) | TR both (n = 133) |
|---|---|---|---|---|---|---|---|
| A | 0.924, 0.919 | 0.891 (188/211) [0.841, 0.930] | 3/3, p = 1 | 0.953 | 2/4, p = 0.688 | 0.949 | 0.857 |
| B | 0.919, 0.915 | 0.891 (188/211) [0.841, 0.930] | — | 0.943 | — | 0.923 | 0.872 |
| D | 0.910, 0.919 | 0.896 (189/211) [0.846, 0.934] | 3/4, p = 1 | 0.934 | 3/1, p = 0.625 | 0.923 | 0.880 |
| E | 0.915, 0.934 | 0.905 (191/211) [0.857, 0.941] | 3/6, p = 0.508 | 0.943 | 2/2, p = 1 | 0.936 | 0.887 |

| arm (40 KU, observations on) | per repeat | both correct [95% CI] | vs B-obs | ≥ 1 correct | vs B-obs |
|---|---|---|---|---|---|
| B-obs | 0.950, 0.975 | 0.950 (38/40) [0.831, 0.994] | — | 0.975 | — |
| E-obs | 0.950, 0.950 | 0.925 (37/40) [0.796, 0.984] | 2/1, p = 1 | 0.975 | 0/0, p = 1 |

- **Null on answer accuracy, both with and without observations.** No arm differs from its reference: E vs B is 3/6 (p = 0.508) on both-correct and 2/2 (p = 1) on at-least-one.
- **A and B give the noise floor of this setup.** They share the ranking code path, yet their both-correct outcomes differ on 6 of 211 questions. E vs B differs on 9, D vs B on 7.
- **Judged E is a fresh re-run.** Both repeats ran from scratch at 07:07–07:55 on the P5 worktree (`728b265c9`) in lifecycle mode (`eval/judged/rerun_E.sh`). The pre-outage partial E-r1 (189/211) is kept but used for no number.
- **Judged E recalled what D recalled (measured for this report, `lifecycle.json` `recall_check`).** Per question, the top-10 (document, type, mentioned_at) sequence of E equals D's in 211/211 (repeat 1) and 209/211 (repeat 2), the same agreement as D's two repeats with each other (210/211). E and B agree in 158/211, as D and B do, because of the retention factor. So without observations E behaves as D at recall, as §11 item 4 predicted. Beyond the top 10 the contexts are not deterministic: two repeats of one arm (B, D or E) have identical full contexts in only 22–26 of 211 questions.
- **Exploratory, not a pre-registered comparison:** E vs D both-correct 2/4 (p = 0.69). Hypothesis: the 0.905 vs 0.896 difference is answer and judge sampling plus the non-deterministic context tail; it is inside the A/B noise floor. Check: a third repeat of D and E.
- **The judged E-obs runs used the lifecycle current view.** Their top-10 signatures match the endpoint E-obs arm in 40/40 (repeat 1) and 39/40 (repeat 2).

### 5.5 LoCoMo locomo10 retrieval (1,540 queries, frozen arm-A banks, observations off, P5 code; reference B)

| arm | gold@5 [95% CI] | vs B | gold@10 | vs B | gold@20 | vs B | any gold |
|---|---|---|---|---|---|---|---|
| A | 0.902 | 0/0 | 0.947 | 0/0 | 0.975 | 0/0 | 0.994 |
| B | 0.902 [0.886, 0.916] | — | 0.947 | — | 0.975 | — | 0.994 |
| D | 0.886 [0.869, 0.902] | 32/8, p = 1.8e-4 | 0.939 | 15/3, p = 0.0075 | 0.975 | 4/4, p = 1 | 0.994 |
| E | 0.886 | 32/8, p = 1.8e-4 | 0.939 | 15/3, p = 0.0075 | 0.975 | 4/4, p = 1 | 0.994 |
| E alpha = 0 (**exploratory**: added after the D/E regression was seen) | 0.902 | **0/0** | 0.947 | **0/0** | 0.975 | **0/0** | 0.994 |

- **The regression is significant but inside the 3-point non-inferiority margin.** It is 1.6 points at k = 5 and 0.8 at k = 10.
- **Exploratory: the retention decay causes all of it.** With alpha = 0, E is identical to B at every k. That arm was decided after the regression was seen (STATE 2026-10-17 23:10 states the hypothesis and launches the check).
- **Where it falls.** It is concentrated in temporal questions (11 worse / 0 better) and open-domain (15/3). The queries that got worse at k = 5 have older gold items (STATE 2026-10-17 23:10).
- Hypothesis for the mechanism: AMB's LoCoMo `query_timestamp` often precedes later sessions. Future-dated memories are clamped to R = 1 while older relevant ones decay. The check: re-run D with `query_timestamp` set to the conversation's last session time plus one day.

### 5.6 Real-model classification eval (27 scenarios, arm B = off vs E = lifecycle, Haiku 5.5 consolidation, Sonnet 5.5 judge)

| metric (arm E) | P3 code, K = 3 (162 runs) | P5 code, K = 1 (54 runs) |
|---|---|---|
| false retirement on controls (unit: scenario-run) | **0/51**, one-sided 95% upper 5.7% | 0/17, upper 16.2% |
| same, unit: control scenario | 0/17, upper 16.2% | — |
| control runs with a wrong `uncertain` label | 1/51 (upper 9.0%) | — |
| supersede recall / precision | 54/57 = 0.95 [0.85, 0.99] / 60/65 = 0.92 [0.83, 0.97] | 18/19 = 0.95 / 20/21 = 0.95 |
| refute recall: strict / either label | 1/3 / 3/3 | 1/1 |
| conflict recall / precision | 6/6 / 6/7 | 2/2 / 2/2 |
| raw-fact leak (a retired observation leaves its source active) | 0/67 | 0/22 |
| judged pass, E vs B | 79/81 = 0.98 vs 73/81 = 0.90 | 25/27 = 0.93 vs 23/27 = 0.85 |
| input tokens, E / B (same call count) | 2.0× | 1.52× |

- **Pooled false retirement.** Pooling P3 and P5 gives 0/68 with a one-sided upper bound of 4.3%. This pooling was computed for this report and is the author's judgment, not pre-registered.
- **Why B fails some scenarios.** In environment-divergence, B rewrote "50 → 5" in 3 of 3 runs. E kept both environments, because their scope fingerprints differ.
- **E fails two scenarios.**
  - employer-supersedes: "four years at Google" was left current.
  - recency-is-when-stated: a recency judgment was missed, and nothing was wrongly retired.
- **All 5 supersede false positives are refute-vs-supersede confusion.**
- **The one conflict false positive** (noisy-service-catalogue run 2) put two different services ("invoice-sink" and "invoice-sync") into a `conflicts` relation and marked both `uncertain`. Nothing was retired, so it is not a false retirement under the pre-declared rule, but it is a wrong label on protected memories.

### 5.7 context_bench (Track 2; 303 + 9 memories, 131 queries: 85 dev / 46 holdout, 8 holdout queries expect abstention; injection cap 10)

Baselines on the base server, holdout, each at its dev-selected threshold:

| arm | precision | useful recall | pollution | polluted queries | abstention acc | leak@10 | tokens/query |
|---|---|---|---|---|---|---|---|
| A no gate | 0.099 | 0.978 | 0.891 | 1.000 | 0.000 | 0.068 | 194.1 |
| B similarity ≥ 0.80 | 0.471 | 0.522 | 0.529 | 0.239 | 0.875 | 0.039 | 22.1 |
| C `min_scores.final` ≥ 0.90 | 0.427 | 0.696 | 0.566 | 0.370 | 0.500 | 0.027 | 35.0 |
| C' tags (hard scope) | 0.099 | 0.978 | 0.891 | 1.000 | 0.000 | 0.000 | 192.3 |
| C' + C | 0.438 | 0.696 | 0.554 | 0.326 | 0.750 | 0.000 | 33.6 |
| D Jev pruning | 0.571 | 0.957 | 0.418 | 0.565 | 0.000 | 0.130 | 37.5 |
| **C' + D tags + Jev (pre-registered comparator)** | 0.625 | 0.978 | 0.360 | 0.522 | 0.000 | 0.000 | 35.3 |

T2 arms, settings frozen on dev (`chosen.json` sha256 `02b02eec…`), each run once on the holdout. In the McNemar column, the pair is `baseline-only / arm-only` polluted queries, following the convention of section 5:

| arm | scorer, mode, thresholds | dev pollution / recall / abstention | **holdout pollution** [bootstrap 95%] | **holdout useful recall** [bootstrap 95%] | **abstention** | leak@10 | precision | polluted queries, arm vs baseline (baseline-only / arm-only, McNemar) | stale admission | tokens/query |
|---|---|---|---|---|---|---|---|---|---|---|
| E-det (exploratory) | deterministic, filter, 0.85 / 0.90 | 0.394 / 0.855 / 0.688 | 0.486 [0.362, 0.588] | 0.761 [0.652, 0.868] | 5/8 | 0.000 | 0.500 | 21 vs 24 (9/6, p = 0.61) | 0.103 | 37.2 |
| E-jev | jev, filter, 0.80 / 0.80 | 0.208 / 0.895 / 1.000 | 0.263 [0.146, 0.365] (exploratory) | 0.848 [0.738, 0.940] | 7/8 | 0.000 | 0.722 | 11 vs 24 (15/2, p = 0.0023) (exploratory) | 0.135 | 30.7 |
| F (exploratory) | hybrid, strict, 0.80 / 0.85 | 0.108 / 0.882 / 1.000 | 0.240 [0.125, 0.346] | 0.804 [0.682, 0.911] | 8/8 | 0.000 | 0.755 | 10 vs 24 (15/1, p = 0.0005) | 0.174 | 25.5 |
| tags + Jev | base server | — | 0.360 | 0.978 | 0/8 | 0.000 | 0.625 | — | 0.090 | 35.3 |

**Endpoint status** (amendment of 2026-10-18 01:45):
- **Every E-det and F endpoint is exploratory.** Their weights come from dev passes 2 and 3, which were added after dev results were seen.
- **E-jev:** its pollution, precision, polluted-query and token endpoints are exploratory, because a tie-break chose among four dev-tied settings. Those settings give holdout pollution 0.307, 0.295, 0.276 and 0.263. Its recall (0.848), abstention (0.875), leakage (0) and calibration are confirmatory.

Other measurements on the holdout:
- **E-det is worse than the baseline (exploratory).** Its pollution is 0.486 against tags + Jev's 0.360, at a useful recall of 0.761 against 0.978. Its polluted-query count barely changed (21 vs 24, p = 0.61).
- **Every arm got worse from dev to holdout.** Pollution rose for F (0.108 → 0.240), E-jev (0.208 → 0.263) and E-det (0.394 → 0.486).
- **The informational holdout sweep finds no grid setting that meets the quality items** for any arm. So the negative result does not depend on where the thresholds were frozen.
- **Calibration.** Holdout Brier is 0.048 for E-det, 0.068 for E-jev and 0.046 for F, against 0.033 for a constant base-rate score. ECE is 0.084, 0.188 and 0.143. AUC is 0.981, 0.995 and 0.996.
- **Stale memories cause half of F's pollution.** Every candidate's server truth_status was `active`, because the bench runs without consolidation. So the currentness gate never fired, and only the scorer could reject a superseded memory.
- **Latency under heavy load** (load average 53 / 68 / 101; not the pre-registered number, which is in §6). The admission phase p95 was 2.7 ms for the deterministic path, 425.7 ms for E-jev and 434.3 ms for F. Each Jev path makes one TypeSafe call per recall, with p50 / p95 of 352 / 418 ms.

### 5.8 Harm benchmark (Track 2 downstream; answer Haiku 5.5, judge Sonnet 5.5, 2 repeats)

The **original** suite is pre-registered: 24 tasks (8 trap tasks H, 8 useful-memory tasks U, 8 neutral tasks N) over a 152-memory bank. `gated` is F at its dev setting, so it is exploratory. `ungated-nopreamble` is exploratory too.

| condition | success | success U | harm H | harm when the trap was injected | trap in top-k (H) | injected memories / tokens | answer input tokens |
|---|---|---|---|---|---|---|---|
| none | 0.667 | 0.000 | 0.000 | — | — | 0 / 0 | 190 |
| ungated top-10 | 1.000 | 1.000 | 0.000 | 0.000 (n = 16) | 8/8 | 10.0 / 581 | 1,101 |
| ungated, no preamble (exploratory) | 1.000 | 1.000 | 0.000 | 0.000 (n = 16) | 8/8 | 10.0 / 553 | 1,050 |
| gated (exploratory) | 1.000 | 1.000 | 0.000 | — (n = 0) | 0/8 | 0.5 / 51 | 277 |

- **Memory helps here.** none vs ungated success: 0 / 8 discordant tasks, p = 0.008 (all 8 U tasks).
- **The gate changes no task.** ungated vs gated: 0 discordant tasks.
- **It abstains where nothing helps.** The gate injected nothing on 15 of 24 tasks: all 8 traps and 7 of the 8 neutral tasks.
- **It cut injected tokens by 91%.**
- **Harm sits at the floor.** Ungated harm is already 0/16 with the trap injected 16/16, so this suite cannot show the gate reducing harm.
- **The preamble does not explain that floor.** The preamble hypothesis is refuted: removing it changes nothing.

**H2** is exploratory, designed after the base result was seen: 16 tasks in 4 groups of 4, over a 174-memory bank.

| condition | success | harm H | harm when the trap was injected | trap in top-k | injected tokens |
|---|---|---|---|---|---|
| none | 0.219 | 0.000 | — | — | 0 |
| ungated | 0.719 | 0.292 | 0.292 (n = 24) | 12/12 | 564 |
| gated | 0.562 | 0.417 | 0.556 (n = 18) | 9/12 | 101 |

| group | success none / ungated / gated | harm none / ungated / gated |
|---|---|---|
| cross_module | 0.125 / 0.750 / 0.250 | 0 / 0 / 0.250 |
| stale | 0.000 / 1.000 / 1.000 | 0 / 0 / 0 |
| joke_episode | 0.750 / 0.125 / 0.000 | 0 / 0.875 / 1.000 |
| control | 0.000 / 1.000 / 1.000 | — |

- **Paired tests.** ungated vs gated success: 2 / 0 discordant tasks (M01, M03), p = 0.5. Harm: 0 / 2 (J01, M04), p = 0.5. none vs ungated success_all: 3 / 11, p = 0.057.
- **Both downstream rule items fail on H2:** success 0.562 < 0.669, and harm 0.417 > 0.292.
- **Hypothesis: a lone wrong memory misleads more than the same memory among the full top-10,** because contrasting memories let the model dismiss the trap. It rests on two discordant tasks (J01, M04; p = 0.5), and no arm separates "alone" from "accompanied". Check: re-answer the H2 gated tasks with the admitted memory plus the ungated run's contrasting memories, and compare harm with the gated answers.
- **Episodes are the gate's weak spot.** It rejects jokes (0.51–0.54) but admits debugging episodes (0.90–0.93).

## 6. Performance

Sources: `eval/perf/REPORT.md` (its criteria table and verification paragraph were rewritten after an independent read-only verification, and those verdicts are the ones used here; its closing "What holds" list still carries the earlier wording, "holds … CI included", which the verification superseded), `eval/perf/performance.json` (written by `eval/perf/scripts/analyze.py`, mirrored with the verified verdicts in this directory's `performance.json`) and `eval/perf/raw/`. All measured. No generative LLM call and no Anthropic call: the perf harness uses its mock LLM. Decision-model (TypeSafe) use: 510 proxy-counted calls in the context_bench runs (0 errors), plus about 2,500 uncounted requests from the in-process hybrid admission arm (§7); `eval/perf/REPORT.md`'s "510 TypeSafe calls" leaves the in-process arm out.

**Gates.** Verbatim from `PREREGISTRATION.md`: "recall p95 overhead in lifecycle mode < 10% vs B" and "rank-mode recall p95 overhead < 5% vs B". Track 2's 23:50 amendment adds that admission latency p50/p95 is measured "on a quiet machine (no concurrent eval servers), deterministic path and Jev path, 200 recalls each", with the decision-rule item "deterministic-path p95 admission latency ≤ 150 ms".

**Setup.**
- **Runs:** 2026-10-18 02:14–06:48 (05:14–09:48Z), one benchmark at a time, on the developer's Apple M1 Pro (10 cores, 32 GB), macOS 14.7.1. Arm A is base `5331da2f4` with the 3-line `now`-kwarg harness fix from `60c1cc87f` (`raw/base-harness-now-kwarg.patch`); arms B–E and the admission arms are the clean feature worktree `2f4c7daae`. One pg0 instance per tree (PostgreSQL 18.1.0); both venvs hold identical packages.
- **Recall bank:** the perf-test medium recall bank, 1,000 facts, local bge-small embeddings on CPU, the harness's RRF stand-in for the reranker. Scenario `noobs` has no observations. Scenario `obs` adds one consolidation round (100 observations); nothing is retired, so lifecycle mode finds no retired claims.
- **Design:** 10 arms (noobs) and 5 (obs), 4 rounds, odd rounds in arm order and even rounds reversed. Each run discards 16 warm-up recalls, then makes 400 sequential and 200 concurrent (batches of 8) recalls: 1,600 + 800 samples per arm and scenario.
- **Analysis plan:** `raw/analysis_plan.json`. Measured: its file mtime is 02:13:48 (05:13:48Z), 5 s before the first analysed run's machine-state check began and 45 s before that run was cleared to start (05:14:33Z, after a 40 s load wait). Its own `written_at` field says 05:20Z, later than both, so that field is not its write time. It was written after every quality result in §5 was known, and after a tiny smoke run (05:09:34Z) and one medium pilot of arm E (05:10:29Z) had been run and seen; both are excluded from the analysis. It fixes the decision cell before any analysed perf run: the noobs scenario on both shapes, "holds" when the pooled p95 point estimate is under the bar and "holds with CI" when a 95% cluster bootstrap upper bound is too; obs is "reported alongside".
- **Deviation, undeclared:** DESIGN §11.4 asks for the "median of ≥ 5 runs at `--scale large`"; the plan used 4 pooled medium rounds. The pre-registered gates themselves name no scale.
- **Machine:** each run waited while the 1-minute load was above 8 (at most 10 minutes). Background load was about 3–4 cores (browsers, WindowServer, IDE); 3 noobs round-4 runs started above the ceiling after the maximum wait. Quiet in the plan's sense, not idle.

**Verification (from `eval/perf/REPORT.md`; the A/A check was not re-run here).** Every pooled quantile, overhead and CI reproduces within Monte Carlo error. But the analysis plan's cluster bootstrap resamples within runs and ignores between-run variance: in an A/A check (one arm's 4 runs split into two pseudo-arms, 60 comparisons) its 95% CI excluded 0 in 62% of p95 cases against a nominal 5%. So the verdicts are stated as point estimates with run-level intervals. **Recomputed for this report** (`performance.json` `run_level_recomputation`): a paired t interval over the 4 rounds' per-round overheads, and DESIGN §11.4's form (median of per-run p95), for the Track 1 arms against B and for the in-process admission arms against E-ext. It reproduces the run-level intervals the perf report quotes.

Recall overhead vs the reference arm (B, or A for B):

| comparison | scenario | shape | p95, pooled [cluster CI] | p95, run-level t95 | p95, median of per-run | p50, pooled | mean, pooled |
|---|---|---|---|---|---|---|---|
| E lifecycle vs B | noobs | seq | +1.1% [−1.6, +2.6] | [−9.6, +10.6] | +3.8% | +8.6% | +4.9% |
| E lifecycle vs B | noobs | conc | −3.0% [−11.4, +7.1] | [−13.9, +29.9] | +10.7% | +14.2% | +12.6% |
| E lifecycle vs B | obs | seq | +0.9% [−0.3, +2.1] | [−6.6, +7.9] | +1.3% | +7.5% | +4.3% |
| E lifecycle vs B | obs | conc | **+22.3%** [+13.7, +31.3] | [−18.1, +55.0] | +20.9% | +16.6% | +22.8% |
| D rank vs B | noobs | seq | +1.6% [−1.0, +3.2] | [−10.2, +12.0] | +4.5% | +9.1% | +5.7% |
| D rank vs B | noobs | conc | −7.2% [−14.6, −0.3] | [−15.9, +17.0] | +3.9% | +11.4% | +3.1% |
| D rank vs B | obs | seq | +4.9% [+3.8, +6.5] | [−3.8, +11.3] | +6.2% | +12.0% | +10.6% |
| D rank vs B | obs | conc | **+38.9%** [+18.9, +54.1] | [−33.1, +72.3] | +5.9% | +13.4% | +17.3% |
| C observe vs B | noobs | seq | +7.9% [+3.8, +10.9] | [−5.9, +26.9] | +6.0% | +12.3% | +16.0% |
| C observe vs B | noobs | conc | +2.8% [−10.8, +13.2] | [−9.4, +18.3] | +2.8% | +9.4% | +6.2% |
| C observe vs B | obs | seq | +2.5% [+1.3, +3.7] | [−2.6, +6.0] | +3.1% | +7.3% | +5.9% |
| C observe vs B | obs | conc | +7.1% [+1.0, +25.0] | [+7.0, +16.8] | +11.2% | +12.2% | +11.5% |
| B off vs A | noobs | seq | −1.7% [−5.3, +1.9] | [−6.1, +3.7] | −3.2% | −1.4% | −5.9% |
| B off vs A | noobs | conc | +8.8% [+1.0, +19.3] | [−27.5, +28.1] | −6.6% | +0.9% | +4.4% |
| B off vs A | obs | seq | +0.2% [−0.7, +1.1] | [−4.1, +7.7] | +0.5% | −0.7% | +0.4% |
| B off vs A | obs | conc | +1.6% [−2.8, +7.5] | [−11.0, +12.8] | +0.1% | +1.2% | +1.9% |

Absolute pooled p95 (ms), noobs sequential / concurrent: A 266.8 / 735.6, B 262.4 / 800.7, D 266.6 / 742.8, E 265.2 / 776.8. Obs: B 258.4 / 661.6, D 271.2 / 918.8, E 260.6 / 809.2.

**Verdicts (verified).**
- **Lifecycle p95 < 10%:** the point estimate is under the bar on the primary noobs scenario, on both shapes. It is not resolved once run-to-run variance is counted. With observations, the sequential cell is the one Track 1 cell resolved under its bar (+0.9%, run-level [−6.6, +7.9]); the concurrent point estimate is over the bar (+22.3%) and unresolved at run level, while the concurrent p50 overhead with observations is resolved, +17.9% [+7.0, +28.8] at run level.
- **Rank p95 < 5%:** the point estimate is under the bar on the primary scenario, not resolved, and fragile to round 1 (below). With observations, both cells are unresolved at run level; the concurrent point estimate is over the bar (+38.9%) and comes mostly from round 1 (+68%), and the median-of-per-run form gives +5.9%.
- **Some pooled results are implausible,** which the verification attributes to the anti-conservative bootstrap: D faster than B at noobs concurrent p95 (−7.2%, CI excluding 0), and B vs A concurrent p95 +8.8% [+1.0, +19.3] while every other B vs A cell is near 0.
- **Under DESIGN §11.4's form** (median of per-run p95; 4 medium runs, not 5 large), noobs lifecycle concurrent is +10.7% and obs rank sequential +6.2%, both over their bars.

**What the p95 measures (measured).**
- **Garbage-collection pauses.** About 12% of sequential recalls in every arm, A included, carry an extra ~190 ms, so the sequential p95 (about 255–283 ms everywhere) measures that stall, not the per-recall work. In a diagnostic run (`raw/diag`, arm B), 47 CPython gen-2 collections over 100 ms (median 170 ms) fell in a window the length of the sequential phase, and 47 sequential recalls took over 150 ms: a count match, not a per-sample alignment.
- **The feature's per-recall cost shows at the median:** sequential p50 +6.6 ms (+9.1%) for rank and +6.3 ms (+8.6%) for lifecycle; concurrent p50 +11% / +14%.
- **Concurrency.** In the noobs phase traces, `adaptive_state_load` and `lifecycle_claim_load` take about 2 ms p50 and 2.4–3.6 ms p95 sequentially, and 35–45 ms p50 and 275–304 ms p95 under c = 8. Hypothesis, untested: event-loop or connection-pool queueing. Check: an event-loop lag probe and a pre-warmed pool.
- **Round 1 (exploratory sensitivity, written after the pooled results were seen; `raw/sensitivity_rounds.json`).** Round 1 B ran slow in every quantile (sequential p50 85 ms vs 71 ms in rounds 2–4), and C was slow in round 1 too (+40% sequential p50), cause unknown. Without round 1: rank noobs p95 +4.0% [+3.0, +5.4] sequential and +8.1% [−2.8, +13.4] concurrent, so the concurrent point estimate is over 5%; lifecycle +3.6% sequential and +9.5% [−2.0, +22.2] concurrent. Dropping round 4 instead (where the load-ceiling breaches happened) leaves the noobs verdicts unchanged.
- **Observe mode (Track 1).** Observe costs more than rank at the sequential median (+12.3% vs +9.1%), but not without round 1 (+10.2% vs +12.1%).
- **Every C/D/E recall's trace carries its `adaptive.mode`,** and no B recall's does, so the modes were live.

**Context admission latency (Track 2).** Primary: context_bench through the HTTP server, 85 dev queries × 3 passes = 255 recalls per mode, 0 failures, lifecycle mode, recall reinforcement off, the local cross-encoder on CPU; the holdout was not touched. The `plain` variant sends no trace and the admission variants do, so end-to-end latency is confounded and only the phase time is used. Each mode ran once, so the bracket is a within-run bootstrap.

| run | mode | phase p50 | phase p95 [CI] | phase p99 | recall p50 / p95 | abstained |
|---|---|---|---|---|---|---|
| deterministic (E-det weights, 0.85 / 0.90) | admission off (`plain`, no trace) | — | — | — | 812.9 / 985.1 | — |
| deterministic | observe | 0.89 | 3.18 [2.93, 3.39] | 3.89 | 851.1 / 1,026.2 | 39 |
| deterministic | filter | 0.63 | 1.17 [1.04, 1.44] | 1.89 | 814.4 / 943.2 | 39 |
| deterministic | strict | 0.86 | 3.11 [2.94, 3.22] | 3.69 | 829.2 / 984.3 | 39 |
| hybrid (F, 0.80 / 0.85) | strict | 344.9 | 432.5 [409.4, 445.2] | 491.1 | 1,209.4 / 1,363.8 | 66 |
| jev (E-jev, 0.80 / 0.80) | filter | 347.7 | 432.1 [416.9, 451.5] | 586.2 | 1,172.8 / 1,285.8 | 65 |

All in ms. Secondary, in-process on the perf bank (`context_admission` phase, p50 / p95 ms, pooled over 4 rounds, with the range of the per-round p95 where it matters; recall p95 overhead as a pooled point estimate with its run-level t95 interval, as in the Track 1 table above):

| arm | phase, sequential | phase, concurrent (c = 8) | recall p95 vs E-ext, sequential | recall p95 vs E-ext, concurrent |
|---|---|---|---|---|
| E+observe (deterministic) | 2.41 / 3.24 | 40.25 / **273.93** (per round 258–287) | +0.0% [−5.3, +4.4] | +22.1% (+158 ms) [−9.0, +40.4], unresolved |
| E+filter (deterministic) | 0.51 / 0.79 | 0.50 / 0.89 | −4.9% [−9.2, −2.0] | −9.8% [−14.4, −2.6] |
| E+strict (deterministic) | 2.62 / 3.54 | 24.55 / 68.63 (per round 55–196) | −4.2% [−8.8, +1.1] | +1.1% [−14.8, +11.6] |
| E+strict (hybrid) | 274.5 / 357.6 | 276.1 / 380.7 | +104% (+282 ms) [+90.2, +117.5] | +36% (+258 ms) [+17.7, +49.0] |

E-ext is lifecycle recall made the way the HTTP endpoint makes it (`_external=True`, `purpose=auto_context`), admission off. Observe's concurrent phase p95 is stable across rounds (258–287 ms), but its recall p95 overhead is not: per round it is +36.2%, −0.7%, +9.9% and +17.4%, so the pooled +22.1% is a point estimate that is not resolved at run level. The cluster-bootstrap CI that `eval/perf/performance.json` gives it ([+10.3, +36.2]) is the anti-conservative one and is not used. Strict's concurrent phase p95 varies more across rounds (55–196 ms; 196 ms in round 2). Filter and strict return 5–6 results instead of 300, which is why they are faster than E-ext. The in-process hybrid arm abstained on 1,761 of 2,400 recalls because the perf bank has no scope metadata. Hypothesis from code (`context_admission/selection.py`, `pipeline.py`), consistent with the measured gap: strict, and observe (which simulates strict), run redundancy suppression, which reads the admitted candidates' embeddings on its own connection; filter does not.

**Verdict (verified): the deterministic-path p95 ≤ 150 ms holds for sequential traffic,** at 1.2–3.2 ms (about 50× margin), on both benches. Under c = 8 in-process, observe reaches 274 ms p95, which fails the bar, and strict 69 ms pooled (196 ms in one round).

**Retain and consolidation per mode (mock LLM).** The perf-test `retain` and `consolidation` suites at medium, 3 interleaved rounds, medians:

| arm | retain, 1,000 items (vs B) | consolidation, 100 memories (vs B) | consolidation per round |
|---|---|---|---|
| A | 8.12 s, 123.1 items/s (−2.7%) | 4.18 s (−0.6%) | — |
| B | 8.35 s, 119.8 items/s | 4.21 s, 23.8 memories/s | 4.09 / 4.54 / 4.21 |
| C | 8.28 s (−0.8%) | 4.18 s (−0.7%) | — |
| D | 8.37 s (+0.2%) | 4.32 s (+2.7%) | — |
| E | 8.38 s (+0.4%) | 4.94 s, 20.3 memories/s (+17.4%) | 4.63 / 4.94 / 4.99 |

In the obs runs (4 rounds, medians), consolidation took 5.12 s for B, 5.57 s for D (+8.7%), 5.64 s for E (+10.2%) and 5.41 s for A (+5.7%). Retain shows no mode effect beyond ±4%. Lifecycle consolidation is +10–17% and rank +3–9%; the A–B difference of up to ±6% is the noise floor. These are whole-batch times; per-request percentiles are not measured.

**Storage growth (exact row counts, the runs' banks).** After retaining 1,000 facts every arm and mode has the same rows (memory_units 1,000, unit_entities 2,999, memory_links 30.2–30.6 k) and 0 rows in all six adaptive tables. After consolidating 100 observations, A, B and C still have no adaptive rows; D and E add 100 `memory_retention_state` rows (115–123 kB) and 100 `memory_reinforcement_events` rows (172–188 kB), one each per observation, from the `observation_new_evidence` reinforcement source (on by default). Claim state, evidence, relations and shadow stay empty because the mock retires nothing, so growth from real supersession is not measured.

**FK bulk delete** (`fk_delete_bench.py`, pg0, 50,000 units, feature mode off, 3 repeats with base and feature interleaved, median seconds; commit ≤ 3 ms everywhere):

| scenario | base (3 FKs) | feature (13 FKs) | Δ | ratio | Δ per row |
|---|---|---|---|---|---|
| empty side tables, SQL | 0.245 | 1.115 | +0.87 s | 4.55× | +17.4 µs |
| base side rows (30 links per unit = 1.5 M, 3 unit_entities per unit, 10 k entities), SQL | 2.590 | 3.975 | +1.38 s | 1.53× | +27.7 µs |
| all side rows (+ dense adaptive rows), SQL, feature only | — | 4.254 | +1.66 s vs base base_side | 1.64× | — |
| engine `delete_bank`, base side rows | 2.917 | 3.741 | +0.82 s | 1.28× | — |
| engine `delete_bank`, all side rows | — | 4.544 | +1.63 s vs base | 1.56× | — |

- The 10 new FKs are 6 CASCADE, DEFERRABLE INITIALLY DEFERRED (retention_state, claim_state.memory_id, claim_evidence ×2, reinforcement_events, claim_shadow) and 4 immediate SET NULL (claim_state.superseded_by_id and inherited_from_id, lifecycle_relations source and target).
- P1's earlier measurement (0.418 → 1.867 s, 4.47×) is reproduced in ratio (4.55×).
- At realistic link density the relative cost falls, but the absolute cost per deleted unit is +17–33 µs. Projected, not measured: about +17–35 s per 1 M-unit bank delete.
- Relations survive a SQL delete as SET NULL tombstones (5,000 in the all-side scenario).

**Measured earlier, under load (indicative only):** the admission phase on the context_bench holdout runs, at load averages 53 / 68 / 101: deterministic p50 / p95 / max 1.0 / 2.7 / 3.8 ms, E-jev 359 / 426 / 596 ms, F 361 / 434 / 538 ms (`context-bench/t2/holdout/holdout-primary.json`). The MMR redundancy pass over 30 candidates takes 0.57 ms at 384 dimensions and 1.56 ms at 1,536 (`impl/T2-NOTES.md`).

R39 coverage (REQUIREMENTS R39 lists retain/recall/consolidation p50/p95/p99, DB queries, write QPS, storage growth, LLM calls and tokens):

| R39 metric | status | source |
|---|---|---|
| recall p50 / p95 / p99 | measured | `performance.json` `recall_noobs`, `recall_obs` |
| retain p50 / p95 / p99 | not measured per request; whole-batch time and items/s per round | `official_suites`; obs runs' `retain_sync_s` |
| consolidation p50 / p95 / p99 | not measured per request; whole-round time and memories/s (mock LLM) | `official_suites`; obs runs' `consolidation_s` |
| storage growth | measured without retirements (rows and bytes after retain and after consolidation); growth from real supersession not measured | obs runs' `table_rows`, `table_bytes_round1` |
| write QPS | retain throughput only: A 123.1, B 119.8, E 119.3 items/s | `official_suites` |
| DB queries (statements per recall) | not measured: neither `eval/perf/scripts` nor `hindsight-dev/benchmarks/perf` reads `pg_stat_statements` | — |
| WAL bytes (DESIGN §11.4) | not measured: no script collects it | — |
| LLM calls and tokens | measured (§7), except the in-process hybrid admission arm's decision-model requests (about 2,500, at least one per recall), which were not counted and whose tokens were not recorded; 0 generative calls on recall in every mode (asserted by a test, `R59-CHECKLIST.md` row 7) | `performance.json` `llm_calls_and_tokens` |

## 7. Costs

Real-LLM spend ledger. The self-imposed cap was $150. The account credit ran out at 01:09:41 on 2026-10-18; the user added $4 at 02:01 and asked for essential work only (judged E and this report).

| item | spend | source |
|---|---|---|
| AMB validity-gate pilot | $2.60 | `eval/amb-pilot.md` §7 |
| freeze of the arm-A banks (211 LongMemEval + 10 LoCoMo archives) | $32.11 (run 1 $12.13 + run 2 $19.98) | STATE 2026-10-17 14:30 |
| classification eval, P3 (162 runs plus pilot, server and judge) | $3.16 | `eval/classify/REPORT.md` |
| observations-on consolidation, 2 conditions × 40 KU banks | ≈ $41: a projection from the measured $0.40 (base) and $0.62 (lifecycle) per bank; no final total was recorded | STATE 2026-10-17 15:35 |
| classification eval re-run, P5 (54 runs) | $0.954 (server $0.570 + judge $0.384) | `eval/classify/raw-p5/score.md` |
| judged AMB before the outage (A×2, B×2, D×2, B-obs×2, E-obs×2, and the unused partial E-r1, $1.218) | $10.17 | computed from `eval/judged/usage-judged-*.jsonl` with `eval/scripts/amb_cost.py` (rows before 06:00) |
| judged E re-run after the top-up (E-r1 $1.358 + E-r2 $1.358, 07:07–07:55) | $2.72 | same script, rows after 06:00 (`usage-judged-E-r1.jsonl` holds both the partial and the re-run) |
| harm benchmark (5 runs, of which gated original $0.214 and gated H2 $0.140 after the top-up) | $1.32 | `eval/harm-bench/summary.json`, `eval/harm-bench/REPORT.md` |
| R55 last-write-wins, 12 banks attempted | not recorded. Hypothesis: at most about 12 × $0.62 ≈ $7.4 | STATE 2026-10-18 02:00 |
| `hs_llm_core` lifecycle judge tests (4) | not recorded ("cents", P3 notes) | `impl/P3-NOTES.md` |
| perf workflow, LoCoMo arms, retrieval endpoints, memory_lifecycle, context_bench | $0 Anthropic | — |
| **total recorded** | **$53.03**, plus ≈ $41 projected, plus the unrecorded R55 and judge-test items | — |

**Spend after the top-up (measured for this report):** $3.07 of the $4, that is $0.354 for the gated harm answers and $2.716 for the judged E re-run. STATE's 07:56 note estimated about $2.75 in total (judged E about $2.4); the recomputation from the usage logs with the project's price table is $0.32 higher. Hypothesis: about $0.93 of the top-up is left, if `amb_cost.py`'s prices match billing; check: the Anthropic console balance. The judged-run figures count AMB-side answer and judge tokens; the servers import frozen banks with observations off, so they make no extraction or consolidation calls (hypothesis from the run configuration; check: the `llm_requests` rows of the srv-e banks).

TypeSafe (Jev) usage:
- context_bench baselines: 962 calls, 2,674,713 input / 658,137 output tokens (`context-bench/summary.json`);
- T2 holdout: 92 calls;
- harm-bench gated runs: 48 + 32 calls;
- perf, counted: 510 calls through the counting proxy in the context_bench runs (255 hybrid + 255 jev);
- perf, not counted: the in-process E+strict (hybrid) arm used the real decision model on every recall, outside the counting proxy (`run_arm.py` passes the real `TYPESAFE_API_KEY` with no base-URL override): 2,400 analysed recalls, 64 warm-ups and a 32-recall smoke run, so about 2,500 more requests and about 3,000 for the perf workflow in all. Measured (`performance.json` `llm_calls_and_tokens.perf_runs`): the runs' env sets the `typesafe` provider, every one of the 2,400 analysed recalls spent over 100 ms in the admission phase (p50 274.5 ms), and none recorded a `jev_error`. That each made at least one request is inferred from those measurements and the code; their tokens were not recorded;
- 0 non-200 responses among the counted calls, and 0 `jev_error` in the in-process arm.

TypeSafe was not priced from any invoice. Upstream PR [#5664](https://github.com/vectorize-io/hindsight/pull/5664) states a price of $0.042 per million input tokens with output free; at that rate the context_bench baselines' input would cost about $0.11 (hypothesis: upstream's stated price, not checked against billing).

Token and time overheads:
- **Lifecycle consolidation input tokens.** +100% at P3, where the classification eval measured 2.0×. +52% at P5, after the schema `$defs` hoist: 2.83 M vs 1.86 M over 169 calls each, or 16.8 k vs 11.0 k per call. The number of calls is unchanged.
- **AMB observations-on consolidation at P3:** $0.62 vs $0.40 per bank (+55%), at about 143 vs 140 calls.
- **The schema alone, measured with the local tokenizer** at a pool of 10 facts / 50 retirable ids / 120 shown sources: 1,229 tokens off, 16,841 lifecycle before the hoist, 5,810 after.
- **Consolidation time on the mock LLM:** lifecycle +10–17%, rank +3–9% (§6). This excludes model time, which the token overhead above drives.
- **Admission reduces injected tokens.** Harm original: 51 vs 581 per task. context_bench holdout tokens/query: F 25.5 vs 35.3 for tags + Jev (exploratory).
- **Recall:** no generative tokens; median latency +9% sequential for rank and lifecycle (§6).

## 8. Pre-registered decision rules, applied mechanically

Rules: `eval/PREREGISTRATION.md`, current sha256 `8ca8f269…7e17`; provenance of the text in §13. Each subsection gives the mechanical outcome first and the author's judgment second, labelled as such.

### 8.1 Track 1: "opt-in lifecycle" is recommended iff all four items hold

| # | item | measured | result |
|---|---|---|---|
| 1 | memory_lifecycle holdout stale@5: E ≤ B − 0.10; refuted@5 = 0; hist@5 ≥ B − 0.05 | stale@5 B 0.200 → E 0.000 (−0.200) on both encodings; refuted@5 E 0.000; hist@5 1.000 vs 1.000 | **holds** (stub-scripted conformance suite) |
| 2 | P1-life significant in favour of E (Holm p < 0.05) | 0.225 → 0.825, 0/24, Holm p = 2.4e-7, after the post-hoc source repair (§5.3). On the rows as recorded: 0.850 vs 0.900, 0/2, p = 0.5 | **holds** under the pre-specified mapping rule, but the significance depends entirely on the post-hoc repair, done after an interim scoring at 17:09. The exploratory re-scoring in 5.3 also shows the effect is mostly an attribution effect |
| 3 | real-model false retirement: one-sided 95% CP upper bound < 10% | 0/51 scenario-runs, upper 5.7% (P3). The cluster unit, 0/17 with upper 16.2%, would fail, but it is not the pre-declared unit. P5 re-run: 0/17, upper 16.2% (underpowered alone) | **holds** (pre-declared unit) |
| 4 | recall p95 overhead in lifecycle mode < 10% vs B | Decision cell fixed by `eval/perf/raw/analysis_plan.json` before any analysed perf run (after tiny smoke runs and one medium pilot of arm E, both excluded): no observations, both shapes, pooled point estimate. Sequential +1.1%, concurrent −3.0%. Run-level t95 [−9.6, +10.6] and [−13.9, +29.9]. With observations (reported alongside): sequential +0.9% (run-level [−6.6, +7.9], resolved under the bar), concurrent +22.3% (run-level [−18.1, +55.0]) | **holds** as a point estimate on the decision cell; not resolved at run level (§6) |

**Outcome, mechanical: all four items hold, so the rule as written recommends "opt-in lifecycle".** Item 2 holds only through the post-hoc source repair, and item 4 only as a point estimate on the analysis plan's decision cell. That plan was written after every quality result was known but before any analysed perf run (after tiny smoke runs and one medium pilot of arm E, both excluded), and the pre-registration itself names no scenario or shape for item 4.

**Author's judgment on item 2: it should not count as established.** The endpoint's construct fails on the reference arm: base consolidation states the update inside one dated observation (measured on a hand-read sample, §5.3), and the rule scores that as a miss. Under the two alternative attributions there is no significant difference (newest source 0.925 vs 0.875; raw facts only 0.850 vs 0.900). The judged answer accuracy (0.950 vs 0.925 with observations, 0.905 vs 0.891 without) and the upstream removal of a recall-time stale-value filter after its knowledge-update and contradiction scores did not move (#5284 → #5632, §9) point the same way.

**Author's judgment on the fallback: it depends on the margin reading.** With item 2 not significant, the rule weakens the claim to "safe opt-in, benefit unproven on LongMemEval" only if "E is not worse on any primary or secondary deterministic endpoint beyond the non-inferiority margin" (3 points, point estimate).
- **Literal reading, every primary or secondary deterministic endpoint: the fallback fails.** KU-obs earlier@5 falls from 1.000 to 0.875 (−12.5 points, 5/0, p = 0.0625; history leaving the current top 5 by design), and memory_lifecycle current recall@5 on the all split falls from 0.983 to 0.951 (−3.2 points; the error-injection scenario; holdout 1.000 = 1.000).
- **Lenient reading, only the named non-inferiority endpoints** (TR gold@20, all_gold@20, KU earlier@20, LoCoMo gold@20): **the fallback holds**; every one is unchanged.
- **Correction:** the previous draft applied the lenient reading without flagging it. §8.2 already treats the same ambiguity as making rank's branch b non-confirmatory, so the fallback is treated the same way here.

**Author's judgment on item 4: not established.**
- It is unresolved at run level on the decision cell.
- It is over the bar, as a point estimate, in the configuration this report would recommend (observations on) under 8 concurrent recalls: +22.3%. Sequentially, in that configuration, it is resolved under the bar (+0.9%, run-level [−6.6, +7.9]).
- It is over the bar in DESIGN §11.4's median-of-per-run form (+10.7%, no observations, concurrent).
- The sequential p95 mostly measures garbage-collection pauses that every arm shares, so a small p95 overhead does not show a small per-recall cost; the median cost is +8.6% sequential and +14.2% concurrent.

**Outcome, author's judgment: not "opt-in lifecycle", and not "safe opt-in" either; the rule's residual category, "experimental".** Reaching "safe opt-in, benefit unproven" needs the lenient reading of the fallback margin and the point-estimate reading of item 4; with either strict reading the rule's outcome is "experimental / observe-only".

### 8.2 Track 1: "opt-in rank" is recommended iff (a or b) and c

| # | item | measured | result |
|---|---|---|---|
| a | P1-rank significant in favour of D | 0.819 → 0.861, 0/3, Holm p = 0.25 | **fails** |
| b | memory_lifecycle stale@1 improves with no endpoint regression beyond the 3-point margin | Base-delete: stale@1 0.222 → 0.167 (all), 0.286 → 0.143 (holdout). Base-update (also pre-registered): 0.111 → 0.111 (all), 0.143 → 0.143 (holdout), with every k = 1 metric identical to off. Regressions: D-obs latest_first 0.225 → 0.175 (−5.0 points, 2/0, p = 0.5); LoCoMo gold@5 −1.6 points (32/8, p = 1.8e-4); KU earlier@5 −1.4 points | **does not hold on base-update; non-confirmatory on base-delete** (see below) |
| c | rank-mode recall p95 overhead < 5% vs B | Same decision cell: sequential +1.6%, concurrent −7.2% (run-level [−10.2, +12.0] and [−15.9, +17.0]). Without round 1 (exploratory): concurrent +8.1%. With observations: sequential +4.9% (run-level [−3.8, +11.3]), concurrent +38.9% (run-level [−33.1, +72.3]) | **holds** as a point estimate on the decision cell; author's judgment: not established (§6) |

Why branch b is non-confirmatory:
- **The criterion was written after its own data were known.** The pre-registration's header says the "P1 off/observe/rank results on memory_lifecycle were already known". STATE's P1 entry (~05:40) recorded "rank vs off: k=1 stale 0.222->0.167 … k>=3 unchanged", and `p1/arm-rank-base-delete.json` is from 05:46, before the ~09:45 pre-registration.
- **It replaced a criterion D does not meet.** DESIGN §11.5 asked for "stale_contamination@5 drops", and D's stale@5 is unchanged: 0.233 = 0.233 (all) and 0.200 = 0.200 (holdout).
- **The improvement exists only on base-delete.** On base-update rank changes nothing at k = 1.
- **Even on base-delete it depends on the margin reading.** The rule does not say which endpoints the margin covers: b fails if it covers every pre-registered endpoint (D-obs −5.0 points), and holds if it covers only the named non-inferiority endpoints (TR gold@20, all_gold@20, KU earlier@20, LoCoMo gold@20, all unchanged).

**Outcome, mechanical: "opt-in rank" is not met.** Item a fails, and branch b does not hold on the pre-registered base-update encoding and is non-confirmatory on base-delete. Item c holds as a point estimate, but the rule needs (a or b) and c. The mechanical outcome for rank is "experimental / observe-only". The author's judgment on c (not established) does not change it.

### 8.3 Track 1: optional pieces

The pre-registered rule covers archive, recall reinforcement, historical cues, observe shadow and conflict policy: keep a piece only if its ablation improves at least one endpoint without regressing another beyond the margin, and otherwise report "no measured benefit".

| piece | ablation measured | mechanical outcome |
|---|---|---|
| retention decay (alpha 0.1 vs 0). **Author's extension of the optional-piece rule:** decay is a frozen parameter, not a pre-registered optional piece | **Improves (point estimates only):** P1-rank latest_first +4.2 points (0/3, not significant); memory_lifecycle k = 1 supers@1 0.889 → 0.944 and prec@1 0.931 → 0.948 (all; holdout supers@1 0.857 → 1.000), and stale@1 in rank (base-delete only); KU-obs latest@5 +2.5 points vs nodecay (1/0). **Regresses:** KU-obs earlier@5 −5.0 points vs nodecay (0/2, p = 0.5); LoCoMo gold@5 −1.6 points and gold@10 −0.8 points (the alpha = 0 LoCoMo arm is exploratory) | under the extension, **depends on the margin reading:** "no measured benefit" if the margin covers every endpoint (earlier@5 −5.0 points); "kept" if only the named non-inferiority endpoints (LoCoMo −1.6 points is inside 3) |
| historical-cue detector | `lifecycle-cues` == `lifecycle` on every metric; the bench sends explicit views, so the detector never decides | **no measured benefit** (and not exercised) |
| historical view as the default (pre-registered ablation, not an optional piece) | stale@5 0.411 (all) / 0.400 (holdout), refuted@5 0, hist@5 1.000 | by design: history is shown; not a candidate default |
| archive / prune | no benchmark arm (system story `test_101` only) | **no measured benefit** (not measured) |
| recall reinforcement | no arm. Upstream rejected usage-count reinforcement three times, for non-deterministic recall and write load (§9) | **no measured benefit** (not measured) |
| observe shadow | no arm (system story `test_103` only) | **no measured benefit** (not measured) |
| conflict policy `latest` (R55 last-write-wins) | not run by user decision (8/40 banks, unused) | **no measured benefit** (not measured) |
| D reinforcement off (pre-registered memory_lifecycle ablation) | not run; no reason recorded | — |
| R55 recency-decay-only baseline | not run; no reason recorded. Hypothesis: on banks without reinforcement events D reduces to a decay-only arm, but it was not run as a separate arm; check: count `memory_reinforcement_events` rows on a D bank | — |

**Measured facts about decay, independent of the reading:**
- Its only statistically significant effect is the LoCoMo regression: gold@5 32/8 (p = 1.8e-4) and gold@10 15/3 (p = 0.0075).
- With alpha = 0, E equals B exactly on LoCoMo: 0 discordant at every k (exploratory arm, added after the regression was seen).
- With alpha = 0, E-obs is unchanged on the primary endpoint (latest_first 0.825 both, 0/0; pre-registered arm).
- memory_lifecycle at k = 5 is identical with and without decay (pre-registered ablation).

Every optional piece ships default-off, except decay, which defaults to alpha 0.1. The user's rule is to remove a piece only on measurement. Measurement shows no benefit for decay that survives a significance test.

### 8.4 Track 2

| rule item (holdout) | E-det (exploratory) | E-jev | F (exploratory) |
|---|---|---|---|
| pollution ≤ 0.20 | 0.486, fails | 0.263, fails (exploratory) | 0.240, fails |
| abstention ≥ 0.80 | 0.625, fails | 0.875, holds | 1.000, holds |
| useful recall ≥ 0.85 | 0.761, fails | 0.848, fails (confirmatory) | 0.804, fails |
| gated downstream success ≥ max(none, ungated) − 0.05 | — | — | original 1.000 ≥ 0.950, holds; H2 (exploratory) 0.562 < 0.669, fails |
| gated harm ≤ ungated harm | — | — | original 0 ≤ 0, holds; H2 0.417 > 0.292, fails |
| deterministic-path admission p95 ≤ 150 ms (255 sequential recalls per mode, §6) | observe 3.18 ms, filter 1.17 ms, strict 3.11 ms: **holds** (sequential); under c = 8 in-process, observe 274 ms | — (Jev path: about 432 ms p95) | — (hybrid path: about 432 ms p95) |
| **safe for automatic injection** | **no** | **no** | **no** |
| opt-in filter: pollution drop vs tags + Jev (0.360) ≥ 0.15 | −0.126, fails | +0.097, fails | +0.120, fails |
| opt-in filter: useful recall ≥ 0.80 | fails | holds | holds |
| **opt-in filter** | **no** | **no** | **no** |

**Outcome, mechanical: neither positive branch holds.** The rule's remaining branch is "observe only" or "do not use", and the rule does not choose between them. The latency item holds, but it cannot change the outcome, because the quality items already fail. **Author's judgment on the latency item:** it holds for sequential traffic only; observe's concurrent tail (274 ms phase p95 at c = 8) matters for §8.5's choice of observe.

### 8.5 Recommended defaults (the author's judgment where the rule leaves a choice)

**Track 1: "experimental" (the rule's residual "experimental / observe-only" category). The default for every bank stays `off`, and no safe default is claimed.** A bank may enable lifecycle mode as an experiment, with observations on, at alpha = 0, and with modest recall concurrency. This replaces the previous draft's "safe opt-in, benefit unproven, conditional on perf", for the three reasons in §8.1: item 2 is not counted (attribution artifact), the fallback holds only under the lenient margin reading, and item 4 is not established. The previous draft said the recommendation becomes "experimental / observe-only" if perf failed the 10% bar; perf did not fail it on the plan's decision cell, but it is unresolved there and fails as a point estimate in the recommended configuration under concurrency.

What an experimental bank gets (measured):
- off reproduces base recall and transcripts exactly, and off-mode recall latency is within base's noise floor (deletes are slower, §6);
- on the stub-scripted holdout, lifecycle removes stale and refuted contamination with history intact, and gives `historical`, `as_of` and refutation semantics that base lacks (§3);
- the real model wrongly retired a protected memory in 0 of 51 control runs (one-sided 95% upper bound 5.7%) and kept environment-divergent claims apart where B failed 3/3.

What it costs, or what was not shown (measured unless labelled):
- no measured answer-accuracy gain, with or without observations (0.925 vs 0.950; 0.905 vs 0.891);
- the primary effect is largely an attribution effect (exploratory), and its significance rests on the source repair;
- some real-model labels were wrong (a wrong `uncertain`, a cross-service conflict, refute recall 1/3 strict);
- +52% consolidation input tokens and +10–17% consolidation time on the mock LLM;
- recall median +8.6% sequential; with observations, sequential p95 +0.9% (resolved under the bar at run level), and under 8 concurrent recalls p95 +22.3% (unresolved at run level) and p50 +17.9% at run level [+7.0, +28.8] (resolved);
- a wrong supersession hides more;
- bulk deletes are slower in every mode, off included (+17–33 µs per deleted unit on pg0);
- Oracle has not run since P1: every P2–T2 Oracle path is checked statically only.

Why alpha = 0:
- decay's only significant effect is the LoCoMo regression, and alpha = 0 removes it exactly. That arm is exploratory (decided after the regression was seen), so this part of the recommendation rests partly on an exploratory arm;
- decay adds nothing on the primary endpoint or on memory_lifecycle at k = 5 (both pre-registered arms);
- upstream saw the same mechanism with its own recency boost, which flipped a correct top-1 (#3893), and its maintainers doubt forgetting as a feature (discussion #2921) (§9).

Why rank mode is not recommended on its own: the opt-in-rank rule is not met (item a fails; branch b does not hold on base-update and is non-confirmatory on base-delete), by design decay is all that rank adds on banks without reinforcement events, and its tail latency with observations under concurrency is the worst measured (+38.9% p95 pooled, mostly round 1).

Observe mode for Track 1 is not a cheap data collector: it does not run lifecycle classification, and it costs about as much as rank at recall (sequential p50 +12.3%, +10.2% without round 1).

**Track 2: observe-only, experimental, not for automatic injection.**
- The gate fails every quality item of both positive branches on the holdout; the confirmatory failure is E-jev's useful recall 0.848.
- Its useful behaviours (abstention, fewer polluted queries, fewer tokens) are exploratory.
- On the exploratory H2 suite, the gate increased harm.
- Observe mode leaves payloads byte-identical, so it can collect `would_drop` data on real traffic for a better scorer. With the deterministic scorer it costs 3.2 ms phase p95 and nothing measurable at recall p95 for sequential recalls, but under 8 concurrent recalls its phase reaches 274 ms p95, and recall p95 rises by a pooled 22% (+158 ms) that is not resolved at run level (per round −0.7% to +36.2%; §6). It sends nothing out, but that scorer has the known joke defect. With the jev or hybrid scorer it adds one decision-model request per recall (about 345 ms p50, 432 ms p95) and sends each candidate's memory text and metadata to that provider.
- `filter` must not be used where repository isolation matters, because filter is not a scope gate.
- Upstream policy points the same way: recall answers a match with something, never nothing (#3704), and its maintainers rejected a pre-reflect relevance gate on latency (#4558) (§9).

## 9. Prior work upstream

Source: `eval/PRIOR-WORK.md` (sha256 `896ff2e6…9f01`; file mtime 2026-10-18 02:35:17, after every quality result in §5 and while the perf workflow was running), a verified synthesis that re-opened the sources at base `5331da2f4` (blog posts, code, docs) and on GitHub through `gh`: 61 issues and PRs, discussions #2921 and #2992, and 10 commits, all ancestors of base. Its verification pass corrected several labels of an earlier sweep (`eval/prior-work-items.json`); this section uses the corrected ones. The upstream numbers are the upstream authors' measurements and were not reproduced here. The raw GitHub downloads are in the scratchpad (`verify/`).

**Retention decay and reinforcement.**
- **A bounded recency boost already exists** in base: ±10%, linear over 365 days or exponential with a 90-day half-life ([#2318](https://github.com/vectorize-io/hindsight/pull/2318)). It is anchored on occurrence time; this fork's retention factor is anchored on last reinforcement, a different signal stacked on top. #2318 kept the default and said that changing it would need a LoCoMo or LongMemEval run; the LoCoMo ablation here is that kind of evidence, and it is negative for an extra age penalty (§5.5).
- **A recency boost already reversed a correct reranker order** ([#3893](https://github.com/vectorize-io/hindsight/issues/3893)): the same mechanism as this fork's LoCoMo regression, whose worse-at-5 queries have older gold items.
- **Maintainers are sceptical of forgetting as a feature**, without a measurement ([discussion #2921](https://github.com/vectorize-io/hindsight/discussions/2921)). A contributor's Ebbinghaus prototype linked there was never merged or measured.
- **Usage-count reinforcement was rejected three times** for non-deterministic recall and database write load ([#793](https://github.com/vectorize-io/hindsight/issues/793), [#795](https://github.com/vectorize-io/hindsight/pull/795), [#3093](https://github.com/vectorize-io/hindsight/pull/3093)); the unused `access_count` column was dropped in [#3157](https://github.com/vectorize-io/hindsight/pull/3157). This fork keeps recall reinforcement off by default and never benchmarked it, which fits that history; its other reinforcement sources write one retention row and one event per new observation (§6). Evidence independence is still an open upstream issue ([#5705](https://github.com/vectorize-io/hindsight/issues/5705)).

**Truth lifecycle.**
- **A recall-time stale-value filter was built and then removed** ([#5284](https://github.com/vectorize-io/hindsight/pull/5284) → [#5632](https://github.com/vectorize-io/hindsight/pull/5632)). It was removed because it depended on English (regexes, month names, stopwords); upstream's measurement on 40 questions found knowledge-update and contradiction questions scoring the same with and without it. That agrees with the judged null here (0.950 vs 0.925 with observations, 0.905 vs 0.891 without). This fork classifies inside the existing consolidation call instead, which is language-agnostic but, on this evidence, not better for answers either.
- **Validity windows and `as_of` were proposed several times and closed unmerged** ([#1391](https://github.com/vectorize-io/hindsight/issues/1391), [#1878](https://github.com/vectorize-io/hindsight/pull/1878), [#2174](https://github.com/vectorize-io/hindsight/pull/2174)). #2174 is prior art for the same surface (current and historical views, `as_of`, automatic supersession) and was closed with no recorded reason, so it is not a refutation. A `fact_state` proposal closed as not planned ([#3509](https://github.com/vectorize-io/hindsight/issues/3509)).
- **Upstream assigns supersession to consolidation** ([#1951](https://github.com/vectorize-io/hindsight/issues/1951)), prefers reversible manual invalidation with no state predicate on the read path ([#1976](https://github.com/vectorize-io/hindsight/pull/1976)), and its consolidation rules never let an older fact overwrite a newer belief ([#5627](https://github.com/vectorize-io/hindsight/pull/5627)). This fork's claim filter is exactly a read-path state predicate; its measured cost is about 2.5 ms sequential and 45 ms p50 / 304 ms p95 under 8 concurrent recalls (§6).

**Context admission.**
- **Jev pruning ships, opt-in** ([#4522](https://github.com/vectorize-io/hindsight/pull/4522); [blog 2026-09-24](https://github.com/vectorize-io/hindsight/blob/5331da2f4db369528bb122075341a7de9c3df6df/hindsight-docs/blog/2026-09-24-adding-jev-reranker-what-we-learned.md)). Upstream's only end-to-end check went from 8/10 to 7/10 answerable, and it was not enabled by default. Only the cut exists upstream: abstention, hard scope, MMR and purpose budgets do not (MMR appears in docstrings only).
- **Gating on Jev's confidence was measured and dropped** (#5284, maintainer comment). This fork thresholds the per-candidate relevance probability, a different field, so it is related evidence, not a refutation.
- **Recall policy: when anything matched, recall returns something, even past the token budget** ([#3704](https://github.com/vectorize-io/hindsight/pull/3704)), because an empty answer reads to agents as "no such memory". Strict-mode abstention conflicts with it unless it stays opt-in.
- **A tag-scope gate with abstention was built and closed** because enumerating tags does not scale ([#4009](https://github.com/vectorize-io/hindsight/pull/4009)); `any_strict` tag matching already gives hard scope ([discussion #2992](https://github.com/vectorize-io/hindsight/discussions/2992)).
- **A pre-reflect relevance gate was rejected on latency** ([#4558](https://github.com/vectorize-io/hindsight/issues/4558)). The Jev path measured here costs about 345 ms p50 and 432 ms p95 per recall on a quiet machine (§6), which supports that objection.
- **Existing baselines:** score floors exist in the integrations ([#2575](https://github.com/vectorize-io/hindsight/pull/2575), [#4145](https://github.com/vectorize-io/hindsight/pull/4145), [#5087](https://github.com/vectorize-io/hindsight/pull/5087)), absolute floors lose gold and `final` includes recency ([#5294](https://github.com/vectorize-io/hindsight/issues/5294)), `prefer_observations` was built for deduplication, not for ordering updates ([#2311](https://github.com/vectorize-io/hindsight/pull/2311)), and reading only observations loses current answers ([#5226](https://github.com/vectorize-io/hindsight/issues/5226)). Each matches what §5.1, §5.3 and §5.7 measured for those baselines here.

**Against the Jev blog post ([2026-09-24](https://github.com/vectorize-io/hindsight/blob/5331da2f4db369528bb122075341a7de9c3df6df/hindsight-docs/blog/2026-09-24-adding-jev-reranker-what-we-learned.md)).**
- **Agrees:** pruning keeps very few candidates (1.6 of 30 in the blog; 1.6–1.7 per query here); the prune never returns an empty list (0/8 abstention here); giving the model a "nothing" option costs recall (here 0.978 → 0.804 or 0.848); an absolute threshold has no stable meaning (thresholds drifted from dev to holdout); filtering is a separate, default-off decision (equal success with 91% fewer tokens on the easy harm suite).
- **Disagrees:** the blog calls the score a calibrated probability, but here its ECE is 0.188 and its Brier 0.068, worse than a constant base-rate score (0.033), at AUC 0.995. The blog's 17× precision gain is 5.8–6.3× here (0.099 → 0.571 / 0.625). Its 19% gold loss did not reproduce (useful recall 0.978 → 0.957 with Jev alone, 0.978 with tags + Jev). Its 27–63 ms per query is 345 ms p50 here, 5.5× its 240-candidate figure and 13× its 30-candidate one. The exploratory H2 suite contradicts "a weak memory beats silence" at both ends (§5.8).

**What is new here, relative to upstream.**
- the abstention side of admission, measured: 8 holdout queries that should get nothing, Jev prune 0/8 vs F 8/8 (exploratory), polluted queries 24 → 10 of 46;
- a downstream harm benchmark for gating;
- a decay ablation on LoCoMo and LongMemEval: its only significant effect is negative, and on LoCoMo alpha = 0 equals base exactly;
- a real-model false-retirement rate for consolidation-time lifecycle labels (0/51 control runs);
- calibration and run-to-run variation of the Jev score;
- the large lifecycle ranking gain the synthesis lists as new (latest_first 0.225 → 0.825) is withdrawn here: it is an attribution artifact (§5.3).

**Where it bears on the conclusions:** the judged null and #5632 together argue against claiming an answer benefit (§8.1); #2318, #3893 and #2921 support alpha = 0 (§8.5); #793/#3093 support keeping recall reinforcement off; #3704, #4558 and #4522 support keeping admission observe-only (§8.5); and §11 cites the items that bear on each explanation.

## 10. Known limitations

- **Oracle is verified live only for P1.** P1's migration, FK delete rules and delete-path engine tests passed on a live Oracle 23ai Free (8 passed) before the container was removed. Every later Oracle path (claim CAS, inheritance, archive, transfer, fold, maintenance) is covered only by SQL-rewriter and fake-connection tests, because starting the container had OOM-killed the user's MySQL. The periodic maintenance loop does not run on Oracle.
- **Errors are amplified by a wrong supersession.** The raw facts a wrongly retired observation stood on leave current recall too. memory_lifecycle rec@5 fell from 0.983 to 0.951 under the `supersede_still_true` injection. Recovery is by `PATCH truth_status=active`.
- **FK delete cost in every mode.** Bulk deletes take 17–33 µs more per deleted unit on pg0, even with the feature off: for 50,000 units, 0.245 → 1.115 s with empty side tables and 2.59 → 3.97 s with realistic side rows (engine `delete_bank` 2.92 → 3.74 s). About +17–35 s per 1 M-unit bank delete is a projection, not a measurement; large scale and Oracle are not run.
- **The recall p95 gates are not resolved.** Four pooled medium rounds did not resolve a 5–10% p95 overhead: on the decision cell (no observations) every run-level interval straddles its bar, and with observations only lifecycle sequential is resolved, under its bar (+0.9%, run-level [−6.6, +7.9]); the other three observation cells straddle theirs (§6). The analysis plan's cluster bootstrap is anti-conservative (62% false positives in an A/A check), the p95 mostly measures garbage-collection pauses, round 1 was slow for unknown reasons, and the machine was quiet only in the plan's sense (1-minute load ≤ 8 on 10 cores, browsers and an IDE running). The gate form of DESIGN §11.4 (at least 5 runs at `--scale large`) was not run.
- **Latency under concurrency.** With 8 concurrent recalls, the adaptive state and claim loads go from about 2 ms to 35–45 ms p50 and 275–304 ms p95, and admission observe's phase reaches 274 ms p95. The cause is not established (hypothesis: event-loop or pool queueing, §6).
- **The perf banks have no retired claims.** The mock LLM retires nothing, so the claim filter's cost and storage growth were measured with empty claim tables; a bank with many supersessions may cost more.
- **The deterministic admission scorer admits jokes.** It has no category or intent signal and reads the cross-encoder as relevance. The A6 joke scores 0.862 and is admitted (strict xfail). Re-fitting the weights cannot add the missing signal.
- **Filter mode is not a scope gate.** A candidate from another repository with reranker 0.99 / semantic 0.95 scores 0.709 and is admitted for auto_context. Only `strict` enforces the repository. Memories without repository metadata pass (fail open).
- **Jev is non-deterministic near the threshold.** It takes one decision-model sample per candidate. Across samples, the mean |Δp| is 0.008 and the max 0.11. Dev replays matched 80/85 (E-jev) and 84/85 (F) admitted lists, and one harm-bench replay mismatched (N02).
- **The gate can leave a wrong memory alone in context.** On H2 it admitted a lone wrong-module procedure (M04) or a debugging episode (J01–J04) with nothing beside it, and harm rose from 0.292 to 0.417 (exploratory, p = 0.5). That the isolation causes the harm is a hypothesis (§5.8).
- **The currentness gate was not exercised by context_bench.** Every candidate was `active`, because the bench has no consolidation.
- **Ceiling effects.**
  - AMB latest@5 = 1.000 and earlier@20 = 1.000 for every arm.
  - Judged accuracy is 0.89–0.95.
  - Harm on the original suite is 0 even with traps injected.
  - memory_lifecycle uses author-scripted decisions.
- **Small n's.**
  - KU observations-on: 40.
  - context_bench holdout: 46, of which 8 expect abstention.
  - Refute cases: 3; conflict cases: 6.
  - Harm: 8 trap tasks, and 4 per H2 group (the smallest attainable p is 0.125).
  - memory_lifecycle holdout stale queries: 7.
- **Single-sample judging.** Each judged answer is judged once. The two repeats are unseeded; these models reject seeds and temperature. The A-vs-B noise floor is 6 discordant questions of 211.
- **AMB date bug.** AMB's LongMemEval parser drops the time of day from session and question dates, so the order of same-day sessions is arbitrary. This hits every arm alike.
- **Inputs live in a temporary directory.** The frozen archives, AMB's per-question outputs and the worktrees exist only in the session scratchpad (§13).
- **`hindsight-system-evals/uv.lock` is not committed.** It holds an unexplained 1,402-line rewrite (`revision = 3` → `1`, `upload-time` dropped).
- **Live-server client suites were not run:** Python `test_main_operations.py` / `test_attachment_download.py`, TS `main_operations.test.ts`, and the Go root package.
- **Also not done.**
  - Rust was built only on rustc 1.87 with pinned dependencies.
  - The control-plane UI was not checked in a browser.
  - Observe mode does not run lifecycle classification.
  - Reinforcement events are not exported, only the resulting state.
  - One api-slim test sits at the 300 s timeout edge in the full `-n 8` run.

## 11. Why the benchmarks may not capture the lifecycle gain

**Measured.**
- The ranking moved where the mechanism acts: memory_lifecycle holdout stale 0.200 → 0 (stub-scripted), and P1-life 0.225 → 0.825 (after the source repair, and mostly an attribution artifact, §5.3).
- Answers did not move: B-obs 0.950 vs E-obs 0.925 with observations, B 0.891 vs E 0.905 without.
- Upstream reached the same null by another route: its recall-time stale-value filter left knowledge-update and contradiction scores unchanged and was removed (#5284 → #5632, §9).

Each explanation below is a **hypothesis**, with the experiment that would test it.

1. **Answer ceiling.** Plain Hindsight already answers 95% of the observations-on subset. AMB's answer prompt tells the model to treat the most recent memory as the update, and both values are in context at 4096/2048. Upstream's #5632 null, on a different mechanism, is consistent with this.
   - Test: judged B-obs vs E-obs at a much tighter budget (for example 512 tokens with no chunks), and with AMB's recency instruction removed from the answer prompt. Prediction: B-obs drops more if lifecycle helps.
2. **LongMemEval's knowledge updates are always the later-dated statement.** 77/78 carry `_2`, and the latest-value session's median age is 15.9 days vs 45.0 days for the earlier one. A recency heuristic is enough. Lifecycle's distinctive cases are absent: refutation, scope or environment divergence, an older statement retained later, unresolved conflict.
   - Base consolidation also already refuses to let an older fact overwrite a newer belief (upstream #5627, §9), so base handles the dated-update case by design.
   - Test: a judged question set built from the classification eval's scenarios, where B failed environment-divergence 3/3, scored on answers.
3. **The pre-registered retrieval endpoint rewards provenance.** This one is measured, exploratory, in §5.3: under newest-source attribution there is no gain, and a hand-read sample of 6 questions found base's in-place observations stating the update with dates. Whether that holds for all 28 questions whose top-1 cites both sessions is not checked.
   - Test: a text-level endpoint, pre-registered before scoring: does the top-k state the earlier value as current?
4. **No observations means no claims.** In AMB's default configuration (observations off) E == D, so lifecycle cannot act. Measured since the previous draft: judged E without observations recalls the same top-10 as D in 211/211 and 209/211 questions, and answers 0.905 vs D's 0.896 (exploratory comparison, 2/4, p = 0.69), as predicted.
   - Test: judged E with observations on the full KU 78 plus TR; consolidation costs about $0.62 per bank.
5. **Distraction harm is not what correctness judges score.** A stale value that is outranked may not flip the answer.
   - Test: a judge criterion "mentions the superseded value as current", or harm-bench stale traps. Measured on H2: the stale group's ungated success is 1.0 with the stale trap injected 8/8, so with Haiku 5.5 this hypothesis predicts little gain.
6. **Power.** On n = 40 or 78 the minimum detectable effect is about 10 points (DESIGN §11.3), and the A/B noise floor is 6 of 211 questions; E vs B differs on 9.
   - Test: 3 or more repeats on the full KU set.
7. **memory_lifecycle uses author-scripted decisions,** so its holdout effect measures mechanism conformance, not real-model benefit.
   - Test: freeze K = 3 real-model consolidation replies as fixtures (`lifecycle_bench --fixtures`) and re-run A–E.

## 12. Invariant checklists

Status and evidence come from `eval/R59-CHECKLIST.md` and `eval/A68-CHECKLIST.md`, both audited at `2f4c7daae`. Updates from later results are marked **(updated)**. Tests: `api/` = `hindsight-api-slim/tests/`, `sys/` = `hindsight-system-tests/tests/`.

### 12.1 R59 (Track 1)

| # | invariant | status | evidence (abridged) |
|---|---|---|---|
| 1 | truth status and retrievability are independent | verified by new test | `api/test_adaptive_axis_independence.py` (2 tests); lifecycle-nodecay = lifecycle at k = 5; E-obs-nodecay latest_first = E-obs |
| 2 | event recency and reinforcement recency are independent | verified by new test | axis test (a confirmation moves R, not mentioned_at or the recency boost); `sys/test_86`, `sys/test_87::test_the_ranking_clock_is_not_the_truth_clock` |
| 3 | superseded and refuted are distinct | verified | `api/test_adaptive_lifecycle_consolidation.py::test_refute_is_not_supersede…`, `sys/test_88`; holdout refuted@5 0.200 → 0.000. Real-model caveat: refute recall is 1/3 strict, 3/3 either label |
| 4 | historical truth is preserved | verified | `sys/test_87` (history, as_of), `sys/test_88` audit, `sys/test_100`; hist@5 1.000. Costs: KU-obs earlier@5 0.875; memory_lifecycle base-update hist@1 0.812 → 0.375 |
| 5 | repetition is not independent confirmation | verified | `sys/test_86`; `api/test_adaptive_retention_math.py::test_spaced_reinforcement_beats_ten_same_day_repetitions` |
| 6 | security scope is never weakened | verified | successor same-scope validation, cross-bank 404, write-scope 403, tenant schemas, `sys/test_89`. The Oracle variants were not run |
| 7 | no extra synchronous LLM call on recall | verified by new test | `api/test_context_admission_invariants.py::test_no_recall_calls_a_generative_model_in_any_mode`: 0 calls across 4 mode pairs |
| 8 | feature off reproduces the baseline | verified **(updated)** | golden gate byte-identical; memory_lifecycle 74/74; AMB A = B, 0 discordant; LoCoMo A = B 0/0; recall latency within base's noise floor (§6). Bulk deletes are slower in off mode (+17–33 µs per deleted unit on pg0, §6) |
| 9 | exact cues recover weak memories | verified | `api/test_adaptive_recall_integration.py::test_an_exact_cue_recovers_a_faded_memory…`; admission FIN-1602 test |
| 10 | reinforcement is idempotent | verified | UNIQUE key; `api/test_adaptive_reinforcement_store.py::test_the_same_events_twice_apply_once`; concurrent producers; `sys/test_86` retry |
| 11 | same-session spam cannot make a memory permanent | verified | cooldown, diminishing returns and cap tests; `sys/test_86` |
| 12 | invalidation/restore has defined retention semantics | verified | `api/test_adaptive_reinforcement_store.py::test_invalidate_then_restore_keeps_stability_count_and_keys` and 4 more |
| 13 | export/import preserves lifecycle semantics | verified | claims-store transfer tests; `sys/test_102`; `sys/test_88` re-ingest. Events do not travel, by documented choice |
| 14 | mental models do not retain stale current-state claims | verified | `sys/test_95_mental_model_supersession.py` (4 tests; last run in the T2 full system suite) |
| 15 | no dangling lifecycle relationships after deletes | verified | migration cascade/SET NULL tests (also live on Oracle 23ai in P1); delete-repair tests; `sys/test_91` |
| 16 | recall p95 overhead measured | **measured (updated); gates not resolved** | `eval/perf` (§6): no observations, p95 vs B lifecycle +1.1% seq / −3.0% conc, rank +1.6% / −7.2% (point estimates under the < 10% and < 5% bars); every run-level interval on this decision cell straddles its bar (lifecycle conc [−13.9, +29.9]). With observations, lifecycle seq +0.9% is resolved under its bar (run-level [−6.6, +7.9]); conc p95 lifecycle +22.3%, rank +38.9% (over the bars, unresolved). Median +8.6% / +9.1% sequential. The p95 is dominated by GC pauses; 4 medium rounds replaced DESIGN §11.4's 5 large runs (undeclared deviation) |
| 17 | LongMemEval knowledge-update measured | **measured (updated)** | Retrieval: P1-rank not significant (0.819 → 0.861); P1-life significant (0.225 → 0.825) after the post-hoc source repair, largely attribution (exploratory). Judged KU both-correct: A 0.949, B 0.923, D 0.923, E 0.936; B-obs 0.950 vs E-obs 0.925 (2/1, p = 1). No judged difference anywhere |
| 18 | temporal-reasoning measured | **measured (updated)** | Retrieval non-inferiority met (gold@20 0.992 = 0.992, all_gold@20 0.913 = 0.913). Judged TR both-correct: A 0.857, B 0.872, D 0.880, E 0.887 |
| 19 | new stale-memory benchmark measured | verified | memory_lifecycle holdout stale 0.200 → 0.000, refuted 0.200 → 0.000, hist 1.000 (stub-scripted); real-model false retirement in 0 of 51 control runs |
| 20 | all regressions documented | **documented (updated)** | documented here: LoCoMo D/E gold@5 −1.6 points (32/8) caused by decay (exploratory arm); KU-obs earlier@5 −12.5 points by design; D-obs latest_first −5.0 points (not significant); memory_lifecycle base-update hist@1 0.812 → 0.375; rec@5 0.983 → 0.951 (error amplification); +52% consolidation tokens and +10–17% consolidation time (mock LLM); recall median +9% sequential and, with observations under c = 8, p95 +22% (lifecycle) / +39% (rank), unresolved at run level; FK delete +17–33 µs per deleted unit in every mode, off included; A6 joke defect; H2 gated harm. Not measured: FK delete at large scale and on Oracle, storage growth from real supersession |

### 12.2 A68 (Track 2)

| # | invariant | status | evidence (abridged) and updates |
|---|---|---|---|
| 1 | zero-memory injection is legitimate | verified **(updated)** | `api/test_context_admission*.py` abstention tests, `sys/test_104`. Holdout abstention: F 8/8, E-jev 7/8, E-det 5/8; baseline 0/8 |
| 2 | similarity ≠ admission | **defect found** | The default deterministic scorer admits the A6 joke (0.862, strict xfail). **(updated)** holdout humor_casual pollution: E-det 0.60, E-jev 0.38, F 0.33 vs tags + Jev 0.42. E-jev admitted the same joke on h05 (explicit_search) |
| 3 | hard scope enforced before injection | verified (strict) **(updated)** | strict-mode tests, `sys/test_104`. Holdout leak@10 0.000 for all three arms. Caveats: filter is not a gate (0.709 admitted); unstamped memories pass |
| 4 | soft scope scores rather than filters | verified | `test_soft_scope_scores_rather_than_filters`; `sys/test_104` (branch mismatch still admitted) |
| 5 | stable memories can be rejected | verified by new test | real-score test: importance 1.0, S 1000 d, R 0.99 → `below_threshold` |
| 6 | weak memories admitted under precise cues | verified by new test **(updated)** | FIN-1602 test, R 0.15. Holdout exact_historical_cue recall: E-det 0.60, E-jev 1.00, F 0.80; useful_old: 0.60 / 0.80 / 0.60 |
| 7 | superseded/refuted interact correctly | verified by new test **(updated)** | real-claim tests. Not exercised by context_bench (all candidates active). Holdout stale admission F 0.174 vs baseline 0.090; refuted 0 everywhere |
| 8 | historical queries retrieve history | verified by new test | `test_a_historical_search_reads_the_history_view_and_admits_superseded_memories` |
| 9 | humor routed, not deleted | partially | asked-for joke admitted and never deleted; routing away from technical tasks fails for the deterministic scorer (Defect 1) |
| 10 | no admission event reinforces by itself | verified | 0 events and 0 retention rows after 3 strict recalls; reinforcement forced off while gating |
| 11 | no same-session popularity loop | verified by new test | day 0 / 30 / 60: gated R strictly decays; the observe control is reinforced |
| 12 | budget enforced | verified by new test | `max_memories=2` → 2 returned; token cap → 1 returned |
| 13 | near-duplicates don't flood | verified | MMR and stored-duplicate tests (strict and observe only) |
| 14 | explicit queries use different trade-offs | verified by new test | purpose thresholds; reference-signal detector (8 cases) |
| 15 | feature off reproduces previous behaviour | verified | `model_dump_json` byte-identical; golden gate; memory_lifecycle with admission off 74/74 = P5 |
| 16 | observe changes nothing visible | verified **(updated)** | byte-identical payload tests, with observations and rank scoring. Latency measured (§6): deterministic phase p95 3.18 ms over 255 sequential HTTP recalls; in-process, recall p95 +0.0% vs no admission sequentially (run-level [−5.3, +4.4]); under c = 8 in-process, phase p95 274 ms (258–287 ms per round) and recall p95 +22.1% (+158 ms) pooled, not resolved at run level ([−9.0, +40.4]) |
| 17 | lifecycle tests stay green | verified | 667 passed, 1 xfailed; golden pass; full system suite 537 passed at T2 |
| 18 | pollution benchmark reproducible | partially **(updated)** | Each holdout run replays exactly from its own trace (46/46 per arm); baseline arm A is deterministic (131/131 top-10). The Jev path varies across samples (dev 80/85 and 84/85 identical) |
| 19 | threshold sweep reported | verified **(updated)** | dev sweeps for every arm (`t2/dev/sweep-*.json`); informational holdout sweep (`t2/holdout/holdout-sweep.json`) |
| 20 | downstream impact measured | measured, exploratory **(updated)** | Original: gated success 1.000 = ungated, harm 0 = 0, tokens 51 vs 581. H2: gated success 0.562 vs 0.719, harm 0.417 vs 0.292 (p = 0.5) |
| 21 | regressions reported | **documented (updated)** | documented here: Defect 1; recall loss 0.978 → 0.804 (F, exploratory) and 0.848 (E-jev, confirmatory); E-det pollution 0.486 vs 0.360; stale memories admitted alone; w08 VPN trap admitted by all three arms; H2 harm; filter not a scope gate; dev-to-holdout drift; observe and strict tail latency under c = 8 (274 ms and 69 ms pooled phase p95; strict 196 ms in one round); the Jev and hybrid paths at about 432 ms phase p95 per recall |

## 13. Reproducibility

| item | value |
|---|---|
| base SHA | `5331da2f4db369528bb122075341a7de9c3df6df` (main, 2026-10-16 15:58 −0400) |
| feature head | `554284233f4fdb38d1e065c0d90b94ed9deb38a4`. Commits: `f87df0cdc` harness, `dafd09a64` P1, `3816d0e4b` P2, `b491b2bb9` P3, `d65f17b05` P4, `081025def` classification eval, `60c1cc87f` perf harness fix, `728b265c9` P5, `504ca72e9` T2, `2f4c7daae` harm bench, `5f523c2e7` invariant tests, `554284233` bench arms |
| worktrees (scratchpad) | `hindsight-base` @ 5331da2f4 (arm A, including perf arm A with the 3-line harness patch); `hindsight-p1` @ dafd09a64; `hindsight-p3` @ b491b2bb9 (AMB arms without observations, observations-on consolidation, classification P3); `hindsight-p5` @ 728b265c9 (judged D and the judged E re-run, LoCoMo, classification P5); `hindsight-t2` @ 2f4c7daae (context_bench T2, harm gated, memory_lifecycle T2, perf arms B–E and admission); about 2.0 GB each |
| inputs that exist only in the temporary scratchpad | Scratchpad: `/private/tmp/claude-502/-Users-davidvieira-code-hindsight/823f87da-11e6-43e4-a7a6-b3e4cbaba9e3/scratchpad`. `frozen/` is 136 MB: the 221 arm-A archives in `amb-a-5331da2f4` (90 MB), `obs-cons-base` (19 MB), `obs-cons-life` (21 MB) and the R55 partials (4.1 + 2.0 MB). AMB's per-question outputs for every judged and LoCoMo run are in `amb-checkout/outputs/{longmemeval/judged-*,locomo/locomo-*}` (1.6 GB of 2.3 GB). The dataset is `amb-checkout/.datasets/`, and the server keeper is `scripts/keep_server.py`. `build_artifacts.py`, `judged/score_judged.py`, `run_judged.sh`, `run_locomo.sh` and `start_server.sh` hard-code these paths. They were not copied anywhere persistent. `derived/scratchpad-inputs.sha256` (334 files; file sha256 `a03a54c6…7404`; it was `6648665d…` with 332 files before the judged E re-run) hashes the archives, those outputs (including the judged E re-run and the unused partial E-r1, which was moved to `judged-E-r1-partial-credit/rag/s.json` before the re-run), the keeper and the dataset, so a restored copy can be checked; all 334 verified for this report. Until they are archived, re-scoring the judged and LoCoMo numbers and re-running any AMB arm depend on a temporary directory. The AMB retrieval rows and every other scored source are under `tasks/adaptive-memory/` (persistent, git-excluded) |
| models | server extraction and consolidation `claude-haiku-5-5` (model-default thinking); AMB answer `claude-haiku-5-5` and judge `claude-sonnet-5-5` (model-default effort); classification judge Sonnet 5.5 at effort low; harm answer Haiku 5.5 at effort medium, judge Sonnet 5.5 at effort high; decision model TypeSafe `jev-latest`. The models reject temperature and seeds, so repeats are unseeded |
| embedder / reranker | local `bge-small-en-v1.5` + `ms-marco-MiniLM-L-6-v2` cross-encoder, CPU, HF offline |
| database | embedded PostgreSQL via `pg0-embedded` 0.15.2 (uv.lock), one named pg0 instance per server |
| AMB | ref `aef3702e0586a959c845cc2c61f10d5e8ff0f94a` plus a local patch. The current checkout diff is sha256 `a81267396f066ac9fc85c88c8930c2b8334ffebfcb25aaea8b420ffec521c52d` (re-hashed for this report from `git diff aef3702e…` in the checkout; saved as `derived/amb-patch-current.diff`; it adds `AMB_QUERY_IDS_FILE`, used by the judged runs). Earlier: cumulative `373bf149…d0b21` (`eval/amb-patch-cumulative.diff`, async-ingest dedup, freeze run 2 and later); patch 1 `c11e67af…`; patch 2 `137fee93…` (the pilot ran with `a2a99e7b…` / cumulative `67fa7f06…`) |
| datasets | `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` (277,383,467 B; hashed for this report); `locomo10.json` `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; observations-on subset `eval/arms/obs-ku-qids.txt` `7ba3a529…6211` |
| pre-registration | **Original Track 1 text:** sha256 `e32bd91bcb7bb88988d42712ff89487e808fb8ec81553e85f634fae13b86083e`, computed at 2026-10-17 09:48 BRT (12:48:11Z, "re-hash pre-registration" after a timestamp correction). That value is recorded only in the session transcript, not in any file under `tasks/` (STATE 09:50 points to `eval/PREREGISTRATION.sha256` without quoting it). It is reproducible: sha256 of the current `PREREGISTRATION.md` up to `\n\n## Track 2`, plus `\n`, equals it. So the Track 1 rules are byte-identical to the 09:48 text (measured). **Overwritten file hash:** `eval/PREREGISTRATION.sha256` was overwritten at 20:17:34 BRT with `59908dd1cba786e78d821f4783e80f8815888e788387f98f02a96e664c90ad2f`, the hash of the Track 1 and Track 2 text together (the prefix up to `## Amendments`). So the Track 2 section was appended by 20:17, although its header says "~20:30". **Chain:** `e32bd91b` (09:48) → `59908dd1` (20:17, Track 2 appended) → `8c18958…` (amendment 20:34, judged scoring) → `9ce010dc…8a9d` (amendment 23:50, Track 2 dev selection rule) → `8ca8f269491ca767101648f244506a7140839fb0ffd8fc70efc91c9a94ab7e17` (amendment 01:45, Track 2 selection as applied; current file, in `eval/PREREGISTRATION.md.sha256`). Each is the hash of a prefix of the current file, so every change after 09:48 was append-only (measured). **Already measured before 20:17:** the no-obs fork AMB endpoints (`score-noobs/`, 15:33), P3 memory_lifecycle (14:19–14:42), the classification eval (15:53–16:29) and the interim obs-on scoring (17:09). The Track 2 baselines (context_bench, by 10:55) were known when Track 2 was written, as its section says. The only evidence that the Track 1 rules predate those results is the 09:48 hash, which sits in a transcript, not a project file |
| frozen parameters | alpha 0.1, floor 0.3, base S 30 d, max S 3650 d; type multipliers 0.5 / 1.0 / 2.0; cooldown 6 h; gain 1.0; recall reinforcement off; conflict policy `evidence`; historical cues off |
| context_bench | `chosen.json` sha256 `02b02eecca7408489dc2b45dfbb51b84b24beb1a463f33707256e64f5a510029`; holdout primary scores `4125fe14…`; 4 labels corrected (z01, h10, h12, u08) before any metric was computed |
| perf | Procedure `eval/perf/raw/analysis_plan.json`: file mtime 2026-10-18 02:13:48 BRT, after the smoke run (05:09:34Z) and the medium pilot of arm E (05:10:29Z), both excluded, and before the first analysed run began its machine-state check (05:13:53Z; cleared to start 05:14:33Z); its `written_at` field (05:20Z) is not its write time (§6). Scripts `eval/perf/scripts/`; raw data `eval/perf/raw/` (`machine_state.jsonl` holds 87 records: the smoke and pilot runs, then the 85 analysed runs); output `eval/perf/performance.json` sha256 `09808262f02af8c52450b6dc0554cdb8b09ce26772f4743bc319cd7bf5a9647b`, report `eval/perf/REPORT.md` sha256 `39e29507…a0b4`. pg0 PostgreSQL 18.1.0 on instances `hindsight-perf-a` (A) and `hindsight-perf-f` (feature); Python 3.11.9, torch 2.13.0, asyncpg 0.30.0, identical package sets in both venvs. `run-perf-test.sh` could not run (the venvs have no `perf-test` console script, and syncing was avoided), so the suites were called through the same entry point (command below), verified at tiny scale on both trees first. The in-process hybrid admission arm calls the real TypeSafe decision model (`TYPESAFE_API_KEY` from the secrets file), outside the counting proxy |
| judged E re-run | `eval/judged/rerun_E.sh` (sha256 `9361df84…7037`): starts srv-e on the P5 worktree in lifecycle mode, then `run_judged.sh` for E-r1 and E-r2 on the frozen arm-A archives; scored file `eval/judged/judged-noobs.md` (sha256 `e4bc585a…27e6`), exit codes 0 / 0 in `eval/judged/exit-codes` |

Exact commands, as recorded in the source reports and scripts. Secrets come from `~/.config/hindsight-adaptive/secrets.env` (`ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`); `<scratchpad>` is the directory above.

```bash
# memory_lifecycle, from hindsight-system-tests/
uv run python -m hindsight_system_tests.lifecycle_bench run --arm A --api-dir <base>/hindsight-api-slim --encoding base-delete --out <out>/arm-A-base-delete.json
uv run python -m hindsight_system_tests.lifecycle_bench run --arm lifecycle --set ADAPTIVE_MEMORY_MODE=lifecycle --encoding base-delete --api-dir <feature>/hindsight-api-slim --out <out>/p5-lifecycle-base-delete.json
uv run python -m hindsight_system_tests.lifecycle_bench compare <A>.json <arm>.json --rows

# AMB LongMemEval retrieval endpoints
# arm files: tasks/adaptive-memory/eval/arms/arms-A.json (A), arms-fork-noobs.json (B, C, D, E),
#            arms-cons-base.json (B-obs, D-obs, P-obs, T-obs), arms-cons-life.json (E-obs, E-obs-hist, E-obs-nodecay, E-obs-off)
bash tasks/adaptive-memory/eval/arms/start_server.sh <worktree> <state-dir> <pg0-instance> [ADAPTIVE_MEMORY_MODE=...]
# observations off
uv run --project <worktree>/hindsight-system-evals python tasks/adaptive-memory/eval/arms/shared_import_eval.py \
  --server-url-file <state-dir>/server.url --archives <scratchpad>/frozen/amb-a-5331da2f4 --dataset-file <longmemeval_s_cleaned.json> \
  --categories knowledge-update,temporal-reasoning --arms tasks/adaptive-memory/eval/arms/arms-fork-noobs.json --out <run dir>
# observations on: one consolidation per condition (servers srv-cb on pg0 amb-cons-base, srv-cl on amb-cons-life, P3 worktree)
uv run --project <worktree>/hindsight-system-evals python tasks/adaptive-memory/eval/arms/shared_import_eval.py \
  --server-url-file tasks/adaptive-memory/eval/arms/srv-cl/server.url --archives <scratchpad>/frozen/amb-a-5331da2f4 \
  --dataset-file <longmemeval_s_cleaned.json> --categories knowledge-update --question-ids tasks/adaptive-memory/eval/arms/obs-ku-qids.txt \
  --arms tasks/adaptive-memory/eval/arms/arms-cons-life.json --observations --consolidation-config '{"adaptive_memory_mode":"lifecycle"}' \
  --keep-banks --concurrency 4 --export-dir <scratchpad>/frozen/obs-cons-life --out tasks/adaptive-memory/eval/arms/obs-cons-life
#   (cons-base: srv-cb, arms-cons-base.json, '{"adaptive_memory_mode":"off"}', frozen/obs-cons-base, obs-cons-base)
# REQUIRED before scoring an obs-on run, with its server still up: resolve observation sources (keeps rows-unresolved/)
python3 tasks/adaptive-memory/eval/arms/resolve_observations.py <state-dir>/server.url <run dir>
python3 tasks/adaptive-memory/eval/arms/score_endpoints.py --dataset-file <lme json> --runs <run dirs> --reference B --out <out>   # --reference B-obs for obs-on

# judged LongMemEval (budget 4096/2048) and LoCoMo retrieval; obs-on judged runs import frozen/obs-cons-{base,life} with obs-ku-qids.txt
tasks/adaptive-memory/eval/judged/run_judged.sh <server-state-dir> <archive-dir> judged-<ARM>-r<k> [qids-file]
tasks/adaptive-memory/eval/judged/run_locomo.sh <server-state-dir> locomo-<ARM>
python3 tasks/adaptive-memory/eval/judged/score_judged.py locomo <out.md> B A=locomo-A B=locomo-B D=locomo-D E=locomo-E E-nodecay=locomo-E-nodecay
# judged E re-run after the credit outage (srv-e, P5 worktree, ADAPTIVE_MEMORY_MODE=lifecycle; move any partial output aside first)
tasks/adaptive-memory/eval/judged/rerun_E.sh
python3 tasks/adaptive-memory/eval/judged/score_judged.py lme tasks/adaptive-memory/eval/judged/judged-noobs.md B \
  A=judged-A-r1,judged-A-r2 B=judged-B-r1,judged-B-r2 D=judged-D-r1,judged-D-r2 E=judged-E-r1,judged-E-r2
# spend from the usage logs (AMB-side answer + judge)
python3 tasks/adaptive-memory/eval/scripts/amb_cost.py tasks/adaptive-memory/eval/judged/usage-judged-*.jsonl

# real-model classification eval, P3 (K = 3: three parallel runs, one repeat each), from the repo root
bash tasks/adaptive-memory/eval/arms/start_server.sh <scratchpad>/hindsight-p3 tasks/adaptive-memory/eval/classify/srv amb-classify
C=/Users/davidvieira/code/hindsight/tasks/adaptive-memory/eval/classify
for r in 0 1 2; do REPEATS=3 $C/run.sh $C/raw/runs-r$r.jsonl $C/raw/runs-r$r.log "r$r" & done
(cd hindsight-system-evals && uv run python -m hindsight_system_evals.lifecycle_eval $C/raw/runs-r*.jsonl) > $C/raw/report-generated.md
touch $C/srv/STOP
# P5 re-run (K = 1)
bash tasks/adaptive-memory/eval/arms/start_server.sh <scratchpad>/hindsight-p5 $C/srv-p5 amb-classify-p5
REPEATS=1 $C/run-p5.sh $C/raw-p5/runs.jsonl $C/raw-p5/runs.log
(cd hindsight-system-evals && uv run python -m hindsight_system_evals.lifecycle_eval $C/raw-p5/runs.jsonl) > $C/raw-p5/score.md
touch $C/srv-p5/STOP

# context_bench, from hindsight-system-tests/ (eval/context-bench/REPORT.md "Running more arms")
uv run python -m hindsight_system_tests.context_bench run --label base-local --api-dir $BASE --variants local-baselines
# the pre-registered comparator tags + Jev (C'+D) and its repeat (TYPESAFE_API_KEY)
uv run python -m hindsight_system_tests.context_bench run --label base-jev --profile jev --pg0 hindsight-ctxbench-jev:15531 --api-dir $BASE --variants plain,tags
uv run python -m hindsight_system_tests.context_bench run --label base-jev-repeat --profile jev --pg0 hindsight-ctxbench-jev:15531 \
  --api-dir $BASE --variants plain --queries <holdout ids> --reuse <out>/runs/base-jev.json
uv run python -m hindsight_system_tests.context_bench report
# T2 dev selection: REPORT.md "Running an arm at its chosen setting"; freezes t2/chosen.json (sha256 02b02eec…)
# T2 holdout: each arm once; refuses to run unless chosen.json hashes to 02b02eec… and never overwrites a holdout run
uv run --frozen python ../tasks/adaptive-memory/eval/context-bench/t2/holdout/run_holdout.py E-det   # then E-jev, F
uv run --frozen python ../tasks/adaptive-memory/eval/context-bench/t2/holdout/score_holdout.py primary
uv run --frozen python ../tasks/adaptive-memory/eval/context-bench/t2/holdout/score_holdout.py sweep

# harm bench, from hindsight-system-tests/ (forms from the harm_bench CLI docstring; the recorded base-h2 run reused an h2 probe's recalls)
uv run python -m hindsight_system_tests.harm_bench run --label base --conditions none,ungated --api-dir <base>/hindsight-api-slim
uv run python -m hindsight_system_tests.harm_bench run --label base-nopreamble --conditions ungated-nopreamble --reuse-recalls <out>/runs/base.json --budget-usd 1
uv run python -m hindsight_system_tests.harm_bench run --label base-h2 --suite h2 --conditions none,ungated --api-dir <base>/hindsight-api-slim --budget-usd 1.5
# gated = arm F's frozen setting: checks chosen.json's sha256, copies F's server env verbatim, pins every bank-config field
# (strict, auto 0.80, explicit 0.85) as HINDSIGHT_API_* env; --profile jev-admission --repeats 2 --pg0 hindsight-harmbench:15541 --budget-usd 3
uv run --frozen python ../tasks/adaptive-memory/eval/harm-bench/run_gated.py original --recall-only   # then: original --resume; same for h2
uv run python -m hindsight_system_tests.harm_bench report

# perf (no Anthropic call; the hybrid admission arm and the context_bench jev/hybrid runs call TypeSafe; eval/perf/REPORT.md "Reproduce"), from the repo root
S=tasks/adaptive-memory/eval/perf/scripts
set -a; source ~/.config/hindsight-adaptive/secrets.env; set +a; unset ANTHROPIC_API_KEY
python3 $S/run_rounds.py --block noobs --rounds 1-4
python3 $S/run_rounds.py --block obs --rounds 1-4
$S/run_rest.sh                                   # suites x3, context_bench det/hybrid/jev, FK delete x3
<t2>/.venv/bin/python $S/analyze.py              # writes performance.json
<t2>/.venv/bin/python $S/sensitivity.py          # exploratory round-dropping sensitivity
python3 $S/report_tables.py                      # prints the REPORT.md tables
# the perf-test suites, called directly in each tree (run_suite.py does this per mode)
uv run --no-sync python -c "from benchmarks.perf.system_perf import main; main()" --scale medium --suite retain --suite consolidation --output <out>.json

# gates
cd hindsight-api-slim && uv run pytest tests -q -m "not hs_llm_mat and not hs_llm_core and not oracle" -p no:cacheprovider
cd hindsight-system-tests && uv run pytest tests/ -q --ignore=tests/test_99_golden_off_mode.py
cd hindsight-system-tests && HINDSIGHT_SYSTEST_GOLDEN_DIR=../tasks/adaptive-memory/golden/base HINDSIGHT_SYSTEST_DATABASE_URL=pg0://hindsight-golden:15519 \
  uv run pytest tests/test_99_golden_off_mode.py

# this directory's JSON (no LLM, no server; reads the scratchpad's AMB outputs, eval/perf/performance.json and eval/perf/raw/; it also
# recomputes the run-level perf intervals (Track 1 arms vs B, admission arms vs E-ext), the per-round admission phase p95,
# the in-process hybrid arm's decision-model use and the judged E vs D top-10 check), from the repo root
python3 -I artifacts/adaptive-memory/derived/build_artifacts.py
python3 -I artifacts/adaptive-memory/derived/latest_first_attribution.py tasks/adaptive-memory/eval/arms <lme json> artifacts/adaptive-memory/derived/latest-first-attribution-exploratory.json
(cd <scratchpad> && shasum -a 256 -c /Users/davidvieira/code/hindsight/artifacts/adaptive-memory/derived/scratchpad-inputs.sha256)
```

## 14. Incidents

- **Usage limit, 2026-10-17 ~03:30–04:10.** The account session usage limit (HTTP 429) was hit with about 8 concurrent agents. The P1 engine, integration and review agents stopped mid-work, and so did the AMB pilot, the lifecycle bench and the context bench agents. Work resumed at 04:12 with less parallelism: one combined review agent per phase, and the context bench deferred. Cached P1 storage and config results were replayed.
- **Oracle container OOM-killed the user's MySQL, overnight 2026-10-17.** Starting an Oracle 23ai container on the 2.84 GiB Docker VM killed the user's MySQL `db` container (03:46:41Z), which has since been restarted. P1's Oracle tests ran on that container before it was removed. After that:
  - no Oracle container was started again;
  - client generation ran through a memory-capped shim, then host `java -Xmx1g`;
  - all of the user's containers stayed up.
- **AMB ingest crash, 2026-10-17 morning.** Freeze run 1 stopped at 87 of 211 banks ($12.13): AMB's async ingest did not dedupe a repeated session id, so the server returned 400. AMB was patched (cumulative diff `373bf149…`) and the run resumed with `--skip-ingested`.
- **Observation sources not recorded, 2026-10-17 afternoon.** The observations-on driver recorded empty source lists for observations; the sources were re-resolved post-hoc at 20:15 under the pre-specified rule (§5.3).
- **Machine memory saturated, 2026-10-17 ~14:30.** The memory compressor held about 15 GB, so the observations-on conditions waited for the servers without observations to stop. The context bench holdout ran under load averages of 53 / 68 / 101, which is why its latency numbers are not gate numbers; the gate numbers come from the later perf workflow (§6).
- **Anthropic credit exhausted, 2026-10-18 01:09:41.** Calls failed with HTTP 400 "credit balance is too low". Lost or blocked:
  - judged E-r1 (189/211 saved, exit 1) and E-r2 (started at 01:10, exit 1, nothing saved): both re-run from scratch at 07:07–07:55, after the perf workflow, with the partial output moved aside; exit 0 / 0;
  - the harm-bench gated answers: completed after the credit was restored, recalls reused;
  - R55 last-write-wins: 8 clean banks, 4 failed and quarantined (`rows-failed-credit`, `frozen/obs-cons-latest-failed`), 28 not run.

  The credit was restored at 02:01, when the user added $4. The user then chose "essential only": judged E and this report. Spent since: $3.07 (§7).
- **Time-reading errors.** The orchestrator reported time-reading errors that were made and corrected during the session. No file under `tasks/` records them individually, so this report cannot list them. Hypothesis about their source: run manifests stamp UTC (`+00:00`) while STATE and the reports use BRT (UTC−3). Every time in this report has been checked against one of those two conventions. One more was found for this revision (measured): the perf analysis plan's `written_at` field says 05:20Z, but its file mtime is 05:13:48Z, after the smoke run (05:09:34Z) and the medium pilot of arm E (05:10:29Z), and the first analysed run began its machine-state check at 05:13:53Z and was cleared to start at 05:14:33Z. The previous draft quoted 05:20Z and said the plan preceded "any perf run"; §6, §8.1 and the JSON artifacts now use the mtime and say "any analysed perf run".
- **Perf verdicts rewritten, 2026-10-18 ~07:08.** The perf workflow's first verdicts ("holds, CI included") rested on the analysis plan's cluster bootstrap; an independent read-only verification found it anti-conservative (62% false positives in an A/A check), and `eval/perf/REPORT.md` was rewritten to point estimates with run-level intervals before this report used it (§6).
- **Perf TypeSafe count incomplete.** `eval/perf/REPORT.md` reports 510 TypeSafe calls, the proxy-counted context_bench runs only. The in-process hybrid admission arm also called the real decision model on every recall (about 2,500 requests, not counted, tokens not recorded; measured from the raw samples for this report, §7).

## 15. Follow-up exploratory tests (after the report)

The user asked whether the benchmarks simply miss the gain. Three tests aimed at the settings where a gain was most plausible were pre-registered as one amendment before any ran (`PREREGISTRATION.md`, "three exploratory follow-up tests", sha256 `e08371a6…`), plus a held-out check of the one that passed ("X3b", `8be4c11a…`). All are exploratory. Each result was re-derived by an independent read-only verifier, which agreed with every verdict. Full reports: `tasks/adaptive-memory/eval/followups/X{1,2,3,3b}/`.

| test | question | result | verdict |
|---|---|---|---|
| X1, all-or-nothing admission (offline, $0) | Keep the gate's abstention but inject the full ungated top-10 whenever it admits anything | context_bench holdout: useful recall 0.913, abstention 8/8. Original harm suite: equal to ungated (success 1.000, harm 0) with 221 vs 581 injected tokens. H2: success 0.563 vs bar 0.669 (the gate abstains on the cross-module tasks M01–M03, where the ungated top-10 helped) | **not met** |
| X2, coding-agents injection budget ($0.26) | Does lifecycle help when context is tight (`types=['observation']`, budget low, `max_tokens` 2000, as the coding-agents integration recalls)? | LongMemEval KU obs subset, judged: E-obs 35/40 vs B-obs 37/40 (0/2, p = 0.5) | **not met** |
| X3, scope separation in the base consolidation prompt ($0.42) | Render the metadata keys environment/service/module as `[scope: …]` in the consolidation prompt plus one rule, on unmodified base | classification eval, K=1: environment-divergence B′ 1/1 vs B 0/4; all 22 scenarios B always passed still pass; judged 26/27 vs B's P5 run 23/27; +1.5% input tokens | **met** (in-sample: the prompt's example reused the scenario's values) |
| X3b, held-out check of X3 ($0.055) | Neutral example; four new scenarios written blind to the prompt (service divergence, module divergence, a real change within one scope, a global fact then a scoped one); base vs fix | B′ 5/5, unmodified base 1/5 (only the within-scope change); no scenario where base passes and the fix fails. Verifier note: the scenario author did not open the prompt but did read the X3 report's one-line summary of the rule | **met** (K=1) |

**What this changes.** The one positive finding is small and independent of the lifecycle machinery: base Hindsight's consolidation never sees retain metadata, so a value that differs by environment, service or module is read as a state change and the observation is rewritten in place. Showing the scope keys to the model fixes that in every scenario tried (6/6 across X3 and X3b, K=1), at +1.5% tokens, with no new tables or config. It is prepared as a single commit (`335e10b6d`, branch `fix/consolidation-scope-separation` on 5331da2f4, not pushed): prompt and consolidator change, prompt unit tests, an `hs_llm_core` judge test (passed), a system story that fails without the fix, and a docs paragraph. K=1 on six scenarios is weak evidence; it is a PR candidate, not a proven gain. The lifecycle and admission verdicts above are unchanged by X1 and X2.

**Jev ecosystem (survey, 2026-10-18).** Projects that use Jev as their main engine exist but are small and new (largest: Jev-Mem, 219 stars, a research prototype with its own LoCoMo claim). Larger systems use it as an optional component, as Hindsight does. Other projects report the same pattern we measured: good ranking, poor calibration, abstention by threshold costs recall, 260–410 ms per call. Several pin the Jev version because `jev-latest` moves to 1.14.0 around 2026-10-21, which one project measured as worse; Hindsight and every Jev number in this report use `jev-latest`. Details: `tasks/adaptive-memory/eval/JEV-PROJECTS.md`.
