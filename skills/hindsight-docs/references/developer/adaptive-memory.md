
# Adaptive Memory: Truth and Retention Over Time

A memory bank keeps everything it is told. That is the right default for an audit trail, but an agent that recalls "the billing service runs on Java 21" a month after it was upgraded to Java 25 is giving a wrong answer with a correct citation. Adaptive memory gives each memory two pieces of state that change over time, and keeps them apart:

| | What should be believed | What should be remembered |
|---|---|---|
| Question | Is this still the current state? | How readily does it come back without a strong cue? |
| State | A **claim**: `active`, `uncertain`, `superseded` or `refuted`, with the interval it was current | **Retention**: stability, last reinforcement, importance, exemption |
| Changed by | Consolidation's lifecycle labels, or the lifecycle `PATCH` API | Reinforcement events only (confirmations, independent restatements, successful use) |
| Used by recall as | A filter (and an annotation) | A ranking penalty, experimental and off by default (`adaptive_memory_score_alpha` is `0`) |

Neither axis is derived from the other. A memory that faded is still true, and a memory that was confirmed a hundred times can still be superseded. The forgetting curve never decides truth.

The feature is **off by default**, per bank, and with it off Hindsight behaves exactly as it does without it. Every setting is listed under [Adaptive memory](./configuration.md#adaptive-memory) in the configuration reference.

## Modes

`adaptive_memory_mode` is a hierarchical setting: set it server-wide with `HINDSIGHT_API_ADAPTIVE_MEMORY_MODE`, or per bank through the [bank config API](./api/memory-banks.md):

### Python

```python
# Turn adaptive memory on for one bank (it is off by default).
await client.aupdate_bank_config(BANK_ID, adaptive_memory_mode="lifecycle")
```

### Node.js

```javascript
// Turn adaptive memory on for one bank (it is off by default).
await client.updateBankConfig(BANK_ID, { adaptiveMemoryMode: 'lifecycle' });
```

### CLI

```bash
# Turn adaptive memory on for one bank (it is off by default).
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"adaptive_memory_mode": "lifecycle"}}'
```

### Go

```go
// Turn adaptive memory on for one bank (it is off by default).
client.BanksAPI.UpdateBankConfig(ctx, memBankID).
	BankConfigUpdate(hindsight.BankConfigUpdate{
		Updates: map[string]interface{}{"adaptive_memory_mode": "lifecycle"},
	}).Execute()
```

| Mode | What it reads | What it writes | What a caller sees |
|---|---|---|---|
| `off` (default) | Nothing | Nothing | Byte-identical to a server without the feature: the same recall payloads, traces, consolidation prompt and schema, MCP tool list, metrics and transfer archives. |
| `observe` | Retention state for recall candidates | Only observe-mode shadow records (see [below](#enabling-observe-mode-safely)) | Results and their order are unchanged. The recall trace and the metrics report what `rank` would have done. |
| `rank` | Retention state | Retention rows and reinforcement events; archive state when archiving is on | Confirmations, importance and exemptions are accepted. The final recall score is multiplied by a retention factor only when `adaptive_memory_score_alpha` is above `0`, which needs the server-level `HINDSIGHT_API_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL`; at the default `0` ranking is the same as `off`. |
| `lifecycle` | Retention and claim state | Everything `rank` writes, plus claims, claim evidence and lifecycle relations | `rank`, plus: consolidation labels state changes, superseded and refuted memories leave current recall, results carry a `lifecycle` annotation, history is available through `lifecycle_view` and `as_of`, and mental models citing retired memories are refreshed. |

Every mode above `off` applies only to banks whose store keeps adaptive state; the built-in PostgreSQL and Oracle stores do, a custom memories extension does not unless it says so, and on such a bank every mode behaves as `off`.

## Personal profile

One block of server settings turns on the parts of adaptive memory that showed a benefit in the benchmarks and leaves out the parts that did not:

```bash
HINDSIGHT_API_ENABLE_OBSERVATIONS=true
HINDSIGHT_API_ADAPTIVE_MEMORY_MODE=lifecycle
HINDSIGHT_API_ADAPTIVE_MEMORY_SCORE_ALPHA=0
HINDSIGHT_API_CONTEXT_ADMISSION_MODE=observe
HINDSIGHT_API_CONTEXT_ADMISSION_SCORER=deterministic
HINDSIGHT_API_CONTEXT_ADMISSION_EXPERIMENTAL=false
HINDSIGHT_API_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL=false
```

The same block is in `.env.example`, commented out. The first four settings are hierarchical, so a single bank can take the profile through the [bank config API](./api/memory-banks.md) instead (`enable_observations`, `adaptive_memory_mode`, `adaptive_memory_score_alpha`, `context_admission_mode`); the last three are server-level. What each line does:

- **Observations on.** Lifecycle labels come from consolidation, which builds observations. Without observations consolidation labels nothing, and only the explicit lifecycle `PATCH` changes a claim.
- **`lifecycle` mode.** The [truth lifecycle](#truth-lifecycle): `active`, `uncertain`, `superseded` and `refuted` claims with their validity interval, current recall without retired claims, history through `lifecycle_view` and `as_of`, lifecycle relations and claim evidence that link an old claim to what replaced it (with snapshots of both ends), raw facts that inherit a retirement, scope-aware contradictions (a fact from another `environment`, `service` or `module` never retires a claim), and mental models refreshed when a memory they cite is retired.
- **Alpha `0`.** Retention state and reinforcement are still recorded and shown (`GET` on a memory, the trace's `score_components`, the metrics), but they never change ranking: while every candidate is active, a lifecycle-mode recall returns the same memories in the same order as `off`. It is the default; the line only makes it explicit.
- **Admission in `observe` with the deterministic scorer.** Every external recall reports in `trace.context_admission` and the metrics what `strict` would have injected, and returns exactly what it would without admission. No memory is sent to a decision model.
- **Admission's experimental switch off.** `filter`, `strict`, `jev` and `hybrid` are rejected, so a bank-config write cannot turn automatic filtering on.
- **Decay switch off.** A positive `adaptive_memory_score_alpha` is rejected, and a stored or tenant one reads as `0`, so a bank-config write cannot put retention back into ranking either.

Everything else stays at its default: archiving, pruning, recall reinforcement and the historical-cue detector are off, and a conflict stays `uncertain` until a later consolidation or an explicit `PATCH` settles it. The profile costs a larger consolidation prompt and schema, the claim and retention reads of every lifecycle recall, and the admission decisions of every external recall, which show most in tail latency under concurrent recalls.

### Optional: recall checks

Three [recall checks](./configuration.md#recall-checks) passed their benchmark bar and can be added to the profile, each on its own. A fourth, the Jev rerank, can be added as well for its retrieval gains, although it missed its bar on one endpoint (below). They need a TypeSafe key, and they are the only part of the profile that sends anything to a decision model: the query for all four, and the recalled memories for the gate, the flag and the rerank.

```bash
HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_PROVIDER=typesafe
HINDSIGHT_API_RECALL_CHECKS_DECISION_MODEL_API_KEY=your-typesafe-api-key
HINDSIGHT_API_RECALL_INJECTION_GATE=true
HINDSIGHT_API_RECALL_CONDITIONAL_RECENCY=true
HINDSIGHT_API_RECALL_RELEVANCE_CHECK=true
HINDSIGHT_API_RECALL_JEV_RERANK=true
```

The provider is set for the `recall_checks` operation only, not through `HINDSIGHT_API_DECISION_MODEL_PROVIDER`: the global setting would also replace recall's local reranker with the decision model and change consolidation and reflect. The model stays at the `typesafe` default, `jev-1.14.0`; the checks' questions and thresholds were developed on `jev-1.13.0` and re-measured unchanged on `jev-1.14.0`, where none of the three got worse, and the rerank was measured on `jev-1.14.0`. The four check settings are hierarchical, so a bank can take them through the bank config API instead (`recall_injection_gate`, `recall_conditional_recency`, `recall_relevance_check`, `recall_jev_rerank`). The evidence for each:

- **Poisoning guard** (`recall_injection_gate`). On jevmem's held-out set it withheld 40 of 42 planted instructions and none of its 42 legitimate rules, nor any of 519 ordinary memories (detection's 95% interval starts at 0.84). It changes what recall returns. If TypeSafe does not answer (an outage, an out-of-credit key), the memories it could not check are returned unchecked and the server logs a warning, so an outage never empties recall.
- **Conditional recency** (`recall_conditional_recency`). Applying the recency boost only to questions about the current or latest state raised LoCoMo temporal gold@5 from 0.916 to 0.947 (10 queries gained, none lost, exact McNemar p = 0.002) and left LongMemEval's knowledge-update latest@5 and earlier@5 and temporal-reasoning gold@5 unchanged; on the held-out half alone the gain (3 queries, none lost) is not significant. Every other question is ranked without the recency boost.
- **Nothing-relevant flag** (`recall_relevance_check`). It sets the response's `nothing_relevant` and removes nothing. On held-out coding-agent queries it flagged 7 of 8 that needed no memory and 3 of 38 that did, and 4 of 38 when the same requests were sent again, so treat it as a hint.
- **Jev rerank** (`recall_jev_rerank`). The local reranker's top 100 go to Jev in one request, with the date of the question and of each memory, and Jev's order replaces theirs; a failed request keeps the local order. On recorded LoCoMo and LongMemEval pools it put a gold memory in the top five for 0.961 of LoCoMo questions against 0.902 for the local order (114 gained, 23 lost, exact McNemar p < 0.001) and all of a temporal-reasoning question's evidence in the top five for 0.827 against 0.701 (18 gained, 2 lost). It missed its bar on knowledge updates: the latest value came first for 0.764 of those questions against 0.819 (8 gained, 12 lost, not significant), so where the newest value of a fact is what matters most, measure on your own bank before keeping it.

Each check adds TypeSafe requests to every REST and MCP recall on the bank: one for conditional recency, asked while retrieval runs; one for the flag; one for the rerank, about 10,000 input tokens for 100 memories; and one per ten screened memories (results and their source facts) for the gate. The rerank, the gate and the flag run after retrieval, one after the other, so their round trips add to recall latency.

If TypeSafe stops answering (unreachable, failing, rate-limited, or the key refused or out of credit), recall falls back the way each bullet above says, and it does not wait long for it: each request has [`HINDSIGHT_API_RECALL_CHECKS_TIMEOUT`](./configuration.md#recall-checks) (2 s) with its one retry, a recall stops asking after its first request that gets no answer, and after five such requests in a row a circuit breaker stops every recall from asking for 30 s, then lets one request probe the endpoint. While it is open a recall takes what it takes without the checks, and it returns what it would without them, so the poisoning guard is off until TypeSafe answers again. The server logs one warning when the breaker opens and one when it closes.

## Event time vs memory time

Five timestamps describe a memory, and they answer different questions. Mixing them up is the most common way to get history wrong.

| Field | Answers | Set by | Moves? |
|---|---|---|---|
| `occurred_start` / `occurred_end` | When did the thing the memory describes happen? | Fact extraction, from the content | Only when the fact is edited |
| `mentioned_at` | When was it said? | The retain request's `timestamp` (or the time of retain) | Never for a raw fact |
| `last_reinforced_at` | When was the memory last confirmed by independent evidence? | Reinforcement events | Forward only, on an applied reinforcement |
| `valid_from` / `valid_to` | Over which interval was this claim considered the current state? | Consolidation's lifecycle labels and the lifecycle API | When the claim is superseded, refuted or restored |

An example: on day 30 a conversation says "we upgraded billing to Java 25 last week". The new fact's `mentioned_at` is day 30 and its `occurred_start` is about day 23. The old "runs on Java 21" claim gets `valid_to` = day 30 and the new one `valid_from` = day 30, because the change instant is taken from when the facts asserting the new state were mentioned (falling back to when they occurred). Neither memory's `last_reinforced_at` changes. A `null` `valid_from` means "since before anything recorded", never "since it was mentioned".

Recall has two clocks of its own. [`query_timestamp`](./api/recall.md#query_timestamp) anchors ranking: recency and retrievability are computed as of that instant. [`as_of`](./api/recall.md#lifecycle_view-and-as_of) selects which claims were valid at an instant. Setting one never moves the other.

## Truth lifecycle

In `lifecycle` mode every memory can hold a claim:

| Status | Meaning | Current recall | Historical recall | `as_of` recall |
|---|---|---|---|---|
| `active` | The current state (the default; most memories never get a claim row) | Shown | Shown | Shown if valid at that instant |
| `uncertain` | Conflicting evidence with no clear winner | Shown, annotated | Shown, annotated | Shown if valid at that instant |
| `superseded` | Was true, and a later state replaced it | Hidden | Shown, annotated | Shown if valid at that instant |
| `refuted` | Was never true | Hidden | Hidden | Hidden |

Nothing is deleted. A retired memory keeps its row, its text, its history and an auditable trail: the claim records how it was retired (`retired_via`), the reason, its successor and the raw facts that justified the change (`evidence_memory_ids`), and a lifecycle relation (`supersedes`, `refutes`, `refines`, `splits`, `merges`, `conflicts`) links it to the memory that changed it. The relation keeps a snapshot of both ends, so it survives either end being deleted.

### How claims change

**Consolidation.** In `lifecycle` mode the consolidation prompt asks the model to label the changes it already makes, inside the same LLM call. It does not add a call:

- An update or a new observation can `supersede` an existing observation (a later state) or `refute` it (it was wrong).
- `conflicts_with` marks both sides `uncertain`.
- `refines` adds a narrower observation and keeps the broader one current.
- Two or more new observations that `split_from` the same observation split it: the over-general one is superseded by its narrower children.
- `merged_into` folds a duplicate into a survivor. A merged-away observation is a rewording, not an earlier state, so it never appears in a historical or `as_of` view.

Every label is validated before it is applied. The target must be in the batch's pool and still current, it must carry exactly the batch's tags, and the successor's evidence must not be older than the target's. A label that fails a check is dropped and counted (`lifecycle_transitions_dropped`), and the rest of the consolidation is applied as usual. Lifecycle changes never cross a tag scope: two memories with different tags are never compared, so a lifecycle change cannot leak across a boundary Hindsight isolates. Within one tag scope, the metadata keys in `HINDSIGHT_API_ADAPTIVE_MEMORY_SCOPE_KEYS` (by default `environment`, `service`, `module`) separate claims further: "prod runs Java 25" does not supersede "staging runs Java 21". The consolidation prompt shows the model each of those keys on every fact, along with `environment`, `service`, `module` and `repository`, so it keeps two scopes' values apart before this check has to drop a label; `repository` is shown but does not separate claims here unless it is added to the setting.

**Raw facts inherit.** Observations are not the only thing recall returns: by default it also returns the raw facts they were built from. When an observation is superseded or refuted, its raw source facts inherit the status (`retired_via: inherited`), so "runs on Java 21" stops answering as current along with the observation. A split parent's raw facts become `superseded` with `retired_via: splits` and no successor: the narrower observations keep them as evidence, not as sources, so they do not hold the over-general fact current. Some facts are never made to inherit:

- experience facts and facts with an `occurred_start`, because a dated event stays true;
- facts whose tags differ from the observation's;
- facts that another current observation still stands on.

**Explicitly.** `PATCH /v1/default/banks/{bank_id}/memories/{memory_id}/lifecycle` with `truth_status` (`superseded`, `refuted`, `uncertain` or `active` to restore), an optional `superseded_by` and a `reason`. See [Memories](./api/memories.md#adaptive-memory-lifecycle).

**Uncertainty is not resolved by repetition.** A conflict resolves on its own only through `HINDSIGHT_API_ADAPTIVE_MEMORY_SOURCE_TRUST`: a side wins only if its evidence comes from a more trusted metadata `source` and is also the more recent. That map is empty by default, so by default a conflict stays `uncertain` until a later consolidation or an explicit `PATCH` settles it. In reflect, uncertain memories are reported as a conflict rather than settled by recency.

### History views

A recall request picks a view with `lifecycle_view` (`current` or `historical`) or `as_of` (an ISO 8601 instant, which wins over `lifecycle_view`). Without either, the bank's `adaptive_memory_default_view` applies, or the historical-cue detector when `adaptive_memory_historical_cues` is on: an external query such as "what did we use before the upgrade?" is then answered from the historical view, and "…before 2026-02-01" as of that date. The view is applied before the reranker, so a retired memory never takes a slot from a current one. If the claim state cannot be read, a lifecycle-mode recall fails with HTTP 503 and `Retry-After` instead of returning memories it could not check.

The MCP recall tools have no view parameter, so their tool list is the same in every mode: they use the bank's default view and its historical cues. In reflect, the agent's recall tool gains an `include_history` flag in `lifecycle` mode, so the agent asks for history only when the question is about the past.

Results whose claim is not plain active carry a [`lifecycle`](./api/recall.md#lifecycle) annotation:

```json
{
  "id": "4c1e…",
  "text": "The billing service runs Java 21",
  "type": "observation",
  "lifecycle": {
    "truth_status": "superseded",
    "valid_from": null,
    "valid_to": "2026-01-31T00:00:00Z",
    "superseded_by": "9a07…"
  }
}
```

### Example: Java 21 to Java 25

1. **Day 0.** An inventory document says "The billing service runs on Java 21". Consolidation creates the observation *billing runs Java 21*.
2. **Day 30.** A new session says "The billing service was upgraded to Java 25". With the bank in `lifecycle` mode, consolidation emits the update it always would, labelled `supersedes`. That writes a new observation, *billing runs Java 25*, citing the new fact. The Java 21 observation is retired as superseded, with `valid_to` = day 30 and the Java 25 observation as its successor. The raw "runs on Java 21" fact inherits the status.
3. **Recall.** "Which Java version does billing run?" returns only the Java 25 observation and fact. With `"lifecycle_view": "historical"` the Java 21 pair comes back too, annotated. With `"as_of": "2026-01-15T00:00:00Z"` (day 15) only the Java 21 pair comes back.
4. **Mental models.** A mental model that cited the Java 21 observation is refreshed in full over current recall. Its refresh trace records `retraction.cause` and `successor_ids`.
5. **Audit.** `GET /v1/default/banks/{bank_id}/memories/{memory_id}` on the Java 21 fact shows the claim under `lifecycle.claim`: `truth_status: superseded`, `retired_via: inherited`, `inherited_from_id` (the observation) and `evidence_memory_ids` (the Java 25 fact).

Deleting the Java 25 document takes away all the evidence for the change, so the Java 21 claim reverts to active. Editing the Java 25 fact keeps it retired.

### Example: a false Redis claim

1. The README says "The checkout service caches sessions in Redis". Consolidation creates an observation from it.
2. A later correction says "The checkout service never used Redis; its sessions are stored in Postgres". Consolidation labels its update `refutes`.
3. The Redis observation and the Redis fact are refuted. They are hidden from current recall, from historical recall and from every `as_of` instant, because "it was Redis, then it became Postgres" would be history worth showing, and "it was never Redis" is not.
4. They stay auditable: `GET` on the Redis fact shows the claim, the reason and the correcting memory, and the `refutes` relation keeps both texts.
5. Re-ingesting the README extracts the Redis fact again under a new id. Its claim is carried forward to the re-extracted fact (matched by document and normalized text), so the refuted claim does not come back as a fresh, current memory.

The same correction can be made by hand:

### Python

```python
# A fact turned out to be wrong; point at the memory that corrects it.
await client.aupdate_memory_lifecycle(
    BANK_ID,
    memory_id,
    truth_status="refuted",
    superseded_by=other_id,
    reason="Recorded for the wrong subject",
)

# Keep an entry readily recallable however rarely it is used.
await client.aupdate_memory_lifecycle(BANK_ID, other_id, hold=True)
```

### Node.js

```javascript
// A fact turned out to be wrong; point at the memory that corrects it.
await client.updateMemoryLifecycle(BANK_ID, memoryId, {
    truthStatus: 'refuted',
    supersededBy: otherId,
    reason: 'Recorded for the wrong subject',
});

// Keep an entry readily recallable however rarely it is used.
await client.updateMemoryLifecycle(BANK_ID, otherId, { hold: true });
```

### CLI

```bash
# A fact turned out to be wrong; point at the memory that corrects it.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID/lifecycle" \
  -H "Content-Type: application/json" \
  -d '{"truth_status": "refuted", "superseded_by": "'"$OTHER_ID"'", "reason": "Recorded for the wrong subject"}'

# Keep an entry readily recallable however rarely it is used.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$OTHER_ID/lifecycle" \
  -H "Content-Type: application/json" \
  -d '{"hold": true}'
```

### Go

```go
// A fact turned out to be wrong; point at the memory that corrects it.
refute := hindsight.NewUpdateMemoryLifecycleRequest()
refute.SetTruthStatus("refuted")
refute.SetSupersededBy(otherID)
refute.SetReason("Recorded for the wrong subject")
client.MemoryAPI.UpdateMemoryLifecycle(ctx, memBankID, memoryID).
	UpdateMemoryLifecycleRequest(*refute).Execute()

// Keep an entry readily recallable however rarely it is used.
hold := hindsight.NewUpdateMemoryLifecycleRequest()
hold.SetHold(true)
client.MemoryAPI.UpdateMemoryLifecycle(ctx, memBankID, otherID).
	UpdateMemoryLifecycleRequest(*hold).Execute()
```

## Retention lifecycle

Retention models how readily a memory comes back without a strong cue. It is computed, never stored: `R = exp(-days_since_last_reinforced / stability_days)`, so `R` is `1` right after reinforcement and about `0.37` after one stability period.

**Initial stability** is deterministic and costs no LLM call: `base_stability_days × type multiplier × (0.5 + importance)`. With the defaults (30 days; experience `0.5`, world `1.0`, observation `2.0`; importance `0.5`), an unreinforced world fact has a stability of 30 days and an observation 60. A raw fact with no stored row is anchored on its `mentioned_at`.

**Ranking.** In `rank` and `lifecycle` the final recall score is multiplied by `(1 - alpha) + alpha × (floor + (1 - floor) × R)`. The default alpha is `0`, which makes the factor exactly `1`: retention is tracked and reported but never moves a score, because decay showed no measured ranking benefit in the benchmarks. A positive `adaptive_memory_score_alpha` is experimental: the server must set `HINDSIGHT_API_ADAPTIVE_MEMORY_DECAY_EXPERIMENTAL=true`, and without it a positive alpha fails start-up or the bank-config write, and a stored or tenant one reads as `0`. With it, alpha `0.1` and the default floor `0.3` give a factor between `0.93` and `1`. It only penalises, and it never removes a memory, so an old memory that matches a precise query is still found. Importance above `0.5` raises a memory's own floor.

**Reinforcement** raises stability. Each event has a source weight:

| Source | Weight | Produced by |
|---|---|---|
| `explicit_confirmation` | 1.0 | `POST /v1/default/banks/{bank_id}/memories/{memory_id}/confirm` |
| `independent_retain_confirmation` | 0.8 | Consolidation: an observation restated, unchanged, by a fact from a source it has not heard from before |
| `successful_reflect_use` | 0.6 | A successful reflect that cited the memory (server switch, on by default) |
| `observation_new_evidence` | 0.5 | Consolidation: an observation gaining a new source from new provenance |
| `final_external_recall` | 0.2 | An HTTP or MCP recall that returned the memory (server switch, off by default) |

Nothing else reinforces. Being a recall candidate, an internal recall, graph maintenance, a failed reflect or a mental-model refresh never counts. Neither does an observation being recalled: that does not reinforce its raw sources. The gain of one event is `reinforcement_gain × source weight × independence × spacing / sqrt(1 + reinforcement_count)`:

- **Spacing** is `1 - R` just before the event, with a minimum of `0.05`. A reconfirmation right after the last one adds little, and one that arrives after the memory has faded adds a lot.
- **Diminishing returns.** Each further reinforcement counts for less.
- **Cooldown.** An event within `HINDSIGHT_API_ADAPTIVE_MEMORY_REINFORCEMENT_COOLDOWN_HOURS` (6) of the last reinforcement adds nothing.
- **Independence** comes from the evidence key (next section).

Stability is capped at `max_stability_days` (3650).

**Importance and exemption.** A retain item's `importance` and `forgetting_exempt`, or the lifecycle `PATCH`, set a memory's retention policy. Importance is a prior: it scales initial stability, raises the ranking floor and protects from archiving, and an ordinary memory that keeps being confirmed still becomes stable. An exempt memory always ranks as fully retrievable. `hold: true` is an exemption that also guarantees the memory is never archived or pruned.

**Archiving and pruning** are separate server switches, both off by default:

- With `HINDSIGHT_API_ADAPTIVE_MEMORY_ARCHIVE_ENABLED`, a maintenance job archives memories whose retrievability stayed below the threshold for the grace period. An archived memory stays stored and is returned only on a strong cue. Any applied reinforcement, a `hold`, or `retention_state: "active"` brings it back. Important memories, exempt ones and evidence for a non-active claim are never archived.
- With `HINDSIGHT_API_ADAPTIVE_MEMORY_PRUNE_ENABLED` as well, memories archived longer than `HINDSIGHT_API_ADAPTIVE_MEMORY_PRUNE_AFTER_DAYS` are deleted. This cannot be undone.

## Four distinctions

### Recency decay is not forgetting

The `HINDSIGHT_API_RECENCY_DECAY_*` settings apply one small freshness adjustment to every memory by the age of its event dates, keep no per-memory state and are on in every mode. Retention is per memory and changes with evidence: a five-year-old fact that keeps being confirmed stays strong, and a fact from last week that nobody ever confirmed fades. Both run together, and neither one hides a memory. Archiving is the only part of retention that takes a memory out of ordinary recall, it is off by default, and it is reversible.

### Superseded is not refuted

"Billing ran Java 21, then moved to Java 25" and "checkout never used Redis" are different facts about the past. A superseded memory was true for an interval: historical recall and `as_of` show it, and its `valid_from` and `valid_to` say when. A refuted memory was never true: no view shows it, only `GET` on the memory and the relation trail do. Both are retired from current recall, both keep their text, and both can be restored with `truth_status: "active"`.

### Repetition is not independent confirmation

Saying something ten times in one conversation is one piece of evidence. Every reinforcement event carries an evidence key, taken from the strongest provenance the memory has, in this order:

1. `metadata.source_event_id`
2. `metadata.session_id`
3. a client-supplied `document_id`
4. `metadata.conversation_id`
5. `metadata.tool_execution_id`
6. a server-minted document id (half weight)
7. the memory id itself (half weight)

The ledger records each key once per memory and source type, so a retried confirmation with the same `source_key`, or the same session restating a fact, changes nothing. Bursts without provenance are absorbed by the cooldown. Observation proof counts are not a stability input, because they count repetitions.

With explicit confirmations of an ordinary world fact (initial stability 30 days, the defaults otherwise):

| Pattern | Stability | Retrievability on day 200 |
|---|---|---|
| Ten confirmations on day 0, the day the fact was mentioned | 30 days (all ten fall inside the cooldown) | about 0.001 |
| One confirmation on day 30 and one on day 120, from different sources | 49 days after day 30, 78 days after day 120 | about 0.36 |

Repetition also never decides truth: with no source trust configured, a conflict stays `uncertain` however many times one side is repeated.

### Curation invalidation is not refutation

[Invalidating a memory](./api/memories.md#invalidate-a-memory-reversible) (`PATCH /memories/{id}` with `state: "invalidated"`) is an operator's temporary removal, available in every mode: the row moves to the curation archive, the observations built on it are rebuilt, and restoring it brings back its retention, claim and links unchanged. It says "do not use this", not what is true. Refutation is a claim on a memory that stays live: the memory remains stored and auditable, records why it was wrong and what corrects it, propagates to the raw facts an observation stood on, and exists only in `lifecycle` mode. Use invalidation for a misextraction or noise; use refutation when the memory states something false that the bank should remember was false.

## Enabling observe mode safely

`observe` changes nothing a caller can see, so it can be turned on for one bank in production:

1. `PATCH /v1/default/banks/{bank_id}/config` with `{"updates": {"adaptive_memory_mode": "observe"}}`.
2. Recall with `"trace": true` and read `trace.adaptive`:
   - `mode`, `alpha` and `floor`: the parameters in force.
   - `scored`: false for the decision-model and interleave orders, which are never re-weighted.
   - `states_loaded`: how many candidates had stored state; the rest used defaults.
   - `simulated_final_ids` and `top_k_changed`: the ids `rank` would have returned, and whether they differ from the ids actually returned.
   - `state_load_error`: set when the state could not be read; the recall then continued unchanged.

   Each entry of `trace.reranked` carries the per-memory inputs in `score_components`: `retrievability`, `stability_days`, `importance`, `reinforcement_count`, `retention_factor` and `adaptive_factor`.
3. Watch the [metrics](./monitoring.md#adaptive-memory-metrics):
   - `hindsight_adaptive_memory_top_k_changed_total` against your recall count: how often `rank` would change what callers see.
   - `hindsight_adaptive_memory_retrievability_bucket` and `hindsight_adaptive_memory_stability_days_bucket`: the distribution of what recall returns.
   - `hindsight_adaptive_memory_state_load_errors_total`: should stay at zero.
4. Look at the shadow records. Observe mode does not change the consolidation prompt, so it labels nothing. When consolidation deletes an observation that it replaced, observe mode records, on each raw fact that would have inherited a retirement, what `lifecycle` mode would have done. `GET` on the fact shows it under `lifecycle.shadow` (`would_status`, the deciding observation, the evidence ids). Recall never reads these records.
5. Move to `lifecycle` once you want superseded memories out of current recall. At the default alpha of `0`, `rank` changes no order, so `simulated_final_ids` matters only for a bank that raises `adaptive_memory_score_alpha` on a server with the decay switch.

Prefer not to downgrade a bank that holds claims. `rank`, `observe` and `off` keep the claim rows but stop applying them, so recall shows superseded and refuted memories again, unannotated. Consolidation in the lower mode can then rewrite a retired observation as if it were current.

## What else carries the state

- **Curation.** Invalidate then restore keeps a memory's stability, reinforcement count and claim.
- **Re-ingest.** Re-extracted facts inherit the retention and claim of the old fact with the same document and normalized text. A fact that changed text starts fresh, and its old state is dropped and counted.
- **Transfer.** Export, import and clone carry retention, claims, claim evidence and lifecycle relations, re-keyed onto the target's ids. See [Memory banks](./api/memory-banks.md#bank-transfer-export--import).
- **Deletes.** Deleting memories removes their retention rows, events and evidence. A claim whose evidence is all gone reverts to active. Relations stay as tombstones with their snapshots.

## Context admission

The two axes above decide what is believed and what is remembered. A third question is separate from both: what should enter an agent's context for the task in front of it. A memory can be current and well remembered and still be the wrong thing to inject: a joke that shares words with a deploy question, another project's deployment, a debugging session from a different subsystem. Recall returns the best candidates it has, and the best of a bad set is still irrelevant. Context admission is a gate after recall's ranking that judges each top candidate for the task and may admit none. It has its own setting, `context_admission_mode` (off by default, listed under [Context admission](./configuration.md#context-admission) in the configuration reference), and works with or without adaptive memory.

Only `observe` is available by default. The modes that change what recall returns (`filter`, `strict`) and the decision-model scorers (`jev`, `hybrid`) met none of their benchmark bars, so they are experimental and need the server-level `HINDSIGHT_API_CONTEXT_ADMISSION_EXPERIMENTAL=true`; without it, setting one fails start-up or the bank-config write. `observe` always scores with the deterministic scorer, whatever the switch says, so it never sends a memory to a decision model.

When lifecycle state is there, the gate uses it without letting it stand in for relevance: a refuted memory is never admitted, a superseded one only for a historical recall, an uncertain one only with a margin above the threshold, and retrievability and importance are two inputs among many. A memory with high stability and no bearing on the task is rejected; a faded memory that answers the question exactly is admitted.

### Purpose and context

A recall request says why it is made (`purpose`) and where the caller works (`context`). See [`purpose` and `context`](./api/recall.md#purpose-and-context) for the request fields.

| `purpose` | For | Threshold |
|---|---|---|
| `auto_context` | Memories injected that nobody asked for, such as a coding agent's first-prompt recall | `context_admission_threshold_auto` (0.6). A query that refers to shared history ("what did we decide", "last time", "do you remember") lowers it by `context_admission_reference_bonus` |
| `explicit_search` (default) | A person or an agent that asked for memories | `context_admission_threshold_explicit` (0.35) |
| `historical_search` | How things were | The explicit threshold. Superseded memories are admissible, and in `lifecycle` mode the view is historical unless the request names one |

`context` keys are compared with memory metadata of the same name, each a match, a mismatch or unknown. Soft keys (module, branch, environment, service) move the score and never filter on their own: a decision made on `main` usually still holds on a feature branch. `repository` is a hard key in `strict` mode: when the request and the memory both name one and they differ, the memory is rejected, so another project's memories never reach the context. Any key in `HINDSIGHT_API_CONTEXT_ADMISSION_HARD_SCOPE_KEYS` is compared, whether or not it is also a soft key. An observation carries no metadata of its own, so it is compared on the values of the facts it was consolidated from: it matches when the request's value is among them and mismatches when none of them is it, which keeps another repository's observations out of `strict` recalls too. The keys only work if the same values are written at retain as `metadata` and sent at recall; the [coding-agents integration](../sdks/integrations/coding-agents.md) stamps `repository`, `branch` and `module` on what it retains and sends them with its automatic recall. MCP recall takes no purpose or context and is judged as `explicit_search`, because an MCP recall is the agent searching on purpose.

### Empty is an answer

In `filter` and `strict` (experimental), a recall can come back with `results: []` because no candidate cleared the threshold. That is the gate working, not a failure: injecting the least bad memory is how a joke ends up in a deployment answer. To tell an abstention from an empty bank, read the response's `context_admission` object, present only in these two modes: `abstained` is true when candidates were judged and none was admitted, and `candidates` is 0 when retrieval found nothing to judge. With `"trace": true`, `trace.context_admission` also has `reasons`, which counts the rejections (`wrong_scope`, `below_threshold`, `superseded`, `refuted`, `redundant`, `budget`). `strict` also suppresses near-duplicates (the kept memory's `decisions[]` entry counts them in `duplicates`) and stops at `context_admission_max_memories` memories or `context_admission_max_tokens` tokens. The token budget is filled by value per token; when the memory count is what binds, the most valuable memories are kept first, and a near-duplicate takes the place of its longer twin when only the twin fits.

The deterministic scorer leans on the cross-encoder's score. When recall's reranker scores by rank instead (`reranking: "rrf"`, an `rrf` reranker or a failover onto one, the `typesafe` and `decision_model` rerankers), those scores are positions, which would admit the top of any pool, so the scorer reads the semantic similarity in their place and drops the rank feature; `trace.context_admission.relevance_source` is then `semantic`.

### Rolling it out

1. `PATCH /v1/default/banks/{bank_id}/config` with `{"updates": {"context_admission_mode": "observe"}}`. Nothing a caller sees changes.
2. Recall with `"trace": true` and the `purpose` and `context` your callers will send, and read `trace.context_admission`:
   - `decisions[]`: one per judged candidate, with `base_retrieval_score`, `scope_score`, `task_relevance`, `expected_utility`, `truth_status`, `retrievability`, `admission_score`, `admitted`, `reason`, the per-key `scope` result and the scorer's input `features`;
   - `shadow_final_ids`, `would_drop` and `would_change_top_k`: what `strict` would have returned, and how it differs.
3. Watch the [metrics](./monitoring.md#context-admission-metrics): `would_drop_count`, `would_abstain_count`, `would_change_top_k` and `cross_scope_candidates_detected`. Observe simulates `strict`, so it counts hard scope, duplicates and the budget too; `filter` drops fewer.
4. Keep `observe` for normal use. To experiment with automatic injection, start the server with `HINDSIGHT_API_CONTEXT_ADMISSION_EXPERIMENTAL=true` and move a bank to `filter` (or `strict` for hard scope and the budget) once the would-drop memories look right. Keep `purpose: "explicit_search"` where people search, which uses the lower threshold.

While a bank is in `filter` or `strict`, the recalls it gates never count as reinforcement, so a memory cannot earn its own admission by being admitted.

## Known limitations

- **Oracle is verified statically only.** The Oracle paths (claim CAS, inheritance, archiving, transfer) are covered by SQL-rewriter and fake-connection tests, not yet by runs against a live Oracle 23ai. The periodic maintenance loop does not run on Oracle, so there archiving happens only after consolidations.
- **Lifecycle classification is only as good as the consolidation model.** The labels come from the consolidation LLM call. A weaker model misses changes (the stale memory stays current) or mislabels them, and the lifecycle prompt and response schema make each consolidation call larger. Track `hindsight_adaptive_memory_lifecycle_transitions_dropped_total` and try `lifecycle` on a representative bank before enabling it widely.
- **A wrong supersession hides more than one memory.** When an observation is wrongly superseded or refuted, the raw facts it stood on inherit the status and leave current recall too. `GET` the memory to see what retired it, and `PATCH …/lifecycle` with `truth_status: "active"` on the observation to restore it together with the facts that inherited from it.
- **Bulk deletes cost more, in every mode.** The adaptive tables hold foreign keys to `memory_units`, so each deleted memory is checked against them even when they are empty. Measured on embedded PostgreSQL, deleting 50,000 memories took 1.9 s instead of 0.4 s, about 29 µs more per row. This affects bank deletes, document deletes and chunk replacement on servers with the feature off as well.
- **Context admission is not calibrated yet.** The deterministic scorer's weights and the two thresholds are hand-set starting values, not fitted to labelled data, and no gating mode met its benchmark bars, which is why `filter` and `strict` are experimental. Run `observe`, and fit `HINDSIGHT_API_CONTEXT_ADMISSION_WEIGHTS` on your own recalls from the trace's `features` before experimenting with `filter`. The deterministic scorer has no signal for humor or casual memories; routing those relies on the experimental `jev` or `hybrid` scorer.
- **Reinforcement events are not exported.** A transfer carries each memory's resulting state, not the event ledger, so after an import a confirmation re-sent with an old `source_key` reinforces the memory again.
