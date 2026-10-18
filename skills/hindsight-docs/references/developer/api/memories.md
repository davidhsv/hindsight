
# Memories

A **memory unit** is the atomic fact Hindsight extracts and stores. This page covers the endpoints for working with individual memory units — reading and listing them, inspecting how a derived observation evolved, and **curating** them (correcting, retiring, or restoring). Ingesting and querying memories is covered separately in [Retain](./retain.md) and [Recall](./recall.md).

{/* Import raw source files */}

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/v1/default/banks/{bank}/memories/list` | List/filter memory units in a bank |
| `GET` | `/v1/default/banks/{bank}/memories/{id}` | Fetch a single memory unit |
| `GET` | `/v1/default/banks/{bank}/memories/{id}/history` | Refresh history of a derived observation |
| `PATCH` | `/v1/default/banks/{bank}/memories/{id}` | Curate: edit / invalidate / restore |
| `POST` | `/v1/default/banks/{bank}/memories/{id}/confirm` | [Adaptive memory](../adaptive-memory.md): confirm that a memory still holds |
| `PATCH` | `/v1/default/banks/{bank}/memories/{id}/lifecycle` | Adaptive memory: retention policy and truth status |
| `DELETE` | `/v1/default/banks/{bank}/memories/{id}/observations` | Clear a memory's derived observations |

## List memory units

List the memory units in a bank. The response includes each unit's `fact_type` (`world` | `experience` | `observation`), `state` (`valid` | `invalidated`), metadata, entities, occurred dates, and — for facts a user has edited — an `edited_at` timestamp. Invalidated rows are **included by default** so curation stays auditable; filter with `state=`.

Narrow the results with query parameters: `type=` (fact type), `q=` (full-text search over text and context), `document_id=` (a single source document), and `entity_id=` (memory units linked to a given entity). The `entity_id` filter is an exact reverse lookup over stored entity links — not a text or semantic match — so you can list an entity's evidence (e.g. its observations, with `type=observation`) without scanning every memory. Because entity links exist only for live units, combining `entity_id` with `state=invalidated` returns nothing.

### Time window

`start_date=` and `end_date=` restrict the results to a half-open range `[start, end)`, and `time_field=` chooses which timestamp they apply to:

| `time_field` | Means |
|---|---|
| `created_at` (default) | When the memory was ingested |
| `updated_at` | When it was last written |
| `mentioned_at` | When the fact was mentioned |
| `occurred_start` / `occurred_end` | When the fact itself occurred |

`time_field` also becomes the sort order (newest first), and **memories with no value on that column are excluded** — so `total` counts only the memories carrying that timestamp, and can be `0` on a bank that is not empty. Omit all three parameters to keep the default listing, which orders by `mentioned_at` then `created_at` and drops nothing.

This filters, unlike recall's `temporal_window`, which only ranks — use it when you need an exhaustive, time-bounded read rather than the best matches.

### Python

```python
# List memory units in a bank. Invalidated rows are included by default.
memories = await client.memory.list_memories(bank_id=BANK_ID)
for unit in memories.items:
    print(f"- [{unit.fact_type}] {unit.text}")

# Filter to only the invalidated facts (e.g. to review duplicates).
invalidated = await client.memory.list_memories(bank_id=BANK_ID, state="invalidated")
print(f"{len(invalidated.items)} invalidated fact(s)")
```

### Node.js

```javascript
// List memory units in a bank. Invalidated rows are included by default.
const memories = await client.listMemories(BANK_ID);
for (const unit of memories.items) {
    console.log(`- [${unit.fact_type}] ${unit.text}`);
}

// Filter to only the invalidated facts (e.g. to review duplicates).
const invalidated = await client.listMemories(BANK_ID, { state: 'invalidated' });
console.log(`${invalidated.items.length} invalidated fact(s)`);
```

### CLI

```bash
# List memory units in a bank (invalidated rows are included by default)
hindsight memory list "$BANK_ID"

# Filter to only the invalidated facts (e.g. to review duplicates)
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/list?state=invalidated"
```

### Go

```go
// List memory units in a bank. Invalidated rows are included by default.
memories, _, _ := client.MemoryAPI.ListMemories(ctx, memBankID).Execute()
for _, unit := range memories.GetItems() {
	fmt.Printf("- [%v] %v\n", unit.GetFactType(), unit.GetText())
}

// Filter to only the invalidated facts (e.g. to review duplicates).
invalidated, _, _ := client.MemoryAPI.ListMemories(ctx, memBankID).State("invalidated").Execute()
fmt.Printf("%d invalidated fact(s)\n", len(invalidated.GetItems()))
```

## Fetch a single memory unit

Fetch a memory unit by ID, including its content, metadata, entities, timestamps, tags, and curation state.

For a bank whose [adaptive memory](../adaptive-memory.md) mode is not `off`, or a memory that has adaptive state stored, the response also carries a [`lifecycle`](#the-lifecycle-block) block. An off-mode bank's memories with nothing stored keep exactly the payload they always had.

### Python

```python
# Fetch a single memory unit (includes metadata, entities, dates, and state).
memory = await client.memory.get_memory(bank_id=BANK_ID, memory_id=memory_id)

print(f"Text: {memory['text']}")
print(f"Type: {memory['type']}  Entities: {memory['entities']}")
```

### Node.js

```javascript
// Fetch a single memory unit (metadata, entities, dates, state).
const memory = await (
    await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${memoryId}`)
).json();
console.log(`Text: ${memory.text}`);
console.log(`Type: ${memory.type}  Entities: ${memory.entities}`);
```

### CLI

```bash
# Fetch a single memory unit (metadata, entities, dates, state)
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID"
```

### Go

```go
// Fetch a single memory unit (metadata, entities, dates, state).
memory, _, _ := client.MemoryAPI.GetMemory(ctx, memBankID, memoryID).Execute()
fmt.Printf("Memory: %v\n", memory)
```

For a **derived observation**, the history endpoint returns how it was refreshed over time as new source facts arrived:

### Python

```python
# Get the refresh history of a derived observation.
history = await client.memory.get_observation_history(
    bank_id=BANK_ID, memory_id=observation.id
)
print(f"Observation history entries: {len(history)}")
```

### Node.js

```javascript
// Get the refresh history of a derived observation.
const history = await (
    await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${observation.id}/history`)
).json();
console.log(`Observation history entries: ${history.length}`);
```

### CLI

```bash
# Get the refresh history of a derived observation
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$OBSERVATION_ID/history"
```

### Go

```go
// Get the refresh history of a derived observation.
history, _, _ := client.MemoryAPI.GetObservationHistory(ctx, memBankID, observationID).Execute()
fmt.Printf("Observation history: %v\n", history)
```

## Curation: editing, invalidating & pruning

Memory is append-only by design — but sometimes a stored fact is **wrong**, has gone **stale**, or is a **duplicate**. Curation lets you correct or retire individual memories without losing the audit trail. Retired facts are moved out of the active set, so recall never returns them, while remaining fully recoverable.

### When to reach for what

Not every "bad memory" needs the same tool. Pick by *why* it's bad:

| The memory is… | Use | Why |
|---|---|---|
| **Wrong because the whole bank extracts badly** (e.g. consistently wrong subject) | Fix the bank's `retain_mission` / `observations_mission`, then **reprocess** the document | Systematic problems are best fixed at the source, then replayed — see [Retain](./retain.md) and [Observations](../observations.md). |
| **Wrong as a one-off** (a single misextracted fact) | **Edit** the memory | Corrects the fact and regenerates everything derived from it. |
| **No longer true, with nothing to replace it** (decommissioned server, a tool that was fixed, a role that changed) | **Invalidate** the memory | Nothing in the pipeline knows the world changed, so you tell it explicitly. |
| **A duplicate or superseded fact** | **Invalidate** the memory | Removes the noise from recall while keeping the audit trail. |
| **Superseded by a newer fact you're storing anyway** (e.g. "likes BMW" → "likes Toyota") | Just retain the new fact | Consolidation already reconciles in-stream contradictions into a single observation. |

The rule of thumb: **if Hindsight could have known, let consolidation handle it; if only you know, curate it.**

Only raw **world** and **experience** facts can be curated. Observations are *derived* — they regenerate from their sources, so you curate the underlying facts, not the observation. A `PATCH` on an observation returns `400`.

### Edit a memory

Correct what the LLM extracted. You can change the **text**, **context**, **occurred dates**, **fact type**, and **entities** — anything the extractor could have gotten wrong. Hindsight re-embeds the fact, drops the observations and links derived from the old version, and re-consolidates, so downstream knowledge reflects the correction. Edited facts are marked with an `edited_at` timestamp (surfaced as an **Edited** badge in the control plane).

Correcting a fact never discards the **cause-and-effect relationships** it takes part in: those come from reading the original source, so an edit (or an invalidate/restore round-trip) keeps them intact rather than dropping links nothing could rebuild.

You don't need to rebuild anything yourself: an edit **automatically recomputes the knowledge graph and links** in the background. The fact's entity associations are re-resolved from the new text/entities, its temporal and semantic links are re-derived, and consolidation re-runs — all triggered by the edit. The PATCH returns as soon as the change is committed; the graph/observation rebuild happens asynchronously right after.

### Python

```python
# Correct the fact's text. Re-embeds, drops derived observations/links,
# re-consolidates, and recomputes the graph automatically.
await client.memory.update_memory(
    bank_id=BANK_ID,
    memory_id=memory_id,
    update_memory_request=UpdateMemoryRequest(
        text="The user visited Paris in 2023.",
        reason="wrong subject",
    ),
)
```

### Node.js

```javascript
// Correct the fact's text. Re-embeds, drops derived observations/links,
// re-consolidates, and recomputes the graph automatically.
await patchMemory(memoryId, { text: 'The user visited Paris in 2023.', reason: 'wrong subject' });
```

### CLI

```bash
# Correct the fact's text. Re-embeds, drops derived observations/links,
# re-consolidates, and recomputes the graph automatically.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
  -H "Content-Type: application/json" \
  -d '{"text": "The user visited Paris in 2023.", "reason": "wrong subject"}'
```

### Go

```go
// Correct the fact's text. Re-embeds, drops derived observations/links,
// re-consolidates, and recomputes the graph automatically.
client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
	UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
		Text:   *hindsight.NewNullableString(hindsight.PtrString("The user visited Paris in 2023.")),
		Reason: *hindsight.NewNullableString(hindsight.PtrString("wrong subject")),
	}).Execute()
```

You can correct the dates, fact type, and entities the same way. For `context`, `occurred_start`, and `occurred_end`, an empty string `""` clears the field and omitting it leaves it unchanged. For `entities`, a list **replaces** the fact's entity set and `[]` detaches them all; omitting it leaves them unchanged.

### Resolving entity names

`resolve_entities` controls how the names in `entities` are matched to entities in the bank:

| Value | Behaviour |
| --- | --- |
| `true` (default) | What retain does. Each name is resolved against the bank, so a name close to one already there may resolve to that existing entity instead, based on name similarity plus how strongly it co-occurs with the other names you sent. |
| `false` | The names are taken literally. An existing entity is reused only when its name matches case-insensitively, any other name creates a new entity, and names in the same request are never merged with each other. |

**Pass `false` when you are correcting a fact by hand.** With resolution on, a name that is close to one already in the bank can be matched onto that neighbour rather than the entity you named — `Dr. Waller` onto a `Dr Wall` typo, `Alice Smith` onto `Alice` — and because the edit succeeds normally the substitution is not obvious from the response. Resolution is right for names that came out of extraction, where spelling varies and the bank's existing entity is usually the one meant; it is wrong when you already know which entity you want. The default stays `true` so existing callers are unaffected.

The same flag exists on [retain](./retain.md#resolve_entities) for the entities you supply there.

### Python

```python
# Correct dates, fact type, and entities in one call. "" clears a field;
# entities replaces the set ([] detaches all); omit to leave unchanged.
# resolve_entities=False keeps the entity names you wrote from being matched
# onto a similar entity that already exists.
await client.memory.update_memory(
    bank_id=BANK_ID,
    memory_id=memory_id,
    update_memory_request=UpdateMemoryRequest(
        occurred_start="2023-06-01",
        fact_type="experience",
        entities=["Alice", "Paris"],
        resolve_entities=False,
    ),
)
```

### Node.js

```javascript
// Correct dates, fact type, and entities in one call. "" clears a field;
// entities replaces the set ([] detaches all); omit to leave unchanged.
// resolve_entities: false keeps the entity names you wrote from being matched
// onto a similar entity that already exists.
await patchMemory(memoryId, {
    occurred_start: '2023-06-01',
    fact_type: 'experience',
    entities: ['Alice', 'Paris'],
    resolve_entities: false,
});
```

### CLI

```bash
# Correct dates, fact type, and entities in one call. "" clears a field;
# entities replaces the set ([] detaches all); omit to leave unchanged.
# resolve_entities false keeps the entity names you wrote from being matched
# onto a similar entity that already exists.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
  -H "Content-Type: application/json" \
  -d '{"occurred_start": "2023-06-01", "fact_type": "experience", "entities": ["Alice", "Paris"], "resolve_entities": false}'
```

### Go

```go
// Correct dates, fact type, and entities in one call. "" clears a field;
// entities replaces the set ([] detaches all); omit to leave unchanged.
// ResolveEntities false keeps the entity names you wrote from being matched
// onto a similar entity that already exists.
client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
	UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
		OccurredStart:   *hindsight.NewNullableString(hindsight.PtrString("2023-06-01")),
		FactType:        *hindsight.NewNullableString(hindsight.PtrString("experience")),
		Entities:        []string{"Alice", "Paris"},
		ResolveEntities: hindsight.PtrBool(false),
	}).Execute()
```

### Invalidate a memory (reversible)

Soft-retire a fact. An invalidated memory:

- **disappears from recall**, consolidation, and the knowledge graph,
- has its **links pruned** and its **derived observations re-computed** without it (cause-and-effect relationships are kept aside and come back if you restore it),
- **stays in the bank** for audit (visible via the memory and document views), and
- can be **restored** at any time.

### Python

```python
# Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
# derived observations recomputed without it — but kept for audit.
await client.memory.update_memory(
    bank_id=BANK_ID,
    memory_id=memory_id,
    update_memory_request=UpdateMemoryRequest(
        state="invalidated",
        reason="server decommissioned 2026-06-01",
    ),
)
```

### Node.js

```javascript
// Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
// derived observations recomputed without it — but kept for audit.
await patchMemory(memoryId, { state: 'invalidated', reason: 'server decommissioned 2026-06-01' });
```

### CLI

```bash
# Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
# derived observations recomputed without it — but kept for audit.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
  -H "Content-Type: application/json" \
  -d '{"state": "invalidated", "reason": "server decommissioned 2026-06-01"}'
```

### Go

```go
// Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
// derived observations recomputed without it — but kept for audit.
client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
	UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
		State:  *hindsight.NewNullableString(hindsight.PtrString("invalidated")),
		Reason: *hindsight.NewNullableString(hindsight.PtrString("server decommissioned 2026-06-01")),
	}).Execute()
```

Restoring moves the fact back into the active set, brings back the cause-and-effect relationships it took part in, and re-consolidates:

### Python

```python
# Restore a previously invalidated fact.
await client.memory.update_memory(
    bank_id=BANK_ID,
    memory_id=memory_id,
    update_memory_request=UpdateMemoryRequest(state="valid"),
)
```

### Node.js

```javascript
// Restore a previously invalidated fact.
await patchMemory(memoryId, { state: 'valid' });
```

### CLI

```bash
# Restore a previously invalidated fact.
curl -s -X PATCH "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$MEMORY_ID" \
  -H "Content-Type: application/json" \
  -d '{"state": "valid"}'
```

### Go

```go
// Restore a previously invalidated fact.
client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
	UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
		State: *hindsight.NewNullableString(hindsight.PtrString("valid")),
	}).Execute()
```

Behind the scenes, invalidating **moves** the row out of the active `memory_units` table into a separate archive, so recall and consolidation never need a "skip invalidated" filter — the rows simply aren't there.

> **📝 Documents are the source of truth**
>
A memory is extracted from a document. Editing or invalidating a memory does **not** change the document it came from — that's deliberate: the document stays as an accurate historical record. As a result, **reprocessing a document resets curation** of the facts it produced (extraction runs fresh from the original text). Fix systematic issues at the mission level and reprocess; use edit/invalidate for the residue.
### A pruning workflow

To clean up duplicates and reclaim noise: cluster duplicates from `memories/list`, then **invalidate** them — recall is clean immediately, and the audit trail is preserved.

## Adaptive memory lifecycle

These endpoints read and change a memory's [adaptive memory](../adaptive-memory.md) state: its **retention** (how readily it comes back) and, in `lifecycle` mode, its **claim** (whether it is still the current state). They work on facts and observations alike, unlike curation above, which only applies to raw facts. The Python and TypeScript clients wrap them; the CLI has no command for them yet, so its examples call the HTTP API.

### The lifecycle block

`GET /v1/default/banks/{bank}/memories/{id}` adds `lifecycle`:

| Field | Contents |
|---|---|
| `retention` | `retrievability` (computed now), `stability_days`, `importance`, `last_reinforced_at`, `reinforcement_count`, `access_count`, `last_recalled_at`, `forgetting_exempt`, `exemption_reason`, `retention_state` (`active` or `archived`), `below_threshold_since`, `archived_at`. `materialized: false` means no row is stored yet and the values are the defaults the memory is ranked with. |
| `claim` | `truth_status`, `confidence`, `valid_from`, `valid_to`, `retired_via` (`supersedes`, `refutes`, `merges`, `splits`, `explicit`, `inherited`), `superseded_by_id`, `inherited_from_id`, `reason`, `evidence_memory_ids`. `null` for a memory that never had a claim, which reads as active. |
| `relations_out` / `relations_in` | Lifecycle relations from and to this memory (`supersedes`, `refutes`, `refines`, `splits`, `merges`, `conflicts`), each with its `origin` (`consolidation`, `explicit` or `inherited`), reason and snapshots of both ends. An end that was deleted is `null`; its snapshot remains. |
| `recent_events` | The last 20 reinforcement events: `source_type`, `source_key`, `event_time`, `weight`, `independence_weight`, `applied_gain`, `outcome` (`applied`, `cooldown`, `initial`). |
| `shadow` | Observe mode only: what `lifecycle` mode would have done to this fact (`would_status`, the deciding observation, evidence ids). Never used by recall. |
| `archived`, `stored` | Whether the memory is archived, and whether any adaptive state is stored for it. |

A bank switched back to `off` still shows what it stored, so the audit trail survives a downgrade.

### Confirm a memory

`POST /v1/default/banks/{bank}/memories/{id}/confirm` records explicit evidence that a memory still holds. It is the strongest reinforcement (weight `1.0`), so the memory stays readily recallable for longer. All fields are optional:

| Field | Meaning |
|---|---|
| `source_key` | Identifies this piece of evidence. A memory is confirmed at most once per key, so a retry is reported as `deduplicated` and changes nothing. Omitted, every request counts as a separate confirmation. |
| `event_time` | When the confirmation happened (ISO 8601, defaults to now). It may be in the past, not in the future. |
| `reason` | A note for the audit log; not stored with the memory. |

The response says what happened: `recorded`, `deduplicated`, `outcome` (`applied`, `cooldown` when it came too soon after the last reinforcement to add stability, or `deduplicated`), and the memory's `stability_days` and `reinforcement_count` after the call. Needs mode `rank` or `lifecycle` with reinforcement enabled (`409` otherwise; `observe` never changes state); `404` for a memory that is missing or invalidated.

### Python

```python
# Record that a memory still holds; it stays readily recallable for longer.
result = await client.aconfirm_memory(
    BANK_ID,
    other_id,
    source_key="incident-2291",  # the same key again is a no-op
)
print(result.outcome, result.stability_days)
```

### Node.js

```javascript
// Record that a memory still holds; it stays readily recallable for longer.
const result = await client.confirmMemory(BANK_ID, otherId, {
    sourceKey: 'incident-2291', // the same key again is a no-op
});
console.log(result.outcome, result.stability_days);
```

### CLI

```bash
# Record that a memory still holds; the same source_key again is a no-op.
curl -s -X POST "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/$OTHER_ID/confirm" \
  -H "Content-Type: application/json" \
  -d '{"source_key": "incident-2291"}'
```

### Go

```go
// Record that a memory still holds; the same source key again is a no-op.
confirm := hindsight.NewConfirmMemoryRequest()
confirm.SetSourceKey("incident-2291")
result, _, _ := client.MemoryAPI.ConfirmMemory(ctx, memBankID, otherID).
	ConfirmMemoryRequest(*confirm).Execute()
if result != nil {
	fmt.Println(result.GetOutcome(), result.GetStabilityDays())
}
```

### Change retention or truth status

`PATCH /v1/default/banks/{bank}/memories/{id}/lifecycle` takes any of these fields; omitted ones are left as they are, and at least one is required. It returns the memory's lifecycle block after the change.

| Field | Mode | Effect |
|---|---|---|
| `importance` | `rank`, `lifecycle` | How much the memory matters, in `[0, 1]` (`0.5` is neutral). Rescales the stability it has learned by the ratio of the new and old importance, so earlier reinforcement is kept. At or above `HINDSIGHT_API_ADAPTIVE_MEMORY_ARCHIVE_PROTECT_IMPORTANCE` it is never archived. |
| `forgetting_exempt`, `exemption_reason` | `rank`, `lifecycle` | `true` exempts the memory from retention decay (it always ranks as fully retrievable); `false` lifts the exemption and clears its reason. |
| `hold` | `rank`, `lifecycle` | `true` places the memory on hold: never archived or pruned, always fully retrievable (stored as an exemption with reason `hold`). Not combined with `forgetting_exempt` or `exemption_reason`. |
| `retention_state` | `rank`, `lifecycle` | `"active"` returns an archived memory to normal recall. |
| `truth_status` | `lifecycle` | `superseded` or `refuted` retires the memory from current recall; `active` restores it; `uncertain` marks it contested. |
| `superseded_by` | `lifecycle` | With `superseded` or `refuted`, the memory that replaces this one: a different live memory of the same bank with exactly the same tags. |
| `reason` | `lifecycle` | Why the truth status changed; recorded on the claim and its relation. |

Retiring an **observation** retires the raw facts that only it stood on (experience facts and dated events excepted). Retiring a **raw fact** rebuilds the observations built on it from their other sources. Restoring an observation restores the facts that inherited its status. The change is recorded as an `explicit` relation, and mental models that cited the memory are refreshed.

Errors: `409` when the bank's mode does not allow the change or the memory is already retired (restore it first); `404` when the memory or `superseded_by` is missing or invalidated; `400` for a successor with different tags, a successor built on the memory itself, or a successor in another claim scope.

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

A retain item can set the same retention policy at write time with `importance` and `forgetting_exempt` (see [Retain](./retain.md#importance-and-forgetting_exempt)).
