# Monitoring

Hindsight provides comprehensive observability through Prometheus metrics, OpenTelemetry distributed tracing, and pre-built Grafana dashboards.

## Local Development

For local observability, use the Grafana LGTM (Loki, Grafana, Tempo, Mimir) all-in-one stack:

```bash
./scripts/dev/start-monitoring.sh
```

This starts a single Docker container providing:
- **Grafana UI**: http://localhost:3000 (anonymous admin access)
- **Traces (Tempo)**: OTLP endpoint at http://localhost:4318 (HTTP) and http://localhost:4317 (gRPC)
- **Metrics (Prometheus/Mimir)**: Scrapes http://localhost:8888/metrics automatically
- **Logs (Loki)**: Available for log aggregation
- **Pre-built Dashboards**: Hindsight Operations, LLM Metrics, API Service

**Enable tracing in your API:**
```bash
export HINDSIGHT_API_OTEL_TRACES_ENABLED=true
export HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

:::note Production Deployment
The local monitoring stack is for development only. In production, deploy Grafana LGTM separately or use commercial platforms (Grafana Cloud, DataDog, New Relic, etc.).
:::

## Grafana Dashboards

Pre-built dashboards are available in [`monitoring/grafana/dashboards/`](https://github.com/anthropics/hindsight/tree/main/monitoring/grafana/dashboards). Import these JSON files into your Grafana instance:

| Dashboard | Description |
|-----------|-------------|
| **Hindsight Operations** | Operation rates, latency percentiles, per-bank metrics |
| **Hindsight LLM Metrics** | LLM calls, token usage, latency by scope/provider |
| **Hindsight API Service** | HTTP requests, error rates, DB pool, process metrics |

The dashboards are automatically provisioned when using the monitoring stack script.

## Metrics Endpoint

Hindsight exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8888/metrics
```

## Health Endpoints

The API server (port 8888) and every worker (port 8889) expose the same three
endpoints. They answer two different questions, and pointing a probe at the wrong
one is the difference between riding out a database blip and turning it into an
outage:

| Endpoint | Checks the database? | Use for |
|----------|----------------------|---------|
| `/health/live` | No | Liveness probes |
| `/health/ready` | Yes | Readiness probes |
| `/health` | Yes | Readiness — alias of `/health/ready`, kept for compatibility |

**`/health/live`** returns 200 whenever the process can serve a request, and does
no I/O to get there:

```json
{ "status": "alive", "version": "0.4.0", "uptime_seconds": 812.4 }
```

Answering at all is the check. Hindsight runs request handlers and task work on a
single event loop, so a loop wedged by a blocking call cannot respond inside the
probe timeout — which is exactly the failure a restart fixes. The worker's payload
adds `worker_id`, `is_shutdown`, and `seconds_since_last_poll` (the age of its last
completed claim cycle, `null` before the first one). That last field is there to
alert on: it never changes the status code, because a poller stalled behind a
saturated database is the case where restarting makes things worse.

**`/health` and `/health/ready`** acquire a pooled connection and run `SELECT 1`,
returning 200 when the database is reachable and 503 when it is not. The payload
separates the two ways that can go wrong — `db_acquire_ms` and `db_pool_waiting`
point at pool exhaustion, a slow query at the database itself:

```json
{ "status": "healthy", "database": "connected", "db_acquire_ms": 0.4, "db_pool_waiting": 0 }
```

:::warning Never point a liveness probe at a dependency check
A liveness failure means "restart this process." If your liveness probe checks the
database, then a slow database restarts every pod at once: in-flight requests are
dropped, claimed async operations are requeued with `retry_count` incremented
toward the permanent-failure cliff, and each restarted pod reconnects to re-warm
its pool against a database that is already struggling. Readiness failing is the
correct response — it takes the pod out of the Service and puts it back when the
database recovers.

The bundled Helm chart is wired this way already (`livenessProbe` → `/health/live`,
`readinessProbe` → `/health`). If you wrote your own manifests against an older
version, move the liveness path over.
:::

## Available Metrics

:::note High-cardinality labels are opt-in
`bank_id` and `tenant` (the tenant schema) are left off every metric by default, because each
one adds a series set per bank or tenant. Turn them on with
`HINDSIGHT_API_METRICS_INCLUDE_BANK_ID=true` / `HINDSIGHT_API_METRICS_INCLUDE_TENANT=true`
only on deployments with few banks or tenants. The backlog gauges always carry `tenant`.
:::

### Operation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.operation.duration` | Histogram | operation, bank_id, source, budget, max_tokens, success, memories_backend | Duration of operations in seconds |
| `hindsight.operation.total` | Counter | operation, bank_id, source, budget, max_tokens, success, memories_backend | Total number of operations executed |

**Labels:**
- `operation`: Operation type (`retain`, `recall`, `reflect`, plus async worker task types such as `consolidation`)
- `bank_id`: Memory bank identifier
- `source`: Where the operation was triggered from (`api`, `reflect`, `internal`, `worker`)
- `budget`: Budget level if specified (`low`, `mid`, `high`)
- `max_tokens`: Max tokens if specified
- `success`: Whether the operation succeeded (`true`, `false`)
- `memories_backend`: The store serving the bank, when the memories extension names one
  (`MemoriesExtension.backend_name_for`). Absent by default, so existing series are unchanged; a
  deployment whose banks live in different stores uses it to compare their latency without
  turning on `tenant`. Recall phase metrics recorded inside the operation carry it too.

The `source` label allows distinguishing between:
- `api`: Direct API calls from clients
- `reflect`: Internal recall calls made during reflect operations
- `internal`: Other internal operations
- `worker`: Async worker completions recorded when a claimed task reaches a terminal outcome

For `source="worker"`, the `success` label is a completion-throughput signal:
`false` means the task raised out to the poller after retries were exhausted or
an unexpected error occurred. Failures handled inside the executor and returned
normally still record `success="true"` here; use
`hindsight_async_operations{status="failed"}` for authoritative async operation
failure status.

### Retain Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.retain.documents.total` | Counter | outcome, bank_id | Documents processed by retain, by extraction outcome |

**Labels:**
- `outcome`: `facts` when the document has memory units after the retain, `no_facts` when it has none
- `bank_id`: Memory bank identifier

`outcome="no_facts"` is the signal that a document was stored but produced no memories. Those documents are invisible to `recall` and `reflect` until they are [reprocessed](/developer/retain#when-a-mission-excludes-everything-in-a-document) — nothing else reports them, because the retain itself succeeded. A rising share usually means a [retain mission](/developer/retain#steering-extraction-with-a-mission) is excluding more than intended:

```promql
sum(rate(hindsight_retain_documents_total{outcome="no_facts"}[15m]))
  / sum(rate(hindsight_retain_documents_total[15m]))
```

### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.llm.duration` | Histogram | provider, model, scope, success | Duration of LLM API calls in seconds |
| `hindsight.llm.calls.total` | Counter | provider, model, scope, success | Total number of LLM API calls |
| `hindsight.llm.tokens.input` | Counter | provider, model, scope, success, token_bucket | Input tokens for LLM calls |
| `hindsight.llm.tokens.output` | Counter | provider, model, scope, success, token_bucket | Output tokens from LLM calls |

**Labels:**
- `provider`: LLM provider (`openai`, `anthropic`, `gemini`, `groq`, `ollama`, `lmstudio`, `bedrock`, `litellm`)
- `model`: Model name (e.g., `gpt-4`, `claude-3-sonnet`)
- `scope`: What the LLM call is for (`memory`, `reflect`, `consolidation`, `answer`)
- `success`: Whether the call succeeded (`true`, `false`)
- `token_bucket`: Token count bucket for cardinality control (`0-100`, `100-500`, `500-1k`, `1k-5k`, `5k-10k`, `10k-50k`, `50k+`)

### Consolidation Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.consolidation.batch_failures` | Counter | failure_class, error_type | Consolidation LLM batch calls that failed, including those whose facts were later recovered |

**Labels:**
- `failure_class`: How the call was treated (`fail_fast` — the model returned something the response schema rejects, so a re-send of the same payload cannot help; `retry` — transport-shaped, an unchanged re-send may succeed; `propagate` — not a batch failure, re-raised to the task handler)
- `error_type`: Exception class name (e.g. `ValidationError`, `JSONDecodeError`)

This is not the same signal as the `failed_consolidation` field in bank stats. That
field counts *facts* still waiting to be consolidated after a failure; when a batch
call fails, consolidation halves the batch and retries, so a call that failed at
batch size 8 and succeeded at size 1 leaves `failed_consolidation` at 0. Everything
the failed response asked for is dropped, though — including any observations it
wanted to delete — so a bank can look completely healthy while its
supersession cleanup does nothing.

Alert on the schema-rejection rate, which no other metric exposes:

```promql
sum(rate(hindsight_consolidation_batch_failures_total{failure_class="fail_fast"}[15m]))
```

A sustained non-zero rate usually means the consolidation model does not reliably
emit valid JSON for the response schema. Switching to a model with stronger
structured output, or enabling `HINDSIGHT_API_LLM_STRICT_SCHEMA_CONSOLIDATION` if
the provider supports grammar-enforced schemas, is the usual fix.

The same count is reported per run as `llm_batch_failures` in the consolidation
operation's result, and the consolidation log summary prints a warning line
whenever it is non-zero. Both count *attempts* — one batch call retried three times
contributes three — so they are not bounded by the number of batches in the run.

### Adaptive Memory Metrics

Recorded only for banks whose adaptive memory mode is not `off` (see [Adaptive memory](/developer/configuration#adaptive-memory)); a server with the feature off exports none of them. Exported names are the OTel instrument name with dots turned into underscores and `_total` added to counters; histograms export `_bucket`, `_sum` and `_count`. The first column is the name the requirements use for each signal.

| Signal | Exported (Prometheus) | OTel instrument | Type | Labels | Recorded when |
|--------|-----------------------|-----------------|------|--------|---------------|
| `adaptive_memory_state_initialized_total` | `hindsight_adaptive_memory_state_initialized_total` | `hindsight.adaptive_memory.state_initialized` | Counter | - | A reinforcement or an explicit retention setting created a memory's retention row |
| `reinforcement_events_total` | `hindsight_adaptive_memory_reinforcement_events_total` | `hindsight.adaptive_memory.reinforcement_events` | Counter | source_type, outcome | A reinforcement event was applied, anchored a new row (`initial`) or fell in the cooldown |
| `reinforcement_events_deduplicated_total` | `hindsight_adaptive_memory_reinforcement_events_deduplicated_total` | `hindsight.adaptive_memory.reinforcement_events_deduplicated` | Counter | source_type | An event's key had already reinforced the memory; nothing changed |
| (operational) | `hindsight_adaptive_memory_reinforcement_dropped_total` | `hindsight.adaptive_memory.reinforcement_dropped` | Counter | producer, reason | A producer gave up on a reinforcement (`error`, or `overflow` of in-flight background writes) |
| `supersessions_total` | `hindsight_adaptive_memory_supersessions_total` | `hindsight.adaptive_memory.supersessions` | Counter | origin | A claim was superseded (its compare-and-set won) |
| `refutations_total` | `hindsight_adaptive_memory_refutations_total` | `hindsight.adaptive_memory.refutations` | Counter | origin | A claim was refuted |
| `uncertain_claims_total` | `hindsight_adaptive_memory_uncertain_claims_total` | `hindsight.adaptive_memory.uncertain_claims` | Counter | origin | A claim was marked uncertain |
| `splits_total` | `hindsight_adaptive_memory_splits_total` | `hindsight.adaptive_memory.splits` | Counter | origin | An observation was split into narrower ones |
| `merges_total` | `hindsight_adaptive_memory_merges_total` | `hindsight.adaptive_memory.merges` | Counter | origin | An observation was merged into a survivor |
| (operational) | `hindsight_adaptive_memory_lifecycle_transitions_dropped_total` | `hindsight.adaptive_memory.lifecycle_transitions_dropped` | Counter | reason | Consolidation dropped a lifecycle label that failed validation; `reason` says which check refused it |
| `recall_superseded_filtered_total` | `hindsight_adaptive_memory_recall_superseded_filtered_total` | `hindsight.adaptive_memory.recall_superseded_filtered` | Counter | view | A lifecycle-mode recall removed candidates because their claim is superseded |
| `recall_refuted_filtered_total` | `hindsight_adaptive_memory_recall_refuted_filtered_total` | `hindsight.adaptive_memory.recall_refuted_filtered` | Counter | view | A lifecycle-mode recall removed candidates because their claim is refuted |
| (operational) | `hindsight_adaptive_memory_historical_cue_fired_total` | `hindsight.adaptive_memory.historical_cue_fired` | Counter | cue | The historical-cue detector turned an external recall historical |
| `archive_suppressed_total` | `hindsight_adaptive_memory_archive_suppressed_total` | `hindsight.adaptive_memory.archive_suppressed` | Counter | - | `rank`/`lifecycle` with archiving on: a recall dropped archived memories after scoring (normalized reranker score below `ADAPTIVE_MEMORY_ARCHIVE_RECALL_MIN_SCORE`) |
| `archived_total` | `hindsight_adaptive_memory_archived_total` | `hindsight.adaptive_memory.archived` | Counter | - | The `adaptive_memory_maintenance` job archived memories |
| `pruned_total` | `hindsight_adaptive_memory_pruned_total` | `hindsight.adaptive_memory.pruned` | Counter | - | The `adaptive_memory_maintenance` job deleted long-archived memories (prune on) |
| `retrievability_distribution` | `hindsight_adaptive_memory_retrievability_bucket` | `hindsight.adaptive_memory.retrievability` | Histogram (0.05 to 1.0, step 0.05) | mode | Once per scored recall, one value per returned memory |
| `stability_days_distribution` | `hindsight_adaptive_memory_stability_days_bucket` | `hindsight.adaptive_memory.stability_days` | Histogram (log, 1 to 3650 days) | mode | Once per scored recall, one value per returned memory |
| `adaptive_memory_rerank_changes_total` | `hindsight_adaptive_memory_rerank_changes_total` | `hindsight.adaptive_memory.rerank_changes` | Counter | mode | `rank`/`lifecycle`: the returned ids differ from the factor-free order's |
| `top_k_changed_total` | `hindsight_adaptive_memory_top_k_changed_total` | `hindsight.adaptive_memory.top_k_changed` | Counter | mode | `observe`: the simulated factor-applied ids differ from the ones returned |
| (operational) | `hindsight_adaptive_memory_state_load_errors_total` | `hindsight.adaptive_memory.state_load_errors` | Counter | mode | A recall's retention state could not be loaded (it continued with factor 1) |
| (operational) | `hindsight_adaptive_memory_carry_forward_matched_total` | `hindsight.adaptive_memory.carry_forward_matched` | Counter | - | A re-ingested fact's lifecycle state moved to its re-extracted twin |
| (operational) | `hindsight_adaptive_memory_carry_forward_unmatched_total` | `hindsight.adaptive_memory.carry_forward_unmatched` | Counter | - | A re-ingested fact with lifecycle state had no re-extracted twin (its state was dropped) |
| (operational) | `hindsight_adaptive_memory_claims_reverted_total` | `hindsight.adaptive_memory.claims_reverted` | Counter | reason | A retired claim returned to active because all its evidence was deleted (`evidence_deleted`) or not re-extracted (`reingest_unmatched`) |

**Labels:**
- `mode`: The bank's effective adaptive memory mode (`observe`, `rank`, `lifecycle`)
- `origin`: Who decided the transition (`consolidation`, or `explicit` for the lifecycle PATCH API)
- `view`: The claim view the recall applied (`current`, `historical`, `as_of`)
- `cue`: The historical-cue pattern that matched (a fixed list, e.g. `used_to`, `before_change`, `before_date`)
- `source_type`, `outcome`, `producer`, `reason`: Bounded enums, listed in the descriptions above

A recall in `lifecycle` mode whose claim state cannot be read fails with HTTP 503 rather than return unchecked memories; those show up in the HTTP metrics below as `status_code="503"` on the recall endpoint.

### HTTP Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.http.duration` | Histogram | method, endpoint, status_code, status_class | Duration of HTTP requests in seconds |
| `hindsight.http.requests.total` | Counter | method, endpoint, status_code, status_class | Total number of HTTP requests |
| `hindsight.http.requests.in_progress` | UpDownCounter | method, endpoint | Number of HTTP requests currently being processed |

**Labels:**
- `method`: HTTP method (`GET`, `POST`, `PUT`, `DELETE`)
- `endpoint`: Template of the route the request matched, with path parameters named rather than filled in (e.g. `/v1/default/banks/{bank_id}/documents/{document_id}`, `/v1/default/banks/{bank_id}/operations/{operation_id}`), so the label has one value per route however many banks, documents or ids are requested. The template does not include a `base_path` / root-path prefix. Requests that match no route (404s) are labelled `unmatched`; a request matched by a route that has no path template is labelled `unknown_route`.
- `status_code`: HTTP status code (`200`, `400`, `500`, etc.)
- `status_class`: Status code class (`2xx`, `4xx`, `5xx`)

### Database Pool Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.db.pool.size` | Gauge | - | Current number of connections in the pool |
| `hindsight.db.pool.idle` | Gauge | - | Number of idle connections in the pool |
| `hindsight.db.pool.min` | Gauge | - | Minimum pool size |
| `hindsight.db.pool.max` | Gauge | - | Maximum pool size |

### Process Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `hindsight.process.cpu.seconds` | Gauge | type | Process CPU time in seconds |
| `hindsight.process.memory.bytes` | Gauge | type | Process memory usage in bytes |
| `hindsight.process.open_fds` | Gauge | - | Number of open file descriptors |
| `hindsight.process.threads` | Gauge | - | Number of active threads |

**Labels:**
- `type` (CPU): `user` or `system`
- `type` (Memory): `rss_max` (maximum resident set size)

### Histogram Buckets

Custom bucket boundaries are configured for better percentile accuracy:

**Operation Duration Buckets (seconds):**
```
0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0
```

**LLM Duration Buckets (seconds):**
```
0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0
```

**HTTP Duration Buckets (seconds):**
```
0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
```

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'hindsight'
    static_configs:
      - targets: ['localhost:8888']
```

## Example Queries

### Average operation latency by type
```promql
rate(hindsight_operation_duration_sum[5m]) / rate(hindsight_operation_duration_count[5m])
```

### LLM calls per minute by provider
```promql
rate(hindsight_llm_calls_total[1m]) * 60
```

### P95 LLM latency
```promql
histogram_quantile(0.95, rate(hindsight_llm_duration_bucket[5m]))
```

### Total tokens consumed by model
```promql
sum by (model) (hindsight_llm_tokens_input_total + hindsight_llm_tokens_output_total)
```

### Internal vs API recall operations
```promql
sum by (source) (rate(hindsight_operation_total{operation="recall"}[5m]))
```

### HTTP requests per second by endpoint
```promql
sum by (endpoint) (rate(hindsight_http_requests_total[1m]))
```

### HTTP error rate (5xx)
```promql
sum(rate(hindsight_http_requests_total{status_class="5xx"}[5m])) / sum(rate(hindsight_http_requests_total[5m]))
```

### P95 HTTP latency
```promql
histogram_quantile(0.95, sum by (le) (rate(hindsight_http_duration_seconds_bucket[5m])))
```

### Database pool utilization
```promql
hindsight_db_pool_size / hindsight_db_pool_max
```

### Active database connections
```promql
hindsight_db_pool_size - hindsight_db_pool_idle
```

### CPU usage rate
```promql
rate(hindsight_process_cpu_seconds{type="user"}[1m])
```

---

## Distributed Tracing

Hindsight supports OpenTelemetry distributed tracing for memory operations and LLM calls, following GenAI semantic conventions v1.37+.

### Configuration

See [Configuration - OpenTelemetry Tracing](./configuration#opentelemetry-tracing) for environment variables.

**Quick Start:**
```bash
# Enable tracing
export HINDSIGHT_API_OTEL_TRACES_ENABLED=true
export HINDSIGHT_API_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# View traces with Grafana LGTM (local dev)
./scripts/dev/start-monitoring.sh
# Open http://localhost:3000 → Explore → Tempo
```

Supports any OTLP-compatible backend (Grafana LGTM, Langfuse, OpenLIT, DataDog, New Relic, Honeycomb, [Pydantic Logfire](https://logfire.pydantic.dev), etc.).

### Span Hierarchy

**Parent Spans (Operations):**
- `hindsight.retain` - Memory ingestion
- `hindsight.recall` - Memory retrieval
  - `hindsight.recall_embedding` - Query embedding
  - `hindsight.recall_retrieval` - Parallel search (semantic, BM25, graph, temporal)
  - `hindsight.recall_fusion` - Reciprocal Rank Fusion
  - `hindsight.recall_rerank` - Cross-encoder reranking
- `hindsight.reflect` - Agentic reasoning
  - `hindsight.reflect_tool_call` - Tool execution (recall, lookup, etc.)
- `hindsight.consolidation` - Observation synthesis
- `hindsight.mental_model_refresh` - Mental model updates

**Child Spans (LLM Calls):**
- Named by scope (e.g., `hindsight.memory`, `hindsight.reflect`)
- Contain full prompts/completions as events
- Follow GenAI semantic conventions for attributes

### Span Attributes

**Operation Spans:**
- `hindsight.operation` - Operation type
- `hindsight.bank_id` - Memory bank ID
- `hindsight.query` - Query text (truncated to 100 chars)
- `hindsight.fact_types` - Fact types for recall
- `hindsight.thinking_budget` - Budget allocation
- `hindsight.max_tokens` - Token limit

**LLM Spans (GenAI Semantic Conventions):**
- `gen_ai.operation.name` - Always `"chat"`
- `gen_ai.provider.name` - Provider (`openai`, `anthropic`, `google`, etc.)
- `gen_ai.request.model` - Model name
- `gen_ai.usage.input_tokens` - Input tokens
- `gen_ai.usage.output_tokens` - Output tokens
- `hindsight.scope` - LLM call purpose (`memory`, `reflect`, `consolidation`, etc.)

**Events:**
- `gen_ai.client.inference.operation.details` - Full prompts and completions

### Trace Context Propagation

Hindsight participates in your existing traces rather than starting parallel ones. When a caller sends W3C trace context (the standard `traceparent` header, which most OpenTelemetry HTTP client instrumentations add automatically), Hindsight continues that trace: the API request appears as a server span under the caller, and every memory operation and LLM call it triggers nests beneath it.

Requests that arrive without trace context still start their own trace, so nothing changes for un-instrumented callers.

Health and metrics endpoints are excluded from tracing so probe traffic doesn't drown out real work. Set `OTEL_PYTHON_FASTAPI_EXCLUDED_URLS` to a comma-separated list of URL patterns to override this.

### Worker Processes

Standalone worker processes honour the same `HINDSIGHT_API_OTEL_*` variables as the API and export their own spans. This matters for deployments that run dedicated workers, since consolidation, background retain and mental model refresh — most of the long-running work and token spend — happen there.

Give the worker its own `HINDSIGHT_API_OTEL_SERVICE_NAME` to separate it from the API in your tracing backend. When left unset, workers report themselves as `hindsight-worker`.

Worker spans are currently their own traces: a background operation is not linked to the request that queued it, because it runs long after that request has returned.
