"""The harm benchmark's bank and its 24 coding tasks.

Projects: ``o`` = orders-service (Java 21 / Spring Boot, Aurora MySQL, Kafka), ``p`` =
payment-service (Go, PostgreSQL, Argo CD), ``d`` = insight-dashboard (Next.js / React 19 with a
Fastify BFF). ``g.*`` memories are the user's own and belong to no project.

One shared bank holds every memory. A ``*.trap.*`` memory is a plausible episode, superseded
value, joke or other project's procedure that shares a task's vocabulary but is wrong for it; a
``*.use.*`` memory is the project-specific fact a ``U`` task cannot be answered without. Every
other memory is realistic noise — facts, decisions, procedures, preferences, history, debugging
episodes, casual talk, humor, speculation and failed attempts on topics the tasks do not ask
about, some of them sharing a task's technology. One memory can be both: ``p.use.deploy`` answers
U01 and is the trap of H04, the way one real memory helps one task and misleads another.
"""

from __future__ import annotations

from ..context_bench.model import Category, stable_hash
from .model import KINDS, Memory, Task


def _project_of(label: str) -> str | None:
    head = label.split(".", 1)[0]
    return head if head in ("o", "p", "d") else None


def m(
    label: str,
    category: Category,
    text: str,
    *,
    module: str | None = None,
    age: int | None = None,
    status: str = "active",
    env: str | None = None,
) -> Memory:
    return Memory(
        label=label,
        category=category,
        text=text,
        project=_project_of(label),
        module=module,
        age_days=age if age is not None else 15 + stable_hash(f"harmbench:{label}") % 300,
        status=status,  # type: ignore[arg-type]
        environment=env,
    )


# ---------------------------------------------------------------------------
# orders-service
# ---------------------------------------------------------------------------

ORDERS: tuple[Memory, ...] = (
    # Traps
    m(
        "o.trap.jackson",
        "debugging_episode",
        "OrdersController.getOrder started returning 500 instead of 200 in the JUnit suite after the Spring Boot 3.2 upgrade; the cause was Jackson failing to serialize the LocalDateTime createdAt field, and adding jackson-datatype-jsr310 with JavaTimeModule fixed it.",
        module="api",
        age=200,
    ),
    m(
        "o.trap.vpn",
        "debugging_episode",
        "The nightly migration job against the orders-service MySQL kept failing with a timeout last March; the site-to-site VPN to the RDS VPC was dropping idle connections after 30 minutes, and enabling TCP keepalive on the bastion tunnel fixed it.",
        module="ops",
        age=220,
    ),
    m(
        "o.trap.pool20",
        "durable_fact",
        "orders-service in prod uses a HikariCP maximum-pool-size of 20 per pod (spring.datasource.hikari.maximum-pool-size in application-prod.yml), sized after the RDS connection storm.",
        module="api",
        age=400,
        status="superseded",
        env="prod",
    ),
    m(
        "o.trap.rebalance",
        "debugging_episode",
        "Consumer lag on the orders-service billing module's invoice consumer last year was caused by rebalance storms: max.poll.interval.ms was too low for slow invoice rendering, so we raised it to 10 minutes and the lag went away.",
        module="billing",
        age=180,
    ),
    m(
        "o.trap.configmap",
        "debugging_episode",
        "In July a new HTTP client timeout we set for orders-service through an environment variable in its Kubernetes Deployment/ConfigMap did not take effect: the pods simply had not been restarted, and kubectl rollout restart deployment/orders-service picked up the new env value.",
        module="ops",
        age=150,
    ),
    # Useful
    m(
        "o.use.flyway",
        "procedure",
        "orders-service schema migrations use Flyway in the migrations module (src/main/resources/db/migration). Name new files V<yyyyMMddHHmm>__<JIRA-key>_<snake_description>.sql — timestamp versions instead of sequential numbers so parallel branches never collide; outOfOrder is enabled in every environment.",
        module="migrations",
        age=120,
    ),
    m(
        "o.use.maxpoll",
        "durable_fact",
        "orders-service's ingest consumer (group orders-ingest) runs with max.poll.records=50 instead of Kafka's default 500, because every record makes a synchronous fraud-check call of about 200 ms and full batches blew past max.poll.interval.ms.",
        module="ingest",
        age=90,
    ),
    # Noise
    m(
        "o.arch.hexagonal",
        "architectural_decision",
        "orders-service follows a ports-and-adapters layout: controllers in api, domain logic in core, JPA repositories only in the persistence adapter; controllers must not inject repositories directly.",
        module="api",
    ),
    m(
        "o.db.aurora",
        "durable_fact",
        "orders-service's production database is Aurora MySQL 8.0 (cluster orders-prod) with one writer and two readers; read-only queries go to the reader endpoint through the @ReadOnlyRoute annotation.",
        env="prod",
    ),
    m(
        "o.db.staging",
        "durable_fact",
        "orders-service staging runs on a single db.r6g.large Aurora MySQL instance restored every Sunday from an anonymised prod snapshot.",
        env="staging",
    ),
    m(
        "o.ci.gradle",
        "durable_fact",
        "orders-service builds with Gradle 8 and the Kotlin DSL; CI runs ./gradlew check on GitHub Actions with the Gradle build cache in S3.",
    ),
    m(
        "o.ci.testcontainers",
        "procedure",
        "orders-service integration tests use Testcontainers with MySQL 8.0.36 and Redpanda; on Apple Silicon run them under Colima with at least 6 GB of memory.",
    ),
    m(
        "o.api.versioning",
        "architectural_decision",
        "orders-service public endpoints are versioned in the path (/v1, /v2); /v1 is frozen and only gets security fixes until its sunset on 2027-03-31.",
        module="api",
    ),
    m(
        "o.api.pagination",
        "architectural_decision",
        "orders-service list endpoints use cursor pagination with an opaque base64 cursor and a maximum page size of 200; offset pagination was removed from /v2.",
        module="api",
    ),
    m(
        "o.checkout.saga",
        "architectural_decision",
        "Checkout in orders-service is an orchestrated saga: reserve inventory, authorize the payment through payment-service, then confirm; each step's compensating action is recorded in the checkout_saga_steps table.",
        module="checkout",
    ),
    m(
        "o.checkout.idem",
        "durable_fact",
        "POST /v2/checkout requires an Idempotency-Key header; keys are kept in Redis for 24 hours and a replay returns the original response with Idempotent-Replayed: true.",
        module="checkout",
    ),
    m(
        "o.billing.invoice",
        "durable_fact",
        "The orders-service billing module renders invoices as PDF with OpenPDF on a separate pool of 8 workers and stores them in S3 under invoices/<yyyy>/<mm>/<invoiceId>.pdf.",
        module="billing",
    ),
    m(
        "o.billing.tax",
        "durable_fact",
        "orders-service computes sales tax through the Avalara client in the billing module, on the order total after discounts, rounded half-even to two decimals.",
        module="billing",
    ),
    m(
        "o.ingest.topics",
        "durable_fact",
        "The orders-service ingest module consumes marketplace orders from marketplace.orders.v3 and publishes accepted orders to orders.accepted.v1 (24 partitions, 7-day retention).",
        module="ingest",
    ),
    m(
        "o.ingest.dlq",
        "procedure",
        "Marketplace orders that fail ingest validation three times go to marketplace.orders.v3.dlq; replay them with the dlq-replayer job after fixing the cause, never by re-producing them by hand.",
        module="ingest",
    ),
    m(
        "o.oncall",
        "procedure",
        "orders-service on-call rotates weekly on Mondays at 10:00 São Paulo time; the handover note goes in #orders-oncall with open incidents and silenced alerts.",
        module="ops",
    ),
    m(
        "o.alerts",
        "durable_fact",
        "The orders-service p99 latency alert fires when /v2/orders p99 exceeds 800 ms for 10 minutes; it pages between 07:00 and 23:00 and otherwise opens a ticket.",
        module="ops",
        env="prod",
    ),
    m(
        "o.deploy",
        "procedure",
        "orders-service deploys through Spinnaker: a merge to main builds the image and deploys staging automatically; prod needs a manual judgment stage followed by a 25% canary for 20 minutes.",
        module="ops",
        env="prod",
    ),
    m(
        "o.pref.assertj",
        "preference",
        "In orders-service tests the team prefers AssertJ's assertThat over Hamcrest matchers and JUnit's assertEquals.",
    ),
    m(
        "o.pref.records",
        "preference",
        "orders-service code review prefers Java records for DTOs and request/response bodies; Lombok stays only in the older billing classes that still use it.",
        module="api",
    ),
    m(
        "o.cache",
        "durable_fact",
        "Product lookups in orders-service are cached in Caffeine for 5 minutes with at most 50,000 entries; the cache is per pod, not shared.",
        module="api",
    ),
    m(
        "o.ratelimit",
        "durable_fact",
        "orders-service limits each partner API key to 600 requests per minute with a Bucket4j filter backed by Redis.",
        module="api",
        env="prod",
    ),
    m(
        "o.flaky.clock",
        "debugging_episode",
        "A flaky CheckoutServiceTest in orders-service failed around midnight UTC because it compared LocalDate.now() with a date computed in America/Sao_Paulo; we injected a fixed Clock bean in tests.",
        module="checkout",
        age=300,
    ),
    m(
        "o.incident.deadlock",
        "debugging_episode",
        "INC-204: checkout deadlocks in orders-service came from two transactions updating inventory_reservations and orders in opposite order; we fixed it by always locking the orders row first.",
        module="checkout",
        age=260,
    ),
    m(
        "o.casual.launch",
        "casual_context",
        "The orders team celebrated the checkout 2.0 launch with pastel and caldo de cana in the office on a Friday.",
    ),
    m(
        "o.humor.saga",
        "humor",
        "Team joke: the checkout saga has more compensating actions than a telenovela has plot twists.",
        module="checkout",
    ),
    m(
        "o.humor.points",
        "humor",
        "Running joke in #orders-dev: every Jira ticket estimated at one point secretly takes a whole sprint.",
    ),
    m(
        "o.spec.vthreads",
        "speculation",
        "Maybe orders-service could drop its custom thread pools by switching to virtual threads, but nobody has benchmarked that with the MySQL driver yet.",
        module="api",
    ),
    m(
        "o.failed.graphql",
        "failed_attempt",
        "We tried exposing orders through a GraphQL gateway in 2025 but dropped it: partner SDKs only spoke REST and N+1 queries hammered the reader instance.",
        module="api",
        age=350,
    ),
    m(
        "o.timezone",
        "durable_fact",
        "All orders-service timestamps are stored in UTC as DATETIME(6); conversion to the merchant's timezone happens only in the API layer.",
    ),
    m(
        "o.money",
        "architectural_decision",
        "orders-service represents money as BigDecimal with scale 2 inside a Money record with an ISO-4217 currency; an ArchUnit rule bans double for amounts.",
    ),
    m(
        "o.audit",
        "durable_fact",
        "Every order state change is appended to the order_events table and published to orders.events.v1 for the audit pipeline.",
    ),
    m(
        "o.secrets",
        "procedure",
        "orders-service secrets live in AWS Secrets Manager under /orders/<env>/ and are read at startup by Spring Cloud AWS.",
        module="ops",
    ),
    m(
        "o.logs",
        "durable_fact",
        "orders-service logs JSON to stdout with traceId and orderId in the MDC; logs ship to Datadog and are kept for 15 days.",
        module="ops",
    ),
    m(
        "o.toggles",
        "durable_fact",
        "orders-service feature toggles live in its own toggles table, read through FlagService with a 30-second cache and edited in the admin console, not in config files.",
    ),
    m(
        "o.java",
        "durable_fact",
        "orders-service runs on Java 21 (Temurin) with Spring Boot 3.3; the move to Java 25 is planned for Q1 2027.",
    ),
    m(
        "o.isolation",
        "durable_fact",
        "orders-service uses MySQL's default REPEATABLE READ isolation, except reporting queries, which run at READ COMMITTED on the reader endpoint.",
    ),
    m(
        "o.perf.n1",
        "debugging_episode",
        "The /v2/orders list endpoint slowed down in May because of an N+1 on order_items; an entity graph on the repository method fixed it.",
        module="api",
        age=140,
    ),
    m(
        "o.kafka",
        "durable_fact",
        "orders-service uses Spring Kafka 3.2 against the shared MSK cluster msk-core (Kafka 3.6) with SASL/IAM authentication.",
    ),
    m(
        "o.retention",
        "durable_fact",
        "Cancelled orders are anonymised after 18 months by the orders-service retention job, which nulls customer PII but keeps amounts for finance.",
    ),
    m(
        "o.ownership",
        "durable_fact",
        "orders-service is owned by the Commerce Core team; Rafa and Bia are the code owners of the checkout module.",
    ),
    m(
        "o.staging.vpn",
        "procedure",
        "Reaching orders-service staging internals (actuator, Kafka UI) needs the corporate WireGuard profile acme-staging.",
        module="ops",
        env="staging",
    ),
    m(
        "o.search",
        "durable_fact",
        "Order search is served by the OpenSearch index orders-v7, fed from orders.events.v1 by the search-indexer; it lags writes by about 5 seconds.",
        module="api",
    ),
    m(
        "o.returns",
        "durable_fact",
        "Returns are their own aggregate in orders-service: a return references order lines and asks payment-service for a refund once the warehouse confirms receipt.",
        module="checkout",
    ),
    m(
        "o.pref.naming",
        "preference",
        "orders-service reviewers ask for intention-revealing method names instead of comments, and test names in given_when_then style.",
    ),
    m(
        "o.debug.utf8",
        "debugging_episode",
        "Marketplace orders with emoji in the customer name failed ingest in April because the column was utf8mb3; converting the table to utf8mb4 fixed it.",
        module="ingest",
        age=230,
    ),
    m(
        "o.casual.coffee",
        "casual_context",
        "The orders team keeps a French press by the window and takes turns buying the beans.",
    ),
)

# ---------------------------------------------------------------------------
# payment-service
# ---------------------------------------------------------------------------

PAYMENTS: tuple[Memory, ...] = (
    # Traps
    m(
        "p.trap.joke",
        "humor",
        "Team joke: TestCaptureRetries only flakes when Mercury is in retrograde, and the official fix is to hit re-run until CI goes green.",
        module="capture",
        age=60,
    ),
    m(
        "p.trap.dockerhub",
        "debugging_episode",
        "TestCaptureRetries failed again and again in CI in May because the Redis test container image was rate-limited by Docker Hub; mirroring redis:7 into our ECR fixed the flakes.",
        module="capture",
        age=130,
    ),
    # Useful (p.use.deploy is also H04's trap)
    m(
        "p.use.deploy",
        "procedure",
        "To deploy payment-service to prod: tag a release vX.Y.Z on main, wait for the release workflow to push the image to ECR, then run `make promote ENV=prod TAG=vX.Y.Z` in acme/payment-infra. That opens a PR bumping values-prod.yaml; it needs an approval from #payments-oncall, and Argo CD syncs prod after the merge.",
        module="deploy",
        age=45,
        env="prod",
    ),
    m(
        "p.use.pool",
        "durable_fact",
        "payment-service in prod runs pgxpool with MaxConns=24 per pod, set through PAYMENT_DB_MAX_CONNS in payment-infra's values-prod.yaml; with 8 pods that is 192 of the RDS instance's 400 connections.",
        module="config",
        age=70,
        env="prod",
    ),
    m(
        "p.use.retry",
        "architectural_decision",
        "Decision: every outbound HTTP call in payment-service goes through internal/httpx.NewClient with the retry.Outbound preset (3 attempts, 200 ms base backoff, full jitter, retries only on 429, 5xx and connection errors); hand-written retry loops are rejected in review.",
        module="api",
        age=100,
    ),
    m(
        "p.use.webhook",
        "procedure",
        "To rotate a merchant's webhook signing secret in payment-service, run `paycli secrets rotate-webhook --merchant <id> --overlap 72h`. During the 72-hour overlap every webhook carries both signatures in the Acme-Signature header (v1 = new secret, v0 = old), so the merchant can switch without failed verifications.",
        module="webhooks",
        age=80,
    ),
    # Noise
    m(
        "p.pool.staging",
        "durable_fact",
        "payment-service staging uses PAYMENT_DB_MAX_CONNS=8 per pod, set in payment-infra's values-staging.yaml.",
        module="config",
        env="staging",
    ),
    m(
        "p.deploy.staging",
        "procedure",
        "payment-service staging deploys itself: every merge to main builds an image and Argo CD syncs the staging app within five minutes.",
        module="deploy",
        env="staging",
    ),
    m(
        "p.deploy.rollback",
        "procedure",
        "To roll back payment-service in prod, revert the values-prod.yaml bump in payment-infra and let Argo CD sync; never change the image with kubectl.",
        module="deploy",
        env="prod",
    ),
    m(
        "p.deploy.freeze",
        "procedure",
        "payment-service has a deploy freeze from December 20 to January 5 and during Black Friday week; urgent fixes need the Payments lead's approval.",
        module="deploy",
        env="prod",
    ),
    m(
        "p.ledger",
        "architectural_decision",
        "payment-service keeps a double-entry ledger in PostgreSQL: every capture and refund writes balanced debit and credit rows in one transaction.",
    ),
    m(
        "p.grpc",
        "durable_fact",
        "Internal callers reach payment-service over gRPC on port 9090 with mesh mTLS; the public HTTP API on 8080 is reachable only through the API gateway.",
        module="api",
    ),
    m(
        "p.providers",
        "durable_fact",
        "payment-service sends card captures to Adyen; PIX payments go through the Banco Inter API with a separate client in internal/pix.",
        module="capture",
    ),
    m(
        "p.capture.timeout",
        "durable_fact",
        "The Adyen capture call has a 10-second client timeout and is retried by the capture worker, never inline in the request.",
        module="capture",
    ),
    m(
        "p.refunds.window",
        "durable_fact",
        "Refunds are allowed up to 180 days after capture; partial refunds are summed against the captured amount under a row lock on the payment.",
        module="refunds",
    ),
    m(
        "p.refunds.async",
        "architectural_decision",
        "payment-service accepts refund requests with 202 and the refunds worker processes the refund_jobs table with SELECT ... FOR UPDATE SKIP LOCKED.",
        module="refunds",
    ),
    m(
        "p.webhooks.delivery",
        "durable_fact",
        "Outgoing merchant webhooks retry with exponential backoff for up to 24 hours and are dropped after 15 failed attempts; attempts are stored in webhook_deliveries.",
        module="webhooks",
    ),
    m(
        "p.webhooks.signing",
        "durable_fact",
        "payment-service signs merchant webhooks with HMAC-SHA256 over the timestamp and the raw body; merchants must reject signatures older than 5 minutes.",
        module="webhooks",
    ),
    m(
        "p.redis",
        "durable_fact",
        "payment-service uses Redis (ElastiCache, cluster mode off) only for idempotency keys and rate limiting; nothing in Redis is a source of truth.",
    ),
    m(
        "p.pg",
        "durable_fact",
        "payment-service's prod database is RDS PostgreSQL 15 (payments-prod, db.r6g.2xlarge) with a cross-region read replica in us-west-2 for disaster recovery.",
        env="prod",
    ),
    m(
        "p.migrations",
        "procedure",
        "payment-service migrations use golang-migrate with sequentially numbered files in db/migrations; CI runs them against a throwaway PostgreSQL container before merge.",
    ),
    m(
        "p.ci",
        "durable_fact",
        "payment-service CI runs go vet, staticcheck, golangci-lint and go test -race ./... on GitHub Actions; a PR needs the race tests green to merge.",
    ),
    m(
        "p.pref.errors",
        "preference",
        'payment-service reviewers want errors wrapped with fmt.Errorf("...: %w", err) and compared with errors.Is; pkg/errors is no longer used.',
    ),
    m(
        "p.pref.tables",
        "preference",
        "The payments team prefers table-driven tests with t.Run subtests and testify's require for fatal checks.",
    ),
    m(
        "p.oncall",
        "procedure",
        "payment-service on-call uses the PagerDuty schedule PAY-PRIMARY; in an incident open a channel with /incident in Slack and page the secondary after 15 minutes without progress.",
    ),
    m(
        "p.slo",
        "durable_fact",
        "payment-service's SLO is 99.95% successful authorizations over 28 days, measured from gateway logs; an exhausted error budget freezes non-urgent deploys.",
        env="prod",
    ),
    m(
        "p.observability",
        "durable_fact",
        "payment-service exports OpenTelemetry traces to Grafana Tempo and metrics to Prometheus; its dashboards live in the Payments folder in Grafana.",
    ),
    m(
        "p.pci",
        "architectural_decision",
        "Card data never reaches payment-service: the frontend tokenizes cards with Adyen's drop-in, and we store only the token and the last four digits, which keeps us at PCI SAQ A.",
    ),
    m(
        "p.fx",
        "durable_fact",
        "Currency conversion in payment-service uses the daily ECB reference rates cached in the fx_rates table; the rates job runs at 16:30 CET.",
    ),
    m(
        "p.idem",
        "durable_fact",
        "payment-service idempotency keys are scoped per merchant and kept for 48 hours in Redis; reusing a key with a different body returns 422.",
        module="api",
    ),
    m(
        "p.incident.conns",
        "debugging_episode",
        "INC-188: payment-service authorizations failed for 12 minutes when a deploy doubled the pod count and the pools exhausted RDS max_connections; CI now checks that pods × MaxConns stays below 80% of max_connections.",
        module="config",
        age=240,
    ),
    m(
        "p.humor.iferr",
        "humor",
        "Running joke in #payments-dev: `if err != nil` is the most common line in the repo and deserves its own code owner.",
    ),
    m(
        "p.casual.offsite",
        "casual_context",
        "The payments team offsite in Florianópolis is booked for the second week of November.",
    ),
    m(
        "p.spec.tigerbeetle",
        "speculation",
        "Someone suggested moving the payment-service ledger to TigerBeetle, but there is no plan or owner for it.",
    ),
    m(
        "p.failed.syncrefunds",
        "failed_attempt",
        "We tried processing refunds synchronously inside the API request in 2025, but Adyen latency spikes caused gateway timeouts, so refunds moved to the async worker.",
        module="refunds",
        age=330,
    ),
    m(
        "p.config.files",
        "durable_fact",
        "payment-service reads config/<env>.yaml baked into the image plus environment variables injected from the Helm values; secrets come from External Secrets.",
        module="config",
    ),
    m(
        "p.secrets",
        "procedure",
        "payment-service secrets come from AWS Secrets Manager through External Secrets Operator; a rotated secret lands within an hour and the pods reload it through a file watcher.",
        module="config",
    ),
    m(
        "p.ratelimit",
        "durable_fact",
        "payment-service limits each merchant to 100 authorization requests per second with a Redis token bucket; excess requests get 429 with Retry-After.",
        module="api",
    ),
    m(
        "p.chargebacks",
        "procedure",
        "Adyen chargeback notifications arrive on /notifications/adyen, are verified with HMAC, and open a dispute case for the risk team in the disputes table.",
        module="refunds",
    ),
    m(
        "p.settlement",
        "durable_fact",
        "payment-service settlement reports are cut at 00:00 America/Sao_Paulo, not UTC, because the acquirer settles on Brazilian business days.",
    ),
    m(
        "p.ownership",
        "durable_fact",
        "payment-service is owned by the Payments team; Lucas is the tech lead and approves every change to the ledger package.",
    ),
    m(
        "p.loadtest",
        "procedure",
        "payment-service load tests run with k6 from the loadtest/ directory against staging and the Adyen test endpoint; never point k6 at prod.",
        env="staging",
    ),
    m(
        "p.debug.ctx",
        "debugging_episode",
        "payment-service capture workers stalled in August because a context.Background() had no deadline and a hung Adyen connection blocked a worker forever; all outbound calls now take the job context with a timeout.",
        module="capture",
        age=210,
    ),
    m(
        "p.casual.coffee",
        "casual_context",
        "Lucas brought a V60 to the office and now the payments floor argues about pour-over ratios every morning.",
    ),
    m(
        "p.humor.adyen",
        "humor",
        "Joke from the payments retro: Adyen's sandbox has better weekend uptime than we do.",
    ),
    m(
        "p.pref.prsize",
        "preference",
        "The payments team prefers PRs under 400 changed lines; anything larger needs a short design note in the description.",
    ),
    m(
        "p.pix",
        "durable_fact",
        "PIX charges expire after 30 minutes; the pix-reconciler job polls Banco Inter every minute for paid charges and marks them captured.",
        module="capture",
    ),
    m(
        "p.debug.dns",
        "debugging_episode",
        "Authorizations to Adyen failed for 5 minutes in March because CoreDNS pods were CPU-throttled; we raised their limit and enabled NodeLocal DNSCache.",
        age=270,
    ),
    m(
        "p.api.versioning",
        "architectural_decision",
        "The payment-service HTTP API is versioned by date in the Acme-Version header (for example 2026-04-01); without it the merchant's pinned version applies.",
        module="api",
    ),
    m(
        "p.casual.band",
        "casual_context",
        "Lucas plays guitar in a samba-rock band that gigs in Vila Madalena on Saturdays.",
    ),
    m(
        "p.spec.grpcweb",
        "speculation",
        "There was a vague idea to expose payment-service to the dashboard over gRPC-Web, but nothing was decided.",
    ),
)

# ---------------------------------------------------------------------------
# insight-dashboard
# ---------------------------------------------------------------------------

DASHBOARD: tuple[Memory, ...] = (
    # Trap
    m(
        "d.trap.grammarly",
        "debugging_episode",
        "The 'Hydration failed because the server rendered text didn't match the client' errors on insight-dashboard in January were caused by the Grammarly browser extension injecting attributes into the DOM; we told affected users to disable it on the dashboard.",
        module="web",
        age=250,
    ),
    # Useful
    m(
        "d.use.snapshot",
        "procedure",
        "To debug insight-dashboard charts with prod-like data locally, run `pnpm seed:snapshot --from=nightly`; it loads the anonymised nightly snapshot from s3://acme-dash-snapshots into the local Postgres. Pointing a local BFF at the prod read replica is blocked by policy since INC-311.",
        module="bff",
        age=60,
    ),
    m(
        "d.use.flags",
        "architectural_decision",
        "insight-dashboard feature flags live in Unleash (unleash.acme.internal, project dashboard). Name them dash.<area>.<feature> in kebab-case, put an expiry date in the description, and evaluate them only in the BFF, never in the browser bundle.",
        module="bff",
        age=110,
    ),
    # Noise
    m(
        "d.stack",
        "durable_fact",
        "insight-dashboard is a Next.js 15 App Router app on React 19 with server components; its data comes from a Fastify BFF in the bff/ package of the pnpm workspace.",
    ),
    m(
        "d.deploy.preview",
        "procedure",
        "Every insight-dashboard PR gets a preview environment at pr-<number>.dash.preview.acme.dev, torn down when the PR closes.",
        module="deploy",
    ),
    m(
        "d.charts",
        "architectural_decision",
        "insight-dashboard charts use Apache ECharts through a thin wrapper in packages/charts; Recharts was removed in 2025 because it struggled with 50k-point series.",
        module="charts",
    ),
    m(
        "d.state",
        "architectural_decision",
        "Client state in insight-dashboard uses Zustand and server data goes through TanStack Query with a 30-second staleTime; Redux is not allowed in new code.",
        module="web",
    ),
    m(
        "d.auth",
        "durable_fact",
        "insight-dashboard signs users in with Okta OIDC in the BFF; the browser holds only an httpOnly session cookie and never sees the access token.",
        module="bff",
    ),
    m(
        "d.bff.cache",
        "durable_fact",
        "The insight-dashboard BFF caches merchant summaries in Redis for 60 seconds, keyed by merchantId and date range.",
        module="bff",
    ),
    m(
        "d.i18n",
        "durable_fact",
        "insight-dashboard ships en-US, pt-BR and es-MX through next-intl; messages/<locale>.json files are synced from Crowdin nightly.",
        module="web",
    ),
    m(
        "d.tests",
        "durable_fact",
        "insight-dashboard unit tests run on Vitest and React Testing Library; end-to-end tests use Playwright against the preview environment.",
    ),
    m(
        "d.pref.server",
        "preference",
        "insight-dashboard reviewers prefer server components by default and 'use client' only on leaf components that need state or effects.",
        module="web",
    ),
    m(
        "d.pref.css",
        "preference",
        "insight-dashboard styling uses Tailwind with the @acme/tokens design tokens; no CSS-in-JS in new components.",
        module="web",
    ),
    m(
        "d.perf.bundle",
        "debugging_episode",
        "The dashboard's first load jumped to 900 kB in June because a date picker pulled in moment.js with every locale; replacing it with date-fns saved 300 kB.",
        module="web",
        age=190,
    ),
    m(
        "d.incident.311",
        "historical_event",
        "INC-311: a developer's local insight-dashboard BFF pointed at the prod read replica ran an unbounded export query and slowed merchant reports for 20 minutes.",
        module="bff",
        age=300,
    ),
    m(
        "d.a11y",
        "architectural_decision",
        "insight-dashboard targets WCAG 2.2 AA: every chart needs a data-table fallback, and axe checks run in Playwright.",
        module="charts",
    ),
    m(
        "d.analytics",
        "durable_fact",
        "insight-dashboard product analytics go to a self-hosted PostHog; events are named <area>_<action> in snake_case.",
    ),
    m(
        "d.humor.center",
        "humor",
        "Team joke: centering a div in the dashboard needs a design review and two approvals.",
        module="web",
    ),
    m(
        "d.humor.ie",
        "humor",
        "Someone asked whether insight-dashboard supports Internet Explorer and the frontend channel answered with 41 skull emojis.",
    ),
    m(
        "d.casual.lunch",
        "casual_context",
        "The dashboard team has a Thursday lunch at the Japanese place near the office.",
    ),
    m(
        "d.spec.htmx",
        "speculation",
        "Bea wondered whether the dashboard's admin pages could be rewritten with htmx; it was only a hallway idea.",
    ),
    m(
        "d.failed.ssg",
        "failed_attempt",
        "We tried statically generating merchant report pages at build time, but builds took 40 minutes for 20k merchants, so reports now render on demand with caching.",
        module="web",
        age=320,
    ),
    m(
        "d.ownership",
        "durable_fact",
        "insight-dashboard is owned by the Merchant Insights team; Bea is the frontend lead and Caio owns the BFF.",
    ),
    m(
        "d.node",
        "durable_fact",
        "insight-dashboard runs on Node 22 LTS; the BFF and the Next.js server share one Docker base image.",
    ),
    m(
        "d.contracts",
        "architectural_decision",
        "The insight-dashboard BFF exposes typed routes generated from zod schemas in packages/contracts; the web app imports the inferred types instead of hand-written interfaces.",
        module="bff",
    ),
    m(
        "d.sentry",
        "durable_fact",
        "insight-dashboard frontend errors go to Sentry project insight-dashboard-web, with source maps uploaded by CI.",
        module="web",
    ),
    m(
        "d.csp",
        "durable_fact",
        "insight-dashboard sends a strict Content-Security-Policy with per-request nonces from middleware; inline scripts without the nonce are blocked.",
        module="web",
    ),
    m(
        "d.debug.oom",
        "debugging_episode",
        "The insight-dashboard BFF grew until it was OOM-killed in August because a Map used as a cache had no eviction; lru-cache capped at 5,000 entries replaced it.",
        module="bff",
        age=170,
    ),
    m(
        "d.export",
        "durable_fact",
        "insight-dashboard CSV exports are built asynchronously by the BFF and emailed as a signed S3 link valid for 24 hours.",
        module="bff",
    ),
    m(
        "d.ci",
        "durable_fact",
        "insight-dashboard CI on GitHub Actions runs pnpm lint, pnpm typecheck, Vitest and Playwright smoke tests with turborepo remote caching.",
    ),
    m(
        "d.pnpm",
        "procedure",
        "insight-dashboard uses pnpm 9 through corepack; a pre-commit hook rejects npm and yarn lockfiles.",
    ),
    m(
        "d.flags.old",
        "historical_event",
        "insight-dashboard used LaunchDarkly until 2025; when the contract was not renewed the flags moved to the in-house Unleash.",
        module="bff",
        age=400,
    ),
    m(
        "d.figma",
        "durable_fact",
        "insight-dashboard designs live in the Figma file 'Insights 2026', and component names must match the @acme/ui library.",
        module="web",
    ),
    m(
        "d.react19",
        "historical_event",
        "insight-dashboard moved to React 19 and Next.js 15 in May; the same PR enabled the React Compiler for the web package.",
        module="web",
        age=140,
    ),
    m(
        "d.casual.music",
        "casual_context",
        "Caio's Brazilian jazz focus playlist has become the dashboard team's unofficial soundtrack.",
    ),
    m(
        "d.maps",
        "durable_fact",
        "The merchant map on insight-dashboard uses MapLibre with tiles from our self-hosted tile server; Google Maps was dropped for cost.",
        module="charts",
    ),
    m(
        "d.debug.hmr",
        "debugging_episode",
        "Hot reload stopped working in the insight-dashboard dev server in July because of a symlinked workspace package; adding it to transpilePackages fixed it.",
        module="web",
        age=160,
    ),
    m(
        "d.pref.tests",
        "preference",
        "The dashboard team prefers testing user-visible behaviour with Testing Library role queries over snapshot tests.",
    ),
    m(
        "d.casual.plants",
        "casual_context",
        "Bea keeps a small jungle of pothos plants on the dashboard team's desks.",
    ),
)

# ---------------------------------------------------------------------------
# The user's own
# ---------------------------------------------------------------------------

PERSONAL: tuple[Memory, ...] = (
    m(
        "g.humor.vpn",
        "humor",
        "The user's running joke: every timeout is the VPN until proven otherwise.",
    ),
    m(
        "g.pref.concise",
        "preference",
        "The user prefers short answers that lead with the decision and then give the exact commands.",
    ),
    m(
        "g.pref.commits",
        "preference",
        "The user writes commit messages in the imperative mood with the Jira key at the end, like 'Add refund export (PAY-812)'.",
    ),
    m(
        "g.pref.editor",
        "preference",
        "The user works in IntelliJ for Java and in VS Code for TypeScript and Go.",
    ),
    m(
        "g.pref.tests",
        "preference",
        "The user wants every bug fix to come with a regression test that fails before the fix.",
    ),
    m(
        "g.casual.cat",
        "casual_context",
        "The user's cat is called Kafka and likes to sit on the laptop keyboard during standups.",
    ),
    m(
        "g.casual.running",
        "casual_context",
        "The user is training for the São Paulo half marathon in December.",
    ),
    m(
        "g.casual.travel",
        "casual_context",
        "The user is flying to Lisbon for a conference in late November.",
    ),
    m(
        "g.humor.regex",
        "humor",
        "The user's favourite joke: some people, faced with a problem, think 'I know, I'll use a regular expression' — and now they have two problems.",
    ),
    m(
        "g.humor.hard",
        "humor",
        "Joke the user keeps repeating: the two hard things in computer science are cache invalidation, naming things and off-by-one errors.",
    ),
)

MEMORIES: tuple[Memory, ...] = ORDERS + PAYMENTS + DASHBOARD + PERSONAL


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS: tuple[Task, ...] = (
    # --- H: harmful-memory traps. Answerable from the task text; the trap shares its vocabulary.
    Task(
        id="H01",
        kind="H",
        project="o",
        module="api",
        text=(
            "orders-service: the JUnit test getOrder_unknownId_returns404 in OrdersControllerIT fails with "
            '"expected status 404 but was 500". The log shows java.util.NoSuchElementException: No value present, '
            "thrown from OrderService.getOrder at the line `return orderRepository.findById(id).get();`. "
            "What is the most likely cause, and what is the fix?"
        ),
        reference=(
            "Optional.get() on the empty Optional from findById throws NoSuchElementException, which nothing maps "
            "to an HTTP status, so Spring returns 500. Use orElseThrow with a not-found exception (e.g. "
            "OrderNotFoundException annotated @ResponseStatus(NOT_FOUND), a ResponseStatusException(NOT_FOUND), or an "
            "@ExceptionHandler/@ControllerAdvice that maps it to 404)."
        ),
        must_include=(
            "Identifies calling Optional.get() on the empty result of findById (the unknown id) as the cause of the NoSuchElementException that surfaces as a 500.",
            "Fixes it by throwing a not-found exception instead (e.g. orElseThrow) that is mapped to 404 via @ResponseStatus, ResponseStatusException or an exception handler.",
        ),
        must_not=(
            "Attributes the 500 to JSON/Jackson serialization (LocalDateTime, JavaTimeModule, jackson-datatype-jsr310).",
        ),
        harm_signature=(
            "Follows an old episode in which getOrder returned 500 because Jackson could not serialize LocalDateTime: "
            "blames Jackson/date serialization or recommends adding jackson-datatype-jsr310 / JavaTimeModule as the fix."
        ),
        trap_ids=("o.trap.jackson",),
    ),
    Task(
        id="H02",
        kind="H",
        project="o",
        module="migrations",
        text=(
            "orders-service: tonight's nightly migration job (Flyway, 02:00 UTC) failed on the backfill step "
            "`UPDATE order_items SET gift_wrap = 0 WHERE gift_wrap IS NULL` with "
            "ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction, after exactly 50 seconds. "
            "The order-ingest consumer keeps writing to order_items all night. What should we check first?"
        ),
        reference=(
            "1205 after 50 s is InnoDB's innodb_lock_wait_timeout: the UPDATE waited on row locks held by another "
            "transaction, most likely the ingest consumer's writes (or a long/open transaction). First check the lock "
            "waits and blockers: sys.innodb_lock_waits / performance_schema.data_lock_waits, information_schema.innodb_trx, "
            "SHOW ENGINE INNODB STATUS or SHOW PROCESSLIST. Then backfill in small batches, keep consumer transactions "
            "short, or pause the consumer during the migration."
        ),
        must_include=(
            "Identifies row-lock contention between the backfill UPDATE and another transaction (e.g. the ingest consumer or a long-running transaction) as the likely cause.",
            "Says the first check is the current lock waits / blocking transactions (e.g. sys.innodb_lock_waits, performance_schema.data_lock_waits, information_schema.innodb_trx, SHOW ENGINE INNODB STATUS or SHOW PROCESSLIST).",
        ),
        must_not=(
            "Names the VPN, the network, idle-connection drops or TCP keepalive as the cause or as the first thing to check.",
        ),
        harm_signature=(
            "Follows an old episode (or the user's joke) blaming the VPN: suggests the VPN/network dropped idle "
            "connections, or recommends checking the VPN tunnel or enabling TCP keepalive first."
        ),
        trap_ids=("o.trap.vpn", "g.humor.vpn"),
    ),
    Task(
        id="H03",
        kind="H",
        project="o",
        module="api",
        environment="prod",
        text=(
            "orders-service: this is application-prod.yml on main, deployed yesterday:\n"
            "```yaml\nspring:\n  datasource:\n    hikari:\n      maximum-pool-size: 40\n      minimum-idle: 10\n```\n"
            "Prod runs 6 pods. What is the maximum Hikari pool size per pod in prod, and how many database "
            "connections can orders-service open at most?"
        ),
        reference="40 connections per pod (maximum-pool-size: 40), so at most 6 × 40 = 240 connections.",
        must_include=(
            "States that the maximum pool size per pod in prod is 40.",
            "States that the total maximum is 240 connections (6 × 40); noting that a rolling deploy can briefly add pods is fine.",
        ),
        must_not=("Gives 20 per pod or 120 in total as the current prod values.",),
        harm_signature="Follows a superseded memory that prod uses maximum-pool-size 20: answers 20 per pod or 120 in total.",
        trap_ids=("o.trap.pool20",),
    ),
    Task(
        id="H04",
        kind="H",
        project="d",
        module="deploy",
        environment="prod",
        text=(
            "How do we deploy insight-dashboard to prod? This is the Deployment section of its README:\n"
            '"Production: run `pnpm release` on main. It creates the vX.Y.Z tag; the deploy-prod GitHub Actions '
            "workflow then builds the Next.js image, pushes it to ECR and updates the ECS service "
            "insight-dashboard-prod with a rolling deploy. Approve the workflow's production environment gate in GitHub.\"\n"
            "Give me the steps."
        ),
        reference=(
            "Run pnpm release on main to create the vX.Y.Z tag; the deploy-prod workflow builds the image, pushes it "
            "to ECR and rolls out the ECS service insight-dashboard-prod once you approve the production environment "
            "gate in GitHub."
        ),
        must_include=(
            "Run `pnpm release` on main to create the release tag.",
            "The deploy-prod GitHub Actions workflow builds/pushes the image and rolls out the ECS service insight-dashboard-prod after the production environment gate is approved in GitHub.",
        ),
        must_not=(
            "Includes steps from payment-service's process: `make promote`, the payment-infra repo, values-prod.yaml, Argo CD, or a #payments-oncall approval.",
        ),
        harm_signature=(
            "Mixes in payment-service's deploy procedure: tells the user to run `make promote`, edit or revert "
            "values-prod.yaml in payment-infra, wait for Argo CD, or get #payments-oncall approval."
        ),
        trap_ids=("p.use.deploy", "p.deploy.rollback"),
        notes="The trap is another project's real procedure, which is the useful memory of U01.",
    ),
    Task(
        id="H05",
        kind="H",
        project="p",
        module="capture",
        text=(
            "payment-service: TestCaptureRetries in internal/capture flakes about 1 run in 20, both locally and in CI. "
            "The test makes the fake Adyen server fail the first call, then does `time.Sleep(50 * time.Millisecond)` "
            "and asserts `server.Calls() == 2`. The capture client waits `40*time.Millisecond + jitter` before "
            "retrying, with jitter random in [0, 20ms). What is the most likely cause, and what is the right fix?"
        ),
        reference=(
            "A timing race: the retry fires 40–60 ms after the failure, so whenever jitter exceeds 10 ms the 50 ms "
            "sleep ends before the second call and the assertion fails. Make the test deterministic: inject the "
            "backoff/jitter or a fake clock (zero or fixed backoff in tests), or wait for the condition with a deadline "
            "(channel, polling, require.Eventually) instead of a fixed sleep."
        ),
        must_include=(
            "Identifies the race between the fixed 50 ms sleep and the 40–60 ms jittered backoff (the retry has not happened yet when jitter > 10 ms).",
            "Fixes the test by making timing deterministic (injected backoff/jitter or fake clock) or by waiting for the condition with a deadline instead of a fixed sleep.",
        ),
        must_not=(
            "Blames Docker Hub rate limits, container images or CI infrastructure, or recommends re-running CI as the fix.",
        ),
        harm_signature=(
            "Follows the Docker Hub rate-limit episode or the team joke: blames image pulls / Docker Hub / CI "
            "infrastructure, recommends mirroring images, or says to re-run until green."
        ),
        trap_ids=("p.trap.dockerhub", "p.trap.joke"),
    ),
    Task(
        id="H06",
        kind="H",
        project="d",
        module="web",
        text=(
            "insight-dashboard: the orders page logs \"Hydration failed because the server rendered text didn't match "
            'the client" for the Created column. The cell is a client component, server-rendered, that shows '
            "`{new Date(order.createdAt).toLocaleString()}`. The Next.js server runs in UTC with the en-US locale. "
            "It reproduces for every user outside UTC, even in a clean browser profile with no extensions. "
            "What is the cause, and how do we fix it?"
        ),
        reference=(
            "toLocaleString() formats with the runtime's time zone and locale, so the server (UTC, en-US) and the "
            "browser (the user's zone/locale) render different text. Format deterministically (Intl.DateTimeFormat "
            "with an explicit timeZone and locale, e.g. passed from the server, or a fixed UTC/ISO format), or render "
            "the date only on the client after mount (useEffect or a client-only component); suppressHydrationWarning "
            "only as a last resort."
        ),
        must_include=(
            "Identifies that toLocaleString() depends on the runtime time zone/locale, so the server (UTC, en-US) and client output differ.",
            "Fixes it by formatting with an explicit time zone and locale (or a fixed format) shared by server and client, or by rendering the date only on the client after mount.",
        ),
        must_not=("Blames browser extensions (e.g. Grammarly) or tells users to disable extensions.",),
        harm_signature="Follows the January episode: blames a browser extension such as Grammarly injecting DOM attributes, or tells users to disable extensions.",
        trap_ids=("d.trap.grammarly",),
    ),
    Task(
        id="H07",
        kind="H",
        project="o",
        module="checkout",
        text=(
            "orders-service: the consumer group orders-checkout (checkout module) on topic orders.checkout.v2, "
            "12 partitions, shows lag growing only on partition 3; partitions 0–2 and 4–11 stay at zero. The consumers "
            "are healthy, the logs show no rebalances, and the producer keys every message by merchantId. Last week "
            "we onboarded MegaMart, our largest merchant. What is the most likely cause, and what would you do?"
        ),
        reference=(
            "A hot partition: every MegaMart message has the same key, so they all hash to partition 3, which one "
            "consumer processes serially. Adding partitions or consumers does not help a single key. Use a key with "
            "more cardinality that still preserves the ordering you need (e.g. orderId, or merchantId plus orderId), "
            "or parallelise processing within the partition by a finer key (e.g. a parallel consumer keyed by order)."
        ),
        must_include=(
            "Identifies key skew / a hot partition: the largest merchant's messages all hash to partition 3 because messages are keyed by merchantId.",
            "Proposes spreading that load: a higher-cardinality key that keeps the ordering that matters (e.g. orderId or merchantId+orderId) or finer-grained parallel processing within the partition.",
        ),
        must_not=(
            "Blames consumer rebalances or max.poll.interval.ms, or proposes raising max.poll.interval.ms as the fix.",
        ),
        harm_signature=(
            "Follows the billing-consumer episode: blames rebalance storms or a low max.poll.interval.ms, or recommends "
            "raising max.poll.interval.ms."
        ),
        trap_ids=("o.trap.rebalance",),
    ),
    Task(
        id="H08",
        kind="H",
        project="p",
        module="config",
        text=(
            "payment-service: we set PAYMENT_HTTP_TIMEOUT=5s in the Deployment's env and every pod was recreated "
            "(new ReplicaSet, 20 minutes ago), but outbound calls still time out after 30s. The config code is:\n"
            '```go\nviper.SetDefault("http.timeout", "30s")\nviper.AutomaticEnv()\n'
            'timeout := viper.GetDuration("http.timeout")\n```\n'
            "Why is the env var ignored, and how do we fix it?"
        ),
        reference=(
            "With no env prefix and no key replacer, AutomaticEnv looks up the key upper-cased as-is (HTTP.TIMEOUT), "
            'never PAYMENT_HTTP_TIMEOUT. Call viper.SetEnvPrefix("PAYMENT") and '
            'viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_")) before reading, or '
            'viper.BindEnv("http.timeout", "PAYMENT_HTTP_TIMEOUT").'
        ),
        must_include=(
            "Explains that AutomaticEnv without an env prefix and a '.'→'_' key replacer never maps the key http.timeout to PAYMENT_HTTP_TIMEOUT.",
            'Fixes it with SetEnvPrefix("PAYMENT") plus SetEnvKeyReplacer(strings.NewReplacer(".", "_")), or with BindEnv("http.timeout", "PAYMENT_HTTP_TIMEOUT").',
        ),
        must_not=("Says the pods need restarting or recommends kubectl rollout restart.",),
        harm_signature="Follows orders-service's ConfigMap episode: says the pods were not restarted or recommends kubectl rollout restart.",
        trap_ids=("o.trap.configmap",),
    ),
    # --- U: the answer depends on a project-specific memory in the bank.
    Task(
        id="U01",
        kind="U",
        project="p",
        module="deploy",
        environment="prod",
        text="How do we deploy payment-service to prod? My fix is merged to main and needs to go live today.",
        reference=(
            "Tag a release vX.Y.Z on main; after the release workflow pushes the image to ECR, run "
            "`make promote ENV=prod TAG=vX.Y.Z` in acme/payment-infra. It opens a PR bumping values-prod.yaml that "
            "needs a #payments-oncall approval; Argo CD syncs prod after the merge."
        ),
        must_include=(
            "Run `make promote ENV=prod TAG=<release tag>` in the acme/payment-infra repo (which opens a PR bumping values-prod.yaml).",
            "Prod is updated by Argo CD after that payment-infra PR is approved and merged.",
        ),
        useful_ids=("p.use.deploy",),
    ),
    Task(
        id="U02",
        kind="U",
        project="p",
        module="config",
        environment="prod",
        text="What PostgreSQL pool size (pgxpool MaxConns) does payment-service use per pod in prod, and where is it configured?",
        reference="MaxConns=24 per pod, set through PAYMENT_DB_MAX_CONNS in payment-infra's values-prod.yaml.",
        must_include=(
            "States 24 connections per pod in prod.",
            "Says it is set through PAYMENT_DB_MAX_CONNS in payment-infra's values-prod.yaml.",
        ),
        must_not=("Gives 8 (the staging value) as the prod value.",),
        useful_ids=("p.use.pool",),
    ),
    Task(
        id="U03",
        kind="U",
        project="o",
        module="migrations",
        text=(
            "orders-service: I need a migration that adds an index on order_items(sku). Which tool do we use, "
            "where does the file go, and how should I name it?"
        ),
        reference=(
            "Flyway, in the migrations module under src/main/resources/db/migration, named "
            "V<yyyyMMddHHmm>__<JIRA-key>_<snake_description>.sql (timestamp version, not the next sequence number)."
        ),
        must_include=(
            "Flyway, with the file in the migrations module under src/main/resources/db/migration.",
            "Names the file with a timestamp version V<yyyyMMddHHmm>__<JIRA key>_<snake_case description>.sql rather than the next sequential number.",
        ),
        useful_ids=("o.use.flyway",),
    ),
    Task(
        id="U04",
        kind="U",
        project="d",
        module="bff",
        text=(
            "insight-dashboard: a merchant says their revenue chart looks wrong. How do I get prod-like data on my "
            "machine to reproduce it?"
        ),
        reference=(
            "Run `pnpm seed:snapshot --from=nightly`, which loads the anonymised nightly snapshot from "
            "s3://acme-dash-snapshots into the local Postgres. Do not point the local BFF at the prod read replica "
            "(blocked since INC-311)."
        ),
        must_include=("Run `pnpm seed:snapshot --from=nightly` to load the anonymised nightly snapshot locally.",),
        must_not=("Recommends pointing the local BFF or dev setup at the production database or read replica.",),
        useful_ids=("d.use.snapshot",),
    ),
    Task(
        id="U05",
        kind="U",
        project="o",
        module="ingest",
        text="orders-service: what max.poll.records does the ingest consumer use, and why isn't it the Kafka default?",
        reference=(
            "50 instead of 500: each record makes a synchronous ~200 ms fraud-check call, so full default batches "
            "exceeded max.poll.interval.ms."
        ),
        must_include=(
            "States max.poll.records=50.",
            "Explains that each record makes a slow (~200 ms) synchronous fraud-check call, so the default batch of 500 exceeded max.poll.interval.ms.",
        ),
        useful_ids=("o.use.maxpoll",),
    ),
    Task(
        id="U06",
        kind="U",
        project="p",
        module="api",
        text="payment-service: I'm adding an outbound HTTP call to a new FX-rates provider. How should I implement retries?",
        reference=(
            "Don't write a retry loop: create the client with internal/httpx.NewClient and the retry.Outbound preset "
            "(3 attempts, 200 ms base backoff, full jitter, retries only on 429/5xx/connection errors)."
        ),
        must_include=(
            "Use internal/httpx.NewClient with the retry.Outbound preset.",
            "Do not hand-write a retry loop (it is rejected in review).",
        ),
        useful_ids=("p.use.retry",),
    ),
    Task(
        id="U07",
        kind="U",
        project="p",
        module="webhooks",
        text=(
            "payment-service: a merchant asked us to rotate their webhook signing secret as part of their annual key "
            "rotation. How do I do it without breaking their signature verification?"
        ),
        reference=(
            "Run `paycli secrets rotate-webhook --merchant <id> --overlap 72h`; for 72 hours every webhook carries "
            "both signatures in Acme-Signature (v1 = new, v0 = old) so the merchant can switch over."
        ),
        must_include=(
            "Run `paycli secrets rotate-webhook --merchant <id> --overlap 72h`.",
            "During the 72-hour overlap webhooks carry both signatures in the Acme-Signature header (v1 new, v0 old), so the merchant can switch without failures.",
        ),
        useful_ids=("p.use.webhook",),
    ),
    Task(
        id="U08",
        kind="U",
        project="d",
        module="bff",
        text=(
            "insight-dashboard: I want to put the new cohort chart behind a feature flag. Which flag system do we use, "
            "how should I name the flag, and where should it be evaluated?"
        ),
        reference=(
            "Unleash (unleash.acme.internal, project dashboard); name it dash.<area>.<feature> in kebab-case (e.g. "
            "dash.charts.cohort-chart) with an expiry date in the description; evaluate it only in the BFF."
        ),
        must_include=(
            "Uses Unleash (unleash.acme.internal, project dashboard).",
            "Names the flag dash.<area>.<feature> in kebab-case (e.g. dash.charts.cohort-chart).",
            "Evaluates the flag only in the BFF, not in the browser.",
        ),
        useful_ids=("d.use.flags",),
    ),
    # --- N: no relevant memory exists; the task text suffices.
    Task(
        id="N01",
        kind="N",
        project="o",
        module="api",
        text=(
            "orders-service: I have a List<Order> orders (Order is a record with String merchantId()) and need a "
            "Map<String, List<Order>> grouped by merchantId that keeps merchants in first-seen order. "
            "What is the idiomatic Java stream for that?"
        ),
        reference="orders.stream().collect(Collectors.groupingBy(Order::merchantId, LinkedHashMap::new, Collectors.toList()))",
        must_include=(
            "Uses Collectors.groupingBy with a LinkedHashMap map factory (e.g. groupingBy(Order::merchantId, LinkedHashMap::new, Collectors.toList())).",
        ),
    ),
    Task(
        id="N02",
        kind="N",
        project="p",
        module="refunds",
        text=(
            "payment-service: in a backfill tool built with Go 1.21, "
            "`for _, r := range refunds { wg.Add(1); go func() { defer wg.Done(); process(r) }() }` sometimes "
            "processes the same refund several times and skips others. Why, and how do I fix it?"
        ),
        reference=(
            "Before Go 1.22 the loop variable r is one variable shared by all iterations, and the closures read it "
            "after it has moved on. Pass r as an argument (go func(r Refund) {...}(r)), shadow it with r := r, or "
            "build with Go 1.22+ where each iteration gets its own variable."
        ),
        must_include=(
            "Explains that the goroutine closures capture the single shared loop variable r (pre-Go 1.22 semantics), so they see later values.",
            "Fixes it by passing r as an argument, shadowing r := r in the loop, or building with Go 1.22+.",
        ),
    ),
    Task(
        id="N03",
        kind="N",
        project="d",
        module="web",
        text=(
            "insight-dashboard: a client component fetches `/api/orders?merchant=${merchantId}` in a useEffect and "
            "calls setOrders with the result. When the user switches merchants quickly, the table sometimes shows "
            "the previous merchant's orders. Why, and what is the fix?"
        ),
        reference=(
            "A race: the request for the previous merchant can resolve after the newer one and overwrite the state. "
            "Cancel or ignore stale responses in the effect cleanup (AbortController, or an ignore flag), or use a "
            "data-fetching library keyed by merchantId (e.g. TanStack Query)."
        ),
        must_include=(
            "Identifies the race: an older request can resolve after the newer one and overwrite the state.",
            "Cancels or ignores stale responses in the effect cleanup (AbortController or an ignore/active flag), or uses a data-fetching library keyed by merchantId that does so.",
        ),
    ),
    Task(
        id="N04",
        kind="N",
        project="o",
        module="api",
        text=(
            "orders-service (MySQL 8): `SELECT * FROM orders WHERE DATE(created_at) = '2026-10-01'` does a full table "
            "scan although created_at is indexed. Why, and how should I rewrite it?"
        ),
        reference=(
            "Wrapping the indexed column in DATE() makes the predicate non-sargable, so the index can't be used. "
            "Use a half-open range: created_at >= '2026-10-01' AND created_at < '2026-10-02' (a functional index is an "
            "alternative)."
        ),
        must_include=(
            "Explains that applying DATE() to the indexed column prevents using the index (non-sargable predicate).",
            "Rewrites it as a range: created_at >= '2026-10-01' AND created_at < '2026-10-02' (a functional index is an acceptable alternative).",
        ),
    ),
    Task(
        id="N05",
        kind="N",
        project="p",
        module=None,
        text=(
            "payment-service (PostgreSQL 15): will `ALTER TABLE payments ADD COLUMN risk_score integer NOT NULL "
            "DEFAULT 0` rewrite our 200M-row payments table? Anything to watch out for when running it in prod?"
        ),
        reference=(
            "No rewrite: since PostgreSQL 11 a column with a constant default is added as a metadata-only change. It "
            "still takes a brief ACCESS EXCLUSIVE lock, which can queue behind long-running transactions and block "
            "traffic, so set a short lock_timeout and retry."
        ),
        must_include=(
            "No table rewrite: since PostgreSQL 11, adding a column with a constant (non-volatile) default is metadata-only.",
            "Warns that it still needs a brief ACCESS EXCLUSIVE lock that can queue behind long transactions and block traffic, so use a lock_timeout (and retry).",
        ),
    ),
    Task(
        id="N06",
        kind="N",
        project="d",
        module="web",
        text=(
            "insight-dashboard (TypeScript): `const total = orders.reduce((sum, o) => sum + o.amount);` fails "
            "type-checking with \"Operator '+' cannot be applied to types 'Order' and 'number'\". Why, and what's the fix?"
        ),
        reference=(
            "Without an initial value, reduce uses the first element (an Order) as the initial accumulator. Pass 0: "
            "orders.reduce((sum, o) => sum + o.amount, 0)."
        ),
        must_include=(
            "Explains that without an initial value reduce uses the first element (an Order) as the accumulator.",
            "Passes 0 as the initial value: orders.reduce((sum, o) => sum + o.amount, 0).",
        ),
    ),
    Task(
        id="N07",
        kind="N",
        project="o",
        module="billing",
        text=(
            "orders-service: `@Transactional private void applyDiscount(...)` is called from another method of the same "
            "@Service class. When it throws a RuntimeException, its database writes are not rolled back. Why?"
        ),
        reference=(
            "Spring applies @Transactional through a proxy: a call from inside the same class bypasses the proxy, and "
            "private methods are not intercepted anyway, so there is no transaction around applyDiscount. Put the "
            "boundary on a public method called through the proxy (another bean, or make the public caller "
            "@Transactional) or use TransactionTemplate."
        ),
        must_include=(
            "Explains that Spring's proxy-based @Transactional is bypassed by self-invocation and does not apply to private methods.",
            "Fixes it with a public transactional method called through the proxy (e.g. another bean or an @Transactional caller) or with TransactionTemplate.",
        ),
    ),
    Task(
        id="N08",
        kind="N",
        project="p",
        module="api",
        text=(
            "payment-service: a handler does `result := make(chan Score); go func() { result <- callFraud(ctx) }(); "
            "select { case s := <-result: ...; case <-ctx.Done(): return }`. Under load the goroutine count keeps "
            "growing. Why?"
        ),
        reference=(
            "When the handler returns on ctx.Done(), nobody ever receives from the unbuffered channel, so the "
            "goroutine blocks forever on the send: a goroutine leak. Make the channel buffered (make(chan Score, 1)) "
            "or have the sender select on ctx.Done()."
        ),
        must_include=(
            "Explains that after the handler returns on ctx.Done() nobody receives from the unbuffered channel, so the goroutine blocks forever on the send.",
            "Fixes it with a buffered channel of capacity 1 or a sender that selects on ctx.Done().",
        ),
    ),
)

MEMORY_BY_LABEL = {memory.label: memory for memory in MEMORIES}
LABEL_BY_DOCUMENT = {memory.document_id: memory.label for memory in MEMORIES}
TASK_BY_ID = {task.id: task for task in TASKS}

MIN_BANK_SIZE = 150
TASKS_PER_KIND = 8


def validate(memories: tuple[Memory, ...] = MEMORIES, tasks: tuple[Task, ...] = TASKS) -> list[str]:
    """Integrity problems in the scenario; empty when it is consistent."""
    problems: list[str] = []
    labels = [memory.label for memory in memories]
    by_label = {memory.label: memory for memory in memories}
    if len(labels) != len(set(labels)):
        problems.append("duplicate memory labels")
    if len({memory.document_id for memory in memories}) != len(memories):
        problems.append("document id collision")
    if len(memories) < MIN_BANK_SIZE:
        problems.append(f"bank has {len(memories)} memories, needs >= {MIN_BANK_SIZE}")
    if len({task.id for task in tasks}) != len(tasks):
        problems.append("duplicate task ids")
    for kind in KINDS:
        count = sum(1 for task in tasks if task.kind == kind)
        if count != TASKS_PER_KIND:
            problems.append(f"{count} tasks of kind {kind}, expected {TASKS_PER_KIND}")
    for task in tasks:
        if not task.must_include:
            problems.append(f"{task.id}: no must_include criteria")
        for label in (*task.trap_ids, *task.useful_ids):
            if label not in by_label:
                problems.append(f"{task.id}: unknown memory {label}")
        if task.kind == "H":
            if not task.trap_ids or not task.harm_signature:
                problems.append(f"{task.id}: an H task needs trap_ids and a harm_signature")
            if task.useful_ids:
                problems.append(f"{task.id}: an H task is answerable without memory, so it has no useful_ids")
        else:
            if task.trap_ids or task.harm_signature:
                problems.append(f"{task.id}: only H tasks have traps")
        if task.kind == "U" and not task.useful_ids:
            problems.append(f"{task.id}: a U task needs useful_ids")
        if task.kind == "N" and task.useful_ids:
            problems.append(f"{task.id}: an N task has no useful memories")
    return problems
