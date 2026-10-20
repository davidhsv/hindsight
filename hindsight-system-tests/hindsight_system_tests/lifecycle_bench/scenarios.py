"""The memory_lifecycle scenarios: the 13 lifecycle situations plus error injection.

Each scenario is one bank. Raw facts are labelled by ``document_id`` (one document per
labelled memory), observations through the documents of the facts they cite. Patterns are
case-insensitive regexes over observation text (see ``labels.py`` for how they combine with
citations); they were written with the scenario, before any arm ran.

The holdout split is whole scenarios (banks and their consolidation payloads together): the
third of scenario ids that sort first by sha256 (``holdout_ids``). It is scored separately
and is not to be iterated on.
"""

from __future__ import annotations

import hashlib

from .model import (
    Conflict,
    ConflictSide,
    Create,
    Historical,
    Item,
    Merge,
    Query,
    Reinforce,
    Scenario,
    Split,
    Step,
    Supersede,
    decision_facts,
)

# ---------------------------------------------------------------------------
# Lifecycle situations
# ---------------------------------------------------------------------------

PREFERENCE_REVERSAL = Scenario(
    id="preference_reversal",
    title="A stated preference is reversed",
    covers=("preference reversal",),
    steps=(
        Step(
            items=(
                Item(
                    doc="pref-tabs",
                    content="I like tabs for indenting Python files.",
                    facts=("The user prefers tabs for indentation in Python files",),
                    day=0,
                    session="pref-s1",
                    entities=("user", "Python"),
                ),
                Item(
                    doc="pref-editor",
                    content="I write Python in Neovim.",
                    facts=("The user edits Python code in Neovim",),
                    day=0,
                    minute=5,
                    session="pref-s1",
                    entities=("user", "Neovim"),
                ),
            ),
            decisions=(
                Create("The user prefers tabs for indentation in Python files", ("prefers tabs",)),
                Create("The user edits Python code in Neovim", ("in Neovim",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="pref-spaces",
                    content="I changed my mind about indentation: from now on use four spaces in Python files, not tabs.",
                    facts=("The user now prefers four spaces instead of tabs for indentation in Python files",),
                    day=40,
                    session="pref-s2",
                    entities=("user", "Python"),
                ),
            ),
            decisions=(
                Supersede(
                    target="prefers tabs for indentation",
                    text="The user prefers four spaces for indentation in Python files",
                    facts=("four spaces instead of tabs",),
                    folded_text="The user prefers four spaces for indentation in Python files; earlier they preferred tabs",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="pref-q1",
            text="Does the user prefer tabs or spaces for indentation?",
            kind="current",
            day=50,
            required_active=("pref-spaces",),
            forbidden_stale=("pref-tabs",),
            current_pattern=r"\bspaces\b",
            stale_pattern=r"\btabs\b",
        ),
        Query(
            id="pref-q2",
            text="How should I indent Python code for this user?",
            kind="current",
            day=50,
            required_active=("pref-spaces",),
            forbidden_stale=("pref-tabs",),
            current_pattern=r"\bspaces\b",
            stale_pattern=r"\btabs\b",
        ),
        Query(
            id="pref-q3",
            text="What indentation did the user prefer before switching to spaces?",
            kind="historical",
            day=50,
            allowed_historical=("pref-tabs",),
            current_context=("pref-spaces",),
            current_pattern=r"\bspaces\b",
            stale_pattern=r"\btabs\b",
        ),
        Query(
            id="pref-q4",
            text="Which indentation style does the user prefer?",
            kind="as_of",
            day=50,
            as_of=20,
            allowed_historical=("pref-tabs",),
            current_context=("pref-spaces",),
            current_pattern=r"\bspaces\b",
            stale_pattern=r"\btabs\b",
        ),
        Query(
            id="pref-q5",
            text="Which editor does the user write Python in?",
            kind="current",
            day=50,
            required_active=("pref-editor",),
            current_pattern=r"Neovim",
        ),
    ),
    expected_states={"pref-tabs": "superseded", "pref-spaces": "active", "pref-editor": "active"},
)

_JAVA_CURRENT, _JAVA_STALE = r"\b25\b", r"\b21\b"

VERSION_UPGRADE = Scenario(
    id="version_upgrade",
    title="A runtime is upgraded (Java 21 -> 25)",
    covers=("software-version upgrade",),
    steps=(
        Step(
            items=(
                Item(
                    doc="java-21",
                    content="The billing service runs on Java 21 with the Temurin JDK.",
                    facts=("The billing service runs on Java 21",),
                    day=0,
                    session="java-s1",
                    entities=("billing service", "Java"),
                ),
                Item(
                    doc="build-maven",
                    content="The billing service is built with Maven.",
                    facts=("The billing service is built with Maven",),
                    day=0,
                    minute=5,
                    session="java-s1",
                    entities=("billing service", "Maven"),
                ),
            ),
            decisions=(
                Create("The billing service runs on Java 21", ("runs on Java 21",)),
                Create("The billing service is built with Maven", ("built with Maven",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="java-25",
                    content="We upgraded the billing service from Java 21 to Java 25 this sprint.",
                    facts=("The billing service was upgraded from Java 21 to Java 25",),
                    day=60,
                    session="java-s2",
                    entities=("billing service", "Java"),
                ),
            ),
            decisions=(
                Supersede(
                    target="runs on Java 21",
                    text="The billing service runs on Java 25",
                    facts=("upgraded from Java 21 to Java 25",),
                    folded_text="The billing service runs on Java 25; it was upgraded from Java 21",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="java-q1",
            text="Which Java version does the billing service run on?",
            kind="current",
            day=90,
            required_active=("java-25",),
            forbidden_stale=("java-21",),
            current_pattern=_JAVA_CURRENT,
            stale_pattern=_JAVA_STALE,
        ),
        Query(
            id="java-q2",
            text="Which JDK do I need to build the billing service?",
            kind="current",
            day=90,
            required_active=("java-25",),
            forbidden_stale=("java-21",),
            current_pattern=_JAVA_CURRENT,
            stale_pattern=_JAVA_STALE,
        ),
        Query(
            id="java-q3",
            text="What Java version did the billing service use before the upgrade?",
            kind="historical",
            day=90,
            allowed_historical=("java-21",),
            current_context=("java-25",),
            current_pattern=_JAVA_CURRENT,
            stale_pattern=_JAVA_STALE,
        ),
        Query(
            id="java-q4",
            text="Which Java version does the billing service run on?",
            kind="as_of",
            day=90,
            as_of=30,
            allowed_historical=("java-21",),
            current_context=("java-25",),
            current_pattern=_JAVA_CURRENT,
            stale_pattern=_JAVA_STALE,
        ),
        Query(
            id="java-q5",
            text="How is the billing service built?",
            kind="current",
            day=90,
            required_active=("build-maven",),
            current_pattern=r"Maven",
        ),
    ),
    expected_states={"java-21": "superseded", "java-25": "active", "build-maven": "active"},
)

INFRA_MIGRATION = Scenario(
    id="infra_migration",
    title="CI moves from Jenkins to GitHub Actions",
    covers=("infrastructure migration",),
    steps=(
        Step(
            items=(
                Item(
                    doc="ci-jenkins",
                    content="Our CI pipeline runs on Jenkins at ci.internal.",
                    facts=("The CI pipeline runs on Jenkins",),
                    day=0,
                    session="ci-s1",
                    entities=("CI pipeline", "Jenkins"),
                ),
                Item(
                    doc="deploy-k8s",
                    content="Deployments go to the Kubernetes cluster prod-east.",
                    facts=("Deployments go to the Kubernetes cluster prod-east",),
                    day=0,
                    minute=5,
                    session="ci-s1",
                    entities=("Kubernetes", "prod-east"),
                ),
            ),
            decisions=(
                Create("The CI pipeline runs on Jenkins", ("runs on Jenkins",)),
                Create("Deployments go to the Kubernetes cluster prod-east", ("prod-east",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="ci-gha",
                    content="We finished moving CI from Jenkins to GitHub Actions and shut Jenkins down.",
                    facts=("The CI pipeline moved from Jenkins to GitHub Actions and Jenkins was shut down",),
                    day=90,
                    session="ci-s2",
                    entities=("CI pipeline", "GitHub Actions", "Jenkins"),
                ),
            ),
            decisions=(
                Supersede(
                    target="runs on Jenkins",
                    text="The CI pipeline runs on GitHub Actions",
                    facts=("moved from Jenkins to GitHub Actions",),
                    folded_text="The CI pipeline runs on GitHub Actions; it previously ran on Jenkins, which was shut down",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="ci-q1",
            text="Where does our CI pipeline run?",
            kind="current",
            day=120,
            required_active=("ci-gha",),
            forbidden_stale=("ci-jenkins",),
            current_pattern=r"GitHub Actions",
            stale_pattern=r"Jenkins",
        ),
        Query(
            id="ci-q2",
            text="Which system should I check when a CI build fails?",
            kind="current",
            day=120,
            required_active=("ci-gha",),
            forbidden_stale=("ci-jenkins",),
            current_pattern=r"GitHub Actions",
            stale_pattern=r"Jenkins",
        ),
        Query(
            id="ci-q3",
            text="Which CI system did we use before GitHub Actions?",
            kind="historical",
            day=120,
            allowed_historical=("ci-jenkins",),
            current_context=("ci-gha",),
            current_pattern=r"GitHub Actions",
            stale_pattern=r"Jenkins",
        ),
        Query(
            id="ci-q4",
            text="Where do deployments go?",
            kind="current",
            day=120,
            required_active=("deploy-k8s",),
            current_pattern=r"prod-east",
        ),
    ),
    expected_states={"ci-jenkins": "superseded", "ci-gha": "active", "deploy-k8s": "active"},
)

_REDIS_CURRENT, _REDIS_REFUTED = r"no cache|never deployed", r"uses (a )?Redis"

FALSE_FACT_CORRECTION = Scenario(
    id="false_fact_correction",
    title="A claim is corrected as never true (Redis proposed, never deployed)",
    covers=("false fact correction",),
    steps=(
        Step(
            items=(
                Item(
                    doc="redis-claim",
                    content="We are putting Redis in front of the orders API as a cache.",
                    facts=("The orders API uses Redis as a cache",),
                    day=0,
                    session="redis-s1",
                    entities=("orders API", "Redis"),
                ),
                Item(
                    doc="orders-db",
                    content="The orders API stores its data in PostgreSQL.",
                    facts=("The orders API stores its data in PostgreSQL",),
                    day=0,
                    minute=5,
                    session="redis-s1",
                    entities=("orders API", "PostgreSQL"),
                ),
            ),
            decisions=(
                Create("The orders API uses a Redis cache", ("uses Redis as a cache",)),
                Create("The orders API stores its data in PostgreSQL", ("in PostgreSQL",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="redis-fix",
                    content=(
                        "Correction: Redis was only ever proposed for the orders API and was never deployed; "
                        "nothing caches the orders API."
                    ),
                    facts=(
                        "Redis was only proposed for the orders API and never deployed; the orders API has no cache",
                    ),
                    day=20,
                    session="redis-s2",
                    entities=("orders API", "Redis"),
                ),
            ),
            decisions=(
                Supersede(
                    relation="refutes",
                    target="uses a Redis cache",
                    text="The orders API has no cache; Redis was proposed but never deployed",
                    facts=("never deployed",),
                    folded_text="The orders API has no cache; Redis was proposed but never deployed",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="redis-q1",
            text="Does the orders API use a cache?",
            kind="current",
            day=40,
            required_active=("redis-fix",),
            forbidden_refuted=("redis-claim",),
            current_pattern=_REDIS_CURRENT,
            refuted_pattern=_REDIS_REFUTED,
        ),
        Query(
            id="redis-q2",
            text="What caching layer sits in front of the orders API?",
            kind="current",
            day=40,
            required_active=("redis-fix",),
            forbidden_refuted=("redis-claim",),
            current_pattern=_REDIS_CURRENT,
            refuted_pattern=_REDIS_REFUTED,
        ),
        Query(
            id="redis-q3",
            text="Is Redis deployed for the orders API?",
            kind="current",
            day=40,
            required_active=("redis-fix",),
            forbidden_refuted=("redis-claim",),
            current_pattern=_REDIS_CURRENT,
            refuted_pattern=_REDIS_REFUTED,
        ),
        Query(
            id="redis-q4",
            text="Where does the orders API store its data?",
            kind="current",
            day=40,
            required_active=("orders-db",),
            forbidden_refuted=("redis-claim",),
            current_pattern=r"PostgreSQL",
            refuted_pattern=_REDIS_REFUTED,
        ),
    ),
    expected_states={"redis-claim": "refuted", "redis-fix": "active", "orders-db": "active"},
)

_EXPORT_CURRENT, _EXPORT_STALE = r"decommissioned", r"cron job|02:00"

RETIRED_STATE = Scenario(
    id="retired_state",
    title="A historically true state is retired with no successor",
    covers=("historically true but retired state",),
    steps=(
        Step(
            items=(
                Item(
                    doc="export-job",
                    content="A nightly cron job exports orders to the data warehouse at 02:00.",
                    facts=("A nightly cron job exports orders to the data warehouse at 02:00",),
                    day=0,
                    session="export-s1",
                    entities=("orders export", "data warehouse"),
                ),
                Item(
                    doc="warehouse",
                    content="Our data warehouse is Snowflake.",
                    facts=("The data warehouse is Snowflake",),
                    day=0,
                    minute=5,
                    session="export-s1",
                    entities=("data warehouse", "Snowflake"),
                ),
            ),
            decisions=(
                Create("A nightly cron job exports orders to the data warehouse at 02:00", ("exports orders",)),
                Create("The data warehouse is Snowflake", ("Snowflake",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="export-gone",
                    content="We decommissioned the nightly orders export job; nothing replaces it.",
                    facts=("The nightly orders export job was decommissioned and nothing replaces it",),
                    day=100,
                    session="export-s2",
                    entities=("orders export",),
                ),
            ),
            decisions=(
                Supersede(
                    target="exports orders",
                    text="The nightly orders export job was decommissioned",
                    facts=("decommissioned",),
                    folded_text=(
                        "A nightly cron job used to export orders to the data warehouse at 02:00; it was decommissioned"
                    ),
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="export-q1",
            text="Is there a nightly job exporting orders to the warehouse?",
            kind="current",
            day=130,
            required_active=("export-gone",),
            forbidden_stale=("export-job",),
            current_pattern=_EXPORT_CURRENT,
            stale_pattern=_EXPORT_STALE,
        ),
        Query(
            id="export-q2",
            text="What runs every night at 02:00?",
            kind="current",
            day=130,
            required_active=("export-gone",),
            forbidden_stale=("export-job",),
            current_pattern=_EXPORT_CURRENT,
            stale_pattern=_EXPORT_STALE,
        ),
        Query(
            id="export-q3",
            text="What did the nightly export job do before it was decommissioned?",
            kind="historical",
            day=130,
            allowed_historical=("export-job",),
            current_context=("export-gone",),
            current_pattern=_EXPORT_CURRENT,
            stale_pattern=_EXPORT_STALE,
        ),
        Query(
            id="export-q4",
            text="Is there a nightly job exporting orders to the warehouse?",
            kind="as_of",
            day=130,
            as_of=50,
            allowed_historical=("export-job",),
            current_context=("export-gone",),
            current_pattern=_EXPORT_CURRENT,
            stale_pattern=_EXPORT_STALE,
        ),
        Query(
            id="export-q5",
            text="Which data warehouse do we use?",
            kind="current",
            day=130,
            required_active=("warehouse",),
            current_pattern=r"Snowflake",
        ),
    ),
    expected_states={"export-job": "superseded", "export-gone": "active", "warehouse": "active"},
)

CONFLICTING_SOURCES = Scenario(
    id="conflicting_sources",
    title="Docs and build config disagree, with no trust differential",
    covers=("conflicting sources",),
    steps=(
        Step(
            items=(
                Item(
                    doc="readme-node",
                    content="README: the frontend requires Node 18.",
                    facts=("The frontend requires Node 18 according to the README",),
                    day=10,
                    session="conflict-s1",
                    source="docs",
                    entities=("frontend", "Node"),
                ),
                Item(
                    doc="readme-jdk",
                    content="README: the API client generator needs JDK 17.",
                    facts=("The API client generator needs JDK 17 according to the README",),
                    day=10,
                    minute=5,
                    session="conflict-s1",
                    source="docs",
                    entities=("API client generator", "JDK"),
                ),
            ),
            decisions=(
                Create("The frontend requires Node 18", ("requires Node 18",)),
                Create("The API client generator needs JDK 17", ("needs JDK 17",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="ci-node",
                    content="CI config: the frontend pipeline runs on Node 20.",
                    facts=("The frontend CI pipeline runs on Node 20",),
                    day=12,
                    session="conflict-s2",
                    source="ci",
                    entities=("frontend", "Node"),
                ),
                Item(
                    doc="pom-jdk",
                    content="pom.xml compiles the API client generator with release 21.",
                    facts=("The pom compiles the API client generator with Java release 21",),
                    day=12,
                    minute=5,
                    session="conflict-s3",
                    source="build_config",
                    entities=("API client generator", "Java"),
                ),
            ),
            decisions=(
                Conflict(
                    target="requires Node 18",
                    text="The frontend CI pipeline runs on Node 20",
                    facts=("runs on Node 20",),
                    folded_text="The README says the frontend requires Node 18, but the CI pipeline runs it on Node 20",
                ),
                Conflict(
                    target="needs JDK 17",
                    text="The API client generator is compiled with Java release 21",
                    facts=("release 21",),
                    folded_text=(
                        "The README says the API client generator needs JDK 17, but the pom compiles it with Java release 21"
                    ),
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="conflict-q1",
            text="Which Node version does the frontend need?",
            kind="current",
            day=20,
            required_active=("readme-node", "ci-node"),
            conflict_sides=(ConflictSide(("readme-node",), r"Node 18"), ConflictSide(("ci-node",), r"Node 20")),
        ),
        Query(
            id="conflict-q2",
            text="What Node version should I install to work on the frontend?",
            kind="current",
            day=20,
            required_active=("readme-node", "ci-node"),
            conflict_sides=(ConflictSide(("readme-node",), r"Node 18"), ConflictSide(("ci-node",), r"Node 20")),
        ),
        Query(
            id="conflict-q3",
            text="Which Java version does the API client generator need?",
            kind="current",
            day=20,
            required_active=("readme-jdk", "pom-jdk"),
            conflict_sides=(ConflictSide(("readme-jdk",), r"\b17\b"), ConflictSide(("pom-jdk",), r"\b21\b")),
        ),
        Query(
            id="conflict-q4",
            text="Which JDK compiles the API client generator?",
            kind="current",
            day=20,
            required_active=("readme-jdk", "pom-jdk"),
            conflict_sides=(ConflictSide(("readme-jdk",), r"\b17\b"), ConflictSide(("pom-jdk",), r"\b21\b")),
        ),
    ),
    expected_states={
        "readme-node": "uncertain",
        "ci-node": "uncertain",
        "readme-jdk": "uncertain",
        "pom-jdk": "uncertain",
    },
)

SCOPE_DIVERGENCE = Scenario(
    id="scope_divergence",
    title="Different scopes hold different truths (module tags; prod vs test)",
    covers=("scope-specific divergence",),
    steps=(
        Step(
            items=(
                Item(
                    doc="api-java",
                    content="backend-api runs on Java 25.",
                    facts=("backend-api runs on Java 25",),
                    day=0,
                    session="scope-s1",
                    tags=("module:backend-api",),
                    entities=("backend-api", "Java"),
                ),
                Item(
                    doc="worker-java",
                    content="legacy-worker is pinned to Java 11.",
                    facts=("legacy-worker is pinned to Java 11",),
                    day=0,
                    minute=5,
                    session="scope-s1",
                    tags=("module:legacy-worker",),
                    entities=("legacy-worker", "Java"),
                ),
                Item(
                    doc="flags-prod",
                    content="In production the feature-flag service reads from flags.prod.internal.",
                    facts=("In production the feature-flag service reads from flags.prod.internal",),
                    day=0,
                    minute=10,
                    session="scope-s1",
                    tags=("env:prod",),
                    entities=("feature-flag service",),
                ),
                Item(
                    doc="flags-test",
                    content="In the test environment the feature-flag service reads from flags.test.internal.",
                    facts=("In the test environment the feature-flag service reads from flags.test.internal",),
                    day=0,
                    minute=15,
                    session="scope-s1",
                    tags=("env:test",),
                    entities=("feature-flag service",),
                ),
            ),
            decisions=(
                Create("backend-api runs on Java 25", ("backend-api runs on Java 25",)),
                Create("legacy-worker is pinned to Java 11", ("pinned to Java 11",)),
                Create(
                    "In production the feature-flag service reads from flags.prod.internal", ("flags.prod.internal",)
                ),
                Create(
                    "In the test environment the feature-flag service reads from flags.test.internal",
                    ("flags.test.internal",),
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="scope-q1",
            text="Which Java version does the service run on?",
            kind="current",
            day=40,
            required_active=("api-java",),
            tags=("module:backend-api",),
            tags_match="any_strict",
            scope_tags=("module:backend-api",),
        ),
        Query(
            id="scope-q2",
            text="Which Java version does the service run on?",
            kind="current",
            day=40,
            required_active=("worker-java",),
            tags=("module:legacy-worker",),
            tags_match="any_strict",
            scope_tags=("module:legacy-worker",),
        ),
        Query(
            id="scope-q3",
            text="Which host does the feature-flag service read from?",
            kind="current",
            day=40,
            required_active=("flags-prod",),
            tags=("env:prod",),
            tags_match="any_strict",
            scope_tags=("env:prod",),
        ),
        Query(
            id="scope-q4",
            text="Which host does the feature-flag service read from?",
            kind="current",
            day=40,
            required_active=("flags-test",),
            tags=("env:test",),
            tags_match="any_strict",
            scope_tags=("env:test",),
        ),
        Query(
            id="scope-q5",
            text="Which Java versions do backend-api and legacy-worker run on?",
            kind="current",
            day=40,
            required_active=("api-java", "worker-java"),
        ),
        Query(
            id="scope-q6",
            text="Which host does the feature-flag service read from in each environment?",
            kind="current",
            day=40,
            required_active=("flags-prod", "flags-test"),
        ),
    ),
    expected_states={doc: "active" for doc in ("api-java", "worker-java", "flags-prod", "flags-test")},
)


def _spaced_vs_burst(
    *,
    id: str,
    title: str,
    covers: tuple[str, ...],
    prefix: str,
    spaced_claim: str,
    spaced_contents: tuple[str, str, str],
    burst_claim: str,
    burst_count: int,
    queries: tuple[str, ...],
) -> Scenario:
    """A claim confirmed in three independent sessions (days 0 / 30 / 120) beside one repeated
    ``burst_count`` times inside a single session on day 0, asked about with an equal cue."""
    spaced_docs = tuple(f"{prefix}-spaced-{i}" for i in (1, 2, 3))
    burst_docs = tuple(f"{prefix}-burst-{i:02d}" for i in range(1, burst_count + 1))
    burst_items = tuple(
        Item(
            doc=doc,
            content=f"Message {i} of the same chat: {burst_claim[0].lower()}{burst_claim[1:]}.",
            facts=(burst_claim,),
            day=0,
            minute=10 + i,
            session=f"{prefix}-burst-session",
        )
        for i, doc in enumerate(burst_docs, start=1)
    )
    first = Step(
        items=(
            Item(
                doc=spaced_docs[0],
                content=spaced_contents[0],
                facts=(spaced_claim,),
                day=0,
                session=f"{prefix}-session-1",
            ),
            *burst_items,
        ),
        decisions=(Create(spaced_claim, (spaced_claim,)), Create(burst_claim, (burst_claim,))),
    )
    later = tuple(
        Step(
            items=(Item(doc=doc, content=content, facts=(spaced_claim,), day=day, session=f"{prefix}-session-{n}"),),
            decisions=(Reinforce(target=spaced_claim, facts=(spaced_claim,), fallback_text=spaced_claim),),
        )
        for n, (doc, content, day) in enumerate(zip(spaced_docs[1:], spaced_contents[1:], (30, 120)), start=2)
    )
    return Scenario(
        id=id,
        title=title,
        covers=covers,
        steps=(first, *later),
        queries=tuple(
            Query(
                id=f"{prefix}-q{n}",
                text=text,
                kind="current",
                day=150,
                required_active=("spaced", "burst"),
                spaced=("spaced",),
                contrast=("burst",),
            )
            for n, text in enumerate(queries, start=1)
        ),
        groups={"spaced": spaced_docs, "burst": burst_docs},
        expected_states={doc: "active" for doc in (*spaced_docs, *burst_docs)},
    )


SAME_SESSION_NOISE = _spaced_vs_burst(
    id="same_session_noise",
    title="One claim repeated 10x in one session vs one confirmed across spaced sessions",
    covers=("repeated same-session noise", "independent reconfirmation"),
    prefix="noise",
    spaced_claim="Production hotfixes need two reviewers",
    spaced_contents=(
        "Production hotfixes need two reviewers.",
        "Reminder for the new on-call: production hotfixes need two reviewers.",
        "Confirmed again in the quarterly review: production hotfixes need two reviewers.",
    ),
    burst_claim="Production hotfixes need a linked incident ticket",
    burst_count=10,
    queries=(
        "What do production hotfixes need?",
        "What is required before shipping a production hotfix?",
        "Which approval rule applies to production hotfixes?",
    ),
)

INDEPENDENT_RECONFIRMATION = _spaced_vs_burst(
    id="independent_reconfirmation",
    title="Three independent spaced confirmations vs three same-session repeats",
    covers=("independent reconfirmation",),
    prefix="recon",
    spaced_claim="Staging deploys require a green smoke test",
    spaced_contents=(
        "Staging deploys require a green smoke test.",
        "As agreed again today, staging deploys require a green smoke test.",
        "Still true after the pipeline rework: staging deploys require a green smoke test.",
    ),
    burst_claim="Staging deploys require a QA sign-off",
    burst_count=3,
    queries=(
        "What do staging deploys require?",
        "Is there a gate before deploying to staging?",
        "What must happen before a staging deploy goes out?",
    ),
)

LOW_VALUE_DECAY = Scenario(
    id="low_value_decay",
    title="An old incidental fact under a weak cue and under an exact cue",
    covers=("low-value fact decay",),
    steps=(
        Step(
            items=(
                Item(
                    doc="pizza-old",
                    content="At the 2024 offsite the team ordered pizza from Luigi's.",
                    facts=("At the 2024 offsite the team ordered pizza from Luigi's",),
                    day=0,
                    session="decay-s1",
                    entities=("team", "Luigi's"),
                ),
            ),
            decisions=(Create("The team ordered pizza from Luigi's at the 2024 offsite", ("pizza from Luigi's",)),),
        ),
        Step(
            items=(
                Item(
                    doc="lunch-thai",
                    content="Our Friday team lunch is usually Thai food from Bangkok Garden.",
                    facts=("The team's Friday lunch is usually Thai food from Bangkok Garden",),
                    day=300,
                    session="decay-s2",
                    entities=("team", "Bangkok Garden"),
                ),
                Item(
                    doc="caterer",
                    content="All-hands catering is handled by GreenLeaf.",
                    facts=("All-hands catering is handled by GreenLeaf",),
                    day=300,
                    minute=5,
                    session="decay-s2",
                    entities=("GreenLeaf",),
                ),
            ),
            decisions=(
                Create("The team's Friday lunch is usually Thai food from Bangkok Garden", ("Bangkok Garden",)),
                Create("All-hands catering is handled by GreenLeaf", ("GreenLeaf",)),
            ),
        ),
    ),
    queries=(
        Query(
            id="decay-q1",
            text="What food does the team usually get?",
            kind="current",
            day=365,
            required_active=("lunch-thai",),
            current_pattern=r"Thai|Bangkok Garden",
        ),
        Query(
            id="decay-q2",
            text="Where does the team get food from?",
            kind="current",
            day=365,
            required_active=("lunch-thai", "caterer"),
        ),
        Query(
            id="decay-q3",
            text="Which pizzeria did the team order from at the 2024 offsite?",
            kind="current",
            day=365,
            required_active=("pizza-old",),
            current_pattern=r"Luigi",
        ),
        Query(
            id="decay-q4",
            text="Luigi's pizza at the 2024 offsite",
            kind="current",
            day=365,
            required_active=("pizza-old",),
            current_pattern=r"Luigi",
        ),
    ),
    expected_states={doc: "active" for doc in ("pizza-old", "lunch-thai", "caterer")},
)

_BUNDLER_CURRENT, _BUNDLER_STALE = r"Vite", r"webpack"

EXACT_HISTORICAL_CUE = Scenario(
    id="exact_historical_cue",
    title="Asking what was used before the upgrade (webpack -> Vite)",
    covers=("exact historical cue",),
    steps=(
        Step(
            items=(
                Item(
                    doc="bundler-webpack",
                    content="The web app is bundled with webpack 4.",
                    facts=("The web app is bundled with webpack 4",),
                    day=0,
                    session="bundle-s1",
                    entities=("web app", "webpack"),
                ),
                Item(
                    doc="test-jest",
                    content="The web app's unit tests run on Jest.",
                    facts=("The web app's unit tests run on Jest",),
                    day=0,
                    minute=5,
                    session="bundle-s1",
                    entities=("web app", "Jest"),
                ),
            ),
            decisions=(
                Create("The web app is bundled with webpack 4", ("bundled with webpack 4",)),
                Create("The web app's unit tests run on Jest", ("run on Jest",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="bundler-vite",
                    content="We replaced webpack 4 with Vite 5 as the web app's bundler.",
                    facts=("The web app's bundler was switched from webpack 4 to Vite 5",),
                    day=80,
                    session="bundle-s2",
                    entities=("web app", "Vite", "webpack"),
                ),
            ),
            decisions=(
                Supersede(
                    target="bundled with webpack 4",
                    text="The web app is bundled with Vite 5",
                    facts=("from webpack 4 to Vite 5",),
                    folded_text="The web app is bundled with Vite 5; it was switched from webpack 4",
                ),
            ),
        ),
    ),
    queries=(
        *(
            Query(
                id=f"bundle-q{n}",
                text=text,
                kind="historical",
                day=100,
                allowed_historical=("bundler-webpack",),
                current_context=("bundler-vite",),
                current_pattern=_BUNDLER_CURRENT,
                stale_pattern=_BUNDLER_STALE,
            )
            for n, text in enumerate(
                (
                    "What did we use to bundle the web app before the upgrade?",
                    "Which bundler did the web app use before Vite?",
                    "What was the web app's original bundler?",
                ),
                start=1,
            )
        ),
        Query(
            id="bundle-q4",
            text="Which bundler does the web app use?",
            kind="as_of",
            day=100,
            as_of=40,
            allowed_historical=("bundler-webpack",),
            current_context=("bundler-vite",),
            current_pattern=_BUNDLER_CURRENT,
            stale_pattern=_BUNDLER_STALE,
        ),
        Query(
            id="bundle-q5",
            text="Which bundler does the web app use?",
            kind="current",
            day=100,
            required_active=("bundler-vite",),
            forbidden_stale=("bundler-webpack",),
            current_pattern=_BUNDLER_CURRENT,
            stale_pattern=_BUNDLER_STALE,
        ),
        Query(
            id="bundle-q6",
            text="How are the web app's unit tests run?",
            kind="current",
            day=100,
            required_active=("test-jest",),
            current_pattern=r"Jest",
        ),
    ),
    expected_states={"bundler-webpack": "superseded", "bundler-vite": "active", "test-jest": "active"},
)

_JF_STALE = r"always uses JOIN FETCH"

SPLIT_GENERALIZATION = Scenario(
    id="split_generalization",
    title="A generalization splits into two scoped rules (JOIN FETCH)",
    covers=("split/generalization",),
    steps=(
        Step(
            items=(
                Item(
                    doc="jf-general",
                    content="We always use JOIN FETCH in JPA queries to avoid N+1 selects.",
                    facts=("The team always uses JOIN FETCH in JPA queries to avoid N+1 selects",),
                    day=0,
                    session="jpa-s1",
                    entities=("JOIN FETCH", "JPA"),
                ),
            ),
            decisions=(
                Create(
                    "The team always uses JOIN FETCH in JPA queries to avoid N+1 selects", ("always uses JOIN FETCH",)
                ),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="jf-paging",
                    content=(
                        "JOIN FETCH breaks pagination on collection associations, so paged list endpoints use "
                        "EntityGraph with batch fetching instead."
                    ),
                    facts=(
                        "Paged list endpoints use EntityGraph with batch fetching because JOIN FETCH breaks "
                        "pagination on collections",
                    ),
                    day=50,
                    session="jpa-s2",
                    entities=("JOIN FETCH", "EntityGraph"),
                ),
                Item(
                    doc="jf-detail",
                    content="Single-entity detail endpoints keep using JOIN FETCH.",
                    facts=("Single-entity detail endpoints keep using JOIN FETCH",),
                    day=50,
                    minute=5,
                    session="jpa-s2",
                    entities=("JOIN FETCH",),
                ),
            ),
            decisions=(
                Split(
                    target="always uses JOIN FETCH",
                    children=(
                        Create(
                            "Detail endpoints that load a single entity use JOIN FETCH to avoid N+1 selects",
                            ("keep using JOIN FETCH",),
                        ),
                        Create(
                            "Paged list endpoints use EntityGraph with batch fetching instead of JOIN FETCH, because "
                            "JOIN FETCH breaks pagination",
                            ("breaks pagination",),
                        ),
                    ),
                    folded_text=(
                        "The team uses JOIN FETCH for single-entity detail queries, but paged list endpoints use "
                        "EntityGraph with batch fetching because JOIN FETCH breaks pagination"
                    ),
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="jf-q1",
            text="How should a paged list endpoint fetch its associations?",
            kind="current",
            day=70,
            required_active=("jf-paging",),
            forbidden_stale=("jf-general",),
            current_pattern=r"EntityGraph|batch fetching",
            stale_pattern=_JF_STALE,
        ),
        Query(
            id="jf-q2",
            text="Should a single-entity detail endpoint use JOIN FETCH?",
            kind="current",
            day=70,
            required_active=("jf-detail",),
            forbidden_stale=("jf-general",),
            current_pattern=r"detail|single.entity",
            stale_pattern=_JF_STALE,
        ),
        Query(
            id="jf-q3",
            text="Do we always use JOIN FETCH in JPA queries?",
            kind="current",
            day=70,
            required_active=("jf-paging", "jf-detail"),
            forbidden_stale=("jf-general",),
            current_pattern=r"EntityGraph|batch fetching|detail|single.entity",
            stale_pattern=_JF_STALE,
        ),
        Query(
            id="jf-q4",
            text="What was our original rule for JOIN FETCH?",
            kind="historical",
            day=70,
            allowed_historical=("jf-general",),
            current_context=("jf-paging", "jf-detail"),
            current_pattern=r"EntityGraph|batch fetching|detail|single.entity",
            stale_pattern=_JF_STALE,
        ),
    ),
    # Semantic truth: the blanket rule no longer holds as stated, so its raw fact is superseded
    # with the observation it was consolidated into, and current recall answers from the facets.
    expected_states={"jf-general": "superseded", "jf-paging": "active", "jf-detail": "active"},
)

MERGED_DUPLICATES = Scenario(
    id="merged_duplicates",
    title="Two duplicate observations are merged and keep all their evidence",
    covers=("merged duplicate evidence",),
    steps=(
        Step(
            items=(
                Item(
                    doc="dup-a",
                    content="Payments are reconciled every night at 1am UTC.",
                    facts=("Payments are reconciled every night at 01:00 UTC",),
                    day=0,
                    session="dup-s1",
                    entities=("payment reconciliation",),
                ),
                Item(
                    doc="payouts",
                    content="Merchant payouts are sent every Friday.",
                    facts=("Merchant payouts are sent every Friday",),
                    day=0,
                    minute=5,
                    session="dup-s1",
                    entities=("merchant payouts",),
                ),
            ),
            decisions=(
                Create("Payments are reconciled nightly at 01:00 UTC", ("reconciled every night",)),
                Create("Merchant payouts are sent every Friday", ("payouts are sent",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="dup-b",
                    content="The payment reconciliation job runs nightly at one in the morning UTC.",
                    facts=("The payment reconciliation job runs nightly at 01:00 UTC",),
                    day=5,
                    session="dup-s2",
                    entities=("payment reconciliation",),
                ),
            ),
            # The model misses the match and creates a near-duplicate sibling.
            decisions=(
                Create(
                    "The payment reconciliation job runs nightly at 01:00 UTC", ("reconciliation job runs nightly",)
                ),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="dup-c",
                    content="Reminder: payment reconciliation happens nightly at 1am UTC.",
                    facts=("Payment reconciliation happens nightly at 01:00 UTC",),
                    day=10,
                    session="dup-s3",
                    entities=("payment reconciliation",),
                ),
            ),
            decisions=(
                Merge(
                    survivor="Payments are reconciled nightly",
                    merged="reconciliation job runs nightly",
                    facts=("reconciliation happens nightly",),
                    text="Payments are reconciled nightly at 01:00 UTC",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="dup-q1",
            text="When are payments reconciled?",
            kind="current",
            day=20,
            required_active=("dup-a", "dup-b", "dup-c"),
            current_pattern=r"01:00|1am",
        ),
        Query(
            id="dup-q2",
            text="What time does the payment reconciliation job run?",
            kind="current",
            day=20,
            required_active=("dup-a", "dup-b", "dup-c"),
            current_pattern=r"01:00|1am",
        ),
        Query(
            id="dup-q3",
            text="When are merchant payouts sent?",
            kind="current",
            day=20,
            required_active=("payouts",),
            current_pattern=r"Friday",
        ),
    ),
    expected_states={doc: "active" for doc in ("dup-a", "dup-b", "dup-c", "payouts")},
    dedup_merges=(("reconciliation job runs nightly", "Payments are reconciled nightly"),),
)

# ---------------------------------------------------------------------------
# Error injection
# ---------------------------------------------------------------------------

CROSS_SCOPE_SUPERSEDE = Scenario(
    id="cross_scope_supersede",
    title="Error injection: a supersede naming an observation from another scope",
    covers=("error: cross-scope supersede attempt",),
    steps=(
        Step(
            items=(
                Item(
                    doc="web-deploy",
                    content="The web team deploys on Tuesdays.",
                    facts=("The web team deploys on Tuesdays",),
                    day=0,
                    session="xs-s1",
                    tags=("team:web",),
                    entities=("web team",),
                ),
            ),
            decisions=(Create("The web team deploys on Tuesdays", ("deploys on Tuesdays",)),),
        ),
        Step(
            items=(
                Item(
                    doc="mobile-deploy",
                    content="The mobile team deploys on Thursdays.",
                    facts=("The mobile team deploys on Thursdays",),
                    day=20,
                    session="xs-s2",
                    tags=("team:mobile",),
                    entities=("mobile team",),
                ),
            ),
            decisions=(
                Supersede(
                    target="web team deploys on Tuesdays",
                    foreign_target=True,
                    text="The mobile team deploys on Thursdays",
                    facts=("deploys on Thursdays",),
                    folded_text="The mobile team deploys on Thursdays; deploys used to happen on Tuesdays",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="xs-q1",
            text="When does the web team deploy?",
            kind="current",
            day=30,
            required_active=("web-deploy",),
            current_pattern=r"Tuesday",
        ),
        Query(
            id="xs-q2",
            text="When does the mobile team deploy?",
            kind="current",
            day=30,
            required_active=("mobile-deploy",),
            current_pattern=r"Thursday",
        ),
        Query(
            id="xs-q3",
            text="What day do deploys happen?",
            kind="current",
            day=30,
            required_active=("web-deploy",),
            tags=("team:web",),
            tags_match="any_strict",
            scope_tags=("team:web",),
        ),
        Query(
            id="xs-q4",
            text="What day do deploys happen?",
            kind="current",
            day=30,
            required_active=("mobile-deploy",),
            tags=("team:mobile",),
            tags_match="any_strict",
            scope_tags=("team:mobile",),
        ),
    ),
    # The server must refuse to retire across a scope boundary.
    expected_states={"web-deploy": "active", "mobile-deploy": "active"},
)

_REGION_NEW = r"us-west-2|eu-central-1"

DOUBLE_SUPERSEDE = Scenario(
    id="double_supersede",
    title="Error injection: A -> B and A -> C in one response",
    covers=("error: A->B and A->C in one response",),
    steps=(
        Step(
            items=(
                Item(
                    doc="region-east",
                    content="The primary database runs in us-east-1.",
                    facts=("The primary database runs in us-east-1",),
                    day=0,
                    session="dbl-s1",
                    entities=("primary database", "us-east-1"),
                ),
            ),
            decisions=(Create("The primary database runs in us-east-1", ("in us-east-1",)),),
        ),
        Step(
            items=(
                Item(
                    doc="region-west",
                    content="The primary database was moved to us-west-2.",
                    facts=("The primary database was moved to us-west-2",),
                    day=30,
                    session="dbl-s2",
                    entities=("primary database", "us-west-2"),
                ),
                Item(
                    doc="region-eu",
                    content="The primary database was moved to eu-central-1.",
                    facts=("The primary database was moved to eu-central-1",),
                    day=30,
                    minute=5,
                    session="dbl-s3",
                    entities=("primary database", "eu-central-1"),
                ),
            ),
            decisions=(
                Supersede(
                    target="in us-east-1",
                    text="The primary database runs in us-west-2",
                    facts=("moved to us-west-2",),
                    folded_text="The primary database runs in us-west-2; it previously ran in us-east-1",
                ),
                Supersede(
                    target="in us-east-1",
                    text="The primary database runs in eu-central-1",
                    facts=("moved to eu-central-1",),
                    folded_text="The primary database runs in eu-central-1; it previously ran in us-east-1",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="dbl-q1",
            text="Where does the primary database run?",
            kind="current",
            day=40,
            required_active=("moved",),
            forbidden_stale=("region-east",),
            current_pattern=_REGION_NEW,
            stale_pattern=r"us-east-1",
            conflict_sides=(
                ConflictSide(("region-west",), r"us-west-2"),
                ConflictSide(("region-eu",), r"eu-central-1"),
            ),
        ),
        Query(
            id="dbl-q2",
            text="Which AWS region hosts the primary database?",
            kind="current",
            day=40,
            required_active=("moved",),
            forbidden_stale=("region-east",),
            current_pattern=_REGION_NEW,
            stale_pattern=r"us-east-1",
            conflict_sides=(
                ConflictSide(("region-west",), r"us-west-2"),
                ConflictSide(("region-eu",), r"eu-central-1"),
            ),
        ),
        Query(
            id="dbl-q3",
            text="Where did the primary database run before it moved?",
            kind="historical",
            day=40,
            allowed_historical=("region-east",),
            current_context=("moved",),
            current_pattern=_REGION_NEW,
            stale_pattern=r"us-east-1",
        ),
    ),
    groups={"moved": ("region-west", "region-eu")},
    expected_states={"region-east": "superseded"},
)

SUPERSEDE_STILL_TRUE = Scenario(
    id="supersede_still_true",
    title="Error injection: a supersede that retires a claim that is still true",
    covers=("error: superseding a still-true claim",),
    steps=(
        Step(
            items=(
                Item(
                    doc="lint-eslint",
                    content="The frontend is linted with ESLint.",
                    facts=("The frontend is linted with ESLint",),
                    day=0,
                    session="lint-s1",
                    entities=("frontend", "ESLint"),
                ),
                Item(
                    doc="fmt-prettier",
                    content="The frontend is formatted with Prettier.",
                    facts=("The frontend is formatted with Prettier",),
                    day=0,
                    minute=5,
                    session="lint-s1",
                    entities=("frontend", "Prettier"),
                ),
            ),
            decisions=(
                Create("The frontend is linted with ESLint", ("linted with ESLint",)),
                Create("The frontend is formatted with Prettier", ("formatted with Prettier",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="ci-format",
                    content="CI now also runs prettier --check on the frontend.",
                    facts=("CI runs prettier --check on the frontend",),
                    day=40,
                    session="lint-s2",
                    entities=("CI", "Prettier"),
                ),
            ),
            decisions=(
                # The injected error: the new fact adds a check, it does not end ESLint.
                Supersede(
                    target="linted with ESLint",
                    text="CI checks frontend formatting with Prettier",
                    facts=("prettier --check",),
                    folded_text="CI checks frontend formatting with Prettier",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="lint-q1",
            text="How is the frontend linted?",
            kind="current",
            day=50,
            required_active=("lint-eslint",),
            current_pattern=r"ESLint",
        ),
        Query(
            id="lint-q2",
            text="What does CI check on the frontend?",
            kind="current",
            day=50,
            required_active=("ci-format", "lint-eslint"),
        ),
        Query(
            id="lint-q3",
            text="Which tools run on the frontend code?",
            kind="current",
            day=50,
            required_active=("lint-eslint", "fmt-prettier", "ci-format"),
        ),
    ),
    # No expectation: the payload is wrong on purpose and the server cannot know it. The
    # behavioural metrics show what the false retirement costs.
)

_PY_CURRENT, _PY_STALE = r"3\.12", r"3\.10"

OUT_OF_ORDER_OLD_FACT = Scenario(
    id="out_of_order_old_fact",
    title="Error injection: an old statement arrives after its replacement",
    covers=("error: out-of-order old fact",),
    steps=(
        Step(
            items=(
                Item(
                    doc="py-310",
                    content="The data pipeline runs on Python 3.10.",
                    facts=("The data pipeline runs on Python 3.10",),
                    day=10,
                    session="py-s1",
                    entities=("data pipeline", "Python"),
                ),
            ),
            decisions=(Create("The data pipeline runs on Python 3.10", ("runs on Python 3.10",)),),
        ),
        Step(
            items=(
                Item(
                    doc="py-312",
                    content="We moved the data pipeline from Python 3.10 to Python 3.12.",
                    facts=("The data pipeline moved from Python 3.10 to Python 3.12",),
                    day=60,
                    session="py-s2",
                    entities=("data pipeline", "Python"),
                ),
            ),
            decisions=(
                Supersede(
                    target="runs on Python 3.10",
                    text="The data pipeline runs on Python 3.12",
                    facts=("to Python 3.12",),
                    folded_text="The data pipeline runs on Python 3.12; it previously ran on Python 3.10",
                ),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="py-310-late",
                    content="Setup notes from March: the data pipeline is on Python 3.10.",
                    facts=("The data pipeline is on Python 3.10",),
                    day=20,
                    session="py-s3",
                    entities=("data pipeline", "Python"),
                ),
            ),
            decisions=(
                Historical(
                    base_target="Python 3.12",
                    base_text="The data pipeline runs on Python 3.12; earlier it ran on Python 3.10",
                    lifecycle_target="runs on Python 3.10",
                    facts=("is on Python 3.10",),
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="py-q1",
            text="Which Python version does the data pipeline run on?",
            kind="current",
            day=80,
            required_active=("py-312",),
            forbidden_stale=("old310",),
            current_pattern=_PY_CURRENT,
            stale_pattern=_PY_STALE,
        ),
        Query(
            id="py-q2",
            text="What Python should I use for the data pipeline?",
            kind="current",
            day=80,
            required_active=("py-312",),
            forbidden_stale=("old310",),
            current_pattern=_PY_CURRENT,
            stale_pattern=_PY_STALE,
        ),
        Query(
            id="py-q3",
            text="Which Python version did the data pipeline use before 3.12?",
            kind="historical",
            day=80,
            allowed_historical=("old310",),
            current_context=("py-312",),
            current_pattern=_PY_CURRENT,
            stale_pattern=_PY_STALE,
        ),
        Query(
            id="py-q4",
            text="Which Python version does the data pipeline run on?",
            kind="as_of",
            day=80,
            as_of=30,
            allowed_historical=("old310",),
            current_context=("py-312",),
            current_pattern=_PY_CURRENT,
            stale_pattern=_PY_STALE,
        ),
    ),
    groups={"old310": ("py-310", "py-310-late")},
    expected_states={"py-310": "superseded", "py-310-late": "superseded", "py-312": "active"},
)

_SHARED = "shared"

SHARED_SCOPE_PROVENANCE = Scenario(
    id="shared_scope_provenance",
    title="Shared-scope bank: provenance-tagged raw facts under untagged observations",
    covers=("shared scope with provenance tags",),
    steps=(
        Step(
            items=(
                Item(
                    doc="pkg-npm",
                    content="The monorepo uses npm workspaces.",
                    facts=("The monorepo uses npm workspaces",),
                    day=0,
                    session="pkg-s1",
                    tags=("source:chat", "harness:claude-code"),
                    observation_scopes=_SHARED,
                    entities=("monorepo", "npm"),
                ),
                Item(
                    doc="node-pin",
                    content="The monorepo pins Node 20 in .nvmrc.",
                    facts=("The monorepo pins Node 20 in .nvmrc",),
                    day=0,
                    minute=5,
                    session="pkg-s1",
                    tags=("source:chat", "harness:claude-code"),
                    observation_scopes=_SHARED,
                    entities=("monorepo", "Node"),
                ),
            ),
            decisions=(
                Create("The monorepo uses npm workspaces", ("uses npm workspaces",)),
                Create("The monorepo pins Node 20 in .nvmrc", ("pins Node 20",)),
            ),
        ),
        Step(
            items=(
                Item(
                    doc="pkg-pnpm",
                    content="We switched the monorepo from npm workspaces to pnpm workspaces.",
                    facts=("The monorepo switched from npm workspaces to pnpm workspaces",),
                    day=45,
                    session="pkg-s2",
                    tags=("source:chat", "harness:codex"),
                    observation_scopes=_SHARED,
                    entities=("monorepo", "pnpm"),
                ),
            ),
            decisions=(
                Supersede(
                    target="uses npm workspaces",
                    text="The monorepo uses pnpm workspaces",
                    facts=("to pnpm workspaces",),
                    folded_text="The monorepo uses pnpm workspaces; it previously used npm workspaces",
                ),
            ),
        ),
    ),
    queries=(
        Query(
            id="pkg-q1",
            text="Which package manager does the monorepo use?",
            kind="current",
            day=60,
            required_active=("pkg-pnpm",),
            forbidden_stale=("pkg-npm",),
            current_pattern=r"pnpm",
            stale_pattern=r"\bnpm\b",
        ),
        Query(
            id="pkg-q2",
            text="How do I install dependencies in the monorepo?",
            kind="current",
            day=60,
            required_active=("pkg-pnpm",),
            forbidden_stale=("pkg-npm",),
            current_pattern=r"pnpm",
            stale_pattern=r"\bnpm\b",
        ),
        Query(
            id="pkg-q3",
            text="Which package manager did the monorepo use before pnpm?",
            kind="historical",
            day=60,
            allowed_historical=("pkg-npm",),
            current_context=("pkg-pnpm",),
            current_pattern=r"pnpm",
            stale_pattern=r"\bnpm\b",
        ),
        Query(
            id="pkg-q4",
            text="Which Node version does the monorepo pin?",
            kind="current",
            day=60,
            required_active=("node-pin",),
            current_pattern=r"Node 20",
        ),
    ),
    expected_states={"pkg-npm": "superseded", "pkg-pnpm": "active", "node-pin": "active"},
)

SCENARIOS: tuple[Scenario, ...] = (
    PREFERENCE_REVERSAL,
    VERSION_UPGRADE,
    INFRA_MIGRATION,
    FALSE_FACT_CORRECTION,
    RETIRED_STATE,
    CONFLICTING_SOURCES,
    SCOPE_DIVERGENCE,
    SAME_SESSION_NOISE,
    INDEPENDENT_RECONFIRMATION,
    LOW_VALUE_DECAY,
    EXACT_HISTORICAL_CUE,
    SPLIT_GENERALIZATION,
    MERGED_DUPLICATES,
    CROSS_SCOPE_SUPERSEDE,
    DOUBLE_SUPERSEDE,
    SUPERSEDE_STILL_TRUE,
    OUT_OF_ORDER_OLD_FACT,
    SHARED_SCOPE_PROVENANCE,
)


def holdout_ids(scenarios: tuple[Scenario, ...] = SCENARIOS) -> frozenset[str]:
    """The holdout third: whole scenarios, the ``round(n / 3)`` ids that sort first by sha256.

    Hash-ordered so the split is fixed by the ids alone, not by catalogue order or by anyone
    choosing which banks to hide.
    """
    ranked = sorted(scenarios, key=lambda s: hashlib.sha256(s.id.encode()).hexdigest())
    return frozenset(s.id for s in ranked[: round(len(scenarios) / 3)])


def validate(scenarios: tuple[Scenario, ...] = SCENARIOS) -> list[str]:
    """Authoring mistakes that would make a run lie: duplicate ids, a label naming no document,
    an extraction needle another item's content contains, a decision citing no item's fact."""
    problems: list[str] = []
    ids = [s.id for s in scenarios]
    if len(set(ids)) != len(ids):
        problems.append("duplicate scenario ids")
    query_ids = [q.id for s in scenarios for q in s.queries]
    if len(set(query_ids)) != len(query_ids):
        problems.append("duplicate query ids")
    for scenario in scenarios:
        docs = {item.doc for item in scenario.items}
        if len(docs) != len(scenario.items):
            problems.append(f"{scenario.id}: duplicate document ids")
        for query in scenario.queries:
            labels = (
                *query.required_active,
                *query.forbidden_stale,
                *query.forbidden_refuted,
                *query.allowed_historical,
                *query.current_context,
                *query.spaced,
                *query.contrast,
                *(doc for side in query.conflict_sides for doc in side.docs),
            )
            for label in labels:
                if not set(scenario.docs_of(label)) <= docs:
                    problems.append(f"{scenario.id}/{query.id}: label {label!r} names no document")
        for doc in scenario.expected_states:
            if doc not in docs:
                problems.append(f"{scenario.id}: expected state for unknown document {doc!r}")
        for index, step in enumerate(scenario.steps):
            contents = [item.content for item in step.items]
            for a in contents:
                if sum(a in b for b in contents) > 1:
                    problems.append(f"{scenario.id} step {index}: content {a!r} is inside another item's content")
            step_facts = [fact for item in step.items for fact in item.facts]
            for decision in step.decisions:
                for needle in decision_facts(decision):
                    if not any(needle in fact for fact in step_facts):
                        problems.append(f"{scenario.id} step {index}: decision needle {needle!r} matches no fact")
            for fact in step_facts:
                if not any(needle in fact for decision in step.decisions for needle in decision_facts(decision)):
                    problems.append(f"{scenario.id} step {index}: fact {fact!r} is covered by no decision")
    return problems
