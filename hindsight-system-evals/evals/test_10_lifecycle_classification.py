"""Does lifecycle-mode consolidation take the right action on a real model — and only then?

``adaptive_memory_mode=lifecycle`` asks the consolidation model to *label* a change:
a successor that ``supersedes`` an outdated observation, one that ``refutes`` a false
one, two claims that ``conflict``. The server then retires the old observation (and
the raw facts it stood on) instead of rewriting it. A wrong label is worse than none:
a true belief retired drops out of recall. So every ``test_09`` scenario, plus a few
small controls and two conflicts, runs under two arms on separate banks —

* **B**: ``adaptive_memory_mode=off`` (the base consolidation prompt);
* **E**: ``adaptive_memory_mode=lifecycle``;

— and each run is classified deterministically from the server's state after it
settles (claims and relation rows through ``GET /memories/{id}``, see
``hindsight_system_evals/lifecycle_eval.py``) against what the scenario expects:
a supersede, a refute, a conflict, or — in a control — nothing retired at all. The
judged must/trap checks of ``test_09`` run too; in E they grade the current view
(retired observations left out, as recall leaves them out).

Expensive and repeated, so opt-in: ``--full``, ``HINDSIGHT_EVAL_LIFECYCLE_REPEATS=K``
(default 1) and ``HINDSIGHT_EVAL_LIFECYCLE_OUTPUT=runs.jsonl`` for the records, which
``python -m hindsight_system_evals.lifecycle_eval runs.jsonl`` turns into the report.
"""

from __future__ import annotations

import os

import pytest
from hindsight_client import Hindsight

from evals.test_09_consolidation import _SCENARIOS
from hindsight_system_evals import Target, judge
from hindsight_system_evals.consolidation import (
    Fact,
    Scenario,
    Wave,
    noisy_personal_bank,
    noisy_service_bank,
    noisy_service_bank_with_mission,
    run_scenario,
)
from hindsight_system_evals.lifecycle_eval import (
    CONTROL_KINDS,
    ExpectedChange,
    FactRef,
    LifecycleSpec,
    RunCost,
    RunRecord,
    classify,
    fetch_state,
    retired_fn,
)
from hindsight_system_evals.pages import SettleFn

_MODES = {"B": "off", "E": "lifecycle"}
_REPEATS = int(os.getenv("HINDSIGHT_EVAL_LIFECYCLE_REPEATS", "1"))


def _w(*facts: Fact, no_new_observations: bool = False) -> Wave:
    return Wave(tuple(facts), no_new_observations=no_new_observations)


#: Small controls test_09 lacks: a restatement in a new session, a refinement, two
#: services with different versions, the same setting in two environments (told apart
#: only by ``metadata.environment``), events that accumulate, and a scoped exception.
#: Plus two conflicts: the same thing stated two ways on the same day.
_EXTRA: dict[str, Scenario] = {
    s.id: s
    for s in (
        Scenario(
            id="restated-new-session",
            origin="control: the same claim restated in a later session",
            waves=(
                _w(Fact("The user's main code editor is Neovim.", "2026-03-02", ("user", "Neovim"))),
                _w(
                    Fact("The user still uses Neovim as their main code editor.", "2026-05-11", ("user", "Neovim")),
                    no_new_observations=True,
                ),
            ),
            must=("An observation records that the user's main code editor is Neovim.",),
            traps=("Some observation says the user switched editors or stopped using Neovim.",),
        ),
        Scenario(
            id="refinement-adds-detail",
            origin="control: a later fact adds detail to a belief without changing it",
            waves=(
                _w(Fact("The team deploys the API service to AWS.", "2026-04-01", ("API service", "AWS"))),
                _w(
                    Fact(
                        "The team deploys the API service to AWS in the us-east-1 region on ECS Fargate.",
                        "2026-04-20",
                        ("API service", "AWS", "ECS Fargate"),
                    )
                ),
            ),
            must=("The observations record that the API service is deployed to AWS us-east-1 on ECS Fargate.",),
            traps=("Some observation says the API service moved away from AWS or is no longer deployed to AWS.",),
        ),
        Scenario(
            id="different-services-versions",
            origin="control: two different services with different versions",
            waves=(
                _w(Fact("The auth-service runs on Python 3.11.", "2026-05-04", ("auth-service", "Python"))),
                _w(Fact("The billing-service runs on Python 3.12.", "2026-05-25", ("billing-service", "Python"))),
            ),
            must=("The observations say auth-service runs on Python 3.11 and billing-service on Python 3.12.",),
            traps=("Some observation gives auth-service a Python version other than 3.11, or says it was upgraded.",),
        ),
        Scenario(
            id="environment-divergence",
            origin="control: one setting with different values in prod and test (metadata.environment)",
            waves=(
                _w(
                    Fact(
                        "The orders database connection pool size is 50.",
                        "2026-06-01",
                        ("orders database",),
                        metadata=(("environment", "prod"),),
                    )
                ),
                _w(
                    Fact(
                        "The orders database connection pool size is 5.",
                        "2026-06-15",
                        ("orders database",),
                        metadata=(("environment", "test"),),
                    )
                ),
            ),
            must=("The observations still record a connection pool size of 50 as well as a pool size of 5.",),
            traps=("Some observation says the pool size of 50 was replaced, reduced or changed to 5.",),
        ),
        Scenario(
            id="events-accumulate",
            origin="control: a second trip does not undo the first",
            waves=(
                _w(Fact("The user visited Kyoto in October 2025.", "2025-11-03", ("user", "Kyoto"))),
                _w(Fact("The user visited Seoul in April 2026.", "2026-04-12", ("user", "Seoul"))),
            ),
            must=("The observations record both the Kyoto trip (October 2025) and the Seoul trip (April 2026).",),
            traps=("Some observation says the Kyoto trip did not happen or was replaced by the Seoul trip.",),
        ),
        Scenario(
            id="preference-exception",
            origin="control: a scoped exception, not a reversal",
            waves=(
                _w(Fact("The user prefers dark mode in their code editor.", "2026-02-02", ("user",))),
                _w(Fact("The user prefers light mode when presenting slides.", "2026-02-20", ("user",))),
            ),
            must=(
                "The observations record the dark-mode preference for the editor and the light-mode preference "
                "for slides.",
            ),
            traps=("Some observation says the user no longer prefers dark mode in their code editor.",),
        ),
        Scenario(
            id="conflict-same-day-date",
            origin="conflict: two dates for one event, stated the same day",
            waves=(
                _w(
                    Fact(
                        "The quarterly planning offsite is on 14 October 2026.",
                        "2026-09-01",
                        ("quarterly planning offsite",),
                    )
                ),
                _w(
                    Fact(
                        "The quarterly planning offsite is on 21 October 2026.",
                        "2026-09-01",
                        ("quarterly planning offsite",),
                    )
                ),
            ),
            must=(
                "The observations do not settle on one date as certain: both 14 and 21 October are recorded, or "
                "the date is marked as disputed or uncertain.",
            ),
            traps=("Some observation states one of the two dates as settled and the other as wrong or replaced.",),
        ),
        Scenario(
            id="conflict-same-day-gateway",
            origin="conflict: two values for one setting, stated the same day",
            waves=(
                _w(Fact("The Berlin office VPN gateway is vpn-ber-1.", "2026-08-12", ("Berlin office", "VPN"))),
                _w(Fact("The Berlin office VPN gateway is vpn-ber-2.", "2026-08-12", ("Berlin office", "VPN"))),
            ),
            must=(
                "The observations do not settle on one gateway as certain: both vpn-ber-1 and vpn-ber-2 are "
                "recorded, or the gateway is marked as disputed or uncertain.",
            ),
            traps=("Some observation states one gateway as settled and the other as wrong or replaced.",),
        ),
    )
}

_ALL: dict[str, Scenario] = {
    **_SCENARIOS,
    **{s.id: s for s in (noisy_service_bank(), noisy_personal_bank(), noisy_service_bank_with_mission())},
    **_EXTRA,
}


def _ref(scenario_id: str, wave: int, *indexes: int) -> tuple[FactRef, ...]:
    """Facts of a scenario by wave (1-based) and index within it."""
    facts = _ALL[scenario_id].waves[wave - 1].facts
    return tuple(FactRef(wave, facts[j].text) for j in indexes)


def _supersede(
    label: str, stale: tuple[FactRef, ...], new: tuple[FactRef, ...], **kw: tuple[FactRef, ...]
) -> ExpectedChange:
    return ExpectedChange(label, "supersede", stale, new, **kw)


def _horse(wave: int) -> tuple[FactRef, ...]:
    return _ref("horse-farm", wave, 0)


_SERVICE_CHANGES = (
    _supersede(
        "ledger-sync-port",
        (FactRef(1, "The ledger-sync service listens on port 8443."),),
        (FactRef(2, "The ledger-sync service moved from port 8443 to port 9443."),),
    ),
    _supersede(
        "billing-gateway-owner",
        (FactRef(1, "The billing-gateway service is owned by team Atlas."),),
        (FactRef(2, "Ownership of the billing-gateway service moved from team Atlas to team Nimbus."),),
    ),
)

#: What each scenario expects of lifecycle consolidation. Anything not named is protected.
_SPECS: dict[str, LifecycleSpec] = {
    "restatements-merge": LifecycleSpec("control"),
    "employer-supersedes": LifecycleSpec(
        "change",
        (
            _supersede(
                "alice-employer",
                # Current employment (works at, commutes to, four years at Google) ...
                _ref("employer-supersedes", 1, 0, 2, 4),
                _ref("employer-supersedes", 2, 0),
                # ... and the past events of the same tenure, which may go with it.
                allowed=_ref("employer-supersedes", 1, 1, 3),
            ),
        ),
    ),
    "retired-conventions": LifecycleSpec(
        "change",
        tuple(
            _supersede(label, _ref("retired-conventions", 1, j), _ref("retired-conventions", 2, j))
            for j, label in enumerate(("fnm-to-nub", "npx-to-bun", "corepack-removed", "autocommit-default"))
        ),
    ),
    # React is the OLDER statement although retained later: the change runs React -> Svelte.
    "recency-is-when-stated": LifecycleSpec(
        "change",
        (_supersede("react-older", _ref("recency-is-when-stated", 2, 0), _ref("recency-is-when-stated", 1, 0)),),
    ),
    "correction-retires-false-belief": LifecycleSpec(
        "change",
        (
            ExpectedChange(
                "keepass-false",
                "refute",
                _ref("correction-retires-false-belief", 1, 0, 1, 2),
                _ref("correction-retires-false-belief", 2, 0),
            ),
        ),
    ),
    "horse-farm": LifecycleSpec(
        "change",
        (
            _supersede("count-2-to-1", _horse(2), _horse(6)),
            _supersede("count-1-to-5", _horse(6), _horse(7)),
            _supersede("buttercup-sold", _horse(4), _horse(5)),
            _supersede("shadow-died", _horse(9) + _horse(10), _horse(11)),
        ),
    ),
    "no-arithmetic": LifecycleSpec("control"),
    "cascade-to-group": LifecycleSpec(
        "change", (_supersede("ben-left", _ref("cascade-to-group", 1, 0, 1), _ref("cascade-to-group", 2, 0)),)
    ),
    "resolve-reference": LifecycleSpec("control"),
    "distinct-entities-stay-apart": LifecycleSpec(
        "change",
        (
            _supersede(
                "ingest-batch", _ref("distinct-entities-stay-apart", 1, 3), _ref("distinct-entities-stay-apart", 2, 0)
            ),
        ),
    ),
    "language-chinese": LifecycleSpec("control"),
    "language-mixed-update": LifecycleSpec(
        "change",
        (_supersede("zhang-employer", _ref("language-mixed-update", 1, 0), _ref("language-mixed-update", 2, 0, 1)),),
    ),
    "default-drops-chatter": LifecycleSpec("control"),
    "mission-keeps-ephemeral": LifecycleSpec("control"),
    "mission-forbids-names": LifecycleSpec("control"),
    "shared-scope-merges-across-tags": LifecycleSpec("control"),
    "noisy-service-catalogue": LifecycleSpec("lookalike", _SERVICE_CHANGES),
    "noisy-service-catalogue-mission": LifecycleSpec("lookalike", _SERVICE_CHANGES),
    "noisy-personal-contacts": LifecycleSpec(
        "lookalike",
        (
            _supersede(
                "maria-city",
                (FactRef(1, "Maria lives in Turin."),),
                (FactRef(2, "Maria moved from Turin to Bologna."),),
            ),
            _supersede(
                "paolo-job",
                (FactRef(1, "Paolo works as a teacher."),),
                (FactRef(2, "Paolo left his job as a teacher and now works as a pilot."),),
            ),
        ),
    ),
    **{sid: LifecycleSpec("control") for sid in _EXTRA if not sid.startswith("conflict-")},
    **{
        sid: LifecycleSpec("change", (ExpectedChange("two-values", "conflict", _ref(sid, 1, 0), _ref(sid, 2, 0)),))
        for sid in _EXTRA
        if sid.startswith("conflict-")
    },
}


@pytest.mark.full
def test_every_scenario_has_a_spec() -> None:
    assert set(_SPECS) == set(_ALL)


@pytest.mark.full
@pytest.mark.parametrize("rep", range(_REPEATS), ids=lambda r: f"r{r}")
@pytest.mark.parametrize("arm", sorted(_MODES))
@pytest.mark.parametrize("scenario_id", list(_ALL))
async def test_lifecycle_classification(
    client: Hindsight, target: Target, bank_id: str, settled: SettleFn, scenario_id: str, arm: str, rep: int
) -> None:
    scenario, spec = _ALL[scenario_id], _SPECS[scenario_id]
    lifecycle = _MODES[arm] == "lifecycle"
    before = judge.USAGE.calls, judge.USAGE.input_tokens, judge.USAGE.output_tokens
    outcome = await run_scenario(
        client,
        bank_id,
        settled,
        scenario,
        adaptive_memory_mode=_MODES[arm],
        retired=retired_fn(target.url, target.api_key) if lifecycle else None,
    )
    state = await fetch_state(client, target.url, target.api_key, bank_id)
    cl = classify(scenario, spec, state, snapshot_ids={o.id for snap in outcome.snapshots for o in snap})
    c = outcome.cost
    record = RunRecord(
        scenario=scenario_id,
        arm=arm,
        rep=rep,
        bank_id=bank_id,
        kind=spec.kind,
        multi_wave=len(scenario.waves) > 1,
        judged_passed=outcome.passed,
        traps_hit=outcome.traps_hit,
        missed=outcome.missed,
        problems=outcome.problems,
        cost=RunCost(
            calls=c.calls,
            input_tokens=c.input_tokens,
            cached_tokens=c.cached_tokens,
            output_tokens=c.output_tokens,
            llm_seconds=c.llm_seconds,
            wall_seconds=c.wall_seconds,
            judge_calls=judge.USAGE.calls - before[0],
            judge_input_tokens=judge.USAGE.input_tokens - before[1],
            judge_output_tokens=judge.USAGE.output_tokens - before[2],
        ),
        lifecycle=cl,
    )
    if path := os.getenv("HINDSIGHT_EVAL_LIFECYCLE_OUTPUT"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
    # The one property gated per run: a control's true beliefs stay current. Everything
    # else (recall of the expected action, the judged checks) is a rate, read from the report.
    if spec.kind in CONTROL_KINDS:
        assert not cl.false_retirement, "\n".join(
            [f"{scenario_id}/{arm} — bank {bank_id}: truth-retired protected memories"]
            + [f"  - {e.fact_type} {e.status} via {e.retired_via}: {e.text}" for e in cl.false_retirements]
        )
