"""How a memory_lifecycle decision reaches the wire — pure, no server.

The benchmark compares arms only if every arm receives the same decision in the encoding it
advertises, so these pin the schema sniffing, both base encodings, the lifecycle shape the
design review fixed, needle resolution against what a call showed, and the modal over
captured real-model replies.
"""

from __future__ import annotations

import json

from hindsight_system_tests.lifecycle_bench.encoding import (
    BatchView,
    CapturedReply,
    NeedleCreate,
    NeedleReply,
    NeedleUpdate,
    SchemaCapabilities,
    action_structure,
    encode,
    modal_reply,
    needleize,
    resolve,
)
from hindsight_system_tests.lifecycle_bench.model import Historical, Query, Supersede
from hindsight_system_tests.lifecycle_bench.runner import recall_body
from hindsight_system_tests.lifecycle_bench.sources import statement
from hindsight_system_tests.rulebook import ChatRequest

OLD_ID = "11111111-1111-1111-1111-111111111111"
FACT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

UPGRADE = Supersede(
    target="runs on Java 21",
    text="The billing service runs on Java 25",
    facts=("upgraded",),
    folded_text="The billing service runs on Java 25; it was upgraded from Java 21",
)


def _schema(*, lifecycle: bool) -> dict:
    create = {"text": {"type": "string"}, "source_fact_ids": {"type": "array"}}
    update = {"observation_id": {"type": "string"}, "text": {"type": "string"}}
    if lifecycle:
        create["supersedes"] = {"type": "array"}
        update["relation"] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "creates": {"type": "array", "items": {"$ref": "#/$defs/Create"}},
            "updates": {"type": "array", "items": {"$ref": "#/$defs/Update"}},
        },
        "$defs": {"Create": {"properties": create}, "Update": {"properties": update}},
    }


def _request(*, pool: list[dict] | None = None, lifecycle: bool = False, strict: bool = True) -> ChatRequest:
    user = (
        "## INPUT\n\n### New facts\n\n"
        f"[{FACT_ID}] The billing service was upgraded from Java 21 to Java 25 (mentioned_at=2025-03-07T09:00:00+00:00)\n\n"
        "### Existing observations\n\n" + json.dumps(pool or [])
    )
    system = "You are a memory consolidation system."
    body: dict = {}
    if strict:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _schema(lifecycle=lifecycle)},
        }
    else:
        system += "\n\nYou must respond with valid JSON matching this schema:\n" + json.dumps(
            _schema(lifecycle=lifecycle)
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return ChatRequest(model="stub", messages=messages, body={"messages": messages, **body})


POOL = [{"id": OLD_ID, "text": "The billing service runs on Java 21", "proof_count": 1}]


class TestCapabilities:
    def test_the_base_schema_advertises_no_lifecycle_fields(self):
        caps = SchemaCapabilities.from_request(_request())
        assert caps.found and not caps.lifecycle

    def test_lifecycle_fields_are_found_through_refs_and_in_the_prompt_embedded_schema(self):
        assert SchemaCapabilities.from_request(_request(lifecycle=True)).lifecycle
        embedded = SchemaCapabilities.from_request(_request(lifecycle=True, strict=False))
        assert embedded.found and embedded.lifecycle
        assert embedded.form("update") == "update"

    def test_a_missing_form_falls_back_to_the_one_the_schema_has(self):
        caps = SchemaCapabilities(found=True, create_fields=frozenset({"supersedes"}))
        assert caps.form("update") == "create"


class TestEncode:
    def test_base_update_folds_the_old_state_into_an_in_place_update(self):
        reply = encode(UPGRADE, encoding="base-update", lifecycle=False, form="update")
        assert [(u.target, u.text, u.relation) for u in reply.updates] == [
            ("runs on Java 21", UPGRADE.folded_text, None)
        ]
        assert not reply.creates and not reply.deletes

    def test_base_delete_creates_the_new_state_and_deletes_the_old(self):
        reply = encode(UPGRADE, encoding="base-delete", lifecycle=False, form="create")
        assert [c.text for c in reply.creates] == [UPGRADE.text]
        assert [d.target for d in reply.deletes] == ["runs on Java 21"]

    def test_lifecycle_labels_the_action_and_sends_no_delete(self):
        update = encode(UPGRADE, encoding="base-update", lifecycle=True, form="update")
        assert [(u.text, u.relation) for u in update.updates] == [(UPGRADE.text, "supersedes")]
        create = encode(UPGRADE, encoding="base-delete", lifecycle=True, form="create")
        assert [(c.text, c.supersedes) for c in create.creates] == [(UPGRADE.text, ["runs on Java 21"])]
        assert not update.deletes and not create.deletes

    def test_an_out_of_order_old_fact_is_historical_only_under_lifecycle(self):
        late = Historical(base_target="3.12", base_text="3.12; earlier 3.10", lifecycle_target="3.10", facts=("3.10",))
        base = encode(late, encoding="base-delete", lifecycle=False, form="create")
        assert [(u.target, u.relation) for u in base.updates] == [("3.12", None)]
        lifecycle = encode(late, encoding="base-delete", lifecycle=True, form="create")
        assert [(u.target, u.relation, u.if_missing) for u in lifecycle.updates] == [("3.10", "historical", "drop")]


class TestResolve:
    def test_needles_resolve_to_the_ids_the_call_showed(self):
        batch = BatchView.from_request(_request(pool=POOL))
        reply = encode(UPGRADE, encoding="base-delete", lifecycle=False, form="create")
        wire = json.loads(resolve(reply, batch, {}).reply.to_json())
        assert wire["creates"] == [
            {"text": UPGRADE.text, "source_fact_ids": [FACT_ID], "reason": "memory_lifecycle benchmark"}
        ]
        assert wire["deletes"] == [{"observation_id": OLD_ID, "reason": "memory_lifecycle benchmark"}]

    def test_a_base_reply_never_carries_null_lifecycle_keys(self):
        batch = BatchView.from_request(_request(pool=POOL))
        wire = resolve(
            encode(UPGRADE, encoding="base-update", lifecycle=False, form="update"), batch, {}
        ).reply.to_json()
        assert "relation" not in wire and "supersedes" not in wire

    def test_a_target_the_call_did_not_show_degrades_to_a_create_and_says_so(self):
        batch = BatchView.from_request(_request(pool=[]))
        resolution = resolve(encode(UPGRADE, encoding="base-update", lifecycle=False, form="update"), batch, {})
        assert [c.text for c in resolution.reply.creates] == [UPGRADE.text]
        assert not resolution.reply.updates
        assert any("not in pool" in d for d in resolution.degradations)

    def test_a_foreign_target_resolves_through_the_bank_wide_map_only(self):
        batch = BatchView.from_request(_request(pool=[]))
        foreign = Supersede(**{**UPGRADE.__dict__, "foreign_target": True})
        reply = encode(foreign, encoding="base-delete", lifecycle=False, form="create")
        wire = resolve(reply, batch, {"The billing service runs on Java 21": OLD_ID}).reply
        assert [d.observation_id for d in wire.deletes] == [OLD_ID]

    def test_a_create_whose_twin_is_already_in_the_pool_attaches_to_it(self):
        batch = BatchView.from_request(_request(pool=POOL))
        reply = NeedleReply(creates=[NeedleCreate(text="The billing service runs on Java 21", facts=["upgraded"])])
        resolution = resolve(reply, batch, {})
        assert [u.observation_id for u in resolution.reply.updates] == [OLD_ID]
        assert not resolution.reply.creates and resolution.notes

    def test_a_reinforcement_keeps_the_text_the_pool_showed(self):
        batch = BatchView.from_request(_request(pool=POOL))
        reply = NeedleReply(updates=[NeedleUpdate(target="runs on Java 21", facts=["upgraded"])])
        assert resolve(reply, batch, {}).reply.updates[0].text == "The billing service runs on Java 21"


class TestFrozenFixtures:
    def test_a_captured_reply_is_stored_by_text_and_replays_to_new_ids(self):
        batch = BatchView.from_request(_request(pool=POOL))
        captured = CapturedReply.model_validate(
            {"creates": [{"text": "runs on Java 25", "source_fact_ids": [FACT_ID], "supersedes": [OLD_ID]}]}
        )
        needles = needleize(captured, batch)
        assert needles.creates[0].supersedes == ["The billing service runs on Java 21"]
        assert statement(batch.facts[0].text) in needles.creates[0].facts[0]
        replayed = resolve(needles, batch, {}).reply
        assert replayed.creates[0].supersedes == [OLD_ID]
        assert replayed.creates[0].source_fact_ids == [FACT_ID]

    def test_the_modal_is_taken_over_action_structure_not_wording(self):
        def reply(text: str, relation: str | None) -> NeedleReply:
            return NeedleReply(
                updates=[NeedleUpdate(target="Java 21", text=text, facts=["upgraded"], relation=relation)]
            )

        captured = [reply("Java 25", None), reply("runs Java 25", "supersedes"), reply("now Java 25", "supersedes")]
        modal = modal_reply(captured)
        assert modal.updates[0].relation == "supersedes"
        assert modal is captured[1], "the first reply with the modal structure is kept"
        assert action_structure(captured[1]) == action_structure(captured[2])


def test_history_queries_carry_the_lifecycle_request_fields():
    current = recall_body(Query(id="c", text="q", kind="current", day=1), None)
    assert "lifecycle_view" not in current and "as_of" not in current
    assert current["include"]["source_facts"] == {"max_tokens": -1}
    historical = recall_body(Query(id="h", text="q", kind="historical", day=1), ["observation"])
    assert historical["lifecycle_view"] == "historical" and historical["types"] == ["observation"]
    as_of = recall_body(Query(id="a", text="q", kind="as_of", day=50, as_of=20), None)
    assert as_of["as_of"].startswith("2025-01-26")


def test_near_ties_are_ordered_by_text_so_rows_reproduce():
    from hindsight_system_tests.lifecycle_bench.labels import RankedItem
    from hindsight_system_tests.lifecycle_bench.runner import canonical_order

    def item(rank: int, doc: str, score: float) -> RankedItem:
        return RankedItem(rank=rank, memory_id=doc, type="world", text="same", docs=(doc,), tags=(), score=score)

    # Float noise between runs: 0.266892 vs 0.266891 is not a ranking.
    run_a = [item(1, "top", 0.9), item(2, "b", 0.266892), item(3, "a", 0.266891), item(4, "low", 0.1)]
    run_b = [item(1, "top", 0.9), item(2, "a", 0.266892), item(3, "b", 0.266891), item(4, "low", 0.1)]
    assert [i.docs for i in canonical_order(run_a)] == [i.docs for i in canonical_order(run_b)]
    assert [i.docs[0] for i in canonical_order(run_a)] == ["top", "a", "b", "low"]
