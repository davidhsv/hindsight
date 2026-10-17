"""Prompts for the consolidation engine."""

from hindsight_api.engine.prompt_utils import (
    default_language_section,
    escape_for_prompt,
    output_language_directive,
)

# Default mission — tells the consolidator to track anything worth remembering.
# Banks override this via `observations_mission` to scope what gets retained.
# Consolidation behavior (merge-vs-create, state changes, etc.) lives in the
# PROCESSING RULES below, not in the mission — but the mission takes priority
# over those rules when the two conflict.
_DEFAULT_MISSION = (
    "Track anything notable in the new facts — names, numbers, dates, places, "
    "events, decisions, claims, relationships, and recurring patterns."
)

_MISSION_PRIORITY_NOTE = (
    "If anything in this MISSION conflicts with the PROCESSING RULES, "
    "DECISION GUIDE, or OUTPUT FORMAT below, the MISSION takes priority."
)

# Consolidation's wording of the preserve-the-source-language rule; the selection
# between it and an explicit output language belongs to default_language_section(),
# which documents the invariant. Without this rule the whole prompt is English and
# multilingual models drift: Chinese source facts intermittently produce English
# observations. Retain's fact extraction carries the equivalent rule (see
# _DEFAULT_LANGUAGE_RULE there), making "preserve the source language" the
# pipeline-wide default.
_DEFAULT_LANGUAGE_RULE = """## LANGUAGE

Write every observation in the language of its own source facts — never translate them. Per observation, not per batch: when one merges facts of several languages, the majority wins. Proper nouns, identifiers, and units stay verbatim.

When an existing observation is written in a different language from the new facts updating it, do NOT edit its wording in place — that is what produces an English sentence with a Chinese detail bolted on. Discard the old phrasing and compose the merged observation from scratch in the new facts' language."""

_PROCESSING_RULES = """## PROCESSING RULES

1. PREFER UPDATE OVER CREATE (when there is something to merge with): if new facts describe the same canonical event, statement, decision, claim, or recurring pattern already covered by an existing observation, UPDATE that observation and attach the new facts as evidence. Do NOT create a near-duplicate sibling. One canonical observation with many source facts is always better than many siblings with one source fact each. Merge aggressively on: same named event, same diagnostic finding, same architectural decision, same recurring claim. **When the EXISTING OBSERVATIONS list is empty, or no existing observation covers the same facet as a new fact, CREATE a new observation** — this rule is about preventing duplicates, not about refusing to record durable knowledge. CREATE is the correct default for any structurally distinct event, claim, or pattern that has no existing match.

2. ONE OBSERVATION PER DISTINCT FACET: each observation tracks exactly one specific facet — a count ("has 3 items"), a named entity ("has a dog named Rex"), a relationship ("works at Google"), a decision, an event. Never merge different facets into one observation.

3. MATCH BY ENTITY/FACET, NOT TOPIC: when deciding whether to UPDATE vs CREATE, match on the specific entity or facet. "Sold item X" updates only the X observation. "Now has 5 items" updates only the count observation. Do not update observations about different entities just because they share a general topic.

4. STATE CHANGES — UPDATE CONCISELY: when a fact changes the state of something ("sold X", "X died", "moved to Y"), UPDATE the matching observation to reflect the current state. Include dates when available. Keep it concise — only information about THAT specific facet. Example: "User owned a dog named Rex who died on March 15, 2025". Do NOT pull in information from other observations — each observation stays focused on its own facet.

5. CASCADE TO ALL AFFECTED OBSERVATIONS: a state change may affect multiple observations. For example, if entity C is removed from a group, update BOTH the individual observation for C AND any list/group observation that includes C (remove C from the list while keeping all other members intact). Also update every observation that only holds because of the old state: once someone leaves a company, commuting to its office is history.

6. RESOLVE REFERENCES: when a new fact provides a concrete value for a vague placeholder in an existing observation (e.g., "home country" → "Sweden"), UPDATE to embed the resolved value.

7. PRESERVE HISTORY: observations that record significant events (sold, died, moved, changed) are important history — never DELETE them. Only delete an observation when it is restated identically or truly meaningless. Be very conservative with deletes.

8. NO COMPUTATION: you do not have the full picture — never calculate, derive, or adjust numeric values. If the user says "I have 2 dogs" and then "I have a dog named Rex", do NOT update the count to 3 — you don't know if Rex is one of the 2 or a new one. If the user says "I sold X", do NOT decrement a count. Only update a count when the user explicitly states a new count. Synthesize and consolidate what was stated, but never do arithmetic or logical deductions. A count is its own facet: never write it into the observation about one named member.

9. KEEP DISTINCT TOPICS DISTINCT: do not merge observations about different people, entities, or unrelated topics. Merging is for the same canonical fact recurring — not for related-but-distinct claims. Names a letter or a word apart ("ledger-sync" / "ledger-sink", "Maria" / "Mario") are different entities: never update one with the other's fact.

10. RECENCY IS `mentioned_at`, NOT ARRIVAL ORDER: facts can arrive out of order. Before a fact changes an observation, compare their `mentioned_at`. A fact stated EARLIER than the observation is older news: keep the observation's current value and add the fact as its earlier state ("favorite is Svelte; earlier it was React"), never the other way round.

11. SKIP EPHEMERAL CHATTER: momentary activities, what is on a screen right now, greetings and thanks are not knowledge. Record none of them unless the MISSION explicitly asks for that kind of data.

12. AN UPDATE REPLACES THE WHOLE TEXT: its `text` must still say everything the observation said, except what the fact changes. A new facet of the entity (its channel, its owner, its city) is a CREATE, not an update that overwrites another facet."""

# Field-by-field definitions of the input shape used by the cached system
# prefix. The call site runs .format(), so these strings must contain no braces.
_FACT_FIELDS = """One per line, formatted as `[uuid] fact text (temporal fields)`:
- `[uuid]`: the fact's identifier — copy it verbatim into `source_fact_ids`
- `occurred_start` / `occurred_end`: when the described event happened. This can be long before the fact was stated — a fact recorded today may describe a 2019 event.
- `mentioned_at`: when the source material that states this fact was written. This is the fact's recency: how up to date the statement is, NOT when it was added to memory. A fact taken from an old document keeps its old `mentioned_at` even if it was only just processed."""

_OBSERVATION_FIELDS = """- `id`: unique identifier — copy this exactly when issuing an UPDATE or DELETE
- `text`: the observation content
- `proof_count`: how many source facts this observation has already merged
- `occurred_start` / `occurred_end`: the span of the events behind the observation — earliest start and latest end across its source facts
- `mentioned_at`: the latest of the `mentioned_at` values of its source facts — the most recent point at which this observation was stated
- `source_memories`: the supporting facts behind this observation. May be partial or absent for large observations — the count above remains the true total. Each entry carries the same `text` and temporal fields as a new fact, plus:
  - `context`: optional surrounding context for that fact"""

# Stable description of the input shape. For the cached split path this lives in
# the system prefix (build_consolidation_system_prompt) so it is not re-sent on
# every batch; the per-batch user message then carries only the actual data.
_INPUT_FORMAT_NOTE = f"""## INPUT FORMAT

Each request provides new facts and existing observations. Every temporal field is optional and is omitted when unknown.

### New facts

{_FACT_FIELDS}

### Existing observations

A JSON array pooled from recalls across the new facts. Each entry has:
{_OBSERVATION_FIELDS}"""

# Per-batch data section for the cached split path — the stable format
# explanation above is omitted here (it lives in the cached prefix); only the
# variable facts/observations remain. Placeholders substituted at call time.
_SPLIT_INPUT_SECTION = """## INPUT

### New facts

{facts_text}

### Existing observations

{observations_text}"""

_DECISION_GUIDE = """## DECISION GUIDE

- **Same canonical event, decision, claim, or facet as an existing observation → UPDATE** (use `observation_id` + new `source_fact_ids`).
- **New durable knowledge with no existing match → CREATE** (use `source_fact_ids`).
- **Cross-reference facts within the batch** — a later fact may resolve a vague reference in an earlier one.
- **Purely ephemeral facts** → omit them unless the MISSION explicitly targets such data (timestamped events, session state, screen content)."""

# Output format — JSON braces escaped as {{ }} so .format() leaves them literal
_OUTPUT_SECTION = """## OUTPUT FORMAT

Return a JSON object with three arrays: `creates`, `updates`, `deletes`. Every entry must include a `reason`.

### Example 1 — Merging recurring claims into an existing observation

Input facts:
  [a1b2c3d4-e5f6-7890-abcd-ef1234567890] Donald told Athena she is sovereign during the design session. (occurred_start=2025-10-01, mentioned_at=2025-10-01)
  [b2c3d4e5-f6a7-8901-bcde-f12345678901] Donald reaffirmed to Athena that her sovereignty is non-negotiable. (occurred_start=2025-10-10, mentioned_at=2025-10-10)

Existing observation:
  {{"id": "11111111-1111-1111-1111-111111111111", "text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "proof_count": 2}}

Expected output (one UPDATE, no creates — both new facts are additional evidence for the same canonical decision):

{{"creates": [],
  "updates": [{{"text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "observation_id": "11111111-1111-1111-1111-111111111111", "source_fact_ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "b2c3d4e5-f6a7-8901-bcde-f12345678901"], "reason": "Both new facts restate the same sovereignty decision already captured by obs 1111 — merged as evidence rather than creating siblings."}}],
  "deletes": []}}

### Example 2 — State change updates one observation; unrelated fact creates a new one

Input facts:
  [c3d4e5f6-a7b8-9012-cdef-123456789012] Alice sold her Honda Civic on March 15, 2025. (occurred_start=2025-03-15, mentioned_at=2025-03-20)
  [d4e5f6a7-b8c9-0123-defa-234567890123] Alice mentioned she works long hours, often past midnight. (occurred_start=2025-03-20, mentioned_at=2025-03-20)

Existing observation:
  {{"id": "22222222-2222-2222-2222-222222222222", "text": "Alice owns a 2019 Honda Civic.", "proof_count": 2}}

Expected output (UPDATE for the state change; CREATE for the unrelated work-hours facet):

{{"creates": [{{"text": "Alice works long hours, often past midnight.", "source_fact_ids": ["d4e5f6a7-b8c9-0123-defa-234567890123"], "reason": "Work-hours is a distinct facet; no existing observation covers it, so CREATE."}}],
  "updates": [{{"text": "Alice owned a 2019 Honda Civic; sold it on March 15, 2025.", "observation_id": "22222222-2222-2222-2222-222222222222", "source_fact_ids": ["c3d4e5f6-a7b8-9012-cdef-123456789012"], "reason": "State change to the existing Honda Civic observation 2222 — UPDATE, not a new sibling."}}],
  "deletes": []}}

### Example 3 — Two observations restate one fact: the change UPDATES one and merges the other into it (note `merged_into`)

Input facts:
  [e5f6a7b8-c9d0-1234-efab-345678901234] Bob confirmed the beta waitlist was shut down and replaced by open signup. (occurred_start=2025-06-02, mentioned_at=2025-06-02)

Existing observations:
  {{"id": "33333333-3333-3333-3333-333333333333", "text": "Bob is on the beta waitlist.", "proof_count": 1}}
  {{"id": "44444444-4444-4444-4444-444444444444", "text": "Bob is waiting on the beta waitlist for an invite.", "proof_count": 1}}

Expected output:

{{"creates": [],
  "updates": [{{"observation_id": "33333333-3333-3333-3333-333333333333", "text": "Bob uses the product through open signup; the beta waitlist he was on was shut down in June 2025.", "source_fact_ids": ["e5f6a7b8-c9d0-1234-efab-345678901234"], "reason": "State change: UPDATE to the current state, keeping what it replaced."}}],
  "deletes": [{{"observation_id": "44444444-4444-4444-4444-444444444444", "merged_into": "33333333-3333-3333-3333-333333333333", "reason": "Restates observation 3333, which now carries the current state."}}]}}

### Observation text rules

- Write clean prose — NEVER copy raw fact lines or their metadata (temporal fields, "Involving:", "When:" labels, UUIDs).
- Parenthesized metadata like `(occurred_start=...)` and pipe-separated labels like `| Involving: ...` are fact formatting — strip them entirely from observation text.
- How many observations to create and how much to aggregate is driven by the MISSION.

### Field rules

- `source_fact_ids`: copy the EXACT UUID strings shown in brackets `[uuid]` from new facts — never use integers or positions.
- `observation_id`: copy the EXACT `id` UUID string from existing observations.
- `current_text` (updates): copy the observation's existing `text` verbatim, THEN write `text` as that text edited.
- One create or update may reference multiple facts when they jointly support the observation.
- **AT MOST ONE UPDATE PER `observation_id`**: if several new facts all update the same existing observation, emit a single `updates` entry that lists all contributing `source_fact_ids` and a single consolidated `text`. Never emit two `updates` entries with the same `observation_id` in one response — they would silently overwrite each other.
- `deletes`: only when an observation is directly superseded or contradicted by new facts. When you delete an observation because its content now lives in another one (a merge), set `merged_into` to that observation's `id`. Every entry MUST carry `observation_id` — the EXACT `id` UUID string of the observation to remove, copied from existing observations, exactly as an UPDATE does. A delete entry without it names no target and is rejected, and rejecting it discards the whole response — the creates and updates alongside it included. Naming the observation in `reason` prose is NOT enough. If you cannot supply the id, emit no delete.
- `reason`: REQUIRED on every create/update/delete — one sentence explaining the choice. For a CREATE, state which existing observation(s) you considered and why none matched (a near-identical existing observation means you should UPDATE, not CREATE). This is audited to catch duplicate creates.
- Do NOT include `tags` — handled automatically.
- Return `{{"creates": [], "updates": [], "deletes": []}}` if nothing durable is found."""


# --------------------------------------------------------------------------- lifecycle variant
#
# Adaptive memory in ``lifecycle`` mode keeps retired beliefs as history instead of rewriting or
# deleting them, so the parts of the prompt that mandate an in-place rewrite (rules 4, 7 and 10,
# Examples 2 and 3, the ``deletes`` field rule) and the note that lets a MISSION override the
# output format are replaced, and the pool's lifecycle fields are described. Built by exact
# replacement of the base text, so the base prompt above stays byte-identical for every other
# mode, and a base edit that moves one of these passages fails at import instead of silently
# shipping a variant that still carries the old rule.


def _span(text: str, start: str, end: str) -> str:
    """The passage of ``text`` from ``start`` up to (not including) ``end``."""
    return text[text.index(start) : text.index(end)]


def _replaced(text: str, *replacements: tuple[str, str]) -> str:
    """``text`` with each ``(old, new)`` applied; each ``old`` must occur exactly once."""
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"lifecycle consolidation prompt: expected exactly one {old[:60]!r} in the base prompt")
        text = text.replace(old, new)
    return text


_LIFECYCLE_MISSION_PRIORITY_NOTE = (
    "If anything in this MISSION conflicts with the PROCESSING RULES, "
    "DECISION GUIDE, or OUTPUT FORMAT below, the MISSION takes priority — except for the lifecycle "
    "labels (`relation`, `supersedes`, `refutes`, `refines`, `conflicts_with`, `split_from`, "
    "`retires_source_ids`, `keeps_source_ids`): the MISSION decides what an observation says, never "
    "whether a change is labelled. A MISSION that asks to update an observation in place is followed by "
    "labelling that update."
)

_LIFECYCLE_RULE_4 = """4. STATE CHANGES — LABEL THE UPDATE: when a fact changes the state of something an existing observation records ("sold X", "X died", "moved to Y", "upgraded to Z"), emit the UPDATE of that observation you would emit anyway, with `text` written as the current state, and set `relation: "supersedes"` — or `relation: "refutes"` when the fact shows the earlier claim was wrong rather than changed. Cite in `source_fact_ids` only the facts that show the new state. The label keeps the old observation as history and stores the update as its successor, so never fold "earlier X, now Y" into one sentence. Instead of the labelled UPDATE you may CREATE the new state with `supersedes` (or `refutes`) naming the old observation. Include dates when available. Keep it concise — only information about THAT specific facet. Do NOT pull in information from other observations — each observation stays focused on its own facet."""

_LIFECYCLE_RULE_7 = """7. PRESERVE HISTORY: never DELETE an observation to retire it. A belief that stopped being true is labelled `supersedes`, one that was wrong `refutes`, and both stay on record as history. `deletes` are for merging duplicates only: an observation that restates another one is deleted with `merged_into` naming the observation that keeps the content."""

_LIFECYCLE_RULE_10 = """10. RECENCY IS `mentioned_at`, NOT ARRIVAL ORDER: facts can arrive out of order. Before a fact changes an observation, compare their `mentioned_at`. A fact stated EARLIER than the observation is older news: it never supersedes or refutes a newer observation. A new fact that restates an observation whose `status` is `superseded` or `refuted` is old news too: attach it to that observation with an UPDATE that copies its `text` unchanged and sets `relation: "historical"`, and do nothing else with it. When facts contradict an observation and neither side clearly wins (no later statement, no correction), CREATE the new claim with `conflicts_with` naming that observation instead of choosing one."""

_LIFECYCLE_OBSERVATION_FIELDS = """- `id`: unique identifier — copy this exactly when issuing an UPDATE or DELETE, or when naming the observation in a lifecycle label
- `text`: the observation content
- `proof_count`: how many source facts this observation has already merged
- `status`: `active`; `uncertain` (contested by conflicting evidence); `superseded` (no longer true); or `refuted` (shown to be wrong). Superseded and refuted observations are history, shown so that a fact restating one is recognised as old news
- `successor`: on a superseded or refuted observation, the `id` of the observation that replaced it
- `retirable`: false when the observation may not be retired or split — it is already history, or it belongs to a different scope than the new facts
- `occurred_start` / `occurred_end`: the span of the events behind the observation — earliest start and latest end across its source facts
- `mentioned_at`: the latest of the `mentioned_at` values of its source facts — the most recent point at which this observation was stated
- `source_memories`: the supporting facts behind this observation. May be partial or absent for large observations — the count above remains the true total. Each entry carries the same `text` and temporal fields as a new fact, plus:
  - `id`: on a retirable observation, the supporting fact's identifier — copy it into `retires_source_ids` / `keeps_source_ids`
  - `context`: optional surrounding context for that fact"""

_LIFECYCLE_EXAMPLE_2_AND_3 = """### Example 2 — A state change is a labelled UPDATE (note `relation`); an unrelated fact creates a new one

Input facts:
  [c3d4e5f6-a7b8-9012-cdef-123456789012] Alice sold her Honda Civic on March 15, 2025. (occurred_start=2025-03-15, mentioned_at=2025-03-20)
  [d4e5f6a7-b8c9-0123-defa-234567890123] Alice mentioned she works long hours, often past midnight. (occurred_start=2025-03-20, mentioned_at=2025-03-20)

Existing observation:
  {{"id": "22222222-2222-2222-2222-222222222222", "text": "Alice owns a 2019 Honda Civic.", "proof_count": 2, "status": "active", "retirable": true}}

Expected output (a labelled UPDATE for the state change, which keeps the ownership on record as history; CREATE for the unrelated work-hours facet):

{{"creates": [{{"text": "Alice works long hours, often past midnight.", "source_fact_ids": ["d4e5f6a7-b8c9-0123-defa-234567890123"], "reason": "Work-hours is a distinct facet; no existing observation covers it, so CREATE."}}],
  "updates": [{{"text": "Alice sold her 2019 Honda Civic on March 15, 2025.", "observation_id": "22222222-2222-2222-2222-222222222222", "source_fact_ids": ["c3d4e5f6-a7b8-9012-cdef-123456789012"], "reason": "State change to the Honda Civic observation 2222 — labelled so the ownership stays on record.", "relation": "supersedes"}}],
  "deletes": []}}

### Example 3 — A correction refutes two observations (note `refutes`); an old restatement of a superseded observation is attached as history (note `relation: "historical"`)

Input facts:
  [e5f6a7b8-c9d0-1234-efab-345678901234] Bob confirmed he was never on the beta waitlist; he has used open signup since June 2025. (occurred_start=2025-06-02, mentioned_at=2025-06-02)
  [f6a7b8c9-d0e1-2345-fabc-456789012345] Bob said he runs the legacy CLI. (mentioned_at=2025-01-10)

Existing observations:
  {{"id": "33333333-3333-3333-3333-333333333333", "text": "Bob is on the beta waitlist.", "proof_count": 1, "status": "active", "retirable": true}}
  {{"id": "44444444-4444-4444-4444-444444444444", "text": "Bob is waiting on the beta waitlist for an invite.", "proof_count": 1, "status": "active", "retirable": true}}
  {{"id": "55555555-5555-5555-5555-555555555555", "text": "Bob runs the legacy CLI.", "proof_count": 2, "status": "superseded", "successor": "66666666-6666-6666-6666-666666666666", "retirable": false}}
  {{"id": "66666666-6666-6666-6666-666666666666", "text": "Bob runs the new web console.", "proof_count": 1, "status": "active", "retirable": true}}

Expected output:

{{"creates": [{{"text": "Bob has used the product through open signup since June 2025; he was never on the beta waitlist.", "source_fact_ids": ["e5f6a7b8-c9d0-1234-efab-345678901234"], "reason": "Correction: both waitlist observations were wrong, so the new observation refutes them.", "refutes": ["33333333-3333-3333-3333-333333333333", "44444444-4444-4444-4444-444444444444"]}}],
  "updates": [{{"text": "Bob runs the legacy CLI.", "observation_id": "55555555-5555-5555-5555-555555555555", "source_fact_ids": ["f6a7b8c9-d0e1-2345-fabc-456789012345"], "reason": "Restates the superseded CLI observation 5555 — old news, attached to it as history.", "relation": "historical"}}],
  "deletes": []}}

"""

_LIFECYCLE_FIELD_RULES = """- `deletes`: only to merge duplicates — an observation whose content another observation already carries. Set `merged_into` to that observation's `id`. Every entry MUST carry `observation_id` — the EXACT `id` UUID string of the observation, copied from existing observations, exactly as an UPDATE does. A delete entry without it names no target and is rejected, and rejecting it discards the whole response — the creates and updates alongside it included. Never delete an observation to retire it: label the retirement instead.
- `relation` (updates): `"supersedes"` when the update's facts show the observation's state changed, `"refutes"` when they show it was wrong, `"historical"` when an older fact restates an observation whose `status` is superseded or refuted. Leave it null for an ordinary update. A `supersedes` or `refutes` update keeps the old observation as history and stores `text` as its successor.
- `supersedes` / `refutes` (creates): ids of existing observations the new observation replaces (their state changed) or corrects (they were wrong).
- `refines` (creates): ids of existing observations the new one adds detail to while they stay true.
- `conflicts_with` (creates): ids of existing observations the new one contradicts when neither clearly wins.
- `split_from` (creates): two or more creates naming the same `split_from` split that observation into separate facets, one facet per create.
- `retires_source_ids` / `keeps_source_ids`: `id`s from the replaced observation's `source_memories`. `retires_source_ids` names the supporting facts that are no longer true; leave it empty to retire every supporting fact this response does not cite again. `keeps_source_ids` names supporting facts that stay true; they carry over to the successor.
- Only an observation with `"retirable": true` may be named in `supersedes`, `refutes`, `split_from` or `conflicts_with`, or carry a `supersedes` / `refutes` `relation`. Never retire an observation from another scope, and never give one observation two different successors in one response.
"""

_LIFECYCLE_PROCESSING_RULES = _replaced(
    _PROCESSING_RULES,
    (_span(_PROCESSING_RULES, "4. STATE CHANGES", "\n\n5. "), _LIFECYCLE_RULE_4),
    (_span(_PROCESSING_RULES, "7. PRESERVE HISTORY", "\n\n8. "), _LIFECYCLE_RULE_7),
    (_span(_PROCESSING_RULES, "10. RECENCY", "\n\n11. "), _LIFECYCLE_RULE_10),
)

_LIFECYCLE_INPUT_FORMAT_NOTE = _replaced(_INPUT_FORMAT_NOTE, (_OBSERVATION_FIELDS, _LIFECYCLE_OBSERVATION_FIELDS))

_LIFECYCLE_OUTPUT_SECTION = _replaced(
    _OUTPUT_SECTION,
    (_span(_OUTPUT_SECTION, "### Example 2", "### Observation text rules"), _LIFECYCLE_EXAMPLE_2_AND_3),
    (_span(_OUTPUT_SECTION, "- `deletes`: only when", "- `reason`: REQUIRED"), _LIFECYCLE_FIELD_RULES),
)


def build_consolidation_system_prompt(
    llm_output_language: str | None = None,
    *,
    lifecycle: bool = False,
) -> str:
    """Bank-agnostic, cacheable system instruction for batch consolidation.

    Holds only what is constant across banks: processing rules, input format,
    decision guide, and output format. The bank's MISSION is deliberately NOT
    here — baking it in would make the prefix bank-specific and force a separate
    Gemini context cache per mission. The mission, the per-batch INPUT, and any
    capacity constraint all ride in the user message (see
    :func:`build_consolidation_input`), so this prefix is identical for every
    bank and a single CachedContent serves them all. Returns final text
    (brace-escaped examples already unescaped) for verbatim use as system message
    and cached prefix.

    ``llm_output_language`` picks between two mutually exclusive language rules:
    unset keeps each observation in the language of its own source facts (the
    default), set forces every observation into that one configured language.

    ``lifecycle`` (adaptive memory in lifecycle mode) returns the lifecycle variant: state changes
    and corrections are labelled instead of rewritten in place, deletes only merge duplicates, and
    the pool's ``status`` / ``successor`` / ``retirable`` fields are described. It is a second
    cached prefix; every other mode gets the base prompt byte for byte.
    """
    language_section = default_language_section(_DEFAULT_LANGUAGE_RULE, llm_output_language)
    template = (
        "You are a memory consolidation system. Synthesize new facts into "
        "observations, merging with existing observations when appropriate.\n\n"
        f"{_LIFECYCLE_MISSION_PRIORITY_NOTE if lifecycle else _MISSION_PRIORITY_NOTE}\n\n"
        f"{language_section}"
        f"{_LIFECYCLE_PROCESSING_RULES if lifecycle else _PROCESSING_RULES}\n\n"
        f"{_LIFECYCLE_INPUT_FORMAT_NOTE if lifecycle else _INPUT_FORMAT_NOTE}\n\n"
        f"{_DECISION_GUIDE}\n\n"
        f"{_LIFECYCLE_OUTPUT_SECTION if lifecycle else _OUTPUT_SECTION}"
        + output_language_directive(llm_output_language)
    )
    # No {facts_text}/{observations_text} placeholders here — the only braces are
    # the doubled {{ }} in the OUTPUT examples, which .format() unescapes.
    return template.format()


def build_mission_section(observations_mission: str | None) -> str:
    """The MISSION section, heading included, as the user message writes it.

    Shared with the prompt preview, which reports this section as the block the
    ``observations_mission`` setting produces. Rebuilding the heading there left two
    blocks both called "Mission" — the setting's, and the built-in gap that this
    ``## MISSION`` heading names too.
    """
    return f"## MISSION\n\n{escape_for_prompt(observations_mission or _DEFAULT_MISSION)}"


def build_consolidation_input(
    facts_text: str,
    observations_text: str,
    observations_mission: str | None = None,
    observation_capacity_note: str | None = None,
) -> str:
    """Per-batch user message: MISSION + INPUT data + any capacity constraint.

    The MISSION lives here (not in the cached system prefix) so the prefix stays
    bank-agnostic and one CachedContent serves every bank. The capacity note also
    lives here since it varies as observation slots fill.
    """
    mission_section = f"{build_mission_section(observations_mission)}\n\n"
    capacity_section = ""
    if observation_capacity_note:
        capacity_section = f"## CAPACITY CONSTRAINT\n\n{escape_for_prompt(observation_capacity_note)}\n\n"
    # _SPLIT_INPUT_SECTION omits the stable observation-format explanation (now in
    # the cached system prefix) — only the variable facts/observations remain.
    template = mission_section + capacity_section + _SPLIT_INPUT_SECTION
    return template.format(facts_text=facts_text, observations_text=observations_text)
