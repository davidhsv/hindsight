"""Serialization schema for the document transfer archive (manifest + per-document payloads).

The archive is a ZIP:

    manifest.json              -- TransferManifest
    documents/000000.json      -- TransferDocument (one file per document)
    documents/000001.json
    ...

Documents are stored under a zero-padded index rather than their id so that
arbitrary document ids (which may contain path-unsafe characters) never leak
into archive entry names. The real id lives inside each payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Bump when the archive layout changes in a backward-incompatible way.
SCHEMA_VERSION = 1

# Whole-bank transfer table classifications shared by export and import.
# Child history is always carried after its mental-model parent; operational
# history is optional and included only when the caller requests it.
CARRIED_HISTORY_TABLES = ("mental_model_history",)
HISTORY_TABLES = ("audit_log", "llm_requests")
# Logical tree carried as typed rows (not raw dicts) and restored parent-first
# after its backing mental models exist.
KNOWLEDGE_TABLES = ("knowledge_pages",)

# Operational rows that back the memories: carried with ``data`` and written to
# ``data/<table>.json``. Not replayed like documents — restored verbatim with
# their ids remapped (see importer._restore_operational_rows), because every one
# of them is keyed by a globally-unique id or references a row whose id the
# replay regenerates.
OPERATIONAL_TABLES = ("async_operations", "graph_maintenance_queue", "entity_maintenance_queue")
# Attachment bytes live in file storage, not in a column, so the rows travel with
# the blobs they point at (``blobs/``) rather than on their own.
ATTACHMENT_TABLES = ("attachments",)


@dataclass(frozen=True)
class TransferScope:
    """Which slices of a bank an archive carries.

    Three booleans rather than a component list, so a caller never has to know
    the table layout. The mapping, which the API documents verbatim:

    ``data`` — the memories and everything derived from them:
        documents, chunks, memory_units (replayed and re-embedded), consolidated
        observations, entities/links (rebuilt from the replay), attachments and
        their bytes, the curation archive (``invalidated_memory_units``), the
        async operations log, the graph/entity maintenance queues, and the
        knowledge the bank synthesized from all of it — mental models, their
        refresh history, and the knowledge-page tree over them.
    ``bank_config`` — how the bank is set up:
        the ``banks`` row (per-bank config overrides), directives and webhooks.
    ``history`` — operational history: ``audit_log`` and ``llm_requests``.

    Mental models and knowledge pages sit under ``data`` rather than
    ``bank_config`` because they are a reading of the bank's facts, not a
    setting: a mental model's ``based_on`` evidence cites memory units by id, so
    carrying it without them would restore a synthesis whose grounding resolves
    to nothing. Their refresh history follows them for the same reason, and is
    not ``history``: it is the state of a mental model, not a log about it.
    """

    data: bool = True
    bank_config: bool = True
    history: bool = False

    def __post_init__(self) -> None:
        if not (self.data or self.bank_config or self.history):
            raise ValueError("A transfer must include at least one of data, bank_config or history")


class TransferScopeManifest(BaseModel):
    """The scope an archive was produced with, recorded in the manifest.

    Defaults describe a pre-scope archive: whole-bank archives carried data and
    bank config, and named their history separately via ``includes_history``.
    """

    data: bool = True
    bank_config: bool = True
    history: bool = False


class TransferAttachment(BaseModel):
    """An attachment row plus the archive entry holding its bytes.

    ``storage_key`` is the source instance's key, which the target does not reuse
    — the key encodes the source tenant and bank (see ``bank_storage_prefix``), so
    import recomputes it and rewrites the row. ``entry`` names the ZIP member the
    bytes live in, kept opaque (an ordinal) because a storage key is caller-shaped
    and must never become a path inside the archive.
    """

    bank_id: str
    #: The document that owns it. An attachment belongs to one document, so a
    #: bank carrying the same image in two documents exports two of these.
    document_id: str
    attachment_hash: str
    short_id: str
    media_type: str
    byte_size: int
    kind: str = "image"
    #: The name this document gave it, which belongs to the reference rather than
    #: the bytes and so travels with the row.
    filename: str | None = None
    created_at: datetime | None = None
    entry: str


ObservationScopes = Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]]
BankRowsJSONEncoding = Literal["decoded", "serialized"]


class TransferCausalRelation(BaseModel):
    """A causal edge from this fact to an earlier fact in the same document.

    ``target_fact_index`` is the ordinal of the target fact within the document's
    ``facts`` list (not a database id), so it survives transfer to a new bank.
    """

    relation_type: str
    target_fact_index: int


def _absent(value: object) -> bool:
    """``exclude_if`` for the adaptive-memory fields: an archive of a bank that never used adaptive
    memory carries none of them, so it stays byte-identical to one written before they existed."""
    return value is None


#: Archive entries for the lifecycle audit trail and claim evidence (DESIGN.md §7). Written only when
#: non-empty, for the same reason; they cite memories by their source ids (``source_id``).
LIFECYCLE_RELATIONS_ENTRY = "lifecycle_relations.json"
CLAIM_EVIDENCE_ENTRY = "claim_evidence.json"


class TransferRetention(BaseModel):
    """A memory's learned retention (``memory_retention_state``), carried so an import keeps it."""

    importance: float
    stability_days: float
    last_reinforced_at: datetime
    reinforcement_count: int = 0
    access_count: int = 0
    last_recalled_at: datetime | None = None
    forgetting_exempt: bool = False
    exemption_reason: str | None = None


class TransferClaim(BaseModel):
    """A memory's claim row (``memory_claim_state``). The two pointers name source ids; the import
    rewrites them to the target's ids and drops (and counts) the ones that did not come back."""

    truth_status: str
    confidence: float | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retired_via: str | None = None
    superseded_by_id: str | None = None
    inherited_from_id: str | None = None
    scope_fingerprint: str | None = None
    reason: str | None = None
    #: When the claim was decided: invalidate → restore re-attaches edges only to claims older
    #: than the invalidation, so the import keeps the source's value.
    updated_at: datetime | None = None


class TransferClaimEvidence(BaseModel):
    """One ``memory_claim_evidence`` row: claim ``memory_id`` rests on fact ``evidence_memory_id``
    (both source ids)."""

    memory_id: str
    evidence_memory_id: str
    created_at: datetime | None = None


class TransferLifecycleRelation(BaseModel):
    """One lifecycle relation. A None endpoint is a tombstone (the memory is gone, or was not
    exported); the snapshots keep what it said."""

    source_id: str
    relation_type: str
    source_memory_id: str | None = None
    target_memory_id: str | None = None
    source_snapshot: dict = Field(default_factory=dict)
    target_snapshot: dict = Field(default_factory=dict)
    origin: str
    reason: str | None = None
    confidence: float | None = None
    evidence: dict = Field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class ImportedMemoryState:
    """The adaptive state of one imported memory, its claim pointers already on target ids."""

    unit_id: str
    claim: TransferClaim | None
    retention: TransferRetention | None


@dataclass(frozen=True)
class LifecycleRelationsRestore:
    """What restoring the lifecycle relations wrote: the source relation id -> new id of each row
    written, and how many endpoints did not resolve on the target (written as tombstones)."""

    id_map: dict[str, str] = field(default_factory=dict)
    dropped_endpoints: int = 0


@dataclass(frozen=True)
class ClaimEvidenceRestore:
    """What restoring the claim evidence wrote, and how many rows named a memory that did not come back."""

    inserted: int = 0
    dropped: int = 0


class TransferFact(BaseModel):
    """One extracted fact (memory unit) without its embedding or database id.

    Everything here is reused verbatim on import except the embedding, which is
    regenerated by the target bank's model, and the entity ids, which are
    re-resolved against the target bank by canonical name.
    """

    text: str
    # The source id is carried only so whole-bank imports can rewrite persisted
    # mental-model evidence after the fact is assigned a new target id.
    source_id: str | None = None
    fact_type: str
    context: str | None = None
    # event_date is a fallback used only when both occurred_start and
    # mentioned_at are absent, to satisfy the NOT NULL event_date column.
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    observation_scopes: ObservationScopes | None = None
    # Ordinal of the source chunk within the document (parsed from chunk_id).
    chunk_index: int | None = None
    # Entity canonical names; re-resolved against the target bank on import.
    entities: list[str] = Field(default_factory=list)
    causal_relations: list[TransferCausalRelation] = Field(default_factory=list)
    # Consolidation lifecycle timestamps, carried verbatim by a whole-bank
    # transfer so imported facts keep their exact consolidation eligibility: an
    # already-consolidated or failed fact is never re-consolidated on the target,
    # and the maintenance reconciler sees no phantom backlog. Absent in archives
    # produced before these were added (-> None), in which case the importer
    # falls back to marking only observation-referenced facts consolidated.
    created_at: datetime | None = None
    consolidated_at: datetime | None = None
    consolidation_failed_at: datetime | None = None
    # Adaptive-memory state (retention and claim), present only for a fact that has some. Events
    # (the reinforcement ledger) do not travel: after an import, evidence the source already
    # counted can reinforce the fact again.
    retention: TransferRetention | None = Field(default=None, exclude_if=_absent)
    claim: TransferClaim | None = Field(default=None, exclude_if=_absent)


class TransferChunk(BaseModel):
    """A raw text chunk of the source document, reused verbatim."""

    chunk_index: int
    chunk_text: str


class TransferObservationSource(BaseModel):
    """A reference to a source fact of an observation, by document + ordinal.

    Observations span documents and reference their source facts by unit id;
    those ids don't survive transfer, so each source is carried as the
    (document_id, fact_index) of the fact within the exported document set.
    """

    document_id: str
    fact_index: int


class TransferObservation(BaseModel):
    """A consolidated observation (``fact_type='observation'``).

    Observations are bank-level (not tied to one document), carry no embedding
    (re-generated on import) and no entity/link associations — retrieval reaches
    entities/links through their source facts. Only exported when explicitly
    requested, and only when every source resolves within the archive.
    """

    # Observations also receive fresh ids on import, so their source id is
    # needed when rewriting mental-model based_on references.
    source_id: str | None = None
    text: str
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    observation_scopes: ObservationScopes | None = None
    proof_count: int = 1
    sources: list[TransferObservationSource] = Field(default_factory=list)
    # As on TransferFact: present only for an observation with adaptive state.
    retention: TransferRetention | None = Field(default=None, exclude_if=_absent)
    claim: TransferClaim | None = Field(default=None, exclude_if=_absent)


class TransferDocument(BaseModel):
    """A single document plus its chunks and extracted facts."""

    id: str
    original_text: str | None = None
    retain_params: dict | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    chunks: list[TransferChunk] = Field(default_factory=list)
    facts: list[TransferFact] = Field(default_factory=list)


class TransferKnowledgePage(BaseModel):
    """One node of the knowledge-base tree (a folder or a page).

    Carried verbatim across a whole-bank transfer so the folder/page hierarchy,
    ``managed`` flags, and ordering survive. IDs are preserved (a page's
    ``mental_model_id`` and a node's ``parent_id`` must still resolve on the
    target), and ``bank_id`` is re-applied to the target bank on import. A folder
    has ``kind='folder'`` and ``mental_model_id=None``; a page has ``kind='page'``
    and points at its backing mental model. No derived search state lives here —
    that is on ``mental_models`` and regenerated on the target.
    """

    id: str
    parent_id: str | None = None
    kind: Literal["folder", "page"]
    name: str
    mental_model_id: str | None = None
    sort_order: int = 0
    managed: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TransferManifest(BaseModel):
    """Top-level archive descriptor (``manifest.json``).

    The bank-level fields default to a documents-only archive so older
    document-only archives (and the document import path) keep parsing
    unchanged; ``export_bank`` populates them for a whole-bank archive.
    """

    schema_version: int = SCHEMA_VERSION
    source_bank_id: str
    exported_at: datetime | None = None
    document_count: int = 0
    fact_count: int = 0
    observation_count: int = 0
    mental_model_count: int = 0
    knowledge_page_count: int = 0
    # "documents" = doc/fact/observation subset; "bank" = whole-bank export
    # (also carries bank config, mental models, directives, webhooks).
    archive_type: Literal["documents", "bank"] = "documents"
    directive_count: int = 0
    webhook_count: int = 0
    # True when --include-history carried audit_log / llm_requests. Kept as the
    # wire name it has always had; `scope.history` says the same thing for
    # archives produced with an explicit scope.
    includes_history: bool = False
    # What the producer was asked to carry. Absent on pre-scope archives, which
    # the importer reads as "data + bank_config" (what a whole-bank export was).
    scope: TransferScopeManifest | None = None
    attachment_count: int = 0
    operation_count: int = 0
    invalidated_memory_count: int = 0
    # How JSON/JSONB values in bank/history row files were represented by the
    # producing connection. Absent on legacy v1 archives; import treats those as
    # decoded because the released producer was the codec-enabled admin CLI.
    bank_rows_json_encoding: BankRowsJSONEncoding | None = None
