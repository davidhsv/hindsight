"""A chunk the provider refuses costs that chunk, not the whole document — when the bank says so.

Anthropic (and the Claude Code and Cursor providers) refuse some requests on content-policy
grounds, and a refusal is deterministic: the same chunk earns it on every replay. By default
one refused chunk fails the retain, so no document is reported complete while missing a
chunk's facts.

`retain_skip_refused_chunks` is the bank's opt-out. The promise has three parts: the retain
succeeds, the refused chunk is still stored (so the document is whole and a reprocess sees it),
and every other chunk's facts reach recall. Off, the retain fails after asking the provider
once — a refusal is not worth a retry — but what it already committed stays: streaming retain
writes chunks as they extract and does not roll them back, so the document holds every chunk
but the refused one and recall serves their facts.

Runs on the Anthropic-provider server, since the refusal is that provider's signal. It has no
worker, so the operation-level records (`extraction_errors_count` for a skip, which
HINDSIGHT_API_FAIL_ON_EXTRACTION_ERRORS reads, and `kept_after_failure` for a failed retain)
are covered by the api-slim suite instead.

The names are unusual on purpose: the stub is shared with every other server in the session.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CHUNK_SIZE = 200
FILLER = "He keeps his notes in a green ledger by the door. " * 2
ORCHARD = "Pellinore Vask tends a pear orchard outside Ghent."
REFUSED = "Refusable passage: Pellinore Vask describes the lock on the shed."
OBOE = "Pellinore Vask plays the oboe every Sunday."
# Each paragraph fits a chunk and no two do, so the document is exactly these three chunks.
CONTENT = "\n\n".join(f"{sentence} {FILLER}" for sentence in (ORCHARD, REFUSED, OBOE))


@pytest.fixture
def _extraction(llm) -> None:
    llm.on_step("extract_facts", contains="Refusable passage").refuses()
    llm.on_step("extract_facts", contains="pear orchard").returns(
        extracted(
            fact(
                "Pellinore Vask tends a pear orchard outside Ghent",
                who="Pellinore Vask",
                where="Ghent",
                entities=["Pellinore Vask", "Ghent"],
            )
        )
    )
    llm.on_step("extract_facts", contains="oboe").returns(
        extracted(fact("Pellinore Vask plays the oboe every Sunday", who="Pellinore Vask", entities=["Pellinore Vask"]))
    )
    # The retain enqueues consolidation; another server's worker may claim it, so it is
    # declared rather than left to whichever provider happens to run it.
    llm.on_step("consolidate").returns(consolidation())


def _refused_calls(llm) -> int:
    # `anthropic-version` marks a call that came in through /v1/messages, i.e. from this server.
    return sum(1 for call in llm.calls if "Refusable passage" in call.all_text and "anthropic-version" in call.headers)


@pytest.mark.usefixtures("_extraction")
async def test_a_refused_chunk_is_skipped_and_the_rest_is_kept(wif_client, llm, bank_id):
    await wif_client.banks.update_bank_config(
        bank_id, {"updates": {"retain_chunk_size": CHUNK_SIZE, "retain_skip_refused_chunks": True}}
    )

    await wif_client.aretain(bank_id=bank_id, content=CONTENT, document_id="d1")
    await wait_until_settled(wif_client, bank_id)

    # The document is whole: the refused chunk is stored with the others, in place.
    chunks = await wif_client.documents.list_document_chunks(bank_id, "d1")
    assert [chunk.chunk_text.split(".")[0] for chunk in chunks.items] == [
        "Pellinore Vask tends a pear orchard outside Ghent",
        "Refusable passage: Pellinore Vask describes the lock on the shed",
        "Pellinore Vask plays the oboe every Sunday",
    ]
    document = await wif_client.documents.get_document(bank_id, "d1")
    assert document.original_text == CONTENT

    # Every chunk but the refused one reached recall.
    response = await wif_client.arecall(bank_id=bank_id, query="What does Pellinore Vask do?")
    assert sorted(result.text for result in response.results) == [
        "Pellinore Vask plays the oboe every Sunday | Involving: Pellinore Vask",
        "Pellinore Vask tends a pear orchard outside Ghent | Involving: Pellinore Vask",
    ]
    assert _refused_calls(llm) == 1


@pytest.mark.usefixtures("_extraction")
async def test_off_a_refused_chunk_fails_the_retain_and_keeps_what_it_committed(wif_client, llm, bank_id):
    await wif_client.banks.update_bank_config(bank_id, {"updates": {"retain_chunk_size": CHUNK_SIZE}})

    with pytest.raises(ApiException) as raised:
        await wif_client.aretain(bank_id=bank_id, content=CONTENT, document_id="d1")
    assert "refused by provider content policy" in str(raised.value)

    await wait_until_settled(wif_client, bank_id)
    chunks = await wif_client.documents.list_document_chunks(bank_id, "d1")
    assert [chunk.chunk_text.split(".")[0] for chunk in chunks.items] == [
        "Pellinore Vask tends a pear orchard outside Ghent",
        "Pellinore Vask plays the oboe every Sunday",
    ]
    response = await wif_client.arecall(bank_id=bank_id, query="What does Pellinore Vask do?")
    assert sorted(result.text for result in response.results) == [
        "Pellinore Vask plays the oboe every Sunday | Involving: Pellinore Vask",
        "Pellinore Vask tends a pear orchard outside Ghent | Involving: Pellinore Vask",
    ]
    # A refusal is permanent: the provider is asked once, not retried.
    assert _refused_calls(llm) == 1
