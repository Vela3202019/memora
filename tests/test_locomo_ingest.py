import asyncio
import json

import pytest

from evals.locomo.ingest import extract_conversation, _cache_path
from evals.locomo.load_locomo import (
    LOCOMO_PREAMBLE,
    LocomoConversation,
    LocomoSession,
)


def _conv():
    return LocomoConversation(
        conv_id="conv-test",
        speaker_a="Caroline",
        speaker_b="Melanie",
        sessions=[
            LocomoSession(1, "2023-05-08T13:56:00", "Caroline: Hey Mel!"),
            LocomoSession(2, "2023-06-09T10:00:00", "Melanie: Back again."),
        ],
        questions=[],
    )


class FakeBackend:
    def __init__(self):
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append((system, user))
        return {"experiences": [{"description": "x"}], "emotions": [],
                "truths": [], "relationships": {}}


def test_extract_writes_cache_and_reuses_it(tmp_path):
    conv, backend = _conv(), FakeBackend()
    first = asyncio.run(extract_conversation(conv, backend, "v2", tmp_path))
    assert len(first) == 2
    assert len(backend.calls) == 2
    # preamble made it into the prompt, session dates into the user content
    assert LOCOMO_PREAMBLE.split(".")[0] in backend.calls[0][0]
    assert "2023-05-08" in backend.calls[0][1]
    assert _cache_path("conv-test", "v2", tmp_path).exists()

    # second run: zero LLM calls, identical result
    second = asyncio.run(extract_conversation(conv, backend, "v2", tmp_path))
    assert len(backend.calls) == 2
    assert second == json.loads(json.dumps(first))


def test_cache_key_includes_prompt_version(tmp_path):
    conv, backend = _conv(), FakeBackend()
    asyncio.run(extract_conversation(conv, backend, "v2", tmp_path))
    asyncio.run(extract_conversation(conv, backend, "v1", tmp_path))
    assert len(backend.calls) == 4  # v1 did not reuse v2's cache


class ErrorBackend:
    async def complete_json(self, system, user):
        raise RuntimeError("llm down")


def test_extraction_error_raises_and_leaves_no_cache(tmp_path):
    with pytest.raises(RuntimeError):
        asyncio.run(extract_conversation(_conv(), ErrorBackend(), "v2", tmp_path))
    assert not _cache_path("conv-test", "v2", tmp_path).exists()
