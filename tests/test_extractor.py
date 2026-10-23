import asyncio

from memora.llm.base import LLMBackend
from memora.memory.diary_extractor import build_prompt, extract_from_document
from memora.prompts.diary_extraction_prompt import PROMPT_VERSIONS
from memora.sources.base import SourceDocument


class FakeBackend(LLMBackend):
    def __init__(self):
        self.seen = []

    async def complete_json(self, system, user):
        self.seen.append((system, user))
        return {"experiences": [], "emotions": [], "truths": [],
                "relationships": {}}


def test_build_prompt_core_unchanged_without_preamble():
    assert build_prompt("", "v2") == PROMPT_VERSIONS["v2"]


def test_build_prompt_prepends_preamble():
    p = build_prompt("PREAMBLE TEXT", "v2")
    assert p.startswith("PREAMBLE TEXT")
    assert p.endswith(PROMPT_VERSIONS["v2"])


def test_extract_from_document_routes_through_backend():
    doc = SourceDocument(doc_id="/x/a.md", text="I went hiking.",
                         timestamp="2026-07-01T09:00:00",
                         source_type="markdown", metadata={})
    backend = FakeBackend()
    result = asyncio.run(extract_from_document(doc, backend=backend))
    assert result["source_metadata"]["doc_id"] == "/x/a.md"
    assert result["source_metadata"]["timestamp"] == "2026-07-01T09:00:00"
    (system, user) = backend.seen[0]
    assert system == PROMPT_VERSIONS["v3"]
    assert "I went hiking." in user
    assert "2026-07-01" in user
