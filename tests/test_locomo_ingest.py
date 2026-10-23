import asyncio
import json

import pytest

from evals.locomo import ingest
from evals.locomo.ingest import (
    _cache_path,
    extract_conversation,
    store_extractions,
    use_eval_neo4j,
    wipe_graph,
)
from evals.locomo.load_locomo import (
    LOCOMO_PREAMBLE,
    LocomoConversation,
    LocomoSession,
)

MODEL = "claude-subscription:claude-sonnet-5"


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
    first = asyncio.run(extract_conversation(conv, backend, MODEL, "v2", tmp_path))
    assert len(first) == 2
    assert len(backend.calls) == 2
    # preamble made it into the prompt, session dates into the user content
    assert LOCOMO_PREAMBLE.split(".")[0] in backend.calls[0][0]
    assert "2023-05-08" in backend.calls[0][1]
    assert _cache_path("conv-test", "v2", MODEL, tmp_path).exists()

    # second run: zero LLM calls, identical result
    second = asyncio.run(extract_conversation(conv, backend, MODEL, "v2", tmp_path))
    assert len(backend.calls) == 2
    assert second == json.loads(json.dumps(first))


def test_cache_key_includes_prompt_version(tmp_path):
    conv, backend = _conv(), FakeBackend()
    asyncio.run(extract_conversation(conv, backend, MODEL, "v2", tmp_path))
    asyncio.run(extract_conversation(conv, backend, MODEL, "v1", tmp_path))
    assert len(backend.calls) == 4  # v1 did not reuse v2's cache


def test_cache_key_includes_extractor_model(tmp_path):
    conv, backend = _conv(), FakeBackend()
    asyncio.run(extract_conversation(conv, backend, MODEL, "v3", tmp_path))
    asyncio.run(extract_conversation(
        conv, backend, "openai-compatible:gpt-5.2", "v3", tmp_path))
    assert len(backend.calls) == 4  # gpt-5.2 did not reuse Sonnet's cache


class ErrorBackend:
    async def complete_json(self, system, user):
        raise RuntimeError("llm down")


def test_extraction_error_raises_and_leaves_no_cache(tmp_path):
    with pytest.raises(RuntimeError):
        asyncio.run(extract_conversation(_conv(), ErrorBackend(), MODEL, "v2", tmp_path))
    assert not _cache_path("conv-test", "v2", MODEL, tmp_path).exists()


def test_store_extractions_passes_facts(monkeypatch):
    stored = []

    class FakeStorage:
        async def store_extracted_data(self, user_id, extracted_data, call_id):
            stored.append(extracted_data)

    import memora.memory.storage as storage_mod
    monkeypatch.setattr(storage_mod, "MemoryStorage", FakeStorage)
    fact = {"subject": "Caroline", "predicate": "adopted", "object": "a dog"}
    asyncio.run(store_extractions([{"facts": [fact]}], "locomo"))
    assert stored[0]["facts"] == [fact]


class FakeResult:
    def __init__(self, record):
        self.record = record

    async def single(self):
        return self.record


class FakeSession:
    def __init__(self, foreign_user):
        self.foreign_user = foreign_user
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.queries.append(query)
        if "RETURN u.userId" in query:
            return FakeResult(
                {"userId": self.foreign_user} if self.foreign_user else None)
        return FakeResult(None)


class FakeClient:
    def __init__(self, uri, foreign_user=None):
        self.uri = uri
        self.session = FakeSession(foreign_user)

    async def get_async_session(self):
        return self.session


def _deleted(client):
    return any("DETACH DELETE" in q for q in client.session.queries)


@pytest.fixture
def fake_client(monkeypatch):
    import memora.memory.client as client_mod

    def install(client):
        monkeypatch.setattr(client_mod, "get_neo4j_client", lambda: client)
        return client
    return install


def test_use_eval_neo4j_refuses_when_eval_uri_missing(monkeypatch):
    monkeypatch.setattr(ingest, "load_dotenv", lambda: None)
    monkeypatch.delenv("MEMORA_EVAL_NEO4J_URI", raising=False)
    monkeypatch.setenv("NEO4J_URI", "bolt://mine:7687")
    with pytest.raises(RuntimeError, match="MEMORA_EVAL_NEO4J_URI"):
        use_eval_neo4j()
    assert ingest.os.environ["NEO4J_URI"] == "bolt://mine:7687"


def test_use_eval_neo4j_refuses_when_eval_uri_equals_neo4j_uri(monkeypatch):
    monkeypatch.setattr(ingest, "load_dotenv", lambda: None)
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_URI", "bolt://mine:7687")
    monkeypatch.setenv("NEO4J_URI", "bolt://mine:7687")
    with pytest.raises(RuntimeError, match="equals NEO4J_URI"):
        use_eval_neo4j()


def test_use_eval_neo4j_points_neo4j_env_at_eval_instance(monkeypatch):
    monkeypatch.setattr(ingest, "load_dotenv", lambda: None)
    monkeypatch.setenv("NEO4J_URI", "bolt://mine:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "me")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_URI", "bolt://eval:7687")
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_USERNAME", "eval")
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_PASSWORD", "evalpw")
    use_eval_neo4j()
    env = ingest.os.environ
    assert (env["NEO4J_URI"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"]) \
        == ("bolt://eval:7687", "eval", "evalpw")


def test_wipe_refuses_non_eval_database(monkeypatch, fake_client):
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_URI", "bolt://eval:7687")
    client = fake_client(FakeClient("bolt://mine:7687"))
    with pytest.raises(RuntimeError, match="not the eval-only Neo4j"):
        asyncio.run(wipe_graph("locomo"))
    assert client.session.queries == []


def test_wipe_refuses_without_eval_uri(monkeypatch, fake_client):
    monkeypatch.delenv("MEMORA_EVAL_NEO4J_URI", raising=False)
    client = fake_client(FakeClient("bolt://mine:7687"))
    with pytest.raises(RuntimeError, match="not the eval-only Neo4j"):
        asyncio.run(wipe_graph("locomo"))
    assert client.session.queries == []


def test_wipe_refuses_when_graph_holds_other_users(monkeypatch, fake_client):
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_URI", "bolt://eval:7687")
    client = fake_client(FakeClient("bolt://eval:7687", foreign_user="vela"))
    with pytest.raises(RuntimeError, match="'vela'"):
        asyncio.run(wipe_graph("locomo"))
    assert not _deleted(client)


def test_wipe_deletes_eval_only_graph(monkeypatch, fake_client):
    monkeypatch.setenv("MEMORA_EVAL_NEO4J_URI", "bolt://eval:7687")
    client = fake_client(FakeClient("bolt://eval:7687"))
    asyncio.run(wipe_graph("locomo"))
    assert _deleted(client)
