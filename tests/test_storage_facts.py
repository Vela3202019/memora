import asyncio
import itertools

import pytest

import memora.memory.storage as storage_module
from memora.memory.storage import MemoryStorage


class FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class FakeSession:
    def __init__(self, client):
        self.client = client

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.client.calls.append((query, params))
        if "RETURN elementId(f) as fact_id" in query:
            return FakeResult({"fact_id": f"fact-{next(self.client.ids)}"})
        if "RETURN elementId(exp) as exp_id" in query:
            return FakeResult({"exp_id": f"exp-{next(self.client.ids)}"})
        if "RETURN elementId(t) as truth_id" in query:
            return FakeResult({"truth_id": f"truth-{next(self.client.ids)}"})
        return FakeResult(None)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.ids = itertools.count()

    async def get_async_session(self):
        return FakeSession(self)

    def queries_with(self, fragment):
        return [(q, p) for q, p in self.calls if fragment in q]


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()

    async def fake_embed(text):
        return [0.0]

    monkeypatch.setattr(storage_module, "get_neo4j_client", lambda: fake)
    monkeypatch.setattr(storage_module, "embed", fake_embed)
    return fake


def store(storage, predicate, obj, valid_from):
    return asyncio.run(storage.store_fact(
        user_id="u1", subject="user", predicate=predicate, object=obj,
        statement=f"I {predicate} {obj}", stated_by="user", valid_from=valid_from,
    ))


def test_single_valued_fact_closes_previous_one(client):
    storage = MemoryStorage()
    old_id = store(storage, "lives_in", "Boston", "2026-01-01")
    new_id = store(storage, "lives_in", "Seattle", "2026-06-01")

    supersedes = client.queries_with("SUPERSEDES")
    assert len(supersedes) == 2  # one attempt per single-valued store
    query, params = supersedes[1]
    # The old current fact gets closed at the new fact's valid_from and linked
    assert "SET old.valid_to = date($valid_from)" in query
    assert "CREATE (new)-[:SUPERSEDES]->(old)" in query
    assert "old.valid_to IS NULL" in query
    assert "old.object <> $object" in query
    # Only this user's facts are candidates
    assert "(:User {userId: $userId})-[:KNOWS]->(old:Fact" in query
    assert params == {
        "fact_id": new_id, "userId": "u1", "subject": "user",
        "predicate": "lives_in", "object": "Seattle", "valid_from": "2026-06-01",
    }
    assert new_id != old_id
    # The old fact is kept: nothing is ever deleted
    assert not any("DELETE" in q for q, _ in client.calls)


def test_new_fact_is_current_and_known_by_user(client):
    storage = MemoryStorage()
    store(storage, "lives_in", "Boston", "2026-01-01")

    query, params = client.queries_with("CREATE (f:Fact")[0]
    assert "valid_to: null" in query
    assert "CREATE (u)-[:KNOWS]->(f)" in query
    assert params["valid_from"] == "2026-01-01"


def test_multi_valued_fact_does_not_supersede(client):
    storage = MemoryStorage()
    store(storage, "likes", "jazz", "2026-01-01")
    store(storage, "likes", "hiking", "2026-06-01")

    assert len(client.queries_with("CREATE (f:Fact")) == 2
    assert client.queries_with("SUPERSEDES") == []


def test_store_extracted_data_routes_facts_with_experience_links(client):
    storage = MemoryStorage()
    extracted = {
        "experiences": [
            {"description": "I met Ken for lunch", "type": "friendship"},
            {"description": "I moved into my new flat", "type": "personal_growth"},
        ],
        "emotions": [],
        "truths": [],
        "facts": [
            {"subject": "user", "predicate": "has_friend", "object": "Ken",
             "statement": "I have a friend named Ken", "stated_by": "user"},
            {"subject": "user", "predicate": "lives_in", "object": "Seattle",
             "statement": "I live in Seattle", "stated_by": "user"},
        ],
        "relationships": {
            "fact_learned_in_experience": [
                {"fact_index": 0, "experience_index": 0},
                {"fact_index": 1, "experience_index": 1},
                {"fact_index": 5, "experience_index": 0},  # out of range: ignored
            ],
        },
        "message_metadata": {"timestamp": "2026-07-05T09:30:00"},
    }
    asyncio.run(storage.store_extracted_data("u1", extracted, call_id="doc-1"))

    exp_ids = [f"exp-{i}" for i in range(2)]
    fact_ids = ["fact-2", "fact-3"]

    created = client.queries_with("CREATE (f:Fact")
    assert [p["object"] for _, p in created] == ["Ken", "Seattle"]
    # valid_from is the document's date
    assert all(p["valid_from"] == "2026-07-05" for _, p in created)

    links = [(p["fact_id"], p["experience_id"])
             for _, p in client.queries_with("MERGE (f)-[:LEARNED_IN]->(exp)")]
    assert links == [(fact_ids[0], exp_ids[0]), (fact_ids[1], exp_ids[1])]


def test_store_extracted_data_without_facts(client):
    storage = MemoryStorage()
    extracted = {
        "experiences": [], "emotions": [],
        "truths": [{"content": "I love mornings", "type": "preference"}],
        "relationships": {},
        "message_metadata": {"timestamp": "2026-07-05T09:30:00"},
    }
    asyncio.run(storage.store_extracted_data("u1", extracted))

    assert client.queries_with("CREATE (f:Fact") == []
    assert len(client.queries_with("CREATE (t:Truth")) == 1
