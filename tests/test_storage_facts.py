import asyncio
import itertools

import pytest

import memora.memory.storage as storage_module
from memora.memory.storage import MemoryStorage, dedupe_facts


class InMemoryGraphStorage(MemoryStorage):
    """MemoryStorage whose graph writes/reads go to in-memory dicts."""

    def __init__(self):
        super().__init__()
        self.ids = itertools.count()
        self.facts = {}        # id -> fact properties (incl. userId)
        self.supersedes = []   # (new_id, old_id)
        self.learned_in = []   # (fact_id, experience_id)

    async def store_user(self, user_id, properties=None):
        pass

    async def store_experience(self, user_id, description, timestamp, **kwargs):
        return f"exp-{next(self.ids)}"

    async def _create_fact(self, user_id, subject, predicate, object, statement,
                           stated_by, valid_from, embedding):
        fact_id = f"fact-{next(self.ids)}"
        self.facts[fact_id] = {
            "userId": user_id, "subject": subject, "predicate": predicate,
            "object": object, "valid_from": valid_from, "valid_to": None,
        }
        return fact_id

    async def _current_facts(self, user_id):
        return [
            {"id": i, **{k: f[k] for k in ("subject", "predicate", "object", "valid_from")}}
            for i, f in self.facts.items()
            if f["userId"] == user_id and f["valid_to"] is None
        ]

    async def _close_fact(self, fact_id, valid_to, superseded_by):
        self.facts[fact_id]["valid_to"] = valid_to
        self.supersedes.append((superseded_by, fact_id))

    async def _link_fact_to_experience(self, fact_id, experience_id):
        self.learned_in.append((fact_id, experience_id))

    def current(self, predicate):
        return {i: f for i, f in self.facts.items()
                if f["predicate"] == predicate and f["valid_to"] is None}


@pytest.fixture
def graph(monkeypatch):
    async def fake_embed(text):
        return [0.0]

    monkeypatch.setattr(storage_module, "get_neo4j_client", lambda: None)
    monkeypatch.setattr(storage_module, "embed", fake_embed)
    return InMemoryGraphStorage()


def store(graph, predicate, obj, valid_from, user_id="u1"):
    return asyncio.run(graph.store_fact(
        user_id=user_id, subject="user", predicate=predicate, object=obj,
        statement=f"I {predicate} {obj}", stated_by="user", valid_from=valid_from,
    ))


def ingest_entry(graph, facts, date, links=()):
    asyncio.run(graph.store_extracted_data("u1", {
        "experiences": [{"description": f"entry {date}"}],
        "emotions": [], "truths": [], "facts": facts,
        "relationships": {"fact_learned_in_experience": list(links)},
        "message_metadata": {"timestamp": f"{date}T09:00:00"},
    }))


def lives_in(city):
    return {"subject": "user", "predicate": "lives_in", "object": city,
            "statement": f"I live in {city}", "stated_by": "user"}


def test_single_valued_fact_closes_previous_one(graph):
    boston = store(graph, "lives_in", "Boston", "2026-01-01")
    seattle = store(graph, "lives_in", "Seattle", "2026-06-01")

    assert list(graph.current("lives_in")) == [seattle]
    assert graph.facts[boston]["valid_to"] == "2026-06-01"
    assert graph.supersedes == [(seattle, boston)]
    assert len(graph.facts) == 2  # nothing deleted


def test_new_fact_is_current_with_document_date(graph):
    fact_id = store(graph, "lives_in", "Boston", "2026-01-01")

    assert graph.facts[fact_id]["valid_from"] == "2026-01-01"
    assert graph.facts[fact_id]["valid_to"] is None
    assert graph.facts[fact_id]["userId"] == "u1"


def test_multi_valued_fact_does_not_supersede(graph):
    store(graph, "likes", "jazz", "2026-01-01")
    store(graph, "likes", "hiking", "2026-06-01")

    assert len(graph.current("likes")) == 2
    assert graph.supersedes == []


def test_other_users_facts_are_not_superseded(graph):
    other = store(graph, "lives_in", "Boston", "2026-01-01", user_id="u2")
    store(graph, "lives_in", "Seattle", "2026-06-01")

    assert graph.facts[other]["valid_to"] is None
    assert graph.supersedes == []


def test_older_fact_ingested_late_does_not_close_newer_one(graph):
    seattle = store(graph, "lives_in", "Seattle", "2026-06-01")
    boston = store(graph, "lives_in", "Boston", "2026-01-01")

    assert list(graph.current("lives_in")) == [seattle]
    assert graph.facts[boston]["valid_to"] == "2026-06-01"
    assert graph.supersedes == [(seattle, boston)]
    for fact in graph.facts.values():
        assert fact["valid_to"] is None or fact["valid_to"] >= fact["valid_from"]


def test_restated_fact_keeps_one_current_fact_with_earliest_date(graph):
    ingest_entry(graph, [lives_in("Seattle")], "2026-01-10")
    ingest_entry(graph, [lives_in("Seattle")], "2026-03-10")

    current = list(graph.current("lives_in").values())
    assert [(f["object"], f["valid_from"]) for f in current] == [("Seattle", "2026-01-10")]

    ingest_entry(graph, [lives_in("Boston")], "2026-05-10")

    assert len(graph.supersedes) == 1
    current = list(graph.current("lives_in").values())
    assert [(f["object"], f["valid_from"]) for f in current] == [("Boston", "2026-05-10")]


def test_store_extracted_data_links_facts_to_experiences(graph):
    asyncio.run(graph.store_extracted_data("u1", {
        "experiences": [
            {"description": "I met Ken for lunch", "type": "friendship"},
            {"description": "I moved into my new flat", "type": "personal_growth"},
        ],
        "emotions": [], "truths": [],
        "facts": [
            {"subject": "user", "predicate": "has_friend", "object": "Ken",
             "statement": "I have a friend named Ken", "stated_by": "user"},
            lives_in("Seattle"),
        ],
        "relationships": {
            "fact_learned_in_experience": [
                {"fact_index": 0, "experience_index": 0},
                {"fact_index": 1, "experience_index": 1},
                {"fact_index": 5, "experience_index": 0},  # out of range: ignored
            ],
        },
        "message_metadata": {"timestamp": "2026-07-05T09:30:00"},
    }, call_id="doc-1"))

    by_object = {f["object"]: i for i, f in graph.facts.items()}
    assert all(f["valid_from"] == "2026-07-05" for f in graph.facts.values())
    assert graph.learned_in == [(by_object["Ken"], "exp-0"), (by_object["Seattle"], "exp-1")]


def test_dedupe_collapses_repeats_and_remaps_links():
    ken = {"subject": "user", "predicate": "has_friend", "object": "Ken"}
    seattle = {"subject": "user", "predicate": "lives_in", "object": "Seattle"}
    facts, links = dedupe_facts(
        [ken, seattle, dict(ken)],
        [{"fact_index": 0, "experience_index": 0},
         {"fact_index": 1, "experience_index": 1},
         {"fact_index": 2, "experience_index": 2}],
        current=set(),
    )

    assert facts == [ken, seattle]
    assert links == [{"fact_index": 0, "experience_index": 0},
                     {"fact_index": 1, "experience_index": 1},
                     {"fact_index": 0, "experience_index": 2}]


def test_dedupe_drops_facts_already_current():
    ken = {"subject": "user", "predicate": "has_friend", "object": "Ken"}
    seattle = {"subject": "user", "predicate": "lives_in", "object": "Seattle"}
    facts, links = dedupe_facts(
        [seattle, ken],
        [{"fact_index": 0, "experience_index": 0},
         {"fact_index": 1, "experience_index": 0}],
        current={("user", "lives_in", "Seattle")},
    )

    assert facts == [ken]
    assert links == [{"fact_index": 0, "experience_index": 0}]
