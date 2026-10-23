"""LoCoMo ingestion: wipe graph -> extract sessions (cached) -> store.

Extraction is the expensive stage (~300 turns x 10 conversations of LLM
calls), and the eval wipes Neo4j between conversations, so raw extraction
JSON is cached per (conversation, prompt_version, extractor model). Re-runs
with an unchanged extraction config rebuild the graph from cache with zero LLM
calls.

The wipe only ever runs against a separate eval-only Neo4j: use_eval_neo4j()
points this process (and the MCP servers ask() spawns) at
MEMORA_EVAL_NEO4J_URI, and wipe_graph() refuses any other database or one that
holds users other than the eval user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from memora.llm.base import LLMBackend
from memora.memory.diary_extractor import extract_from_document
from memora.sources.base import SourceDocument

from evals.locomo.load_locomo import LOCOMO_PREAMBLE, LocomoConversation

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(conv_id: str, prompt_version: str, model_id: str,
                cache_dir: Path = CACHE_DIR) -> Path:
    model_tag = re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
    return cache_dir / f"{conv_id}-{prompt_version}-{model_tag}.json"


async def extract_conversation(
    conv: LocomoConversation,
    backend: LLMBackend,
    model_id: str,
    prompt_version: str = "v3",
    cache_dir: Path = CACHE_DIR,
) -> list:
    """Extract all sessions of one conversation; returns list of extraction dicts.

    Serves from cache when present; otherwise extracts and writes the cache
    BEFORE any storage happens, so a mid-run crash never loses paid LLM work.
    """
    cache_file = _cache_path(conv.conv_id, prompt_version, model_id, cache_dir)
    if cache_file.exists():
        logger.info("cache hit: %s", cache_file.name)
        return json.loads(cache_file.read_text())

    extractions = []
    for session in conv.sessions:
        doc = SourceDocument(
            doc_id=f"{conv.conv_id}-session_{session.session_id}",
            text=session.transcript,
            timestamp=session.date_time,
            source_type="conversation",
            metadata={},
        )
        extracted = await extract_from_document(
            doc, preamble=LOCOMO_PREAMBLE,
            prompt_version=prompt_version, backend=backend,
        )
        if extracted.get("error"):
            raise RuntimeError(
                f"extraction failed for {doc.doc_id}: {extracted['error']}"
            )
        extractions.append(extracted)
        logger.info(
            "extracted %s: %d exp / %d emo / %d truths",
            doc.doc_id,
            len(extracted.get("experiences", [])),
            len(extracted.get("emotions", [])),
            len(extracted.get("truths", [])),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(extractions, ensure_ascii=False, indent=1)
    )
    return extractions


def use_eval_neo4j() -> None:
    """Point every Neo4j client in this process at the eval-only instance.

    Must run before any Neo4j client is created. Child processes (the memora
    MCP server ask() spawns) inherit the overridden environment.
    """
    load_dotenv()
    eval_uri = os.getenv("MEMORA_EVAL_NEO4J_URI")
    if not eval_uri:
        raise RuntimeError(
            "MEMORA_EVAL_NEO4J_URI is required: the LoCoMo eval wipes its "
            "graph, so it only runs against a separate eval-only Neo4j"
        )
    if eval_uri == os.getenv("NEO4J_URI"):
        raise RuntimeError(
            "MEMORA_EVAL_NEO4J_URI equals NEO4J_URI; refusing to wipe the "
            "Neo4j that holds your memories"
        )
    os.environ["NEO4J_URI"] = eval_uri
    os.environ["NEO4J_USERNAME"] = os.environ["MEMORA_EVAL_NEO4J_USERNAME"]
    os.environ["NEO4J_PASSWORD"] = os.environ["MEMORA_EVAL_NEO4J_PASSWORD"]


async def wipe_graph(user_id: str) -> None:
    """Delete ALL nodes and relationships of the eval-only graph."""
    from memora.memory.client import get_neo4j_client

    client = get_neo4j_client()
    eval_uri = os.getenv("MEMORA_EVAL_NEO4J_URI")
    if not eval_uri or client.uri != eval_uri:
        raise RuntimeError(
            f"refusing to wipe {client.uri}: not the eval-only Neo4j "
            "(MEMORA_EVAL_NEO4J_URI); call use_eval_neo4j() first"
        )
    session = await client.get_async_session()
    async with session:
        result = await session.run(
            "MATCH (u:User) WHERE u.userId IS NULL OR u.userId <> $userId "
            "RETURN u.userId AS userId LIMIT 1",
            userId=user_id,
        )
        foreign = await result.single()
        if foreign is not None:
            raise RuntimeError(
                f"refusing to wipe {client.uri}: it holds User "
                f"{foreign['userId']!r}, not only the eval user {user_id!r}"
            )
        await session.run("MATCH (n) DETACH DELETE n")


async def store_extractions(
    extractions: list, user_id: str
) -> None:
    from memora.memory.storage import MemoryStorage

    storage = MemoryStorage()
    for extracted in extractions:
        meta = extracted.get("source_metadata", {})
        await storage.store_extracted_data(
            user_id=user_id,
            extracted_data={
                "experiences": extracted.get("experiences", []),
                "emotions": extracted.get("emotions", []),
                "truths": extracted.get("truths", []),
                "facts": extracted.get("facts", []),
                "relationships": extracted.get("relationships", {}),
                "message_metadata": {"timestamp": meta.get("timestamp")},
            },
            call_id=meta.get("doc_id"),
        )


async def ingest_conversation(
    conv: LocomoConversation,
    backend: LLMBackend,
    model_id: str,
    prompt_version: str = "v3",
    user_id: str = "locomo",
    cache_dir: Path = CACHE_DIR,
) -> None:
    """Full per-conversation ingest: extract (cached) -> wipe graph -> store."""
    extractions = await extract_conversation(
        conv, backend, model_id, prompt_version, cache_dir
    )
    await wipe_graph(user_id)
    await store_extractions(extractions, user_id)
