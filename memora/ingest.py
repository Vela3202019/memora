"""Ingest pipeline: discover documents from configured sources, extract, store."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from memora.config import Config
from memora.llm import get_backend
from memora.llm.base import LLMBackend
from memora.memory.diary_extractor import extract_from_document
from memora.sources import build_source

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path.home() / ".memora" / "state" / "ingest.json"
MIN_CONTENT_LENGTH = 10


class IngestState:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or DEFAULT_STATE_PATH)
        self._seen: dict = {}
        if self.path.exists():
            self._seen = json.loads(self.path.read_text())

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_processed(self, doc_id: str, text: str) -> bool:
        return self._seen.get(doc_id) == self._hash(text)

    def mark(self, doc_id: str, text: str) -> None:
        self._seen[doc_id] = self._hash(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._seen, indent=2))


async def ingest(
    config: Config,
    source_name: Optional[str] = None,
    dry_run: bool = False,
    state_path: Optional[Path] = None,
    backend: Optional[LLMBackend] = None,
) -> dict:
    backend = backend or get_backend(config.llm)
    state = IngestState(state_path)
    summary = {"total": 0, "processed": 0, "skipped": 0, "errors": 0,
               "details": []}

    sources = [
        build_source(s) for s in config.sources
        if source_name is None or s.name == source_name
    ]
    if source_name and not sources:
        logger.warning(f"No source named {source_name!r} in config")

    for source in sources:
        for doc in source.discover():
            summary["total"] += 1
            if len(doc.text.strip()) < MIN_CONTENT_LENGTH:
                summary["skipped"] += 1
                summary["details"].append(
                    {"doc": doc.doc_id, "status": "skipped", "reason": "too short"})
                continue
            if not dry_run and state.is_processed(doc.doc_id, doc.text):
                summary["skipped"] += 1
                summary["details"].append(
                    {"doc": doc.doc_id, "status": "skipped",
                     "reason": "already processed"})
                continue
            try:
                extracted = await extract_from_document(
                    doc, preamble=source.prompt_preamble, backend=backend
                )
                if extracted.get("error"):
                    # The extractor caught an error and returned an empty result;
                    # surface it instead of counting an empty doc as processed.
                    raise RuntimeError(extracted["error"])
                if not dry_run:
                    # Local import: Neo4j connection only needed when storing
                    from memora.memory.storage import MemoryStorage
                    await MemoryStorage().store_extracted_data(
                        user_id=config.user_id,
                        extracted_data={
                            "experiences": extracted.get("experiences", []),
                            "emotions": extracted.get("emotions", []),
                            "truths": extracted.get("truths", []),
                            "facts": extracted.get("facts", []),
                            "relationships": extracted.get("relationships", {}),
                            "message_metadata": {"timestamp": doc.timestamp},
                        },
                        call_id=doc.doc_id,
                    )
                    state.mark(doc.doc_id, doc.text)
                summary["processed"] += 1
                summary["details"].append({
                    "doc": doc.doc_id, "status": "processed",
                    "experiences": len(extracted.get("experiences", [])),
                    "emotions": len(extracted.get("emotions", [])),
                    "truths": len(extracted.get("truths", [])),
                })
            except Exception as e:
                logger.error(f"Error processing {doc.doc_id}: {e}")
                summary["errors"] += 1
                summary["details"].append(
                    {"doc": doc.doc_id, "status": "error", "error": str(e)})
    return summary
