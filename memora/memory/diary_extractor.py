"""
Diary extractor module for extracting memories from diary entries.
Processes markdown diary files and extracts experiences, emotions, and truths.
"""

import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from memora.prompts.diary_extraction_prompt import PROMPT_VERSIONS
from memora.llm.base import LLMBackend
from memora.llm.openai_compat import OpenAICompatBackend
from memora.sources.base import SourceDocument

load_dotenv()

logger = logging.getLogger(__name__)


def parse_diary_filename(filename: str) -> Dict[str, str]:
    """
    Parse diary filename to extract UUID and date.
    
    Filename format: [UUID]-[YYYY-MM-DD-HH-MM-SS].md
    Example: [34B84EC6-BCB1-4652-9EFA-6BC0D9B17698]-[2026-01-04-13-01-20].md
    
    Returns:
        Dict with 'diary_id', 'date', 'datetime'
    """
    # Extract UUID and datetime from filename
    pattern = r'\[([A-F0-9-]+)\]-\[(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\]\.md'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Invalid diary filename format: {filename}")
    
    uuid = match.group(1)
    year, month, day, hour, minute, second = match.groups()[1:]
    
    date_str = f"{year}-{month}-{day}"
    datetime_str = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    
    return {
        "diary_id": uuid,
        "date": date_str,
        "datetime": datetime_str
    }


def load_diary(file_path: str) -> Dict[str, Any]:
    """
    Load a diary markdown file.
    
    Returns:
        Dict with 'diary_id', 'date', 'datetime', 'content'
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Diary file not found: {file_path}")
    
    # Parse filename for metadata
    metadata = parse_diary_filename(path.name)
    
    # Read content
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    return {
        "diary_id": metadata["diary_id"],
        "date": metadata["date"],
        "datetime": metadata["datetime"],
        "content": content
    }


def build_prompt(preamble: str, prompt_version: str = "v3") -> str:
    core = PROMPT_VERSIONS[prompt_version]
    return f"{preamble}\n\n{core}" if preamble else core


async def extract_from_document(
    doc: SourceDocument,
    preamble: str = "",
    prompt_version: str = "v3",
    backend: Optional[LLMBackend] = None,
) -> Dict[str, Any]:
    """Extract memories from any SourceDocument via the configured backend."""
    backend = backend or OpenAICompatBackend(base_url=None, model="gpt-5.2")
    date = doc.timestamp[:10]
    user_content = (
        f"ENTRY TO ANALYZE (Source: {doc.source_type}, "
        f"ID: {doc.doc_id}, Date: {date}):\n\n{doc.text}"
    )
    try:
        extracted = await backend.complete_json(
            build_prompt(preamble, prompt_version), user_content
        )
    except Exception as e:
        logger.error(f"Error extracting from {doc.doc_id}: {e}")
        extracted = {"experiences": [], "emotions": [], "truths": [],
                     "relationships": {}, "error": str(e)}
    extracted["source_metadata"] = {
        "doc_id": doc.doc_id,
        "timestamp": doc.timestamp,
        "source_type": doc.source_type,
        "content_preview": doc.text[:200],
    }
    return extracted


async def extract_from_diary(
    diary_id: str,
    diary_date: str,
    content: str,
    prompt_version: str = "v3",
    model: str = "gpt-5.2",
    backend: Optional[LLMBackend] = None,
) -> Dict[str, Any]:
    """Back-compat wrapper used by evals and the legacy entrypoint."""
    doc = SourceDocument(
        doc_id=diary_id, text=content,
        timestamp=f"{diary_date}T00:00:00", source_type="markdown",
        metadata={},
    )
    backend = backend or OpenAICompatBackend(base_url=None, model=model)
    result = await extract_from_document(
        doc, prompt_version=prompt_version, backend=backend
    )
    result["diary_metadata"] = {
        "diary_id": diary_id,
        "diary_date": diary_date,
        "content_preview": content[:200] if content else "",
    }
    return result

