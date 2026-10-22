import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memora.config import EmbeddingsConfig, load_config
from memora.memory.client import get_neo4j_client

load_dotenv()

logger = logging.getLogger(__name__)


def _embeddings_config() -> EmbeddingsConfig:
    try:
        return load_config().embeddings
    except Exception:
        return EmbeddingsConfig()  # config optional: default OpenAI


_cfg = _embeddings_config()
EMBEDDING_MODEL = _cfg.model
# Non-default models: recreate vector indexes with the model's dimension.
EMBEDDING_DIMENSIONS = 1536

_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY") or "unused",
                      base_url=_cfg.base_url)


async def embed(text: str) -> list[float]:
    """Generate a vector embedding for the given text."""
    response = await _openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def setup_vector_indexes() -> None:
    """
    Create Neo4j vector indexes for Experience and Truth nodes.
    Safe to run multiple times (uses IF NOT EXISTS).
    """
    client = get_neo4j_client()
    session = await client.get_async_session()

    async with session:
        await session.run("""
            CREATE VECTOR INDEX experience_embedding IF NOT EXISTS
            FOR (n:Experience) ON n.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        await session.run("""
            CREATE VECTOR INDEX truth_embedding IF NOT EXISTS
            FOR (n:Truth) ON n.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        await session.run("""
            CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
            FOR (n:Fact) ON n.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}
        """)

    logger.info("Vector indexes created (or already exist).")
