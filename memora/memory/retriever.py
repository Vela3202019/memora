import logging
import os
from typing import Any

from memora.memory.client import get_neo4j_client
from memora.memory.embedder import embed

logger = logging.getLogger(__name__)

USER_ID = os.getenv("USER_ID", "default_user")


async def search_experiences(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Vector similarity search over Experience nodes.
    Also traverses EVOKED edges to include associated emotions.
    """
    vector = await embed(query)
    client = get_neo4j_client()
    session = await client.get_async_session()

    async with session:
        result = await session.run(
            """
            CALL db.index.vector.queryNodes('experience_embedding', $top_k, $vector)
            YIELD node AS exp, score
            OPTIONAL MATCH (exp)-[:EVOKED]->(em:Emotion)
            RETURN
                exp.description   AS description,
                exp.type          AS type,
                exp.date          AS date,
                exp.location      AS location,
                exp.significance  AS significance,
                exp.context       AS context,
                collect(em.name)  AS emotions,
                score
            ORDER BY score DESC
            """,
            top_k=top_k,
            vector=vector,
        )
        return [dict(r) for r in await result.data()]


async def search_truths(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Vector similarity search over Truth nodes.
    Also traverses DISTILLED_FROM edges to include source experiences.
    """
    vector = await embed(query)
    client = get_neo4j_client()
    session = await client.get_async_session()

    async with session:
        result = await session.run(
            """
            CALL db.index.vector.queryNodes('truth_embedding', $top_k, $vector)
            YIELD node AS t, score
            OPTIONAL MATCH (t)-[r:DISTILLED_FROM]->(exp:Experience)
            RETURN
                t.content              AS content,
                t.type                 AS type,
                t.first_synthesized    AS first_synthesized,
                collect({
                    description: exp.description,
                    weight: r.contribution_weight
                })                     AS source_experiences,
                score
            ORDER BY score DESC
            """,
            top_k=top_k,
            vector=vector,
        )
        return [dict(r) for r in await result.data()]


async def get_experiences_by_emotion(emotion_name: str) -> list[dict[str, Any]]:
    """
    Find all experiences that evoked a specific emotion, ordered by intensity.
    """
    client = get_neo4j_client()
    session = await client.get_async_session()

    async with session:
        result = await session.run(
            """
            MATCH (u:User {userId: $user_id})-[:HAD]->(exp:Experience)
            MATCH (exp)-[r:EVOKED]->(em:Emotion)
            WHERE toLower(em.name) = toLower($emotion_name)
            RETURN
                exp.description  AS description,
                exp.date         AS date,
                exp.location     AS location,
                exp.significance AS significance,
                r.intensity      AS intensity
            ORDER BY r.intensity DESC
            """,
            user_id=USER_ID,
            emotion_name=emotion_name,
        )
        return [dict(r) for r in await result.data()]


async def get_emotional_patterns() -> list[dict[str, Any]]:
    """
    Return emotion frequencies and average intensities for the user,
    ordered by how often each emotion was felt.
    """
    client = get_neo4j_client()
    session = await client.get_async_session()

    async with session:
        result = await session.run(
            """
            MATCH (u:User {userId: $user_id})-[r:FELT]->(em:Emotion)
            RETURN
                em.name         AS emotion,
                em.valence      AS valence,
                COUNT(r)        AS times_felt,
                AVG(r.intensity) AS avg_intensity
            ORDER BY times_felt DESC
            """,
            user_id=USER_ID,
        )
        return [dict(r) for r in await result.data()]
