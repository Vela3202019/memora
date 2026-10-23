
import logging
from typing import Dict, List, Any, Optional
from memora.memory.client import get_neo4j_client
from memora.memory.embedder import embed

logger = logging.getLogger(__name__)


def dedupe_facts(
    facts: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    current: set,
) -> tuple:
    """
    Deterministic post-extraction dedup of facts, keyed on
    (subject, predicate, object):

    - repeats within one extraction collapse into the first occurrence
    - facts matching one of the user's current facts (`current`) are dropped,
      so the existing node is kept

    Returns (facts, links) with each link's fact_index remapped to the kept
    list; links to dropped facts are removed.
    """
    kept: List[Dict[str, Any]] = []
    first_index: Dict[tuple, int] = {}
    index_map: Dict[int, int] = {}
    for i, fact in enumerate(facts):
        key = (fact.get("subject"), fact.get("predicate"), fact.get("object"))
        if key in current:
            continue
        if key not in first_index:
            first_index[key] = len(kept)
            kept.append(fact)
        index_map[i] = first_index[key]
    remapped = [
        {**rel, "fact_index": index_map[rel.get("fact_index")]}
        for rel in links
        if rel.get("fact_index") in index_map
    ]
    return kept, remapped


class MemoryStorage:
    
    def __init__(self):
        self.client = get_neo4j_client()
    
    async def store_user(self, user_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Create or update User node.
        
        Args:
            user_id: Unique user identifier
            properties: Additional user properties
        """
        session = await self.client.get_async_session()
        
        try:
            async with session:
                props = properties or {}
                props["userId"] = user_id
                
                query = """
                MERGE (u:User {userId: $userId})
                SET u += $properties
                SET u.lastUpdated = datetime()
                RETURN u
                """
                
                await session.run(query, userId=user_id, properties=props)
                logger.info(f"Stored User node: {user_id}")
                
        except Exception as e:
            logger.error(f"Error storing user: {e}")
            raise
    
    async def store_experience(
        self,
        user_id: str,
        description: str,
        timestamp: str,
        experience_type: Optional[str] = None,
        location: Optional[str] = None,
        significance: Optional[str] = None,
        people_involved: Optional[List[str]] = None,
        context: Optional[str] = None,
        call_id: Optional[str] = None
    ) -> str:
        """
        Create Experience node and link to User.
        Date is derived from the transcript timestamp.
        Returns Neo4j elementId for the created experience.
        
        Args:
            user_id: User identifier
            description: The experience description
            timestamp: When experience was mentioned (from transcript timestamp, ISO format)
            experience_type: Type/category of experience (e.g., career, relationship, hobbies, etc.)
            location: Location/place where experience occurred
            significance: Why it mattered
            people_involved: List of people who were there
            context: Other contextual information
            call_id: Call identifier
            
        Returns:
            The Neo4j elementId of the created experience
        """
        session = await self.client.get_async_session()
        
        # Extract date from timestamp (YYYY-MM-DD format)
        # Timestamp format: "2025-08-05T22:00:32.317Z" -> "2025-08-05"
        date = timestamp[:10] if timestamp and len(timestamp) >= 10 else None
        
        embedding = await embed(description) if description else None

        try:
            async with session:
                query = """
                MATCH (u:User {userId: $userId})
                CREATE (exp:Experience {
                    description: $description,
                    type: $experience_type,
                    date: $date,
                    createdAt: datetime($timestamp),
                    location: $location,
                    significance: $significance,
                    people_involved: $people_involved,
                    context: $context,
                    callId: $call_id,
                    embedding: $embedding
                })
                CREATE (u)-[:HAD {timestamp: datetime($timestamp)}]->(exp)
                RETURN elementId(exp) as exp_id
                """

                result = await session.run(
                    query,
                    userId=user_id,
                    description=description,
                    experience_type=experience_type,
                    date=date,
                    timestamp=timestamp,
                    location=location,
                    significance=significance,
                    people_involved=people_involved,
                    context=context,
                    call_id=call_id,
                    embedding=embedding,
                )
                
                record = await result.single()
                exp_id = record["exp_id"] if record else None
                
                logger.debug(f"Stored experience: {description[:50]}... (id: {exp_id})")
                return exp_id
                
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
            raise
    
    
    async def store_emotion(
        self,
        user_id: str,
        name: str,
        intensity: float,
        valence: str,
        context: str,
        timestamp: str,
        experience_id: Optional[str] = None,
        call_id: Optional[str] = None
    ) -> None:
        """
        Create or merge Emotion node and link to User and Experience.
        Emotions are reused - same emotion name creates/updates the same node.
        
        Args:
            user_id: User identifier
            name: Emotion name (e.g., "joy", "peace", "aliveness")
            intensity: Intensity level (0-1 scale)
            valence: positive/negative/neutral
            context: Context of emotion
            timestamp: When emotion was expressed
            experience_id: Optional experience ID this emotion was evoked by
            call_id: Call identifier
        """
        session = await self.client.get_async_session()
        
        try:
            async with session:
                # MERGE emotion by name (reuse same emotion node)
                emotion_name = name.lower()  # Normalize to lowercase
                query = """
                MATCH (u:User {userId: $userId})
                MERGE (em:Emotion {name: $emotion_name})
                ON CREATE SET em.createdAt = datetime()
                SET em.valence = $valence
                MERGE (u)-[r:FELT {
                    when: datetime($timestamp)
                }]->(em)
                SET r.intensity = $intensity
                SET r.context = $context
                SET r.callId = $call_id
                RETURN em
                """
                
                await session.run(
                    query,
                    userId=user_id,
                    emotion_name=emotion_name,
                    intensity=intensity,
                    valence=valence,
                    context=context,
                    timestamp=timestamp,
                    call_id=call_id
                )
                
                # Link emotion to experience if provided
                if experience_id:
                    await self._link_emotion_to_experience(
                        emotion_name,
                        experience_id,
                        intensity,
                        timestamp
                    )
                
                logger.debug(f"Stored emotion: {name} (intensity: {intensity}, valence: {valence})")
                
        except Exception as e:
            logger.error(f"Error storing emotion: {e}")
            raise
    
    async def _link_emotion_to_experience(
        self,
        emotion_name: str,
        experience_id: str,
        intensity: float,
        timestamp: str
    ) -> None:
        """Link an Emotion to an Experience (experience evoked emotion)."""
        session = await self.client.get_async_session()
        
        try:
            async with session:
                query = """
                MATCH (exp:Experience)
                WHERE elementId(exp) = $experience_id
                MATCH (em:Emotion {name: $emotion_name})
                MERGE (exp)-[r:EVOKED]->(em)
                SET r.intensity = $intensity
                SET r.timestamp = datetime($timestamp)
                """
                
                await session.run(
                    query,
                    experience_id=experience_id,
                    emotion_name=emotion_name,
                    intensity=intensity,
                    timestamp=timestamp
                )
        except Exception as e:
            logger.debug(f"Could not link emotion to experience: {e}")
    
    async def store_truth(
        self,
        user_id: str,
        content: str,
        truth_type: str,
        first_synthesized: str,
        experience_ids: Optional[List[str]] = None
    ) -> str:
        """
        Create Truth node (PersonalSemanticMemory) and link to User and Experiences.
        Returns Neo4j elementId for the created truth.

        Args:
            user_id: User identifier
            content: The truth content (e.g., "I thrive in solitary exploration at night")
            truth_type: Type of truth (pattern, principle, value, etc.)
            first_synthesized: When this truth was first synthesized (ISO date string)
            experience_ids: List of experience IDs this truth was distilled from

        Returns:
            The Neo4j elementId of the created truth
        """
        session = await self.client.get_async_session()

        embedding = await embed(content) if content else None

        try:
            async with session:
                # Create truth node
                query = """
                MATCH (u:User {userId: $userId})
                CREATE (t:Truth {
                    content: $content,
                    type: $truth_type,
                    first_synthesized: date($first_synthesized),
                    createdAt: datetime(),
                    embedding: $embedding
                })
                CREATE (u)-[:GOT {
                    discovered_at: date($first_synthesized)
                }]->(t)
                RETURN elementId(t) as truth_id
                """

                result = await session.run(
                    query,
                    userId=user_id,
                    content=content,
                    truth_type=truth_type,
                    first_synthesized=first_synthesized,
                    embedding=embedding,
                )
                
                record = await result.single()
                t_id = record["truth_id"] if record else None
                
                # Link to experiences
                if experience_ids and t_id:
                    for exp_id in experience_ids:
                        await self._link_truth_to_experience(
                            t_id,
                            exp_id,
                            first_synthesized
                        )
                
                logger.debug(f"Stored truth: {content[:50]}... (type: {truth_type}, id: {t_id})")
                return t_id
                
        except Exception as e:
            logger.error(f"Error storing truth: {e}")
            raise
    
    # Predicates where a new value replaces the old one (someone lives in ONE
    # city). Multi-valued predicates (likes, has_pet, has_friend...) must NOT
    # auto-supersede — a new liked thing doesn't end the old one.
    SINGLE_VALUED_PREDICATES = {
        "lives_in", "works_at", "works_as", "is_named", "has_partner",
    }

    async def store_fact(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        object: str,
        statement: str,
        stated_by: Optional[str],
        valid_from: str,
        experience_id: Optional[str] = None,
    ) -> str:
        """
        Create a Fact node (persisting attribute/relationship as a
        subject-predicate-object triple), bi-temporal:

        - valid_from = the source document's date (never LLM-inferred)
        - valid_to = null while current
        - For single-valued predicates, an existing current fact with the same
          (subject, predicate), a different object and valid_from <= the new
          fact's is closed (valid_to = new fact's valid_from) and linked via
          (new)-[:SUPERSEDES]->(old). If a current fact is newer than the new
          one (an older document ingested late), the new fact is closed instead
          at the earliest such fact's valid_from, linked via
          (newer)-[:SUPERSEDES]->(new). Nothing is ever deleted.

        The create, current-fact read and closes run in one transaction.
        Returns the Neo4j elementId of the created fact.
        """
        fact = {
            "subject": subject, "predicate": predicate, "object": object,
            "statement": statement, "stated_by": stated_by,
            "embedding": await embed(statement) if statement else None,
        }
        session = await self.client.get_async_session()

        try:
            async with session:
                fact_id = await session.execute_write(
                    self._write_fact, user_id, fact, valid_from
                )
        except Exception as e:
            logger.error(f"Error storing fact: {e}")
            raise

        logger.debug(f"Stored fact: {statement[:50]}... (id: {fact_id})")
        if experience_id and fact_id:
            await self._link_fact_to_experience(fact_id, experience_id)
        return fact_id

    async def _write_facts(
        self,
        tx,
        user_id: str,
        facts: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        valid_from: str,
    ) -> tuple:
        """
        Transaction function: dedupe one extraction's facts against the
        user's current facts and store the rest. A duplicate of a current fact
        dated earlier than it moves that fact's valid_from back to this date.
        Returns (fact_ids, links) with links remapped to fact_ids.
        """
        current = {
            (f["subject"], f["predicate"], f["object"]): f
            for f in await self._current_facts(tx, user_id)
        }
        for fact in facts:
            existing = current.get((fact.get("subject"), fact.get("predicate"), fact.get("object")))
            if existing and valid_from < existing["valid_from"]:
                await self._backdate_fact(tx, existing["id"], valid_from)
                existing["valid_from"] = valid_from
        facts, links = dedupe_facts(facts, links, set(current))
        fact_ids = [await self._write_fact(tx, user_id, fact, valid_from) for fact in facts]
        return fact_ids, links

    async def _write_fact(
        self, tx, user_id: str, fact: Dict[str, Any], valid_from: str
    ) -> Optional[str]:
        """Transaction function: create one fact and apply supersession."""
        subject, predicate, object = fact.get("subject"), fact.get("predicate"), fact.get("object")
        fact_id = await self._create_fact(tx, user_id, fact, valid_from)

        if predicate in self.SINGLE_VALUED_PREDICATES and fact_id:
            newer = []
            for old in await self._current_facts(tx, user_id):
                if (old["id"] == fact_id or old["subject"] != subject
                        or old["predicate"] != predicate or old["object"] == object):
                    continue
                if old["valid_from"] <= valid_from:
                    await self._close_fact(tx, old["id"], valid_from, fact_id)
                else:
                    newer.append(old)
            if newer:
                successor = min(newer, key=lambda f: f["valid_from"])
                await self._close_fact(tx, fact_id, successor["valid_from"], successor["id"])
        return fact_id

    async def _backdate_fact(self, tx, fact_id: str, valid_from: str) -> None:
        """
        Move a fact's valid_from back and re-close the facts it SUPERSEDES at
        that date, never below a predecessor's own valid_from.
        """
        await self._set_valid_from(tx, fact_id, valid_from)
        for predecessor in await self._predecessors(tx, fact_id):
            await self._set_valid_to(
                tx, predecessor["id"], max(valid_from, predecessor["valid_from"])
            )

    async def _create_fact(
        self, tx, user_id: str, fact: Dict[str, Any], valid_from: str
    ) -> Optional[str]:
        """Create a current Fact node known by the user; return its elementId."""
        result = await tx.run(
            """
            MATCH (u:User {userId: $userId})
            CREATE (f:Fact {
                subject: $subject,
                predicate: $predicate,
                object: $object,
                statement: $statement,
                stated_by: $stated_by,
                valid_from: date($valid_from),
                valid_to: null,
                createdAt: datetime(),
                embedding: $embedding
            })
            CREATE (u)-[:KNOWS]->(f)
            RETURN elementId(f) as fact_id
            """,
            userId=user_id,
            subject=fact.get("subject"),
            predicate=fact.get("predicate"),
            object=fact.get("object"),
            statement=fact.get("statement"),
            stated_by=fact.get("stated_by"),
            valid_from=valid_from,
            embedding=fact.get("embedding"),
        )
        record = await result.single()
        return record["fact_id"] if record else None

    async def _current_facts(self, tx, user_id: str) -> List[Dict[str, Any]]:
        """The user's current facts (valid_to IS NULL), valid_from as an ISO date string."""
        result = await tx.run(
            """
            MATCH (:User {userId: $userId})-[:KNOWS]->(f:Fact)
            WHERE f.valid_to IS NULL
            RETURN elementId(f) AS id, f.subject AS subject,
                   f.predicate AS predicate, f.object AS object,
                   toString(f.valid_from) AS valid_from
            """,
            userId=user_id,
        )
        return [dict(r) for r in await result.data()]

    async def _predecessors(self, tx, fact_id: str) -> List[Dict[str, Any]]:
        """Facts this fact SUPERSEDES, valid_from as an ISO date string."""
        result = await tx.run(
            """
            MATCH (f:Fact)-[:SUPERSEDES]->(old:Fact) WHERE elementId(f) = $fact_id
            RETURN elementId(old) AS id, toString(old.valid_from) AS valid_from
            """,
            fact_id=fact_id,
        )
        return [dict(r) for r in await result.data()]

    async def _close_fact(self, tx, fact_id: str, valid_to: str, superseded_by: str) -> None:
        """Set a fact's valid_to and link (superseded_by)-[:SUPERSEDES]->(fact)."""
        await tx.run(
            """
            MATCH (old:Fact) WHERE elementId(old) = $fact_id
            MATCH (new:Fact) WHERE elementId(new) = $superseded_by
            SET old.valid_to = date($valid_to)
            CREATE (new)-[:SUPERSEDES]->(old)
            """,
            fact_id=fact_id,
            valid_to=valid_to,
            superseded_by=superseded_by,
        )

    async def _set_valid_from(self, tx, fact_id: str, valid_from: str) -> None:
        await tx.run(
            "MATCH (f:Fact) WHERE elementId(f) = $fact_id SET f.valid_from = date($valid_from)",
            fact_id=fact_id,
            valid_from=valid_from,
        )

    async def _set_valid_to(self, tx, fact_id: str, valid_to: str) -> None:
        await tx.run(
            "MATCH (f:Fact) WHERE elementId(f) = $fact_id SET f.valid_to = date($valid_to)",
            fact_id=fact_id,
            valid_to=valid_to,
        )

    async def _link_fact_to_experience(self, fact_id: str, experience_id: str) -> None:
        """Link a Fact to the Experience it was learned in."""
        session = await self.client.get_async_session()

        try:
            async with session:
                await session.run(
                    """
                    MATCH (f:Fact) WHERE elementId(f) = $fact_id
                    MATCH (exp:Experience) WHERE elementId(exp) = $experience_id
                    MERGE (f)-[:LEARNED_IN]->(exp)
                    """,
                    fact_id=fact_id,
                    experience_id=experience_id,
                )
        except Exception as e:
            logger.debug(f"Could not link fact to experience: {e}")

    async def _link_truth_to_experience(
        self,
        truth_id: str,
        experience_id: str,
        synthesis_date: str,
        contribution_weight: Optional[float] = None
    ) -> None:
        """Link a Truth to an Experience (truth distilled from experience)."""
        session = await self.client.get_async_session()
        
        try:
            async with session:
                query = """
                MATCH (t:Truth)
                WHERE elementId(t) = $truth_id
                MATCH (exp:Experience)
                WHERE elementId(exp) = $experience_id
                MERGE (t)-[r:DISTILLED_FROM]->(exp)
                SET r.synthesis_date = date($synthesis_date)
                SET r.contribution_weight = $contribution_weight
                """
                
                await session.run(
                    query,
                    truth_id=truth_id,
                    experience_id=experience_id,
                    synthesis_date=synthesis_date,
                    contribution_weight=contribution_weight
                )
        except Exception as e:
            logger.debug(f"Could not link truth to experience: {e}")
    
    async def store_extracted_data(
        self,
        user_id: str,
        extracted_data: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> None:
        """
        Store all extracted data from a single message.
        Handles relationships by array index mapping.
        
        Args:
            user_id: User identifier
            extracted_data: Full extraction result from memory_extractor
            call_id: Optional call/session identifier
        """
        timestamp = extracted_data.get("message_metadata", {}).get("timestamp")
        
        # Ensure user exists
        await self.store_user(user_id)
        
        # Store experiences first and collect their IDs (index -> ID mapping)
        experience_ids = []
        experiences = extracted_data.get("experiences", [])
        for experience in experiences:
            exp_id = await self.store_experience(
                user_id=user_id,
                description=experience.get("description"),
                timestamp=timestamp,
                experience_type=experience.get("type"),
                location=experience.get("location"),
                significance=experience.get("significance"),
                people_involved=experience.get("people_involved"),
                context=experience.get("context"),
                call_id=call_id
            )
            experience_ids.append(exp_id)
        
        # Store emotions and collect their names (index -> name mapping)
        emotion_names = []
        emotions = extracted_data.get("emotions", [])
        for emotion in emotions:
            emotion_name = emotion.get("name", "").lower()
            emotion_names.append(emotion_name)
            
            await self.store_emotion(
                user_id=user_id,
                name=emotion.get("name"),
                intensity=emotion.get("intensity"),
                valence=emotion.get("valence"),
                context=emotion.get("context"),
                timestamp=timestamp,
                experience_id=None,  # Will link via relationships
                call_id=call_id
            )
        
        # Store truths and collect their IDs (index -> ID mapping)
        truth_ids = []
        truths = extracted_data.get("truths", [])
        for truth in truths:
            truth_id = await self.store_truth(
                user_id=user_id,
                content=truth.get("content"),
                truth_type=truth.get("type"),
                first_synthesized=truth.get("first_synthesized"),
                experience_ids=[]  # Will link via relationships
            )
            truth_ids.append(truth_id)

        # Store facts and collect their IDs (index -> ID mapping).
        # valid_from is the document's date, never an LLM-inferred one.
        facts = [
            {**fact, "embedding": await embed(fact["statement"]) if fact.get("statement") else None}
            for fact in extracted_data.get("facts", [])
        ]
        session = await self.client.get_async_session()
        async with session:
            fact_ids, fact_links = await session.execute_write(
                self._write_facts,
                user_id,
                facts,
                extracted_data.get("relationships", {}).get("fact_learned_in_experience", []),
                timestamp[:10],
            )

        # Process relationships by index mapping
        relationships = extracted_data.get("relationships", {})
        
        # Link experiences to emotions (experience_evoked_emotion)
        for rel in relationships.get("experience_evoked_emotion", []):
            exp_idx = rel.get("experience_index")
            emo_idx = rel.get("emotion_index")
            intensity = rel.get("intensity")
            
            if exp_idx is not None and emo_idx is not None:
                if 0 <= exp_idx < len(experience_ids) and 0 <= emo_idx < len(emotion_names):
                    exp_id = experience_ids[exp_idx]
                    emotion_name = emotion_names[emo_idx]
                    
                    await self._link_emotion_to_experience(
                        emotion_name,
                        exp_id,
                        intensity,
                        timestamp
                    )
        
        # Link truths to experiences (truth_distilled_from_experience)
        for rel in relationships.get("truth_distilled_from_experience", []):
            truth_idx = rel.get("truth_index")
            exp_indices = rel.get("experience_indices")
            contribution_weights = rel.get("contribution_weights")
            
            if truth_idx is not None and 0 <= truth_idx < len(truth_ids):
                truth_id = truth_ids[truth_idx]
                synthesis_date = rel.get("synthesis_date")
                
                for i, exp_idx in enumerate(exp_indices):
                    if 0 <= exp_idx < len(experience_ids):
                        exp_id = experience_ids[exp_idx]
                        weight = contribution_weights[i] if contribution_weights and i < len(contribution_weights) else None
                        
                        await self._link_truth_to_experience(
                            truth_id,
                            exp_id,
                            synthesis_date,
                            weight
                        )

        # Link facts to experiences (fact_learned_in_experience)
        for rel in fact_links:
            fact_idx = rel.get("fact_index")
            exp_idx = rel.get("experience_index")

            if fact_idx is not None and exp_idx is not None:
                if 0 <= fact_idx < len(fact_ids) and 0 <= exp_idx < len(experience_ids):
                    await self._link_fact_to_experience(
                        fact_ids[fact_idx],
                        experience_ids[exp_idx]
                    )

        logger.info(f"Stored all extracted data for message at {timestamp}")

