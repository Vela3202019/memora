# Memory Schema Documentation

## Overview

This system models personal long-term memory using a graph database (Neo4j). The memory structure consists of two main types:

1. **Episodic Memory (Experiences)**: Specific events, moments, or occurrences the user has had
2. **Personal Semantic Memory (Truths)**: Abstracted self-knowledge and principles distilled from experiences

## Memory Architecture

### Episodic Memory
Episodic memory captures concrete, specific experiences with temporal and spatial context. Each experience represents a single event or moment - multiple experiences are never mixed into a single entry.

### Personal Semantic Memory
Personal semantic memory contains abstracted truths about the self that are:
- Distilled from multiple experiences
- Generalizable across contexts
- Subjectively bound to the individual
- Examples: patterns, values, principles, needs, preferences

## Graph Schema

### Node Types

#### User
The central node representing the person whose memory is being stored.

**Properties:**
- `userId` (string, required): Unique user identifier
- `lastUpdated` (datetime): Last update timestamp
- Additional custom properties can be added

**Example:**
```cypher
(u:User {userId: "user_001"})
```

#### Experience
Represents a specific episodic memory - a single event, moment, or occurrence.

**Properties:**
- `description` (string, required): What happened (concise, vivid)
- `type` (string, optional): Category/type (e.g., "career", "relationship", "hobbies", "location", "family", "health")
- `date` (date, optional): Date when experience occurred (derived from transcript timestamp)
- `createdAt` (datetime, required): When the experience was recorded
- `location` (string, optional): Where it happened
- `significance` (string, optional): Why it mattered
- `people_involved` (list[string], optional): Who was there
- `context` (string, optional): Other contextual information
- `callId` (string, optional): Identifier for the call/session where this was mentioned

**Important:** Each experience captures only ONE specific event. Multiple distinct experiences are stored as separate nodes.

**Example:**
```cypher
(exp:Experience {
  description: "Hiking alone at Green Gulch at night",
  type: "hobbies",
  date: date("2025-03-20"),
  location: "Green Gulch, Marin",
  significance: "felt alive",
  context: "weekend trip to meditation center"
})
```

#### Emotion
Represents an emotional state. Emotions are reused - the same emotion name refers to the same node.

**Properties:**
- `name` (string, required): Emotion name (e.g., "joy", "peace", "aliveness", "anxiety")
- `valence` (string, required): "positive", "negative", or "neutral"
- `createdAt` (datetime): When the emotion node was first created

**Example:**
```cypher
(em:Emotion {
  name: "aliveness",
  valence: "positive"
})
```

#### Truth
Represents personal semantic memory - abstracted self-knowledge distilled from experiences.

**Properties:**
- `content` (string, required): The truth statement (e.g., "I thrive in solitary exploration at night")
- `type` (string, required): Type of truth - "pattern", "value", "principle", "need", or "preference"
- `first_synthesized` (date, required): When this truth was first synthesized
- `createdAt` (datetime, required): When the truth node was created

**Example:**
```cypher
(t:Truth {
  content: "I thrive in solitary exploration at night",
  type: "pattern",
  first_synthesized: date("2025-12-01")
})
```

#### Fact
A persisting attribute or relationship stored as a subject–predicate–object triple.

**Properties:**
- `subject` (string, required): Who or what the fact is about (e.g., "user")
- `predicate` (string, required): The relation (e.g., "lives_in", "has_friend")
- `object` (string, required): The value (e.g., "Seattle")
- `statement` (string, required): The fact as a sentence
- `stated_by` (string, optional): Who stated it
- `valid_from` (date, required): The source document's date (never LLM-inferred)
- `valid_to` (date, optional): Null while the fact is current; set when it is superseded
- `createdAt` (datetime, required): When the fact node was created

For single-valued predicates (`lives_in`, `works_at`, `works_as`, `is_named`, `has_partner`) a new value closes the user's previous current fact instead of deleting it. Other predicates (e.g. `likes`, `has_friend`) accumulate. Facts are never deleted.

**Example:**
```cypher
(f:Fact {
  subject: "user",
  predicate: "lives_in",
  object: "Seattle",
  statement: "I live in Seattle",
  stated_by: "user",
  valid_from: date("2026-06-01"),
  valid_to: null
})
```

### Relationships

#### User -[:HAD]-> Experience
Indicates that a user had a specific experience.

**Properties:**
- `timestamp` (datetime, required): When the experience was mentioned in the conversation

**Example:**
```cypher
(u:User)-[:HAD {timestamp: datetime("2025-08-05T22:00:32.317Z")}]->(exp:Experience)
```

#### User -[:FELT]-> Emotion
Indicates that a user felt a specific emotion at a particular time.

**Properties:**
- `when` (datetime, required): When the emotion was expressed
- `intensity` (float, required): Intensity level (0-1 scale)
- `context` (string, required): What triggered or relates to this emotion
- `callId` (string, optional): Identifier for the call/session

**Example:**
```cypher
(u:User)-[:FELT {
  when: datetime("2025-03-20T22:30:00"),
  intensity: 0.95,
  context: "while hiking alone at night",
  callId: "call_001"
}]->(em:Emotion)
```

#### Experience -[:EVOKED]-> Emotion
Indicates that an experience evoked a specific emotion.

**Properties:**
- `intensity` (float, required): Intensity of the emotion during this experience
- `timestamp` (datetime, required): When this relationship was established

**Example:**
```cypher
(exp:Experience)-[:EVOKED {
  intensity: 0.95,
  timestamp: datetime("2025-03-20T22:30:00")
}]->(em:Emotion)
```

#### User -[:GOT]-> Truth
Indicates that a user discovered/learned a truth about themselves.

**Properties:**
- `discovered_at` (date, required): When the truth was discovered

**Example:**
```cypher
(u:User)-[:GOT {
  discovered_at: date("2025-12-01")
}]->(t:Truth)
```

#### User -[:KNOWS]-> Fact
Indicates that a user knows a fact.

**Example:**
```cypher
(u:User)-[:KNOWS]->(f:Fact)
```

#### Fact -[:LEARNED_IN]-> Experience
Indicates the experience a fact was learned in.

**Example:**
```cypher
(f:Fact)-[:LEARNED_IN]->(exp:Experience)
```

#### Fact -[:SUPERSEDES]-> Fact
Points from a new value of a single-valued fact to the value it replaced. The replaced fact's `valid_to` is set to the new fact's `valid_from`.

**Example:**
```cypher
(new:Fact {object: "Seattle"})-[:SUPERSEDES]->(old:Fact {object: "Boston", valid_to: date("2026-06-01")})
```

#### Truth -[:DISTILLED_FROM]-> Experience
Indicates that a truth was synthesized/distilled from specific experiences.

**Properties:**
- `synthesis_date` (date, required): When the synthesis occurred
- `contribution_weight` (float, optional): Weight indicating how much this experience contributed to the truth

**Example:**
```cypher
(t:Truth)-[:DISTILLED_FROM {
  synthesis_date: date("2025-12-01"),
  contribution_weight: 0.35
}]->(exp:Experience)
```

## Visual Structure

```
User (Vela)
├─[:HAD]─> Experience 1 (Green Gulch hiking)
│ └─[:EVOKED]─> Emotion (aliveness)
│
├─[:HAD]─> Experience 2 (Cambridge walk)
│ └─[:EVOKED]─> Emotion (aliveness)
│
├─[:HAD]─> Experience 3 (Williamsburg walk)
│ └─[:EVOKED]─> Emotion (aliveness)
│
├─[:FELT]─> Emotion (aliveness)
│
└─[:GOT]─> Truth ("I thrive in solitary exploration at night")
    ├─[:DISTILLED_FROM]─> Experience 1
    ├─[:DISTILLED_FROM]─> Experience 2
    └─[:DISTILLED_FROM]─> Experience 3
```

## Data Flow

1. **Extraction**: Conversational messages are analyzed using LLM to extract:
   - Experiences (episodic memories)
   - Emotions
   - Truths (explicit self-knowledge)
   - Relationships between them (using array indices)

2. **Storage**: Extracted data is stored in Neo4j:
   - Experiences are created with date derived from transcript timestamp
   - Emotions are merged by name (reused across experiences)
   - Truths are created and linked to contributing experiences
   - Relationships are established using Neo4j elementIds

3. **Relationships**: Connections are mapped using array indices from extraction:
   - `experience_evoked_emotion`: Maps which experiences evoked which emotions
   - `truth_distilled_from_experience`: Maps which experiences contributed to which truths

## Key Design Principles

1. **One Experience Per Node**: Each experience node represents exactly one event/moment. Multiple experiences are never combined.

2. **Emotion Reuse**: Emotions are stored as reusable nodes. The same emotion (e.g., "joy") can be felt multiple times and linked to multiple experiences.

3. **Truth Synthesis**: Truths represent abstracted knowledge that emerges from multiple experiences. The `DISTILLED_FROM` relationship tracks which experiences contributed to each truth.

4. **Temporal Context**: Dates are derived from transcript timestamps, not extracted from content. This ensures consistency and accuracy.

5. **No Manual IDs**: All nodes use Neo4j's built-in `elementId()` for identification, ensuring stability and avoiding ID conflicts.

## Example Queries

### Get all experiences for a user
```cypher
MATCH (u:User {userId: "user_001"})-[:HAD]->(exp:Experience)
RETURN exp
ORDER BY exp.createdAt DESC
```

### Find emotions evoked by an experience
```cypher
MATCH (exp:Experience)-[:EVOKED]->(em:Emotion)
WHERE elementId(exp) = "experience_id"
RETURN em, exp
```

### Get truths distilled from experiences
```cypher
MATCH (t:Truth)-[:DISTILLED_FROM]->(exp:Experience)
WHERE elementId(t) = "truth_id"
RETURN exp, t
```

### Find all experiences that contributed to a truth
```cypher
MATCH (t:Truth)-[r:DISTILLED_FROM]->(exp:Experience)
WHERE elementId(t) = "truth_id"
RETURN exp, r.contribution_weight
ORDER BY r.contribution_weight DESC
```

### Get user's emotional patterns
```cypher
MATCH (u:User {userId: "user_001"})-[:FELT]->(em:Emotion)
MATCH (exp:Experience)-[:EVOKED]->(em)
RETURN em.name, em.valence, COUNT(exp) as frequency
ORDER BY frequency DESC
```

## Notes

- All timestamps are stored in ISO 8601 format
- Dates are extracted from timestamps (first 10 characters: YYYY-MM-DD)
- Experience types are dynamically determined (career, relationship, hobbies, etc.)
- Truth types are: pattern, value, principle, need, preference
- Emotion intensity is on a 0-1 scale
- No fallback/default values are used - all fields must be explicitly provided
