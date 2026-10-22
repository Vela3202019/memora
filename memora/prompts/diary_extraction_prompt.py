"""
Extraction prompt template for analyzing diary entries.
Extracts experiences, emotions, and truths from personal reflective writing.
"""

DIARY_EXTRACTION_PROMPT = """You are an expert at analyzing personal diary entries and extracting meaningful memories from someone's life.

You will receive a diary entry written by the user. Your task is to extract ALL memories they share.

## EXTRACTION PROCESS

**Step 1: Read through the entire diary entry carefully**
**Step 2: Extract from each paragraph/section:**
- Every experience mentioned (if multiple, create separate entries for each)
- Every emotion expressed or implied (if multiple, create separate entries for each)
- Every truth/self-insight stated (if multiple, create separate entries for each)

**ONE entry may contain MULTIPLE experiences/emotions/truths - separate each into its own entry.**

---

## WHAT TO EXTRACT

### EXPERIENCES (events, moments, occurrences from the user's ACTUAL LIFE)

Fields:
- description: What happened (concise, vivid, first-person "I")
- type: One of:
  - friendship: experiences with friends
  - family: experiences with family members
  - romantic: experiences with romantic partners
  - career: work, professional achievements, job-related
  - education: school, college, university, courses, studying
  - health: physical/mental health experiences
  - hobbies: personal interests, activities, sports
  - travel: trips, relocations, exploring new places
  - personal_growth: learning, self-discovery, milestones
- location: Where it happened (if mentioned, else null)
- significance: Why it mattered (if evident, else null)
- people_involved: Other people mentioned (not the user)
- context: Other contextual information

**IMPORTANT:** 
- ONE experience = ONE specific event. Never combine multiple events.
- "I met Sarah and Dara today" = 2 separate experiences if meaningful interactions occurred with each

---

### EMOTIONS (feelings the user expresses)

Fields:
- name: ONE word (joy, anxiety, peace, aliveness, gratitude, pride, freedom, blessed, etc.)
- intensity: 0.0-1.0 (0.0 = barely felt, 1.0 = intense)
- valence: positive/negative/neutral
- context: What triggered this emotion

---

### TRUTHS (self-knowledge the user states about themselves)

Fields:
- content: ONE specific insight (first-person "I")
- type: One of:
  - pattern: What I DO — recurring behavior ("I always...", "I tend to...")
  - belief: What I THINK — values, principles ("I believe...")
  - preference: What I LIKE — attractions, tastes ("I prefer...", "I'm drawn to...")
  - goal: What I WANT — intentions ("I want to...", "My goal is...")
- confidence: 0.0-1.0 how explicitly they stated this

---

### RELATIONSHIPS (connections between extracted items)

experience_evoked_emotion:
- Links an experience to the emotion it caused
- Example: {"experience_index": 0, "emotion_index": 0, "intensity": 0.9}

truth_distilled_from_experience:
- Links a truth to the experiences that support it
- contribution_weights should sum to ~1.0
- synthesis_date: Use the diary date provided
- Example: {"truth_index": 0, "experience_indices": [0, 1], "contribution_weights": [0.6, 0.4], "synthesis_date": "2025-08-05"}

---

## OUTPUT FORMAT (JSON)

{
  "experiences": [
    {
      "description": "I had a beautiful conversation with Dara about her startup journey",
      "type": "friendship",
      "location": "San Francisco",
      "significance": "I learned about her passion and determination",
      "people_involved": ["Dara"],
      "context": "Catching up with a friend"
    }
  ],
  "emotions": [
    {
      "name": "aliveness",
      "intensity": 0.9,
      "valence": "positive",
      "context": "Talking deeply with friends made me feel alive"
    }
  ],
  "truths": [
    {
      "content": "I feel alive when talking deep with someone",
      "type": "pattern",
      "confidence": 0.9
    }
  ],
  "relationships": {
    "experience_evoked_emotion": [
      {"experience_index": 0, "emotion_index": 0, "intensity": 0.9}
    ],
    "truth_distilled_from_experience": [
      {"truth_index": 0, "experience_indices": [0], "contribution_weights": [1.0], "synthesis_date": "2025-12-03"}
    ]
  }
}

---

## CRITICAL INSTRUCTIONS

1. **Read the ENTIRE diary entry** - don't skip any section
2. **Extract EVERYTHING:** Every experience, emotion, and truth mentioned
3. **SEPARATE multiple items:** If entry mentions 3 experiences, create 3 separate entries
4. **Never combine items** - each experience/emotion/truth is a separate entry
5. **Extract from reflections too** - diary entries often contain insights and realizations
6. **People matter** - extract names of people involved in experiences
7. **Goals and intentions count** - "I want to..." or "I set a goal to..." are truths
8. **Use first-person "I/me/my"** in all descriptions

Example: If diary says "I met Sarah and Hadassah. Sarah told me about her boyfriend. Hadassah is very mature for 17."
→ Extract:
  1. Experience: Meeting Sarah
  2. Experience: Meeting Hadassah
  3. Experience: Sarah sharing about her relationship (people_involved: Sarah)"""


# V2 = V1 + explicit exclusion rules. The only change is the EXCLUSIONS block below,
# so eval deltas between versions isolate the effect of negative instructions.
DIARY_EXTRACTION_PROMPT_V2 = DIARY_EXTRACTION_PROMPT + """

---

## EXCLUSIONS — WHAT MUST NOT BE EXTRACTED

A memory system that fabricates memories is worse than one that misses them. Apply these rules strictly:

1. **Not the user's** — Stories other people told, things that happened to friends, content from
   books/podcasts/movies. The user HEARING the story can be an experience; the story itself is not.
   - "Priya told me how she got stuck in the Denver airport" → the user's experience is the
     conversation with Priya. Getting stuck in Denver is Priya's memory, never the user's.
2. **Not yet real** — Future plans, intentions to act, and hypotheticals are not experiences.
   - "Tomorrow I fly to Seattle" / "if the demo goes well we might land the client" → NOT experiences.
   - A stated intention may still be a GOAL truth ("I want to...") when the user frames it as one.
3. **Not stated** — Extract an emotion only if it is named or unmistakably implied by the user's
   own words. Do not infer how the user "must have felt" from events alone.
   - Fixing a hard bug does NOT imply relief. Fog clearing does NOT imply awe.
4. **Not claimed** — Extract a truth only when the user asserts self-knowledge. Never generalize
   a single event into a pattern or belief yourself.
5. **Dreams stay dreams** — "I dreamt I was flying" → the experience is having the dream,
   never the dream's content as a real event.
6. **Negation is not occurrence** — "I almost called him" / "I didn't go to the party" describe
   events that did NOT happen. They are not experiences (though the user's reflection on them
   may contain an emotion or truth).

When uncertain whether something qualifies: extract fewer, higher-confidence memories."""


# V3 = V2 minus Truth confidence (everything stored is asserted, not scored)
# plus a FACTS layer: persisting attributes/relationships as
# subject-predicate-object triples. Built by transforming V2 so the certified
# v2 text is never edited in place.
_V3_BASE = (
    DIARY_EXTRACTION_PROMPT_V2
    .replace("- confidence: 0.0-1.0 how explicitly they stated this\n", "")
    .replace(
        '      "type": "pattern",\n      "confidence": 0.9\n',
        '      "type": "pattern"\n',
    )
    .replace(
        "When uncertain whether something qualifies: extract fewer, "
        "higher-confidence memories.",
        "When uncertain whether something qualifies: extract fewer, "
        "clearly-supported memories.",
    )
)

DIARY_EXTRACTION_PROMPT_V3 = _V3_BASE + """

---

## FACTS (persisting attributes and relationships)

Besides experiences (events), also extract FACTS: attributes or relationships
that remain true over a period of time, as opposed to something that happened.
"Melanie adopted a dog last May" is an experience; "Melanie's dog is named
Max" is a fact.

Fields:
- subject: canonical name of who/what the fact is about. NEVER a pronoun
  ("she", "he", "they") — always the person's name. For the diary author,
  use "user" unless their name is known.
- predicate: concise snake_case attribute. Prefer this vocabulary when it
  fits: has_pet, works_at, works_as, lives_in, has_family_member,
  has_friend, has_partner, studies, owns, likes, dislikes, has_hobby,
  has_health_condition, has_goal, is_named. Otherwise coin a short
  snake_case predicate.
- object: the value (concise noun phrase)
- statement: one natural-language sentence stating the fact, using canonical
  names ("Melanie has a dog named Max")
- stated_by: canonical name of the speaker who stated it ("user" for the
  diary author unless their name is known)

Do NOT include dates in facts — the entry's date is attached automatically.
Only extract facts actually stated; never invent or infer them. A fact and an
experience can both come from the same sentence.

Add to the OUTPUT JSON:

  "facts": [
    {
      "subject": "Melanie",
      "predicate": "has_pet",
      "object": "dog named Max",
      "statement": "Melanie has a dog named Max",
      "stated_by": "Melanie"
    }
  ]

And optionally, when a fact was learned during a specific extracted
experience, add to "relationships":

  "fact_learned_in_experience": [
    {"fact_index": 0, "experience_index": 0}
  ]"""


PROMPT_VERSIONS = {
    "v1": DIARY_EXTRACTION_PROMPT,
    "v2": DIARY_EXTRACTION_PROMPT_V2,
    "v3": DIARY_EXTRACTION_PROMPT_V3,
}

