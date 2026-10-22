# Memora — Personal Memory Graph

Point Memora at your files — diaries, notes, chat exports, call transcripts — and it extracts the memories worth keeping into a queryable Neo4j graph: the **experiences** you lived, the **emotions** you felt, and the **truths** you've learned about yourself. Then talk to it, or wire it into Claude / Cursor as an MCP tool.

## Evaluated, not vibes

Extraction quality is measured by an automated eval suite ([`evals/`](evals/README.md)):
14 gold-labeled cases covering a taxonomy of failure modes (hallucinated memories,
dream/negation/secondhand boundaries, atomicity, bilingual entries, relationship edges),
scored by deterministic checks plus a cross-family LLM judge (DeepSeek judging GPT
extractions), with repeat trials and reproducible run manifests.

Latest result — prompt v1 → v2 (3 trials × 14 cases): **mean F1 0.838 → 0.926,
hallucinations −78%** (23 → 5 per trial). The eval's regression guards also surface the
cost: v2 mildly over-suppresses own-past recollections — the next iteration target.
See [`evals/results/compare-v1-vs-v2.md`](evals/results/compare-v1-vs-v2.md).

## The three memory types

Memora models memory the way human memory actually stratifies — and keeps the layers as distinct node types so you can query each on its own terms.

| Type | Cognitive analogue | What it is | Example |
|------|--------------------|------------|---------|
| **Experience** | Episodic memory | A specific event you lived through | "I finished my first marathon in Berlin" |
| **Emotion** | Affective tag | What you felt, with intensity and valence | "pride" (0.9, positive) |
| **Truth** | Semantic self-knowledge | A belief/pattern/preference/goal distilled from experiences | "I keep going when it stops being fun" |

The relationships are the point: an **Experience `EVOKED` an Emotion**, and a **Truth is `DISTILLED_FROM` Experiences**. Emotion isn't decoration — memory research finds that emotional salience is a primary driver of what we encode and later recall, so Memora treats it as a first-class edge that weights and links the graph, not a sentiment score. Truths are never invented from a single event; they're derived knowledge, synthesized from the episodic layer beneath them.

This mapping — episodic memory → Experience, semantic self-knowledge → Truth, emotional salience → Emotion — is the intellectual core of the project. For the full reasoning and its grounding in memory science (Tulving's episodic/semantic distinction, emotional modulation of consolidation), see **[docs/MEMORY_MODEL.md](docs/MEMORY_MODEL.md)**.

## Quick Start

```bash
pip install -e .
docker compose up -d          # local Neo4j (or use Neo4j Aura free tier)
memora init                   # onboarding wizard: sources, backend, database
memora ingest --dry-run       # preview what would be extracted
memora ingest                 # extract and store memories
memora chat                   # talk to your memory graph
memora mcp                    # expose it to Claude Desktop / Cursor via MCP
```

## Configuring sources

Memora ingests files from anywhere on your machine. Two source types:

- **markdown** — diaries, notes, Obsidian vaults (`.md`/`.txt`)
- **conversation** — ChatGPT/Claude export JSON, speaker-labeled call transcripts

`memora init` writes `~/.memora/config.yaml`; edit it by hand any time:

```yaml
user_id: me
sources:
  - name: journal
    type: markdown
    paths: [~/Documents/journal]
    include: ["**/*.md"]
  - name: chatgpt
    type: conversation
    paths: [~/Downloads/chatgpt-export]
llm:
  backend: claude-subscription   # default: your Claude plan, no API bill
embeddings:
  model: text-embedding-3-small  # needs OPENAI_API_KEY
neo4j:
  uri: bolt://localhost:7687
  user: neo4j                    # password from NEO4J_PASSWORD
```

Secrets never live in the config file — only paths and non-secret settings, so it's
safe to share. Timestamps are resolved per document: native timestamps (chat exports) →
filename patterns → markdown front-matter `date:` → file modified time. Never inferred
from content by the LLM.

## LLM backends

| backend | auth | notes |
|---|---|---|
| `claude-subscription` (default) | Claude Code login | shells out to `claude -p`; zero marginal cost on your plan. Do NOT set `ANTHROPIC_API_KEY` or the CLI bills the API instead. |
| `openai-compatible` | `OPENAI_API_KEY` + `base_url` | OpenAI, DeepSeek, Ollama, OpenRouter, Groq |
| `anthropic` | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| `gemini` | `GEMINI_API_KEY` | `pip install google-genai` |

Extraction quality (F1 0.926, above) is certified for `gpt-5.2` only. Other models
vary — measure any backend yourself:

```bash
python -m evals.run_extraction_eval --backend claude-subscription --trials 3
```

## Fully local setup

No API keys, nothing leaves your machine: Docker Neo4j + [Ollama](https://ollama.com).

```yaml
llm:
  backend: openai-compatible
  base_url: http://localhost:11434/v1
  model: llama3.1
embeddings:
  base_url: http://localhost:11434/v1
  model: nomic-embed-text      # note: 768 dims — recreate vector indexes to match
```

## Query via MCP

The MCP (Model Context Protocol) server exposes memory retrieval as tools for AI agents
(Claude Desktop, Cursor, VS Code, etc.). Zero extra dependencies — stdio JSON-RPC.

| Tool | Description |
|------|-------------|
| `search_experiences` | Semantic vector search over life events |
| `search_truths` | Semantic search over personal beliefs, patterns, and goals |
| `get_experiences_by_emotion` | All experiences tied to a specific emotion |
| `get_emotional_patterns` | Emotion frequency, intensity, and valence stats |

Configure in your MCP client:

```json
{
  "mcpServers": {
    "memora": {
      "command": "memora",
      "args": ["mcp"]
    }
  }
}
```

## Project structure

```
memora/
├── cli.py               # memora init / ingest / chat / mcp
├── config.py            # ~/.memora/config.yaml loader
├── init_wizard.py       # onboarding wizard
├── ingest.py            # discover → extract → store pipeline
├── sources/             # markdown + conversation adapters (pluggable)
├── llm/                 # backend layer (claude-cli, openai-compat, anthropic, gemini)
├── memory/              # Neo4j client, storage, embedder, retriever, extractor
├── prompts/             # shared core extraction prompt
├── agents/              # chat agent
├── mcp/server.py        # MCP stdio server
└── resources/           # bundled docker-compose.yml
evals/                   # extraction eval suite (see evals/README.md)
docs/MEMORY_SCHEMA.md    # full Neo4j schema
```

## Neo4j schema

### Nodes
- **User** — `{userId}`
- **Experience** — `{description, type, date, location, significance}`
- **Emotion** — `{name, valence}`
- **Truth** — `{content, type, first_synthesized}`
- **Fact** — `{subject, predicate, object, statement, stated_by, valid_from, valid_to}`.
  A persisting attribute or relationship as a subject–predicate–object triple.
  `valid_from` is the source document's date; `valid_to` is null while the fact is current.
  For single-valued predicates (`lives_in`, `works_at`, `works_as`, `is_named`,
  `has_partner`) a new value closes the user's previous current fact instead of
  deleting it. Other predicates (e.g. `likes`, `has_friend`) accumulate.

### Relationships
- `(User)-[:HAD]->(Experience)`
- `(User)-[:FELT]->(Emotion)`
- `(User)-[:GOT]->(Truth)`
- `(Experience)-[:EVOKED]->(Emotion)`
- `(Truth)-[:DISTILLED_FROM]->(Experience)`
- `(User)-[:KNOWS]->(Fact)`
- `(Fact)-[:LEARNED_IN]->(Experience)`
- `(Fact)-[:SUPERSEDES]->(Fact)` — new value → the value it replaced (whose `valid_to` is set)

### Query examples

```cypher
-- Get all experiences for a user
MATCH (u:User {userId: "me"})-[:HAD]->(exp:Experience)
RETURN exp.description, exp.type, exp.date, exp.location

-- Emotions with intensity
MATCH (u:User {userId: "me"})-[f:FELT]->(em:Emotion)
RETURN em.name, f.intensity, f.context, em.valence

-- Truths / self-knowledge
MATCH (u:User {userId: "me"})-[:GOT]->(t:Truth)
RETURN t.content, t.type

-- Facts that are true now
MATCH (u:User {userId: "me"})-[:KNOWS]->(f:Fact)
WHERE f.valid_to IS NULL
RETURN f.subject, f.predicate, f.object, f.valid_from
```

## Reference

**Experience types:** `friendship`, `family`, `romantic`, `career`, `education`, `health`, `hobbies`, `travel`, `personal_growth`

**Truth types:**

| Type | Description | Example |
|------|-------------|---------|
| `pattern` | Recurring behavior | "I always choose what I really want" |
| `belief` | Values, principles | "I believe deep connections matter" |
| `preference` | Tastes, attractions | "I'm drawn to challenging work" |
| `goal` | Intentions, aspirations | "I want to build great AI products" |

## Contributing

**Add a new source type:** write one adapter in `memora/sources/` implementing the
`Source` interface (`discover()` yielding `SourceDocument`s, plus a `prompt_preamble`),
and register it in `memora/sources/__init__.py`. The extraction, storage, and retrieval
layers are reused as-is.

**Modify the extraction prompt:** `memora/prompts/diary_extraction_prompt.py`. The
`PROMPT_VERSIONS` dict is what the eval suite grades — add a version, then run
`python -m evals.run_extraction_eval` to measure the change before shipping it.

**Run tests:** `python -m pytest tests/`
