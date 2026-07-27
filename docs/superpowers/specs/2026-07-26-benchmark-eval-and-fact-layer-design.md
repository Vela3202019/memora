# Benchmark eval harness + Fact layer design

## Goal

Measure Memora against three long-term-memory benchmarks — **LoCoMo**, **LongMemEval**,
**BEAM** — starting with an honest baseline, then extending the schema where the failure
data justifies it. Extensions are chosen by one rule: *would this make Memora better even
if the benchmarks didn't exist?* Retrieval-hyperparameter tuning or format-specific hacks
that only move a leaderboard number are out of scope.

Cost ladder, cheapest/smallest first: **LoCoMo → LongMemEval-S → BEAM-1M**. BEAM scales to
1M–10M tokens per conversation; ingesting that via per-turn LLM extraction on
`claude-subscription` is a real cost/time commitment, budgeted explicitly and attempted last.

## Part 1 — Eval harness (build first, against LoCoMo)

New code under `evals/locomo/`, isolated from the existing `evals/run_extraction_eval.py`
suite (that suite measures extraction quality against gold-labeled diary cases; this one
measures end-to-end retrieve-then-answer QA accuracy).

Pipeline, per LoCoMo conversation:

1. **Ingest** — feed the conversation through the existing `sources/conversation.py`
   adapter → `ingest.py`. Each conversation gets its own clean graph (Neo4j wiped between
   conversations — vector indexes are currently global, not user-scoped).

   Two LoCoMo-specific requirements the current adapter does NOT meet, both harness-level
   (the diary path is untouched):

   - **Two-speaker extraction preamble.** The existing `ConversationSource.prompt_preamble`
     says "extract ONLY the USER's memories," and v2 exclusion rule #1 discards anything
     that is "not the user's." LoCoMo conversations are two *peers* (e.g. Melanie ↔
     Caroline) and questions ask about both — under the current prompt stack the extractor
     would correctly-but-uselessly suppress nearly everything, and the baseline would
     measure a format mismatch, not memory quality. The LoCoMo loader uses its own
     preamble: extract memories for BOTH named speakers, keeping the speaker's name in
     every description.
   - **Speaker names must survive formatting.** `_fmt()` in `conversation.py` collapses
     speakers to `[User]`/`[Assistant]` labels (both LoCoMo speakers would map to
     `[Assistant]`). The LoCoMo loader emits its own transcript format with real speaker
     names preserved — attribution (`subject`/`stated_by`, Part 2) is impossible if names
     are erased upstream.

   **Extraction cache:** raw extraction JSON is cached to disk per (conversation,
   prompt-version) before storage. Ingestion is by far the expensive stage (~300 turns ×
   10 conversations of LLM extraction); wiping Neo4j between conversations must not mean
   re-paying extraction on every eval re-run. Re-runs with an unchanged extraction config
   rebuild the graph from cache and only re-run answer→judge.

2. **Answer** — call `memora.agents.qa.ask()` (the new service-level QA entry point, Part
   3). The agent sees **only** what Memora stored — never raw conversation turns. This is
   the honesty rule: if a fact fell through extraction, the miss is the measurement, not
   papered over by letting the LLM read the transcript.
3. **Judge** — DeepSeek (`DEEPSEEK_API_KEY`), correct/incorrect against gold answer,
   cross-family judge per the existing eval philosophy (avoids self-preference bias).

New files:
- `evals/locomo/load_locomo.py` — parse LoCoMo dataset into (conversation, questions[]) with category labels; emits speaker-name-preserving transcripts + the two-speaker preamble (see step 1).
- `evals/locomo/run_locomo_eval.py` — orchestrates the 3 stages; `--limit` / `--category` flags for cheap iteration; manages the extraction cache.
- `evals/locomo/judge.py` — DeepSeek correctness judge.

No eval-local answerer: answering goes through `memora.agents.qa.ask()` (Part 3) so the
benchmark scores the code path the product actually ships, not a parallel eval-only
implementation. Benchmark-specific concerns (category labels, judge formatting, top_k
overrides for counting questions) stay in the harness — LoCoMo's shape must not leak into
the service API.
- Results → `evals/locomo/results/` as JSON (per-question: retrieved nodes, answer, verdict) + Markdown report with **per-category accuracy** — this table is what drives which schema extensions (Part 2) are worth keeping.

Same harness shape is reused for LongMemEval and BEAM (different loader/category labels),
once LoCoMo proves the pipeline out.

## Part 2 — Schema extension: the `Fact` node

Baseline run happens **before** this part ships, so we can prove each extension actually
moves the per-category numbers rather than building blind.

### Why

LoCoMo/LongMemEval/BEAM mostly probe flat facts ("her dog's name", "which city he moved
to"). Memora's current extraction (v2 prompt) deliberately suppresses exactly this kind of
mundane detail in favor of emotionally-salient autobiographical memory. Result: most
benchmark answers simply aren't stored. Fixing this is also a real product gap — a
personal memory system that can't recall a stated fact is missing something real.

### Fact vs. Experience — what goes where

The two node types answer different question shapes; Fact is not a replacement for
Experience or a primary answer source, it fills a gap Experience's shape doesn't cover.

- **Experience** = something that *happened*, bounded at a point/period in time. Answers
  "what happened" / "what did X do." e.g. "Melanie adopted a dog last May."
- **Fact** = a *persisting attribute or relationship*, true over an interval, not itself an
  event. Answers "what is true about X." e.g. "Melanie's dog is named Max."

Benchmark categories split along this line too: LoCoMo's `single-session-user` and
`multi-session` questions are mostly event questions — Experience territory, already
handled today. Only attribute-style questions ("what's her dog's name," "what city does he
live in now") are what Fact is for. One conversation turn can produce both an Experience
(the event) and a Fact (an attribute mentioned in passing) — the same way today's extractor
already splits one diary entry into Experience + Emotion + Truth.

### Shape — reified subject–predicate–object triple

```
(:Fact {
  subject:     "Melanie",                        // who/what it's about
  predicate:   "has_pet",                         // normalized attribute (controlled vocab)
  object:      "dog named Max",
  statement:   "Melanie has a dog named Max",      // NL rendering, embedded for vector search
  valid_from:  "2023-05-01",                       // session date it became true
  valid_to:    null,                               // null = currently true
  stated_by:   "Melanie"                           // provenance / speaker attribution
})
```

No `confidence` field. Everything stored in the graph is asserted as accurate by default —
if it was extracted and written, it's treated as true until superseded (below), not as a
probabilistic claim to be weighed at read time.

**Decision: drop `confidence` from `Truth` too**, not just Fact. This removes an
already-shipped field, so it's a small migration, not just "don't add it to the new node":

- `memora/prompts/diary_extraction_prompt.py` — remove the `confidence` field/instruction from the Truth extraction schema (lines ~68, ~111, ~172 today ask the LLM for a 0.0–1.0 confidence and to "extract fewer, higher-confidence memories" — that judgment call moves to a plain include/exclude decision, no score).
- `memora/memory/storage.py` — drop the `confidence` param from the Truth-write function and Cypher (`confidence: $confidence` in both the query and the `truth.get("confidence")` call site).
- `memora/memory/retriever.py` — drop `t.confidence AS confidence` from `search_truths`.
- `memora/mcp/server.py` — update the `search_truths` tool description, which currently advertises "confidence scores."
- `docs/MEMORY_SCHEMA.md` — remove `confidence` from the documented Truth schema.
- `evals/run_extraction_eval.py` — remove the confidence-range schema-violation check (line ~191-193); the v2 gold dataset (`evals/dataset.json`) likely has `confidence` values on expected Truths too and needs those stripped so gold no longer expects a field that doesn't exist.

This is a real (if small) migration of already-shipped, eval-certified code — called out
explicitly rather than folded silently into "add Fact," since it changes the v2 baseline
the eval suite currently reports against.

Reified as a node (not a plain edge) because the retriever is vector-search-based — a Fact
needs its own embedding to be searchable, same pattern as Experience/Truth.

Predicates are normalized to a controlled vocabulary by the extractor LLM at extraction
time (e.g. `has_pet`, `works_at`, `lives_in`), not matched by embedding similarity — simpler
and debuggable, and the extractor already calls an LLM per document.

**Subjects are normalized too:** the extractor is instructed to always use the canonical
speaker/person name as `subject` — never pronouns. SUPERSEDES matching (below) is exact on
`(subject, predicate)`; "Melanie" vs. "she" vs. "Mel" would silently fail to match and
stale facts would survive. One sentence in the extraction prompt, but load-bearing.

### Relationship to the existing episodic/affective layer

Fact is a new, independent flat layer — it does **not** replace or merge into
Experience/Emotion/Truth. Optional link when a fact's provenance is a specific memory:

```
(:Fact)-[:LEARNED_IN]->(:Experience)
```

So flat recall ("what's her dog's name") and the emotional graph (why that day mattered)
coexist without collapsing into each other.

### Bi-temporal update (knowledge-update / contradiction-resolution categories)

Applies to **both** `Fact` and `Truth{type: preference}` — preferences are also time-varying
(BEAM's "preference following" category tests evolving preferences), so the same mechanism
covers both node types. Never delete; supersede:

```
1. MATCH existing current fact: (f) WHERE f.subject = $s AND f.predicate = $p AND f.valid_to IS NULL
2. close it:  SET f.valid_to = $new_session_date
3. link it:   CREATE (new)-[:SUPERSEDES]->(old)
```

Retrieval defaults to `valid_to IS NULL` (current truth) — this is what answers
knowledge-update questions correctly. History is preserved via the SUPERSEDES chain for
temporal questions ("what did he use before?"). One mechanism serves knowledge-update,
temporal-reasoning, and contradiction-resolution (BEAM) categories — contradiction
resolution is not built as a separate mechanism; "latest wins via SUPERSEDES" is treated as
sufficient for the QA-style questions these benchmarks ask.

### Speaker / entity attribution

Already present in the shape above: `subject` (who the fact is about) vs. `stated_by` (who
said it) — two properties, no new node type. Covers LoCoMo's two-speaker conversations
("what did Melanie say she did?" → filter `stated_by = "Melanie"`). If richer person-graphs
are needed later, promote `subject`/`stated_by` to `:Entity` nodes — explicitly deferred,
not needed for the baseline extensions.

### Code touched
- `memora/memory/diary_extractor.py` — add a Fact-extraction profile emitting normalized triples.
- `memora/memory/storage.py` — SUPERSEDES/close-old-fact write logic + `fact_embedding` vector index.
- `memora/memory/retriever.py` — `search_facts()`, filters `valid_to IS NULL` by default.
- `memora/mcp/server.py` — expose `search_facts` as the 5th MCP tool, so `ask()` (Part 3)
  and external MCP clients can reach the Fact layer.

## Part 3 — `ask()`: grounded QA as a service capability

Retrieve-then-answer QA is core memory-service functionality, not eval scaffolding — so it
lives in the product, and the eval calls it. Memora today has two answer surfaces and
neither fits factual QA: `chat_agent.py` is a persona-heavy companion (insight/comfort,
hard-coded OpenAI `gpt-4o`), and the MCP server only exposes raw search tools (the client
LLM does the synthesis). `ask()` is the missing third surface: terse, grounded, abstaining
factual recall.

### Shape

New `memora/agents/qa.py`: `async def ask(question: str, top_k: int = ...) -> str`.

**Agentic tool-calling, not fixed fan-out.** The answering LLM gets ALL of Memora's
retrieval tools and decides which to call, in as many rounds as it needs:
`search_experiences`, `search_truths`, `get_experiences_by_emotion`,
`get_emotional_patterns`, plus `search_facts` once Part 2 ships. Multi-hop questions
(LoCoMo `multi-hop`, BEAM `multi-session reasoning`) benefit directly — the agent can
chain a second search on what the first one returned, which a one-shot fan-out cannot.

**Implementation: two paths, all backends supported.**

- `claude-subscription` (default): `ask()` shells out to
  `claude -p --mcp-config <memora mcp>` so Claude runs the agentic retrieval loop natively
  against the same MCP server Claude Desktop / Cursor users get. No tool-loop code, and
  the benchmark measures the true product configuration: "Claude + Memora's MCP tools."
- **API-key backends** (`openai-compatible`, `anthropic`, `gemini`): a Python tool loop in
  `qa.py` — the pattern already exists in `chat_agent.py` (OpenAI tool-calling over the
  retriever functions). Tool definitions are declared once (name/description/params,
  mirroring the MCP tool list) and adapted per SDK. The `LLMBackend` abstraction gains an
  optional tool-calling method alongside `complete_json`; `openai-compatible` covers
  OpenAI/DeepSeek/Ollama/Groq/OpenRouter in one implementation, `anthropic` and `gemini`
  map the same definitions to their SDKs' tool APIs.

Either path, the tools only return stored nodes, so the honesty rule holds by
construction. The eval's headline run uses `claude-subscription`; `--backend` stays
selectable like the existing extraction eval, so any backend can be measured.

- `memora/mcp/server.py` — gains the `search_facts` tool in Part 2 (5th tool).
- Chat vs. QA stay separate layers: persona (chat) vs. recall (`ask()`), each testable on
  its own terms.

### Abstention

No numeric confidence threshold, no retrieval-score cutoff — deliberately rejected as an
untunable knob. Abstention is entirely the agent's job: the `ask()` system prompt requires
answering **strictly** from retrieved memories, and saying "I don't know" when they don't
contain the answer. The LLM judges sufficiency directly from content, not from a score.

### Aggregation / counting ("how many times did I…")

No new collect/count query path. Retrieval mechanism is unchanged — the harness passes a
larger `top_k` (e.g. 50) for count-style questions and the agent counts what comes back.
Known limitation, accepted for the baseline: if true matches exceed `top_k`, or vector
ranking doesn't surface all of them, the count will be wrong. Revisit only if the
per-category failure table (Part 1) shows this actually bites.

### Summarization

Deferred out of the LoCoMo milestone. LoCoMo's summarization task is separate from its QA
pairs, and published harnesses (mem0, Zep) score QA only. Build a summarization prompt
mode only when we reach a benchmark task that actually needs it (BEAM). Prompt-only when
it comes — no schema or retrieval change.

## Explicitly deferred / out of scope

- **Instruction-following** (BEAM category) — about assistant behavior, not memory recall; not modeled.
- **Standalone contradiction-detection mechanism** — SUPERSEDES/latest-wins is treated as sufficient for QA-style contradiction questions; no separate detector built.
- **Entity graph promotion** for subject/stated_by — flat string attributes are sufficient until failure data says otherwise.
- **Retrieval-confidence scoring of any kind** — rejected per user direction; abstention is answerer-judgment-based only.

## Validation order

1. Ship `ask()` (Part 3) + harness (Part 1). Run LoCoMo baseline with **current** Memora
   schema (no Fact node yet, current 4 MCP tools) — establish the honest starting point
   and per-category miss table. "Current schema" means current *node types*; the
   two-speaker LoCoMo preamble (Part 1, step 1) is part of the harness, not a schema
   change.
2. Ship Fact node + bi-temporal update + attribution + confidence removal (Part 2);
   re-run the existing extraction eval (`run_extraction_eval.py`) after the confidence
   migration to re-establish its baseline.
3. Re-run LoCoMo — compare per-category deltas against step 1's table. Confirms which
   extensions actually moved numbers.
4. Extend harness to LongMemEval-S, then BEAM-1M, reusing the same pipeline with new
   loaders/category labels (summarization mode built if/when BEAM needs it).
