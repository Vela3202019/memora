# Benchmark Eval + Fact Layer Implementation Plan

**Goal:** Measure Memora on LoCoMo (then LongMemEval-S, then BEAM-1M) via a new service-level `ask()` QA capability, establish an honest baseline, then ship the Fact layer (bi-temporal, speaker-attributed) and prove each extension moves the per-category numbers.

**Spec:** `docs/superpowers/specs/2026-07-26-benchmark-eval-and-fact-layer-design.md` — read fully before starting. The spec is the authority; this plan is the sequencing.

**Architecture:** Phase A builds `ask()` (both backend paths) + the LoCoMo harness and runs the baseline with the current schema. Phase B ships the confidence removal + Fact node + bi-temporal update and re-runs for deltas. Phase C extends to LongMemEval-S and BEAM-1M.

## Global Constraints

- Branch: `feature/benchmark-eval` off `main` (pull latest first). Conventional commits, no Co-Authored-By, git email `velapod@gmail.com`, confirm each commit message with the user.
- Headline eval runs use `claude-subscription`; ensure `ANTHROPIC_API_KEY` is NOT set in the environment for those runs (otherwise `claude -p` silently bills the API).
- Judge: DeepSeek via `DEEPSEEK_API_KEY` (cross-family, matches existing eval philosophy).
- The v2 extraction prompt is eval-certified and must not be edited in place. Schema-changing prompt edits (confidence removal, Fact extraction) land as a **new `v3` entry in `PROMPT_VERSIONS`** and get measured by the existing extraction eval before becoming the default.
- The existing extraction eval (`python -m evals.run_extraction_eval`) and unit tests (`python -m pytest tests/`) must stay green after every task (against whichever prompt version they certify at that point).
- Benchmark-specific concerns (category labels, top_k overrides, judge formatting) stay in `evals/locomo/` — never leak into `memora/`.
- Honesty rule: `ask()` and everything under it may only see stored graph nodes, never raw benchmark transcripts.

---

## Phase A — `ask()` + LoCoMo harness + baseline

### Task 1: `ask()` — claude-subscription path (MCP attach)

**Files:** create `memora/agents/qa.py`, `tests/test_qa.py`; export from `memora/agents/__init__.py`.

**Interface:** `async def ask(question: str, top_k: int = 5) -> str`.

- [ ] Write a strict grounded system prompt: answer ONLY from retrieved memories via the tools; terse factual answers; say "I don't know" when the memories don't contain the answer. No confidence language.
- [ ] Implement the claude-subscription path: shell out to `claude -p` with a generated `--mcp-config` pointing at `memora mcp` (stdio), parse the reply using the same envelope handling as `memora/llm/claude_cli.py`.
- [ ] `top_k` plumbs through to the MCP tool descriptions/prompt as the suggested search size.
- [ ] Unit tests: prompt construction + mcp-config generation + reply parsing (mock the subprocess; no live Neo4j/CLI in unit tests).
- [ ] Manual smoke: with local Neo4j up and some ingested data, `python -c "import asyncio; from memora.agents.qa import ask; print(asyncio.run(ask('...')))"` → verify it calls tools and abstains on an unanswerable question.
- [ ] Commit.

### Task 2: `ask()` — API-key backend path (Python tool loop)

**Files:** modify `memora/agents/qa.py`, `memora/llm/base.py`, `memora/llm/openai_compat.py`, `memora/llm/anthropic_api.py`, `memora/llm/gemini_api.py`; tests in `tests/test_qa.py`.

- [ ] Declare tool definitions ONCE in `qa.py` (name/description/params mirroring `memora/mcp/server.py`'s tool list), mapped per SDK.
- [ ] Add an optional tool-calling method to `LLMBackend` (e.g. `complete_tools(system, user, tools, dispatch) -> str`); implement for `openai-compatible` first (covers OpenAI/DeepSeek/Ollama/Groq/OpenRouter — reuse the loop pattern from `chat_agent.py`), then `anthropic` and `gemini`.
- [ ] `ask()` routes: claude-subscription → MCP path (Task 1); everything else → tool loop dispatching directly to `memora.memory.retriever` functions.
- [ ] Unit tests with a fake backend: multi-round tool calls, abstention, dispatch wiring.
- [ ] Commit.

### Task 3: LoCoMo loader

**Files:** create `evals/locomo/__init__.py`, `evals/locomo/load_locomo.py`, `tests/test_locomo_loader.py`. Dataset JSON goes under `evals/locomo/data/` (gitignore if large; document download source/URL + sha in the module docstring).

- [ ] Parse LoCoMo JSON → `LocomoConversation(conv_id, sessions=[(session_date, [(speaker_name, text), ...])], questions=[LocomoQuestion(text, gold_answer, category)])`.
- [ ] Emit per-session transcripts preserving REAL speaker names ("Melanie: ...") — do NOT route through `ConversationSource._fmt()`.
- [ ] Define the LoCoMo two-speaker preamble: extract memories for BOTH named speakers, keep the speaker's name in every description, never use pronouns for the subject.
- [ ] Session dates map to document timestamps (never LLM-inferred).
- [ ] Unit tests on a small fixture: speaker names preserved, categories parsed, dates resolved.
- [ ] Commit.

### Task 4: Ingest orchestration + extraction cache

**Files:** create `evals/locomo/ingest.py`; tests `tests/test_locomo_ingest.py`.

- [ ] Per conversation: wipe Neo4j → extract each session (LoCoMo preamble + current default prompt version) → store.
- [ ] Cache raw extraction JSON to `evals/locomo/cache/<conv_id>-<prompt_version>.json` BEFORE storage; on re-run with same (conv_id, prompt_version), rebuild the graph from cache with zero LLM calls.
- [ ] Unit test cache round-trip (fake extractor); manual smoke: ingest 1 conversation end-to-end, inspect graph in Neo4j browser.
- [ ] Commit.

### Task 5: Judge + orchestrator + report

**Files:** create `evals/locomo/judge.py`, `evals/locomo/run_locomo_eval.py`; results to `evals/locomo/results/`.

- [ ] `judge.py`: DeepSeek correct/incorrect vs gold answer; handles adversarial/unanswerable category (gold = abstain → "I don't know" judged correct).
- [ ] `run_locomo_eval.py`: per conversation — ingest (Task 4) → for each question call `ask()` → judge. Flags: `--limit N` (conversations), `--category`, `--backend`, `--top-k`; count-style categories pass a larger top_k (e.g. 50).
- [ ] Output: JSON (per-question: question, category, retrieved/tool-call trace if available, answer, gold, verdict) + Markdown report with per-category accuracy table + run manifest (dataset hash, prompt version, backend, git rev) — same reproducibility bar as `run_extraction_eval.py`.
- [ ] Smoke: `python -m evals.locomo.run_locomo_eval --limit 1`.
- [ ] Commit.

### Task 6: Baseline run

- [ ] Verify `ANTHROPIC_API_KEY` unset; Neo4j up; full run: all conversations, backend `claude-subscription`.
- [ ] Commit results JSON + Markdown. This per-category table is the Phase B justification — expect single-hop attribute questions and knowledge-update to be the craters.

---

## Phase B — Fact layer + confidence removal

### Task 7: Prompt v3 — confidence removal + Fact extraction

**Files:** modify `memora/prompts/diary_extraction_prompt.py` (new `PROMPT_VERSIONS["v3"]`), `evals/run_extraction_eval.py`, `evals/dataset.json`, `docs/MEMORY_SCHEMA.md`.

- [ ] v3 = v2 minus Truth `confidence` (include/exclude judgment, no score) plus Fact extraction: subject–predicate–object triples, controlled predicate vocab (`has_pet`, `works_at`, `lives_in`, ... — define the initial vocab list in the prompt), canonical-name subjects (never pronouns), NL `statement` rendering.
- [ ] Extraction eval updates: drop the confidence-range schema check; strip `confidence` from gold Truths in `dataset.json`; add deterministic Fact schema checks (predicate in vocab, subject non-pronoun).
- [ ] Run `python -m evals.run_extraction_eval --prompt-version v3 --trials 3`; compare v2→v3 with `compare_runs`; v3 becomes default only if Experience/Emotion/Truth metrics hold (Fact is additive).
- [ ] Commit.

### Task 8: Fact storage + bi-temporal SUPERSEDES

**Files:** modify `memora/memory/storage.py`, `memora/memory/client.py` (index creation if that's where indexes live); tests `tests/test_fact_storage.py`.

- [ ] `Fact` node per spec shape (no confidence field anywhere); `fact_embedding` vector index; `LEARNED_IN` edge when provenance experience exists.
- [ ] Write path: on same `(subject, predicate)` with `valid_to IS NULL` and different object → close old (`valid_to = new session date`), `(new)-[:SUPERSEDES]->(old)`. Same mechanism applied to `Truth{type: preference}`.
- [ ] Remove `confidence` from Truth writes (param + Cypher, storage.py lines ~236–284, ~403).
- [ ] Tests against local Neo4j: create → supersede → chain of 3; preference supersede.
- [ ] Commit.

### Task 9: Fact retrieval + MCP tool + `ask()` pickup

**Files:** modify `memora/memory/retriever.py`, `memora/mcp/server.py`, `memora/agents/qa.py` (tool defs), `memora/agents/chat_agent.py` untouched.

- [ ] `search_facts(query, top_k, include_superseded=False)` — vector search, default filter `valid_to IS NULL`.
- [ ] Drop `t.confidence` from `search_truths`; update MCP `search_truths` description (remove "confidence scores").
- [ ] Add `search_facts` as 5th MCP tool + to the qa.py tool definitions (both `ask()` paths see it automatically).
- [ ] Tests + commit.

### Task 10: Re-run LoCoMo, measure deltas

- [ ] Re-ingest with v3 (cache keys include prompt version, so this is a fresh extraction pass — budget for it), full run, compare per-category vs Task 6 baseline.
- [ ] Write up deltas in `evals/locomo/results/compare-baseline-vs-fact.md`. Update README eval section.
- [ ] Commit; PR for user review (draft description first, per workflow).

---

## Phase C — LongMemEval-S, then BEAM-1M

### Task 11: LongMemEval-S

- [ ] New loader `evals/longmemeval/load_longmemeval.py` (categories: single-session-user/-preference/-assistant, multi-session, temporal-reasoning, knowledge-update); reuse ingest/judge/orchestrator via shared helpers (refactor out of `evals/locomo/` only what both need — no speculative framework).
- [ ] Run, report per-category. Abstention category is the `ask()` prompt's proving ground.

### Task 12: BEAM-1M

- [ ] Budget check FIRST: estimate extraction volume (~1M tokens/conversation × 100 conversations is not feasible on subscription — decide subset or API budget with user before running anything).
- [ ] Loader + summarization prompt mode for `ask()` if BEAM's task set requires it (deferred until here by spec).
- [ ] Run whatever subset the budget decision allows; report with explicit coverage caveats — no silent truncation.
