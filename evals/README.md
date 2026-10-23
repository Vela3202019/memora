# memora evals

Automated evaluation of the diary → memory extraction pipeline. No Neo4j required.

```bash
# full suite, 3 trials, current prompt (v3)
python -m evals.run_extraction_eval --trials 3

# single case, older prompt version
python -m evals.run_extraction_eval --case dream-boundary --prompt-version v1

# compare two runs (e.g. prompt versions)
python -m evals.compare_runs evals/results/run-A-v1.json evals/results/run-B-v2.json
```

Requires `OPENAI_API_KEY` (extractor) and `DEEPSEEK_API_KEY` (judge). Alternatively,
`--judge claude-subscription` judges via `claude -p` at no API cost (`--judge-model` must
differ from a Claude extractor's model); see `--help` for extractor backends.
Results land in `evals/results/` as JSON (full per-trial detail) + Markdown (report).

## Design principles

1. **The judge is a different model family from the extractor.** GPT extractions are graded
   by DeepSeek. A model grading its own family's output exhibits self-preference bias; separating
   them is table stakes for a trustworthy LLM-judge signal.
2. **Deterministic before probabilistic.** Everything that can be checked without a model —
   enum types, value ranges, relationship index validity, weight sums — is checked in plain
   Python first. The judge is only asked questions that require semantic judgment.
3. **Hallucination is the primary error class.** A personal memory system that fabricates
   memories is worse than one that misses them. Every unmatched extraction is classified as
   *grounded* (harmless extra — doesn't hurt precision) or *hallucinated* (unsupported or
   trap-violating — hurts precision). Traps are encoded per-case so the judge grades against
   explicit boundaries, not vibes.
4. **Variance is measured, not hidden.** `--trials N` repeats the suite; reports show
   mean±std. A single-run eval of a stochastic system is an anecdote.
5. **Every run is reproducible.** The manifest records dataset hash, prompt version + hash,
   extractor/judge models, temperatures, and git revision. Two runs are comparable only if
   their dataset hashes match — `compare_runs` warns otherwise.
6. **Guards against over-correction.** The dataset includes regression-guard cases where the
   *right* behavior is to extract (own-past recollections, real actions toward future events),
   so tightening the prompt against hallucination can't silently trade away recall.

## Dataset taxonomy (14 cases)

| case | probes |
|---|---|
| `simple-explicit` | baseline extraction of explicit experience/emotion/truth |
| `multi-experience-separation` | atomicity — 3 events must become 3 experiences |
| `implied-emotion-no-truth` | implied emotion recall; refusing to invent truths |
| `reflection-heavy` | distinguishing truth types (pattern/belief/goal) |
| `distractors-hypotheticals` | future plans, hypotheticals, others' stories |
| `mundane-minimal` | over-extraction pressure on low-content entries |
| `negation-not-occurrence` | events that almost happened must not be extracted |
| `media-secondhand` | podcast/book content is not lived experience |
| `dream-boundary` | dream content vs. the experience of dreaming |
| `bilingual-mixed` | code-switched Chinese/English extraction |
| `long-entry-relationships` | relationship edges (EVOKED / DISTILLED_FROM), scored deterministically via judge alignment |
| `emotion-only-venting` | pure mood entries must not manufacture events/truths |
| `past-recollection-guard` | REGRESSION GUARD: own-past memories must still be extracted |
| `future-vs-real-action-guard` | REGRESSION GUARD: real actions toward future events count |

## Metrics

Per memory type (experiences / emotions / truths):

- **precision** = 1 − hallucination rate among extracted items
- **recall** = matched required gold items / required gold items
  (gold items marked `optional: true` never count against recall; emotions accept listed synonyms)
- **F1**, reported as mean±std across trials

Plus: schema violations (deterministic), atomicity violations (one extraction spanning
multiple events), and **edge recall** for cases with `gold_relationships` — gold edges are
mapped through the judge's item alignment and checked against extracted edges in plain Python.

## Prompt versions

- `v1` — original extraction prompt ("extract everything").
- `v2` — v1 + six explicit exclusion rules (not-the-user's, not-yet-real, not-stated,
  not-claimed, dreams-stay-dreams, negation-is-not-occurrence). Additive only, so
  v1→v2 deltas isolate the effect of negative instructions.
- `v3` (default) — v2 minus truth `confidence`, plus a facts layer (subject–predicate–object
  triples). Facts get deterministic schema checks but are not judge-scored.

## Results: prompt v1 → v2 (July 2026)

3 trials × 14 cases each, extractor `gpt-5.2`, judge `deepseek-chat`
(full reports in `results/compare-v1-vs-v2.md`):

| metric | v1 | v2 | Δ |
|---|---|---|---|
| mean F1 | 0.838 | **0.926** | +0.088 |
| hallucinations / trial | 23.0 | **5.0** | −78% |
| schema violations | 1 | 0 | — |

Biggest wins were exactly the targeted failure modes: distractors (+0.31 F1),
emotion-only venting (+0.33), dream boundary (+0.25), multi-experience emotion
invention (+0.26). The regression guards caught the expected cost: v2 occasionally
over-suppresses legitimate own-past memories and real actions toward future events
(−0.04 to −0.09 F1 on guard cases) — the documented next iteration target.

**Judge calibration:** the primary DeepSeek judge was cross-checked by an independent
Claude judge re-grading the full v2 run (`judge_agreement.py`). Mean inter-judge
agreement (Jaccard): **0.927 on gold-item matching, 0.881 on hallucination flags** —
disagreements concentrate on borderline optional items, not on the headline failures.

## Limitations (read before trusting numbers)

- Gold labels were authored alongside the system, not by independent annotators; treat
  absolute scores as internal signal and deltas between versions as the reliable measurement.
- The dataset is synthetic and English/Chinese only; real diary corpora are messier.
- The judge's grounded/hallucinated split is itself an LLM judgment. Spot-check
  `hallucination_details` in the JSON output when a number surprises you.
- 14 cases is enough to catch failure-mode regressions, not to certify accuracy to a decimal.
