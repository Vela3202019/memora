"""
Extraction eval for memora.

Runs the diary extractor against a gold-labeled dataset and scores it with:
  1. Deterministic schema checks (types, ranges, relationship indices)
  2. LLM-judge alignment of extracted items to gold items
     -> precision / recall / F1 per memory type
     -> hallucination detection (extracted items not grounded in the entry)
     -> atomicity violations (one extraction combining multiple events)
  3. Deterministic relationship-edge scoring for cases with gold edges,
     using the judge's item alignment as the index mapping.

Design principles:
  - The judge is a DIFFERENT model family from the extractor (DeepSeek judges
    OpenAI extractions) to avoid self-preference bias.
  - --trials N repeats the whole suite to expose run-to-run variance;
    aggregates report mean +/- std.
  - Every run writes a manifest (dataset hash, prompt hash, models, git rev)
    so results are reproducible and comparable across prompt versions.

Usage (from repo root):
    python -m evals.run_extraction_eval --prompt-version v2 --trials 3
    python -m evals.run_extraction_eval --case dream-boundary
    python -m evals.compare_runs evals/results/run-A.json evals/results/run-B.json

Requires OPENAI_API_KEY (extractor) and DEEPSEEK_API_KEY (judge). No Neo4j needed.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from memora.config import LLMConfig
from memora.llm import get_backend
from memora.memory.diary_extractor import extract_from_diary
from memora.prompts.diary_extraction_prompt import PROMPT_VERSIONS

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "deepseek-chat")
JUDGE_BASE_URL = os.getenv("EVAL_JUDGE_BASE_URL", "https://api.deepseek.com")

EXPERIENCE_TYPES = {
    "friendship", "family", "romantic", "career", "education",
    "health", "hobbies", "travel", "personal_growth",
}
TRUTH_TYPES = {"pattern", "belief", "preference", "goal"}
VALENCES = {"positive", "negative", "neutral"}

JUDGE_PROMPT = """You are grading a memory-extraction system against a gold reference.

You will receive:
- DIARY: the original diary entry
- GOLD: memories a careful human annotator extracted
- EXTRACTED: memories the system extracted
- TRAPS: content that must NOT be extracted (may be empty)

Grade as follows:

1. MATCHING - For each gold item, decide if some extracted item expresses the same memory
   (paraphrase is fine; type field may differ - judge the content). One extracted item can
   match at most one gold item. Emotions match if the extracted name is the gold name or a
   listed synonym, with the same valence. Gold items marked "optional": true never count
   against recall when missed.

2. UNMATCHED EXTRACTED ITEMS - For each extracted item that matches no gold item, classify:
   - "grounded": genuinely present in the diary, annotator just didn't list it (harmless extra)
   - "hallucinated": not supported by the diary text, OR violates a listed trap
     (future plans, hypotheticals, dream content as real events, negated/non-occurring events,
     another person's or media-sourced experience presented as the user's, or an invented
     emotion/truth the user never expressed)

3. ATOMICITY - Count extracted experiences that combine two or more distinct events.

Return JSON only, no markdown fences:
{
  "matches": {
    "experiences": [{"gold_index": 0, "extracted_index": 1}],
    "emotions":    [{"gold_index": 0, "extracted_index": 0}],
    "truths":      [{"gold_index": 0, "extracted_index": 0}]
  },
  "unmatched_extracted": {
    "experiences": [{"extracted_index": 2, "classification": "hallucinated", "reason": "..."}],
    "emotions": [],
    "truths": []
  },
  "missed_gold": {
    "experiences": [0],
    "emotions": [],
    "truths": []
  },
  "atomicity_violations": [{"extracted_index": 0, "reason": "combines bike ride and dinner"}]
}"""


# ---------------------------------------------------------------------------
# Judge — a DIFFERENT model family from the extractor, to avoid self-preference.
# Default: DeepSeek (API key). Alternative: the local Claude subscription CLI
# (`claude -p`), which judges GPT extractions cross-family at no API cost.
# ---------------------------------------------------------------------------

_judge_client = None
_judge_backend = None  # set to a ClaudeCLIBackend when --judge claude-subscription


def get_judge_client() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for the judge")
        _judge_client = AsyncOpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    return _judge_client


def build_judge_payload(case: dict, extracted: dict) -> str:
    return json.dumps({
        "DIARY": case["diary"],
        "GOLD": case["gold"],
        "EXTRACTED": {
            "experiences": extracted.get("experiences", []),
            "emotions": extracted.get("emotions", []),
            "truths": extracted.get("truths", []),
        },
        "TRAPS": case.get("traps", []),
    }, ensure_ascii=False, indent=2)


async def judge_case(case: dict, extracted: dict) -> dict:
    payload = build_judge_payload(case, extracted)
    if _judge_backend is not None:
        return await _judge_backend.complete_json(JUDGE_PROMPT, payload)
    response = await get_judge_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

PRONOUN_SUBJECTS = {"i", "me", "she", "he", "they", "her", "him", "them", "it"}
PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def check_schema(extracted: dict, prompt_version: str = "v2") -> list:
    """Deterministic structural checks. Returns list of violation strings."""
    violations = []
    experiences = extracted.get("experiences", [])
    emotions = extracted.get("emotions", [])
    truths = extracted.get("truths", [])

    for i, exp in enumerate(experiences):
        if not exp.get("description"):
            violations.append(f"experience[{i}] missing description")
        if exp.get("type") not in EXPERIENCE_TYPES:
            violations.append(f"experience[{i}] invalid type: {exp.get('type')!r}")

    for i, em in enumerate(emotions):
        if not em.get("name"):
            violations.append(f"emotion[{i}] missing name")
        elif len(str(em["name"]).split()) > 2:
            violations.append(f"emotion[{i}] name not concise: {em['name']!r}")
        if em.get("valence") not in VALENCES:
            violations.append(f"emotion[{i}] invalid valence: {em.get('valence')!r}")
        intensity = em.get("intensity")
        if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
            violations.append(f"emotion[{i}] intensity out of range: {intensity!r}")

    for i, t in enumerate(truths):
        if not t.get("content"):
            violations.append(f"truth[{i}] missing content")
        if t.get("type") not in TRUTH_TYPES:
            violations.append(f"truth[{i}] invalid type: {t.get('type')!r}")
        if prompt_version in ("v1", "v2"):
            # v1/v2 schema includes confidence; dropped from v3 onward —
            # everything stored is asserted, not scored.
            confidence = t.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                violations.append(f"truth[{i}] confidence out of range: {confidence!r}")
        elif "confidence" in t:
            violations.append(f"truth[{i}] has confidence (removed in v3+)")

    for i, f in enumerate(extracted.get("facts", []) or []):
        subject = (f.get("subject") or "").strip()
        if not subject:
            violations.append(f"fact[{i}] missing subject")
        elif subject.lower() in PRONOUN_SUBJECTS:
            violations.append(f"fact[{i}] pronoun subject: {subject!r}")
        if not PREDICATE_RE.match(f.get("predicate") or ""):
            violations.append(f"fact[{i}] predicate not snake_case: {f.get('predicate')!r}")
        if not f.get("statement"):
            violations.append(f"fact[{i}] missing statement")

    rels = extracted.get("relationships", {}) or {}
    for j, rel in enumerate(rels.get("experience_evoked_emotion", []) or []):
        if not (0 <= rel.get("experience_index", -1) < len(experiences)):
            violations.append(f"evoked[{j}] experience_index out of range")
        if not (0 <= rel.get("emotion_index", -1) < len(emotions)):
            violations.append(f"evoked[{j}] emotion_index out of range")
    for j, rel in enumerate(rels.get("truth_distilled_from_experience", []) or []):
        if not (0 <= rel.get("truth_index", -1) < len(truths)):
            violations.append(f"distilled[{j}] truth_index out of range")
        for idx in rel.get("experience_indices", []):
            if not (0 <= idx < len(experiences)):
                violations.append(f"distilled[{j}] experience_index {idx} out of range")
        weights = rel.get("contribution_weights", [])
        if weights and abs(sum(weights) - 1.0) > 0.15:
            violations.append(f"distilled[{j}] contribution_weights sum to {sum(weights):.2f}")

    return violations


def score_edges(case: dict, extracted: dict, verdict: dict) -> dict:
    """
    Deterministic relationship-edge scoring for cases with gold_relationships.
    Uses the judge's item alignment to map gold indices -> extracted indices,
    then checks whether the extractor produced the corresponding edges.
    """
    gold_rels = case.get("gold_relationships")
    if not gold_rels:
        return None

    def mapping(kind):
        return {
            m["gold_index"]: m["extracted_index"]
            for m in verdict.get("matches", {}).get(kind, [])
        }

    exp_map, emo_map, truth_map = mapping("experiences"), mapping("emotions"), mapping("truths")
    ext_rels = extracted.get("relationships", {}) or {}

    ext_evoked = {
        (r.get("experience_index"), r.get("emotion_index"))
        for r in ext_rels.get("experience_evoked_emotion", []) or []
    }
    ext_distilled = {
        (r.get("truth_index"), frozenset(r.get("experience_indices", [])))
        for r in ext_rels.get("truth_distilled_from_experience", []) or []
    }

    gold_edges = 0
    matched_edges = 0
    for edge in gold_rels.get("experience_evoked_emotion", []):
        gold_edges += 1
        ei, mi = edge["experience_index"], edge["emotion_index"]
        if ei in exp_map and mi in emo_map and (exp_map[ei], emo_map[mi]) in ext_evoked:
            matched_edges += 1
    for edge in gold_rels.get("truth_distilled_from_experience", []):
        gold_edges += 1
        ti = edge["truth_index"]
        wanted = {exp_map[i] for i in edge["experience_indices"] if i in exp_map}
        if ti in truth_map and any(
            t == truth_map[ti] and wanted & set(exps)
            for t, exps in ext_distilled
        ):
            matched_edges += 1

    n_extracted_edges = len(ext_evoked) + len(ext_distilled)
    recall = matched_edges / gold_edges if gold_edges else 1.0
    return {
        "gold_edges": gold_edges,
        "extracted_edges": n_extracted_edges,
        "matched_edges": matched_edges,
        "edge_recall": round(recall, 3),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_case(case: dict, extracted: dict, verdict: dict) -> dict:
    scores = {}
    for kind in ("experiences", "emotions", "truths"):
        gold_items = case["gold"].get(kind, [])
        required = [g for g in gold_items if not g.get("optional")]
        n_extracted = len(extracted.get(kind, []))
        matched = len(verdict.get("matches", {}).get(kind, []))
        hallucinated = [
            u for u in verdict.get("unmatched_extracted", {}).get(kind, [])
            if u.get("classification") == "hallucinated"
        ]
        if n_extracted:
            precision = (n_extracted - len(hallucinated)) / n_extracted
        else:
            # Extracting nothing is only correct when nothing was required.
            precision = 0.0 if required else 1.0
        recall = min(matched, len(required)) / len(required) if required else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores[kind] = {
            "gold": len(required),
            "extracted": n_extracted,
            "matched": matched,
            "hallucinated": len(hallucinated),
            "hallucination_details": hallucinated,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    scores["atomicity_violations"] = verdict.get("atomicity_violations", [])
    scores["edges"] = score_edges(case, extracted, verdict)
    return scores


def case_summary(scores: dict) -> dict:
    f1s = [scores[k]["f1"] for k in ("experiences", "emotions", "truths")]
    return {
        "mean_f1": sum(f1s) / len(f1s),
        "hallucinations": sum(
            scores[k]["hallucinated"] for k in ("experiences", "emotions", "truths")
        ),
        "recall_misses": sum(
            scores[k]["gold"] - min(scores[k]["matched"], scores[k]["gold"])
            for k in ("experiences", "emotions", "truths")
        ),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_case(case: dict, prompt_version: str, model: str, backend=None) -> dict:
    start = time.monotonic()
    extracted = await extract_from_diary(
        diary_id=f"eval-{case['id']}",
        diary_date="2026-07-05",
        content=case["diary"],
        prompt_version=prompt_version,
        model=model,
        backend=backend,
    )
    extraction_ms = int((time.monotonic() - start) * 1000)
    schema_violations = check_schema(extracted, prompt_version)
    verdict = await judge_case(case, extracted)
    scores = score_case(case, extracted, verdict)
    return {
        "id": case["id"],
        "extraction_ms": extraction_ms,
        "schema_violations": schema_violations,
        "scores": scores,
        "summary": case_summary(scores),
        "extracted": extracted,
        "verdict": verdict,
    }


async def run_trial(cases, prompt_version, model, backend=None):
    sem = asyncio.Semaphore(4)

    async def bounded(case):
        async with sem:
            return await run_case(case, prompt_version, model, backend)

    return list(await asyncio.gather(*(bounded(c) for c in cases)))


def aggregate(trials: list) -> dict:
    per_case = {}
    for trial in trials:
        for r in trial:
            per_case.setdefault(r["id"], []).append(r)

    case_stats = {}
    for case_id, runs in per_case.items():
        f1s = [r["summary"]["mean_f1"] for r in runs]
        halls = [r["summary"]["hallucinations"] for r in runs]
        case_stats[case_id] = {
            "trials": len(runs),
            "mean_f1": round(statistics.mean(f1s), 3),
            "std_f1": round(statistics.stdev(f1s), 3) if len(f1s) > 1 else 0.0,
            "mean_hallucinations": round(statistics.mean(halls), 2),
            "schema_violations": sum(len(r["schema_violations"]) for r in runs),
            "atomicity_violations": sum(
                len(r["scores"]["atomicity_violations"]) for r in runs
            ),
            "edge_recall": (
                round(statistics.mean(
                    r["scores"]["edges"]["edge_recall"] for r in runs
                ), 3)
                if runs[0]["scores"].get("edges") else None
            ),
        }

    all_f1 = [r["summary"]["mean_f1"] for t in trials for r in t]
    all_hall = [r["summary"]["hallucinations"] for t in trials for r in t]
    return {
        "cases": case_stats,
        "overall": {
            "mean_f1": round(statistics.mean(all_f1), 3),
            "std_f1": round(statistics.stdev(all_f1), 3) if len(all_f1) > 1 else 0.0,
            "total_hallucinations": sum(all_hall),
            "hallucinations_per_trial": round(sum(all_hall) / len(trials), 2),
            "total_schema_violations": sum(
                len(r["schema_violations"]) for t in trials for r in t
            ),
            "total_atomicity_violations": sum(
                len(r["scores"]["atomicity_violations"]) for t in trials for r in t
            ),
        },
    }


def build_manifest(args, dataset_raw: str, n_cases: int) -> dict:
    try:
        git_rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=EVALS_DIR.parent,
        ).stdout.strip()
    except OSError:
        git_rev = "unknown"
    prompt_text = PROMPT_VERSIONS[args.prompt_version]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_rev": git_rev,
        "dataset_sha256": hashlib.sha256(dataset_raw.encode()).hexdigest()[:12],
        "n_cases": n_cases,
        "trials": args.trials,
        "extractor_model": args.model,
        "backend": args.backend,
        "prompt_version": args.prompt_version,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
        "judge": ("claude-subscription"
                  if args.judge == "claude-subscription"
                  else f"deepseek:{JUDGE_MODEL}"),
        "extraction_temperature": 0.1,
        "judge_temperature": 0.0,
    }


def render_report(manifest: dict, agg: dict) -> str:
    lines = [
        "# memora extraction eval",
        "",
        f"Run {manifest['timestamp']} · prompt **{manifest['prompt_version']}** "
        f"(`{manifest['prompt_sha256']}`) · extractor `{manifest['extractor_model']}` · "
        f"judge `{manifest['judge']}` · {manifest['trials']} trial(s) × "
        f"{manifest['n_cases']} cases · dataset `{manifest['dataset_sha256']}` · "
        f"rev `{manifest['git_rev']}`",
        "",
        "| case | F1 (mean±std) | halluc./trial | schema | atomicity | edge recall |",
        "|---|---|---|---|---|---|",
    ]
    for case_id, s in agg["cases"].items():
        edge = s["edge_recall"] if s["edge_recall"] is not None else "—"
        lines.append(
            f"| {case_id} | {s['mean_f1']:.3f}±{s['std_f1']:.3f} "
            f"| {s['mean_hallucinations']} | {s['schema_violations']} "
            f"| {s['atomicity_violations']} | {edge} |"
        )
    o = agg["overall"]
    lines += [
        "",
        f"**Overall:** F1 {o['mean_f1']:.3f}±{o['std_f1']:.3f} · "
        f"{o['hallucinations_per_trial']} hallucinations/trial · "
        f"{o['total_schema_violations']} schema violations · "
        f"{o['total_atomicity_violations']} atomicity violations",
        "",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run memora extraction eval")
    parser.add_argument("--case", help="Run a single case by id")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--prompt-version", default="v2", choices=sorted(PROMPT_VERSIONS))
    parser.add_argument("--model", default="gpt-5.2", help="Extractor model")
    parser.add_argument(
        "--backend",
        choices=["openai-compatible", "claude-subscription", "anthropic", "gemini"],
        default="openai-compatible",
        help="LLM backend for the extractor (default preserves current behavior)",
    )
    parser.add_argument(
        "--judge",
        choices=["deepseek", "claude-subscription"],
        default="deepseek",
        help="Judge backend. 'deepseek' (default) needs DEEPSEEK_API_KEY; "
             "'claude-subscription' judges via `claude -p` at no API cost.",
    )
    args = parser.parse_args()

    dataset_raw = (EVALS_DIR / "dataset.json").read_text()
    cases = json.loads(dataset_raw)["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"No case with id {args.case!r}")

    if args.judge == "claude-subscription":
        from memora.llm.claude_cli import ClaudeCLIBackend
        global _judge_backend
        # Eval model policy: Sonnet only — don't inherit the CLI's default model.
        _judge_backend = ClaudeCLIBackend(model="claude-sonnet-5")
    else:
        get_judge_client()  # fail fast if DEEPSEEK_API_KEY is missing
    backend = get_backend(LLMConfig(backend=args.backend, model=args.model))
    manifest = build_manifest(args, dataset_raw, len(cases))
    logger.info("Manifest: %s", json.dumps(manifest))

    trials = []
    for t in range(args.trials):
        logger.info("Trial %d/%d", t + 1, args.trials)
        trials.append(await run_trial(cases, args.prompt_version, args.model, backend))

    agg = aggregate(trials)
    report = render_report(manifest, agg)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"run-{stamp}-{args.prompt_version}"
    (RESULTS_DIR / f"{base}.json").write_text(json.dumps(
        {"manifest": manifest, "aggregate": agg, "trials": trials},
        ensure_ascii=False, indent=2,
    ))
    (RESULTS_DIR / f"{base}.md").write_text(report)
    print(report)
    print(f"Full results: evals/results/{base}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
