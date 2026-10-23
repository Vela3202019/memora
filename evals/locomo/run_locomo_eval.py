"""LoCoMo end-to-end eval: ingest -> ask() -> judge -> per-category report.

Usage:
    python -m evals.locomo.run_locomo_eval --limit 1 --max-questions 5   # smoke
    python -m evals.locomo.run_locomo_eval                               # full

The answerer is memora.agents.qa.ask() — the shipped QA path — and it only
sees what Memora stored (honesty rule). Requires a separate eval-only Neo4j
(MEMORA_EVAL_NEO4J_URI / _USERNAME / _PASSWORD; it is wiped per conversation),
OPENAI_API_KEY (embeddings and the default openai judge; DEEPSEEK_API_KEY for
--judge deepseek). Extraction + answering default to the Claude subscription
with Sonnet: verify ANTHROPIC_API_KEY is NOT set.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from memora.agents.qa import ask
from memora.config import LLMConfig
from memora.llm import get_backend
from memora.prompts.diary_extraction_prompt import PROMPT_VERSIONS

from evals.locomo.ingest import ingest_conversation, use_eval_neo4j
from evals.locomo.judge import configure_judge, judge_answer
from evals.locomo.load_locomo import DATA_PATH, load_locomo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("locomo-eval")

RESULTS_DIR = Path(__file__).parent / "results"

# Eval model policy: Sonnet only (user directive) — applies to extraction
# and answering on the claude-subscription backend.
DEFAULT_MODEL = "claude-sonnet-5"
COUNT_TOP_K = 50  # "how many times..." needs breadth, not the 5 nearest


def _question_top_k(question: str, default: int) -> int:
    return COUNT_TOP_K if "how many" in question.lower() else default


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


async def eval_question(q, llm_config: LLMConfig, top_k: int,
                        sem: asyncio.Semaphore) -> dict:
    record = {
        "question": q.question,
        "gold": q.answer,
        "category": q.category_name,
    }
    async with sem:
        try:
            answer = await ask(
                q.question,
                top_k=_question_top_k(q.question, top_k),
                llm_config=llm_config,
            )
            record["answer"] = answer
            verdict = await judge_answer(q.question, q.answer, answer)
            record.update(verdict)
        except Exception as e:
            logger.error("question failed: %s (%s)", q.question[:60], e)
            record.update(answer=None, correct=False, reason="",
                          error=str(e))
    return record


def summarize(records: list) -> tuple[dict, int, int, int]:
    """Per-category and overall counts; errored records are not scored."""
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0, "errors": 0})
    for r in records:
        c = by_cat[r["category"]]
        if "error" in r:
            c["errors"] += 1
        else:
            c["n"] += 1
            c["correct"] += bool(r.get("correct"))
    total_n = sum(c["n"] for c in by_cat.values())
    total_correct = sum(c["correct"] for c in by_cat.values())
    total_errors = sum(c["errors"] for c in by_cat.values())
    return dict(by_cat), total_n, total_correct, total_errors


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of conversations (default all 10)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Max questions per conversation (smoke runs)")
    parser.add_argument("--category", default=None,
                        help="Only questions of this category name")
    parser.add_argument("--backend", default="claude-subscription")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default="v3",
                        choices=list(PROMPT_VERSIONS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qa-concurrency", type=int, default=4)
    parser.add_argument("--judge", default="openai",
                        choices=["openai", "deepseek"],
                        help="Judge family — must differ from the answerer's "
                             "(Claude), so GPT or DeepSeek")
    args = parser.parse_args()

    use_eval_neo4j()
    judge_id = configure_judge(args.judge)
    llm_config = LLMConfig(backend=args.backend, model=args.model)
    extractor_backend = get_backend(llm_config)
    conversations = load_locomo()
    if args.limit:
        conversations = conversations[:args.limit]

    manifest = {
        "benchmark": "locomo10",
        "dataset_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()).hexdigest()[:12],
        "prompt_version": args.prompt_version,
        "backend": args.backend,
        "model": args.model,
        "judge": judge_id,
        "top_k": args.top_k,
        "count_top_k": COUNT_TOP_K,
        "git_rev": _git_rev(),
        "conversations": len(conversations),
    }
    logger.info("manifest: %s", json.dumps(manifest))

    sem = asyncio.Semaphore(args.qa_concurrency)
    all_records = []
    for conv in conversations:
        logger.info("=== %s (%s & %s) ===",
                    conv.conv_id, conv.speaker_a, conv.speaker_b)
        await ingest_conversation(
            conv, extractor_backend, f"{args.backend}:{args.model}",
            prompt_version=args.prompt_version,
        )
        questions = [
            q for q in conv.questions
            if args.category is None or q.category_name == args.category
        ]
        if args.max_questions:
            questions = questions[:args.max_questions]
        records = await asyncio.gather(*[
            eval_question(q, llm_config, args.top_k, sem) for q in questions
        ])
        for r in records:
            r["conv_id"] = conv.conv_id
        all_records.extend(records)
        _, n, done, errors = summarize(all_records)
        logger.info("%s: %d/%d correct so far (%d errors)",
                    conv.conv_id, done, n, errors)

    # ---- report ----
    by_cat, total_n, total_correct, total_errors = summarize(all_records)

    ts = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"run-{ts}-{args.prompt_version}.json"
    json_path.write_text(json.dumps(
        {"manifest": manifest, "records": all_records},
        ensure_ascii=False, indent=1))

    lines = [
        f"# LoCoMo run {ts}",
        "",
        f"- prompt: `{args.prompt_version}` | backend: `{args.backend}`"
        f" | model: `{args.model}` | judge: `{judge_id}`",
        f"- conversations: {len(conversations)} | scored: {total_n}"
        f" | errors (not scored): {total_errors}"
        f" | git: `{manifest['git_rev']}`"
        f" | dataset: `{manifest['dataset_sha256']}`",
        "",
        f"**Overall accuracy: {total_correct}/{total_n}"
        f" ({total_correct / total_n:.1%})**" if total_n else "no questions",
        "",
        "| category | n | correct | accuracy | errors |",
        "|---|---|---|---|---|",
    ]
    for cat in sorted(by_cat):
        c = by_cat[cat]
        lines.append(
            f"| {cat} | {c['n']} | {c['correct']}"
            f" | {c['correct'] / c['n']:.1%} | {c['errors']} |" if c["n"] else
            f"| {cat} | 0 | 0 | n/a | {c['errors']} |"
        )
    md_path = RESULTS_DIR / f"run-{ts}-{args.prompt_version}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nresults: {json_path}\n         {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
