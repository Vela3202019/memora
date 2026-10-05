"""
Judge calibration: re-grade a run's extractions with a second judge (Claude via
the `claude` CLI - uses a Claude subscription, no API key) and measure agreement
with the primary DeepSeek judge.

An LLM-judge is only trustworthy if independent judges reach the same verdicts.
This reports, per memory type:
  - match agreement: do both judges match the same gold items?
  - hallucination agreement: do both judges flag the same extracted items?

Usage:
    python -m evals.judge_agreement evals/results/run-XXXX-v2.json [--max-cases N]
"""

import argparse
import json
import subprocess
from pathlib import Path

from evals.run_extraction_eval import JUDGE_PROMPT, build_judge_payload

EVALS_DIR = Path(__file__).parent


def claude_judge(case: dict, extracted: dict) -> dict:
    prompt = JUDGE_PROMPT + "\n\n" + build_judge_payload(case, extracted)
    result = subprocess.run(
        ["claude", "-p", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:500]}")
    text = result.stdout.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end + 1])


def verdict_sets(verdict: dict):
    """Reduce a verdict to comparable sets."""
    matches = {
        kind: {(m["gold_index"], m["extracted_index"])
               for m in verdict.get("matches", {}).get(kind, [])}
        for kind in ("experiences", "emotions", "truths")
    }
    hallucinated = {
        kind: {u["extracted_index"]
               for u in verdict.get("unmatched_extracted", {}).get(kind, [])
               if u.get("classification") == "hallucinated"}
        for kind in ("experiences", "emotions", "truths")
    }
    return matches, hallucinated


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-judge agreement check")
    parser.add_argument("run_file", help="A run-*.json produced by run_extraction_eval")
    parser.add_argument("--max-cases", type=int, default=14)
    args = parser.parse_args()

    run = json.loads(Path(args.run_file).read_text())
    dataset = json.loads((EVALS_DIR / "dataset.json").read_text())
    cases_by_id = {c["id"]: c for c in dataset["cases"]}

    trial = [r for r in run["trials"][0] if "harness_error" not in r][: args.max_cases]
    rows = []
    for r in trial:
        case = cases_by_id[r["id"]]
        claude_verdict = claude_judge(case, r["extracted"])
        m1, h1 = verdict_sets(r["verdict"])
        m2, h2 = verdict_sets(claude_verdict)
        match_agr = sum(jaccard(m1[k], m2[k]) for k in m1) / 3
        hall_agr = sum(jaccard(h1[k], h2[k]) for k in h1) / 3
        rows.append({"id": r["id"], "match_agreement": round(match_agr, 3),
                     "hallucination_agreement": round(hall_agr, 3)})
        print(f"{r['id']:<32} match {match_agr:.2f}  halluc {hall_agr:.2f}")

    mean_match = sum(x["match_agreement"] for x in rows) / len(rows)
    mean_hall = sum(x["hallucination_agreement"] for x in rows) / len(rows)
    summary = {
        "run_file": args.run_file,
        "primary_judge": run["manifest"]["judge"],
        "secondary_judge": "claude-cli",
        "cases": rows,
        "mean_match_agreement": round(mean_match, 3),
        "mean_hallucination_agreement": round(mean_hall, 3),
    }
    out = Path(args.run_file).with_suffix(".agreement.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nMean agreement - matches: {mean_match:.3f}, hallucinations: {mean_hall:.3f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
