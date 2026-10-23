"""
Compare two eval runs (e.g. prompt v1 vs v2).

Usage:
    python -m evals.compare_runs evals/results/run-A-v1.json evals/results/run-B-v2.json
"""

import argparse
import json
from pathlib import Path


def label(manifest: dict) -> str:
    return f"{manifest['prompt_version']} ({manifest['extractor_model']})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    a = json.loads(Path(args.baseline).read_text())
    b = json.loads(Path(args.candidate).read_text())
    la, lb = label(a["manifest"]), label(b["manifest"])

    for path, run in ((args.baseline, a), (args.candidate, b)):
        if "mean_f1" not in run["aggregate"]["overall"]:
            raise SystemExit(f"{path} has no scored cases - nothing to compare")

    if a["manifest"]["dataset_sha256"] != b["manifest"]["dataset_sha256"]:
        print("WARNING: runs used different dataset versions - deltas are not comparable\n")

    lines = [
        f"# Eval comparison: {la} -> {lb}",
        "",
        f"Baseline: {a['manifest']['timestamp']} ({a['manifest']['trials']} trials) · "
        f"Candidate: {b['manifest']['timestamp']} ({b['manifest']['trials']} trials) · "
        f"judge {b['manifest']['judge']}",
        "",
        f"| case | F1 {la} | F1 {lb} | Δ F1 | halluc. {la} | halluc. {lb} |",
        "|---|---|---|---|---|---|",
    ]
    for case_id, sa in a["aggregate"]["cases"].items():
        sb = b["aggregate"]["cases"].get(case_id)
        if not sb or "mean_f1" not in sa or "mean_f1" not in sb:
            continue
        delta = sb["mean_f1"] - sa["mean_f1"]
        arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "·")
        lines.append(
            f"| {case_id} | {sa['mean_f1']:.3f} | {sb['mean_f1']:.3f} "
            f"| {arrow} {delta:+.3f} | {sa['mean_hallucinations']} "
            f"| {sb['mean_hallucinations']} |"
        )

    oa, ob = a["aggregate"]["overall"], b["aggregate"]["overall"]
    lines += [
        "",
        f"**Overall F1:** {oa['mean_f1']:.3f} -> {ob['mean_f1']:.3f} "
        f"({ob['mean_f1'] - oa['mean_f1']:+.3f})",
        f"**Hallucinations/trial:** {oa['hallucinations_per_trial']} -> "
        f"{ob['hallucinations_per_trial']}",
        f"**Schema violations:** {oa['total_schema_violations']} -> "
        f"{ob['total_schema_violations']}",
        "",
    ]
    report = "\n".join(lines)
    print(report)

    out = Path(args.candidate).parent / (
        f"compare-{a['manifest']['prompt_version']}-vs-"
        f"{b['manifest']['prompt_version']}.md"
    )
    out.write_text(report)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
