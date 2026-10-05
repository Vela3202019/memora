import asyncio

import evals.run_extraction_eval as ev

CASE = {
    "id": "c1",
    "diary": "Ran a marathon.",
    "gold": {"experiences": [{"description": "Ran a marathon"}], "emotions": [], "truths": []},
}


def _scored(case_id, f1=1.0, hallucinations=0):
    return {
        "id": case_id,
        "schema_violations": [],
        "scores": {"atomicity_violations": [], "edges": None},
        "summary": {"mean_f1": f1, "hallucinations": hallucinations, "recall_misses": 0},
    }


def _failed(case_id):
    return {"id": case_id, "harness_error": "billing_not_active"}


def test_run_case_reports_extractor_failure_without_judging(monkeypatch):
    async def failing_extract(**kwargs):
        return {"experiences": [], "emotions": [], "truths": [], "error": "billing_not_active"}

    async def judge_must_not_run(case, extracted):
        raise AssertionError("a failed extraction must not be judged")

    monkeypatch.setattr(ev, "extract_from_diary", failing_extract)
    monkeypatch.setattr(ev, "judge_case", judge_must_not_run)

    result = asyncio.run(ev.run_case(CASE, "v2", "gpt-5.2"))

    assert result["harness_error"] == "billing_not_active"
    assert "scores" not in result


def test_aggregate_excludes_harness_errors_from_scores():
    trials = [[_scored("a", f1=0.8), _failed("b")], [_scored("a", f1=0.6), _failed("b")]]

    agg = ev.aggregate(trials)

    assert agg["overall"]["mean_f1"] == 0.7
    assert agg["overall"]["harness_errors"] == 2
    assert agg["cases"]["a"]["harness_errors"] == 0
    assert agg["cases"]["b"] == {"trials": 0, "harness_errors": 2}


def test_aggregate_of_a_fully_failed_run_has_no_quality_score():
    agg = ev.aggregate([[_failed("a"), _failed("b")]])

    assert agg["overall"] == {"scored_runs": 0, "harness_errors": 2}


def test_report_states_harness_errors():
    manifest = {
        "timestamp": "t", "prompt_version": "v3", "prompt_sha256": "p",
        "extractor_model": "m", "judge": "j", "trials": 1, "n_cases": 2,
        "dataset_sha256": "d", "git_rev": "r",
    }
    partial = ev.render_report(manifest, ev.aggregate([[_scored("a"), _failed("b")]]))
    failed = ev.render_report(manifest, ev.aggregate([[_failed("a"), _failed("b")]]))

    assert "1 of 2 case runs failed to run" in partial
    assert "| b | not scored" in partial
    assert "No case was scored" in failed
    assert "F1" not in failed.split("**Overall:**")[1]
