import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy")  # memora.agents package import needs it

from evals.locomo.run_locomo_eval import summarize


def test_errored_questions_are_not_scored():
    records = [
        {"category": "multi-hop", "correct": True},
        {"category": "multi-hop", "correct": False},
        {"category": "multi-hop", "correct": False, "error": "claude -p failed"},
        {"category": "temporal", "correct": False, "error": "rate limit"},
    ]
    by_cat, total_n, total_correct, total_errors = summarize(records)
    assert (total_n, total_correct, total_errors) == (2, 1, 2)
    assert by_cat["multi-hop"] == {"n": 2, "correct": 1, "errors": 1}
    assert by_cat["temporal"] == {"n": 0, "correct": 0, "errors": 1}
