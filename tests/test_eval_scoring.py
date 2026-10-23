from evals.run_extraction_eval import case_summary, score_case

EMPTY_VERDICT = {"matches": {}, "unmatched_extracted": {}, "missed_gold": {}}


def _case(experiences=(), emotions=(), truths=()):
    return {
        "gold": {
            "experiences": list(experiences),
            "emotions": list(emotions),
            "truths": list(truths),
        }
    }


def test_empty_extraction_with_required_gold_scores_zero():
    case = _case(experiences=[{"description": "Ran a marathon"}])
    scores = score_case(case, {"experiences": [], "emotions": [], "truths": []}, EMPTY_VERDICT)

    assert scores["experiences"]["precision"] == 0.0
    assert scores["experiences"]["recall"] == 0.0
    assert scores["experiences"]["f1"] == 0.0


def test_empty_extraction_without_required_gold_is_correct():
    case = _case(emotions=[{"name": "calm", "optional": True}])
    scores = score_case(case, {"experiences": [], "emotions": [], "truths": []}, EMPTY_VERDICT)

    assert scores["experiences"]["precision"] == 1.0
    assert scores["emotions"]["precision"] == 1.0
    assert scores["emotions"]["f1"] == 1.0


def test_extracting_nothing_does_not_earn_a_passing_case_score():
    case = _case(
        experiences=[{"description": "Ran a marathon"}],
        emotions=[{"name": "pride"}],
        truths=[{"content": "I keep going when it stops being fun"}],
    )
    scores = score_case(case, {}, EMPTY_VERDICT)

    assert case_summary(scores)["mean_f1"] == 0.0


def test_precision_counts_only_hallucinated_items():
    case = _case(experiences=[{"description": "Ran a marathon"}])
    extracted = {"experiences": [{"description": "Ran a marathon"}, {"description": "Won it"}]}
    verdict = {
        "matches": {"experiences": [{"gold_index": 0, "extracted_index": 0}]},
        "unmatched_extracted": {
            "experiences": [{"extracted_index": 1, "classification": "hallucinated"}]
        },
    }
    scores = score_case(case, extracted, verdict)

    assert scores["experiences"]["precision"] == 0.5
    assert scores["experiences"]["recall"] == 1.0
