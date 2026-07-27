import json

from evals.locomo.load_locomo import (
    LOCOMO_PREAMBLE,
    _parse_date_time,
    load_locomo,
)

FIXTURE = {
    "sample_id": "conv-test",
    "qa": [
        {"question": "What did Caroline research?", "answer": "Adoption agencies",
         "evidence": ["D2:8"], "category": 1},
        {"question": "When did X happen?", "answer": 2023, "category": 2},
        {"question": "Trick question?", "adversarial_answer": "nope",
         "category": 5},
    ],
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hi!",
             "blip_caption": "a dog on a beach"},
        ],
        "session_2_date_time": "10:00 am on 9 June, 2023",
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "Back again."},
        ],
    },
}


def _load_fixture(tmp_path):
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps([FIXTURE]))
    return load_locomo(p)[0]


def test_parse_date_time():
    assert _parse_date_time("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"


def test_speaker_names_preserved(tmp_path):
    conv = _load_fixture(tmp_path)
    assert "Caroline: Hey Mel!" in conv.sessions[0].transcript
    assert "[User]" not in conv.sessions[0].transcript
    assert "[Assistant]" not in conv.sessions[0].transcript


def test_photo_caption_inlined(tmp_path):
    conv = _load_fixture(tmp_path)
    assert "Melanie: Hi! [shares a photo: a dog on a beach]" \
        in conv.sessions[0].transcript


def test_sessions_sorted_with_dates(tmp_path):
    conv = _load_fixture(tmp_path)
    assert [s.session_id for s in conv.sessions] == [1, 2]
    assert conv.sessions[1].date_time == "2023-06-09T10:00:00"


def test_questions_categories_and_adversarial(tmp_path):
    conv = _load_fixture(tmp_path)
    q1, q2, q5 = conv.questions
    assert q1.category_name == "multi-hop"
    assert q1.answer == "Adoption agencies"
    assert q2.answer == "2023"  # non-string gold answers coerced
    assert q5.category_name == "adversarial"
    assert q5.answer is None  # gold behavior is abstention


def test_preamble_demands_both_speakers_and_no_pronouns():
    assert "BOTH" in LOCOMO_PREAMBLE
    assert "never pronouns" in LOCOMO_PREAMBLE


def test_real_dataset_loads():
    convs = load_locomo()
    assert len(convs) == 10
    total_qa = sum(len(c.questions) for c in convs)
    assert total_qa == 1986
    c = convs[0]
    assert {c.speaker_a, c.speaker_b} == {"Caroline", "Melanie"}
    # every session has a parseable ISO date
    for conv in convs:
        for s in conv.sessions:
            assert s.date_time[4] == "-" and "T" in s.date_time
