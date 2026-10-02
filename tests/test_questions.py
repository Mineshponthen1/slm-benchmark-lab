import json
from collections import Counter
from pathlib import Path

QUESTIONS_FILE = Path(__file__).parent.parent / "questions.json"
QUESTIONS = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def test_exam_has_30_questions():
    assert len(QUESTIONS) == 30


def test_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_is_complete():
    for q in QUESTIONS:
        assert q["question"].strip(), f"Q{q['id']} has no question text"
        assert q["category"].strip(), f"Q{q['id']} has no category"
        assert len(q["accept"]) >= 1, f"Q{q['id']} has no accepted answers"


def test_each_category_has_5_questions():
    counts = Counter(q["category"] for q in QUESTIONS)
    assert all(n == 5 for n in counts.values()), counts