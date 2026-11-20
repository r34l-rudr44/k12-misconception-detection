"""Pure helpers of the text classifier. Skipped when torch is not installed (the model code runs on a cloud GPU)."""

import numpy as np
import pytest

pytest.importorskip("torch")
import text_cls as tc  # noqa: E402


def test_clean_gold_drops_annotator_notes():
    gold = "\n".join(
        [
            "L1 | 3/4 + 1/4 = 4/8",
            "L2 | = 1/2 [student answered the question correctly]",
            "L3 | <misconseption> adds denominators",
            "a line without a label",
            "L4 | = 1/2",
        ]
    )
    assert tc.clean_gold(gold) == "L1 | 3/4 + 1/4 = 4/8\nL4 | = 1/2"
    assert tc.clean_gold(float("nan")) == ""


def test_map3():
    P = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.0, 0.1, 0.2, 0.7]])
    y = np.array([0, 2, 2, 0])  # rank 1, rank 2, rank 3, not in the top 3
    assert tc.map3(P, y) == pytest.approx((1 + 1 / 2 + 1 / 3 + 0) / 4)


def test_build_text_ends_with_the_student_work():
    row = type("Row", (), {"grade": 8, "topic_name": "Aljabar", "question_text": "2x = 6", "correct_answer": "3"})()
    text = tc.build_text(row, "L1 | x = 3")
    assert text.startswith("Grade 8. Topic: Aljabar") and text.endswith("Student work:\nL1 | x = 3")
