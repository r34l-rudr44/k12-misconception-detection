"""Parsing of the labeller's replies and the cost arithmetic (no API calls)."""

import pytest

import api_label as al

VALID = ["FRA-01", "FRA-02", "ALG-03", "NONE"]


def test_parse_takes_the_last_answer_line():
    text = "The student adds denominators.\nANSWER: ALG-03\nOn reflection:\nANSWER: fra-01, NONE, FRA-02."
    assert al.parse(text, VALID) == ["FRA-01", "NONE", "FRA-02"]


def test_parse_drops_unknown_and_repeated_codes():
    assert al.parse("ANSWER: FRA-01, XYZ-99, FRA-01, `NONE`", VALID) == ["FRA-01", "NONE"]
    assert al.parse("no answer line here", VALID) == []


def test_transcript_block():
    text = "TRANSCRIPTION:\nL1 | 2 + 2 = 5\nL2 | = 5\nEND TRANSCRIPTION\nThe sum is wrong.\nANSWER: NONE"
    assert al.transcript(text) == "L1 | 2 + 2 = 5\nL2 | = 5"
    assert al.transcript("ANSWER: NONE") == ""


def test_screen_parse():
    assert al.screen_parse("FINAL: 3/4 | MATCH: yes") == "YES"
    assert al.screen_parse("FINAL: 12 | MATCH: NO") == "NO"
    assert al.screen_parse("cannot read the photo") == ""


def test_cost_counts_cache_and_batch_discount():
    usage = {
        "input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    p_in, p_out = al.PRICES["claude-sonnet-5-5"]
    full = p_in * (1 + 1.25 + 0.1) + p_out
    assert al.cost("claude-sonnet-5-5", usage, batch=False) == pytest.approx(full)
    assert al.cost("claude-sonnet-5-5", usage, batch=True) == pytest.approx(full / 2)
    assert al.cost("claude-haiku-5-5", {"input_tokens": 10, "output_tokens": 0}, batch=False) > 0


def test_user_text_lists_the_question_prior():
    row = type(
        "Row", (), {"grade": 7, "topic_name": "Pecahan", "question_text": "1/2 + 1/3 = ...", "correct_answer": "5/6"}
    )()
    text = al.user_text(row, [("FRA-01", 4), ("NONE", 2)])
    assert "FRA-01 (4), NONE (2)" in text and "Answer key: 5/6" in text
    assert "none recorded" in al.user_text(row, None)
    assert "TRANSCRIPTION:" in al.user_text(row, None, transcribe=True)
