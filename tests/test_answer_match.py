"""The answer matcher on formats seen in the transcriptions and answer keys."""

import pytest

import answer_match as am

CASES = [
    # (student's last line, answer key, expected match)
    ("L1 | = 4.000,", "Rp4.000", 1),  # currency prefix, thousands separator
    ("L1 | = 87000", "Rp87.000", 1),
    ("L1 | = Rp. 180.000,00", "Rp180.000,00", 1),
    ("L1 | 3.625", "3,625", 1),  # ambiguous separator: both readings are tried
    ("L1 | x = 15", "15 tahun", 1),  # unit word in the key
    ("L1 | luas = 30 cm", "30 cm", 1),
    ("L1 | Total = 276 cm²", r"$276\ \text{cm}^2$", 1),  # LaTeX unit
    ("L1 | Q = 1/2", "$0{,}5$", 1),  # fraction vs decimal comma
    ("L1 | = 7,214", "7,21", 1),  # rounds to the decimals of the key
    ("L1 | 6/4 = 1 2/4 = 1 1/2", r"$1\dfrac{1}{2}$", 1),  # mixed number
    ("L1 | = 1 39/45", r"$1\dfrac{13}{15}$", 1),
    ("L1 | 3/√5 = 3√5/5", r"$\dfrac{3\sqrt{5}}{5}$", 1),  # expression key, compared as a string
    ("L1 | 2⁸ = 256", "$2^8$", 1),
    ("L1 | = 5p^3q^2", "$5p^3q^2$", 1),
    ("L1 | = 3π(16) = 48 π cm²", r"$48\pi$", 1),
    ("L1 | = a³ b⁵", r"$a^3 \times b^5$", 1),
    ("L1 | jawaban: A", "B", 0),  # multiple-choice letter
    ("L1 | 35/15 = 2 5/15", "$2$", 0),  # mixed number is not its whole part
    ("L1 | fpb = 28", "3", 0),
    ("L1 | hasil 25", "5", 0),  # no substring match on numbers
    ("L1 | = -11/4", r"$-\dfrac{11}{7}$", 0),
    ("L1 | 12 [crossed out 15]", "15", 0),  # annotations are ignored
    ("L1 | = 0,85", "85%", 0),
]


@pytest.mark.parametrize("text, key, expected", CASES)
def test_match(text, key, expected):
    assert am.match(text, key, 2) == expected


def test_only_the_tail_is_checked():
    text = "L1 | 2 x 6 = 12\nL2 | 12 + 3 = 15\nL3 | 15 - 20 = -5"
    assert am.match(text, "15", tail=1) == 0
    assert am.match(text, "15", tail=2) == 1
    assert am.match(text, "12", tail=2) == 0


def test_empty_work_never_matches():
    assert am.match("", "5") == 0
    assert am.match(float("nan"), "5") == 0


def test_lines_of_strips_line_labels():
    assert am.lines_of("L1 | a = 1\n\nL2 | b = 2") == ["a = 1", "b = 2"]


def test_key_values():
    assert am.key_values("Rp4.000") == ([4000.0, 4.0], None)  # both readings
    assert am.key_values("1.500.000") == ([1500000.0], None)
    assert am.key_values(r"$2\dfrac{5}{8}$") == ([2.625], None)
    values, string = am.key_values("$5p^3q^2$")
    assert values is None and string == "5p^3q^2"
