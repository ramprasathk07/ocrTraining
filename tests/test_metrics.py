import pytest

from evaluation.metrics import aer, aksharas, cer, matra_error_rate, one_minus_ned, wer


@pytest.mark.parametrize(
    ("text", "units"),
    [
        ("क्षत्रिय", ["क्ष", "त्रि", "य"]),  # conjuncts + matra stay one akshara
        ("हिंदी", ["हिं", "दी"]),
        ("आँख", ["आँ", "ख"]),  # independent vowel + candrabindu
        ("क़लम", ["क़", "ल", "म"]),  # nukta
        ("सत्‌", ["स", "त्‌"]),  # final virama (+ ZWNJ)
        ("ab 1", ["a", "b", " ", "1"]),
    ],
)
def test_aksharas(text, units):
    assert aksharas(text) == units


def test_aer_counts_a_conjunct_error_once():
    # क्ष → क: CER sees 2 deleted code points, AER one wrong akshara of three
    pair = [("क्षत्रिय", "कत्रिय")]
    assert cer(pair) == pytest.approx(2 / 8)
    assert aer(pair) == pytest.approx(1 / 3)


def test_matra_error_rate():
    assert matra_error_rate([("कि", "का")]) == 1.0  # one matra, substituted
    assert matra_error_rate([("किताब", "कताब")]) == pytest.approx(1 / 2)  # ि dropped, ा kept
    assert matra_error_rate([("किताब", "किटाब")]) == 0.0  # consonant error only
    with pytest.raises(ValueError):
        matra_error_rate([("abc", "abd")])


def test_one_minus_ned_is_bounded_per_item():
    assert one_minus_ned([("abc", "abd")]) == pytest.approx(2 / 3)
    assert one_minus_ned([("ab", "x" * 98), ("", "")]) == pytest.approx((0 + 1) / 2)


def test_cer_counts_code_points():
    assert cer([("abc", "abd")]) == pytest.approx(1 / 3)


def test_cer_is_corpus_level():
    assert cer([("ab", "ab"), ("cd", "")]) == pytest.approx(2 / 4)


def test_wer_on_hindi():
    assert wer([("नमस्ते मेरा नाम", "नमस्ते मेरा")]) == pytest.approx(1 / 3)


def test_nfc_score_ignores_nukta_encoding_but_raw_score_does_not():
    pair = [("क़", "क़")]
    assert cer(pair) == pytest.approx(2.0)
    assert cer(pair, nfc=True) == 0.0


def test_empty_references_have_no_rate():
    with pytest.raises(ValueError):
        cer([("", "x")])
