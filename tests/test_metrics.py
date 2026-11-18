import pytest

from evaluation.metrics import cer, wer


def test_cer_counts_code_points():
    assert cer([("abc", "abd")]) == pytest.approx(1 / 3)


def test_cer_is_corpus_level():
    assert cer([("ab", "ab"), ("cd", "")]) == pytest.approx(2 / 4)


def test_wer_on_hindi():
    assert wer([("थाना अध्यक्ष अररिया", "थाना अध्यक्ष")]) == pytest.approx(1 / 3)


def test_nfc_score_ignores_nukta_encoding_but_raw_score_does_not():
    pair = [("क़", "क़")]
    assert cer(pair) == pytest.approx(2.0)
    assert cer(pair, nfc=True) == 0.0


def test_empty_references_have_no_rate():
    with pytest.raises(ValueError):
        cer([("", "x")])
