"""Corpus-level text metrics. Error rates are total edits / total reference length.

Characters are Unicode code points, so a wrong matra or nukta counts as an error.
`nfc=True` scores both sides after NFC; raw scores are reported next to it.

- CER / WER: code points / whitespace-separated words.
- AER (akshara error rate): units are aksharas, the written syllables of Devanagari: a
  consonant cluster joined by virama, with its nukta, matras and marks (`aksharas`). A
  conjunct or a matra error costs one akshara, however many code points it has.
- Matra (diacritic) error rate: edits that touch a dependent sign (matra, nukta, virama,
  anusvara, visarga, candrabindu) / dependent signs in the references.
- 1 - NED: mean over items of 1 - edits / max(len(reference), len(hypothesis)), in [0, 1];
  one runaway output cannot push it below 0, unlike CER.
"""

import re
from collections.abc import Callable, Iterable, Sequence

from rapidfuzz.distance import Levenshtein

from dataEngine.text import normalize_text

Pair = tuple[str, str]  # (reference, hypothesis)

_SIGN = "ऀ-ःऺ-़ा-ॏ॑-ॗॢॣ"  # Devanagari dependent
_CONSONANT = "क-हक़-य़ॸ-ॿ"
_JOINER = "‌‍"
_AKSHARA = re.compile(
    rf"(?:[{_CONSONANT}]़?(?:्[{_JOINER}]?[{_CONSONANT}]़?)*|.)[{_SIGN}{_JOINER}]*",
    re.DOTALL,
)
_DIACRITIC = re.compile(f"[{_SIGN}]")


def aksharas(text: str) -> list[str]:
    """Devanagari aksharas; any other character (Latin, digit, space) is its own unit."""
    return _AKSHARA.findall(text)


def _pairs(pairs: Iterable[Pair], nfc: bool) -> Iterable[Pair]:
    for reference, hypothesis in pairs:
        yield (
            (normalize_text(reference), normalize_text(hypothesis))
            if nfc
            else (reference, hypothesis)
        )


def _rate(pairs: Iterable[Pair], units: Callable[[str], Sequence[str]], nfc: bool) -> float:
    edits = total = 0
    for reference, hypothesis in _pairs(pairs, nfc):
        ref, hyp = units(reference), units(hypothesis)
        edits += Levenshtein.distance(ref, hyp)
        total += len(ref)
    if total == 0:
        raise ValueError("all references are empty; the error rate is undefined")
    return edits / total


def cer(pairs: Iterable[Pair], nfc: bool = False) -> float:
    return _rate(pairs, list, nfc)


def wer(pairs: Iterable[Pair], nfc: bool = False) -> float:
    return _rate(pairs, str.split, nfc)


def aer(pairs: Iterable[Pair], nfc: bool = False) -> float:
    return _rate(pairs, aksharas, nfc)


def matra_error_rate(pairs: Iterable[Pair], nfc: bool = False) -> float:
    edits = total = 0
    for reference, hypothesis in _pairs(pairs, nfc):
        total += len(_DIACRITIC.findall(reference))
        for tag, i, j in Levenshtein.editops(reference, hypothesis).as_list():
            ref_sign = tag != "insert" and _DIACRITIC.match(reference[i])
            hyp_sign = tag != "delete" and _DIACRITIC.match(hypothesis[j])
            edits += bool(ref_sign or hyp_sign)
    if total == 0:
        raise ValueError("no dependent signs in the references; the rate is undefined")
    return edits / total


def one_minus_ned(pairs: Iterable[Pair], nfc: bool = False) -> float:
    scores = [1 - Levenshtein.normalized_distance(r, h) for r, h in _pairs(pairs, nfc)]
    if not scores:
        raise ValueError("no items")
    return sum(scores) / len(scores)
