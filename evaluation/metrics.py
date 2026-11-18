"""Corpus-level CER and WER: total edits / total reference length.

Characters are Unicode code points, so a wrong matra or nukta counts as an error.
`nfc=True` scores both sides after NFC; raw scores are reported next to it.
"""

from collections.abc import Callable, Iterable, Sequence

from rapidfuzz.distance import Levenshtein

from dataEngine.text import normalize_text

Pair = tuple[str, str]  # (reference, hypothesis)


def _rate(pairs: Iterable[Pair], units: Callable[[str], Sequence[str]], nfc: bool) -> float:
    edits = total = 0
    for reference, hypothesis in pairs:
        if nfc:
            reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
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
