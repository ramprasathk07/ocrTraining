"""The one text normalizer used by data prep, training targets and evaluation."""

import unicodedata


def normalize_text(text: str) -> str:
    """Return `text` in Unicode NFC.

    The tokenizer has no normalizer, so precomposed and decomposed nukta letters
    become different tokens. Every target and every scored string goes through here.
    """
    return unicodedata.normalize("NFC", text)


def is_valid_utf8(text: str) -> bool:
    """True if `text` can be written as UTF-8 (no lone surrogates)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
