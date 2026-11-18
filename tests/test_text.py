from dataEngine.text import is_valid_utf8, normalize_text


def test_nfc_makes_both_nukta_forms_identical():
    precomposed = "क़"  # क़
    decomposed = "क़"  # क + nukta
    assert precomposed != decomposed
    assert normalize_text(precomposed) == normalize_text(decomposed)


def test_lone_surrogate_is_not_valid_utf8():
    assert is_valid_utf8("थाना अध्यक्ष")
    assert not is_valid_utf8("थाना\udc8d")
