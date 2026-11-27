import pytest
from conftest import GOLDEN, OVERRIDES, needs_golden

from dataEngine.golden import convert_document, parse_group, to_999


@pytest.mark.parametrize(
    ("document_id", "expected"),
    [
        ("Alpha_Beta_XY_07_18", ("Alpha", "Beta")),
        ("Alpha_Two_Words_XY_14_21", ("Alpha", "Two_Words")),
        ("Gamma_Gamma_east_XY_86_24", ("Gamma", "Gamma_east")),
    ],
)
def test_parse_group(document_id, expected):
    assert parse_group(document_id, {}) == expected


def test_parse_group_refuses_to_guess():
    with pytest.raises(ValueError, match="overrides_file"):
        parse_group("scan_0001", {})


def test_override_wins():
    assert parse_group("Delta_Epsilon_13_26", {"Delta_Epsilon_13_26": ["Delta", "Epsilon"]}) == (
        "Delta",
        "Epsilon",
    )


def test_to_999():
    assert to_999(0, 1721) == 0
    assert to_999(1721, 1721) == 999
    assert to_999(264, 1721) == 153


@needs_golden
def test_golden_documents_convert_and_validate():
    documents = sorted(p for p in GOLDEN.iterdir() if (p / "manifest.json").is_file())
    for document in documents[:3]:
        pages, problems = convert_document(document, GOLDEN, OVERRIDES)
        assert problems == []
        assert pages and all(p.page_id.startswith(f"{document.name}/page_") for p in pages)
        for page in pages:
            assert page.image.endswith("/png/page.png")
            for block in page.blocks:
                assert all(0 <= v <= 999 for v in block.bbox_999)
                assert block.crop.startswith(f"{document.name}/") and block.crop.endswith(".png")
