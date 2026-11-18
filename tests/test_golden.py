import pytest
from conftest import GOLDEN, needs_golden

from dataEngine.golden import convert_document, parse_station, to_999


@pytest.mark.parametrize(
    ("document_id", "expected"),
    [
        ("Arariya_SC_ST_PS_07_18", ("Arariya", "SC_ST")),
        ("Arariya_Fulkaha_PS_14_21", ("Arariya", "Fulkaha")),
        ("Aurangabad_Aurangabad_cyber_PS_86_24", ("Aurangabad", "Aurangabad_cyber")),
        ("Begusarai_Bachhwara_PS_75_20", ("Begusarai", "Bachhwara")),
    ],
)
def test_parse_station(document_id, expected):
    assert parse_station(document_id) == expected


def test_parse_station_refuses_to_guess():
    with pytest.raises(ValueError, match="ask for a station mapping"):
        parse_station("scan_0001")


def test_to_999():
    assert to_999(0, 1721) == 0
    assert to_999(1721, 1721) == 999
    assert to_999(264, 1721) == 153


@needs_golden
def test_sample_document_converts_and_validates():
    pages, problems = convert_document(GOLDEN / "Arariya_SC_ST_PS_07_18", GOLDEN)
    assert problems == []
    assert [p.page_id for p in pages] == [
        "Arariya_SC_ST_PS_07_18/page_0001",
        "Arariya_SC_ST_PS_07_18/page_0002",
    ]
    first = pages[0]
    assert first.image == "Arariya_SC_ST_PS_07_18/pages/page_0001/png/page.png"
    title = next(b for b in first.blocks if b.region_id == "r002")
    assert title.bbox_999 == (153, 85, 770, 157)
    assert title.label == "section-title"
    assert title.source.startswith("विषय")
    assert title.english.startswith("Subject")
    rotated = next(b for b in first.blocks if b.region_id == "r005")
    assert rotated.rotation == 90
    assert rotated.crop.endswith("regions/r005.ocr.png")
