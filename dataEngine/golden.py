"""Golden pages (human-checked FIRs in `dataEngine/data/`) → PageRecords, plus validation.

Only `png/page.png` is used as the model image. `image/page_regions.png` has the answer
boxes drawn on it and must never reach a model.
"""

import hashlib
import json
import re
from pathlib import Path

from dataEngine.records import Block, PageRecord
from dataEngine.text import is_valid_utf8, normalize_text

INDICDLP_LABELS = frozenset(
    {
        "advertisement", "answer", "author", "chapter-title", "contact-info", "dateline",
        "figure", "figure-caption", "first-level-question", "flag", "folio", "footer",
        "footnote", "formula", "header", "headline", "index", "jumpline", "options",
        "ordered-list", "page-number", "paragraph", "placeholder-text", "quote", "reference",
        "second-level-question", "section-title", "sidebar", "sub-headline",
        "sub-ordered-list", "sub-section-title", "sub-unordered-list", "subsub-headline",
        "subsub-ordered-list", "subsub-section-title", "subsub-unordered-list", "table",
        "table-caption", "table-of-contents", "third-level-question", "unordered-list",
        "website-link",
    }
)  # fmt: skip
STATUSES = frozenset({"ok", "partial", "illegible", "no_text"})

_STATION = re.compile(r"^(?P<district>[^_]+)_(?P<station>.+)_PS_(?P<case>\d+)_(?P<year>\d{2})$")
_MARKER = re.compile(r"\[(?:illegible|uncertain:[^\[\]]*)\]")


def parse_station(document_id: str) -> tuple[str, str]:
    """Split `<District>_<Station>_PS_<case>_<yy>` into (district, station)."""
    match = _STATION.match(document_id)
    if match is None:
        raise ValueError(
            f"document id {document_id!r} does not match <District>_<Station>_PS_<case>_<yy>; "
            "ask for a station mapping instead of guessing"
        )
    return match["district"], match["station"]


def to_999(value: int, size: int) -> int:
    """Pixel coordinate → the model's 0–999 box scale."""
    return round(value / size * 999)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_page(page_dir: Path, data_root: Path) -> PageRecord:
    """Join page.json, layout regions, source text and English into one PageRecord."""
    page = _read_json(page_dir / "page.json")
    layout = {
        r["region_id"]: r
        for r in _read_json(page_dir / "layout_detection" / "regions.json")["regions"]
    }
    english = {
        r["region_id"]: r for r in _read_json(page_dir / "eng_translation" / "page.json")["regions"]
    }
    source_regions = _read_json(page_dir / "source_language_extraction" / "page.json")["regions"]
    width, height = page["width_px"], page["height_px"]
    district, station = parse_station(page["document_id"])

    blocks = []
    for region in sorted(source_regions, key=lambda r: r["reading_order"]):
        rid = region["region_id"]
        x1, y1, x2, y2 = region["bbox_xyxy_px"]
        # Stored crop paths are absolute paths from another machine; only the file name is reused.
        crop_name = Path(layout[rid]["ocr_crop"]["image_path"]).name
        blocks.append(
            Block(
                region_id=rid,
                bbox_px=(x1, y1, x2, y2),
                bbox_999=(
                    to_999(x1, width),
                    to_999(y1, height),
                    to_999(x2, width),
                    to_999(y2, height),
                ),
                label=region["layout_label"],
                reading_order=region["reading_order"],
                source=normalize_text(region.get("source_text", "")),
                english=normalize_text(english.get(rid, {}).get("english_translation", "")),
                status=region["status"],
                langs=tuple(region["source_languages"]),
                crop=(page_dir / "regions" / crop_name).relative_to(data_root).as_posix(),
                rotation=layout[rid]["text_rotation_degrees"],
            )
        )
    return PageRecord(
        page_id=page["page_id"],
        document_id=page["document_id"],
        district=district,
        police_station=station,
        image=(page_dir / "png" / "page.png").relative_to(data_root).as_posix(),
        image_sha256=page["image_sha256"],
        width=width,
        height=height,
        blocks=tuple(blocks),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_problems(name: str, text: str) -> list[str]:
    problems = []
    if not is_valid_utf8(text):
        problems.append(f"{name} is not valid UTF-8")
    if text.count("[uncertain") + text.count("[illegible") != len(_MARKER.findall(text)):
        problems.append(f"{name} has a malformed [illegible]/[uncertain: …] marker")
    return problems


def validate_page(page: PageRecord, data_root: Path) -> list[str]:
    """Every rule from Stage 0 step 0.3. Returns human-readable problems; empty means valid."""
    problems = []
    image = data_root / page.image
    if not image.is_file():
        problems.append(f"missing image {page.image}")
    elif _sha256(image) != page.image_sha256:
        problems.append(f"image hash differs from page.json: {page.image}")

    orders = sorted(b.reading_order for b in page.blocks)
    if orders != list(range(1, len(orders) + 1)):
        problems.append(f"reading_order is not 1..{len(orders)}: {orders}")

    for b in page.blocks:
        where = f"{b.region_id}:"
        x1, y1, x2, y2 = b.bbox_px
        if not (0 <= x1 < x2 <= page.width and 0 <= y1 < y2 <= page.height):
            problems.append(f"{where} box {b.bbox_px} outside page {page.width}x{page.height}")
        if b.label not in INDICDLP_LABELS:
            problems.append(f"{where} label {b.label!r} not in IndicDLP 42")
        if b.status not in STATUSES:
            problems.append(f"{where} status {b.status!r} unknown")
        if b.status == "ok" and not b.source.strip():
            problems.append(f"{where} status ok but source text empty")
        if b.source.strip() and not b.english.strip():
            problems.append(f"{where} source text without English")
        if b.rotation not in (0, 90, 180, 270):
            problems.append(f"{where} rotation {b.rotation} not a right angle")
        if not (data_root / b.crop).is_file():
            problems.append(f"{where} missing crop {b.crop}")
        problems += _text_problems(f"{where} source", b.source)
        problems += _text_problems(f"{where} english", b.english)
    return [f"{page.page_id}: {p}" for p in problems]


def convert_document(doc_dir: Path, data_root: Path) -> tuple[list[PageRecord], list[str]]:
    """All pages of one document. A page that cannot be read becomes a problem, not a crash."""
    pages, problems = [], []
    for page_dir in sorted((doc_dir / "pages").iterdir()):
        try:
            page = convert_page(page_dir, data_root)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
            problems.append(
                f"{page_dir.relative_to(data_root).as_posix()}: cannot read ({error!r})"
            )
            continue
        page_problems = validate_page(page, data_root)
        problems += page_problems
        if not page_problems:
            pages.append(page)
    return pages, problems
