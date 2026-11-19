"""Record types shared by every pipeline, and JSONL read/write.

PageRecord: one golden page with its blocks (Stage 0 output).
CropRecord: one image crop with its exact text (Stage 1 input, any source).
"""

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

SPLITS = ("train", "dev", "test")
SCRIPTS = ("printed", "handwritten", "mixed", "unknown")
LEVELS = ("word", "line", "region")
PAGE_LEVEL = "page"  # level of every PageRecord, for curricula that mix crops and pages


@dataclass(frozen=True)
class Block:
    """One region of a golden page."""

    region_id: str
    bbox_px: tuple[int, int, int, int]
    bbox_999: tuple[int, int, int, int]
    label: str
    reading_order: int
    source: str
    english: str
    status: str
    langs: tuple[str, ...]
    crop: str  # relative to the golden data root
    rotation: int


@dataclass(frozen=True)
class PageRecord:
    """One golden page. `image` is relative to the golden data root."""

    page_id: str
    document_id: str
    district: str
    police_station: str
    image: str
    image_sha256: str
    width: int
    height: int
    blocks: tuple[Block, ...]

    @property
    def id(self) -> str:
        return self.page_id

    @property
    def level(self) -> str:
        return PAGE_LEVEL

    @property
    def station_key(self) -> str:
        return f"{self.district}/{self.police_station}"

    @property
    def langs(self) -> set[str]:
        return {lang for block in self.blocks for lang in block.langs}


@dataclass(frozen=True)
class CropRecord:
    """One crop and its exact text. `image` is relative to the record file's folder."""

    id: str
    image: str
    text: str
    langs: tuple[str, ...]
    script: str
    level: str
    source: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: split {self.split!r} not in {SPLITS}")
        if self.script not in SCRIPTS:
            raise ValueError(f"{self.id}: script {self.script!r} not in {SCRIPTS}")
        if self.level not in LEVELS:
            raise ValueError(f"{self.id}: level {self.level!r} not in {LEVELS}")


def page_from_dict(data: dict) -> PageRecord:
    blocks = tuple(
        Block(
            **{
                **b,
                "bbox_px": tuple(b["bbox_px"]),
                "bbox_999": tuple(b["bbox_999"]),
                "langs": tuple(b["langs"]),
            }
        )
        for b in data["blocks"]
    )
    return PageRecord(**{**data, "blocks": blocks})


def crop_from_dict(data: dict) -> CropRecord:
    return CropRecord(**{**data, "langs": tuple(data["langs"])})


def relative_path(target: Path, start: Path) -> str:
    """`target` relative to folder `start`, with forward slashes (portable across OSes)."""
    try:
        return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()
    except ValueError as error:  # e.g. different Windows drives
        raise ValueError(f"cannot make {target} relative to {start}: {error}") from error


def write_jsonl(path: Path, records: Iterable[PageRecord | CropRecord]) -> int:
    """Write records as JSON lines (UTF-8, Devanagari kept readable). Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_pages(path: Path) -> list[PageRecord]:
    with path.open(encoding="utf-8") as f:
        return [page_from_dict(json.loads(line)) for line in f]


def read_crops(folder: Path) -> list[CropRecord]:
    """Every `*.jsonl` part in `folder`, sorted by id so order never depends on worker count."""
    parts = sorted(folder.glob("*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"no crop records (*.jsonl) in {folder}")
    records = []
    for part in parts:
        with part.open(encoding="utf-8") as f:
            records.extend(crop_from_dict(json.loads(line)) for line in f)
    return sorted(records, key=lambda r: r.id)
