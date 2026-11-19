"""Task registry: each task turns one record into (prompt, target, image).

Prompts live here, once, so training and evaluation always use the same text.
Add a task = write one function and register it in TASKS with the record type it reads;
configs refer to it by name.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dataEngine.document_ir import IRBlock, serialize
from dataEngine.records import CropRecord, PageRecord, read_crops, read_pages
from dataEngine.splits import read_splits

FREE_OCR_PROMPT = "<image>\nFree OCR."  # DeepSeek-OCR's own prompts
PAGE_PARSE_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


@dataclass(frozen=True)
class TaskSample:
    prompt: str
    target: str
    image: Path


def crop_ocr(record: CropRecord, root: Path) -> TaskSample:
    """Crop image → its exact text (Stage 1)."""
    return TaskSample(prompt=FREE_OCR_PROMPT, target=record.text, image=root / record.image)


def page_blocks(record: PageRecord) -> list[IRBlock]:
    """Ground-truth blocks in reading order (no_text blocks keep their box, empty text)."""
    return [
        IRBlock(b.label, b.bbox_999, b.source)
        for b in sorted(record.blocks, key=lambda b: b.reading_order)
    ]


def page_parse(record: PageRecord, root: Path) -> TaskSample:
    """Page image → grounded source-only blocks (Stage 2; Stage 0 page baseline)."""
    return TaskSample(PAGE_PARSE_PROMPT, serialize(page_blocks(record)), root / record.image)


@dataclass(frozen=True)
class Task:
    build: Callable[[CropRecord | PageRecord, Path], TaskSample]
    records: str  # "crops" (folder of crop records) or "pages" (pages.jsonl)
    prompt: str  # the exact prompt `build` uses; reports version it by content

    def __call__(self, record: CropRecord | PageRecord, root: Path) -> TaskSample:
        return self.build(record, root)


TASKS: dict[str, Task] = {
    "crop_ocr": Task(crop_ocr, "crops", FREE_OCR_PROMPT),
    "page_parse": Task(page_parse, "pages", PAGE_PARSE_PROMPT),
}


def prompt_version(prompt: str) -> str:
    """Content-addressed id: an edited prompt always gets a new version."""
    return "p-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name]


def read_split(
    task: Task, path: Path, split: str, splits: Path | None
) -> list[CropRecord] | list[PageRecord] | None:
    """Records of `split` for one source of `task`; None while that data is not converted yet.

    Crops: `path` is a crop-record folder (records carry their split).
    Pages: `path` is pages.jsonl and `splits` the folder of locked split lists.
    """
    if task.records == "pages":
        if splits is None:
            raise ValueError(f"page source {path} needs a splits folder")
        if not path.is_file():
            return None
        assignment = read_splits(splits)
        return [p for p in read_pages(path) if assignment.get(p.page_id) == split]
    if not any(path.glob("*.jsonl")):
        return None
    return [r for r in read_crops(path) if r.split == split]
