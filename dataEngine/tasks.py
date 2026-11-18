"""Task registry: each task turns one record into (prompt, target, image).

Prompts live here, once, so training and evaluation always use the same text.
Add a task = write one function and register it in TASKS; configs refer to it by name.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dataEngine.records import CropRecord

FREE_OCR_PROMPT = "<image>\nFree OCR."  # DeepSeek-OCR's own prompt


@dataclass(frozen=True)
class TaskSample:
    prompt: str
    target: str
    image: Path


def crop_ocr(record: CropRecord, root: Path) -> TaskSample:
    """Crop image → its exact text (Stage 1)."""
    return TaskSample(prompt=FREE_OCR_PROMPT, target=record.text, image=root / record.image)


TASKS: dict[str, Callable[[CropRecord, Path], TaskSample]] = {
    "crop_ocr": crop_ocr,
}


def get_task(name: str) -> Callable[[CropRecord, Path], TaskSample]:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name]
