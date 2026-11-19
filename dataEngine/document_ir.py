"""Grounded block format (docs/training-plan.md §7): the model's page output, serialized and parsed.

    <|ref|>paragraph<|/ref|><|det|>[[133, 178, 822, 940]]<|/det|>
    block text …

Boxes are 0–999. Block order is reading order. One serializer and one parser, shared by
training targets, evaluation and the PDF renderer.
"""

import ast
import re
from dataclasses import dataclass

from dataEngine.text import is_valid_utf8

_BLOCK = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL)


@dataclass(frozen=True)
class IRBlock:
    label: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 on the 0–999 scale
    text: str


def serialize(blocks: list[IRBlock]) -> str:
    return "\n\n".join(
        f"<|ref|>{b.label}<|/ref|><|det|>[[{b.box[0]}, {b.box[1]}, {b.box[2]}, {b.box[3]}]]<|/det|>"
        + (f"\n{b.text}" if b.text else "")
        for b in blocks
    )


def _box(det: str) -> tuple[int, int, int, int] | None:
    """`[[x1, y1, x2, y2], …]` → one box (the union when the model lists several)."""
    try:
        boxes = ast.literal_eval(det.strip())
    except (ValueError, SyntaxError):
        return None
    if not boxes or not all(
        isinstance(b, list | tuple) and len(b) == 4 and all(isinstance(v, int) for v in b)
        for b in boxes
    ):
        return None
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
        return None
    return x1, y1, x2, y2


def parse(text: str) -> tuple[list[IRBlock], bool]:
    """Blocks in output order, and whether the whole text was well-formed.

    Malformed headers are skipped (and make `valid` false) so one bad box does not hide
    the rest of the page from the metrics.
    """
    matches = list(_BLOCK.finditer(text))
    valid = bool(matches) and is_valid_utf8(text) and not text[: matches[0].start()].strip()
    blocks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        box = _box(match.group(2))
        if box is None or "<|ref|>" in body or "<|det|>" in body:
            valid = False
            continue
        blocks.append(IRBlock(match.group(1).strip(), box, body))
    return blocks, valid and len(blocks) == len(matches)
