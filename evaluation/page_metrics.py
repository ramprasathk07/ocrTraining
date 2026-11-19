"""Page-level metrics on grounded blocks.

- page CER: ground-truth text in reading order vs predicted block texts in output order,
  whitespace collapsed, HTML tags removed from the prediction (the base model writes tables
  as HTML). Output without parsable blocks is scored as plain text.
- block P/R/F1@0.5: one-to-one greedy matching at IoU ≥ 0.5, labels ignored.
- mean IoU: one-to-one greedy matching at any overlap; unmatched ground truth counts 0.
- layout AP@50: the model gives no confidence scores, so predictions are ranked by output
  order (VOC all-point AP per page, then averaged). Class-agnostic, and class-aware (mAP)
  over the ground-truth labels of each page.
- reading order: share of matched block pairs whose relative order agrees.
- valid output: the text parses cleanly and generation ended before max_new_tokens.

Boxes are 0–999 on both sides; IoU does not change under per-axis scaling, so it equals
pixel IoU.
"""

import re
from itertools import combinations

from dataEngine.document_ir import IRBlock, parse
from evaluation.metrics import cer

Box = tuple[int, int, int, int]
_TAGS = re.compile(r"<[^>]+>")


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def greedy_match(gt: list[Box], pred: list[Box], threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one (gt index, pred index, IoU), highest IoU first, IoU ≥ threshold and > 0."""
    pairs = sorted(
        (
            (v, g, p)
            for g, gb in enumerate(gt)
            for p, pb in enumerate(pred)
            if (v := iou(gb, pb)) > 0
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_gt, used_pred, matches = set(), set(), []
    for v, g, p in pairs:
        if v >= threshold and g not in used_gt and p not in used_pred:
            used_gt.add(g)
            used_pred.add(p)
            matches.append((g, p, v))
    return matches


def average_precision(gt: list[Box], pred: list[Box], threshold: float = 0.5) -> float | None:
    """VOC all-point AP with predictions in output order; None without ground truth."""
    if not gt:
        return None
    used, hits = set(), []
    for box in pred:
        best = max(
            (g for g in range(len(gt)) if g not in used and iou(gt[g], box) >= threshold),
            key=lambda g: iou(gt[g], box),
            default=None,
        )
        hits.append(best is not None)
        if best is not None:
            used.add(best)
    precisions, recalls, tp = [], [], 0
    for rank, hit in enumerate(hits, start=1):
        tp += hit
        precisions.append(tp / rank)
        recalls.append(tp / len(gt))
    ap, previous_recall = 0.0, 0.0
    for i, recall in enumerate(recalls):
        if recall > previous_recall:
            ap += (recall - previous_recall) * max(precisions[i:])
            previous_recall = recall
    return ap


def reading_order_accuracy(pairs: list[tuple[int, int]]) -> float | None:
    """(ground-truth position, predicted position) of matched blocks → agreeing pair share."""
    if len(pairs) < 2:
        return None
    agree = [(a[0] < b[0]) == (a[1] < b[1]) for a, b in combinations(pairs, 2)]
    return sum(agree) / len(agree)


def _text(blocks: list[IRBlock], strip_tags: bool) -> str:
    texts = (_TAGS.sub(" ", b.text) if strip_tags else b.text for b in blocks)
    return " ".join(" ".join(t.split()) for t in texts if t.strip())


def score_page(gt_blocks: list[dict], prediction: str, finished: bool) -> dict:
    """Raw counts and per-page values; `aggregate` turns a list of these into metrics."""
    gt = [IRBlock(b["label"], tuple(b["box"]), b["text"]) for b in gt_blocks]
    pred, parsed_ok = parse(prediction)
    gt_boxes, pred_boxes = [b.box for b in gt], [b.box for b in pred]
    matched = greedy_match(gt_boxes, pred_boxes, 0.5)
    class_ap = {
        label: average_precision(
            [b.box for b in gt if b.label == label], [b.box for b in pred if b.label == label]
        )
        for label in {b.label for b in gt}
    }
    return {
        "valid": parsed_ok and finished,
        "gt_blocks": len(gt),
        "pred_blocks": len(pred),
        "matched50": len(matched),
        "iou_sum": sum(v for _, _, v in greedy_match(gt_boxes, pred_boxes, 1e-9)),
        "ap50": average_precision(gt_boxes, pred_boxes),
        "class_ap50": list(class_ap.values()),
        "reading_order": reading_order_accuracy([(g, p) for g, p, _ in matched]),
        "reference": _text(gt, strip_tags=False),
        # no parsable blocks: score the raw text, so reading is measured apart from structure
        "hypothesis": _text(
            pred or [IRBlock("", (0, 0, 1, 1), _strip_grounding(prediction))], True
        ),
    }


def _strip_grounding(text: str) -> str:
    return re.sub(r"<\|/?(ref|det)\|>", " ", text)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def aggregate(pages: list[dict]) -> dict:
    gt = sum(p["gt_blocks"] for p in pages)
    pred = sum(p["pred_blocks"] for p in pages)
    matched = sum(p["matched50"] for p in pages)
    precision = matched / pred if pred else 0.0
    recall = matched / gt if gt else 0.0
    try:
        page_cer = cer([(p["reference"], p["hypothesis"]) for p in pages])
    except ValueError:
        page_cer = None
    return {
        "n": len(pages),
        "page_cer": page_cer,
        "layout_ap50": _mean([p["ap50"] for p in pages]),
        "layout_map50": _mean([ap for p in pages for ap in p["class_ap50"]]),
        "mean_iou": sum(p["iou_sum"] for p in pages) / gt if gt else None,
        "block_precision": precision,
        "block_recall": recall,
        "block_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "reading_order": _mean([p["reading_order"] for p in pages]),
        "valid_output": sum(p["valid"] for p in pages) / len(pages),
    }
