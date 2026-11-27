import pytest

from dataEngine.document_ir import IRBlock, serialize
from evaluation.page_metrics import (
    aggregate,
    average_precision,
    greedy_match,
    iou,
    reading_order_accuracy,
    score_page,
)

A, B, C = (0, 0, 100, 100), (200, 0, 300, 100), (0, 200, 100, 300)


def _gt(*blocks):
    return [{"label": label, "box": list(box), "text": text} for label, box, text in blocks]


def test_iou():
    assert iou(A, A) == 1.0
    assert iou(A, B) == 0.0
    assert iou(A, (50, 0, 150, 100)) == pytest.approx(50 * 100 / (2 * 100 * 100 - 50 * 100))


def test_greedy_match_is_one_to_one():
    matches = greedy_match([A, B], [A, A, B], 0.5)
    assert sorted((g, p) for g, p, _ in matches) == [(0, 0), (1, 2)]


def test_average_precision_uses_output_order():
    assert average_precision([A, B], [A, B]) == 1.0
    # precision after each rank: 0, 1/2, 2/3; interpolated (best later) 2/3 at both recall steps
    assert average_precision([A, B], [C, A, B]) == pytest.approx(2 / 3)
    assert average_precision([A], []) == 0.0
    assert average_precision([], [A]) is None


def test_reading_order_accuracy():
    assert reading_order_accuracy([(0, 0), (1, 1), (2, 2)]) == 1.0
    assert reading_order_accuracy([(0, 2), (1, 1), (2, 0)]) == 0.0
    assert reading_order_accuracy([(0, 0)]) is None


def test_perfect_page():
    gt = _gt(("header", A, "पत्र"), ("paragraph", B, "निवेदन पूर्वक"), ("figure", C, ""))
    prediction = serialize([IRBlock(g["label"], tuple(g["box"]), g["text"]) for g in gt])
    scores = aggregate([score_page(gt, prediction, finished=True)])
    assert scores["page_cer"] == 0.0
    assert scores["layout_ap50"] == scores["layout_map50"] == 1.0
    assert scores["mean_iou"] == scores["block_f1"] == scores["reading_order"] == 1.0
    assert scores["valid_output"] == 1.0


def test_base_model_page_with_other_labels_html_and_swapped_order():
    gt = _gt(("header", A, "पत्र"), ("table", B, "क ख"))
    prediction = (
        "<|ref|>table<|/ref|><|det|>[[200, 0, 300, 100]]<|/det|>\n"
        "<table><tr><td>क ख</td></tr></table>\n\n"
        "<|ref|>title<|/ref|><|det|>[[0, 0, 100, 100]]<|/det|>\nपत्र"
    )
    scores = aggregate([score_page(gt, prediction, finished=True)])
    assert scores["mean_iou"] == 1.0 and scores["block_f1"] == 1.0
    assert scores["layout_ap50"] == 1.0
    assert scores["layout_map50"] == 0.5  # "header" ≠ "title"; "table" matches
    assert scores["reading_order"] == 0.0  # the two blocks come out swapped
    assert scores["page_cer"] > 0  # text order differs; HTML tags themselves are not counted


def test_truncated_or_unstructured_output_is_invalid():
    gt = _gt(("header", A, "पत्र"))
    good = serialize([IRBlock("header", A, "पत्र")])
    assert aggregate([score_page(gt, good, finished=False)])["valid_output"] == 0.0
    plain = aggregate([score_page(gt, "पत्र", finished=True)])
    assert plain["valid_output"] == 0.0 and plain["mean_iou"] == 0.0
    assert plain["page_cer"] == 0.0  # plain text still gets a page CER
