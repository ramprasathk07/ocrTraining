import pytest

from evaluation.report import (
    biggest_changes,
    check_gate,
    markdown_report,
    score,
    validate_gates,
)


def _pred(item_id, target, prediction, script="handwritten", langs=("hi",), source="s"):
    return {
        "id": item_id,
        "task": "crop_ocr",
        "source": source,
        "script": script,
        "langs": list(langs),
        "target": target,
        "prediction": prediction,
    }


def test_score_groups_and_rates():
    scores = score(
        [
            _pred("a", "पत्र", "पत्र"),
            _pred("b", "abcd", "", script="printed", langs=("en",)),
        ]
    )
    assert set(scores) == {
        "all",
        "source=s",
        "script=handwritten",
        "script=printed",
        "lang=hi",
        "lang=en",
        "script=handwritten|lang=hi",
        "script=printed|lang=en",
    }
    assert scores["all"]["exact_match"] == 0.5
    assert scores["all"]["empty_output"] == 0.5
    assert scores["script=printed"]["cer"] == 1.0
    assert scores["script=handwritten|lang=hi"]["cer"] == 0.0


def test_empty_references_give_no_rate():
    assert score([_pred("a", "", "")])["all"]["cer"] is None


def test_gates_pass_fail_and_skip():
    before = {"g": {"cer": 0.20}}
    good = {"g": {"cer": 0.10}}
    assert (
        check_gate({"group": "g", "metric": "cer", "max_relative_change": -0.30}, before, good)[
            "status"
        ]
        == "pass"
    )
    assert (
        check_gate({"group": "g", "metric": "cer", "max_value": 0.05}, before, good)["status"]
        == "fail: max_value"
    )
    assert (
        check_gate(
            {"group": "g", "metric": "cer", "max_absolute_change": 0.0},
            before,
            {"g": {"cer": 0.21}},
        )["status"]
        == "fail: max_absolute_change"
    )
    assert check_gate({"group": "missing", "metric": "cer", "max_value": 1}, before, good)[
        "status"
    ].startswith("skipped")
    single = check_gate({"group": "g", "metric": "cer", "max_absolute_change": 0.0}, {}, good)
    assert single["status"].startswith("pass (no before score")


def test_min_value_gate_for_higher_is_better_metrics():
    after = {"p": {"mean_iou": 0.7}}
    gate = {"group": "p", "metric": "mean_iou", "min_value": 0.8}
    assert check_gate(gate, {}, after)["status"] == "fail: min_value"
    assert check_gate(gate, {}, {"p": {"mean_iou": 0.9}})["status"] == "pass"


def test_page_items_get_page_metrics_under_their_source():
    page = {
        "id": "doc/page_0001",
        "task": "page_parse",
        "source": "golden_pages",
        "script": "page",
        "langs": ["hi"],
        "target": "",
        "prediction": "<|ref|>header<|/ref|><|det|>[[0, 0, 100, 100]]<|/det|>" + "\n" + "पत्र",
        "finished": True,
        "blocks": [{"label": "header", "box": [0, 0, 100, 100], "text": "पत्र"}],
    }
    scores = score([_pred("a", "abcd", "abcd"), page])
    assert scores["source=golden_pages"]["page_cer"] == 0.0
    assert scores["source=golden_pages"]["valid_output"] == 1.0
    assert scores["all"]["n"] == 1  # pages do not mix into crop groups


def test_gate_typos_fail_loudly():
    with pytest.raises(ValueError):
        validate_gates([{"group": "g", "metric": "cer", "max_valu": 0.1}])
    with pytest.raises(ValueError):
        validate_gates([{"group": "g", "metric": "cer"}])


def test_biggest_changes_finds_regressions_and_improvements():
    before = [_pred("a", "abcd", "abcd"), _pred("b", "abcd", "xxxx")]
    after = [_pred("a", "abcd", "abxx"), _pred("b", "abcd", "abcd")]
    changes = biggest_changes(before, after)
    assert [c["id"] for c in changes["regressed"]] == ["a"]
    assert [c["id"] for c in changes["improved"]] == ["b"]


def test_markdown_report_has_tables_gates_and_examples():
    before = [_pred("a", "abcd", "abcd")]
    after = [_pred("a", "abcd", "abxx")]
    scores = [score(before), score(after)]
    gates = [
        [check_gate({"group": "all", "metric": "cer", "max_value": 0.1}, scores[0], scores[1])]
    ]
    prompts = {"crop_ocr": {"version": "p-abc", "prompt": "<image>" + "\n" + "Free OCR."}}
    warnings = ["run/merged: crop_ocr was trained with prompt p-old but evaluated with p-abc"]
    text = markdown_report(
        "T",
        ["start", "run/merged"],
        scores,
        gates,
        [biggest_changes(before, after)],
        ["iam_lines"],
        prompts,
        warnings,
    )
    assert "| all | 1 | 0.00 | 50.00 | +50.00 |" in text
    assert "fail: max_value" in text
    assert "Most regressed: run/merged vs start" in text
    assert "Skipped sources (no records yet): iam_lines" in text
    assert "Extraction field F1" in text  # listed as not measurable yet
    assert "| crop_ocr | `p-abc` | `<image>" + "\\n" + "Free OCR.` |" in text  # newline shown as \n
    assert "**Prompt mismatch:**" in text
