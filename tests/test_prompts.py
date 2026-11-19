import json

from dataEngine.records import CropRecord
from dataEngine.tasks import TASKS, crop_ocr, prompt_version
from evaluation.prompts import drift_warnings, prompts_used, record_run

RAW = {
    "name": "eval_test_data",
    "model": {"name_or_path": "deepseek-ai/DeepSeek-OCR-2", "revision": "aaa02f3811945a91"},
    "eval": {
        "sources": [
            {"name": "printed_words", "task": "crop_ocr"},
            {"name": "golden_pages", "task": "page_parse"},
        ]
    },
}


def test_prompt_version_is_content_addressed():
    assert prompt_version("<image>\nFree OCR.") == prompt_version("<image>\nFree OCR.")
    assert prompt_version("<image>\nFree OCR.") != prompt_version("<image>\nFree OCR. ")
    assert prompt_version("x").startswith("p-") and len(prompt_version("x")) == 12


def test_each_task_declares_the_prompt_it_builds():
    record = CropRecord("s/1", "x.png", "t", ("hi",), "printed", "word", "s", "test")
    assert crop_ocr(record, __import__("pathlib").Path(".")).prompt == TASKS["crop_ocr"].prompt


def test_registry_history_and_leaderboard(repo_tmp):
    prompts = prompts_used(RAW["eval"]["sources"])
    assert set(prompts) == {"crop_ocr", "page_parse"}
    scores = [
        {
            "source=printed_words": {"n": 200, "cer": 0.10, "wer": 0.3, "exact_match": 0.7},
            "source=golden_pages": {"n": 49, "page_cer": 0.4, "mean_iou": 0.6, "valid_output": 1.0},
        }
    ]
    record_run(repo_tmp, RAW, "test", ["start"], scores, prompts)
    record_run(repo_tmp, RAW, "test", ["start"], scores, prompts)

    registry = json.loads((repo_tmp / "registry.json").read_text(encoding="utf-8"))
    assert set(registry) == {p["version"] for p in prompts.values()}  # no duplicate entries
    history = (repo_tmp / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 2
    board = (repo_tmp / "leaderboard.md").read_text(encoding="utf-8")
    assert "cer=10.00%" in board and "page_cer=40.00%" in board


def test_drift_warning_when_eval_prompt_differs_from_training(repo_tmp):
    prompts = prompts_used([{"name": "w", "task": "crop_ocr"}])
    checkpoint = repo_tmp / "checkpoint-100"
    checkpoint.mkdir()
    trained = {"crop_ocr": {"version": "p-0000000000", "prompt": "old"}}
    (repo_tmp / "run_manifest.json").write_text(json.dumps({"prompts": trained}), encoding="utf-8")
    assert drift_warnings([str(checkpoint)], prompts) == [
        f"{checkpoint}: crop_ocr was trained with prompt p-0000000000 "
        f"but evaluated with {prompts['crop_ocr']['version']}"
    ]
    trained["crop_ocr"] = prompts["crop_ocr"]
    (repo_tmp / "run_manifest.json").write_text(json.dumps({"prompts": trained}), encoding="utf-8")
    assert drift_warnings([str(checkpoint)], prompts) == []
