import json
import tomllib
from pathlib import Path

import pytest
from conftest import GOLDEN, OVERRIDES, needs_golden

import evaluation.__main__ as runner
from dataEngine.golden import convert_document
from dataEngine.records import write_jsonl
from dataEngine.sources import golden_regions
from dataEngine.splits import assign_splits, write_splits

RAW = {
    "image": {"base_size": 1024, "image_size": 768, "crop_mode": False},
    "model": {"name_or_path": "m", "revision": "r", "attn_implementation": "eager"},
    "eval": {"output_dir": "outputs/eval/x", "no_repeat_ngram_size": 35},
}


def _item(item_id="a", size=768):
    image_cfg = {**RAW["image"], "image_size": size}
    return {"id": item_id, "prompt": "p", "image_cfg": image_cfg, "max_new_tokens": 10}


def test_eval_dirs_and_labels(repo_tmp):
    assert runner.eval_dir("start", RAW, "dev") == Path("outputs/eval/x/start/dev")
    assert runner.eval_dir("deepseek-ai/DeepSeek-OCR-2", RAW, "dev") == Path(
        "outputs/eval/x/deepseek-ai__DeepSeek-OCR-2/dev"
    )
    assert runner.eval_dir(str(repo_tmp), RAW, "test") == repo_tmp / "eval" / "test"
    assert runner.label("outputs/runs/stage1a_printed/merged") == "stage1a_printed/merged"


def test_fingerprint_changes_with_image_mode_and_weights(repo_tmp):
    base = runner.fingerprint([_item()], RAW, "start")
    assert base == runner.fingerprint([_item()], RAW, "start")
    assert runner.fingerprint([_item(size=1024)], RAW, "start") != base
    (repo_tmp / "adapter_model.safetensors").write_bytes(b"1")
    first = runner.fingerprint([_item()], RAW, str(repo_tmp))
    (repo_tmp / "adapter_model.safetensors").write_bytes(b"22")
    assert runner.fingerprint([_item()], RAW, str(repo_tmp)) != first


@pytest.fixture
def golden_test_set(repo_tmp):
    """Golden test pages → pages.jsonl, all-test splits, region crops (as data_test.toml)."""
    documents = sorted(d for d in GOLDEN.iterdir() if (d / "manifest.json").is_file())
    pages = [p for d in documents for p in convert_document(d, GOLDEN, OVERRIDES)[0]]
    write_jsonl(repo_tmp / "pages.jsonl", pages)
    assignment, info = assign_splits(pages, 1.0, 0.0, 1.0, seed=1)
    write_splits(assignment, info, repo_tmp / "splits")
    source = {
        "name": "golden_regions",
        "pages": str(repo_tmp / "pages.jsonl"),
        "splits": str(repo_tmp / "splits"),
        "data_root": str(GOLDEN),
    }
    golden_regions.convert(source, repo_tmp / "crops", workers=1)
    return repo_tmp, pages


def _raw(tmp: Path) -> dict:
    raw = tomllib.loads(Path("configs/eval/test_data.toml").read_text(encoding="utf-8"))
    raw["eval"]["output_dir"] = str(tmp / "eval")
    raw["eval"]["report_dir"] = str(tmp / "report")
    raw["eval"]["prompts_dir"] = str(tmp / "prompts")
    raw["eval"]["sources"] = [
        {
            "name": "golden_regions",
            "task": "crop_ocr",
            "path": str(tmp / "crops"),
            "limit": 5,
            "max_new_tokens": 9,
        },
        {
            "name": "golden_pages",
            "task": "page_parse",
            "path": str(tmp / "pages.jsonl"),
            "root": str(GOLDEN),
            "splits": str(tmp / "splits"),
            "limit": 3,
            "max_new_tokens": 9,
            "image": {"base_size": 1024, "image_size": 768, "crop_mode": True},
        },
        {
            "name": "not_downloaded",
            "task": "crop_ocr",
            "path": str(tmp / "nothing"),
            "limit": 5,
            "max_new_tokens": 9,
        },
    ]
    return raw


@needs_golden
def test_select_items_samples_each_source_and_skips_missing(golden_test_set):
    tmp, pages = golden_test_set
    assert len(pages) == 49
    items, skipped = runner.select_items(_raw(tmp), "test")
    assert skipped == ["not_downloaded"]
    by_source = {
        s: [i for i in items if i["source"] == s] for s in ("golden_regions", "golden_pages")
    }
    assert len(by_source["golden_regions"]) == 5 and len(by_source["golden_pages"]) == 3
    page = by_source["golden_pages"][0]
    assert page["blocks"] and page["image_cfg"]["crop_mode"] is True
    assert Path(page["image"]).is_file()
    assert items == runner.select_items(_raw(tmp), "test")[0]  # seeded


@needs_golden
def test_whole_pipeline_with_a_stub_model(golden_test_set, monkeypatch):
    """Everything but the model: sampling, sharded loop, scoring, cache, report files."""
    tmp, _ = golden_test_set
    raw = _raw(tmp)
    config = tmp / "eval.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner.tomllib, "loads", lambda _: raw)
    monkeypatch.setattr(runner, "load_tokenizer", lambda cfg: None)

    class Stub:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(runner, "load_model", lambda cfg, adapter: Stub())
    targets = {i["prompt"] + i["image"]: i for i in runner.select_items(raw, "test")[0]}

    def perfect(model, tokenizer, prompt, image, cfg, max_new_tokens, no_repeat):
        item = targets[prompt + str(image)]
        return item["target"], True  # page targets are the serialized ground truth

    monkeypatch.setattr(runner, "generate_text", perfect)
    runner.main(["--config", str(config), "--split", "test", "--checkpoint", "start"])

    summary = json.loads((tmp / "report" / "report_test.json").read_text(encoding="utf-8"))
    pages = summary["scores"][0]["source=golden_pages"]
    assert pages["page_cer"] == 0.0 and pages["mean_iou"] == 1.0 and pages["valid_output"] == 1.0
    assert summary["scores"][0]["source=golden_regions"]["cer"] == 0.0
    report = (tmp / "report" / "report_test.md").read_text(encoding="utf-8")
    assert "Layout mAP@50" in report and "Not measured yet" in report and "## Prompts" in report
    assert len((tmp / "prompts" / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    def fail(*args):
        raise AssertionError("cached predictions should be reused")

    monkeypatch.setattr(runner, "generate_text", fail)
    runner.main(["--config", str(config), "--split", "test", "--checkpoint", "start"])
