import json
from dataclasses import replace
from pathlib import Path

import pytest

from dataEngine.records import PageRecord, write_jsonl
from dataEngine.splits import assign_splits, write_splits
from dataEngine.tasks import get_task, read_split
from training.config import RecordSource, StageConfig, load_config
from training.prompts import register_training_run
from training.stages import check, load_stages, tree

STAGE1A = Path("configs/stage1/1a_printed.toml")


def test_stage_tree_from_configs():
    stages = load_stages()
    assert check(stages) == []
    lines = tree(stages)
    assert lines[0].startswith("base")
    ids = [line.split(":")[0].strip("└─ ").strip() for line in lines if "└─" in line]
    assert ids == ["1A", "1B", "1C"]


def test_stage_tree_problems_are_reported():
    (path, cfg), *_ = load_stages()
    orphan = replace(cfg, stage=StageConfig("2", "pages", starts_from="9Z"))
    looped = replace(cfg, stage=StageConfig("1A", "dup", starts_from="1A"))
    problems = check([(path, cfg), (path, orphan), (path, looped)])
    assert "duplicate stage id 1A" in problems
    assert "2: starts_from '9Z' is not a stage" in problems
    assert any("cycle" in p for p in problems)


def test_page_sources_need_root_and_splits():
    with pytest.raises(ValueError, match="needs `root` and `splits`"):
        RecordSource(task="page_parse", path=Path("pages.jsonl")).validate()
    RecordSource(task="crop_ocr", path=Path("crops")).validate()


def test_read_split_reads_pages_by_locked_splits(repo_tmp):
    pages = [PageRecord(f"d/p{i}", "d", "D", "S", "x.png", "", 10, 10, ()) for i in range(4)]
    write_jsonl(repo_tmp / "pages.jsonl", pages)
    assignment, info = assign_splits(pages, 1.0, 0.0, 1.0, seed=1)
    write_splits(assignment, info, repo_tmp / "splits")
    task = get_task("page_parse")
    test_pages = read_split(task, repo_tmp / "pages.jsonl", "test", repo_tmp / "splits")
    assert [p.id for p in test_pages] == ["d/p0", "d/p1", "d/p2", "d/p3"]
    assert {p.level for p in test_pages} == {"page"}
    assert read_split(task, repo_tmp / "missing.jsonl", "test", repo_tmp / "splits") is None


def test_training_run_registers_its_prompts_separately(repo_tmp):
    cfg = load_config(STAGE1A)
    cfg = replace(cfg, train=replace(cfg.train, prompts_dir=repo_tmp / "training"))
    register_training_run(cfg, STAGE1A)
    history = (repo_tmp / "training" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(history[0])
    assert entry["stage"] == "1A" and entry["starts_from"] == "base"
    assert set(entry["prompts"]) == {"crop_ocr"}
    assert entry["sources"] == [{"name": "mozhi_hi", "task": "crop_ocr", "share": 1.0}]
    runs = (repo_tmp / "training" / "runs.md").read_text(encoding="utf-8")
    assert "| 1A | stage1a_printed | base | configs/stage1/1a_printed.toml |" in runs
