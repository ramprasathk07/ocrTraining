import json

import pytest
from conftest import GOLDEN, needs_golden
from PIL import Image

from dataEngine.golden import convert_document
from dataEngine.records import read_crops, write_jsonl
from dataEngine.sources import golden_regions, labelled_folder
from dataEngine.splits import assign_splits, write_splits
from dataEngine.tasks import FREE_OCR_PROMPT, crop_ocr
from dataEngine.text import normalize_text


def _toy_archive(root, missing_image=False):
    (root / "img").mkdir(parents=True)
    for i in range(3):
        if not (missing_image and i == 1):
            Image.new("RGB", (20, 10), "white").save(root / "img" / f"{i}.png")
    (root / "train.txt").write_text("img/0.png\tनमस्ते\nimg/1.png\tथाना\n", encoding="utf-8")
    (root / "val.txt").write_text("img/2.png\tक़\n", encoding="utf-8")
    return {
        "name": "toy",
        "root": str(root),
        "label_files": {"train": "train.txt", "dev": "val.txt"},
        "separator": "\t",
        "langs": ["hi"],
        "script": "handwritten",
        "level": "word",
    }


def test_labelled_folder_converts_every_split(repo_tmp):
    source = _toy_archive(repo_tmp / "raw")
    out = repo_tmp / "crops"
    assert labelled_folder.convert(source, out, workers=2) == 3
    records = read_crops(out)
    assert [(r.id, r.split) for r in records] == [
        ("toy/dev/0000001", "dev"),
        ("toy/train/0000001", "train"),
        ("toy/train/0000002", "train"),
    ]
    assert records[0].text == normalize_text("क़")
    assert (out / records[1].image).resolve() == (repo_tmp / "raw" / "img" / "0.png").resolve()


def test_labelled_folder_fails_loudly_then_drops_only_when_allowed(repo_tmp):
    source = _toy_archive(repo_tmp / "raw", missing_image=True)
    out = repo_tmp / "crops"
    with pytest.raises(ValueError, match="1 invalid rows"):
        labelled_folder.convert(source, out, workers=1)
    assert labelled_folder.convert({**source, "drop_invalid": True}, out, workers=1) == 2
    problems = json.loads((out / "problems.json").read_text(encoding="utf-8"))
    assert problems[0]["line"] == 2 and problems[0]["split"] == "train"


@needs_golden
def test_golden_pages_to_crop_records(repo_tmp):
    documents = sorted(p for p in GOLDEN.iterdir() if (p / "manifest.json").is_file())
    pages, problems = [], []
    for document in documents:
        doc_pages, doc_problems = convert_document(document, GOLDEN)
        pages += doc_pages
        problems += doc_problems
    assert problems == []
    write_jsonl(repo_tmp / "pages.jsonl", pages)
    assignment, info = assign_splits(pages, 0.15, 0.15, 0.15, seed=1)
    write_splits(assignment, info, repo_tmp / "splits")

    source = {
        "name": "golden_regions",
        "pages": str(repo_tmp / "pages.jsonl"),
        "splits": str(repo_tmp / "splits"),
        "data_root": str(GOLDEN),
    }
    out = repo_tmp / "crops"
    assert golden_regions.convert(source, out, workers=1) == sum(len(p.blocks) for p in pages)
    records = read_crops(out)
    assert all((out / r.image).is_file() for r in records)
    sample = crop_ocr(records[0], out)
    assert sample.prompt == FREE_OCR_PROMPT
    assert sample.target == records[0].text
