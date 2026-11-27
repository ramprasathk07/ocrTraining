# dataEngine/

Turns raw data (golden pages, official public datasets) into validated **records**, and defines the **tasks** that turn a record into one training or evaluation sample. It has no model code and needs no GPU.

```bash
python -m dataEngine --config configs/data/training_data.toml golden                 # 1. golden pages → page records
python -m dataEngine --config configs/data/training_data.toml split                  # 2. lock train/dev/test
python -m dataEngine --config configs/data/training_data.toml --workers 8 crops --source NAME   # 3. crop records
```

Run the steps in that order. `--workers N` uses N CPU processes and gives the same output for any N.

## Files

| File | Does |
|---|---|
| `__main__.py` | the CLI: `golden`, `split`, `crops` steps |
| `golden.py` | reads one golden page folder → `PageRecord`; validation (boxes, labels, statuses, UTF-8); `parse_group` |
| `splits.py` | splits by writer group, so one writer's handwriting never appears in both train and test; writes once, refuses to overwrite |
| `records.py` | `PageRecord`, `Block`, `CropRecord`; JSONL read/write; allowed `SPLITS`, `SCRIPTS`, `LEVELS` |
| `tasks.py` | **training types**: `TASKS` registry (prompt, record kind, builder), `read_split` (shared by training and evaluation), `prompt_version` |
| `document_ir.py` | the grounded block format `<\|ref\|>label<\|/ref\|><\|det\|>[[x1,y1,x2,y2]]<\|/det\|>text`, boxes 0–999: `serialize` / `parse` |
| `prompt_registry.py` | prompt registry files used by training and evaluation (see `report/README.md`) |
| `text.py` | NFC normalisation, UTF-8 check |
| `parallel.py` | `parallel_map` over CPU processes |
| `sources/` | converters: one dataset → crop records (see below) |
| `data/` | **raw data, git-ignored** (contains personal information) |

## Input: `data/` (never committed)

```
data/
  train/                                     golden documents for training (configs/data/training_data.toml)
    <Region>_<Unit>_<XX>_<number>_<yy>/       one golden document
      manifest.json
      pages/page_0001/
        page.json                              size, document id
        png/page.png                           the model image
        image/page_regions.png                 boxes drawn in: never shown to a model
        layout_detection/regions.json          boxes, IndicDLP labels
        source_language_extraction/page.json   source text, reading order, status
        eng_translation/page.json              English per region
        regions/                               region crops
  test_data/
    src_golden/                              49 golden pages kept for testing only
    mozhi/                                   Mozhi printed Hindi test words (images/ + test_gt.txt)
    cvit/                                    identical copy of mozhi (unused)
```

## Output: `outputs/data/` (git-ignored)

| Step | Writes |
|---|---|
| `golden` | `golden/pages.jsonl`, `pages_set_aside.jsonl` (excluded languages), `inventory.json`, `problems.json`. It stops on any problem unless `drop_invalid = true` |
| `split` | `splits/{train,dev,test}.txt` (page ids) + `manifest.json` with hashes. To re-split, delete them by hand |
| `crops` | `crops/<source>/*.jsonl` (one `CropRecord` per line, image path relative to the folder) + `manifest.json` |

## Converters (`sources/`)

| Converter | For |
|---|---|
| `golden_regions` | golden page blocks → region crops, split by the locked splits |
| `labelled_folder` | any official archive laid out as images plus a label file per split (Mozhi, IIIT-HW-Dev, …): set `root`, `label_files`, `separator` from the archive's README |

Converter signature: `convert(source: dict, out_dir: Path, workers: int) -> int`. Register it in `sources/__init__.py: CONVERTERS`.

## Tasks (training types)

| Task | Reads | Prompt | Target |
|---|---|---|---|
| `crop_ocr` | crop folder | `<image>\nFree OCR.` | the crop's text |
| `page_parse` | `pages.jsonl` + splits + image root | `<image>\n<\|grounding\|>Convert the document to markdown.` | the page's blocks in `document_ir` format |

To add a training type:
1. Write `build(record, root) -> TaskSample` in `tasks.py`.
2. Add `Task(build, "crops" | "pages", PROMPT)` to `TASKS`.

Training, evaluation and the prompt registries pick it up by name. Changing a prompt's text gives it a new version, and a model trained on the old prompt is then flagged in eval reports.

Public data must come only from official publishers. The rules are in the root README, under "Data and privacy".
