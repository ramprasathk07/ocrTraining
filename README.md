# HindiDoc-OCR

Fine-tune **DeepSeek-OCR 2 (3B)** to read **Hindi + English police documents (FIRs)**, printed and handwritten. For each page it returns **grounded bilingual blocks**: box, layout label, reading order, source text and English translation. A separate renderer rebuilds the translated PDF from them.

> **Status (2026-11-19):** switched to DeepSeek-OCR 2. The data engine, Stage 1 (1A printed → 1B handwritten → 1C joint) training and the before/after evaluation pipeline are built and tested on CPU. Nothing has been trained yet; the untrained baseline is evaluated on the test data. Next steps: the training smoke tests (`--smoke`) on the GPU, and the official public-data downloads.

| Milestone | Date | Deliverable |
|---|---|---|
| M1 | 2026-12-20 | Hindi FIR reader: reading + layout, compared with the untrained model |
| M2 | 2027-01-17 | Reader + translator in one model, final report |

## How it works

Training runs in stages. Each stage adds a fresh LoRA adapter to the previous merged checkpoint, then merges it in turn. Stage 1 trains LoRA on DeepEncoder V2 attention (r8) and decoder attention (r16). SAM, the conv downsampler, `projector`, router, experts, embeddings and LM head stay frozen.

| Stage | What the model learns | Doc |
|---|---|---|
| 0 | Nothing (data, splits, baseline, smoke tests) | [stage-0](docs/stages/stage-0-data-and-baseline.md) |
| 1 | Read Devanagari + handwritten English: 1A printed → 1B handwritten → 1C joint with golden regions | [stage-1](docs/stages/stage-1-script-adaptation.md) |
| 2 | Whole FIR pages: boxes, labels, reading order, text | [stage-2](docs/stages/stage-2-document-understanding.md) |
| 3 | Hindi + translation, text only (decoder branch) | [stage-3](docs/stages/stage-3-language-branch.md) |
| 4 | Reading + translation in one model | [stage-4](docs/stages/stage-4-joint-ocr-translation.md) |
| 5 | Final test, compression, renderer hand-off | [stage-5](docs/stages/stage-5-release.md) |

The full plan, with decisions, datasets, gates and timeline, is in [docs/training-plan.md](docs/training-plan.md). The tokenizer findings are in [docs/tokenizer-check.md](docs/tokenizer-check.md).

## Repository layout

```
dataEngine/           raw data → validated records
  golden.py             golden FIR pages → page records + validation
  splits.py             train/dev/test by police station, locked once written
  tasks.py              training types: task registry (prompt, record kind) + shared record reader
  document_ir.py        grounded block format: serialize / parse
  prompt_registry.py    prompt registry files (training and evaluation each keep one folder)
  sources/              converter registry: one dataset → crop records
models/deepseek_ocr/  model input encoding, loading, generation (shared by training + evaluation)
training/             config → deterministic data plan → LoRA training (1..N GPUs)
  config.py             [stage], [data], [lora], [train], [smoke] of a run config
  stages.py             python -m training.stages: the stage tree from the configs
  lora.py / inventory.py / smoke.py / prompts.py
evaluation/           before/after runner (sharded), page metrics, report with gates, prompt leaderboard
configs/
  data/                 data engine: training_data.toml, test_data.toml
  stage1/               one run config per sub-stage: 1a_printed, 1b_handwritten, 1c_joint
  eval/                 evaluation-only runs: test_data.toml
tests/                pytest, CPU only, no model download
docs/                 plan, stage docs, tokenizer check
outputs/              data, checkpoints, predictions (git-ignored)
report/               evaluation reports (git-ignored); report/prompts/ tracked
```

Each folder has its own README: [configs](configs/README.md) · [dataEngine](dataEngine/README.md) · [models](models/README.md) · [training](training/README.md) · [evaluation](evaluation/README.md) · [report](report/README.md) · [tests](tests/README.md) · [docs](docs/README.md).

## Stages and training types

**Training type = a task in `dataEngine/tasks.py`.** Each task defines:
- its prompt;
- the record kind it reads;
- how a record becomes (prompt, target, image).

| Task | Records | Used in |
|---|---|---|
| `crop_ocr` | crop folders (`python -m dataEngine … crops`) | Stage 1A / 1B / 1C |
| `page_parse` | `pages.jsonl` + locked splits + image root | Stage 2, page evaluation |

**Stage or sub-stage = one run config in `configs/stage<N>/`.** Its `[stage]` section names it and links it to the stage it starts from:

```toml
[stage]
id = "1B"
description = "handwritten Hindi + English crops, separate adapter on 1A"
starts_from = "1A"   # [model].name_or_path is 1A's merged output; "base" = original OCR-2
```

`python -m training.stages` prints the tree, with each stage's tasks, data mix and status (merged output, report):

```
base  (original model)
    └─ 1A: printed Hindi crops (public data only)
        └─ 1B: handwritten Hindi + English crops, separate adapter on 1A
            └─ 1C: joint printed + handwritten replay with golden FIR regions
```

**How to add things:**

| To add | Do |
|---|---|
| A sub-stage | Add a config to its stage folder |
| A new stage | Add a new `configs/stage<N>/` folder |
| A new training type | Register one function in `TASKS` |

Data sources can mix tasks and record kinds. Each source can set its own `image` mode (e.g. tiles for pages).

## Setup

Python 3.12.

**CPU / data machine:**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

**GPU machine (CUDA 11.8):**

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .[train,dev]
pip install flash-attn==2.7.3 --no-build-isolation      # or set attn_implementation = "eager"
```

The model's own code needs transformers **4.46.3**, which is pinned. Do not upgrade it.

## Quickstart

```bash
# 1. Data engine (add --workers N to use N CPU processes)
python -m dataEngine --config configs/data/training_data.toml golden                          # validate golden pages
python -m dataEngine --config configs/data/training_data.toml split                           # lock train/dev/test
python -m dataEngine --config configs/data/training_data.toml crops --source golden_regions   # region crops

# 2. Check what a config will train (builds the model without weights)
python -m training.inventory --config configs/stage1/1a_printed.toml

# 3. Smoke tests on one GPU (needs only the golden crops)
torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --smoke

# 4. Evaluate before training (the run's starting model)
torchrun --nproc_per_node=N -m evaluation --config configs/stage1/1a_printed.toml --checkpoint start

# 5. Train: the same command on 1 or N GPUs (needs every source in the config)
torchrun --nproc_per_node=N -m training --config configs/stage1/1a_printed.toml

# 6. Evaluate after training: before vs after, with gates
torchrun --nproc_per_node=N -m evaluation --config configs/stage1/1a_printed.toml \
    --checkpoint start --checkpoint outputs/runs/stage1a_printed/merged

# 7. Next sub-stage starts from the best checkpoint, merged
python -m training --config configs/stage1/1a_printed.toml --merge outputs/runs/stage1a_printed/checkpoint-3000
```

**Outputs:**

| Step | Where |
|---|---|
| Data | `outputs/data/` |
| Training | `outputs/runs/<run>/`: checkpoints, TensorBoard logs, `run_manifest.json`, `smoke.json`, `adapter/`, `merged/` |
| Evaluation | predictions per checkpoint in `<checkpoint>/eval/<split>/` (`start` and hub ids under `[eval].output_dir`), private; report in `report/<run>/report_<split>.md` + `.json` (git-ignored: comparisons quote FIR text) |

## Configuration

Every run is one TOML file in `configs/`: data engine (`configs/data/`), one training config per stage/sub-stage (`configs/stage<N>/`), evaluation-only suites (`configs/eval/`). Section-by-section reference, image modes and how to add a run: [configs/README.md](configs/README.md). A misspelled key fails loudly; nothing falls back to a hidden default.

## Extending

| To add | Do |
|---|---|
| A public dataset in "images + label file" form | Add a `[[sources]]` block with `converter = "labelled_folder"` in `configs/data/training_data.toml` |
| A dataset in a new format | Write `convert(source, out_dir, workers)` in `dataEngine/sources/` and register it in `CONVERTERS` |
| A new task (prompt/target) | Write one function in `dataEngine/tasks.py` and register it in `TASKS` |
| A run variant (data mix, LoRA targets, image mode) | Copy a run TOML and change values. No code |

## Evaluation before and after training

- The first `--checkpoint` is "before"; every later one is compared with it. `start` = the config's starting model; a folder or hub id also works.
- Per source, script and language: CER, WER, exact match, change vs before, gates, most improved/regressed items. On pages: page CER, layout AP/mAP@50, box IoU, block F1, reading order, valid output.
- Predictions are cached per checkpoint while items, image mode, generation settings and weights are unchanged.

**Before training, on the test data in `dataEngine/data/test_data`:**

```bash
python -m dataEngine --config configs/data/test_data.toml golden
python -m dataEngine --config configs/data/test_data.toml split
python -m dataEngine --config configs/data/test_data.toml --workers 8 crops --source golden_regions
python -m dataEngine --config configs/data/test_data.toml --workers 8 crops --source printed_words
torchrun --nproc_per_node=N -m evaluation --config configs/eval/test_data.toml --split test --checkpoint start
```

Flags, metric definitions, output paths and caching: [evaluation/README.md](evaluation/README.md). Baseline results (2026-11-19): `report/test_data/` (local only).

## Prompt versions

Prompts are defined once, in `dataEngine/tasks.py`; a version is a hash of the exact text. Training runs register theirs in `report/prompts/training/` (`runs.md`), evaluations in `report/prompts/eval/` (`leaderboard.md`), and reports flag a checkpoint evaluated with a prompt it was not trained on. Details: [report/README.md](report/README.md).

## Multi-GPU

- The GPU count comes only from the launch command (`torchrun --nproc_per_node=N`).
- The config sets the **global** batch, and gradient accumulation is derived from the GPU count. 1 GPU and 8 GPUs train the same recipe.
- The whole sample plan (shuffle, shares, curriculum, epochs) is built once from the seed. Each rank reads its own slice, so every GPU count sees identical data.
- Only rank 0 writes checkpoints and reports. Evaluation splits items across ranks and gathers them on rank 0.
- CUDA and ROCm machines never share one job.

## Data and privacy

- **Golden FIR data never enters git.** `dataEngine/data/` and `outputs/` are git-ignored, because the documents contain names, addresses and caste information. Checkpoints trained on it stay private too.
- **Public data comes only from official publishers:** AI4Bharat, IIIT-Hyderabad CVIT / `ilocr.iiit.ac.in` (MeitY–Bhashini), or the dataset authors' own release. No third-party re-uploads. Record the URL, licence and SHA-256 of each archive at download.
- Several datasets are licensed for research use only, and this project is **research / internal use**.

## Development

```bash
ruff check . && ruff format --check . && pytest
```

Code rules for this repo live in [.claude/skills/ocr-pipeline/SKILL.md](.claude/skills/ocr-pipeline/SKILL.md):
- no duplicated logic
- no hidden defaults
- simple code
- 1→N GPU scaling
- a Decisions log

Read them before changing code.
