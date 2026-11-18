# HindiDoc-OCR

Fine-tune **DeepSeek-OCR (3B)** to read **Hindi + English police documents (FIRs)**, printed and handwritten. For each page it returns **grounded bilingual blocks**: box, layout label, reading order, source text and English translation. A separate renderer rebuilds the translated PDF from them.

> **Status (2026-11-18):** plan approved. The data engine and the Stage 1 training and evaluation code are built and tested on CPU. Nothing has been trained yet. Next steps: the Stage 0 smoke tests on the GPU, and the official public-data downloads.

| Milestone | Date | Deliverable |
|---|---|---|
| M1 | 2026-12-20 | Hindi FIR reader: reading + layout, compared with the untrained model |
| M2 | 2027-01-17 | Reader + translator in one model, final report |

## How it works

Training runs in stages. Each stage adds a fresh LoRA adapter to the previous merged checkpoint, then merges it in turn. The vision encoder, `projector`, router and routed experts stay frozen.

| Stage | What the model learns | Doc |
|---|---|---|
| 0 | Nothing (data, splits, baseline, smoke tests) | [stage-0](docs/stages/stage-0-data-and-baseline.md) |
| 1 | Read Devanagari: printed and handwritten crops | [stage-1](docs/stages/stage-1-script-adaptation.md) |
| 2 | Whole FIR pages: boxes, labels, reading order, text | [stage-2](docs/stages/stage-2-document-understanding.md) |
| 3 | Hindi + translation, text only (decoder branch) | [stage-3](docs/stages/stage-3-language-branch.md) |
| 4 | Reading + translation in one model | [stage-4](docs/stages/stage-4-joint-ocr-translation.md) |
| 5 | Final test, compression, renderer hand-off | [stage-5](docs/stages/stage-5-release.md) |

The full plan, with decisions, datasets, gates and timeline, is in [docs/training-plan.md](docs/training-plan.md). The tokenizer findings are in [docs/tokenizer-check.md](docs/tokenizer-check.md).

## Repository layout

```
dataEngine/         raw data → validated records
  golden.py           golden FIR pages → page records + validation
  splits.py           train/dev/test by police station, locked once written
  tasks.py            task registry: record → (prompt, target, image)
  sources/            converter registry: one dataset → crop records
models/deepseek_ocr/  model input encoding, loading, generation (shared by training + evaluation)
training/           config → deterministic data plan → LoRA training (1..N GPUs) + smoke tests
evaluation/         CER/WER + sharded generate-and-score runner
configs/            data.toml (data engine), one TOML per training run
tests/              pytest, CPU only, no model download
docs/               plan, stage docs, tokenizer check
outputs/            everything generated (git-ignored)
```

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
python -m dataEngine --config configs/data.toml golden                          # validate golden pages
python -m dataEngine --config configs/data.toml split                           # lock train/dev/test
python -m dataEngine --config configs/data.toml crops --source golden_regions   # region crops

# 2. Smoke tests on one GPU (needs only the golden crops)
torchrun --nproc_per_node=1 -m training --config configs/stage1_script.toml --smoke

# 3. Train: the same command on 1 or N GPUs (needs every source in the config)
torchrun --nproc_per_node=N -m training --config configs/stage1_script.toml

# 4. Evaluate: no --checkpoint = the untrained base model
torchrun --nproc_per_node=N -m evaluation --config configs/stage1_script.toml --checkpoint outputs/runs/stage1_script/checkpoint-1000
```

**Outputs:**

| Step | Where |
|---|---|
| Data | `outputs/data/` |
| Training | `outputs/runs/<run>/`: checkpoints, TensorBoard logs, `run_manifest.json`, `smoke.json`, `adapter/`, `merged/` |
| Evaluation | `<checkpoint>/eval_<split>/`: `predictions.jsonl` + `scores.json` |

## Configuration

**`configs/data.toml`:**

| Section | Holds |
|---|---|
| `[golden]` | golden data root, languages to set aside |
| `[split]` | split fractions |
| `[[sources]]` | one block per dataset converter |

**`configs/<run>.toml`:**

| Section | Holds |
|---|---|
| `[model]` | checkpoint + pinned revision |
| `[image]` | the model's own image modes |
| `[lora]` | rank + target-module regex |
| `[train]` | global batch, learning rate, … |
| `[data]` | sources with shares, curriculum, epochs |
| `[smoke]` | smoke-test settings |
| `[eval]` | evaluation settings |

A misspelled key fails loudly; nothing falls back to a hidden default.

## Extending

| To add | Do |
|---|---|
| A public dataset in "images + label file" form | Add a `[[sources]]` block with `converter = "labelled_folder"` in `configs/data.toml` |
| A dataset in a new format | Write `convert(source, out_dir, workers)` in `dataEngine/sources/` and register it in `CONVERTERS` |
| A new task (prompt/target) | Write one function in `dataEngine/tasks.py` and register it in `TASKS` |
| A run variant (data mix, LoRA targets, image mode) | Copy a run TOML and change values. No code |

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
