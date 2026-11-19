---
name: ocr-pipeline
description: Rules for building this repo's OCR / document-intelligence data, training and evaluation pipelines. Use for any code change, new module, refactor, review or design question in this project (dataEngine, models, training, evaluation, configs, tests).
---

# OCR Pipeline Rules

This repo builds three pipelines for an OCR document-intelligence model:

1. **Data** (`dataEngine/`): raw documents → validated, preprocessed, split datasets.
2. **Training** (`training/`): dataset + config → checkpoint.
3. **Evaluation** (`evaluation/`): checkpoint + held-out split → metrics report.

Every rule serves three goals: **no redundant code**, **no hidden assumptions**, **nothing more complex than the task needs**. If a request conflicts with these rules, ask the user before proceeding.

## 1. Never assume. Ask.

The user resolves ambiguity, not Claude. Never fill a gap with a guess. If code needs a value or choice nobody gave, stop and ask.

- Ask with AskUserQuestion: short questions, batched, each with a recommended option and one line on its trade-off.
- Record every answer in **Decisions** (section 8) so it is never asked twice.
- Explain in plain language: what you will do, why, and what it affects. Define any ML term the user may not know.

Open until answered (check section 8 first):

- Task output: plain text, text + layout boxes, key-value fields, tables, or a mix.
- Model: architecture or base checkpoint; train from scratch or fine-tune.
- Framework and Python version.
- Data: sources, file types (PDF, images), annotation format, licence, size.
- Split strategy and sizes.
- Metrics and target values.
- Compute: GPU/CPU, memory, expected training time.
- Config format, experiment tracking, lint/format/test tools.
- Naming convention: existing folder `dataEngine` is camelCase; Python convention is snake_case. Pick one and use it everywhere.

## 2. Structure

Target layout. Change it only with user approval.

```
ocrTraining/
├── dataEngine/    # ingest → validate → preprocess → split → export
├── models/        # model definition only
├── training/      # training loop, checkpointing
├── evaluation/    # metrics, eval runner, reports
├── configs/       # one file per run: data, model, train, eval settings
├── tests/         # mirrors the source layout
├── outputs/       # generated data, checkpoints, reports (gitignored)
└── README.md      # one command per pipeline, how to run it
```

- One responsibility per folder. Data code never trains; training never writes final reports; evaluation never trains.
- Imports flow one way: `training` and `evaluation` import from `dataEngine` and `models`. `dataEngine` and `models` import from neither. `training` may import metric functions from `evaluation` for validation logs.
- Each pipeline has exactly one entry point that takes a config path, e.g. `python -m training --config configs/stage1/1a_printed.toml`.
- Generated files never go inside source folders.
- **Easy to extend.** A new task, dataset or variation is one new function, registered by name, plus a config entry. Never copy a script or fork a pipeline to make a variant.

## 3. No redundant code

- One implementation of each thing. Before writing a function, search the repo for an existing one and reuse or extend it.
- Shared logic lives with its owner: loading and preprocessing in `dataEngine`, model in `models`, metrics in `evaluation`.
- Preprocessing used at training and evaluation time is the same function, never re-implemented. Two copies drift and silently skew results.
- A second copy of any logic means extract it to its owner module now, not later.
- Every setting (paths, hyperparameters, thresholds) is defined once, in config. Code reads it; code never re-declares its own defaults.
- Delete dead code, unused parameters and commented-out code.
- No wrapper that only renames or forwards a call.

## 4. Keep it simple

- Plain functions first. A class only when it holds state (model, dataset).
- No base classes, registries, factories or plugin systems unless two real uses exist and the user agrees.
- Use the framework's own features (data loaders, schedulers, checkpoint saving) before writing your own.
- No new dependency without user approval.
- No options, flags or hooks "for later". Build what is needed now.
- Short functions, clear names, flat code. A reader should understand a file in one pass.

## 5. Code standards

- Type hints on every function signature. A one-line docstring on public functions: what it does, plus the non-obvious why.
- Code passes the lint, format and test tools recorded in Decisions before it is done.
- Validate at boundaries (raw files, annotations, configs) with clear error messages. Never silently drop bad samples: skip them only if config allows, and log the count and reason.
- No hardcoded paths, secrets or magic numbers.
- Use `logging`, not `print`, in pipeline code.
- Tests: every parser, transform, split function and metric gets a small unit test with tiny fixtures in `tests/`. Test metrics against hand-computed values.
- **Scale from 1 to N GPUs (or CPU workers) with no code change.** The count comes from the launch command (`torchrun --nproc_per_node=N` / `accelerate launch`) or a `--workers N` flag, never from the code.
  - No hardcoded devices (`cuda:0`, `.to("cuda")`). Use the device the launcher assigns.
  - Config sets the **global** batch size. Code derives gradient accumulation from the world size, so 1 GPU and 8 GPUs train the same recipe.
  - Data is a deterministic, seeded list of samples. Each rank or worker reads its own shard (DistributedSampler for training, `worker_id` slices for data jobs). Never let two ranks draw their own random mixtures.
  - Only rank 0 writes checkpoints, logs and reports. Evaluation is sharded across ranks and gathered on rank 0.
  - One job never mixes GPU stacks (CUDA and ROCm run as separate jobs).

## 6. ML correctness

- Reproducible runs: seed everything from config. Save the resolved config, code version and dataset version next to every checkpoint and report.
- Splits are created once by `dataEngine`, saved to disk and deterministic.
- Split by document, never by page or crop, so no document appears in two splits.
- The test split is never used for training, tuning or early stopping. Tune on validation only.
- Each data stage writes its output plus a small manifest (sample counts, source, version). The next stage reads those files, not in-memory state.
- Evaluation computes the same metrics the same way every run, so runs stay comparable.

## 7. Before calling a change done

- [ ] Searched for existing code first; no duplicated logic.
- [ ] Nothing assumed; every open question answered by the user and logged in section 8.
- [ ] Lint, format and tests pass.
- [ ] Runs on 1 and N GPUs (or workers) with the same config and the same effective batch.
- [ ] README run commands still correct.
- [ ] It is the simplest version that does the job.
- [ ] Told the user what changed, what was skipped, and any open question.

## 8. Decisions

Record each answer here when the user gives it. Never re-ask a recorded decision; change one only when the user says so.

| Decision | Answer | Date |
|----------|--------|------|
| Plan | `docs/training-plan.md` + `docs/stages/` are the source of truth for stages, data and gates | 2026-11-18 |
| Base model | DeepSeek-OCR 2 (3B), revision aaa02f3811945a91062062994c5c4a3f4c0af2b0. Fine-tune only, never from scratch | 2026-11-19 |
| Stage 1 structure | 1A printed (public only) → 1B handwritten, separate adapter on 1A merged (IIIT-HW-Dev, DohaScript, IAM English) → 1C joint on 1B merged (golden regions only here, balanced replay) | 2026-11-19 |
| Languages (V1) | Hindi + English only (translation target English). Marathi deferred to V2 | 2026-11-18 |
| Time box | M1 results by 2026-12-20, M2 final by 2027-01-17 (hard limit) | 2026-11-18 |
| Task output (V1) | Grounded bilingual blocks: box (0–999) + IndicDLP label + reading order + source + English. No key-value fields | 2026-11-18 |
| Golden data | ~3K pages of Bihar police FIRs/complaints, fully human-checked = ground truth | 2026-11-18 |
| Licence use | Research / internal only. Academic-licence datasets allowed | 2026-11-18 |
| Training method | LoRA per stage + full `projector`. No whole-model full fine-tune. Tokenizer unchanged (see `docs/tokenizer-check.md`) | 2026-11-18 |
| Split | By police station (fallback: by document). Test used only for the M1/M2 reports | 2026-11-18 |
| Compute | A6000 48GB primary. MI300X after a ROCm smoke test. RTX 3060 for data/smoke tests | 2026-11-18 |
| Training framework | HF Trainer + PEFT on the official checkpoint (replaces "Unsloth first"; Unsloth only if too slow). Pinned in `pyproject.toml`: py 3.12, torch 2.6.0, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1 | 2026-11-18 |
| Trainable parts (Stage 1) | LoRA on DeepEncoder V2 attention (r8, α16, lr 5e-5) + decoder attention (r16, α32, lr 1e-4) = 3,047,424 params. SAM, convs, projector, experts, router, embeddings, LM head frozen. Loader disables the official `no_grad` on the vision path | 2026-11-19 |
| Evaluation | Before/after runner: first `--checkpoint` is before; one shared eval suite per stage; report with gates per config | 2026-11-19 |
| Test data + page metrics | `dataEngine/data/test_data` (Mozhi printed words, 49 golden FIR pages) is evaluation-only. Page metrics: page CER, AP/mAP@50 (output-order ranking), box IoU, block F1, reading order, valid output. Field F1 not measurable (no labels); translation reviewed by hand from Stage 4 | 2026-11-19 |
| Stage structure | One run config per stage/sub-stage in `configs/stage<N>/` with `[stage] id, description, starts_from`; `python -m training.stages` is the tree. Training type = task in `dataEngine/tasks.py` (prompt + record kind); training and evaluation share `read_split` | 2026-11-19 |
| Prompt registries | Training: `report/prompts/training/` (runs.md). Evaluation: `report/prompts/eval/` (leaderboard.md). Versions are prompt-text hashes; reports flag training/eval prompt mismatch | 2026-11-19 |
| Station mapping | Document ids that do not fit `<District>_<Station>_PS_<case>_<yy>` go in `[golden.station_overrides]`; the 3 in configs/data/test_data.toml are unconfirmed readings | 2026-11-19 |
| Naming | Keep `dataEngine/` folder; all modules and functions snake_case | 2026-11-18 |
| Config format | TOML (stdlib `tomllib`), one file per run in `configs/` | 2026-11-18 |
| Dev tools | ruff (lint + format), pytest. Training logs: TensorBoard via the Trainer | 2026-11-18 |
| Public data sources | Trusted publishers only: AI4Bharat, IIIT-H CVIT / NLTM ilocr (MeitY-Bhashini), Meta FLORES, or the dataset authors' own release. No third-party re-uploads. Record URL, licence, SHA-256 per archive | 2026-11-18 |
| Extensibility + scale | New task/variation = one registered function + config entry. All code runs 1→N GPUs/workers from the launch command; config sets the global batch | 2026-11-18 |
