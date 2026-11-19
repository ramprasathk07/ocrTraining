# configs/

Every pipeline run is a TOML file here. Code holds no run settings. To change a run, edit or copy its config.

| Folder | What it holds | Read by |
|---|---|---|
| `data/` | data engine configs: where raw data is and where records go | `python -m dataEngine` |
| `stage1/` | one run config per Stage 1 sub-stage | `python -m training`, `python -m evaluation`, `python -m training.stages` |
| `eval/` | evaluation-only runs (no training) | `python -m evaluation` |

Rules for every config:
- **Paths are relative to the repo root.** Run every command from there.
- **Misspelled keys fail.** There is no hidden default.
- **`name` names the run.** It appears in output folders, reports and the prompt registries.

## data/: data engine

| File | Input | Output |
|---|---|---|
| `training_data.toml` | `dataEngine/data/` (golden FIRs) + official public downloads | `outputs/data/{golden,splits,crops/<source>}` |
| `test_data.toml` | `dataEngine/data/test_data/` (49 golden pages + Mozhi test words) | `outputs/data/test/...` |

| Section | Holds |
|---|---|
| `seed` | used for the split |
| `[golden]` | `data_root`, `out_dir`, `exclude_langs` (pages set aside, e.g. Marathi), `drop_invalid` |
| `[golden.station_overrides]` | document ids that don't follow `<District>_<Station>_PS_<case>_<yy>`, mapped to `[district, station]` |
| `[split]` | `test_fraction`, `dev_fraction`, `max_group_fraction` (stations larger than this are split by document) |
| `[[sources]]` | one block per dataset: `name`, `converter` (see `dataEngine/sources/`), `out_dir` + converter settings |

The `test_data.toml` overrides are readings of the file names. `Rail_Patna_Rail_Patna_cyber_PS_09_23` is still unconfirmed.

```bash
python -m dataEngine --config configs/data/training_data.toml golden
python -m dataEngine --config configs/data/training_data.toml split
python -m dataEngine --config configs/data/training_data.toml --workers 8 crops --source golden_regions
```

## stage1/: training runs

Each file is one sub-stage. Each one starts from the merged output of the stage named in `starts_from`:

| File | `[stage].id` | starts from | trains on |
|---|---|---|---|
| `1a_printed.toml` | 1A | `base` (original OCR-2) | Mozhi printed Hindi |
| `1b_handwritten.toml` | 1B | 1A → `outputs/runs/stage1a_printed/merged` | IIIT-HW-Dev, DohaScript, IAM |
| `1c_joint.toml` | 1C | 1B → `outputs/runs/stage1b_handwritten/merged` | golden regions 40% + replay |

`python -m training.stages` prints this tree from the files and checks the chain.

| Section | Holds |
|---|---|
| `name` | run name; output goes to `[train].output_dir` |
| `[stage]` | `id`, `description`, `starts_from` (a stage id, or `base`) |
| `[model]` | `name_or_path` (hub id, or the parent's `merged/` folder), pinned `revision`, `attn_implementation` (`flash_attention_2` or `eager`) |
| `[image]` | default image mode (see below) |
| `[lora]` | default group: `r`, `alpha`, `dropout`, `target_modules` (regex on module names) |
| `[[lora.groups]]` | extra groups inside the targets: `pattern`, `r`, `alpha`, `learning_rate` |
| `[train]` | `output_dir`, `prompts_dir`, `global_batch_size` (same on any GPU count), `per_device_batch_size`, LR schedule, `save_steps`, … |
| `[data]` | `epoch_size`, `num_epochs`, `dev_size`, `max_target_tokens` (longer targets are dropped and counted) |
| `[[data.phases]]` | curriculum: `fraction` of epoch 1 and the allowed `levels` (`word`, `line`, `region`, `page`) |
| `[[data.sources]]` | `name`, `task` (`crop_ocr` / `page_parse`), `path`, `share`; page tasks also need `root` and `splits`; optional own `image` |
| `[smoke]` | smoke-test data (`path`, `task`, `samples`, `steps`) |
| `[eval]` | `output_dir` (predictions), `report_dir`, `prompts_dir`, `no_repeat_ngram_size`, `seed` |
| `[[eval.sources]]` | `name`, `task`, `path`, `limit`, `max_new_tokens`; optional `image`, `root`, `splits` |
| `[[eval.gates]]` | `group` (e.g. `source=mozhi_hi`, `script=handwritten`), `metric`, plus one of `max_value`, `min_value`, `max_absolute_change`, `max_relative_change` |

All three Stage 1 configs share one eval suite, so their numbers are comparable.

## eval/: evaluation-only runs

`test_data.toml` scores a checkpoint on the curated test data, before or after training. It has `name`, `[model]`, `[image]`, `[eval]`, `[[eval.sources]]` and `[[eval.gates]]`, and no `[stage]`, `[train]` or `[data]`.

```bash
torchrun --nproc_per_node=N -m evaluation --config configs/eval/test_data.toml --split test \
    --checkpoint start --checkpoint outputs/runs/stage1a_printed/merged
```

## Image modes

These are the only inputs DeepSeek-OCR 2 accepts.

| Mode | `base_size` / `image_size` / `crop_mode` | Vision positions | Use |
|---|---|---|---|
| 768 view (stretched) | 1024 / 768 / false | 145 | word and line crops |
| 1024 view (padded) | 1024 / 1024 / false | 257 | small crops without stretching |
| tiles | 1024 / 768 / true | 257 + 2–6 tiles × 144 | paragraph regions, full pages |

A source can override the run's mode with `image = { base_size = …, image_size = …, crop_mode = … }`.

## How to add

| To add | Do |
|---|---|
| A sub-stage (e.g. 1D) | Copy a `stage1/` file: new `name`, `[stage]` id and `starts_from`, `[model].name_or_path` = the parent's `merged/` |
| A new stage | Create `configs/stage2/` and put its sub-stage configs there |
| A dataset | Add a `[[sources]]` block to `data/training_data.toml`, convert it, then reference its `out_dir` in a run's `[[data.sources]]` |
| An eval suite | Add a file to `eval/` |
