# evaluation/

Scores one or more checkpoints on fixed, seeded samples, compares each with the first ("before"), checks gates and writes a report. The same command runs on 1 or N GPUs; items are sharded across ranks.

```bash
# Baseline on the curated test data (convert it first: configs/data/test_data.toml)
torchrun --nproc_per_node=N -m evaluation --config configs/eval/test_data.toml --split test --checkpoint start

# Before vs after a stage
torchrun --nproc_per_node=N -m evaluation --config configs/stage1/1a_printed.toml \
    --checkpoint start --checkpoint outputs/runs/stage1a_printed/merged
```

| Flag | Meaning |
|---|---|
| `--config` | any config with `[model]` and `[eval]` (a stage config or `configs/eval/*.toml`) |
| `--checkpoint` | repeatable. Accepts `start` (the config's `[model]`), a folder (a LoRA adapter loads on top of `start`; anything else is a full model) or a hub id. The first one is "before" |
| `--split` | `dev` (default, for development) or `test` (for reports only) |

## Files

| File | Does |
|---|---|
| `__main__.py` | runner: selects items, generates per checkpoint, caches, writes the report and prompt history |
| `metrics.py` | corpus CER / WER (edits ÷ reference length; can exceed 100% when the model over-generates) |
| `page_metrics.py` | page CER, layout AP@50 (with and without labels), mean box IoU, block P/R/F1 @ IoU 0.5, reading-order accuracy, valid output |
| `report.py` | groups scores (by source, script, language), gates, biggest changes, markdown report |
| `prompts.py` | eval prompt registry + leaderboard (`report/prompts/eval/`); warns when a checkpoint is evaluated with a prompt it wasn't trained on |

## Metrics

| Task | Metric | Notes |
|---|---|---|
| `crop_ocr` | CER, WER, exact match | per source, script, language, script × language |
| `page_parse` | page CER | whitespace collapsed, HTML tags stripped; output without blocks is scored as plain text |
| | layout AP@50 / mAP@50 | ranked by output order; mAP needs IndicDLP labels (only after Stage 2) |
| | mean IoU, block F1@0.5 | greedy box matching |
| | reading order | pairwise accuracy over matched blocks only |
| | valid output | parses cleanly **and** stopped at EOS |
| not measured yet | field F1, translation | no field labels; translation comes in Stage 4 |

Full definitions: [plan §9](../docs/training-plan.md#9-evaluation-and-gates).

## Where results go

| Path | Content | In git? |
|---|---|---|
| `[eval].output_dir/<checkpoint>/<split>/` for `start` and hub ids; `<checkpoint>/eval/<split>/` for local folders | `predictions.jsonl` (raw FIR text), `scores.json`, `fingerprint.txt` | no (private) |
| `[eval].report_dir/report_<split>.md` + `.json` | scores, gates, prompts, the most improved and regressed items | no (quotes FIR text) |
| `[eval].prompts_dir` = `report/prompts/eval/` | prompt registry, history, leaderboard | yes |

**Caching:** predictions are reused while the items, image modes, token limits, start model and weight files are unchanged. Change any of them and the checkpoint is generated again.

**Gates:** `[[eval.gates]]` set a `group` and `metric` with `max_value`, `min_value`, `max_absolute_change` or `max_relative_change`. Change bounds compare each later checkpoint with the first.
