# evaluation/

Scores one or more checkpoints on fixed, seeded samples, compares each with the first ("before"), checks gates and writes a report. The same command runs on 1 or N GPUs; items are sharded across ranks.

```bash
# Baseline on the curated test data (convert it first: configs/data/test_data.toml)
torchrun --nproc_per_node=N -m evaluation --config configs/eval/test_data.toml --split test --checkpoint start

# Before vs after a stage
torchrun --nproc_per_node=N -m evaluation --config configs/stage1/1a_printed.toml \
    --checkpoint start --checkpoint outputs/runs/stage1a_printed/merged
```

With W&B / MLflow keys in `.env` (see [tracking/](../tracking/README.md)), every evaluated checkpoint is also logged as one tracker run: its scores, gate results and `report_<split>.json`. This includes `--checkpoint start`, so the untrained baseline is on record. Predictions and the markdown report (they quote private text) are never sent.

| Flag | Meaning |
|---|---|
| `--config` | any config with `[model]` and `[eval]` (a stage config or `configs/eval/*.toml`) |
| `--checkpoint` | repeatable. Accepts `start` (the config's `[model]`), a folder (a LoRA adapter loads on top of `start`; anything else is a full model) or a hub id. The first one is "before" |
| `--split` | `dev` (default, for development) or `test` (for reports only) |

## Files

| File | Does |
|---|---|
| `__main__.py` | runner: selects items, generates per checkpoint, caches, writes the report and prompt history |
| `metrics.py` | corpus CER / AER / WER / matra error rate (edits ÷ reference length; can exceed 100% when the model over-generates), mean 1 - NED (per item, 0–1) |
| `page_metrics.py` | page CER, layout AP@50 (with and without labels), mean box IoU, block P/R/F1 @ IoU 0.5, reading-order accuracy, valid output |
| `report.py` | groups scores (by source, script, language), gates, biggest changes, markdown report |
| `prompts.py` | eval prompt registry + leaderboard (`report/prompts/eval/`); warns when a checkpoint is evaluated with a prompt it wasn't trained on |

## Metrics

| Task | Metric | Notes |
|---|---|---|
| `crop_ocr` | CER, AER, WER, matra error rate, 1 - NED, exact match, mean latency | per source, script, language, script × language |
| `page_parse` | page CER | whitespace collapsed, HTML tags stripped; output without blocks is scored as plain text |
| | layout AP@50 / mAP@50 | ranked by output order; mAP needs IndicDLP labels (only after Stage 2) |
| | mean IoU, block F1@0.5 | greedy box matching |
| | reading order | pairwise accuracy over matched blocks only |
| | valid output | parses cleanly **and** stopped at EOS |
| not measured yet | field F1, translation | no field labels; translation comes in Stage 4 |

**Crop metrics beyond CER** (`metrics.py`):
- **AER** (akshara error rate): edits over aksharas, the written syllables (consonant cluster joined by virama, with its nukta, matras and marks). A wrong conjunct or matra costs one unit, not 2–4 code points.
- **Matra error rate:** edits that touch a dependent sign (matra, nukta, virama, anusvara, visarga, candrabindu) ÷ dependent signs in the references. Undefined (—) for text without them, e.g. English.
- **1 - NED:** mean over items of 1 - edits ÷ max(reference, output length). Bounded 0–1, so a few runaway outputs cannot dominate it the way they dominate corpus CER.
- **Latency:** mean seconds per item for `generate_text` (batch 1, on the eval GPU). Predictions cached before timing existed show —.

Full definitions: [plan §9](../docs/training-plan.md#9-evaluation-and-gates).

## Where results go

| Path | Content | In git? |
|---|---|---|
| `[eval].output_dir/<checkpoint>/<split>/` for `start` and hub ids; `<checkpoint>/eval/<split>/` for local folders | `predictions.jsonl` (raw private text), `scores.json`, `fingerprint.txt` | no (private) |
| `[eval].report_dir/report_<split>.md` + `.json` | scores, gates, prompts, the most improved and regressed items | no (quotes private text) |
| `[eval].prompts_dir` = `report/prompts/eval/` | prompt registry, history, leaderboard | yes |

**Caching:** predictions are reused while the items, image modes, token limits, start model and weight files are unchanged. Change any of them and the checkpoint is generated again.

**Gates:** `[[eval.gates]]` set a `group` and `metric` with `max_value`, `min_value`, `max_absolute_change` or `max_relative_change`. Change bounds compare each later checkpoint with the first.
