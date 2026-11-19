# report/

Human-readable results. Git tracks only `report/prompts/` and this README. Every other report quotes FIR text and stays local.

```
report/
  README.md
  prompts/                       tracked: prompt versions and how metrics move with them
    training/                    written by every training run, at start
      registry.json              prompt version → task, text, first seen, first run (append-only)
      history.jsonl              one line per run: stage, starts_from, start model, config, prompts, data mix
      runs.md                    history as a table, newest first
    eval/                        written by every evaluation
      registry.json              same format as training/
      history.jsonl              one line per evaluation: model, prompts, headline metrics per source
      leaderboard.md             every result so far, newest first
  <run name>/                    git-ignored: one folder per config ([eval].report_dir)
    report_<split>.md / .json    scores, gates, prompts used, most improved/regressed items
```

| Folder | From | Status |
|---|---|---|
| `test_data/` | `configs/eval/test_data.toml` | baseline of the original OCR-2 on the test data (2026-11-19) |
| `stage1a_printed/`, `stage1b_handwritten/`, `stage1c_joint/` | `configs/stage1/*.toml` | written after each sub-stage's evaluation |

## Prompt versions

Prompts are defined only in `dataEngine/tasks.py`, one per task. A version is a hash of the exact text (`p-` + 10 hex), so any edit gives a new id.

| Question | Where to look |
|---|---|
| Which prompt was 1B trained with? | `prompts/training/runs.md` |
| How did CER move across checkpoints and prompts? | `prompts/eval/leaderboard.md` |
| What exactly did prompt `p-…` say? | `registry.json` in either folder |
| When did a prompt change? | `git log -p report/prompts/` |

A report flags a **prompt mismatch** when a checkpoint is evaluated with a different prompt than its `run_manifest.json` records.

To compare a new prompt:
1. Register it as a new task (e.g. `crop_ocr_v2`).
2. Evaluate both tasks on the same items in one run.

A fine-tuned model is tied to its training prompt, so changing that prompt later means retraining.

## Reading a report

- **CER/WER above 100%** means the model wrote much more than the reference (hallucination or repetition), not that the scoring is wrong.
- **Valid output** counts only parsable output that stopped at EOS. Pages that loop to the token limit fail it.
- **Gate rows:** `before` = the first checkpoint, `after` = each later one. A single-checkpoint run shows only `after`.
