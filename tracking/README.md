# tracking/

Logs training and evaluation to **Weights & Biases** and/or **MLflow**. Nothing is logged until keys are set.

```bash
cp .env.example .env     # then fill in the keys; .env is git-ignored
```

Training and evaluation read `.env` at start. A variable already set in the shell wins.

| Tracker | On when | Where runs go |
|---|---|---|
| W&B | `WANDB_API_KEY` is set | project `WANDB_PROJECT`, team `WANDB_ENTITY` |
| MLflow | `MLFLOW_TRACKING_URI` is set | experiment `MLFLOW_EXPERIMENT_NAME` (login: `MLFLOW_TRACKING_USERNAME` / `PASSWORD`, or `MLFLOW_TRACKING_TOKEN`) |

## What is logged

| Run | Name | Content |
|---|---|---|
| Training | the config's `name`, e.g. `stage1a_printed` | loss, learning rate, grad norm, dev loss (every `logging_steps` / `eval_steps`), training arguments. Each saved checkpoint is uploaded if `WANDB_LOG_MODEL=checkpoint` / `HF_MLFLOW_LOG_ARTIFACTS=TRUE` |
| Evaluation | `eval <config> <split> <checkpoint>`, one per `--checkpoint` | every score group (e.g. `source.mozhi_hi/cer`, `script.handwritten/lang.hi/cer`), gate results (`gate/...` = 1 pass / 0 fail, `gates_failed`), params (checkpoint, model path, before/after), `report_<split>.json` |

**Linking runs:**
- **W&B:** runs of one config share a group, with job type `train` or `eval`.
- **MLflow:** runs carry the tags `config`, `stage` and `kind` (`train` / `eval`). Filter in the UI with, for example, `tags.stage = "1A"`.

**Not logged:**
- Predictions and `report_<split>.md`, because they quote FIR text.
- `--smoke` and `--profile` runs.
- `merged/` models (6.7 GB each).

**Privacy:** checkpoints from Stage 1C on are trained on golden FIR data (personal information). Upload them only to a server your team controls, or set the two upload variables to `false` / `FALSE`.

Code: `__init__.py` (`load_env`, `enabled`, `log_eval`). Tests: `tests/test_tracking.py`.
