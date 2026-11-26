"""Experiment tracking with Weights & Biases and/or MLflow, switched on by environment variables.

Keys go in `.env` at the repo root (copy `.env.example`; git-ignored). Training and evaluation
read it at start; a variable already set in the shell wins.
    WANDB_API_KEY        set -> W&B    (project WANDB_PROJECT, team WANDB_ENTITY)
    MLFLOW_TRACKING_URI  set -> MLflow (experiment MLFLOW_EXPERIMENT_NAME)

Training logs through the HF Trainer integrations: loss, learning rate, dev loss, and each saved
checkpoint when WANDB_LOG_MODEL=checkpoint / HF_MLFLOW_LOG_ARTIFACTS=TRUE. Evaluation logs one
run per checkpoint (`log_eval`): scores, gate results and report_<split>.json. Predictions and
the markdown report are never sent: they quote FIR text. Runs of one config share a W&B group
and MLflow tags (`config`, `stage`, `kind` = train / eval).
"""

import json
import os
import re
from pathlib import Path

TRACKERS = {"wandb": "WANDB_API_KEY", "mlflow": "MLFLOW_TRACKING_URI"}


def load_env(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines into os.environ (blank lines and # comment lines skipped)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("\"'"))


def enabled() -> list[str]:
    """Trackers whose key or URI is set, in TRACKERS order."""
    return [name for name, variable in TRACKERS.items() if os.environ.get(variable)]


def label_training_run(config: str, stage: str) -> None:
    """Group/tag the HF Trainer's tracker run like the eval runs of the same config."""
    os.environ.setdefault("WANDB_RUN_GROUP", config)
    os.environ.setdefault("WANDB_JOB_TYPE", "train")
    tags = {"config": config, "stage": stage, "kind": "train"}
    os.environ.setdefault("MLFLOW_TAGS", json.dumps(tags))


def metric_name(key: str) -> str:
    """`script=handwritten|lang=hi` -> `script.handwritten/lang.hi` (characters MLflow accepts)."""
    return re.sub(r"[^\w./ -]", "_", key.replace("=", ".").replace("|", "/"))


def flatten(scores: dict, prefix: str = "") -> dict[str, float]:
    """Nested scores -> {"group/metric": number}; text, None and booleans are left out."""
    flat = {}
    for key, value in scores.items():
        name = prefix + metric_name(str(key))
        if isinstance(value, dict):
            flat |= flatten(value, name + "/")
        elif isinstance(value, int | float) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def gate_metrics(gates: list[dict]) -> dict[str, float]:
    """1 = pass, 0 = fail per gate (skipped gates left out), plus the number failed."""
    values = {
        f"gate/{metric_name(g['group'])}/{g['metric']}": float(g["status"].startswith("pass"))
        for g in gates
        if not g["status"].startswith("skipped")
    }
    return values | {"gates_failed": float(sum(g["status"].startswith("fail") for g in gates))}


def log_eval(
    raw: dict,
    split: str,
    labels: list[str],
    paths: list[str],
    scores: list[dict],
    gate_results: list[list[dict]],
    report_json: Path,
) -> None:
    """One tracker run per checkpoint. Gates belong to the checkpoints after the first
    (or to the only one), as in the report."""
    trackers = enabled()
    if not trackers:
        return
    gates_of = gate_results if len(scores) == 1 else [[], *gate_results]
    stage = raw.get("stage", {}).get("id", "none")  # evaluation-only configs have no stage
    for i, (label, path, checkpoint_scores) in enumerate(zip(labels, paths, scores, strict=True)):
        params = {
            "config": raw["name"],
            "stage": stage,
            "split": split,
            "checkpoint": label,
            "model": path,
            "role": "before" if i == 0 and len(scores) > 1 else "after",
        }
        values = flatten(checkpoint_scores) | gate_metrics(gates_of[i])
        name = f"eval {raw['name']} {split} {label}"
        if "wandb" in trackers:
            _log_wandb(name, raw["name"], params, values, report_json)
        if "mlflow" in trackers:
            _log_mlflow(name, params, values, report_json)


def _log_wandb(name: str, group: str, params: dict, values: dict, report_json: Path) -> None:
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name,
        group=group,
        job_type="eval",
        config=params,
    )
    run.log(values)
    report = wandb.Artifact(f"eval-report-{group}-{params['split']}", type="eval-report")
    report.add_file(str(report_json))  # a copy: later evals rewrite the file
    run.log_artifact(report)
    run.finish()


def _log_mlflow(name: str, params: dict, values: dict, report_json: Path) -> None:
    import mlflow

    if experiment := os.environ.get("MLFLOW_EXPERIMENT_NAME"):
        mlflow.set_experiment(experiment)
    tags = {"config": params["config"], "stage": params["stage"], "kind": "eval"}
    with mlflow.start_run(run_name=name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(values)
        mlflow.log_artifact(str(report_json))
