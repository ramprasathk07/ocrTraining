"""Training side of prompt tracking: which prompts each stage was trained on.

Writes report/prompts/training/ (registry.json, history.jsonl, runs.md) through
dataEngine.prompt_registry, one history line per training run. The same prompts go into
the run's run_manifest.json, where evaluation checks them (prompt mismatch warning).
"""

from pathlib import Path

from dataEngine.prompt_registry import (
    append_history,
    markdown_table,
    now,
    prompt_versions_table,
    register,
    task_prompts,
)
from training.config import RunConfig


def training_prompts(cfg: RunConfig) -> dict[str, dict]:
    """task → {version, prompt} for every task the run trains on."""
    return task_prompts(s.task for s in cfg.data.sources)


def register_training_run(cfg: RunConfig, config_path: Path) -> None:
    folder, prompts = cfg.train.prompts_dir, training_prompts(cfg)
    register(folder, prompts, cfg.name)
    history = append_history(
        folder,
        {
            "date": now(),
            "run": cfg.name,
            "stage": cfg.stage.id,
            "starts_from": cfg.stage.starts_from,
            "start_model": cfg.model.name_or_path,
            "config": config_path.as_posix(),
            "output_dir": cfg.train.output_dir.as_posix(),
            "prompts": {task: used["version"] for task, used in prompts.items()},
            "sources": [
                {"name": s.name, "task": s.task, "share": s.share} for s in cfg.data.sources
            ],
        },
    )
    rows = [
        [
            e["date"][:16],
            e["stage"],
            e["run"],
            e["starts_from"],
            e["config"],
            ", ".join(f"{t} `{v}`" for t, v in e["prompts"].items()),
            ", ".join(f"{s['name']} {s['share']:.0%}" for s in e["sources"]),
        ]
        for e in reversed(history)
    ]
    header = ["date", "stage", "run", "starts from", "config", "prompts", "data"]
    lines = ["# Training runs", "", "Newest first.", "", "## Prompt versions", ""]
    lines += [*prompt_versions_table(folder), "", "## Runs", "", *markdown_table(header, rows)]
    (folder / "runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
