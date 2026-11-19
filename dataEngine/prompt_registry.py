"""Prompt registry files. Training and evaluation each keep one folder:

    report/prompts/training/   which prompts each training run (stage) was trained on
    report/prompts/eval/       which prompts produced each evaluation's numbers

Each folder holds:
    registry.json    version → task, prompt text, first run (entries are only added)
    history.jsonl    one line per run
    <table>.md       the history as a readable table (written by the caller)
"""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from dataEngine.tasks import get_task, prompt_version


def task_prompts(tasks: Iterable[str]) -> dict[str, dict]:
    """task → {version, prompt}, from the task registry (the only place prompts are defined)."""
    return {
        task: {"version": prompt_version(get_task(task).prompt), "prompt": get_task(task).prompt}
        for task in sorted(set(tasks))
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def register(folder: Path, prompts: dict[str, dict], run: str) -> None:
    """Add unseen prompt versions to <folder>/registry.json; existing entries never change."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "registry.json"
    registry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for task, used in prompts.items():
        registry.setdefault(
            used["version"],
            {"task": task, "prompt": used["prompt"], "first_seen": now(), "first_run": run},
        )
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(folder: Path, entry: dict) -> list[dict]:
    """Append one run to <folder>/history.jsonl and return the whole history."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (folder / "history.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    def cell(value: str) -> str:
        return value.replace("\n", "\\n").replace("|", "\\|")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(cell(str(v)) for v in row) + " |" for row in rows]


def prompt_versions_table(folder: Path) -> list[str]:
    registry = json.loads((folder / "registry.json").read_text(encoding="utf-8"))
    rows = [
        [
            f"`{version}`",
            info["task"],
            info["first_seen"][:10],
            info["first_run"],
            f"`{info['prompt']}`",
        ]
        for version, info in registry.items()
    ]
    return markdown_table(["version", "task", "first seen", "first run", "prompt"], rows)
