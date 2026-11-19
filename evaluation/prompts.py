"""Evaluation side of prompt tracking: which prompt produced which numbers, over time.

Writes report/prompts/eval/ (registry.json, history.jsonl, leaderboard.md) through
dataEngine.prompt_registry. A fine-tuned model must be evaluated with the prompts it was
trained on: `drift_warnings` compares them with the training run's run_manifest.json.
"""

import json
from pathlib import Path

from dataEngine.prompt_registry import (
    append_history,
    markdown_table,
    now,
    prompt_versions_table,
    register,
    task_prompts,
)

HEADLINE = {  # metrics shown per task in the leaderboard
    "crop_ocr": ("cer", "wer", "exact_match"),
    "page_parse": ("page_cer", "layout_ap50", "mean_iou", "reading_order", "valid_output"),
}


def prompts_used(sources: list[dict]) -> dict[str, dict]:
    """task → {version, prompt} for every task in the eval sources."""
    return task_prompts(s["task"] for s in sources)


def training_prompts(checkpoint: str) -> dict[str, dict] | None:
    """Prompts recorded when the checkpoint's run trained (run_manifest.json beside or above)."""
    for folder in (Path(checkpoint), Path(checkpoint).parent):
        manifest = folder / "run_manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8")).get("prompts")
    return None


def drift_warnings(checkpoints: list[str], prompts: dict[str, dict]) -> list[str]:
    warnings = []
    for checkpoint in checkpoints:
        trained = training_prompts(checkpoint) or {}
        for task, used in prompts.items():
            if task in trained and trained[task]["version"] != used["version"]:
                warnings.append(
                    f"{checkpoint}: {task} was trained with prompt {trained[task]['version']} "
                    f"but evaluated with {used['version']}"
                )
    return warnings


def _headline(task: str, scores: dict) -> str:
    return ", ".join(
        f"{metric}={100 * scores[metric]:.2f}%"
        for metric in HEADLINE[task]
        if scores.get(metric) is not None
    )


def record_run(
    prompts_dir: Path,
    raw: dict,
    split: str,
    labels: list[str],
    scores: list[dict],
    prompts: dict[str, dict],
) -> None:
    """Register the prompts, append this evaluation to the history, redraw leaderboard.md."""
    register(prompts_dir, prompts, raw["name"])
    results = []
    for label, checkpoint_scores in zip(labels, scores, strict=True):
        for source in raw["eval"]["sources"]:
            group = checkpoint_scores.get(f"source={source['name']}")
            if group is not None:
                results.append(
                    {
                        "checkpoint": label,
                        "source": source["name"],
                        "task": source["task"],
                        "prompt": prompts[source["task"]]["version"],
                        "n": group["n"],
                        "headline": _headline(source["task"], group),
                    }
                )
    history = append_history(
        prompts_dir,
        {
            "date": now(),
            "run": raw["name"],
            "stage": raw.get("stage", {}).get("id"),
            "split": split,
            "model": f"{raw['model']['name_or_path']}@{raw['model']['revision'][:8]}",
            "prompts": {task: used["version"] for task, used in prompts.items()},
            "results": results,
        },
    )
    rows = [
        [
            e["date"][:16],
            e["run"],
            e.get("stage") or "—",
            e["split"],
            r["checkpoint"],
            r["source"],
            f"`{r['prompt']}`",
            r["n"],
            r["headline"],
        ]
        for e in reversed(history)
        for r in e["results"]
    ]
    header = ["date", "run", "stage", "split", "checkpoint", "source", "prompt", "n", "metrics"]
    lines = ["# Evaluation leaderboard", "", "Newest first.", "", "## Prompt versions", ""]
    lines += [
        *prompt_versions_table(prompts_dir),
        "",
        "## Results",
        "",
        *markdown_table(header, rows),
    ]
    (prompts_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
