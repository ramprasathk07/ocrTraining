"""The stage tree, read from the run configs:

    python -m training.stages

Every configs/stage<N>/*.toml is one stage or sub-stage; its [stage].starts_from names the
stage whose merged output it starts from ("base" = the original model). The configs are the
only source of truth: add a sub-stage = add a config. Prints the tree, each stage's tasks and
data, and what exists so far (merged output, evaluation report).
"""

import sys
from collections import defaultdict
from pathlib import Path

from training.config import BASE, RunConfig, load_config

CONFIGS = Path("configs")


def load_stages(root: Path = CONFIGS) -> list[tuple[Path, RunConfig]]:
    return [(path, load_config(path)) for path in sorted(root.glob("stage*/*.toml"))]


def check(stages: list[tuple[Path, RunConfig]]) -> list[str]:
    """Problems with the tree: duplicate ids, unknown starts_from, cycles."""
    ids = [cfg.stage.id for _, cfg in stages]
    problems = [f"duplicate stage id {i}" for i in sorted(set(ids)) if ids.count(i) > 1]
    parents = {cfg.stage.id: cfg.stage.starts_from for _, cfg in stages}
    for stage, parent in parents.items():
        if parent != BASE and parent not in parents:
            problems.append(f"{stage}: starts_from {parent!r} is not a stage")
        seen, current = {stage}, parent
        while current in parents:
            if current in seen:
                problems.append(f"{stage}: starts_from forms a cycle")
                break
            seen.add(current)
            current = parents[current]
    return problems


def tree(stages: list[tuple[Path, RunConfig]]) -> list[str]:
    children = defaultdict(list)
    for path, cfg in stages:
        children[cfg.stage.starts_from].append((path, cfg))
    lines = [BASE + "  (original model)"]

    def walk(parent: str, depth: int) -> None:
        for path, cfg in children[parent]:
            merged = (cfg.train.output_dir / "merged").is_dir()
            report = Path(cfg.raw["eval"]["report_dir"]) / "report_dev.md"
            tasks = sorted({s.task for s in cfg.data.sources})
            data = ", ".join(f"{s.name} {s.share:.0%}" for s in cfg.data.sources)
            pad = "    " * depth
            lines.append(f"{pad}└─ {cfg.stage.id}: {cfg.stage.description}")
            lines.append(f"{pad}     config {path.as_posix()} · tasks {', '.join(tasks)}")
            lines.append(f"{pad}     data   {data}")
            lines.append(
                f"{pad}     status merged {'yes' if merged else 'no'} · "
                f"report {'yes' if report.is_file() else 'no'}"
            )
            walk(cfg.stage.id, depth + 1)

    walk(BASE, 1)
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    stages = load_stages()
    print("\n".join(tree(stages)))
    if problems := check(stages):
        raise SystemExit("stage tree problems:\n" + "\n".join(problems))


if __name__ == "__main__":
    main()
