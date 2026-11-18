"""Official archives laid out as images plus one label file per split (IIIT-HW-Dev, Mozhi).

Each label-file line is `<image path relative to root><separator><text>`, split on the first
separator. Set `root`, `label_files` and `separator` in the source config after reading the
archive's README. Nothing here guesses a format: bad lines or images fail loudly unless the
source sets `drop_invalid = true`, and then every dropped row is listed in problems.json.
"""

import json
from pathlib import Path

from PIL import Image

from dataEngine.parallel import parallel_map
from dataEngine.records import CropRecord, relative_path, write_jsonl
from dataEngine.text import is_valid_utf8, normalize_text

Row = tuple[int, str, str]  # (line number, image path, text)


def read_label_file(path: Path, separator: str) -> list[Row]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        image, found, text = line.partition(separator)
        if not found:
            raise ValueError(f"{path}:{number}: separator {separator!r} not found")
        rows.append((number, image.strip(), text.strip()))
    return rows


def _check_rows(job: tuple[Path, list[Row]]) -> list[tuple[int, str]]:
    """(line number, problem) for each row whose image or text is unusable."""
    root, rows = job
    problems = []
    for number, image, text in rows:
        if not text or not is_valid_utf8(text):
            problems.append((number, "empty or invalid text"))
            continue
        try:
            with Image.open(root / image) as img:
                img.verify()
        except (OSError, SyntaxError) as error:
            problems.append((number, f"unreadable image {image}: {error}"))
    return problems


def convert(source: dict, out_dir: Path, workers: int) -> int:
    root = Path(source["root"])
    records, problems = [], []
    for split, label_file in source["label_files"].items():
        rows = read_label_file(root / label_file, source["separator"])
        shards = [(root, rows[i::workers]) for i in range(max(workers, 1))]
        bad = {n: p for shard in parallel_map(_check_rows, shards, workers) for n, p in shard}
        problems += [{"split": split, "line": n, "problem": p} for n, p in sorted(bad.items())]
        records += [
            CropRecord(
                id=f"{source['name']}/{split}/{number:07d}",
                image=relative_path(root / image, out_dir),
                text=normalize_text(text),
                langs=tuple(source["langs"]),
                script=source["script"],
                level=source["level"],
                source=source["name"],
                split=split,
            )
            for number, image, text in rows
            if number not in bad
        ]

    if problems:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "problems.json").write_text(
            json.dumps(problems, ensure_ascii=False, indent=1), "utf-8"
        )
        if not source.get("drop_invalid", False):
            raise ValueError(f"{len(problems)} invalid rows; see {out_dir / 'problems.json'}")
    return write_jsonl(out_dir / "records.jsonl", records)
