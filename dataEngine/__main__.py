"""Data engine CLI.

    python -m dataEngine --config configs/data.toml golden [--workers N]
    python -m dataEngine --config configs/data.toml split
    python -m dataEngine --config configs/data.toml crops --source NAME [--workers N]

Steps run in that order: validated golden page records → locked splits → crop records per source.
"""

import argparse
import json
import logging
import tomllib
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from dataEngine.golden import convert_document
from dataEngine.parallel import parallel_map
from dataEngine.records import PageRecord, read_pages, write_jsonl
from dataEngine.sources import CONVERTERS
from dataEngine.splits import assign_splits, write_splits

log = logging.getLogger("dataEngine")


def inventory(pages: list[PageRecord]) -> dict:
    blocks = [b for p in pages for b in p.blocks]
    return {
        "documents": len({p.document_id for p in pages}),
        "pages": len(pages),
        "regions": len(blocks),
        "pages_per_station": dict(sorted(Counter(p.station_key for p in pages).items())),
        "regions_per_language": dict(Counter(lang for b in blocks for lang in b.langs)),
        "regions_per_status": dict(Counter(b.status for b in blocks)),
        "regions_per_label": dict(Counter(b.label for b in blocks).most_common()),
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_golden(cfg: dict, workers: int) -> None:
    golden = cfg["golden"]
    data_root, out_dir = Path(golden["data_root"]), Path(golden["out_dir"])
    documents = sorted(p for p in data_root.iterdir() if (p / "manifest.json").is_file())
    results = parallel_map(partial(convert_document, data_root=data_root), documents, workers)
    pages = [page for doc_pages, _ in results for page in doc_pages]
    problems = [problem for _, doc_problems in results for problem in doc_problems]

    _write_json(out_dir / "problems.json", problems)
    if problems and not golden["drop_invalid"]:
        raise SystemExit(
            f"{len(problems)} problems in golden data, see {out_dir / 'problems.json'}. "
            "Fix them, or set drop_invalid = true to skip those pages."
        )
    excluded = set(golden["exclude_langs"])
    kept = [p for p in pages if not p.langs & excluded]
    set_aside = [p for p in pages if p.langs & excluded]
    write_jsonl(out_dir / "pages.jsonl", kept)
    write_jsonl(out_dir / "pages_set_aside.jsonl", set_aside)
    _write_json(
        out_dir / "inventory.json",
        {"kept": inventory(kept), "set_aside": inventory(set_aside), "problems": len(problems)},
    )
    log.info(
        "golden: %d pages kept, %d set aside (%s), %d problems",
        len(kept),
        len(set_aside),
        sorted(excluded),
        len(problems),
    )


def run_split(cfg: dict) -> None:
    split = cfg["split"]
    pages = read_pages(Path(cfg["golden"]["out_dir"]) / "pages.jsonl")
    assignment, info = assign_splits(
        pages,
        split["test_fraction"],
        split["dev_fraction"],
        split["max_group_fraction"],
        cfg["seed"],
    )
    write_splits(assignment, info, Path(split["out_dir"]))
    log.info(
        "split: %s pages; stations split by document: %s",
        info["pages"],
        info["stations_split_by_document"],
    )


def run_crops(cfg: dict, name: str, workers: int) -> None:
    sources = {s["name"]: s for s in cfg["sources"]}
    if name not in sources:
        raise SystemExit(f"unknown source {name!r}; configured: {sorted(sources)}")
    source = sources[name]
    out_dir = Path(source["out_dir"])
    count = CONVERTERS[source["converter"]](source, out_dir, workers)
    _write_json(
        out_dir / "manifest.json",
        {"source": source, "records": count, "created_at": datetime.now(UTC).isoformat()},
    )
    log.info("crops %s: %d records -> %s", name, count, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dataEngine", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="CPU worker processes")
    parser.add_argument("step", choices=["golden", "split", "crops"])
    parser.add_argument("--source", help="source name (crops step)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))

    if args.step == "golden":
        run_golden(cfg, args.workers)
    elif args.step == "split":
        run_split(cfg)
    else:
        if not args.source:
            parser.error("crops needs --source")
        run_crops(cfg, args.source, args.workers)


if __name__ == "__main__":
    main()
