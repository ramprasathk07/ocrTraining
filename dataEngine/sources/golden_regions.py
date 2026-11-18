"""Golden page blocks → region crop records, split by the locked page splits."""

from pathlib import Path

from dataEngine.records import CropRecord, read_pages, relative_path, write_jsonl
from dataEngine.splits import read_splits


def convert(source: dict, out_dir: Path, workers: int) -> int:
    """One record per block. Fast, so it ignores `workers` and runs in one process."""
    pages = read_pages(Path(source["pages"]))
    splits = read_splits(Path(source["splits"]))
    data_root = Path(source["data_root"])
    missing = [p.page_id for p in pages if p.page_id not in splits]
    if missing:
        raise ValueError(
            f"{len(missing)} pages have no split (first: {missing[:3]}); re-run `split`"
        )

    records = [
        CropRecord(
            id=f"{source['name']}/{page.page_id}/{block.region_id}",
            image=relative_path(data_root / block.crop, out_dir),
            text=block.source,
            langs=block.langs,
            # shortcut: script tags come from the Stage 0 dev/test tagging; until then "unknown".
            script="unknown",
            level="region",
            source=source["name"],
            split=splits[page.page_id],
        )
        for page in pages
        for block in page.blocks
    ]
    return write_jsonl(out_dir / "records.jsonl", records)
