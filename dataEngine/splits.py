"""Train/dev/test split by police station, written once and then locked.

One officer writes many FIRs at one station, so splitting by page would leak that
officer's handwriting into test. A station larger than `max_group_fraction` of all
pages is split by document instead, so the targets stay reachable.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from dataEngine.records import SPLITS, PageRecord


def assign_splits(
    pages: list[PageRecord],
    test_fraction: float,
    dev_fraction: float,
    max_group_fraction: float,
    seed: int,
) -> tuple[dict[str, str], dict]:
    """Map page_id → split. Each group goes whole to one split; fill order: test, dev, train."""
    # shortcut: no doc_type stratification yet; add it once the Stage 0 page tags exist.
    total = len(pages)
    by_station: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        by_station[page.station_key].append(page)

    groups: dict[str, list[str]] = {}
    split_by_document = []
    for station, station_pages in sorted(by_station.items()):
        if len(station_pages) > max_group_fraction * total:
            split_by_document.append(station)
            by_document: dict[str, list[str]] = defaultdict(list)
            for page in station_pages:
                by_document[page.document_id].append(page.page_id)
            for document_id, page_ids in by_document.items():
                groups[f"{station}#{document_id}"] = page_ids
        else:
            groups[station] = [p.page_id for p in station_pages]

    order = sorted(groups)
    random.Random(seed).shuffle(order)
    targets = (("test", test_fraction * total), ("dev", dev_fraction * total))
    counts: Counter[str] = Counter()
    assignment = {}
    for key in order:
        split = next((name for name, target in targets if counts[name] < target), "train")
        for page_id in groups[key]:
            assignment[page_id] = split
        counts[split] += len(groups[key])

    info = {
        "seed": seed,
        "group": "police_station",
        "test_fraction": test_fraction,
        "dev_fraction": dev_fraction,
        "max_group_fraction": max_group_fraction,
        "stations_split_by_document": split_by_document,
        "groups": len(groups),
        "pages": {s: counts[s] for s in SPLITS},
    }
    return assignment, info


def write_splits(assignment: dict[str, str], info: dict, out_dir: Path) -> None:
    """Write one sorted page-id list per split plus a hashed manifest. Refuses to overwrite."""
    files = [out_dir / f"{split}.txt" for split in SPLITS]
    if any(f.exists() for f in files):
        raise FileExistsError(
            f"splits already exist in {out_dir}; they are locked. Delete them by hand to re-split."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, path in zip(SPLITS, files, strict=True):
        content = "".join(
            f"{pid}\n" for pid in sorted(p for p, s in assignment.items() if s == split)
        )
        path.write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {**info, "sha256": hashes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_splits(out_dir: Path) -> dict[str, str]:
    """page_id → split, from the locked split files."""
    assignment = {}
    for split in SPLITS:
        for line in (out_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            assignment[line] = split
    return assignment
