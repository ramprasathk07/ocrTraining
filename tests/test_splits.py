from collections import defaultdict

import pytest

from dataEngine.records import PageRecord
from dataEngine.splits import assign_splits, read_splits, write_splits


def _page(page_id: str, unit: str, document: str) -> PageRecord:
    return PageRecord(page_id, document, "D", unit, "x.png", "", 10, 10, ())


def _pages() -> list[PageRecord]:
    small = [_page(f"s{s}/p{i}", f"s{s}", f"s{s}") for s in range(20) for i in range(5)]
    big = [_page(f"big/d{d}/p{i}", "big", f"big{d}") for d in range(4) for i in range(10)]
    return small + big


def test_split_is_deterministic_and_never_splits_a_group():
    pages = _pages()
    assignment, info = assign_splits(pages, 0.15, 0.15, 0.15, seed=7)
    again, _ = assign_splits(list(reversed(pages)), 0.15, 0.15, 0.15, seed=7)
    assert assignment == again

    assert info["groups_split_by_document"] == ["D/big"]  # 40 of 140 pages > 15%
    groups = defaultdict(set)
    for page in pages:
        key = page.document_id if page.unit == "big" else page.unit
        groups[key].add(assignment[page.page_id])
    assert all(len(splits) == 1 for splits in groups.values())

    total = len(pages)
    assert info["pages"]["test"] >= 0.15 * total
    assert info["pages"]["dev"] >= 0.15 * total
    assert sum(info["pages"].values()) == total


def test_splits_are_locked_once_written(repo_tmp):
    assignment, info = assign_splits(_pages(), 0.15, 0.15, 0.15, seed=1)
    write_splits(assignment, info, repo_tmp)
    assert read_splits(repo_tmp) == assignment
    with pytest.raises(FileExistsError):
        write_splits(assignment, info, repo_tmp)
