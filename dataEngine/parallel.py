"""Run CPU work on 1..N worker processes. The worker count comes from `--workers`."""

from collections.abc import Callable, Iterable
from multiprocessing import Pool


def parallel_map[T, R](fn: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """`map` that keeps input order, so results never depend on the worker count."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with Pool(min(workers, len(items))) as pool:
        return pool.map(fn, items)
