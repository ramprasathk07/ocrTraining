from collections import Counter

from dataEngine.records import CropRecord
from training.config import PhaseConfig
from training.data import _Cycler, plan_samples, split_counts


def _record(index: int, level: str) -> CropRecord:
    return CropRecord(
        f"s/{level}/{index:04d}", "x.png", "t", ("hi",), "unknown", level, "s", "train"
    )


def test_split_counts_is_exact():
    assert split_counts(10, [0.5, 0.25, 0.25]) == [5, 3, 2]
    assert sum(split_counts(250_000, [0.5, 0.25, 0.25])) == 250_000


def test_plan_follows_shares_and_curriculum_and_is_deterministic():
    words = [_record(i, "word") for i in range(50)]
    regions = [_record(i, "region") for i in range(20)]
    phases = (PhaseConfig(0.5, ("word",)), PhaseConfig(0.5, ("word", "region")))
    plan = plan_samples([words, regions], [0.5, 0.5], 100, 2, phases, seed=1)

    assert len(plan) == 200
    assert {s for s, _ in plan[:50]} == {0}  # regions sit out the word-only phase
    assert Counter(s for s, _ in plan[50:100]) == {0: 25, 1: 25}
    assert Counter(s for s, _ in plan[100:]) == {0: 50, 1: 50}  # epoch 2: everything mixed
    assert plan == plan_samples([words, regions], [0.5, 0.5], 100, 2, phases, seed=1)


def test_repeats_are_spread_evenly():
    import random

    cycler = _Cycler([0, 1, 2], random.Random(0))
    assert Counter(cycler.take(9)) == {0: 3, 1: 3, 2: 3}
