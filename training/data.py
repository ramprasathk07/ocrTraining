"""Deterministic data mixture → map-style datasets.

The whole sample plan (every epoch, in order) is built once from the seed. Each rank then
reads its own shard of that plan through the Trainer's distributed sampler, so 1 GPU and
N GPUs see the same samples, and no rank draws its own random mixture.
"""

import hashlib
import logging
import math
import random
from dataclasses import dataclass

from torch.utils.data import Dataset

from dataEngine.records import CropRecord, PageRecord
from dataEngine.tasks import get_task, read_split
from models.deepseek_ocr.inputs import ImageConfig, encode_sample, load_image
from training.config import DataConfig, PhaseConfig, SourceConfig

log = logging.getLogger(__name__)

Sample = tuple[int, int]  # (source index, record index)
Record = CropRecord | PageRecord


def split_counts(total: int, weights: list[float]) -> list[int]:
    """Largest-remainder split of `total` by `weights`; exact sum, deterministic."""
    raw = [total * w / sum(weights) for w in weights]
    counts = [math.floor(x) for x in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: (counts[i] - raw[i], i))
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


class _Cycler:
    """Draws without replacement and reshuffles when exhausted, so repeats are spread evenly."""

    def __init__(self, indices: list[int], rng: random.Random) -> None:
        self.indices, self.rng, self.queue = indices, rng, []

    def take(self, count: int) -> list[int]:
        out = []
        while len(out) < count:
            if not self.queue:
                self.queue = self.rng.sample(self.indices, len(self.indices))
            take = min(count - len(out), len(self.queue))
            out += self.queue[:take]
            self.queue = self.queue[take:]
        return out


def plan_samples(
    pools: list[list[CropRecord]],
    shares: list[float],
    epoch_size: int,
    num_epochs: int,
    phases: tuple[PhaseConfig, ...],
    seed: int,
) -> list[Sample]:
    """Every sample of every epoch, in training order.

    Epoch 1 follows `phases` (curriculum); later epochs mix all levels. A source with no
    record of a phase's levels sits that phase out, and the others share its slot.
    """
    rng = random.Random(seed)
    cyclers: dict[tuple[int, tuple[str, ...]], _Cycler] = {}
    plan: list[Sample] = []
    for epoch in range(num_epochs):
        segments = phases if epoch == 0 and phases else (PhaseConfig(1.0, ()),)
        for segment in segments:
            eligible = [
                [i for i, r in enumerate(pool) if not segment.levels or r.level in segment.levels]
                for pool in pools
            ]
            active = [s for s, indices in enumerate(eligible) if indices]
            if not active:
                raise ValueError(f"no records match levels {segment.levels}")
            counts = split_counts(round(segment.fraction * epoch_size), [shares[s] for s in active])
            chunk: list[Sample] = []
            for source, count in zip(active, counts, strict=True):
                cycler = cyclers.setdefault(
                    (source, segment.levels), _Cycler(eligible[source], rng)
                )
                chunk += [(source, i) for i in cycler.take(count)]
            rng.shuffle(chunk)
            plan += chunk
    return plan


def drop_long_targets(
    records: list[Record], source: SourceConfig, tokenizer, max_tokens: int
) -> tuple[list[Record], int]:
    """Remove records whose target exceeds `max_tokens` (never truncate a target)."""
    if not records:
        return [], 0
    task = get_task(source.task)
    targets = [task(r, source.image_root).target for r in records]
    lengths = [len(ids) + 1 for ids in tokenizer(targets, add_special_tokens=False)["input_ids"]]
    kept = [r for r, n in zip(records, lengths, strict=True) if n <= max_tokens]
    return kept, len(records) - len(kept)


@dataclass
class TaskDataset(Dataset):
    """Plan index → encoded sample (image loading and preprocessing run in DataLoader workers)."""

    plan: list[Sample]
    pools: list[list[Record]]
    sources: list[SourceConfig]
    tokenizer: object
    image_cfg: ImageConfig

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict:
        source_index, record_index = self.plan[index]
        source = self.sources[source_index]
        sample = get_task(source.task)(self.pools[source_index][record_index], source.image_root)
        image_cfg = source.image or self.image_cfg
        return encode_sample(
            sample.prompt, sample.target, load_image(sample.image), self.tokenizer, image_cfg
        )


def load_pools(
    sources: tuple[SourceConfig, ...], split: str, tokenizer, max_tokens: int
) -> tuple[list[list[Record]], dict]:
    """Records of one split per source (long targets dropped and counted) + a data fingerprint."""
    pools, stats = [], {}
    for source in sources:
        records = read_split(get_task(source.task), source.path, split, source.splits)
        if records is None:
            raise FileNotFoundError(f"source {source.name!r}: no records at {source.path}")
        records, dropped = drop_long_targets(records, source, tokenizer, max_tokens)
        ids = "\n".join(r.id for r in records).encode("utf-8")
        stats[source.name] = {
            "records": len(records),
            "dropped_long": dropped,
            "ids_sha256": hashlib.sha256(ids).hexdigest(),
        }
        if dropped:
            log.warning(
                "%s/%s: dropped %d records longer than %d target tokens",
                source.name,
                split,
                dropped,
                max_tokens,
            )
        pools.append(records)
    return pools, stats


def build_datasets(
    cfg: DataConfig, tokenizer, image_cfg: ImageConfig, seed: int
) -> tuple[TaskDataset, TaskDataset, dict]:
    """Train dataset (all epochs, planned) and a fixed dev dataset for eval loss."""
    sources = list(cfg.sources)
    shares = [s.share for s in sources]
    train_pools, train_stats = load_pools(cfg.sources, "train", tokenizer, cfg.max_target_tokens)
    for source, pool in zip(sources, train_pools, strict=True):
        if not pool:
            raise ValueError(f"source {source.name!r} has no train records in {source.path}")
    dev_pools, dev_stats = load_pools(cfg.sources, "dev", tokenizer, cfg.max_target_tokens)
    if not any(dev_pools):
        raise ValueError("no source has dev records; eval loss needs at least one")

    train_plan = plan_samples(train_pools, shares, cfg.epoch_size, cfg.num_epochs, cfg.phases, seed)
    dev_plan = plan_samples(dev_pools, shares, cfg.dev_size, 1, (), seed + 1)
    stats = {
        "train": train_stats,
        "dev": dev_stats,
        "train_samples": len(train_plan),
        "dev_samples": len(dev_plan),
    }
    return (
        TaskDataset(train_plan, train_pools, sources, tokenizer, image_cfg),
        TaskDataset(dev_plan, dev_pools, sources, tokenizer, image_cfg),
        stats,
    )
