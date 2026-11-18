"""Run config: one TOML file per run (configs/<run>.toml), loaded into frozen dataclasses.

Unknown keys raise immediately, so a typo never silently falls back to a default.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from dataEngine.records import LEVELS
from dataEngine.tasks import get_task
from models.deepseek_ocr.inputs import ImageConfig
from models.deepseek_ocr.loader import ModelConfig


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path  # folder of crop records written by `python -m dataEngine ... crops`
    share: float  # sampling probability within an epoch
    task: str  # name in dataEngine.tasks.TASKS


@dataclass(frozen=True)
class PhaseConfig:
    fraction: float  # share of epoch 1
    levels: tuple[str, ...]  # record levels allowed in this phase


@dataclass(frozen=True)
class DataConfig:
    sources: tuple[SourceConfig, ...]
    epoch_size: int
    num_epochs: int
    phases: tuple[PhaseConfig, ...]  # epoch-1 curriculum; empty = no curriculum
    dev_size: int
    max_target_tokens: int
    num_workers: int


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: str  # regex, full-matched against module names by PEFT


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    seed: int
    global_batch_size: int  # same recipe on any GPU count; accumulation is derived
    per_device_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler: str
    max_grad_norm: float
    gradient_checkpointing: bool
    logging_steps: int
    save_steps: int
    eval_steps: int


@dataclass(frozen=True)
class SmokeConfig:
    source: str
    samples: int
    steps: int


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: ModelConfig
    image: ImageConfig
    lora: LoraConfig
    data: DataConfig
    train: TrainConfig
    smoke: SmokeConfig
    raw: dict  # the parsed TOML, saved next to every checkpoint


def _close_to_one(values: list[float], what: str) -> None:
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"{what} must sum to 1, got {sum(values)}")


def load_config(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    data = dict(raw["data"])
    sources = tuple(SourceConfig(**{**s, "path": Path(s["path"])}) for s in data.pop("sources"))
    phases = tuple(PhaseConfig(p["fraction"], tuple(p["levels"])) for p in data.pop("phases", []))
    cfg = RunConfig(
        name=raw["name"],
        model=ModelConfig(**raw["model"]),
        image=ImageConfig(**raw["image"]),
        lora=LoraConfig(**raw["lora"]),
        data=DataConfig(sources=sources, phases=phases, **data),
        train=TrainConfig(**{**raw["train"], "output_dir": Path(raw["train"]["output_dir"])}),
        smoke=SmokeConfig(**raw["smoke"]),
        raw=raw,
    )
    _close_to_one([s.share for s in sources], "data.sources shares")
    if phases:
        _close_to_one([p.fraction for p in phases], "data.phases fractions")
    for phase in phases:
        if not set(phase.levels) <= set(LEVELS):
            raise ValueError(f"phase levels {phase.levels} must be within {LEVELS}")
    for source in sources:
        get_task(source.task)
    if cfg.smoke.source not in {s.name for s in sources}:
        raise ValueError(f"smoke.source {cfg.smoke.source!r} is not a data source")
    return cfg


def grad_accum_steps(global_batch_size: int, per_device_batch_size: int, world_size: int) -> int:
    """Accumulation that keeps the global batch fixed for any GPU count."""
    per_step = per_device_batch_size * world_size
    if global_batch_size % per_step:
        raise ValueError(
            f"global_batch_size {global_batch_size} is not divisible by "
            f"per_device_batch_size {per_device_batch_size} × {world_size} GPUs"
        )
    return global_batch_size // per_step
