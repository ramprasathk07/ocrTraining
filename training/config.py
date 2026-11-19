"""Run config: one TOML file per stage or sub-stage (configs/stage<N>/<id>_<name>.toml).

Unknown keys raise immediately, so a typo never silently falls back to a default.
What a run trains is defined by:
  [stage]   which stage it is and which stage's merged output it starts from
  [data]    sources: each names a task (dataEngine/tasks.py) and where its records live
  [lora]    what is adapted; [train] how
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dataEngine.records import LEVELS, PAGE_LEVEL
from dataEngine.tasks import get_task
from models.deepseek_ocr.inputs import ImageConfig
from models.deepseek_ocr.loader import ModelConfig

BASE = "base"  # [stage].starts_from for the original model


@dataclass(frozen=True)
class StageConfig:
    id: str  # e.g. "1A"
    description: str
    starts_from: str  # id of the stage whose merged output is [model].name_or_path, or "base"


@dataclass(frozen=True)
class RecordSource:
    """Where one task's records live. Crops: a record folder. Pages: pages.jsonl + splits + root."""

    task: str  # name in dataEngine.tasks.TASKS
    path: Path
    root: Path | None = None  # page tasks: folder the page image paths are relative to
    splits: Path | None = None  # page tasks: folder of locked split lists
    image: ImageConfig | None = None  # this source's image mode; None = the run's [image]

    @property
    def image_root(self) -> Path:
        return self.root if get_task(self.task).records == "pages" else self.path

    def validate(self) -> None:
        if get_task(self.task).records == "pages" and (self.root is None or self.splits is None):
            raise ValueError(f"page task {self.task!r} at {self.path} needs `root` and `splits`")


@dataclass(frozen=True, kw_only=True)
class SourceConfig(RecordSource):
    name: str
    share: float  # sampling probability within an epoch


@dataclass(frozen=True, kw_only=True)
class SmokeConfig(RecordSource):
    """Any source, independent of the run's data, so smoke tests work before downloads."""

    samples: int
    steps: int


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
class LoraGroup:
    """Different rank, alpha and learning rate for the target modules matching `pattern`."""

    pattern: str  # regex; PEFT matches it as a module-name suffix, the optimizer by search
    r: int
    alpha: int
    learning_rate: float


@dataclass(frozen=True)
class LoraConfig:
    r: int  # default rank; train.learning_rate is the default learning rate
    alpha: int
    dropout: float
    target_modules: str  # regex, full-matched against module names by PEFT
    groups: tuple[LoraGroup, ...]


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    prompts_dir: Path  # training prompt registry (report/prompts/training)
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
class RunConfig:
    name: str
    stage: StageConfig
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


def _paths(table: dict, *keys: str) -> dict:
    """Path-valued keys to Path, and an optional inline `image` table to ImageConfig."""
    out = {**table, **{k: Path(table[k]) for k in keys if k in table}}
    if "image" in table:
        out["image"] = ImageConfig(**table["image"])
    return out


def load_config(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    data = dict(raw["data"])
    sources = tuple(
        SourceConfig(**_paths(s, "path", "root", "splits")) for s in data.pop("sources")
    )
    phases = tuple(PhaseConfig(p["fraction"], tuple(p["levels"])) for p in data.pop("phases", []))
    lora = dict(raw["lora"])
    groups = tuple(LoraGroup(**g) for g in lora.pop("groups", []))
    cfg = RunConfig(
        name=raw["name"],
        stage=StageConfig(**raw["stage"]),
        model=ModelConfig(**raw["model"]),
        image=ImageConfig(**raw["image"]),
        lora=LoraConfig(groups=groups, **lora),
        data=DataConfig(sources=sources, phases=phases, **data),
        train=TrainConfig(**_paths(raw["train"], "output_dir", "prompts_dir")),
        smoke=SmokeConfig(**_paths(raw["smoke"], "path", "root", "splits")),
        raw=raw,
    )
    _close_to_one([s.share for s in sources], "data.sources shares")
    if phases:
        _close_to_one([p.fraction for p in phases], "data.phases fractions")
    allowed = {*LEVELS, PAGE_LEVEL}
    for phase in phases:
        if not set(phase.levels) <= allowed:
            raise ValueError(f"phase levels {phase.levels} must be within {sorted(allowed)}")
    for source in (*sources, cfg.smoke):
        source.validate()
    for pattern in [cfg.lora.target_modules] + [g.pattern for g in groups]:
        re.compile(pattern)
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
