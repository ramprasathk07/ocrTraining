"""Training entry point. The same command runs on 1 or N GPUs:

    torchrun --nproc_per_node=N -m training --config configs/stage1/1a_printed.toml
    torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --smoke
    python -m training --config configs/stage1/1a_printed.toml --merge <checkpoint dir>
    torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --profile 20

The config fixes the global batch; gradient accumulation is derived from the GPU count.
Outputs in [train].output_dir: checkpoints, TensorBoard logs, run_manifest.json,
adapter/ (LoRA weights) and merged/ (base + LoRA, the next stage's input).
--merge turns any saved checkpoint (e.g. the best one by dev CER) into <checkpoint>/merged.
--resume continues an interrupted run from the last checkpoint-N in output_dir.
--profile N records training steps 3 .. N+2 as Chrome traces in <output_dir>/profile/ and stops
(training/profiling.py).
W&B / MLflow: on when their keys are in .env (tracking/__init__.py).
"""

import argparse
import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import peft
import torch
import transformers
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

import tracking
from models.deepseek_ocr.inputs import collate
from models.deepseek_ocr.loader import load_model, load_tokenizer
from training.config import RunConfig, grad_accum_steps, load_config
from training.data import build_datasets
from training.lora import apply_lora, build_optimizer
from training.profiling import ProfileSteps, label_modules
from training.prompts import register_training_run, training_prompts
from training.smoke import run_smoke, write_report

log = logging.getLogger("training")


class PlannedOrderTrainer(Trainer):
    """Reads the pre-planned sample list in order: shuffling, curriculum and epochs are in the plan.

    Under DDP, accelerate hands consecutive batches to different ranks, so every rank gets
    a distinct, deterministic shard.
    """

    def _get_train_sampler(self) -> SequentialSampler:
        return SequentialSampler(self.train_dataset)


def _code_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not a git repository"


def _write_manifest(cfg: RunConfig, world_size: int, accum: int, data_stats: dict) -> None:
    manifest = {
        "config": cfg.raw,
        "resolved": {
            "world_size": world_size,
            "per_device_batch_size": cfg.train.per_device_batch_size,
            "gradient_accumulation_steps": accum,
            "global_batch_size": cfg.train.global_batch_size,
        },
        "stage": {"id": cfg.stage.id, "starts_from": cfg.stage.starts_from},
        "data": data_stats,
        "prompts": training_prompts(cfg),  # evaluation warns when scored with other prompts
        "code_version": _code_version(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
    cfg.train.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.train.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def train(
    cfg: RunConfig, config_path: Path, profile_steps: int | None = None, resume: bool = False
) -> None:
    profiling = profile_steps is not None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accum = grad_accum_steps(
        cfg.train.global_batch_size, cfg.train.per_device_batch_size, world_size
    )
    tokenizer = load_tokenizer(cfg.model)
    model = apply_lora(load_model(cfg.model), cfg)
    train_ds, dev_ds, data_stats = build_datasets(cfg.data, tokenizer, cfg.image, cfg.train.seed)
    trackers = [] if profiling else tracking.enabled()  # a profiling run is not an experiment
    tracking.label_training_run(cfg.name, cfg.stage.id)
    log.info("tracking: TensorBoard%s", "".join(f", {t}" for t in trackers))

    args = TrainingArguments(
        output_dir=str(cfg.train.output_dir),
        run_name=cfg.name,
        seed=cfg.train.seed,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,  # the plan already holds every epoch
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=True,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train.logging_steps,
        save_strategy="no" if profiling else "steps",  # a profiling run saves no model
        save_steps=cfg.train.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        prediction_loss_only=True,  # never gather (batch × seq × 129K-vocab) logits
        label_names=["labels"],  # PeftModel's forward signature hides `labels` from the Trainer
        report_to=["tensorboard", *trackers],
        dataloader_num_workers=cfg.data.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # every LoRA module runs on every token
    )
    trainer = PlannedOrderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=partial(collate, pad_id=tokenizer.pad_token_id),
        optimizers=(build_optimizer(model, cfg), None),  # per-group LRs; Trainer adds the schedule
    )
    if profiling:
        label_modules(model)
        trainer.add_callback(ProfileSteps(profile_steps, cfg.train.output_dir / "profile"))
    elif trainer.is_world_process_zero():
        _write_manifest(cfg, world_size, accum, data_stats)
        register_training_run(cfg, config_path)
    last = None
    if resume and cfg.train.output_dir.is_dir():
        last = get_last_checkpoint(str(cfg.train.output_dir))
    if resume:
        log.info("resuming from %s", last or "the start: no checkpoint yet")
    if last:
        # Trainer's torch.load of rng_state.pth (numpy arrays) fails under torch >= 2.6's
        # weights_only default; the checkpoint is this run's own output
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    trainer.train(resume_from_checkpoint=last)  # the planned order makes the skipped batches exact
    if profiling:
        log.info("profile traces in %s", cfg.train.output_dir / "profile")
        return

    trainer.save_model(str(cfg.train.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(cfg.train.output_dir / "merged", safe_serialization=True)
        tokenizer.save_pretrained(cfg.train.output_dir / "merged")
        log.info("saved adapter/ and merged/ in %s", cfg.train.output_dir)


def merge(cfg: RunConfig, checkpoint: Path) -> None:
    """Base model (cfg.model) + the checkpoint's adapter → <checkpoint>/merged."""
    merged = load_model(cfg.model, adapter=checkpoint).merge_and_unload()
    merged.save_pretrained(checkpoint / "merged", safe_serialization=True)
    load_tokenizer(cfg.model).save_pretrained(checkpoint / "merged")
    log.info("merged %s -> %s", checkpoint, checkpoint / "merged")


def smoke(cfg: RunConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise SystemExit("run --smoke with one process (--nproc_per_node=1)")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    tokenizer = load_tokenizer(cfg.model)
    model = apply_lora(load_model(cfg.model), cfg)
    if not write_report(run_smoke(model, tokenizer, cfg, device), cfg.train.output_dir):
        raise SystemExit("smoke tests failed; see smoke.json")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="run the Stage 0 smoke tests and exit")
    parser.add_argument("--merge", type=Path, help="merge this checkpoint's adapter and exit")
    parser.add_argument(
        "--profile", type=int, metavar="N", help="trace training steps 3..N+2, then stop"
    )
    parser.add_argument(
        "--resume", action="store_true", help="continue from the last checkpoint in output_dir"
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s: %(message)s",
    )
    if args.profile is not None and args.profile < 1:
        parser.error("--profile needs at least 1 step")
    tracking.load_env()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    if args.smoke:
        smoke(cfg)
    elif args.merge:
        merge(cfg, args.merge)
    else:
        train(cfg, args.config, args.profile, args.resume)


if __name__ == "__main__":
    main()
