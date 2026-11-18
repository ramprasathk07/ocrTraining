"""Training entry point. The same command runs on 1 or N GPUs:

    torchrun --nproc_per_node=N -m training --config configs/stage1_script.toml
    torchrun --nproc_per_node=1 -m training --config configs/stage1_script.toml --smoke

The config fixes the global batch; gradient accumulation is derived from the GPU count.
Outputs in [train].output_dir: checkpoints, TensorBoard logs, run_manifest.json,
adapter/ (LoRA weights) and merged/ (base + LoRA, the next stage's input).
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
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments, set_seed

from models.deepseek_ocr.inputs import collate
from models.deepseek_ocr.loader import load_model, load_tokenizer
from training.config import RunConfig, grad_accum_steps, load_config
from training.data import build_datasets
from training.smoke import run_smoke, write_report

log = logging.getLogger("training")


class PlannedOrderTrainer(Trainer):
    """Reads the pre-planned sample list in order: shuffling, curriculum and epochs are in the plan.

    Under DDP, accelerate hands consecutive batches to different ranks, so every rank gets
    a distinct, deterministic shard.
    """

    def _get_train_sampler(self) -> SequentialSampler:
        return SequentialSampler(self.train_dataset)


def apply_lora(model, cfg: RunConfig):
    """LoRA on `lora.target_modules`; PEFT freezes all other weights (vision, projector, router)."""
    lora = cfg.lora
    return get_peft_model(
        model,
        PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            bias="none",
        ),
    )


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
        "data": data_stats,
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


def train(cfg: RunConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accum = grad_accum_steps(
        cfg.train.global_batch_size, cfg.train.per_device_batch_size, world_size
    )
    tokenizer = load_tokenizer(cfg.model)
    model = apply_lora(load_model(cfg.model), cfg)
    train_ds, dev_ds, data_stats = build_datasets(cfg.data, tokenizer, cfg.image, cfg.train.seed)

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
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        prediction_loss_only=True,  # never gather (batch × seq × 129K-vocab) logits
        label_names=["labels"],  # PeftModel's forward signature hides `labels` from the Trainer
        report_to=["tensorboard"],
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
    )
    if trainer.is_world_process_zero():
        _write_manifest(cfg, world_size, accum, data_stats)
    trainer.train()

    trainer.save_model(str(cfg.train.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(cfg.train.output_dir / "merged", safe_serialization=True)
        tokenizer.save_pretrained(cfg.train.output_dir / "merged")
        log.info("saved adapter/ and merged/ in %s", cfg.train.output_dir)


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
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    if args.smoke:
        smoke(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
