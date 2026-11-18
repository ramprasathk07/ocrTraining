"""Stage 0 smoke tests S3–S9 on one GPU, before any long run (docs/stages/stage-0 §0.7).

Each check records pass/fail and its numbers in `<output_dir>/smoke.json`.
The run exits non-zero if any check fails.
"""

import json
import logging
import math
import tempfile
import time
from pathlib import Path

import torch

from dataEngine.records import read_crops
from dataEngine.tasks import get_task
from models.deepseek_ocr.inputs import IGNORE_INDEX, collate, encode_sample, load_image
from models.deepseek_ocr.loader import load_model
from training.config import RunConfig

log = logging.getLogger(__name__)


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    moved["images"] = [(p.to(device), g.to(device)) for p, g in batch["images"]]
    return moved


def _expected_lora_params(model, r: int) -> int:
    return sum(
        r * (m.in_features + m.out_features)
        for _, m in model.named_modules()
        if hasattr(m, "lora_A")
    )


def run_smoke(model, tokenizer, cfg: RunConfig, device: torch.device) -> dict:
    """`model` is the LoRA-wrapped model from training.__main__ (not yet on a device)."""
    source = next(s for s in cfg.data.sources if s.name == cfg.smoke.source)
    task = get_task(source.task)
    records = [r for r in read_crops(source.path) if r.split == "train"]
    samples = [task(r, source.path) for r in records[: cfg.smoke.samples]]
    encoded = [
        encode_sample(s.prompt, s.target, load_image(s.image), tokenizer, cfg.image)
        for s in samples
    ]
    checks: dict[str, dict] = {}

    # S2/S3: LoRA sits on the configured modules and nothing else trains.
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    not_lora = [n for n in trainable if "lora_" not in n]
    lora_modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))
    expected = _expected_lora_params(model, cfg.lora.r)
    checks["S3_freeze"] = {
        "pass": lora_modules > 0 and not not_lora and sum(trainable.values()) == expected,
        "lora_modules": lora_modules,
        "trainable_params": sum(trainable.values()),
        "expected_params": expected,
        "unexpected_trainable": not_lora[:10],
    }

    # S4: loss positions decode to exactly target + EOS.
    first = encoded[0]
    decoded = tokenizer.decode(
        first["input_ids"][first["labels"] != IGNORE_INDEX], clean_up_tokenization_spaces=False
    )
    checks["S4_label_mask"] = {
        "pass": decoded == samples[0].target + tokenizer.eos_token,
        "decoded": decoded[:200],
    }

    model.to(device)
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    batch = _to_device(collate(encoded, tokenizer.pad_token_id), device)

    # S5/S6: finite loss; gradients reach LoRA.
    loss = model(**batch).loss
    loss.backward()
    with_grad = sum(
        1
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    )
    checks["S5_S6_forward_backward"] = {
        "pass": math.isfinite(loss.item()) and with_grad > 0,
        "loss": loss.item(),
        "lora_tensors_with_grad": with_grad,
    }

    # S7: a few steps on the same batch must drive the loss down.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(cfg.smoke.steps):
        optimizer.zero_grad()
        step_loss = model(**batch).loss
        step_loss.backward()
        optimizer.step()
        losses.append(step_loss.item())
    checks["S7_overfit"] = {
        "pass": losses[-1] < 0.5 * losses[0],
        "first": losses[0],
        "last": losses[-1],
    }

    # S8: saved adapter reloads to the same loss.
    model.eval()
    with torch.no_grad():
        before = model(**batch).loss.item()
    with tempfile.TemporaryDirectory() as folder:
        model.save_pretrained(folder)
        reloaded = load_model(cfg.model, adapter=Path(folder)).to(device).eval()
        with torch.no_grad():
            after = reloaded(**batch).loss.item()
    del reloaded
    checks["S8_save_reload"] = {
        "pass": abs(before - after) < 1e-2,
        "before": before,
        "after": after,
    }

    # S9: speed and peak memory on the longest samples at the configured micro-batch.
    longest = sorted(records, key=lambda r: len(r.text), reverse=True)[
        : cfg.train.per_device_batch_size
    ]
    long_samples = [task(r, source.path) for r in longest]
    long_batch = _to_device(
        collate(
            [
                encode_sample(s.prompt, s.target, load_image(s.image), tokenizer, cfg.image)
                for s in long_samples
            ],
            tokenizer.pad_token_id,
        ),
        device,
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start, steps = time.perf_counter(), 3
    for _ in range(steps):
        model(**long_batch).loss.backward()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - start) / steps
    tokens = int(long_batch["attention_mask"].sum())
    checks["S9_speed_memory"] = {
        "pass": True,
        "micro_batch": len(long_samples),
        "seq_len": long_batch["input_ids"].shape[1],
        "seconds_per_step": seconds,
        "tokens_per_second": tokens / seconds,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    return checks


def write_report(checks: dict, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, result in checks.items():
        log.info(
            "%s %s %s",
            "PASS" if result["pass"] else "FAIL",
            name,
            {k: v for k, v in result.items() if k != "pass"},
        )
    return all(result["pass"] for result in checks.values())
