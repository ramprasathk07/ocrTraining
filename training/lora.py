"""LoRA setup and the optimizer, shared by training and the smoke tests."""

import re

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from training.config import RunConfig


def apply_lora(model, cfg: RunConfig):
    """LoRA on `lora.target_modules`, with per-group rank/alpha. PEFT freezes everything else."""
    lora = cfg.lora
    return get_peft_model(
        model,
        PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            rank_pattern={g.pattern: g.r for g in lora.groups},
            alpha_pattern={g.pattern: g.alpha for g in lora.groups},
            bias="none",
        ),
    )


def build_optimizer(model, cfg: RunConfig) -> torch.optim.AdamW:
    """AdamW with one parameter group per LoRA group; the rest use train.learning_rate."""
    groups = {g.pattern: [] for g in cfg.lora.groups}
    default = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pattern = next((g.pattern for g in cfg.lora.groups if re.search(g.pattern, name)), None)
        (groups[pattern] if pattern else default).append(param)
    empty = [pattern for pattern, params in groups.items() if not params]
    if empty:
        raise ValueError(f"LoRA groups match no trainable parameter: {empty}")
    param_groups = [{"params": default, "lr": cfg.train.learning_rate}] if default else []
    param_groups += [{"params": groups[g.pattern], "lr": g.learning_rate} for g in cfg.lora.groups]
    return torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
