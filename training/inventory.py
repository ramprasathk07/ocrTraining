"""Module inventory: parameters per component and what this run's LoRA config would train.

    python -m training.inventory --config configs/stage1/1a_printed.toml

Builds the model on PyTorch's meta device: the real structure and module names, without
downloading weights. Run it after any change to `[lora]`, before spending GPU time.
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from training.config import load_config
from training.lora import apply_lora

# First match wins. Patterns work on plain and PEFT-wrapped parameter names.
COMPONENTS = (
    ("visual tokenizer (SAM + convs)", r"\.sam_model\."),
    ("DeepEncoder V2", r"\.qwen2_model\."),
    ("projector", r"\.projector\."),
    ("decoder attention", r"\.layers\.\d+\.self_attn\."),
    ("decoder routed experts", r"\.mlp\.experts\."),
    ("decoder shared experts", r"\.mlp\.shared_experts\."),
    ("decoder router", r"\.mlp\.gate\."),
    ("decoder dense MLP (layer 0)", r"\.layers\.0\.mlp\."),
    ("embeddings", r"embed_tokens"),
    ("LM head", r"lm_head"),
)


def component_of(name: str) -> str:
    return next(
        (label for label, p in COMPONENTS if re.search(p, name)), "other (norms, separator)"
    )


def count_by_component(model) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "trainable": 0})
    for name, param in model.named_parameters():
        entry = counts[component_of(name)]
        entry["params"] += param.numel()
        entry["trainable"] += param.numel() if param.requires_grad else 0
    return dict(counts)


def lora_ranks(model) -> Counter:
    """(component, rank, alpha) → number of LoRA modules."""
    return Counter(
        (component_of(name + "."), module.r["default"], module.lora_alpha["default"])
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and "default" in module.lora_A
    )


def main() -> None:
    from transformers import AutoConfig, AutoModel

    parser = argparse.ArgumentParser(prog="python -m training.inventory")
    parser.add_argument("--config", type=Path, required=True)
    cfg = load_config(parser.parse_args().config)
    config = AutoConfig.from_pretrained(
        cfg.model.name_or_path, revision=cfg.model.revision, trust_remote_code=True
    )
    with torch.device("meta"):
        model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = apply_lora(model, cfg)

    print(f"{'component':34s} {'params':>14s} {'trainable':>12s}")
    for label, entry in sorted(count_by_component(model).items(), key=lambda kv: -kv[1]["params"]):
        print(f"{label:34s} {entry['params']:14,d} {entry['trainable']:12,d}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'total':34s} {total:14,d} {trainable:12,d}")
    for (component, r, alpha), n in sorted(lora_ranks(model).items()):
        print(f"LoRA: {n:4d} modules in {component} (r={r}, alpha={alpha})")


if __name__ == "__main__":
    main()
