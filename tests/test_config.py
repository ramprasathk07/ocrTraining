import re
from pathlib import Path

import pytest

from training.config import grad_accum_steps, load_config

STAGE1 = sorted(Path("configs/stage1").glob("*.toml"))
DECODER_ATTENTION = [f"model.layers.{i}.self_attn.{p}_proj" for i in range(12) for p in "qkvo"]
ENCODER_ATTENTION = [
    f"model.qwen2_model.model.model.layers.{i}.self_attn.{p}_proj"
    for i in range(24)
    for p in "qkvo"
]
FROZEN = [  # real module names from the OCR-2 inventory
    "model.layers.3.mlp.gate",
    "model.layers.3.mlp.experts.7.gate_proj",
    "model.layers.3.mlp.shared_experts.up_proj",
    "model.layers.0.mlp.gate_proj",
    "model.qwen2_model.model.model.layers.5.mlp.up_proj",
    "model.sam_model.blocks.0.attn.qkv",
    "model.sam_model.net_3",
    "model.projector.layers",
    "lm_head",
]


def test_three_stage1_configs_exist():
    assert [p.name for p in STAGE1] == ["1a_printed.toml", "1b_handwritten.toml", "1c_joint.toml"]


@pytest.mark.parametrize("path", STAGE1, ids=lambda p: p.stem)
def test_stage1_lora_targets_encoder_and_decoder_attention_only(path):
    cfg = load_config(path)
    target = re.compile(cfg.lora.target_modules)
    assert all(target.fullmatch(n) for n in DECODER_ATTENTION + ENCODER_ATTENTION)
    assert len(DECODER_ATTENTION + ENCODER_ATTENTION) == 144
    assert not any(target.fullmatch(n) for n in FROZEN)

    (encoder,) = cfg.lora.groups
    assert (encoder.r, encoder.alpha, encoder.learning_rate) == (8, 16, 5e-5)
    assert (cfg.lora.r, cfg.lora.alpha, cfg.train.learning_rate) == (16, 32, 1e-4)
    # PEFT matches rank_pattern keys as a name suffix: r".*\.{key}$"
    suffix = re.compile(rf".*\.{encoder.pattern}$")
    assert all(suffix.match(n) for n in ENCODER_ATTENTION)
    assert not any(suffix.match(n) for n in DECODER_ATTENTION)


def test_stages_chain_from_each_other():
    configs = [load_config(p) for p in STAGE1]
    assert [(c.stage.id, c.stage.starts_from) for c in configs] == [
        ("1A", "base"),
        ("1B", "1A"),
        ("1C", "1B"),
    ]
    # each stage starts from the merged output of the stage it names
    by_id = {c.stage.id: c for c in configs}
    for c in configs:
        if c.stage.starts_from != "base":
            parent = by_id[c.stage.starts_from]
            assert c.model.name_or_path == (parent.train.output_dir / "merged").as_posix()
    assert by_id["1A"].model.name_or_path == "deepseek-ai/DeepSeek-OCR-2"


@pytest.mark.parametrize(("world_size", "accum"), [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_accumulation_keeps_the_global_batch(world_size, accum):
    assert grad_accum_steps(64, 8, world_size) == accum


def test_uneven_global_batch_is_rejected():
    with pytest.raises(ValueError):
        grad_accum_steps(64, 8, 3)
