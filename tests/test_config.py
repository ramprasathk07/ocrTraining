import re
from pathlib import Path

import pytest

from training.config import grad_accum_steps, load_config

STAGE1 = Path("configs/stage1_script.toml")


def test_stage1_config_loads():
    cfg = load_config(STAGE1)
    assert [s.name for s in cfg.data.sources] == ["iiit_hw_dev", "mozhi_hi", "golden_regions"]
    assert cfg.model.revision == "9f30c71f441d010e5429c532364a86705536c53a"
    assert cfg.lora.r == 16


@pytest.mark.parametrize(("world_size", "accum"), [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_accumulation_keeps_the_global_batch(world_size, accum):
    assert grad_accum_steps(64, 8, world_size) == accum


def test_uneven_global_batch_is_rejected():
    with pytest.raises(ValueError):
        grad_accum_steps(64, 8, 3)


def test_lora_regex_hits_exactly_the_planned_84_modules():
    pattern = re.compile(load_config(STAGE1).lora.target_modules)
    wanted = (
        [f"model.layers.{i}.self_attn.{p}_proj" for i in range(12) for p in "qkvo"]
        + [f"model.layers.0.mlp.{p}_proj" for p in ("gate", "up", "down")]
        + [
            f"model.layers.{i}.mlp.shared_experts.{p}_proj"
            for i in range(1, 12)
            for p in ("gate", "up", "down")
        ]
    )
    frozen = [
        "model.layers.3.mlp.gate",  # router
        "model.layers.3.mlp.experts.7.gate_proj",  # routed expert
        "model.vision_model.transformer.layers.0.self_attn.qkv_proj",
        "model.sam_model.blocks.0.attn.qkv",
        "model.projector.layers",
        "lm_head",
    ]
    assert len(wanted) == 84
    assert all(pattern.fullmatch(name) for name in wanted)
    assert not any(pattern.fullmatch(name) for name in frozen)
