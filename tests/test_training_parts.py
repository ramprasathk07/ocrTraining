import sys
import types
from pathlib import Path

import pytest
import torch

from models.deepseek_ocr.loader import allow_vision_gradients
from training.config import load_config
from training.inventory import component_of
from training.lora import build_optimizer

FAKE_MODELING = """
import torch

class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)

    def forward(self, x):
        with torch.no_grad():  # the official forward wraps the whole vision path like this
            return torch.cat([self.encoder(x)], dim=0)
"""


def test_vision_gradient_fix_lets_encoder_learn():
    module = types.ModuleType("fake_modeling")
    exec(FAKE_MODELING, module.__dict__)
    sys.modules["fake_modeling"] = module
    outer = types.SimpleNamespace(model=module.Vision())
    x = torch.ones(1, 2)

    assert not outer.model(x).requires_grad  # as shipped: gradients are cut
    allow_vision_gradients(outer)
    outer.model(x).sum().backward()
    assert outer.model.encoder.weight.grad is not None
    with torch.no_grad():  # an outer no_grad (evaluation) still holds
        assert not outer.model(x).requires_grad


def test_optimizer_gives_each_lora_group_its_learning_rate():
    cfg = load_config(Path("configs/stage1/1a_printed.toml"))
    model = torch.nn.ModuleDict(
        {"qwen2_model": torch.nn.Linear(2, 2), "layers": torch.nn.Linear(2, 2)}
    )
    optimizer = build_optimizer(model, cfg)
    by_lr = {g["lr"]: {id(p) for p in g["params"]} for g in optimizer.param_groups}
    assert by_lr[1e-4] == {id(p) for p in model["layers"].parameters()}
    assert by_lr[5e-5] == {id(p) for p in model["qwen2_model"].parameters()}


def test_optimizer_refuses_a_group_that_matches_nothing():
    cfg = load_config(Path("configs/stage1/1a_printed.toml"))
    with pytest.raises(ValueError, match="match no trainable parameter"):
        build_optimizer(torch.nn.ModuleDict({"layers": torch.nn.Linear(2, 2)}), cfg)


@pytest.mark.parametrize(
    ("name", "component"),
    [
        ("base_model.model.model.sam_model.net_3.weight", "visual tokenizer (SAM + convs)"),
        (
            "base_model.model.model.qwen2_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight",
            "DeepEncoder V2",
        ),
        ("model.layers.4.self_attn.o_proj.weight", "decoder attention"),
        ("model.layers.4.mlp.experts.9.up_proj.weight", "decoder routed experts"),
        ("model.layers.4.mlp.shared_experts.down_proj.weight", "decoder shared experts"),
        ("model.layers.4.mlp.gate.weight", "decoder router"),
        ("model.layers.0.mlp.gate_proj.weight", "decoder dense MLP (layer 0)"),
        ("model.projector.layers.weight", "projector"),
        ("model.embed_tokens.weight", "embeddings"),
        ("lm_head.weight", "LM head"),
        ("model.view_seperator", "other (norms, separator)"),
    ],
)
def test_component_of(name, component):
    assert component_of(name) == component
