import gzip
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerControl, TrainerState

from training.profiling import _LABELLED, SKIPPED_STEPS, ProfileSteps, label_modules


@pytest.mark.parametrize(
    ("name", "labelled"),
    [
        ("base_model.model.model.sam_model", True),
        ("base_model.model.model.qwen2_model", True),
        ("base_model.model.model.projector", True),
        ("base_model.model.lm_head", True),
        ("base_model.model.model.layers.3.self_attn", True),
        ("base_model.model.model.layers.3.mlp", True),
        ("base_model.model.model.qwen2_model.model.model.layers.3.self_attn", False),
        ("base_model.model.model.layers.3.self_attn.q_proj", False),
        ("base_model.model.model.layers.3.mlp.experts.7", False),
        ("base_model.model.model.sam_model.blocks.0", False),
    ],
)
def test_labelled_modules(name, labelled):
    assert bool(_LABELLED.search(name)) is labelled


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_model = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.lm_head(self.sam_model(x))


def test_profile_records_steps_then_stops(repo_tmp):
    model = Tiny()
    assert label_modules(model) == 2
    callback = ProfileSteps(3, repo_tmp / "profile")
    args, state, control = SimpleNamespace(process_index=0), TrainerState(), TrainerControl()

    callback.on_train_begin(args, state, control)
    for step in range(1, 20):
        model(torch.ones(2, 4)).sum().backward()
        state.global_step = step
        callback.on_step_end(args, state, control)
        if control.should_training_stop:
            break
    callback.on_train_end(args, state, control)

    assert state.global_step == SKIPPED_STEPS + 3
    with gzip.open(repo_tmp / "profile" / "rank0.json.gz", "rt", encoding="utf-8") as f:
        trace = f.read()
    assert '"sam_model"' in trace and '"lm_head"' in trace
    assert "sam_model" in (repo_tmp / "profile" / "rank0_top_ops.txt").read_text(encoding="utf-8")


def test_profile_needs_a_step(repo_tmp):
    with pytest.raises(ValueError, match="at least 1 step"):
        ProfileSteps(0, repo_tmp)
