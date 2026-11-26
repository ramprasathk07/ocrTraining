"""Profile the first optimizer steps of a training run (`python -m training ... --profile N`).

Writes to <output_dir>/profile/, one set per rank:
    rank<r>.json.gz         Chrome trace: open in https://ui.perfetto.dev or chrome://tracing
    rank<r>_top_ops.txt     the most expensive ops over the recorded steps (no viewer needed)

Steps 1-2 are skipped (allocator, cuDNN and dataloader warm-up would distort them); steps
3 .. N+2 are recorded, then training stops. A profiling run saves no model and writes no
run_manifest or prompt-registry entry. `label_modules` names the model's parts in the trace.
How to run it on a server and read the trace: docs/training-throughput.md §8.
"""

import re
from functools import partial
from pathlib import Path

import torch
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import TrainerCallback

SKIPPED_STEPS = 2  # schedule: wait 1 + warmup 1
# SAM, DeepEncoder V2, projector, LM head, and each decoder layer's attention and MLP.
# DeepEncoder V2 has its own `layers.N.self_attn`; it is shown as one block.
_LABELLED = re.compile(
    r"(sam_model|qwen2_model|projector|lm_head)$|^(?!.*qwen2_model).*\.layers\.\d+\.(self_attn|mlp)$"
)


def label_modules(model: torch.nn.Module) -> int:
    """Wrap each labelled module's forward (and its recompute under checkpointing) in a range."""
    count = 0
    for name, module in model.named_modules():
        if _LABELLED.search(name):
            open_ranges: list[record_function] = []
            label = name.removeprefix("base_model.model.")  # PEFT wrapper prefix
            module.register_forward_pre_hook(partial(_open_range, open_ranges, label))
            module.register_forward_hook(partial(_close_range, open_ranges))
            count += 1
    return count


def _open_range(open_ranges: list, label: str, module, args) -> None:
    open_ranges.append(record_function(label).__enter__())


def _close_range(open_ranges: list, module, args, output) -> None:
    open_ranges.pop().__exit__(None, None, None)


class ProfileSteps(TrainerCallback):
    def __init__(self, steps: int, out_dir: Path):
        if steps < 1:
            raise ValueError(f"--profile needs at least 1 step, got {steps}")
        self.steps, self.out_dir, self.rank = steps, out_dir, 0
        # CPU only where the build has no CUPTI (Windows wheels); Linux CUDA wheels add GPU kernels
        activities = torch.profiler.supported_activities()
        self.sort_by = (
            "self_device_time_total"
            if ProfilerActivity.CUDA in activities
            else "self_cpu_time_total"
        )
        self.profiler = profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=steps, repeat=1),
            on_trace_ready=self._save,
            record_shapes=True,
            profile_memory=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.rank = args.process_index
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()
        if state.global_step >= SKIPPED_STEPS + self.steps:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.stop()  # saves the trace too if training ended before the last step

    def _save(self, prof) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(self.out_dir / f"rank{self.rank}.json.gz"))
        table = prof.key_averages().table(sort_by=self.sort_by, row_limit=60)
        (self.out_dir / f"rank{self.rank}_top_ops.txt").write_text(table, encoding="utf-8")
