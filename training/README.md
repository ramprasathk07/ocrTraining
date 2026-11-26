# training/

LoRA training of one stage or sub-stage from one run config (`configs/stage<N>/*.toml`). The same command runs on 1 or N GPUs. The GPU count comes only from `--nproc_per_node`.

```bash
python -m training.stages                                                        # stage tree + status
python -m training.inventory --config configs/stage1/1a_printed.toml             # what LoRA will train (no download)
torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --smoke   # smoke tests
torchrun --nproc_per_node=N -m training --config configs/stage1/1a_printed.toml           # train
python -m training --config configs/stage1/1a_printed.toml --merge outputs/runs/stage1a_printed/checkpoint-3000
torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --profile 20  # traces, then stop
```

W&B / MLflow: switched on by keys in `.env` ([tracking/](../tracking/README.md)). They log loss, learning rate and dev loss, and upload each checkpoint if `.env` asks for it. TensorBoard logs always.

`--profile N` records steps 3…N+2 as Chrome traces in `<output_dir>/profile/` and then stops. It saves no model. How to run it on a server and read the trace: [docs/training-throughput.md §8](../docs/training-throughput.md#8-profiling-on-a-linux-gpu-server).

## Files

| File | Does |
|---|---|
| `__main__.py` | entry point: `train` / `--smoke` / `--merge`; deterministic sampler; writes `run_manifest.json`; registers prompts |
| `config.py` | loads and validates a run config (`[stage]`, `[model]`, `[image]`, `[lora]`, `[train]`, `[data]`, `[smoke]`); `grad_accum_steps` |
| `data.py` | `plan_samples`: the whole sample order (shares, curriculum phases, epochs) built once from the seed; `TaskDataset`; long targets dropped and counted |
| `lora.py` | `apply_lora` (rank/alpha per group), `build_optimizer` (one learning rate per group; fails if a group matches nothing) |
| `inventory.py` | parameters per component and LoRA ranks, built on the meta device |
| `smoke.py` | smoke checks S3–S9 (below), written to `smoke.json` |
| `profiling.py` | `--profile N`: torch profiler over the first steps, named ranges per model part, one trace per rank |
| `prompts.py` | training prompt registry: `report/prompts/training/` |
| `stages.py` | `python -m training.stages`: the stage tree from the configs; fails on duplicate ids, unknown parents or cycles |

## How a stage runs

1. Load `[model].name_or_path`: the original model, or the parent stage's `merged/`.
2. Add a fresh LoRA adapter.
   - Default: decoder attention, r16, α32, lr 1e-4.
   - `[[lora.groups]]`: DeepEncoder V2 attention, r8, α16, lr 5e-5.
   - Frozen: SAM, convs, projector, experts, router, embeddings, LM head.
3. Train on target tokens only, with cross-entropy (the prompt and image are masked).
4. Pick the best checkpoint by dev CER, then `--merge` it → `<checkpoint>/merged`. The next sub-stage's `[model]` points at that folder.

**Multi-GPU:**
- The config fixes `global_batch_size`. Gradient accumulation = global ÷ (per-device × GPUs).
- Each rank reads its own slice of one fixed sample plan, so 1 GPU and 8 GPUs train identical data.
- Only rank 0 writes.

## Smoke tests (`--smoke`, one GPU, needs only `[smoke].path`)

| Check | Verifies |
|---|---|
| S3_freeze | only LoRA parameters train, and their count matches the expected per-component count |
| S4_label_mask | the unmasked labels decode to exactly target + EOS |
| S5_S6_forward_backward | finite loss; every LoRA group gets non-zero gradients |
| S7_overfit | loss on `samples` items drops below half within `steps` steps |
| S8_save_reload | the reloaded adapter gives the same loss (within 0.01) |
| S9_speed_memory | records tokens/s and peak memory on the longest samples (always passes; read the numbers) |

## Outputs: `[train].output_dir` (git-ignored, e.g. `outputs/runs/stage1a_printed/`)

| Path | Content |
|---|---|
| `checkpoint-<step>/` | HF Trainer checkpoints (adapter weights + optimizer state) |
| `runs/` | TensorBoard logs |
| `run_manifest.json` | config, stage, GPU count, gradient accumulation, data counts, prompt versions |
| `smoke.json` | smoke check results |
| `adapter/` | final LoRA weights |
| `merged/` | base + LoRA in one model: the next stage's input |
| `profile/` | `--profile` runs only: `rank<r>.json.gz` (open in ui.perfetto.dev or chrome://tracing) + `rank<r>_top_ops.txt` |

On start, rank 0 also appends the run to `report/prompts/training/` (`registry.json`, `history.jsonl`, `runs.md`).

## Adding training variants

| Variant | Change |
|---|---|
| Different data mix, LoRA targets, image mode or schedule | config only |
| New sub-stage or stage | new config (see `configs/README.md`) |
| New training type | one task in `dataEngine/tasks.py` |
