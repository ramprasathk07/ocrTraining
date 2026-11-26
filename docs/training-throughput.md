# Training throughput plan: DeepSeek-OCR 2

**Goal:** the most training samples per second per GPU, with exactly the same training result.
- Every change must give the same loss and gradients as the official forward pass (§5).
- Every number below marked *estimate* is a back-of-envelope figure until the profile (§8) replaces it.

**Status (2026-11-23):**
- Profiling is built (`--profile N`, §8).
- Phases 1–3 are proposed, not implemented.
- Nothing has been profiled yet.

Model code references are to the pinned revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`. The files are in the Hugging Face cache, under `snapshots/<revision>/`.

## 1. What we measure

| Metric | Definition | Why |
|---|---|---|
| samples/s per GPU | training samples processed per second | the number that matters for crops (Stage 1) |
| target tokens/s | label tokens (not prompt, not image) per second | comparable across tasks |
| positions/s | decoder sequence positions (image + prompt + target) per second | what S9 reports today (`tokens_per_second`) |
| MFU | model FLOPs per sample (§2, without recompute) × samples/s ÷ GPU peak | how much headroom is left |
| peak memory | `torch.cuda.max_memory_allocated` | decides batch size and gradient checkpointing |

Report Stage 1 (crops) and Stage 2 (pages) separately, because their bottlenecks differ (§2).

## 2. Where the compute goes

The architecture, read from `config.json` and the model code:

| Part | Shape | Trainable in Stage 1 |
|---|---|---|
| SAM ViT-B + convs | 12 blocks, width 768, global attention in blocks 2/5/8/11, window attention elsewhere; 16-px patches; `net_2` 256→512, `net_3` 512→896, both stride 2 | frozen, forward only |
| DeepEncoder V2 | Qwen2-style, 24 layers, width 896, 358M parameters; input = n image tokens + n learned queries (n = 144 at 768, 256 at 1024); custom mask (image tokens bidirectional, queries causal) | LoRA r8 on attention, so backward runs |
| projector | Linear 896→1280 | frozen |
| decoder | DeepseekV2, 12 layers, hidden 1280, 10-head MHA; layer 0 dense MLP (6848); layers 1–11 MoE: 64 routed experts (top-6) + 2 shared, expert width 896 | LoRA r16 on attention |
| LM head | 1280 × 129,280 vocab | frozen |

The decoder uses about **408M parameters per token**:

| Piece | Parameters |
|---|---|
| attention | 78.6M |
| dense MLP (layer 0) | 26.3M |
| MoE layers: 11 × (6 routed + 2 shared) | 302.8M |

**How the estimates are computed:**
- Forward FLOPs ≈ 2 × parameters × tokens, plus attention 4 × T² × width per layer.
- Backward through frozen weights computes input gradients only, so it costs about 1× the forward. LoRA weight gradients are negligible.
- Gradient checkpointing adds one more forward (recompute).

**Word or line crop, 768 view** (*estimate*). About 160 decoder positions (145 image + prompt + ~8 target), gradient checkpointing on:

| Part | Forward | Backward | Recompute | Total TFLOP | Share |
|---|---|---|---|---|---|
| SAM + convs (2,304 patches) | 0.46 | — | — | 0.46 | 29% |
| DeepEncoder V2 (288 tokens) | 0.21 | 0.21 | 0.21 | 0.62 | 39% |
| decoder body | 0.13 | 0.13 | 0.13 | 0.39 | 25% |
| LM head, all 160 positions | 0.05 | 0.05 | — | 0.11 | 7% |
| **sample** | | | | **≈1.6** | |

**Page, tiles mode** (*estimate*). One 1024 view plus 6 tiles of 768, so about 1,121 image positions. With a ~1,500-token target the sequence is about 2,700:

| Part | Total TFLOP | Share |
|---|---|---|
| SAM + convs (1 × 4,096 + 6 × 2,304 patches) | 3.7 | 20% |
| DeepEncoder V2 (512 + 6 × 288 tokens) | 4.8 | 26% |
| decoder body including attention | 8.0 | 44% |
| LM head, all positions | 1.8 | 10% |
| **page** | **≈18** | |

What this means:
- **Crops (Stage 1):** ~70% of the compute is on the vision side (SAM + DeepEncoder V2). The decoder is small because targets are short.
- **Pages (Stage 2):** the decoder and long sequences dominate.
- **1A image mode:** a 1024 padded crop costs about **2.9 TFLOP, 1.8× the 768 view**. Set that against its better baseline CER when choosing the 1A mode.
- **Rough ceiling on an RTX A6000:**
  - NVIDIA's dense BF16 tensor peak for the A6000 is 154.8 TFLOPS.
  - At 30–40% MFU that gives about **29–39 crops/s** or **2.6–3.4 pages/s** per GPU.
  - This is a target to measure against, not a promise.

## 3. What slows down the official training forward

These problems come from the model's own code. Ours only calls it.

| # | Where (pinned revision) | What happens | Effect |
|---|---|---|---|
| 1 | `modeling_deepseekv2.py` `DeepseekV2MoE.forward`, lines 612–621 (training branch) | Python loop over 64 experts: `y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])` | Each boolean-mask index forces a GPU→CPU sync: ~128 per MoE layer, ~1,400 per forward, ~2,800 with recompute. Each expert then multiplies only ~100 tokens. The GPU mostly waits. |
| 2 | `modeling_deepseekocr2.py` `DeepseekOCR2Model.forward`, lines 404–495 | Vision runs **one image at a time**: SAM → DeepEncoder V2 → projector per sample, with `torch.sum(...).item()` at lines 404 and 420, then `masked_scatter_` | The part that is ~70% of crop compute runs at batch size 1, with 2 syncs per sample |
| 3 | `modeling_deepseekocr2.py` lines 575–590 | `lm_head` over **every** position, cast to fp32, then cross-entropy over all of them | A crop has ~8 target tokens out of ~160 positions. Logits for a batch of 8 take ~0.7 GB in fp32 plus their gradient, which limits batch size |
| 4 | `modeling_deepseekv2.py` gate, lines 507–533, and `AddAuxiliaryLoss`, 538–555 | In training mode the router's load-balancing loss is active (`aux_loss_alpha = 0.001`, `seq_aux = True`, defaults in `configuration_deepseek_v2.py`) | **Correctness bug, not just speed.** Its gradient flows through the frozen router into our LoRA weights, breaking the plan's "plain cross-entropy only" rule. Setting `alpha = 0` alone crashes: `AddAuxiliaryLoss` asserts on the `None` loss. |
| 5 | `modeling_deepseekv2.py` `ATTENTION_CLASSES`, lines 1230–1239 | The decoder has only `mha_eager` and `mha_flash_attention_2`, no SDPA | Without flash-attn (Windows, the 3060, ROCm until built) attention is eager: O(T²) memory, which matters for pages |

Already reasonable, so not targeted:
- DeepEncoder V2 attention uses SDPA with its custom mask (`deepencoderv2.py` line 205).
- SAM attention uses SDPA with relative-position bias (lines 795–798).

## 4. The plan

### Phase 0: measure (½ day, Linux GPU server)

1. Run `--profile` (§8) on a crop config and, once Stage 2 exists, on a page config.
2. From the trace, note per part (SAM, DeepEncoder V2, decoder attention, decoder MLP/MoE, LM head + loss):
   - time and share;
   - GPU idle share;
   - sync count (`aten::nonzero`, `aten::item`, `cudaStreamSynchronize`);
   - dataloader wait.
3. Fill in the results log (§9).
4. Extend smoke check S9 to report samples/s and target tokens/s next to positions/s.

### Phase 1: exact rewrites in plain PyTorch (1–2 days; also runs on the 3060 and ROCm)

| Step | Change | Code | Fixes |
|---|---|---|---|
| 1a | Replace the MoE training forward:<br>- sort token→expert assignments once;<br>- `bincount` → **one** sync;<br>- contiguous slices per expert;<br>- `index_add_` back with the routing weights;<br>- no aux loss.<br>Same results as the original up to float summation order. | `models/deepseek_ocr/loader.py`, patched like the existing `no_grad` fix | #1, #4 |
| 1b | Own training forward:<br>1. batch all global views of the same size through SAM → DeepEncoder V2 → projector in **one** call, and all tiles in another;<br>2. scatter the results into the token embeddings;<br>3. call the decoder (`DeepseekV2Model.forward`) with `inputs_embeds`;<br>4. apply `lm_head` **only at positions whose shifted label ≠ −100**;<br>5. take cross-entropy in fp32 on those. | new `models/deepseek_ocr/train_forward.py`; `PlannedOrderTrainer.compute_loss` calls it | #2, #3 |
| 1c | Sort samples by input length **within each global batch**, so micro-batches pad less. Each optimizer step still sees the same samples, so results stay identical for any GPU count. | `training/data.py` (`plan_samples`) | padding waste |
| 1d | Per stage: gradient checkpointing off if S9 shows memory fits (it costs ~30% extra compute); then raise `per_device_batch_size` | configs only | recompute cost |
| 1e | Register `mha_sdpa` → `LlamaSdpaAttention` (still present in transformers 4.46.3) and allow `attn_implementation = "sdpa"` | `models/deepseek_ocr/loader.py` | #5 |

Expected: the largest jump of all phases, because 1a and 1b remove the GPU idle time rather than shaving FLOPs. Re-profile after Phase 1 before starting Phase 2.

### Phase 2: kernels (Linux; only for parts the re-profile still shows as expensive)

| Kernel | For | Notes |
|---|---|---|
| Grouped GEMM for MoE experts (ScatterMoE in Triton, or CUTLASS `grouped_gemm`) | decoder MoE, if still >~20% of step time after 1a | experts are frozen, so only the input gradient is needed; check Ampere (sm86) support before adopting |
| FlashAttention-2 (`flash-attn==2.7.3`, already pinned) | decoder attention on the A6000 | matters for Stage 2 sequences of ~2,700+; padding-free packing is possible later because we own the forward after 1b |
| Liger kernels: fused RMSNorm, SwiGLU, RoPE | decoder + DeepEncoder V2 (built on transformers' Qwen2, so the stock Qwen2 patch applies) | mostly memory, which allows bigger batches; needs Triton (Linux); pin a version tested with transformers 4.46.3 |
| Fused linear + cross-entropy (Liger, chunked) | Stage 2 long targets | for crops, step 1b's position slicing is enough |
| Regional `torch.compile` (SAM blocks, decoder layers) | everything, once 1a/1b removed the graph breaks (`.item()`, boolean indexing) | step 1c's length bucketing limits recompiles; use `dynamic=True` |

### Phase 3: structural changes (only if the profile justifies them)

| Change | Gain | Cost |
|---|---|---|
| **Cache SAM features** for crop stages. SAM + convs are frozen and deterministic, so their 144 × 896 output can be precomputed once per crop | removes ~30% of crop compute (§2) | ~0.26 MB per 768-view crop (~52 GB for 200K crops); **rules out pixel augmentation** (blur, rotation, noise), which may matter for handwriting |
| **Faster evaluation:** batch the generation (currently batch 1), or vLLM if it supports OCR-2 (unverified) | wall-clock of every eval run | not training speed; separate from the training loop |

## 5. Verification for every change

| Check | Pass condition | Where |
|---|---|---|
| S10 (new): equivalence | same batch through the official forward and the fast forward: loss difference < 1e-3 (bf16), LoRA-gradient cosine > 0.999 per LoRA group | `training/smoke.py` |
| S7 overfit | still drops below half | existing |
| Loss curve | the first 200 steps of a short run overlay the previous code's curve | TensorBoard |
| Sample plan | `plan_samples` output unchanged (1c reorders only within a global batch) | `tests/test_sampling.py` |
| Speed | re-profile; log samples/s, MFU and peak memory in §9 | §8 |

## 6. GPUs

| | RTX A6000 48 GB (training) | RTX 3060 12 GB, Windows (smoke tests) | MI300X (Stage 3, text only) |
|---|---|---|---|
| Decoder attention | FlashAttention-2 | SDPA after 1e (no flash-attn on Windows) | SDPA, or CK flash-attn for ROCm |
| Phase 1 | yes | yes | yes |
| Phase 2 (Triton) | yes | no | partial |
| GPU kernels in profiler traces | yes | **no**: Windows builds record CPU ops only | yes (ROCm build) |

## 7. Not applicable

| Option | Why not |
|---|---|
| FP8 training | the A6000 is Ampere: no FP8 tensor cores |
| FlashAttention-3 | Hopper only |
| FSDP / ZeRO / expert parallelism | 3.4B parameters in bf16 (~6.8 GB) fit on every GPU; LoRA gradients are a few MB, so plain DDP all-reduce is cheap |
| CUDA graphs | input shapes change every batch |
| QLoRA | slower; only saves memory, which 48 GB doesn't need |

## 8. Profiling on a Linux GPU server

### Run

```bash
# 1. On the server, from the repo root, with the .venv active
nvidia-smi                                                    # the GPU is free
python -c "import torch; print(torch.profiler.supported_activities())"
#    must include ProfilerActivity.CUDA; if only CPU, reinstall the CUDA pip wheel (README Setup)

# 2. The config's [[data.sources]] must be converted (python -m dataEngine ... crops)
#    Run inside tmux so an SSH drop does not kill it
tmux new -s profile
torchrun --nproc_per_node=1 -m training --config configs/stage1/1a_printed.toml --profile 20 \
    2>&1 | tee outputs/profile_1a.log
```

What `--profile N` does (`training/profiling.py`):
- **Records:** training steps **3 … N+2**. Steps 1–2 are skipped, because allocator, cuDNN and dataloader warm-up would distort them. A "step" is one optimizer step: all its gradient-accumulation micro-batches.
- **Labels:**
  - `model.sam_model`, `model.qwen2_model`, `model.projector`, `lm_head`;
  - each decoder layer's `model.layers.N.self_attn` and `model.layers.N.mlp`.
  - These show as named ranges in the forward and, under gradient checkpointing, again in the recompute inside backward.
- **Records too:** op shapes and memory.
- **Then:** stops. It saves no checkpoint, adapter or merged model, and writes no `run_manifest.json` or prompt-registry entry.

**Outputs:** `<[train].output_dir>/profile/`, e.g. `outputs/runs/stage1a_printed/profile/`.

| File | Content |
|---|---|
| `rank<r>.json.gz` | Chrome trace of the recorded steps, one per rank. In it, `ProfilerStep#k` is training step k+1 |
| `rank<r>_top_ops.txt` | the 60 most expensive ops, sorted by self GPU time (CPU time where no GPU is traced) |

**Trace size:**
- With the official MoE loop (§3 #1), one micro-batch produces on the order of 10⁵ trace events. 20 steps × 8 accumulation micro-batches (1A on one GPU) can reach several GB uncompressed: fine for `top_ops.txt`, too big for a browser.
- **To get a trace you can open**, use a profiling copy of the config with `global_batch_size = per_device_batch_size` (1 micro-batch per step), or record fewer steps (`--profile 5`).
- Keep profiling copies **outside** `configs/stage*/`, e.g. `outputs/profile_configs/1a.toml`. Otherwise `python -m training.stages` sees a duplicate stage id.

### Copy to your machine

```bash
scp user@server:/path/to/ocrTraining/outputs/runs/stage1a_printed/profile/rank0.json.gz .
scp user@server:/path/to/ocrTraining/outputs/runs/stage1a_printed/profile/rank0_top_ops.txt .
```

Traces hold op names, shapes and timings, with no document text.

### Open

| Viewer | How | Limits |
|---|---|---|
| **Perfetto** (recommended) | open https://ui.perfetto.dev → *Open trace file* → `rank0.json.gz` | the trace is processed locally in the browser; nothing is uploaded; up to roughly 1–2 GB of trace |
| Perfetto with the native trace processor (large traces) | `curl -LO https://get.perfetto.dev/trace_processor && chmod +x trace_processor`, `gunzip -k rank0.json.gz`, `./trace_processor --httpd rank0.json`, then open ui.perfetto.dev and accept *use native accelerator* | limited by machine memory, not the browser |
| `chrome://tracing` (legacy) | in Chrome: *Load* → the file (gunzip first if it refuses `.gz`) | slow above a few hundred MB |

Useful Perfetto SQL (*Query (SQL)* tab), time per labelled part:

```sql
select name, count(*) as calls, round(sum(dur) / 1e6, 1) as total_ms
from slice
where name like 'model.%' or name = 'lm_head'
group by name order by total_ms desc;
```

### What to look for

| In the trace | Means | Phase-1 step |
|---|---|---|
| Gaps in the GPU stream rows while CPU rows are busy | GPU waiting on Python or syncs | 1a, 1b |
| `aten::nonzero`, `aten::item`, `aten::_local_scalar_dense`, `cudaStreamSynchronize` inside `model.layers.N.mlp` | the MoE expert loop (§3 #1) | 1a |
| Many tiny GEMM kernels per `mlp` range | ~100 tokens per expert | 1a, then grouped GEMM |
| `model.sam_model` / `model.qwen2_model` repeated once per sample inside a micro-batch | batch-1 vision loop (§3 #2) | 1b |
| Long `lm_head` + `log_softmax` / `nll_loss` kernels, big memory spike at the end of forward | full-vocab logits at all positions (§3 #3) | 1b |
| `enumerate(DataLoader)…__next__` taking visible time between steps | input pipeline too slow: raise `num_workers` or pre-decode images | — |
| Labelled ranges inside the backward region | gradient-checkpointing recompute | 1d |
| `nccl:all_reduce` not overlapping compute (multi-GPU only) | communication on the critical path (unexpected with LoRA-sized gradients) | — |

**Multi-GPU:** `--nproc_per_node=N --profile 5` writes one trace per rank. Profile on one GPU first; use multi-GPU only to check communication. For a cross-rank summary (idle vs compute vs communication time per rank), Meta's Holistic Trace Analysis (`pip install HolisticTraceAnalysis`, `TraceAnalysis(trace_dir=...)`) reads these traces. See its docs for the current API.

**Overhead:** shape and memory recording slow the run down. Use traces for **where** time goes, not for absolute speed: absolute speed comes from S9 or a normal run's logged throughput.

## 9. Results log

Add one row per profile or change.

| Date | Config | GPU | Code state | samples/s/GPU | target tok/s | MFU | Peak GB | Top cost in trace | Trace |
|---|---|---|---|---|---|---|---|---|---|
| | | | baseline (official forward) | | | | | | |

## 10. Sources

- Model code and config: `deepseek-ai/DeepSeek-OCR-2` at revision `aaa02f38`: `modeling_deepseekv2.py`, `modeling_deepseekocr2.py`, `deepencoderv2.py`, `configuration_deepseek_v2.py`.
- PyTorch profiler: https://pytorch.org/docs/stable/profiler.html
- Perfetto UI and trace processor: https://perfetto.dev/docs/
- Holistic Trace Analysis: https://github.com/facebookresearch/HolisticTraceAnalysis
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- ScatterMoE: https://github.com/shawntan/scattermoe
- NVIDIA RTX A6000 datasheet (tensor throughput).
