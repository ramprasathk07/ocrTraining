# Stage 1 — Script adaptation on DeepSeek-OCR 2 (printed + handwritten, Hindi + English)

Teach DeepSeek-OCR 2 to **read Hindi characters**, printed and handwritten, and handwritten English, from crops (words, lines, regions). It runs as three sub-stages, so each improvement can be measured separately:

| Sub-stage | Starts from | Learns from | Image mode | Output |
|---|---|---|---|---|
| **1A — printed** | original DeepSeek-OCR 2 | Mozhi printed Hindi | one 768 view | `stage1a_printed/merged` |
| **1B — handwritten** | best 1A, merged | IIIT-HW-Dev, DohaScript (Hindi), IAM (English) | one 768 view | `stage1b_handwritten/merged` |
| **1C — joint** | best 1B, merged | your golden regions + balanced printed/handwritten replay | 1024 global + 768 tiles | `stage1c_joint/merged` |

- Weeks: W2–W3 (2026-11-30 → 12-13).
- Configs: `configs/stage1/1a_printed.toml`, `stage1b_handwritten.toml`, `stage1c_joint.toml`.
- Hardware: A6000.
- GPU-hours (estimates; recompute from smoke S9): 1A 6–11, 1B 9–17, 1C 10–20.

Part of the [training plan](../training-plan.md).

## 1. DeepSeek-OCR 2 architecture and what trains

Measured with `python -m training.inventory --config configs/stage1/1a_printed.toml`, which builds the real model structure on PyTorch's meta device.

| Component | Module path | Params | Stage 1 |
|---|---|---|---|
| Visual tokenizer: SAM ViT-B + conv downsampler (`net_2`, `net_3`) | `model.sam_model` | 95.0M | frozen |
| DeepEncoder V2 (Qwen2, 24 layers, hidden 896) | `model.qwen2_model.model.model.layers.N` | 358.3M | **LoRA r8 on attention** (`self_attn.{q,k,v,o}_proj`) |
| projector (896 → 1280) | `model.projector` | 1.1M | frozen |
| Decoder attention (12 layers) | `model.layers.N.self_attn.{q,k,v,o}_proj` | 78.6M | **LoRA r16** |
| Decoder routed experts (64 × 11 layers) | `model.layers.N.mlp.experts.N.*` | 2,422M | frozen (follow-up experiment) |
| Decoder shared experts / layer-0 dense MLP | `mlp.shared_experts.*` / `model.layers.0.mlp.*` | 75.7M / 26.3M | frozen |
| Router | `model.layers.N.mlp.gate` | 0.9M | frozen |
| Embeddings / LM head | `model.embed_tokens` / `lm_head` | 165.5M each | frozen (tokenizer unchanged) |

**Trainable:**

| LoRA group | Modules | Parameters |
|---|---|---|
| Encoder | 96 | 1,081,344 |
| Decoder | 48 | 1,966,080 |
| **Total** | | **3,047,424** of 3.39B |

Smoke test S3 checks this count.

**Why train both encoder and decoder:** DeepEncoder V2 decides how visual information from Devanagari and English strokes is organised. The decoder learns to transcribe it. Training only one side leaves adaptation unused on the other.

**Important fix (in `models/deepseek_ocr/loader.py`):** the official forward runs SAM, DeepEncoder V2 and the projector inside `torch.no_grad()`. Without a fix, encoder LoRA would receive **zero gradient** and silently not train.
- The loader disables that `no_grad`.
- Frozen parts still build no autograd graph, because nothing in them requires a gradient.
- Smoke test S6 fails if any LoRA group gets no gradient.

**Our recipe vs the paper's:** the OCR-2 authors updated encoder and decoder fully, then froze the encoder and trained the decoder. Ours is a parameter-efficient LoRA adaptation for a smaller domain dataset, not a reproduction of their recipe.

## 2. LoRA settings (all three sub-stages)

| | DeepEncoder V2 attention | Decoder attention |
|---|---|---|
| Rank / alpha / dropout | 8 / 16 / 0.05 | 16 / 32 / 0.05 |
| Learning rate | 5e-5 | 1e-4 |
| Config | `[[lora.groups]]` with `pattern = 'qwen2_model\..*'` | `[lora]` defaults + `train.learning_rate` |

Each sub-stage loads the previous merged checkpoint and adds a **fresh** adapter. So 1B is a separate handwriting adapter on top of 1A: you can measure whether it improves handwriting without damaging printed reading. All are starting values, not proven optima.

## 3. Data

**Rules:**
- **Trusted sources only:**
  - Mozhi and IIIT-HW-Dev from the official CVIT / `ilocr.iiit.ac.in` downloads.
  - DohaScript from its authors.
  - IAM from the FKI group (fki.tic.heia-fr.ch; free for non-commercial research, registration).

  Never use third-party re-uploads. Record URL, licence and SHA-256 at download.
- **Golden regions enter only in 1C.** 1A and 1B use public data only, so the private data is untouched until the joint stage.
- **English:**
  - Handwritten English: IAM in 1B.
  - Printed English: no trusted word/line dataset exists, so 1A is printed Hindi only. English printed text appears in your golden regions in 1C, and the base model already reads printed English well.
- Only train splits. Dev and test never enter training. Text is NFC-normalized.

**Per sub-stage** (shares = sampling probabilities):

| | 1A printed | 1B handwritten | 1C joint |
|---|---|---|---|
| `mozhi_hi` (printed Hindi words; Mozhi has no lines) | 100% | — | 30% |
| `iiit_hw_dev` (handwritten Hindi words) | — | 50% | 15% |
| `dohascript` (handwritten Hindi lines) | — | 25% | 10% |
| `iam_lines` (handwritten English lines) | — | 25% | 5% |
| `golden_regions` (your golden regions, ~21K train) | — | — | 40% |
| Epoch size × epochs | 100K × 2 | 150K × 2 | 100K × 2 |
| Epoch-1 curriculum | words → words + lines | words → words + lines | none (mixed) |

## 4. Task format and image modes

| Prompt | Target |
|---|---|
| `<image>\nFree OCR.` | the exact text: line breaks and `[illegible]` / `[uncertain: …]` markers kept, then the end token |

Image preprocessing copies the model's own `infer()` exactly (`models/deepseek_ocr/inputs.py`). The same code serves training and evaluation.

| Mode (config `[image]`) | What the model sees | Image-token positions |
|---|---|---|
| `crop_mode = false`, `image_size = 768` (1A, 1B) | crop **stretched** to 768×768 | 145 (12×12 queries + separator) |
| `crop_mode = false`, `image_size = 1024` (later run) | crop padded to 1024 | 257 |
| `crop_mode = true` (1C) | 1024 global view; crops larger than 768 also get 2–6 tiles of 768 | 257 + 144 per tile |

DeepEncoder V2 has learned queries only for 768 and 1024 inputs. Other sizes are rejected.

1C uses tiles because golden regions are paragraph-sized. Stretching a whole handwritten letter to 768 would shrink the writing too far.

## 5. Settings (starting values, all sub-stages)

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.01, one parameter group per LoRA group |
| Schedule | cosine, 3% warmup, gradient clipping 1.0 |
| Precision | bf16, gradient checkpointing on |
| Batch | global 64 (fixed for any GPU count), micro-batch 8 |
| Max target length | 1,536 tokens; longer targets are dropped and counted, never truncated |
| Loss | cross-entropy on target tokens only |
| Checkpoints | every 1,000 steps; dev loss every 1,000 steps |

## 6. Evaluation before and after

The same command runs for every sub-stage. The first `--checkpoint` is "before" and later ones are compared with it. `start` means the config's starting model.

```bash
torchrun --nproc_per_node=N -m evaluation --config configs/stage1/1a_printed.toml \
    --checkpoint start --checkpoint outputs/runs/stage1a_printed/checkpoint-2000 \
    --checkpoint outputs/runs/stage1a_printed/merged
```

**What it produces:**
- per checkpoint: `predictions.jsonl` and `scores.json`
- `report/<run>/report_dev.md` (+ `.json`):
  - CER, WER and exact-match per source, per script, per language, and per script × language
  - deltas against "before"
  - gate pass/fail
  - the most improved and most regressed items, with target, before and after

Predictions are cached and reused while items, image mode, generation settings and weight files are unchanged. So "before" is generated only once.

**Eval suite:** identical in all three configs: dev splits of `golden_regions`, `mozhi_hi`, `iiit_hw_dev`, `dohascript`, `iam_lines`. Sources not converted yet are skipped and named in the report.

**Gates** (proposed; confirm against the measured "before"):

| Sub-stage (before) | Gate |
|---|---|
| 1A (original OCR-2) | Mozhi Hindi CER ≤ 3%; handwritten CER and golden CER rise by ≤ 0.5 points |
| 1B (1A merged) | handwritten-Hindi CER drops ≥ 30% relative; printed CER and golden CER rise by ≤ 0.5 points |
| 1C (1B merged) | golden CER drops ≥ 10% relative; handwritten and printed CER rise by ≤ 0.5 points |

**Whole Stage 1** vs the original model, in 1C's image mode:

```
--config configs/stage1/1c_joint.toml --checkpoint deepseek-ai/DeepSeek-OCR-2 --checkpoint outputs/runs/stage1c_joint/merged
```

Use `--merge <checkpoint>` to turn the best checkpoint (by dev CER in the report) into the next sub-stage's start:

```
python -m training --config configs/stage1/1a_printed.toml --merge outputs/runs/stage1a_printed/checkpoint-3000
```

Then point the next config's `[model] name_or_path` at `checkpoint-3000/merged`.

## 7. Follow-up experiments (only after the first baseline)

Run them one at a time, each against the same eval suite, so you can tell where an improvement comes from.

| Experiment | Change in config | When |
|---|---|---|
| Decoder MoE expert LoRA | add `mlp\.experts\.\d+\.(gate|up|down)_proj` to `target_modules` (~73M more params at r16; check with `training.inventory`) | after the 1A attention-only baseline |
| Unfreeze last 2–4 DeepEncoder V2 blocks | small learning rate, its own group | if handwriting plateaus in 1B |
| Visual tokenizer A / B / C | A: all frozen (default) · B: `net_2`/`net_3` trainable · C: last SAM layers selectively trainable | handwriting; B first. Never unfreeze the whole SAM |
| Resolution 1024 | `[image] image_size = 1024` | after 768 works |

The gradient fix in §1 already lets the vision path learn. Nothing else needs patching for these.

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Encoder LoRA silently not training | Smoke S6 checks every LoRA group gets gradients |
| Handwriting adapter damages printed reading | The 1B gate on printed CER. 1B is a separate adapter, so roll back to 1A merged |
| No English printed data in 1A | English is measured in every report. 1C's golden regions include printed English |
| Small crops lose detail when stretched to 768 | Compare with `image_size = 1024` on the dev suite |
| Golden regions overfit in 1C | 40% share (~2 repeats per epoch), early stop by dev CER |
| Public label noise | Spot-check 100 samples per dataset at conversion; drop any set with >5% bad labels |

## 9. Code

| Need | Where |
|---|---|
| OCR-2 preprocessing and token layout (shared by training and evaluation) | `models/deepseek_ocr/inputs.py` |
| Loading, OCR-2 check, vision-gradient fix | `models/deepseek_ocr/loader.py` |
| LoRA groups (rank/alpha per group) and per-group optimizer | `training/lora.py` |
| Module inventory (params and trainable per component) | `training/inventory.py` |
| Training, `--smoke`, `--merge` | `training/__main__.py`, `training/smoke.py` |
| Data mixture and curriculum | `training/data.py` |
| Before/after evaluation, report, gates | `evaluation/__main__.py`, `evaluation/report.py` |
| Converters (golden regions; official archives) | `dataEngine/sources/` |
| Run configs | `configs/stage1/1a_printed.toml`, `stage1b_handwritten.toml`, `stage1c_joint.toml` |
