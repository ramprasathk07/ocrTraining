# Stage 1 — Devanagari script adaptation (printed + handwritten Hindi)

Teach the model to **read Hindi characters**: matras, conjuncts, nukta, digits, and above all handwriting. The training images are crops (words, lines, regions), not full pages, so the model focuses on reading rather than layout.

| | |
|---|---|
| Week | W2 (2026-11-30 → 12-06). Public data is converted during W1 |
| Input checkpoint | Base model chosen in Stage 0 (v1 or OCR-2) |
| Output checkpoint | `stage1` (merged) + `stage1_lora` (adapter) |
| Method | **LoRA r16 on the decoder.** Vision encoder and `projector` frozen |
| Hardware | A6000 |
| GPU-hours | 15–30 (recompute from Stage 0 speed) |

Part of the [training plan](../training-plan.md).

## 1. Why this stage exists

The baseline DeepSeek-OCR barely saw handwritten Devanagari. Your 2.1K training pages alone cannot teach glyph shapes reliably. Public word and line datasets give tens of thousands of writers and words. Your region crops anchor that learning to FIR handwriting.

## 2. Data

**One epoch = 250K samples**, drawn by these shares (shares are probabilities, not counts). Train for 2 epochs.

| Share | Source | Kind | Level | Priority |
|---|---|---|---|---|
| 50% | **Handwritten Hindi**: IIIT-HW-Dev (same data as the IIIT-INDIC-HW-WORDS Hindi part), DohaScript; + ICDAR 2024 HWD crops, IIIT-Indic-HW-UC Hindi if P2 data is ready | handwritten | words, lines | P1 (+P2) |
| 25% | **Printed Hindi**: Mozhi Hindi words + lines (train split only) | printed | words, lines | P1 |
| 25% | **Golden FIR region crops**, train split only, all languages in their natural mix (Hindi + English) | handwritten + printed | regions | P1 |

**Rules:**
- **Trusted sources only.** IIIT-HW-Dev and Mozhi come from the official CVIT / NLTM `ilocr.iiit.ac.in` downloads, and DohaScript from its authors. Never use third-party re-uploads. Record the URL, licence and SHA-256 of each archive at download.
- Only train splits. Dev and test crops from any dataset never enter training.
- Golden crops are ~21K regions (2.1K pages × ~10), so 25% of 250K ≈ 3 repeats per epoch. Do not raise this share; the risk is memorizing them.
- Public datasets keep their official splits. Their test sets are used only for evaluation.
- **Curriculum (epoch 1):**
  1. First third of the steps: word crops only.
  2. Second third: words + lines.
  3. Last third: everything.
- **Epoch 2:** everything, mixed by the shares.
- Text is NFC-normalized ([tokenizer rule 1](../tokenizer-check.md#4-verdict-and-rules)).

## 3. Task format

| Prompt | Target |
|---|---|
| `<image>\nFree OCR.` | the exact text of the crop: original line breaks, `[illegible]` / `[uncertain: …]` markers kept, then end token |

**Images:** use the model's own preprocessing for the chosen mode. It is implemented once in `training/inputs.py` and shared by training and evaluation.
- v1 Small (`base_size = image_size = 640`, `crop_mode = false`) **stretches** the crop to 640×640: 100 vision tokens + 11 row-end tokens = 111 positions.
- v1 Gundam (`crop_mode = true`) **pads** a small crop onto a 1024 canvas: 273 positions.
- Stage 0 picks the mode from measured speed and dev CER. The config holds the three values.

**Loss:** only on target tokens. Prompt and image positions get label -100.

## 4. What trains and what is frozen

| Module | State | Parameters |
|---|---|---|
| `projector` | frozen (the official code runs it without gradients) | — |
| `layers.0-11.self_attn.{q,k,v,o}_proj` | **LoRA r16** | 1,966,080 |
| `layers.0.mlp.{gate,up,down}_proj` (dense) | **LoRA r16** | 390,144 |
| `layers.1-11.mlp.shared_experts.{gate,up,down}_proj` | **LoRA r16** | 1,622,016 |
| `sam_model` (incl. `net_2`, `net_3`) | frozen | |
| `vision_model` (v1) / `qwen2_model` (v2) | frozen | |
| `layers.1-11.mlp.experts.*` (routed), `layers.1-11.mlp.gate` (router) | frozen | |
| `embed_tokens`, `lm_head`, `image_newline`, `view_seperator` | frozen | |

Smoke test S3 in Stage 0 confirms these counts.

**Why LoRA here and not full fine-tuning:**
- 250K crops would be enough data to move more weights. But Stage 2 and Stage 4 must still merge cleanly, and the base model's general reading skill must survive.
- LoRA gives both.
- If the model underfits (dev CER stops improving early while train loss stays high), add LoRA r16 on the routed experts `mlp.experts.*.{gate,up,down}_proj` (+≈73.5M parameters) **before** touching the vision encoder.

## 5. Settings (starting values)

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate, LoRA | 1e-4 |
| Optimizer | AdamW, weight decay 0.01 |
| Schedule | cosine, 3% warmup |
| Gradient clipping | 1.0 |
| Precision | bf16, gradient checkpointing on |
| Batch | micro-batch 16 × accumulation 4 = 64 crops (lower micro-batch if memory requires; keep 64 effective) |
| Max target length | 1,536 tokens (a whole handwritten letter region can reach ~1,100 tokens); group batches by length |
| Epochs | 2 × 250K samples, early stop on dev |
| Evaluation | dev loss every 1,000 steps during training. CER on 2K dev region crops + Mozhi Hindi val via `python -m evaluation` on each saved checkpoint. Keep the best checkpoint by **handwritten-Hindi dev CER** |
| Seed | fixed, recorded in the run manifest |

## 6. Exit gate

Values are proposed. Final values are set in Stage 0 and recorded in Decisions.

| Check | Pass |
|---|---|
| Handwritten-Hindi dev region CER | ≥30% relative drop vs baseline |
| Printed-Hindi dev region CER | below baseline |
| Mozhi Hindi line CER (official test) | ≤3% |
| English dev region CER | at most 0.5 points worse than baseline |
| Output validity | 100% valid UTF-8 |

**If the handwriting check fails:**
1. Sample 200 errors and sort them into two groups:
   - *visual* — wrong glyph read: confused matras, conjuncts, digits.
   - *language* — right glyphs, wrong word or spelling.
2. Mostly language errors → more epochs or the routed-expert LoRA.
3. Mostly visual errors → **Stage 1b**.

## 7. Stage 1b — conditional visual unfreeze

Run **only** if the gate fails because of visual errors. It uses the W8 buffer week; cut-list item 1 drops it first.

| | |
|---|---|
| Input | `stage1` merged |
| Adds | **full training** of the last 4 vision blocks: v1 `vision_model.transformer.layers.20-23` (CLIP-L, ≈12M each, ≈48M total); v2 the last 4 layers of `qwen2_model` (confirm the exact path from S2). Also `projector` |
| Needs | A patched model `forward` that keeps `torch.no_grad()` around `sam_model` but lets gradients reach the last 4 vision blocks and `projector` |
| Keeps | everything else as in Stage 1. `sam_model` stays frozen |
| Learning rates | vision blocks 1e-5, `projector` 2e-5, fresh LoRA r16 at 5e-5 |
| Data / length | same mixture, 1 epoch |
| Output | `stage1b`, compared to `stage1` on the same dev set. Keep whichever wins |

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Tiny word crops look nothing like FIR pages | Golden region crops (25%) and line data keep it realistic. Stage 2 trains on full pages |
| Public label noise | Spot-check 100 samples per dataset at conversion. Drop any dataset with >5% bad labels |
| English reading gets worse | Golden crops include English regions. If the English gate fails, double the English share inside the golden 25% |
| Overfitting golden crops | Early stop on dev. The golden share stays at 25% |

## 9. Code (built 2026-11-18)

| Need | Where |
|---|---|
| Golden pages → validated records, locked splits | `dataEngine/golden.py`, `dataEngine/splits.py` |
| One crop record format for every source | `CropRecord` in `dataEngine/records.py` |
| Golden region crops (from page records, never the raw folders) | `dataEngine/sources/golden_regions.py` |
| Official public archives (images + label file per split), set up through config | `dataEngine/sources/labelled_folder.py` |
| `crop_ocr` task and its prompt | `dataEngine/tasks.py` |
| The model's own preprocessing and token layout, shared by training and evaluation | `models/deepseek_ocr/inputs.py` |
| Shares, curriculum and epochs | `training/data.py` |
| LoRA, Trainer, 1→N GPUs | `training/__main__.py` |
| Smoke tests S3–S9 | `training/smoke.py` |
| CER/WER and sharded evaluation | `evaluation/` |
| Run config | `configs/stage1_script.toml` |

**Mixing:** the plan said `datasets.interleave_datasets`. The code instead builds the whole sample plan once from the seed, and each GPU reads its own slice. That keeps 1 GPU and N GPUs on identical data, which a per-rank interleave cannot guarantee.
