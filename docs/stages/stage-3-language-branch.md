# Stage 3 — Language branch (text-only Hindi + translation)

This is your idea from the ChatGPT discussion: train **a copy of the same decoder, with the same tokenizer, on text only**, so it gets better at Hindi and at Hindi→English translation. Then connect it back to the OCR model in Stage 4. No images are used here.

| | |
|---|---|
| Week | W2–W4 (2026-11-30 → 12-20), **in parallel** with Stages 1–2 |
| Input checkpoint | **The base model chosen in Stage 0** (W0). Not `stage1` / `stage2`; see section 4 |
| Output | `lang_adapter`: a LoRA adapter **ΔB**, not merged into anything yet |
| Method | **LoRA r32 on the decoder only.** Vision modules and `projector` unused and frozen |
| Hardware | MI300X if smoke test S10 passed. Otherwise the A6000 in W4, after Stage 2 finishes |
| GPU-hours | 10–25 |

Part of the [training plan](../training-plan.md).

## 1. Why this stage exists

The OCR decoder learned to *copy* text from images. It was not built to translate. Text-only data is huge and cheap: millions of Hindi sentences and Hindi↔English pairs. Training on it without images is far faster than multimodal training. The open question, answered in Stage 4, is whether this skill transfers into the OCR model **without hurting reading**.

## 2. Data

**Budget: 50–100M tokens, one pass.** Final size is set from the Stage 0 speed measurement.

| Share | Task | Source | Priority |
|---|---|---|---|
| 35% | Hindi language modelling | Sangraha Hindi (verified portion), chunks of up to 2,048 tokens | P1 |
| 40% | Translation: ~90% hi→en, ~10% en→hi | BPCC hi–en. Human-made pairs (BPCC-H) first, then filtered mined pairs | P1 |
| 15% | FIR translation pairs | Golden **train split** regions where the source is Hindi: `source` → `english` | P1 |
| 5% | FIR terms | Short pairs mined from the golden train pairs: legal terms ("धारा", "भा. द. वि." → "I.P.C."), police ranks, form labels | P1 |
| 5% | English language modelling | English side of BPCC as plain text (keeps English fluent) | P1 |

**Filters for BPCC mined pairs:**
- exact duplicates removed
- length ratio between 0.5 and 2.0
- script check (Devanagari source, Latin target)
- **drop pairs whose digit sequences differ between source and target**, so the data itself teaches number preservation

**Other rules:**
- Golden pairs keep their `[illegible]` / `[uncertain: …]` markers on both sides, so the model learns to carry them into the English.
- Only golden **train** pages are used. Dev pairs are for evaluation.
- Text is NFC-normalized.

## 3. Task format

| Task | Input (prompt) | Target (loss here) |
|---|---|---|
| Hindi → English | `Translate this Hindi text to English.\n{hindi}` | English text + end token |
| English → Hindi | `Translate this English text to Hindi.\n{english}` | Hindi text + end token |
| Language modelling | (none) | the whole chunk |

These are the same prompts Stage 4 replays, defined once in config.

The model runs **without an image** through the same model class. Smoke test S5 confirmed how its code handles a missing image.

## 4. What trains and what is frozen

| Module | State |
|---|---|
| `layers.0-11.self_attn.{q,k,v,o}_proj` | **LoRA r32** |
| `layers.0.mlp.{gate,up,down}_proj` | **LoRA r32** |
| `layers.1-11.mlp.shared_experts.{gate,up,down}_proj` | **LoRA r32** |
| `projector` | frozen (unused: no image) |
| `sam_model`, `qwen2_model` | frozen (unused) |
| routed experts, router, `embed_tokens`, `lm_head` | frozen |

**Why it starts from the base model, not from `stage2`:** Stages 1–2 change the decoder through ΔA (merged). Training ΔB on the untouched base W0 keeps both changes measured from the same starting point, so Stage 4 can add them: **W0 + ΔA + λ·ΔB**. Starting from `stage2` would also work mathematically, but then this branch could not run in parallel with Stages 1–2. Parallel running is what makes the 1-month M1 possible.

**Why the same target modules as Stages 1–2:** ΔB must be addable to the same weight matrices.

**Why LoRA and not full fine-tuning:**
- The decoder weights are shared with OCR. A full update would overwrite them, and there would be nothing to scale back if reading gets worse.
- LoRA gives a separate ΔB that Stage 4 can scale with λ or drop entirely.
- `embed_tokens` and `lm_head` stay frozen so the tokenizer–embedding link is identical in both branches.

## 5. Settings (starting values)

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Learning rate | 1e-4 (the chat's 2e-5 is a full-fine-tuning rate) |
| Optimizer / schedule | AdamW wd 0.01, cosine, 1% warmup, gradient clipping 1.0 |
| Sequence length | 2,048 |
| Batch | 64 sequences (MI300X: 16 × 4; A6000: 8 × 8), grouped by length, no packing |
| Precision | bf16. On ROCm use `attn_implementation = "eager"`; the model code supports only `eager` and `flash_attention_2` |
| Evaluation | every 1,000 steps: FLORES-200 **dev** hi→en (500 sentences) + golden dev pairs |

## 6. Exit gate

Values are proposed; final values are set in Stage 0.

| Check | Pass |
|---|---|
| FLORES-200 devtest hi→en chrF++ | ≥ IndicTrans2 (Stage 0 run B4) − 3 |
| Golden dev pairs chrF++ | above the base model's zero-shot text translation |
| Number/ID preservation, golden dev pairs | ≥ 99% |
| COMET (FLORES + golden dev) | reported, not gated |

**Diagnostic, not a gate:** add ΔB at λ = 1 to `stage2` and measure dev crop CER. This shows how much the language branch disturbs reading, and sets the λ values Stage 4 tries.

**If the gate fails:** ΔB is not used. Stage 4 runs as J0 only (translation data mixed in directly), so the M2 deadline still holds.

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| ROCm smoke test fails | Run on the A6000 in W4 (it is mostly free during the M1 report). If that slips too, apply cut-list item 4: no separate branch, the data goes into Stage 4 |
| The OCR decoder is weak at translation, even with LoRA | Stage 4 J0 still trains translation end-to-end. IndicTrans2 stays the external reference in the M2 report |
| Sangraha noise | Use the verified portion only |
| ΔB hurts reading too much (diagnostic) | Stage 4 tries small λ, or drops ΔB |

## 8. Code this stage needs

- **`dataEngine`:**
  - converters for Sangraha, BPCC (with the filters above) and golden pairs into **one text record format** (`prompt`, `target`, `task`, `source_dataset`)
  - the terms list comes from the golden pairs
- **`training`:** same entry point and loader, text-only path (no image). Config `stage3`.
- **`evaluation`:** chrF++ / COMET wrapper and the number-preservation check from Stage 0. No new metric code.
