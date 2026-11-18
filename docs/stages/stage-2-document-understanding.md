# Stage 2 — FIR document understanding

Teach the model to read **whole FIR pages**. That means:
- finding every block
- giving each block its IndicDLP label and box
- putting the blocks in reading order
- writing each block's text

The result is the **M1 checkpoint**: a Hindi FIR reader.

| | |
|---|---|
| Week | W3 (2026-12-07 → 12-13). M1 report in W4 |
| Input checkpoint | `stage1` merged (or `stage1b` if it won) |
| Output checkpoint | `stage2` merged + `stage2_lora` |
| Method | **Fresh LoRA r16 on the decoder.** Vision encoder and `projector` frozen |
| Hardware | A6000 |
| GPU-hours | 15–30 |

Part of the [training plan](../training-plan.md).

## 1. Why this stage exists

Stage 1 taught the model to read crops. Real use is a full page: printed form labels next to handwritten fill-ins, rotated margin notes, stamps, tables. This stage teaches layout and reading order on your human-checked pages. Replay keeps the reading skill from Stage 1.

## 2. Data

**One epoch = 15K samples**, drawn by these shares. Train for 3 epochs.

| Share | Task | Source | Priority |
|---|---|---|---|
| 40% | Full-page grounded parse | Golden train pages (~2.1K, ≈3 views each per epoch) | P1 |
| 10% | Read the text in a given box | Golden train pages, one random block per sample | P1 |
| 15% | Layout only (boxes + labels, no text) | IndicDLP Hindi + English pages (P2) + golden train pages. If IndicDLP access is late: golden pages only | P1/P2 |
| 5% | Page-level free OCR (plain text in reading order) | ICDAR 2024 HWD Hindi pages (P2). Otherwise golden pages as plain text | P1/P2 |
| 30% | Stage 1 replay (crops) | Same sources and shares as Stage 1 | P1 |

**Rules:**
- Golden pages come from the Stage 0 records and train split only.
- Rotated regions keep their **page** box. Their text is the reading of the rotated crop.
- `no_text` blocks (stamps, signatures) appear as a header only.
- `partial` / `illegible` text keeps its markers, so the model learns to flag what it cannot read rather than guess.
- **IndicDLP needs no label mapping**: same 42 classes. Its block order may not match human reading order. Check this at conversion. If it has no reading order, sort top-to-bottom, left-to-right, and keep it on the *layout-only* prompt so it does not teach the parse prompt a different order.

## 3. Task format

The output format is defined once in [training-plan.md §7](../training-plan.md#7-model-input-and-output-format).

| Task | Prompt | Target |
|---|---|---|
| Page parse | `<image>\n<|grounding|>Convert the document to markdown.` | source-only blocks in reading order |
| Read box | `<image>\n<|grounding|>Read the text in <|det|>[[x1, y1, x2, y2]]<|/det|>.` | that block's text |
| Layout only | `<image>\n<|grounding|>Detect the layout.` | block header lines only |
| Page free OCR | `<image>\nFree OCR.` | page text in reading order |
| Replay | as in Stage 1 | as in Stage 1 |

**Example page-parse target** (from `Arariya_SC_ST_PS_07_18/page_0001`, shortened):

```
<|ref|>header<|/ref|><|det|>[[190, 19, 558, 94]]<|/det|>
सेवा में,
थाना अध्यक्ष,
अररिया।

<|ref|>section-title<|/ref|><|det|>[[153, 85, 770, 157]]<|/det|>
विषय :- मारपीट गाली गलौज घड़ी साइकिल एवं रूपया
छीन लेने के सम्बंध में।
```

**Page images:** use the resolution mode chosen in Stage 0 (v1 Gundam or Base, v2 default).

**Output budget:** max 4,096 target tokens. The longest sample page needed 1,406 tokens source-only (see [tokenizer-check](../tokenizer-check.md#3-output-length-on-your-real-pages)). Any page above 4,096 is logged and skipped, never silently cut.

## 4. What trains and what is frozen

Stage 1's adapter is already merged into `stage1`. This stage adds a **fresh** adapter on the same modules.

| Module | State |
|---|---|
| `projector` | frozen (trained only if Stage 1b ran and was merged) |
| `layers.0-11.self_attn.{q,k,v,o}_proj` | **LoRA r16** (new) |
| `layers.0.mlp.{gate,up,down}_proj` | **LoRA r16** (new) |
| `layers.1-11.mlp.shared_experts.{gate,up,down}_proj` | **LoRA r16** (new) |
| `sam_model`, `vision_model` / `qwen2_model` (incl. Stage 1b blocks if merged) | frozen |
| routed experts, router, `embed_tokens`, `lm_head`, `image_newline`, `view_seperator` | frozen |

**Why LoRA:** only ~2.1K training pages, which is the classic small-data case. Full fine-tuning would memorize the pages and wreck the merge with Stage 3.

## 5. Settings (starting values)

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate, LoRA | 5e-5 |
| Optimizer / schedule | AdamW wd 0.01, cosine, 3% warmup, gradient clipping 1.0 |
| Precision | bf16, gradient checkpointing on |
| Batch | micro-batch 1 × accumulation 8 = 8 samples, grouped by length |
| Epochs | 3 × 15K samples, early stop on dev |
| Evaluation | every 500 steps on all dev pages. Keep the best by `page CER + (1 − block F1)` |

## 6. Exit gate

Values are proposed; final values are set in Stage 0.

| Check (dev) | Pass |
|---|---|
| Block F1 @ IoU 0.5 (class-aware) | ≥ 0.85 |
| Reading order, Kendall τ | ≥ 0.90 |
| Page CER, Hindi handwritten and printed | below baseline and below the Stage 1 checkpoint run on pages |
| Crop CER on Stage 1 dev crops | at most 0.5 points worse than Stage 1 |
| Output parses (format + UTF-8) | ≥ 98% |
| Truncated outputs (hit 4,096 tokens) | 0 |

## 7. M1 report (W4, 2026-12-14 → 12-20)

Run `stage2` **once** on the locked test set, next to the Stage 0 baseline-on-test.

**Numbers:**
- page CER/WER for Hindi and English, printed and handwritten
- block F1
- Kendall τ
- parse rate
- seconds per page
- the Stage 3 translation results, if ready

**Pictures:** 10 test pages with predicted boxes drawn and text side by side with the truth. Pick 5 good and 5 bad, chosen by a fixed rule (e.g. best and worst CER) so they are not cherry-picked.

**Text:** what fails most (by layout label and by script), and what Stage 4 will change.

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Dense handwriting unreadable at tile resolution | Compare Gundam with more tiles vs Large 1280 on 50 dev pages. The vision tokens still fit the budget |
| Complex form reading order | OCR-2's encoder was built for reading order; Stage 0 may already pick it. Add more page-parse share (layout-only down) |
| Layout labels noisy (if the human check did not cover them) | Report class-agnostic block F1 next to class-aware. Do not gate on label accuracy until labels are confirmed |
| IndicDLP access late | Layout-only task uses golden pages only. Shares unchanged |

## 9. Code this stage needs

- **`dataEngine`:**
  - **one function per task** that turns a Stage 0 page record into (prompt, target), using the shared Document IR serializer
  - IndicDLP converter into the same page-record format
- **`training`:** same loader and entry point. Config `stage2`.
- **`evaluation`:**
  - block F1 and Kendall τ, using the shared Document IR **parser** (the same one the PDF renderer will use)
  - M1 report runner (metrics + side-by-side images)
