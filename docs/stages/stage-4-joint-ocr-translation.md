# Stage 4 — Joint OCR + translation

Put reading (Stages 1–2) and translation (Stage 3) into **one model** that takes a page image and writes **grounded bilingual blocks**: box, label, Hindi or English source, and the English translation. The output is `hindidoc-v0.1`, the M2 model.

| | |
|---|---|
| Week | W5–W6 (2026-12-21 → 2027-01-03) |
| Input | `stage2` merged (= W0 + ΔA) and `lang_adapter` (ΔB) from Stage 3, if it passed its gate |
| Output checkpoint | `hindidoc-v0.1` merged + its adapter |
| Method | Short integration experiments, then **fresh LoRA r32 on the decoder**. Vision encoder and `projector` frozen |
| Hardware | A6000 |
| GPU-hours | 15–30 (3 short runs + 1 full run) |

Part of the [training plan](../training-plan.md).

## 1. Why this stage exists

The product needs reading and translating in one pass, block by block, so the PDF renderer can put each English block where the Hindi one was. This stage also answers the research question behind your idea: **does a separately trained language branch (ΔB) make the joint model translate better, without making it read worse?**

## 2. Step 1 — Integration experiments (W5)

All runs use the same data, settings and steps. Only the starting weights differ.

| Run | Starting weights | Meaning |
|---|---|---|
| **J0** (control) | `stage2` | No language branch. Translation is learned only from this stage's data, including text-only translation replay |
| **J1-a** | `stage2` + 0.5·ΔB | Language branch merged at half strength |
| **J1-b** | `stage2` + 1.0·ΔB | Language branch merged at full strength |

**Choosing λ:**
- The λ values come from the Stage 3 diagnostic.
- If ΔB at λ=1 raised crop CER by more than 1 point, use λ = 0.25 and 0.5 instead.
- The merge works because ΔA and ΔB are both low-rank changes **to the same weight matrices, measured from the same base W0**, so they add.

**Not a separate experiment:** keeping ΔB as a separate frozen adapter. During training and inference that is **mathematically identical** to J1-b (merge at λ=1), so it would only add machinery.

**Run length:** each short run is one 5K-sample epoch of the mixture below, evaluated on 200 dev pages.

**Winner rule** (dev):
1. Highest translation chrF++ on bilingual blocks, among runs whose page CER is at most 0.5 points worse than `stage2`.
2. If the difference is under 1 chrF++, take **J0**, because it is simpler.

If Stage 3 failed its gate or was cut, skip Step 1 and run J0 directly.

## 3. Step 2 — Full joint run on the winner (W6)

### Data

**One epoch = 15K samples**, drawn by these shares. Train for 2 epochs.

| Share | Task | Source |
|---|---|---|
| 45% | Bilingual page parse | Golden train pages |
| 15% | Source-only page parse | Golden train pages, so the source-only mode keeps working |
| 15% | Bilingual crop | Golden train region crops |
| 15% | Text-only translation replay | Stage 3 data: BPCC hi↔en, golden pairs, FIR terms. Keeps translation from fading while image tasks dominate |
| 10% | Crop OCR replay | Stage 1 mixture |

### Task format

The format is defined once in [training-plan.md §7](../training-plan.md#7-model-input-and-output-format).

| Task | Prompt | Target |
|---|---|---|
| Bilingual page | `<image>\n<|grounding|>Convert the document to markdown and translate it to English.` | blocks with `[EN]` sections |
| Source-only page | `<image>\n<|grounding|>Convert the document to markdown.` | blocks, no `[EN]` |
| Bilingual crop | `<image>\nFree OCR and translate to English.` | source `\n[EN]\n` English |
| Text translation / crop OCR | as in Stages 3 / 1 | as in Stages 3 / 1 |

**English regions:** their `[EN]` part repeats the source. That is how the data already handles English and how the renderer expects it.

**Measured budget:** bilingual pages need 1,420 tokens on average and at most 2,145. The cap stays at 4,096.

### What trains and what is frozen

| Module | State |
|---|---|
| `projector` | frozen |
| `layers.0-11.self_attn.{q,k,v,o}_proj` | **LoRA r32** (new) |
| `layers.0.mlp.{gate,up,down}_proj` | **LoRA r32** (new) |
| `layers.1-11.mlp.shared_experts.{gate,up,down}_proj` | **LoRA r32** (new) |
| `sam_model`, `vision_model` / `qwen2_model` | frozen |
| routed experts, router, `embed_tokens`, `lm_head`, `image_newline`, `view_seperator` | frozen |

**Why LoRA (not full fine-tuning):**
- ~2.1K training pages is still small data.
- A fresh adapter on the merged weights also means a bad run can be thrown away without touching `stage2`.
- The rank goes up to 32 because this stage learns two output languages at once.

### Settings (starting values)

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Learning rate, LoRA | 5e-5 |
| Optimizer / schedule | AdamW wd 0.01, cosine, 3% warmup, gradient clipping 1.0 |
| Precision | bf16, gradient checkpointing on |
| Batch | micro-batch 1 × accumulation 8, grouped by length |
| Max target length | 4,096 |
| Epochs | 2 × 15K samples, early stop on dev |
| Evaluation | every 500 steps on all dev pages |

## 4. Exit gate

Values are proposed; final values are set in Stage 0.

| Check (dev) | Pass |
|---|---|
| Page CER (source text), Hindi handwritten + printed | at most 0.5 points worse than `stage2` |
| Block F1 @ 0.5 / Kendall τ | at most 0.02 below `stage2` / ≥ 0.90 |
| Output parses (format + UTF-8) | ≥ 99% |
| Block recall / block precision ("no invented blocks") | ≥ 0.98 / ≥ 0.98 |
| Every block with source text has an `[EN]` part | 100% |
| Number/ID preservation (source → English) | ≥ 99% |
| `[illegible]` / `[uncertain]` markers carried into English | same count on both sides in ≥ 99% of blocks |
| Translation chrF++ on bilingual dev blocks | ≥ the **two-model pipeline** (`stage2` reads → IndicTrans2 translates) − 3 |
| Human review | 50 random dev blocks checked for names, dates, sections, amounts. Report the error count |

The two-model pipeline is the fair competitor. One model must come close to it to justify the single-model design.

## 5. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Translation training makes reading worse | The OCR gate catches it. Choose a lower λ or J0, and raise the crop-replay share |
| The model translates text it marked illegible (invents content) | Marker-count gate. Add more `partial`/`illegible` pages to the mixture |
| English blocks get "translated" into odd English | Check English-source blocks separately in the dev report |
| Holidays (W5–W6) | Start Step 1 before the break. All runs are unattended jobs |
| The joint model clearly loses to the two-model pipeline | M2 ships the pipeline as the fallback product path and reports the gap honestly. The single-model work continues in V2 |

## 6. Code this stage needs

- **`dataEngine`:** task builders for bilingual page and bilingual crop, using the shared Document IR serializer.
- **`training`:**
  - one small merge function (scaled add of a LoRA delta: `W += λ·B·A·α/r`) — or PEFT's weighted-adapter helpers if they cover a single adapter cleanly
  - same entry point. Configs `stage4` with `lambda` as a value; J0 = 0
- **`evaluation`:**
  - marker-count check, block precision/recall
  - pipeline baseline runner (`stage2` + IndicTrans2)
  - all other metrics from Stage 0
