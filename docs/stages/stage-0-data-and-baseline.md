# Stage 0 — Data and baseline

Turn the golden data into clean, split training records, prove the training setup works, and measure the untrained model. Every later number is compared against this baseline.

| | |
|---|---|
| Week | W1 (2026-11-23 → 11-29). Dataset access requests and downloads start this week |
| Input | `dataEngine/data/` golden pages, base checkpoints `deepseek-ai/DeepSeek-OCR` and `deepseek-ai/DeepSeek-OCR-2` |
| Output | Page records, locked splits, smoke-test report, `baseline.json`, the v1/v2 + resolution choice, gate values |
| Method | **No training** |
| Hardware | RTX 3060 (conversion, small smoke tests), A6000 (smoke tests at full size, baselines) |
| GPU-hours | 5–8 |

Part of the [training plan](../training-plan.md).

## 1. Steps

### 0.1 Inventory

Count the following, across all ~3K pages:
- documents, pages and regions
- per language
- per police station
- per `status`
- per `layout_label`

Then:
- List the Marathi pages and set them aside for V2.
- Write `outputs/stage0/inventory.json`.
- Show the counts to the user before splitting.

### 0.2 Convert golden pages into records

Write one JSON line per page with the fields below. Nothing else goes in: model names, token usage and prompts stay in the source folders as provenance.

| Field | From | Notes |
|---|---|---|
| `page_id` | `page.json` | e.g. `Arariya_SC_ST_PS_07_18/page_0001` |
| `document_id` | `page.json` | |
| `police_station` | parsed from `document_id` | Pattern `<District>_<Station>_PS_<case>_<yy>`. **Check the pattern on all IDs**; ask before guessing for any that do not match |
| `doc_type` | page-level human tag | `handwritten_letter` / `filled_form` / `other` (used for split stratification) |
| `image` | `png/page.png` | Path **relative** to the data root. The stored paths are absolute Linux paths and must be rebased |
| `width`, `height` | `page.json` | pixels |
| `blocks[]` | `source_language_extraction/page.json` + `eng_translation/page.json`, joined on `region_id` | sorted by `reading_order` |
| `blocks[].bbox_px` | `bbox_xyxy_px` | |
| `blocks[].bbox_999` | `round(x / width * 999)`, `round(y / height * 999)` | model coordinates |
| `blocks[].label` | `layout_label` | IndicDLP 42 classes |
| `blocks[].source` | `source_text`, NFC | keeps `[illegible]` / `[uncertain: …]` markers |
| `blocks[].english` | `english_translation`, NFC | |
| `blocks[].status` | `status` | ok / partial / illegible / no_text |
| `blocks[].languages` | `source_languages` | |
| `blocks[].crop` | `regions.json` → `ocr_crop.image_path` (basename) | Gives `rNNN.ocr.png` for rotated regions, `rNNN.png` otherwise |
| `blocks[].script` | dev/test only, see 0.4 | printed / handwritten / mixed |

**Never use `image/page_regions.png` or `image/preview.jpg` as model input.** `page_regions.png` has the answer boxes drawn on it.

### 0.3 Validate (fail loudly, report counts)

**Files and geometry:**
- The image exists and its SHA-256 equals `image_sha256`.
- Every box lies inside the page and has positive area.
- `reading_order` runs 1…n with no gaps or duplicates.
- Every crop file exists.

**Labels and status:**
- Every `label` is in the IndicDLP 42-class list.
- `status = ok` ⇒ `source` is non-empty.
- `source` is non-empty ⇒ `english` is non-empty.
- `[uncertain: …]` and `[illegible]` markers are well-formed.

**Encoding:**
- Text is valid UTF-8 and NFC.

**Confirm with the user:** whether the human check covered boxes and labels as well as text. 86 of the 134 sample regions carry `classification_low_confidence`.

### 0.4 Tags

**Page level, all pages:** tag `doc_type`. A human does this in about 1–2 hours, since letters and forms are easy to tell apart.

**Region level, dev + test only:** tag `script` = printed / handwritten / mixed.
1. Pre-label with a vision model.
2. A human checks 10% of the tags.
3. If spot-check accuracy is below 95%, a human checks all of them.

Training does not need region tags. Only the per-script metrics do.

### 0.5 Split (then lock)

1. **Group = police station.** Assign whole stations to test / dev / train, aiming for 15 / 15 / 70% of pages and stratifying by `doc_type`.
2. If one station holds more than 15% of the pages, split it by `document_id` instead and note this in the manifest.
3. English-only pages follow their document.
4. The split is deterministic: sort the stations, use a fixed seed, and record the seed in the manifest.
5. Write `splits/{train,dev,test}.txt` (page ids), plus the SHA-256 of each list.

**From here on, the test list is read only by the M1/M2 report runs.**

### 0.6 Environment

On the A6000:
- Python 3.12.9, torch 2.6.0 (CUDA 11.8), transformers 4.46.3, flash-attn 2.7.3, PEFT 0.13.2, accelerate 1.1.1.
- Install with `pip install -e .[train]` (versions pinned in `pyproject.toml`).
- Record the exact versions in `outputs/stage0/env.txt`.
- Do not upgrade transformers. The model ships its own code (`trust_remote_code=True`), and Unsloth's notes warn that the latest transformers breaks it.

### 0.7 Smoke tests (all must pass before Stage 1)

| # | Test | Pass condition |
|---|---|---|
| S1 | Load v1 and v2 | No custom-code errors |
| S2 | Module names | `model.named_parameters()` contains every name in the [freeze matrix](../training-plan.md#6-lora-or-full-fine-tuning-what-is-frozen). Save the list |
| S3 | Freeze check (Stage 1 setup) | Trainable = LoRA only: 3,978,240 parameters at r16. Nothing else has `requires_grad` |
| S4 | Label masking | Decoding the tokens whose label ≠ -100 gives exactly the target text + end token |
| S5 | Forward | Finite loss on 1 crop, 1 full page and 1 text-only sample (via the model's no-image path; confirm how its code handles missing images) |
| S6 | Backward | Non-zero gradients on LoRA, none anywhere else |
| S7 | Overfit | 8 samples × ~50 steps: loss falls close to 0 |
| S8 | Save/reload | The reloaded checkpoint gives the same loss on the same batch |
| S9 | Speed + memory | tokens/s and peak GPU memory for: crop micro-batch 16, page micro-batch 1, text micro-batch 16. **Recompute every stage's GPU-hours from these numbers** |
| S10 | ROCm (MI300X) | S1, S5, S6, S8 for the text-only path. Pass ⇒ Stage 3 runs on the MI300X |

### 0.8 Baseline (zero-shot, no training)

| Run | Data | Prompt | Measures |
|---|---|---|---|
| B1 | dev region crops | `<image>\nFree OCR.` | CER/WER per script |
| B2 | dev pages, v1 Base vs Gundam, v2 default | `<image>\n<|grounding|>Convert the document to markdown.` | page CER, block F1@0.5 (**class-agnostic**: the base model uses its own labels), Kendall τ, parse rate, seconds/page |
| B3 | Mozhi Hindi test lines | Free OCR | CER |
| B4 | dev hi→en region pairs + FLORES-200 devtest hi→en | IndicTrans2 (indic-en 1B) | chrF++, COMET, number preservation (the translation bar for Stage 3) |

**Decision rule for model and resolution** (dev only):
1. Lowest handwritten-Hindi CER.
2. If tied: better Kendall τ.
3. If still tied: v1, because more community fine-tunes exist.

For the M1/M2 comparison, also run the chosen configuration once on **test**. That is a fixed reference and does not tune anything.

### 0.9 Set gates

Turn the proposed gates in [training-plan.md §9](../training-plan.md#9-evaluation-and-gates) into numbers using the baseline. Record them in the Decisions table (`.claude/skills/ocr-pipeline/SKILL.md` §8).

## 2. Exit gate

- [ ] Inventory shown to the user. Marathi pages set aside.
- [ ] All validation checks pass, or every failure is listed and accepted by the user.
- [ ] Splits written, hashed and locked. No police station appears in two splits.
- [ ] Smoke tests S1–S9 green (S10 decides where Stage 3 runs).
- [ ] `baseline.json` written. Model version, resolution mode and gate values recorded in Decisions.

## 3. Risks and fallbacks

| Risk | Fallback |
|---|---|
| `document_id` does not encode the police station for some files | Ask the user for a mapping. Do not guess |
| v2 custom code fails under the pinned stack | Use v1. The plan is identical except for the vision module names |
| HF Trainer + PEFT is too slow | Try Unsloth's upload (decoder LoRA only) on the same data |
| Measured speed is far below the estimates | Shrink the Stage 1 epoch size first (it is the largest), then follow the cut list |

## 4. Code this stage needs (owner folder per SKILL.md)

- **`dataEngine`:**
  - golden → page records converter
  - validator
  - splitter
  - **one** `normalize_text` (NFC) used everywhere
  - **one** Document IR `serialize` / `parse` pair, used by data prep, evaluation and the PDF renderer
- **`evaluation`:**
  - CER/WER (raw + NFC)
  - block F1@IoU
  - Kendall τ
  - parse rate
  - chrF++ / COMET wrapper
  - number-preservation check
  - baseline runner
- **`training`:** model loader + freeze/LoRA setup (shared by all stages), smoke-test mode.
- **`configs`:** `stage0_baseline`.
