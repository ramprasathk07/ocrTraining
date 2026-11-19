# Stage 5 — Release check and M2 report

Measure the final model **once** on the locked test set, check that it survives compression, fix its output contract for the PDF renderer, and write the M2 report. **No training.**

| | |
|---|---|
| Week | W7 (2027-01-04 → 01-10). M2 report in W8 (01-11 → 01-17), which is also the buffer week |
| Input | `hindidoc-v0.1`. For comparison: Stage 0 baseline, `stage2` (M1), the two-model pipeline (`stage2` + IndicTrans2) |
| Output | M2 report, quantization results, renderer contract + sample outputs, archived release bundle |
| Method | **None** (evaluation only) |
| Hardware | A6000 |
| GPU-hours | 3–6 |

Part of the [training plan](../training-plan.md).

## 1. Steps

### 5.1 Locked test run (once)

Run all four systems on the test split with the same prompts and settings:
1. baseline
2. `stage2`
3. pipeline
4. `hindidoc-v0.1`

Report every metric from [training-plan.md §9](../training-plan.md#9-evaluation-and-gates), split by:
- script type: printed / handwritten
- text language: Hindi / English
- `doc_type`: letter / form

Add 95% confidence intervals by bootstrapping over **police stations**, not pages, because pages from one station are not independent.

### 5.2 Error analysis

- Break down CER and block F1 by layout label, police station and `doc_type`.
- List the 20 most frequent character confusions (e.g. matra or nukta swaps).

These numbers drive the V1.1 / V2 priorities.

### 5.3 Privacy checks

| Check | How | Report |
|---|---|---|
| Overfitting gap | CER on 200 train pages vs test pages | the gap |
| Memorization probe | 100 text-only prompts made of the first half of a train-set sentence that contains a person's name. Does the model complete the rest exactly? | rate of exact completions |

Whatever the result:
- Checkpoints and data stay private (research / internal use).
- The report states the result.

### 5.4 Compression

Compare bf16 against 8-bit and 4-bit loading on 200 dev pages.

| Measure | Pass |
|---|---|
| Page CER change | at most +0.5 points |
| Translation chrF++ change | reported |
| Peak memory, seconds/page | reported |

The model uses custom code (MoE + custom vision). **Confirm that each quantization path actually loads and runs.** Do not assume it does.

Serving options to check, not assumed:
- the official vLLM support for DeepSeek-OCR, with our fine-tuned checkpoint
- GGUF / llama.cpp support

The edge target device and latency budget are still open. Ask for them at the start of W7.

### 5.5 Renderer contract

The model's text output is turned into JSON by the **same parser** used in evaluation (owned by `dataEngine`).

| Field | Meaning |
|---|---|
| `page_width`, `page_height` | pixels of the input image |
| `blocks[].index` | reading order (0-based, = output order) |
| `blocks[].label` | IndicDLP class |
| `blocks[].bbox_999` / `bbox_px` | model coordinates / converted with the page size |
| `blocks[].source` | source text, with markers |
| `blocks[].english` | translation, with markers |
| `blocks[].flags` | `illegible`, `uncertain`, `no_text` |
| `parse_ok` | false ⇒ the page goes to human review. **Never silently dropped** |

Hand the renderer 20 test-page JSONs, chosen by a fixed rule, plus this table.

### 5.6 M2 report (W8)

1. **Results table:** baseline → M1 → v0.1 → pipeline, all metrics, with confidence intervals.
2. **Gates:** pass/fail for each stage, with numbers.
3. **Samples:** 10 pages (5 best, 5 worst by CER) showing the image, predicted boxes, Hindi and English side by side.
4. **Research answer:** did the language branch (J1) beat J0? By how much, and at what OCR cost?
5. **Limits:**
   - Hindi + English only
   - Bihar FIR style
   - single pages
   - no key-value fields
   - research licence data
6. **Next steps:** V1.1 Marathi, V2 items from [training-plan.md §11](../training-plan.md#11-timeline-8-weeks-and-roadmap), ordered by the error analysis.

### 5.7 Archive the release bundle

Save the following together:
- merged checkpoint + adapters
- every stage config
- split files + their hashes
- `env.txt`
- data manifests
- the OCR-2 baseline report
- all evaluation outputs
- the M2 report

## 2. Exit gate

- [ ] Test numbers for all four systems, with confidence intervals.
- [ ] Quantized CER change within +0.5 points, or a documented reason.
- [ ] Privacy checks reported.
- [ ] Renderer contract + 20 sample JSONs delivered.
- [ ] M2 report delivered by **2027-01-17**.
- [ ] Release bundle archived.

## 3. Risks and fallbacks

| Risk | Fallback |
|---|---|
| A quantization path does not load the custom model | Report bf16 only. Quantization moves to V2 (cut-list item 5) |
| v0.1 loses to the pipeline | Report it honestly. The pipeline is the M2 product path, and the single model continues in V2 |
| Buffer week used up by follow-up experiments or reruns | Keep 5.1, 5.5 and 5.6. Shorten 5.2 to the top 10 errors |

## 4. Code this stage needs

- **`evaluation`:**
  - report runner (all systems, confidence intervals by station)
  - memorization probe
  - quantized-load benchmark
- **`dataEngine`:** the existing Document IR parser. Add JSON export only, no second parser.
