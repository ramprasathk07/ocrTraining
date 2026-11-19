# HindiDoc-OCR V1 — Training Plan

Fine-tune **DeepSeek-OCR (3B)** so that one model reads **Hindi + English** police documents — printed and handwritten — and returns **grounded bilingual blocks**. Each block has a box, a layout label, a reading order, the source text and an English translation. A separate, non-AI renderer turns those blocks into the translated PDF.

**Deadline**

| Milestone | Date | What it shows |
|---|---|---|
| **M1** | **2026-12-20** (1 month) | Hindi FIR reader |
| **M2** | **2027-01-17** (2 months, hard limit) | Hindi FIR reader + translator, final report |

- Written 2026-11-18.
- Status: plan approved, nothing trained yet.
- Source discussion: [ChatGPT export](ChatGPT-Training%20OCR%20Models%20Properly-20261118-2144.md). This plan checks it against the real model code, the real tokenizer and your real data, and corrects it where it was wrong (section 4).

Detail lives in the stage docs. This page summarizes and links.

| Doc | What it covers |
|---|---|
| [tokenizer-check.md](tokenizer-check.md) | Can DeepSeek's tokenizer handle Hindi and English? (Yes, with rules) |
| [stage-0-data-and-baseline.md](stages/stage-0-data-and-baseline.md) | Convert golden data, split it, measure the untrained model |
| [stage-1-script-adaptation.md](stages/stage-1-script-adaptation.md) | Teach the model to read Devanagari, printed and handwritten |
| [stage-2-document-understanding.md](stages/stage-2-document-understanding.md) | Teach whole FIR pages: boxes, labels, reading order, text |
| [stage-3-language-branch.md](stages/stage-3-language-branch.md) | Text-only Hindi + translation training of the decoder |
| [stage-4-joint-ocr-translation.md](stages/stage-4-joint-ocr-translation.md) | Combine reading and translation into one model |
| [stage-5-release.md](stages/stage-5-release.md) | Final test, compression, hand-off to the PDF renderer |

---

## 1. Decisions

| Decision | Answer | Source |
|---|---|---|
| Base model | **DeepSeek-OCR 2** (3B, Jan 2026, Apache-2.0), pinned revision `aaa02f38…`. Replaces the earlier v1-vs-OCR-2 baseline choice | You, 2026-11-19 |
| Stage 1 structure | **1A printed → 1B handwritten (separate adapter) → 1C joint** with the golden regions; each starts from the previous merged checkpoint | You, 2026-11-19 |
| Languages (V1) | **Hindi + English only.** English appears inside FIRs and is the translation target. **Marathi moves to V2** | You, 2026-11-18 |
| Time box | First results in 1 month (M1, 2026-12-20). Done in at most 2 months (M2, 2027-01-17) | You, 2026-11-18 |
| Domain | Police FIRs and complaint letters (Bihar, Hindi) | Data scan |
| Golden data | ~3K pages, **fully human-checked**, used as ground truth. Marathi pages are set aside for V2 | You |
| Licence use | **Research / internal only**, so academic-licence datasets are allowed | You |
| V1 output | Grounded bilingual blocks. **No key-value field extraction in V1** | You |
| Training method | LoRA in every stage. Stage 1: DeepEncoder V2 attention (r8) + decoder attention (r16); SAM, convs, projector, experts, router, embeddings frozen. No full fine-tune of the whole model. Tokenizer unchanged | You, 2026-11-19 (section 6) |
| Training code | HF Trainer + PEFT on the official, pinned checkpoint. Runs on 1→N GPUs. Unsloth only if measured speed is too low | You, 2026-11-18 |
| GPUs | RTX A6000 48GB is primary. MI300X 192GB only after a ROCm smoke test. The local RTX 3060 12GB is for data work and smoke tests | Chat + this machine |
| RL / preference tuning | Not in V1 | Chat |
| PDF rebuild | External deterministic renderer, outside the model | Chat |

## 2. Words used in this plan

| Term | Meaning |
|---|---|
| **Fine-tuning** | Continue training an already-trained model on new data |
| **Full fine-tuning** | Every weight in a part of the model is updated |
| **LoRA** | Instead of changing a big weight matrix, train two small matrices whose product is added to it. Cheap, forgets little, and can be merged into the weights later |
| **Rank (r)** | The size of the LoRA matrices. Higher rank means more capacity |
| **Frozen** | Weights that are not updated in a stage |
| **Merge** | Add the trained LoRA update into the base weights, giving a normal checkpoint |
| **Encoder / decoder** | The encoder turns the image into vision tokens. The decoder (a language model) writes text from them |
| **MoE (mixture of experts)** | Each decoder layer has many small "expert" networks. A **router** picks 6 of 64 for each token. 2 **shared experts** see every token |
| **CER / WER** | Character / word error rate: the edits needed to turn the output into the truth, divided by the truth's length. Lower is better |
| **chrF++, COMET** | Translation quality scores. Higher is better |
| **Replay** | Mixing earlier-stage data into a later stage so the model does not forget it |
| **Gate** | A pass/fail check before moving to the next stage |

## 3. What we start with (verified this session)

### 3.1 Golden data

A sample of 4 documents, 13 pages and 134 regions was scanned in `dataEngine/data/`.

**Content:**
- handwritten Hindi complaint letters
- printed FIR forms filled in by hand
- pages from Bihar police stations (Arariya, Aurangabad, Begusarai), rendered at 200–300 DPI

**Fields per region:**
- box in pixels
- reading order
- layout label from the **IndicDLP 42-class** set
- source text (Hindi or English)
- English translation
- status: ok 108, partial 22, illegible 3, no_text 1
- `[illegible]` and `[uncertain: …]` markers in the text
- 3 regions are rotated 90° and come with pre-rotated crops

**Languages in the sample:** hi 115 regions, en 79 (mixed).

**Missing:** a printed vs handwritten tag per region. Stage 0 adds it.

**Consequence:** handwriting is the core problem, not an optional extra.

### 3.2 The model

DeepSeek-OCR 2. Names and counts come from `python -m training.inventory`, which builds the real model structure without weights.

```
page image
  └─ visual tokenizer: model.sam_model (SAM ViT-B + conv downsampler net_2/net_3)     95.0M
  └─ DeepEncoder V2:   model.qwen2_model (Qwen2, 24 layers, hidden 896)              358.3M
                       attention: qwen2_model.model.model.layers.N.self_attn.{q,k,v,o}_proj
                       learned queries only for 768 (144) and 1024 (256) inputs
  └─ projector:        model.projector (one Linear 896→1280)                           1.1M
                                                                                       ▼
decoder "DeepseekV2": 12 layers, hidden 1280, 10 heads, plain multi-head attention
  model.layers.N.self_attn.{q,k,v,o}_proj                                             78.6M
  layer 0:     dense MLP  mlp.{gate,up,down}_proj                                     26.3M
  layers 1–11: mlp.experts.0..63.*  (64 routed, 6 used per token)                  2,422.2M
               mlp.shared_experts.* (2 shared, every token)                           75.7M
               mlp.gate             (router)                                           0.9M
  embed_tokens, lm_head (vocab 129,280; 165.5M each), max positions 8,192 (vision + text)
3.39B parameters in total
```

- **Box format:** `<|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`, with coordinates scaled to **0–999**.
- **Loss:** `forward(labels=…)` already computes cross-entropy. Positions labelled `-100` are ignored.

**Vision tokens per image:**

| Mode | What the model sees | Image-token positions |
|---|---|---|
| one 768 view (`crop_mode = false`) | image stretched to 768×768 | 145 |
| one 1024 view (`crop_mode = false`) | image padded to 1024 | 257 |
| tiles (`crop_mode = true`, the model's default) | 1024 global view + 2–6 tiles of 768 for images larger than 768 | 257 + 144 per tile |

### 3.3 Tokenizer

DeepSeek's tokenizer handles Hindi **losslessly**: it reproduced all 17,355 characters of your real FIR text exactly. It is costly, though:

| Language | Tokens per word |
|---|---|
| Hindi | 3.1 |
| English | 1.5 |

We keep the tokenizer unchanged and add three rules:
1. Normalize every text to NFC.
2. Check that model output is valid UTF-8.
3. Budget the output tokens per page.

Full numbers, including Marathi for V2, are in [tokenizer-check.md](tokenizer-check.md).

### 3.4 Training tools

**Backend: Hugging Face Trainer + PEFT** on the official checkpoint, calling the model's own `forward(labels=…)`. It scales from 1 to N GPUs with `torchrun` and keeps the pinned stack.

[Unsloth](https://unsloth.ai/docs/models/tutorials/deepseek-ocr-how-to-run-and-fine-tune) also supports DeepSeek-OCR and [OCR-2](https://unsloth.ai/docs/models/deepseek-ocr-2). Its Persian example cut CER from 149% to about 60% with decoder LoRA only. It stays a fallback in case measured speed is too low.

- **Important:** the official `forward` runs SAM, DeepEncoder V2 **and `projector`** inside `torch.no_grad()`, which would give encoder LoRA zero gradient. `models/deepseek_ocr/loader.py` disables that `no_grad`; frozen parts still build no graph. Smoke test S6 checks that every LoRA group gets gradients.
- **Official pinned stack:** Python 3.12.9, torch 2.6.0, transformers 4.46.3, flash-attn 2.7.3, CUDA 11.8.

## 4. Review of the ChatGPT plan

### Kept

- Fine-tune the pretrained DeepSeek-OCR. Do not train from scratch.
- Do not replace the tokenizer.
- LoRA first, with the vision encoder frozen at first.
- Public data teaches the script. Your pages teach the domain.
- Translation stays inside the model. The PDF renderer is external.
- RL is deferred.
- Lock the test set.
- Your idea is kept: a separate text-only decoder branch that is connected back later.

### Changed

| # | ChatGPT said | Reality / change | Why |
|---|---|---|---|
| 1 | Printed government text first, handwriting optional | **Handwriting is core.** Handwritten Hindi datasets are essential in Stage 1 | Your FIRs are handwritten |
| 2 | Hindi + Marathi together, 90-day+ roadmap | **Hindi only, 8 weeks, results at week 4** | Your decision, 2026-11-18 |
| 3 | No official training pipeline; write a custom loop | The model's own `forward(labels=…)` computes the loss, so **HF Trainer + PEFT** works with a small data collator. Unsloth also supports it (fallback) | Model code, Unsloth docs |
| 4 | Train the "2-layer conv compressor" | The bridge is **`projector`** (one Linear layer), and the official code runs it without gradients, so it stays frozen in V1. The 16× conv compressor (`net_2`, `net_3`) sits inside `sam_model`, also frozen | Model code |
| 5 | LoRA on `q/k/v/o_proj`, then expert projections | LoRA on attention **+ layer-0 dense MLP + `shared_experts`** (≈4.0M params at r16), because these see every token. The 64 routed experts (+≈73.5M params, each sees few tokens) are added only if the model underfits. The router stays frozen | Model code + arithmetic |
| 6 | JSON output, boxes 0–1000 | Native grounding format, **boxes 0–999** (section 7). The grounding tags are single special tokens | Model code + tokenizer test |
| 7 | Weighted multi-task loss (λ = 0.40/0.25/…) | One next-token loss. **Task balance comes from sampling shares** | Every task is next-token prediction, so loss weights add tuning work without adding control |
| 8 | (not covered) | Vision and text tokens share 8,192 positions. Measured: your pages need 1,420 output tokens on average and at most 2,145 in bilingual form | Tokenizer test on golden data |
| 9 | DeepSeek-OCR v1 | **Switched to DeepSeek-OCR 2** (Jan 2026): same decoder and tokenizer, Qwen2-based DeepEncoder V2, better reading order on its benchmark | You, 2026-11-19 |
| 10 | Split by document | **Split by police station**, falling back to by document | One officer writes many FIRs. A page-level split leaks their handwriting into test |
| 11 | Feed detected boxes as hints 50% of the time, 20% of them corrupted | **Deferred.** The V1 model detects layout itself | Simpler: one model does the job |
| 12 | Separate translation branch via adapter routing, merge, or full transfer | **Merge with strength λ, gated against a simpler control** (Stage 4). Keeping the branch as a separate frozen adapter equals a merge at λ=1 | Same result, less machinery |
| 13 | Large synthetic rendering programme | **Deferred to V2.** Public printed Hindi (Mozhi) + your pages cover V1 | Saves engineering time inside the time box |
| 14 | Many layout/table datasets | **Dropped for V1:** DocLayNet, DocBank, PubTables-1M, FinTabNet, IndicSTR12, Devanagari-OCR-ICL | Off-domain, different label sets, or a different task. IndicDLP uses the **same 42 labels as your data** |
| 15 | "Gemini outputs are teacher predictions, not truth" | Your 3K pages are **human-checked**, so they are truth | You |
| 16 | MI300X for high-resolution runs | Best used for the **text-only Stage 3** (no vision kernels), after a ROCm smoke test | Lower risk; runs in parallel with the A6000 |
| 17 | LoRA learning rate 2e-5 for the language branch | **1e-4** to start | 2e-5 is a full-fine-tuning rate. LoRA usually needs ~1e-4 |

## 5. Stage map

```
Stage 0  data + baseline (no training)
   │
   ├──► Stage 1  read Devanagari (crops): 1A printed → 1B handwritten → 1C joint
   │        │
   │        ▼
   │    Stage 2  understand FIR pages ──────────┐      ◄── M1 result (week 4)
   │                                            ▼
   └──► Stage 3  language branch (text only) ─► Stage 4  join reading + translation ─► Stage 5 ◄── M2 (week 8)
```

**Every training stage follows the same steps:**
1. Load the previous merged checkpoint.
2. Add a fresh LoRA adapter.
3. Train, with the loss on target tokens only.
4. Merge the adapter.
5. Save both the adapter and the merged checkpoint.

There is one training entry point, with one config per stage.

| # | Stage | Full or LoRA | Trains | Main data | Output | A6000 h* |
|---|---|---|---|---|---|---|
| 0 | [Data + baseline](stages/stage-0-data-and-baseline.md) | none | — | golden pages, public test sets | splits, OCR-2 baseline report | 5–8 |
| 1 | [Script adaptation](stages/stage-1-script-adaptation.md) (1A, 1B, 1C) | LoRA r8 encoder + r16 decoder | DeepEncoder V2 attention + decoder attention | 1A printed Hindi · 1B handwritten Hindi + English · 1C joint + golden regions | `stage1a/1b/1c …/merged` | 25–48 |
| 2 | [Document understanding](stages/stage-2-document-understanding.md) | LoRA r16 | as Stage 1 | full pages, regions, IndicDLP layout | `stage2` → **M1** | 15–30 |
| 3 | [Language branch](stages/stage-3-language-branch.md) | LoRA r32, decoder only | attention, dense MLP, shared experts | Sangraha-hi, BPCC hi↔en, your hi→en pairs | `lang_adapter` | 10–25 |
| 4 | [Joint OCR + translation](stages/stage-4-joint-ocr-translation.md) | merge experiment, then LoRA r32 | attention, dense MLP, shared experts | bilingual pages + replay | `hindidoc-v0.1` | 15–30 |
| 5 | [Release check](stages/stage-5-release.md) | none | — | locked test set | **M2** report | 3–6 |

\* Planning ranges. They are recomputed from the tokens/second measured in Stage 0. Total ≈ **73–147 A6000 hours**, about a third of the A6000's time over 8 weeks.

**The real bottleneck is engineering time, not GPU time.** That is why the plan keeps one training entry point, measures speed first, and has a cut list (section 11).

## 6. LoRA or full fine-tuning? What is frozen?

**Answer: LoRA in every stage, and no full fine-tuning in V1 by default.**

- Stage 1 trains LoRA on DeepEncoder V2 attention (r8) and decoder attention (r16): 3,047,424 parameters.
- The official forward computes the vision path under `torch.no_grad()`. The loader disables that, so encoder LoRA really trains (section 3.4).

**Why not full fine-tune the whole model in V1:**
- *Memory:* 3.39B parameters × ~16 bytes (weights + gradients + Adam states) ≈ 54GB before activations. That does not fit on the A6000.
- *Overfitting:* 3K pages is far too little data to move 3B weights safely.
- *Merging:* the Stage 3 branch must merge back. LoRA updates add cleanly; full updates do not.

**Follow-up experiments if Stage 1 needs more capacity** (one at a time, see [Stage 1 §7](stages/stage-1-script-adaptation.md#7-follow-up-experiments-only-after-the-first-baseline)):
1. LoRA on the decoder's MoE expert projections.
2. Unfreeze the last 2–4 DeepEncoder V2 blocks at a small learning rate (if handwriting plateaus).
3. Make the conv downsampler (`net_2`, `net_3`) trainable; then, selectively, the last SAM layers. Never the whole SAM.

| Module (exact name) | St 1 (1A/1B/1C) | St 2 | St 3 | St 4 |
|---|---|---|---|---|
| `sam_model` (incl. `net_2`, `net_3` downsampler) | frozen | frozen | not used | frozen |
| `qwen2_model…layers.N.self_attn.{q,k,v,o}_proj` (DeepEncoder V2) | **LoRA r8** | frozen† | not used | frozen† |
| rest of `qwen2_model`, `projector`, `view_seperator` | frozen | frozen | not used | frozen |
| `model.layers.N.self_attn.{q,k,v,o}_proj` (decoder) | **LoRA r16** | LoRA r16 | LoRA r32 | LoRA r32 |
| `model.layers.0.mlp.{gate,up,down}_proj` (dense) | frozen | LoRA r16 | LoRA r32 | LoRA r32 |
| `model.layers.1-11.mlp.shared_experts.*` | frozen | LoRA r16 | LoRA r32 | LoRA r32 |
| `model.layers.1-11.mlp.experts.*` (64 routed) | frozen (follow-up 1) | frozen | frozen | frozen |
| `model.layers.1-11.mlp.gate` (router) | frozen | frozen | frozen | frozen |
| `embed_tokens`, `lm_head` | frozen | frozen | frozen | frozen |

† Stage 1's encoder LoRA is merged into the weights. Whether Stages 2 and 4 also adapt the encoder is decided from the Stage 1 results.

`python -m training.inventory --config <run>` prints exactly what a config will train, before any GPU time.

## 7. Model input and output format

**Output:** one block per region, in reading order. Reading order is the order of the blocks, so no separate number is needed.

Real blocks from `Arariya_SC_ST_PS_07_18/page_0001`, text shortened and personal details replaced by placeholders:

```
<|ref|>section-title<|/ref|><|det|>[[153, 85, 770, 157]]<|/det|>
विषय :- मारपीट गाली गलौज घड़ी साइकिल एवं रूपया
छीन लेने के सम्बंध में।
[EN]
Subject :- Regarding assault, verbal abuse, snatching of watch, bicycle and money.

<|ref|>paragraph<|/ref|><|det|>[[133, 178, 822, 940]]<|/det|>
निवेदन पूर्वक कहना है कि मैं [नाम]
उम्र - [आयु] वर्ष पिता [नाम] ग्राम- [ग्राम] ...
[EN]
It is humbly stated that I, [Name],
age - [age] years, father [Name], resident ...
```

- **Boxes:** pixels scaled to 0–999 with `round(x / width * 999)`.
- **Labels:** the IndicDLP 42 classes, the same set your data uses.
- **Text:** NFC-normalized. Line breaks and the `[illegible]` / `[uncertain: …]` markers are kept.
- **`no_text` blocks** (e.g. a stamp) are the header line only.
- **Source-only version:** the same, with the `[EN]` lines removed.
- **Token cost:** `[EN]` costs 3–4 tokens. The `<|ref|>` / `<|det|>` tags are 1 token each.

**Fixed task prompts** (one per task, defined once in code, in the task registry `dataEngine/tasks.py`):

| Task | Prompt | Target |
|---|---|---|
| Crop OCR | `<image>\nFree OCR.` | exact text |
| Page parse | `<image>\n<|grounding|>Convert the document to markdown.` | source-only blocks |
| Read box | `<image>\n<|grounding|>Read the text in <|det|>[[x1, y1, x2, y2]]<|/det|>.` | text in that box |
| Layout only | `<image>\n<|grounding|>Detect the layout.` | block headers only |
| Bilingual page | `<image>\n<|grounding|>Convert the document to markdown and translate it to English.` | bilingual blocks |
| Bilingual crop | `<image>\nFree OCR and translate to English.` | source `\n[EN]\n` English |
| Translate (text only) | `Translate this Hindi text to English.\n{text}` (and the reverse) | translation |
| Language modelling (text only) | (none) | the text itself |

The first two prompts are DeepSeek-OCR's own, so we build on what it already knows.

## 8. Datasets (Hindi + English V1)

- **Trusted sources only.** Public data comes from its original publisher:
  - AI4Bharat (IIT Madras)
  - IIIT-Hyderabad CVIT, and the NLTM OCR portal `ilocr.iiit.ac.in` (MeitY / Bhashini programme)
  - Meta (FLORES)
  - the dataset authors' own release

  No third-party re-uploads. At download, record the URL, licence and SHA-256 of every archive in the data manifest.
- **Priority:** **P1** = needed for M1. **P2** = add if time allows. The plan works without P2.
- **Licence "check":** the licence was not found in this review. Record it at download time. Everything is for research use.

| Dataset | Content | Kind | Size (approx.) | Licence | Stage | Pri | Link |
|---|---|---|---|---|---|---|---|
| **Golden FIR pages** (yours) | Pages with boxes, labels, text, English | handwritten + printed forms | ~3K pages, ~10 regions/page | private | 0–5 | P1 | `dataEngine/data/` |
| IIIT-HW-Dev (= Hindi part of IIIT-INDIC-HW-WORDS; a third-party copy has exactly these split sizes, so it is one dataset, not two) | Handwritten Hindi words | handwritten | ~95K words (69,853 / 12,708 / 12,869) | CC BY 4.0 (ilocr) | 1B, 1C | P1 | [CVIT (official)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data), [ilocr](https://ilocr.iiit.ac.in/dataset/) |
| DohaScript | Continuous handwritten Hindi, 531 writers (lines) | handwritten | 531 pages (full set from the authors; 200-sample preview on their HF repo) | "other" on HF, check | 1B, 1C | P1 | [arXiv 2602.18089](https://arxiv.org/abs/2602.18089), [authors' HF](https://huggingface.co/datasets/SIR-Lab/DohaScript) |
| IAM Handwriting (lines) | Handwritten English lines, ~650 writers | handwritten | ~13K lines | research licence, registration | 1B, 1C | P1 | [FKI group (official)](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) |
| Mozhi (Hindi) | Printed words + lines, official splits | printed | ~80K train words, lines 80/10/10 | check | 1A, 1C, eval | P1 | [cvit](https://cvit.iiit.ac.in/usodi/tdocrmil.php), [portal](https://ilocr.iiit.ac.in/dataset/) |
| ICDAR 2024 HWD | Handwritten Hindi pages, word boxes + text (train ground truth only) | handwritten | check | check | 1 (crops), 2 (pages) | P2 | [ilocr](https://ilocr.iiit.ac.in/icdar_2024_hwd/dataset.html) |
| IIIT-Indic-HW-UC (Hindi) | Camera-captured handwriting | handwritten | Hindi part | check | 1 | P2 | [paper](https://cdn.iiit.ac.in/cdn/ilocr.iiit.ac.in/dataset/static/assets/img/publication/handwritten/unconstrained.pdf) |
| IndicDLP (hi + en), AI4Bharat | Layout boxes, **same 42 labels as your data** | printed | ~119.8K pages, all languages | MIT on the AI4Bharat card (access form) | 2 | P2 | [site](https://indicdlp.github.io/), [HF (AI4Bharat)](https://huggingface.co/datasets/ai4bharat/indicdlp) |
| Sangraha (Hindi) | Monolingual Hindi text | text | very large | CC-BY-4.0 | 3 | P1 | [HF](https://huggingface.co/datasets/ai4bharat/sangraha) |
| BPCC (hi↔en) | Parallel text (human-made + mined) | text | 2.2M human-made pairs (all languages) + mined | mined CC0, human-made CC-BY-4.0 | 3, 4 | P1 | [HF](https://huggingface.co/datasets/ai4bharat/BPCC) |
| IndicTrans2 (indic-en) | Translation model: baseline | model | 1B | MIT | 0, 3 | P1 | [GitHub](https://github.com/AI4Bharat/IndicTrans2) |
| FLORES-200 devtest | Translation test | text | 1,012 sentences | CC-BY-SA 4.0 | eval | P1 | [HF](https://huggingface.co/datasets/facebook/flores) |
| IN22-Gen | Translation test, Indian content | text | check | check | eval | P2 | [HF](https://huggingface.co/datasets/ai4bharat/IN22-Gen) |
| Sarvam indic-ocr-bench (Hindi) | Hand-reviewed OCR blocks | mixed | check | check | eval only | P2 | [HF](https://huggingface.co/datasets/sarvamai/indic-ocr-bench) |

**Do this week (before W1):**
- Request access to the gated sets (IndicDLP, BPCC).
- Start the downloads.

**Moved to V2:**
- All Marathi data: IIIT-Indic-HW-UC Marathi, ICDAR 2025 IHDR, Mozhi Marathi, Sangraha/BPCC Marathi.
- Synthetic rendering.
- OmniDocBench retention test.

**Not used:**
- DocLayNet, DocBank: different label sets.
- PubTables-1M, FinTabNet: tables in financial documents.
- IndicSTR12: scene text.
- Devanagari-OCR-ICL: post-OCR correction, a different task.

## 9. Evaluation and gates

**Rules:**
- Every metric is reported **per script type** (printed / handwritten) and for Hindi and English text separately.
- The test set is used **only for the M1 and M2 reports**, never to choose settings. All tuning uses dev.
- Gate values below are **proposed**. Stage 0 confirms them against the measured baseline and records them in the Decisions table.

**Targets:**

| Metric | Stage 1 | Stage 2 | Final V1 |
|---|---|---|---|
| Printed line CER | ≤3% | ≤3% | ≤3% |
| Government page CER | measure baseline | ≤5% | ≤5% |
| Layout mAP@50 | — | ≥0.85 | ≥0.85 |
| Bounding-box IoU | — | ≥0.80 | ≥0.80 |
| Extraction field F1 | — | ≥0.85 | ≥0.90 |
| Reading-order accuracy | — | ≥0.90 | ≥0.95 |
| Translation faithfulness | — | human-reviewed | human-reviewed |
| Valid structured output | — | ≥98% | ≥99% |

**How each is computed** (`evaluation/report.py`, `evaluation/page_metrics.py`):

| Metric | Computation |
|---|---|
| Crop CER / WER | Corpus-level edits ÷ reference length, raw and NFC. Characters are code points, so matras and nukta count. Reported per source, script, language and script × language. "Printed line CER" uses printed crops: the current Mozhi test set has words only |
| Page CER | Ground-truth block text in reading order vs the predicted blocks' text in output order. Whitespace collapsed; HTML tags removed from the prediction (the base model writes tables as HTML). Output without parsable blocks is scored as plain text |
| Layout AP@50 / mAP@50 | VOC all-point AP at IoU 0.5 per page, averaged. The model gives no confidence scores, so predictions are ranked by output order. Class-agnostic AP, and class-aware mAP over each page's ground-truth labels. The base model uses its own label names, so its class-aware score is near 0 until Stage 2 teaches the IndicDLP labels |
| Bounding-box IoU | One-to-one greedy matching at any overlap; mean over ground-truth blocks, unmatched = 0. Boxes are 0–999 on both sides, and IoU is unchanged by per-axis scaling |
| Block P / R / F1 | One-to-one greedy matching at IoU ≥ 0.5, labels ignored |
| Reading-order accuracy | Share of matched block pairs whose relative order agrees with the ground truth |
| Valid structured output | Output parses into well-formed grounded blocks (0–999 boxes, no leftover tags, valid UTF-8) **and** generation ended before `max_new_tokens` |
| Extraction field F1 | **Not measurable:** no field labels exist in the data, and key-value fields are outside V1 scope |
| Translation faithfulness | **Not measurable before Stage 4** (no translated output). Review by hand then, with number/ID preservation and chrF++/COMET as automatic checks |

**Before-training check on the test data** (`dataEngine/data/test_data`):
- Convert with `configs/data/test_data.toml`.
- Evaluate with `configs/eval/test_data.toml --split test --checkpoint start`.
- The sample:
  - 200 Mozhi printed words
  - 200 golden FIR regions
  - all 49 golden pages (538 blocks)
- `test_data/cvit` is a byte-identical copy of `test_data/mozhi`, so it is used once.

| Stage | Gate (proposed) |
|---|---|
| 1 | Per sub-stage, in each config's `[[eval.gates]]` ([Stage 1 §6](stages/stage-1-script-adaptation.md#6-evaluation-before-and-after)). Overall vs the original OCR-2: handwritten-Hindi CER drops ≥30% relative; Mozhi Hindi CER ≤3%; English CER rises by ≤0.5 points |
| 2 | Block F1@0.5 ≥0.85; Kendall τ ≥0.90; parses ≥98%; crop CER no more than 0.5 points worse than Stage 1 |
| 3 | chrF++ ≥ IndicTrans2 − 3 (FLORES hi→en); number/ID preservation ≥99% |
| 4 | OCR CER no more than 0.5 points worse than Stage 2; parses ≥99%; block recall and precision ≥0.98; number/ID preservation ≥99%; translation chrF++ ≥ two-model pipeline (Stage 2 + IndicTrans2) − 3 |
| 5 | Final numbers on the locked test set; quantized model CER no more than 0.5 points worse |

## 10. Compute, memory, storage

**Machines:**

| Machine | Role |
|---|---|
| A6000 48GB (CUDA) | All vision stages |
| MI300X 192GB (ROCm) | Stage 3 (text only), after a smoke test. CUDA and ROCm cannot share one training job |
| RTX 3060 12GB | Data conversion, smoke tests, small inference |

**Memory** (estimates; Stage 0 measures the real numbers):
- **Weights** in bf16 ≈ 6.8GB (3.39B parameters).
- **Trainable state:** Stage 1 LoRA 3.0M parameters; later stages ≈4–8M. Optimizer state < 0.2GB.
- **Activations** are the real cost:
  - Full page: up to ~1,150 vision tokens + ≤2,200 target tokens. Micro-batch 1 with gradient checkpointing, est. 25–40GB.
  - Crops: 145 image positions at 768, micro-batch 8.
- **Speed risk:** Hugging Face's MoE code loops over experts in Python, so real speed may be lower. Stage 0 measures it.

**Storage:** ≈150GB.
- Public Hindi data ~40GB.
- Merged checkpoints ≈6.7GB each, ~6 of them.
- Logs.

## 11. Timeline (8 weeks) and roadmap

Weeks start on Monday 2026-11-23. This week (11-18 → 11-22): request dataset access, start downloads, set up the environment.

```
            W1    W2    W3    W4    W5    W6    W7    W8
Stage 0     ████
Stage 1           ██████████                           (1A + 1B in W2, 1C in W3)
Stage 2                       ████
M1 report                        ██
Stage 3           ░░░░░░░░░░░░                         (MI300X, parallel)
Stage 4                             ██████████
Stage 5                                         ████
Buffer                                                ████   (follow-up experiments / reruns / M2)
```

| Week | Dates | Work | Exit |
|---|---|---|---|
| W1 | 11-23 → 11-29 | Stage 0: convert, split, smoke tests, OCR-2 baseline (`evaluation --checkpoint start`), measure speed | splits locked, gates set |
| W2 | 11-30 → 12-06 | Stage 1A, then 1B. Stage 3 starts on MI300X | 1A and 1B gates |
| W3 | 12-07 → 12-13 | Stage 1C, then start Stage 2 | 1C gate |
| W4 | 12-14 → 12-20 | Finish Stage 2. **M1 report** on test: Hindi FIR reading + layout vs the original OCR-2. Stage 3 gate | **M1: Hindi FIR reader** |
| W5–6 | 12-21 → 2027-01-03 | Stage 4: J0 vs J1, then the full joint run | Stage 4 gate |
| W7 | 2027-01-04 → 01-10 | Stage 5: test set, quantization, latency, renderer contract | release check |
| W8 | 2027-01-11 → 01-17 | Buffer: follow-up experiments or reruns. **M2 report** | **M2: hindidoc-v0.1** |

**Cut list if behind schedule.** Drop items from the top first. The bottom block is never dropped.

1. Stage 1 follow-up experiments (MoE expert LoRA, encoder/conv unfreeze, 1024 resolution).
2. P2 datasets (ICDAR 2024 HWD, HW-UC, IndicDLP, IN22, Sarvam).
3. J1 merge experiments. Run Stage 4 as J0 only.
4. Stage 3 as a separate branch. Its data goes straight into Stage 4 (J0).
5. Quantization in Stage 5.

**Never cut:**
- Stage 0 splits and baseline
- Stages 1, 2, 4
- the gates
- test-set discipline

**Roadmap after V1:**
- **V1.1 — Marathi:**
  - Public Marathi data (Mozhi-mr, HW-UC-mr, ICDAR 2025 IHDR, Sangraha/BPCC-mr) plus your limited Marathi pages.
  - An active-learning loop: the model reads unlabeled pages, people fix the low-confidence blocks, retrain.
- **V2:**
  - Synthetic rendering.
  - FIR key-value fields (needs a schema + labels).
  - Preference tuning (DPO/GRPO) with checkable rewards: CER, box IoU, number preservation.
  - SAM unfreeze.
  - Edge distillation.
  - Multi-page documents.

## 12. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Engineering time | Weeks slip | One entry point, P1-only data, cut list |
| Stage 1 now has three sub-stages | Stage 2 squeezed into W3–W4 before M1 | Measure speed in Stage 0; shorten 1C epochs before touching Stage 2 |
| Encoder LoRA silently untrained (official `no_grad`) | Stage 1 learns only in the decoder | Loader fix + smoke S6 per-group gradient check |
| Custom model code vs library versions | Training won't start or gives a wrong loss | Pin the stack; Stage 0 smoke tests (load, freeze check, finite loss, gradients flow, save/reload) |
| Slow MoE training | Hours exceed the estimates | Measure in Stage 0; Unsloth; smaller epochs |
| LoRA alone cannot read handwriting | 1B gate fails | Follow-up experiments (encoder blocks, conv downsampler) in the W8 buffer |
| Same writer in train and test | Scores look better than reality | Split by police station |
| Forgetting English text | Mixed pages break | English regions in replay + retention gate |
| Language branch damages OCR | Stage 4 regresses | J0 control + λ sweep + OCR gate |
| The human check covered text but not boxes/labels | Noisy layout supervision (86 of 134 sample regions had low-confidence labels) | Stage 0 confirms what was checked and spot-checks labels |
| Model memorizes names from FIRs | Privacy leak | Data and checkpoints stay private; memorization probe in Stage 5 |
| ROCm problems on MI300X | Stage 3 waits for the A6000 | Smoke test first. Run Stage 3 on the A6000 in W4 (mostly free during the M1 report). If that slips too, apply cut-list item 4 |
| Holidays (W5–6) | Less hands-on time | Stage 4 runs are long, unattended jobs |

## 13. Open items (filled as they become known)

| Item | Needed by | Recorded in |
|---|---|---|
| Hindi page count; number of police stations / documents | Stage 0 (split) | Decisions table |
| What the human check covered (text only, or boxes + labels too) | Stage 0 | Decisions table |
| Image mode for 1A/1B: 768 vs 1024 view | Stage 0 baseline | Decisions table |
| Final gate values | Stage 0 baseline | Decisions table |
| Edge target device and latency budget | Stage 5 | Decisions table |

The Decisions table lives in `.claude/skills/ocr-pipeline/SKILL.md`, section 8.

## 14. Sources

- DeepSeek-OCR: [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR), [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR), [paper](https://arxiv.org/abs/2510.18234)
- DeepSeek-OCR-2: [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), [paper](https://arxiv.org/abs/2601.20552)
- Unsloth: [DeepSeek-OCR fine-tuning](https://unsloth.ai/docs/models/tutorials/deepseek-ocr-how-to-run-and-fine-tune), [DeepSeek-OCR 2](https://unsloth.ai/docs/models/deepseek-ocr-2)
- Datasets: links in section 8.
- Measured in this repo: dataset scan, tokenizer test and token budget ([tokenizer-check.md](tokenizer-check.md)).
