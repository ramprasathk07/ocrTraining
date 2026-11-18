# Tokenizer check: can DeepSeek-OCR handle Hindi?

**Answer: yes, without loss, but at about 2× the token cost of English. Keep the tokenizer and follow the four rules in section 4.**

Checked on 2026-11-18 against the tokenizers of `deepseek-ai/DeepSeek-OCR` and `deepseek-ai/DeepSeek-OCR-2`. They are identical, so everything below applies to both.

Part of the [training plan](training-plan.md).

## 1. What the tokenizer is

| Property | Value | Meaning |
|---|---|---|
| Type | Byte-level BPE | Any Unicode text can be encoded: there is no "unknown" token, and nothing is ever lost |
| Vocabulary | 128,827 base + 830 added tokens (config pads to 129,280) | — |
| Normalizer | none | Text is tokenized exactly as given, so *we* must normalize it consistently |
| Tokens containing Devanagari | 303 (0.23% of the vocabulary) | Hindi words get split into many small pieces |
| Grounding tags | `<|ref|>`, `<|/ref|>`, `<|det|>`, `<|/det|>`, `<|grounding|>`, `<image>` are each **1 token** | Our output format builds on them at no extra cost |

## 2. Results

**Text used:**
- **Hindi:** all `source_text` from the 115 Hindi regions in `dataEngine/data/` (real FIRs).
- **English:** a sample FIR sentence.
- **Marathi:** checked for V2.

| Text | Words | Characters | Tokens | Tokens / word | Characters / token | Exact round-trip |
|---|---|---|---|---|---|---|
| Hindi, real FIR text | 3,024 | 17,355 | 9,473 | **3.13** | 1.83 | ✅ |
| English sample | 17 | 100 | 25 | **1.47** | 4.00 | ✅ |
| Marathi sample (V2) | 47 | 332 | 202 | 4.30 | 1.64 | ✅ (incl. ळ, ऱ, ॲ) |

*Exact round-trip* means that encoding and then decoding gave back the identical string.

**Edge cases:**

| Case | Result |
|---|---|
| Nukta letters, precomposed (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़, U+0958–U+095F) | Round-trip ✅, **16 tokens** |
| Same letters, decomposed (base + ़ U+093C) | Round-trip ✅, **15 tokens**, a *different* token sequence |
| `प्राथमिकी` | `प्र` · `ा` + half of `थ` · rest of `थ` · `म` · `िक` · `ी` |

The last example shows that the byte-level vocabulary can split a single Devanagari character across two tokens.

## 3. Output length on your real pages

Each golden page was turned into the planned output format: one block per region, with tags, box, Hindi text and `[EN]` translation (see [training-plan.md §7](training-plan.md#7-model-input-and-output-format)).

| | Mean tokens / page | Max tokens / page |
|---|---|---|
| Source only (Stage 2) | 944 | 1,406 |
| Bilingual (Stage 4) | 1,420 | 2,145 |

**Budget:**
- The model has 8,192 positions for vision and text together.
- A full page uses ≤ ~1,150 vision tokens (Gundam / OCR-2 default) plus the prompt.
- So a **max output of 4,096 tokens** leaves 1.9× headroom over the longest page seen.

Re-check this on all ~3K pages in Stage 0.

## 4. Verdict and rules

**Keep the tokenizer. No vocabulary extension in V1.** New tokens would need new embeddings trained from scratch. 3K pages cannot do that well, and it would break the merge of the Stage 3 branch.

| # | Rule | Why | Where it is enforced |
|---|---|---|---|
| 1 | **NFC-normalize** every training target and every string we score | Precomposed and decomposed nukta give different tokens. Mixing them teaches the model two spellings of one letter | `dataEngine` (one normalize function, used by data prep and evaluation) |
| 2 | **Check that model output is valid UTF-8** before parsing | A character can be split across tokens, so a generation that stops mid-character produces broken bytes | Evaluation parser; counts as a parse failure |
| 3 | **Report CER raw and NFC-normalized**, and never normalize away matras or nukta | Normalization must not hide real reading errors | `evaluation` metrics |
| 4 | **Budget output tokens**: max 4,096 new tokens per page at first | Hindi costs ~2× English. Verbose JSON would waste the budget, so we use the native tag format | Stage configs |

**Cost to accept:** Hindi pages take about 2× the tokens of the same content in English. That means slower training and inference per page. It is fine for V1. Revisit vocabulary extension only if V2 speed targets require it.

## 5. How to re-run

Run this whenever the base model changes. It needs only `pip install tokenizers` and downloads `tokenizer.json` into memory.

```python
import json, glob, urllib.request
from tokenizers import Tokenizer

REPO = "deepseek-ai/DeepSeek-OCR"  # or deepseek-ai/DeepSeek-OCR-2
with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/main/tokenizer.json") as r:
    tok = Tokenizer.from_str(r.read().decode("utf-8"))

regions = [
    json.load(open(p, encoding="utf-8"))
    for p in glob.glob("dataEngine/data/*/pages/*/source_language_extraction/r*.json")
]
hindi = "\n".join(
    r["source_text"]
    for r in regions
    if "hi" in r.get("source_languages", []) and r.get("source_text")
)

ids = tok.encode(hindi, add_special_tokens=False).ids
print("tokens/word:", len(ids) / len(hindi.split()))
print("exact round-trip:", tok.decode(ids) == hindi)
```

Under PowerShell on Windows, call `sys.stdout.reconfigure(encoding="utf-8")` before printing Devanagari.
