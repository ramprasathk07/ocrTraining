# docs/

The plan and its reasoning. The code follows these documents, and the decisions taken while building are logged in `.claude/skills/ocr-pipeline/SKILL.md`.

| File | Content |
|---|---|
| [training-plan.md](training-plan.md) | main plan: goal, decisions, model facts, stage map, freeze matrix, datasets (official sources + licences), output format, evaluation metrics and gates (§9), timeline, risks |
| [training-throughput.md](training-throughput.md) | training speed for OCR-2: where the compute goes, bottlenecks in the model code (MoE loop, batch-1 vision, full-vocab loss, active aux loss), phased plan with kernels, how to profile on a Linux server and read traces |
| [tokenizer-check.md](tokenizer-check.md) | DeepSeek tokenizer on Hindi: lossless but costly (≈3.1 tokens/word); rules that follow (NFC, UTF-8 check, token budget) |
| [stages/stage-0-data-and-baseline.md](stages/stage-0-data-and-baseline.md) | data conversion, splits, baseline, smoke tests |
| [stages/stage-1-script-adaptation.md](stages/stage-1-script-adaptation.md) | 1A printed → 1B handwritten → 1C joint: data, LoRA targets, gates |
| [stages/stage-2-document-understanding.md](stages/stage-2-document-understanding.md) | full pages: boxes, labels, reading order |
| [stages/stage-3-language-branch.md](stages/stage-3-language-branch.md) | text-only Hindi + translation branch |
| [stages/stage-4-joint-ocr-translation.md](stages/stage-4-joint-ocr-translation.md) | reading + translation in one model |
| [stages/stage-5-release.md](stages/stage-5-release.md) | final test, compression, renderer hand-off |
| `ChatGPT-Training OCR Models Properly-….md` | the original discussion the plan was reviewed against (kept unedited) |

Each stage doc uses the same template:
- goal
- input → output checkpoint
- data
- prompt/target example
- trainable and frozen modules
- hyperparameters
- compute
- exit gate
- risks

Where each stage lives in code: `configs/README.md` (configs) and `training/README.md` (how a stage runs).
