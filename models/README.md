# models/

Model-specific code, shared by training and evaluation so both see exactly the same inputs. Only `deepseek_ocr/` exists today. A second model family would get its own sibling folder with the same three files.

## deepseek_ocr/: DeepSeek-OCR 2 (3B)

| File | Does |
|---|---|
| `inputs.py` | `ImageConfig` (image mode validation), `encode_image` (global view + tiles, mirroring the model's own `infer()`), `encode_sample` (prompt + target → input ids, labels masked to target only), `collate` |
| `loader.py` | `load_tokenizer`, `load_model(cfg, adapter=None)`: official code at the pinned revision, optional LoRA adapter on top; `allow_vision_gradients` |
| `generate.py` | `generate_text(...) -> (text, finished)`: greedy decoding; `finished` = stopped at EOS, not at the token limit |

Model facts the code relies on (revision `aaa02f38`):
- **Parts:** `sam_model` (SAM + convs, 95M) → `qwen2_model` (DeepEncoder V2, 358M) → `projector` (Linear 896→1280) → DeepseekV2 MoE decoder (12 layers, 64 routed + 2 shared experts). 3.39B parameters in total.
- **Input sizes:** the encoder accepts only 768 or 1024 inputs (144 or 256 queries). The image modes are in `configs/README.md`.
- **Vision gradients:** the official `forward` runs vision under `torch.no_grad()`. `allow_vision_gradients` replaces it so LoRA on DeepEncoder V2 receives gradients, and a test covers this.
- **Pinned transformers:** the model's own code needs transformers **4.46.3**.

Check what a config would train without downloading weights:

```bash
python -m training.inventory --config configs/stage1/1a_printed.toml
```
