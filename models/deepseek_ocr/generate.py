"""Greedy generation for one sample, using the same input encoding as training."""

from pathlib import Path

import torch

from models.deepseek_ocr.inputs import EOS_ID, ImageConfig, encode_sample, load_image


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    image: Path,
    cfg: ImageConfig,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
) -> str:
    """Model output for `prompt` + `image`, decoded with special tokens kept.

    Special tokens are kept because grounding tags (<|ref|>, <|det|>) are special tokens.
    """
    # shortcut: one sample per call (no left-padded batching); batch it if evaluation is too slow.
    device = next(model.parameters()).device
    encoded = encode_sample(prompt, None, load_image(image), tokenizer, cfg)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            images=[(encoded["patches"].to(device), encoded["global_view"].to(device))],
            images_seq_mask=encoded["images_seq_mask"].unsqueeze(0).to(device),
            images_spatial_crop=encoded["tiles"].unsqueeze(0),
            do_sample=False,
            eos_token_id=EOS_ID,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            use_cache=True,
        )
    text = tokenizer.decode(
        output[0, input_ids.shape[1] :],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.removesuffix(tokenizer.eos_token).strip()
