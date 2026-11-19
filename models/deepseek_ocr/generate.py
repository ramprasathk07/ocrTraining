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
) -> tuple[str, bool]:
    """(text, finished): the output for `prompt` + `image`, and whether it ended on EOS.

    Special tokens are kept because grounding tags (<|ref|>, <|det|>) are special tokens.
    `finished` is false when generation hit `max_new_tokens` (a truncated output).
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
    new_tokens = output[0, input_ids.shape[1] :]
    finished = bool(len(new_tokens)) and int(new_tokens[-1]) == EOS_ID
    text = tokenizer.decode(
        new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return text.removesuffix(tokenizer.eos_token).strip(), finished
