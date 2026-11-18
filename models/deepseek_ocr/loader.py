"""Load DeepSeek-OCR (official code, pinned revision) and, optionally, a LoRA adapter."""

from dataclasses import dataclass
from pathlib import Path

import torch

# The model code maps only these two (as "mha_eager" / "mha_flash_attention_2"); "sdpa" would crash.
ATTENTION_IMPLEMENTATIONS = ("eager", "flash_attention_2")
SUPPORTED_VISION = (
    "deeplip_b_l"  # v1: SAM + CLIP-L. OCR-2 ("deepencoderv2") needs its own input encoder.
)


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    revision: str
    attn_implementation: str

    def __post_init__(self) -> None:
        if self.attn_implementation not in ATTENTION_IMPLEMENTATIONS:
            raise ValueError(f"attn_implementation must be one of {ATTENTION_IMPLEMENTATIONS}")


def load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.name_or_path, revision=cfg.revision, trust_remote_code=True
    )


def load_model(cfg: ModelConfig, adapter: Path | None = None):
    """bf16 model on CPU (the caller moves it to its device), plus a LoRA adapter if given."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        cfg.name_or_path,
        revision=cfg.revision,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation=cfg.attn_implementation,
    )
    vision = model.config.vision_config
    vision_name = (
        vision.get("model_name")
        if isinstance(vision, dict)
        else getattr(vision, "model_name", None)
    )
    if vision_name != SUPPORTED_VISION:
        raise NotImplementedError(
            f"vision encoder {vision_name!r}: only the DeepSeek-OCR v1 ({SUPPORTED_VISION}) "
            "input encoding exists"
        )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model
