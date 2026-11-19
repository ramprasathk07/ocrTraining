"""Load DeepSeek-OCR 2 (official code, pinned revision) and, optionally, a LoRA adapter."""

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Decoder attention: the model code maps only these two ("mha_eager" / "mha_flash_attention_2").
# DeepEncoder V2 always uses sdpa internally.
ATTENTION_IMPLEMENTATIONS = ("eager", "flash_attention_2")
SUPPORTED_VISION = "deepencoderv2"  # OCR-2: SAM tokenizer + Qwen2-based DeepEncoder V2


class _TorchWithoutNoGrad:
    """`torch` as the model's own module sees it, except that `no_grad` does nothing.

    The official forward runs SAM, DeepEncoder V2 and the projector inside torch.no_grad(),
    which cuts every gradient to encoder LoRA. Without it, frozen parts still record no graph,
    because nothing in them requires a gradient. Inference stays under torch.inference_mode().
    """

    no_grad = staticmethod(contextlib.nullcontext)

    def __getattr__(self, name: str):
        return getattr(torch, name)


def allow_vision_gradients(model) -> None:
    """Let gradients flow through the vision path of a loaded model (see _TorchWithoutNoGrad)."""
    sys.modules[type(model.model).__module__].torch = _TorchWithoutNoGrad()


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
            f"vision encoder {vision_name!r}: only the DeepSeek-OCR 2 ({SUPPORTED_VISION}) "
            "input encoding exists"
        )
    allow_vision_gradients(model)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model
