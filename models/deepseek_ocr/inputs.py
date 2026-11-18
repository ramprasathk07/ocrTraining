"""DeepSeek-OCR v1 input encoding: image preprocessing, token layout, labels, batching.

Mirrors `infer()` and `dynamic_preprocess()` in the model's own code (MIT licence,
deepseek-ai/DeepSeek-OCR at the revision pinned in config). Training, evaluation and the
official inference therefore see identical inputs. OCR-2 uses a different layout (no
row-end tokens, 768 tiles) and needs its own encoder before it can be trained.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_ID = 0
EOS_ID = 1
IGNORE_INDEX = -100
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
TILE_SIZE = 640  # dynamic_preprocess default tile size in v1
PAD_RGB = (127, 127, 127)  # int(0.5 * 255): the model pads with its normalisation mean


@dataclass(frozen=True)
class ImageConfig:
    """The model's own preprocessing knobs (see README: Tiny/Small/Base/Large/Gundam)."""

    base_size: int
    image_size: int
    crop_mode: bool

    def __post_init__(self) -> None:
        if self.crop_mode and self.image_size != TILE_SIZE:
            raise ValueError(
                f"crop_mode tiles are {TILE_SIZE}px in v1; set image_size = {TILE_SIZE}"
            )


@dataclass(frozen=True)
class EncodedImage:
    patches: (
        torch.Tensor
    )  # (tiles, 3, 640, 640), or zeros(1, 3, base, base) when there are no tiles
    global_view: torch.Tensor  # (1, 3, size, size)
    tiles: tuple[int, int]  # (tiles across, tiles down)
    num_tokens: int  # image-token positions this image occupies in the sequence


def load_image(path: Path) -> Image.Image:
    """Open as RGB with EXIF rotation applied, as the model's `load_image` does."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    """ToTensor + Normalize(mean=0.5, std=0.5) → bf16 (3, H, W), as BasicImageTransform."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array - 0.5) / 0.5).permute(2, 0, 1).to(torch.bfloat16)


def _queries(size: int) -> int:
    return math.ceil((size // PATCH_SIZE) / DOWNSAMPLE_RATIO)


def _closest_aspect_ratio(
    aspect_ratio: float, ratios: list[tuple[int, int]], width: int, height: int
) -> tuple[int, int]:
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for ratio in ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * TILE_SIZE * TILE_SIZE * ratio[0] * ratio[1]:
            best = ratio
    return best


def _tile(
    image: Image.Image, min_num: int = 2, max_num: int = 9
) -> tuple[list[Image.Image], tuple[int, int]]:
    """Resize to a grid of 640px tiles closest to the image's aspect ratio (dynamic_preprocess)."""
    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda r: r[0] * r[1],
    )
    across, down = _closest_aspect_ratio(width / height, ratios, width, height)
    resized = image.resize((TILE_SIZE * across, TILE_SIZE * down))
    tiles = [
        resized.crop((x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE))
        for y in range(down)
        for x in range(across)
    ]
    return tiles, (across, down)


def encode_image(image: Image.Image, cfg: ImageConfig) -> EncodedImage:
    """Pixels and token count for one image, following infer() branch by branch."""
    tiles: list[Image.Image] = []
    grid = (1, 1)
    if cfg.crop_mode:
        if image.size[0] > TILE_SIZE or image.size[1] > TILE_SIZE:
            tiles, grid = _tile(image)
        global_view = ImageOps.pad(image, (cfg.base_size, cfg.base_size), color=PAD_RGB)
        base = _queries(cfg.base_size)
        num_tokens = (
            base + 1
        ) * base + 1  # rows of tokens + a row-end token each, + view separator
        if tiles:
            q = _queries(cfg.image_size)
            num_tokens += (q * grid[0] + 1) * (q * grid[1])
    else:
        if cfg.image_size <= TILE_SIZE:
            image = image.resize(
                (cfg.image_size, cfg.image_size)
            )  # the model stretches small modes
        global_view = ImageOps.pad(image, (cfg.image_size, cfg.image_size), color=PAD_RGB)
        q = _queries(cfg.image_size)
        num_tokens = (q + 1) * q + 1

    patches = (
        torch.stack([to_tensor(t) for t in tiles])
        if tiles
        else torch.zeros((1, 3, cfg.base_size, cfg.base_size), dtype=torch.bfloat16)
    )
    return EncodedImage(patches, to_tensor(global_view).unsqueeze(0), grid, num_tokens)


def encode_sample(
    prompt: str, target: str | None, image: Image.Image, tokenizer, cfg: ImageConfig
) -> dict:
    """Token ids, labels and image tensors for one sample.

    Layout: [BOS] prompt-before-<image> [IMAGE × n] prompt-after target [EOS].
    Loss only on target + EOS. `target=None` builds a generation prompt.
    """
    if prompt.count(IMAGE_TOKEN) != 1:
        raise ValueError(f"prompt must contain exactly one {IMAGE_TOKEN}: {prompt!r}")
    before, after = prompt.split(IMAGE_TOKEN)
    encoded = encode_image(image, cfg)
    head = [BOS_ID, *tokenizer.encode(before, add_special_tokens=False)]
    tail = tokenizer.encode(after, add_special_tokens=False)
    prompt_ids = head + [IMAGE_TOKEN_ID] * encoded.num_tokens + tail
    target_ids = (
        [] if target is None else [*tokenizer.encode(target, add_special_tokens=False), EOS_ID]
    )

    image_mask = (
        [False] * len(head) + [True] * encoded.num_tokens + [False] * (len(tail) + len(target_ids))
    )
    return {
        "input_ids": torch.tensor(prompt_ids + target_ids),
        "labels": torch.tensor([IGNORE_INDEX] * len(prompt_ids) + target_ids),
        "images_seq_mask": torch.tensor(image_mask),
        "patches": encoded.patches,
        "global_view": encoded.global_view,
        "tiles": torch.tensor(encoded.tiles),
    }


def _pad(rows: list[torch.Tensor], value: int | bool, length: int) -> torch.Tensor:
    out = torch.full((len(rows), length), value, dtype=rows[0].dtype)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad a list of encoded samples into the keyword arguments of the model's forward."""
    length = max(len(item["input_ids"]) for item in batch)
    input_ids = _pad([item["input_ids"] for item in batch], pad_id, length)
    return {
        "input_ids": input_ids,
        "attention_mask": _pad([torch.ones_like(item["input_ids"]) for item in batch], 0, length),
        "labels": _pad([item["labels"] for item in batch], IGNORE_INDEX, length),
        "images_seq_mask": _pad([item["images_seq_mask"] for item in batch], False, length),
        "images": [(item["patches"], item["global_view"]) for item in batch],
        "images_spatial_crop": torch.stack([item["tiles"] for item in batch]),
    }
