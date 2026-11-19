import pytest
import torch
from PIL import Image

from models.deepseek_ocr.inputs import (
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
    IMAGE_TOKEN_ID,
    ImageConfig,
    collate,
    encode_image,
    encode_sample,
    to_tensor,
)

PROMPT = "<image>\nFree OCR."
VIEW_768 = ImageConfig(base_size=1024, image_size=768, crop_mode=False)
VIEW_1024 = ImageConfig(base_size=1024, image_size=1024, crop_mode=False)
TILED = ImageConfig(base_size=1024, image_size=768, crop_mode=True)


class StubTokenizer:
    """One id per character, far from the special ids."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(c) for c in text]


def test_768_view_layout_and_labels():
    item = encode_sample(
        PROMPT, "अब", Image.new("RGB", (300, 60), "white"), StubTokenizer(), VIEW_768
    )
    ids, labels = item["input_ids"].tolist(), item["labels"].tolist()
    n_image = 145  # 12 x 12 DeepEncoder V2 queries + view separator (no row-end tokens in OCR-2)
    prompt_len = 1 + n_image + len("\nFree OCR.")

    assert ids[0] == BOS_ID
    assert ids[1 : 1 + n_image] == [IMAGE_TOKEN_ID] * n_image
    assert int(item["images_seq_mask"].sum()) == n_image
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == [1000 + ord("अ"), 1000 + ord("ब"), EOS_ID]
    assert item["global_view"].shape == (1, 3, 768, 768)  # stretched, as the model does
    assert not item["patches"].any()  # no tiles: zeros mean "global view only" to the model


def test_1024_view_has_256_queries():
    encoded = encode_image(Image.new("RGB", (300, 60)), VIEW_1024)
    assert encoded.num_tokens == 16 * 16 + 1
    assert encoded.global_view.shape == (1, 3, 1024, 1024)


def test_generation_prompt_has_no_target():
    item = encode_sample(PROMPT, None, Image.new("RGB", (300, 60)), StubTokenizer(), VIEW_768)
    assert (item["labels"] == IGNORE_INDEX).all()
    assert item["input_ids"][-1] != EOS_ID


def test_tiled_full_page():
    encoded = encode_image(Image.new("RGB", (1698, 2325)), TILED)  # a golden FIR page size
    assert encoded.tiles == (2, 3)  # closest grid with 2..6 tiles
    assert encoded.patches.shape == (6, 3, 768, 768)
    assert encoded.num_tokens == 16 * 16 + 1 + (12 * 2) * (12 * 3)


def test_tiled_small_crop_uses_global_view_only():
    encoded = encode_image(Image.new("RGB", (700, 120)), TILED)
    assert encoded.tiles == (1, 1)
    assert encoded.num_tokens == 257
    assert encoded.global_view.shape == (1, 3, 1024, 1024)


def test_wide_image_gets_wide_grid():
    assert encode_image(Image.new("RGB", (3000, 1000)), TILED).tiles == (3, 1)


@pytest.mark.parametrize(
    ("base", "size", "crop"), [(1024, 640, False), (1024, 1024, True), (640, 768, True)]
)
def test_sizes_without_encoder_queries_are_rejected(base, size, crop):
    with pytest.raises(ValueError):
        ImageConfig(base_size=base, image_size=size, crop_mode=crop)


def test_to_tensor_normalises_to_minus_one_one():
    tensor = to_tensor(Image.new("RGB", (2, 1), "white"))
    assert tensor.dtype == torch.bfloat16 and tensor.shape == (3, 1, 2)
    assert float(tensor.max()) == 1.0
    assert float(to_tensor(Image.new("RGB", (2, 1), "black")).min()) == -1.0


def test_collate_right_pads_every_field():
    image = Image.new("RGB", (300, 60))
    short = encode_sample(PROMPT, "अ", image, StubTokenizer(), VIEW_768)
    long = encode_sample(PROMPT, "अबक", image, StubTokenizer(), VIEW_768)
    batch = collate([short, long], pad_id=2)

    assert batch["input_ids"].shape == (2, len(long["input_ids"]))
    assert batch["input_ids"][0, -2:].tolist() == [2, 2]
    assert batch["labels"][0, -2:].tolist() == [IGNORE_INDEX] * 2
    assert batch["attention_mask"][0].tolist().count(0) == 2
    assert not batch["images_seq_mask"][0, -2:].any()
    assert len(batch["images"]) == 2 and batch["images_spatial_crop"].tolist() == [[1, 1], [1, 1]]
