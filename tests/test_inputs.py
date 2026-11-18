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
SMALL = ImageConfig(base_size=640, image_size=640, crop_mode=False)
GUNDAM = ImageConfig(base_size=1024, image_size=640, crop_mode=True)


class StubTokenizer:
    """One id per character, far from the special ids."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + ord(c) for c in text]


def test_small_mode_layout_and_labels():
    item = encode_sample(PROMPT, "अब", Image.new("RGB", (300, 60), "white"), StubTokenizer(), SMALL)
    ids, labels = item["input_ids"].tolist(), item["labels"].tolist()
    n_image = 111  # 10 rows × (10 tokens + row end) + view separator
    prompt_len = 1 + n_image + len("\nFree OCR.")

    assert ids[0] == BOS_ID
    assert ids[1 : 1 + n_image] == [IMAGE_TOKEN_ID] * n_image
    assert int(item["images_seq_mask"].sum()) == n_image
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == [1000 + ord("अ"), 1000 + ord("ब"), EOS_ID]
    assert item["global_view"].shape == (1, 3, 640, 640)
    assert not item[
        "patches"
    ].any()  # no tiles → zeros, which the model reads as "global view only"


def test_generation_prompt_has_no_target():
    item = encode_sample(PROMPT, None, Image.new("RGB", (300, 60)), StubTokenizer(), SMALL)
    assert (item["labels"] == IGNORE_INDEX).all()
    assert item["input_ids"][-1] != EOS_ID


def test_gundam_tiles_a_full_page():
    encoded = encode_image(Image.new("RGB", (1698, 2325)), GUNDAM)  # a golden FIR page size
    assert encoded.tiles == (2, 3)
    assert encoded.patches.shape == (6, 3, 640, 640)
    assert encoded.num_tokens == 17 * 16 + 1 + 21 * 30  # global 16×(16+1)+1, tiles (10·2+1)×(10·3)


def test_gundam_small_crop_uses_global_view_only():
    encoded = encode_image(Image.new("RGB", (300, 60)), GUNDAM)
    assert encoded.tiles == (1, 1)
    assert encoded.num_tokens == 273
    assert encoded.global_view.shape == (1, 3, 1024, 1024)


def test_wide_image_gets_wide_grid():
    assert encode_image(Image.new("RGB", (3000, 1000)), GUNDAM).tiles == (3, 1)


def test_crop_mode_needs_640_tiles():
    with pytest.raises(ValueError):
        ImageConfig(base_size=1024, image_size=1024, crop_mode=True)


def test_to_tensor_normalises_to_minus_one_one():
    tensor = to_tensor(Image.new("RGB", (2, 1), "white"))
    assert tensor.dtype == torch.bfloat16 and tensor.shape == (3, 1, 2)
    assert float(tensor.max()) == 1.0
    assert float(to_tensor(Image.new("RGB", (2, 1), "black")).min()) == -1.0


def test_collate_right_pads_every_field():
    image = Image.new("RGB", (300, 60))
    short = encode_sample(PROMPT, "अ", image, StubTokenizer(), SMALL)
    long = encode_sample(PROMPT, "अबक", image, StubTokenizer(), SMALL)
    batch = collate([short, long], pad_id=2)

    assert batch["input_ids"].shape == (2, len(long["input_ids"]))
    assert batch["input_ids"][0, -2:].tolist() == [2, 2]
    assert batch["labels"][0, -2:].tolist() == [IGNORE_INDEX] * 2
    assert batch["attention_mask"][0].tolist().count(0) == 2
    assert not batch["images_seq_mask"][0, -2:].any()
    assert len(batch["images"]) == 2 and batch["images_spatial_crop"].tolist() == [[1, 1], [1, 1]]
