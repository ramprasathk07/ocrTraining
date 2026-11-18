"""Generate on one split and score it. The same command runs on 1 or N GPUs:

    torchrun --nproc_per_node=N -m evaluation --config configs/stage1_script.toml [--checkpoint DIR]

Without --checkpoint it scores the base model (the Stage 0 baseline). A folder holding
adapter_config.json loads as base + LoRA; any other folder loads as a merged model.
Items are sharded across ranks; rank 0 gathers them and writes predictions.jsonl + scores.json.
"""

import argparse
import json
import logging
import random
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from accelerate import PartialState
from accelerate.utils import gather_object

from dataEngine.records import read_crops
from dataEngine.tasks import get_task
from evaluation.metrics import cer, wer
from models.deepseek_ocr.generate import generate_text
from models.deepseek_ocr.inputs import ImageConfig
from models.deepseek_ocr.loader import ModelConfig, load_model, load_tokenizer

log = logging.getLogger("evaluation")


def select_items(raw: dict, split: str) -> list[dict]:
    """A fixed, seeded subset of each configured source's split."""
    settings = raw["eval"]
    sources = {s["name"]: s for s in raw["data"]["sources"]}
    items = []
    for name in settings["sources"]:
        source = sources[name]
        root, task = Path(source["path"]), get_task(source["task"])
        records = [r for r in read_crops(root) if r.split == split]
        random.Random(settings["seed"]).shuffle(records)
        for record in records[: settings["limit_per_source"]]:
            sample = task(record, root)
            items.append(
                {
                    "id": record.id,
                    "source": name,
                    "script": record.script,
                    "langs": list(record.langs),
                    "prompt": sample.prompt,
                    "image": str(sample.image),
                    "target": sample.target,
                }
            )
    return items


def score(predictions: list[dict]) -> dict:
    """CER/WER (raw and NFC) overall and per source, script and language."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p in predictions:
        pair = (p["target"], p["prediction"])
        keys = ["all", f"source={p['source']}", f"script={p['script']}"] + [
            f"lang={g}" for g in p["langs"]
        ]
        for key in keys:
            groups[key].append(pair)
    scores = {}
    for key, pairs in sorted(groups.items()):
        try:
            scores[key] = {
                "n": len(pairs),
                "cer": cer(pairs),
                "cer_nfc": cer(pairs, nfc=True),
                "wer": wer(pairs),
                "wer_nfc": wer(pairs, nfc=True),
            }
        except ValueError:  # only empty references (e.g. blank stamps): no rate exists
            scores[key] = {
                "n": len(pairs),
                "cer": None,
                "cer_nfc": None,
                "wer": None,
                "wer_nfc": None,
            }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Trainer checkpoint or merged model folder")
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev", help="test only for M1/M2 reports"
    )
    args = parser.parse_args()
    state = PartialState()
    logging.basicConfig(level=logging.INFO if state.is_main_process else logging.WARNING)
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    settings = raw["eval"]

    model_cfg, image_cfg = ModelConfig(**raw["model"]), ImageConfig(**raw["image"])
    adapter = (
        args.checkpoint
        if args.checkpoint and (args.checkpoint / "adapter_config.json").is_file()
        else None
    )
    if args.checkpoint and adapter is None:
        model_cfg = replace(model_cfg, name_or_path=str(args.checkpoint))
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, adapter).to(state.device).eval()

    items = select_items(raw, args.split)
    with state.split_between_processes(items) as shard:
        predictions = [
            {
                **item,
                "prediction": generate_text(
                    model,
                    tokenizer,
                    item["prompt"],
                    Path(item["image"]),
                    image_cfg,
                    settings["max_new_tokens"],
                    settings["no_repeat_ngram_size"],
                ),
            }
            for item in shard
        ]
    predictions = gather_object(predictions)

    if state.is_main_process:
        out_dir = (
            args.checkpoint or Path(raw["train"]["output_dir"]) / "baseline"
        ) / f"eval_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions)
        scores = score(predictions)
        (out_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        log.info("all: %s -> %s", scores["all"], out_dir)


if __name__ == "__main__":
    main()
