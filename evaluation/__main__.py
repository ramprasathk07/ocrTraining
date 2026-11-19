"""Before/after evaluation. The same command runs on 1 or N GPUs:

    torchrun --nproc_per_node=N -m evaluation --config configs/eval/test_data.toml --split test \\
        --checkpoint start [--checkpoint outputs/runs/stage1a_printed/merged ...]

`start` is the config's starting model ([model].name_or_path). A folder with
adapter_config.json loads as start model + LoRA; any other folder, or a hub id such as
deepseek-ai/DeepSeek-OCR-2 (at [model].revision), loads as a full model.
The first --checkpoint is "before"; every later one is compared with it.

Each [[eval.sources]] gives its task, sample size (`limit`), token limit and, optionally,
its own [image] mode. Crop tasks read a crop-record folder; page tasks read pages.jsonl,
the split files and the image root. Items are a fixed, seeded sample, sharded across ranks.
Per checkpoint: predictions.jsonl + scores.json (reused while nothing changed).
Rank 0 writes [eval].report_dir/report_<split>.md and .json, with gates. Predictions (raw
FIR text) stay under [eval].output_dir; the report holds scores only, plus examples when
comparing checkpoints.
"""

import argparse
import hashlib
import json
import logging
import random
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import PartialState
from accelerate.utils import gather_object

from dataEngine.tasks import get_task, page_blocks, read_split
from evaluation.prompts import drift_warnings, prompts_used, record_run
from evaluation.report import biggest_changes, check_gate, markdown_report, score, validate_gates
from models.deepseek_ocr.generate import generate_text
from models.deepseek_ocr.inputs import ImageConfig
from models.deepseek_ocr.loader import ModelConfig, load_model, load_tokenizer

log = logging.getLogger("evaluation")
START = "start"
SOURCE_KEYS = {"name", "task", "path", "limit", "max_new_tokens", "image", "root", "splits"}


def select_items(raw: dict, split: str) -> tuple[list[dict], list[str]]:
    """Seeded sample per eval source; sources without records yet are skipped and named."""
    settings, items, skipped = raw["eval"], [], []
    for source in settings["sources"]:
        if unknown := set(source) - SOURCE_KEYS:
            raise ValueError(f"eval source {source.get('name')}: unknown keys {sorted(unknown)}")
        task = get_task(source["task"])
        splits = Path(source["splits"]) if "splits" in source else None
        records = read_split(task, Path(source["path"]), split, splits)
        if records is None:
            skipped.append(source["name"])
            continue
        root = Path(source["root"] if task.records == "pages" else source["path"])
        image_cfg = source.get("image", raw["image"])
        ImageConfig(**image_cfg)  # fail early on an unsupported mode
        random.Random(settings["seed"]).shuffle(records)
        for record in records[: source["limit"]]:
            sample = task(record, root)
            pages = task.records == "pages"
            item = {
                "id": record.id,
                "source": source["name"],
                "task": source["task"],
                "script": "page" if pages else record.script,
                "langs": sorted(record.langs),
                "prompt": sample.prompt,
                "image": str(sample.image),
                "target": sample.target,
                "image_cfg": image_cfg,
                "max_new_tokens": source["max_new_tokens"],
            }
            if pages:
                item["blocks"] = [
                    {"label": b.label, "box": list(b.box), "text": b.text}
                    for b in page_blocks(record)
                ]
            items.append(item)
    return items, skipped


def eval_dir(checkpoint: str, raw: dict, split: str) -> Path:
    """Local checkpoints keep their results inside; `start` and hub ids go under the run."""
    if checkpoint != START and Path(checkpoint).is_dir():
        return Path(checkpoint) / "eval" / split
    return Path(raw["eval"]["output_dir"]) / checkpoint.replace("/", "__") / split


def label(checkpoint: str) -> str:
    return checkpoint if checkpoint == START else "/".join(Path(checkpoint).parts[-2:])


def fingerprint(items: list[dict], raw: dict, checkpoint: str) -> str:
    """Changes when items, image modes, token limits, start model or weight files change."""
    weights = [] if checkpoint == START else sorted(Path(checkpoint).glob("*.safetensors"))
    key = {
        "items": [(i["id"], i["prompt"], i["image_cfg"], i["max_new_tokens"]) for i in items],
        "model": raw["model"],
        "weights": [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in weights],
        "no_repeat_ngram_size": raw["eval"]["no_repeat_ngram_size"],
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _is_adapter(checkpoint: str) -> bool:
    return (Path(checkpoint) / "adapter_config.json").is_file()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_checkpoint(checkpoint: str, items: list[dict], raw: dict, split: str, state) -> list[dict]:
    """Predictions for every item, from cache when the fingerprint still matches."""
    out_dir, key = eval_dir(checkpoint, raw, split), fingerprint(items, raw, checkpoint)
    if (out_dir / "fingerprint.txt").is_file() and (out_dir / "fingerprint.txt").read_text() == key:
        log.info("%s: reusing predictions in %s", label(checkpoint), out_dir)
        return _read_jsonl(out_dir / "predictions.jsonl")

    model_cfg = ModelConfig(**raw["model"])
    adapter = None
    if checkpoint != START:
        if _is_adapter(checkpoint):
            adapter = Path(checkpoint)
        else:
            model_cfg = replace(model_cfg, name_or_path=checkpoint)
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, adapter).to(state.device).eval()
    no_repeat = raw["eval"]["no_repeat_ngram_size"]
    predictions, started = [], time.perf_counter()
    with state.split_between_processes(items) as shard:
        for done, item in enumerate(shard):
            if done % 10 == 0:
                elapsed = time.perf_counter() - started
                log.info("%s: %d/%d items, %.0fs", label(checkpoint), done, len(shard), elapsed)
            text, finished = generate_text(
                model,
                tokenizer,
                item["prompt"],
                Path(item["image"]),
                ImageConfig(**item["image_cfg"]),
                item["max_new_tokens"],
                no_repeat,
            )
            predictions.append({**item, "prediction": text, "finished": finished})
    predictions = gather_object(predictions)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions)
        (out_dir / "scores.json").write_text(json.dumps(score(predictions), indent=2), "utf-8")
        (out_dir / "fingerprint.txt").write_text(key, encoding="utf-8")
    state.wait_for_everyone()
    return predictions


def write_report(
    raw: dict, split: str, checkpoints: list[str], predictions: list[list[dict]], skipped: list[str]
) -> Path:
    """Scores, gates and biggest changes for all checkpoints → report_<split>.md/.json."""
    gates = raw["eval"].get("gates", [])
    labels = [label(c) for c in checkpoints]
    scores = [score(p) for p in predictions]
    afters = scores[1:] or scores  # a single checkpoint can still meet value gates
    gate_results = [
        [check_gate(g, scores[0] if scores[1:] else {}, s) for g in gates] for s in afters
    ]
    changes = [biggest_changes(predictions[0], p) for p in predictions[1:]]
    used_sources = [s for s in raw["eval"]["sources"] if s["name"] not in skipped]
    prompts = prompts_used(used_sources)
    paths = [raw["model"]["name_or_path"] if c == START else c for c in checkpoints]
    warnings = drift_warnings(paths, prompts)
    out_dir = Path(raw["eval"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Evaluation: {raw['name']} ({split}, {len(predictions[0])} items)"
    report = markdown_report(
        title, labels, scores, gate_results, changes, skipped, prompts, warnings
    )
    (out_dir / f"report_{split}.md").write_text(report, encoding="utf-8")
    summary = {
        "checkpoints": labels,
        "prompts": prompts,
        "prompt_warnings": warnings,
        "scores": scores,
        "gates": gate_results,
        "skipped": skipped,
    }
    (out_dir / f"report_{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record_run(Path(raw["eval"]["prompts_dir"]), raw, split, labels, scores, prompts)
    for warning in warnings:
        log.warning("prompt mismatch: %s", warning)
    failed = sum(g["status"].startswith("fail") for row in gate_results for g in row)
    log.info("report -> %s (%d gate(s) failed)", out_dir / f"report_{split}.md", failed)
    return out_dir / f"report_{split}.md"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        help=f"'{START}', a checkpoint folder or a hub id; repeat it, the first is the 'before'",
    )
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev", help="test only for reports"
    )
    args = parser.parse_args(argv)
    state = PartialState()
    logging.basicConfig(level=logging.INFO if state.is_main_process else logging.WARNING)
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    validate_gates(raw["eval"].get("gates", []))
    checkpoints = args.checkpoint or [START]

    items, skipped = select_items(raw, args.split)
    if not items:
        raise SystemExit("no evaluation items: convert at least one [[eval.sources]] first")
    predictions = [run_checkpoint(c, items, raw, args.split, state) for c in checkpoints]
    if state.is_main_process:
        write_report(raw, args.split, checkpoints, predictions, skipped)


if __name__ == "__main__":
    main()
