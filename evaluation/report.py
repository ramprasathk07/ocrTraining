"""Scoring, before/after comparison, gates and the markdown report (no model, no GPU)."""

from collections import defaultdict

from evaluation.metrics import aer, cer, matra_error_rate, one_minus_ned, wer
from evaluation.page_metrics import aggregate, score_page

PAGE_TASK = "page_parse"
GATE_BOUNDS = ("max_value", "min_value", "max_absolute_change", "max_relative_change")
GATE_KEYS = {"group", "metric", *GATE_BOUNDS}

# (metric, heading, scale): values are shown × scale (100 = percent)
CROP_TABLES = (
    ("cer", "Crop CER % (lower is better)", 100),
    ("aer", "Crop AER, akshara error rate % (lower is better)", 100),
    ("wer", "Crop WER % (lower is better)", 100),
    ("matra_er", "Crop matra / diacritic error rate % (lower is better)", 100),
    ("one_minus_ned", "Crop 1 - NED % (higher is better)", 100),
    ("exact_match", "Crop exact match % (higher is better)", 100),
)
PAGE_TABLES = (
    ("page_cer", "Page CER % (lower is better)", 100),
    ("layout_ap50", "Layout AP@50, class-agnostic % (higher is better)", 100),
    ("layout_map50", "Layout mAP@50, class-aware % (higher is better)", 100),
    ("mean_iou", "Mean box IoU % (higher is better)", 100),
    ("block_f1", "Block F1@0.5 % (higher is better)", 100),
    ("reading_order", "Reading-order accuracy % (higher is better)", 100),
    ("valid_output", "Valid structured output % (higher is better)", 100),
)
LATENCY_TABLE = (
    "latency_s",
    "Mean latency, seconds per item (lower is better; batch 1 on the eval GPU)",
    1,
)
CROP_RATES = (  # (score key, function, nfc)
    ("cer", cer, False),
    ("cer_nfc", cer, True),
    ("aer", aer, False),
    ("wer", wer, False),
    ("wer_nfc", wer, True),
    ("matra_er", matra_error_rate, False),
    ("one_minus_ned", one_minus_ned, False),
)
NOT_MEASURED = (
    "Extraction field F1: the data has no field labels (key-value fields are outside V1 scope).",
    "Translation faithfulness: needs translated output (Stage 4); review it by hand then.",
)


def group_keys(prediction: dict) -> list[str]:
    script = prediction["script"]
    keys = ["all", f"source={prediction['source']}", f"script={script}"]
    for lang in prediction["langs"]:
        keys += [f"lang={lang}", f"script={script}|lang={lang}"]
    return keys


def _latency(predictions: list[dict]) -> float | None:
    """Mean generation seconds; None for predictions cached before timing was recorded."""
    seconds = [p["seconds"] for p in predictions if "seconds" in p]
    return sum(seconds) / len(seconds) if seconds else None


def _score_crops(predictions: list[dict]) -> dict[str, dict]:
    """Per group: the CROP_RATES, exact-match rate, empty-output rate, latency."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        for key in group_keys(p):
            groups[key].append(p)
    scores = {}
    for key, items in groups.items():
        pairs = [(p["target"], p["prediction"]) for p in items]
        entry = {
            "n": len(pairs),
            "exact_match": sum(t == h for t, h in pairs) / len(pairs),
            "empty_output": sum(bool(t.strip()) and not h.strip() for t, h in pairs) / len(pairs),
            "latency_s": _latency(items),
        }
        for metric, rate, nfc in CROP_RATES:
            try:
                entry[metric] = rate(pairs, nfc=nfc)
            except ValueError:  # e.g. every reference empty (blank stamps): no rate exists
                entry[metric] = None
        scores[key] = entry
    return scores


def score(predictions: list[dict]) -> dict[str, dict]:
    """Crop items: text metrics by source, script, language. Page items: page metrics by source."""
    crops = [p for p in predictions if p["task"] != PAGE_TASK]
    scores = _score_crops(crops)
    pages: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        if p["task"] == PAGE_TASK:
            pages[p["source"]].append(p)
    for source, items in pages.items():
        page_scores = [score_page(p["blocks"], p["prediction"], p["finished"]) for p in items]
        scores[f"source={source}"] = {**aggregate(page_scores), "latency_s": _latency(items)}
    return dict(sorted(scores.items()))


def validate_gates(gates: list[dict]) -> None:
    for gate in gates:
        unknown = set(gate) - GATE_KEYS
        if unknown or not {"group", "metric"} <= set(gate) or not set(gate) & set(GATE_BOUNDS):
            raise ValueError(f"bad gate {gate}: needs group, metric and one of {GATE_BOUNDS}")


def check_gate(gate: dict, before: dict, after: dict) -> dict:
    """One gate, comparing `after` scores with `before` scores (empty dict = no before)."""
    b = before.get(gate["group"], {}).get(gate["metric"])
    a = after.get(gate["group"], {}).get(gate["metric"])
    if a is None:
        return {**gate, "before": b, "after": a, "status": "skipped: group not evaluated"}
    failed, skipped = [], []
    if "max_value" in gate and a > gate["max_value"]:
        failed.append("max_value")
    if "min_value" in gate and a < gate["min_value"]:
        failed.append("min_value")
    for bound, change in (
        ("max_absolute_change", None if b is None else a - b),
        ("max_relative_change", None if not b else (a - b) / b),
    ):
        if bound not in gate:
            continue
        if change is None:
            skipped.append(bound)
        elif change > gate[bound]:
            failed.append(bound)
    status = f"fail: {', '.join(failed)}" if failed else "pass"
    if skipped and not failed:
        status += f" (no before score for {', '.join(skipped)})"
    return {**gate, "before": b, "after": a, "status": status}


def _item_cer(prediction: dict) -> float | None:
    try:
        return cer([(prediction["target"], prediction["prediction"])])
    except ValueError:
        return None


def biggest_changes(before: list[dict], after: list[dict], count: int = 10) -> dict[str, list]:
    """Crop items whose CER moved most between two checkpoints, for error analysis."""
    by_id = {p["id"]: p for p in before if p["task"] != PAGE_TASK}
    changes = []
    for p in after:
        old = by_id.get(p["id"])
        if old is None or (cer_before := _item_cer(old)) is None:
            continue
        changes.append(
            {
                "id": p["id"],
                "source": p["source"],
                "target": p["target"],
                "before": old["prediction"],
                "after": p["prediction"],
                "cer_before": cer_before,
                "cer_after": _item_cer(p),
            }
        )
    changes.sort(key=lambda c: c["cer_after"] - c["cer_before"])
    return {
        "improved": [c for c in changes[:count] if c["cer_after"] < c["cer_before"]],
        "regressed": [c for c in reversed(changes[-count:]) if c["cer_after"] > c["cer_before"]],
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def _short(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + "…"


def _table(
    metric: str, heading: str, scale: int, labels: list[str], scores: list[dict]
) -> list[str]:
    groups = sorted(
        g
        for g in set().union(*(s.keys() for s in scores))
        if any(s.get(g, {}).get(metric) is not None for s in scores)
    )
    if not groups:
        return []
    header = ["group", "n", *labels] + [f"Δ {x}" for x in labels[1:]]
    lines = ["", f"## {heading}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group in groups:
        values = [s.get(group, {}).get(metric) for s in scores]
        shown = ["—" if v is None else f"{scale * v:.2f}" for v in values]
        deltas = [
            "—" if v is None or values[0] is None else f"{scale * (v - values[0]):+.2f}"
            for v in values[1:]
        ]
        n = next((s[group]["n"] for s in scores if group in s), 0)
        lines.append("| " + " | ".join([group, str(n), *shown, *deltas]) + " |")
    return lines


def markdown_report(
    title: str,
    labels: list[str],
    scores: list[dict],
    gates: list[list[dict]],
    changes: list[dict],
    skipped_sources: list[str],
    prompts: dict[str, dict],
    prompt_warnings: list[str],
) -> str:
    """One page: prompts, metric tables per group, gates, and the biggest per-item changes."""
    lines = [f"# {title}", ""]
    lines.append(
        f"Before: `{labels[0]}`"
        + (f" · After: {', '.join(f'`{x}`' for x in labels[1:])}" if labels[1:] else "")
    )
    if skipped_sources:
        lines.append(f"\nSkipped sources (no records yet): {', '.join(skipped_sources)}")
    lines += ["", "## Prompts", "", "| task | version | prompt |", "|---|---|---|"]
    for task, used in prompts.items():
        text = used["prompt"].replace("\n", "\\n").replace("|", "\\|")
        lines.append(f"| {task} | `{used['version']}` | `{text}` |")
    lines += [f"\n**Prompt mismatch:** {warning}" for warning in prompt_warnings]
    for metric, heading, scale in (*CROP_TABLES, *PAGE_TABLES, LATENCY_TABLE):
        lines += _table(metric, heading, scale, labels, scores)
    if any(gates):
        lines += [
            "",
            "## Gates",
            "",
            "| checkpoint | group | metric | bound | before | after | status |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, label_gates in zip(labels[-len(gates) :], gates, strict=True):
            for g in label_gates:
                bounds = ", ".join(f"{k}={g[k]}" for k in GATE_BOUNDS if k in g)
                cells = [label, g["group"], g["metric"], bounds, _pct(g["before"])]
                cells += [_pct(g["after"]), g["status"]]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Not measured yet", ""] + [f"- {item}" for item in NOT_MEASURED]
    for label, change in zip(labels[1:], changes, strict=True):
        for kind in ("regressed", "improved"):
            lines += ["", f"## Most {kind}: {label} vs {labels[0]}", ""]
            for c in change[kind]:
                lines += [
                    f"- `{c['id']}` CER {_pct(c['cer_before'])} → {_pct(c['cer_after'])}",
                    f"  - target: {_short(c['target'])}",
                    f"  - before: {_short(c['before'])}",
                    f"  - after: {_short(c['after'])}",
                ]
    return "\n".join(lines) + "\n"
