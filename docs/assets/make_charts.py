"""README charts: Stage 1A training curves (from W&B) and public dev-set scores per checkpoint.

    python docs/assets/make_charts.py --run <entity>/<project>/<run id> \
        --eval start=outputs/eval/test_data/start/dev/predictions.jsonl \
        --eval 400=outputs/runs/stage1a_printed_colab/checkpoint-400/eval/dev/predictions.jsonl ...

Only the public sources in PUBLIC are scored (Mozhi and IIIT-HW dev words), so nothing private
reaches the repo. Writes training.png, eval.png (+ -dark variants) and stage1a_dev_eval.json here.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.report import score

OUT = Path(__file__).resolve().parent
PUBLIC = {"printed_words": "Mozhi printed (dev)", "iiit_hw_dev": "IIIT-HW handwritten (dev)"}
METRICS = (("exact_match", "Exact match %"), ("aer", "AER %"), ("one_minus_ned", "1 - NED %"))
# dataviz reference palette, categorical slots 1-2 (validated light + dark)
THEMES = {
    "": {"surface": "#ffffff", "ink": "#191919", "ink2": "#666666", "grid": "#e6e6e6",
         "series": ("#257de0", "#f47230")},
    "-dark": {"surface": "#191919", "ink": "#efefef", "ink2": "#9c9c9c", "grid": "#333333",
              "series": ("#338ced", "#e76624")},
}  # fmt: skip


def history(run_path: str) -> dict[str, list[tuple[int, float]]]:
    import wandb

    keys = ("train/loss", "train/learning_rate", "train/grad_norm", "eval/loss")
    series: dict[str, dict[int, float]] = {k: {} for k in keys}
    for row in wandb.Api().run(run_path).scan_history():
        step = row.get("train/global_step")
        for key in keys:
            if step is not None and row.get(key) is not None:
                series[key][int(step)] = float(row[key])  # a resumed run re-logs a step: keep last
    return {k: sorted(v.items()) for k, v in series.items()}


def eval_scores(evals: list[str]) -> list[dict]:
    rows = []
    for spec in evals:
        label, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as f:
            preds = [p for p in map(json.loads, f) if p["source"] in PUBLIC]
        scores = score(preds)
        step = 0 if label == "start" else int(label)
        for source, name in PUBLIC.items():
            s = scores[f"source={source}"]
            keys = ("cer", "aer", "wer", "matra_er", "one_minus_ned", "exact_match")
            rows.append({"step": step, "source": name, "n": s["n"], **{k: s[k] for k in keys}})
    return rows


def _style(ax, t: dict, title: str) -> None:
    ax.set_facecolor(t["surface"])
    ax.set_title(title, loc="left", color=t["ink"], fontsize=11)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.tick_params(colors=t["ink2"], labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.set_xlabel("training step", color=t["ink2"], fontsize=9)


def _label_end(ax, xs, ys, text: str, color: str, t: dict) -> None:
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=t["ink"])  # fmt: skip
    ax.plot(xs[-1], ys[-1], "o", ms=8, color=color, mec=t["surface"], mew=2)


def training_chart(h: dict, t: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=t["surface"])
    blue, orange = t["series"]
    loss = h["train/loss"]
    ax = axes[0]
    ax.plot(*zip(*loss, strict=False), color=blue, lw=2, label="train loss")
    _label_end(ax, *zip(*loss, strict=False), "train", blue, t)
    if h["eval/loss"]:
        ev = h["eval/loss"]
        dot = {"marker": "o", "ms": 8, "mec": t["surface"], "mew": 2}
        ax.plot(*zip(*ev, strict=False), color=orange, lw=2, label="dev loss", **dot)
        _label_end(ax, *zip(*ev, strict=False), "dev", orange, t)
    ax.legend(frameon=False, labelcolor=t["ink2"], fontsize=9)
    _style(ax, t, "Loss")
    for ax, key, title in ((axes[1], "train/learning_rate", "Learning rate (decoder LoRA)"),
                           (axes[2], "train/grad_norm", "Gradient norm")):  # fmt: skip
        ax.plot(*zip(*h[key], strict=False), color=blue, lw=2)
        _style(ax, t, title)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].yaxis.get_offset_text().set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def eval_chart(rows: list[dict], t: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 3.6), facecolor=t["surface"])
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for color, name in zip(t["series"], PUBLIC.values(), strict=True):
            pts = sorted((r["step"], 100 * r[metric]) for r in rows if r["source"] == name)
            marks = {"marker": "o", "ms": 8, "mec": t["surface"], "mew": 2, "color": color}
            # dotted: nothing was measured between the base model and the first checkpoint
            ax.plot(*zip(*pts[:2], strict=False), lw=1.5, ls=":", **marks)
            xs, ys = zip(*pts[1:], strict=False)
            ax.plot(xs, ys, lw=2, label=name, **marks)
            _label_end(ax, xs, ys, f"{ys[-1]:.1f}", color, t)
        if metric == "aer":
            ax.set_yscale("log")  # the base model is far above 100 %
        _style(ax, t, title)
        steps = sorted({r["step"] for r in rows})
        ax.set_xticks(steps, ["base" if s == 0 else str(s) for s in steps])
        ax.set_xlabel("checkpoint (training step)", color=t["ink2"], fontsize=9)
    axes[0].legend(frameon=False, labelcolor=t["ink2"], fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="W&B <entity>/<project>/<run id>")
    parser.add_argument(
        "--eval", action="append", default=[], help="<step|start>=<predictions.jsonl>"
    )
    args = parser.parse_args()
    h = history(args.run)
    rows = eval_scores(args.eval)
    (OUT / "stage1a_dev_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for suffix, theme in THEMES.items():
        training_chart(h, theme, OUT / f"training{suffix}.png")
        eval_chart(rows, theme, OUT / f"eval{suffix}.png")
    last = h["train/loss"][-1]
    print(f"train loss at step {last[0]}: {last[1]:.4f}; dev loss points: {h['eval/loss']}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
