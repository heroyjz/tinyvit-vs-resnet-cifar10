#!/usr/bin/env python3
"""
Create a 2x2 panel figure comparing ResNet-18 and TinyViT-5M runs.
Panels:
1. Train Accuracy
2. Val Accuracy
3. Train Loss
4. Val Loss

Each panel overlays four curves:
  - ResNet FT, ResNet Scratch, TinyViT FT, TinyViT Scratch
Colors are consistent per model family (ResNet/TinyViT); line styles distinguish FT vs Scratch.
Final epoch values are annotated on every curve.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-panel ResNet vs TinyViT plots")
    parser.add_argument("--resnet-ft", default="runs/resnet_ft")
    parser.add_argument("--resnet-scratch", default="runs/resnet_scratch")
    parser.add_argument("--tinyvit-ft", default="runs/tinyvit_ft")
    parser.add_argument("--tinyvit-scratch", default="runs/tinyvit_scratch")
    parser.add_argument(
        "--output",
        default="runs/resnet_tinyvit_four_panels.png",
        help="Output image path",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--safe", action="store_true", help="Force Agg backend")
    return parser.parse_args()


def ensure_backend(use_safe: bool) -> None:
    if use_safe:
        matplotlib.use("Agg")


def resolve_metrics_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_dir():
        candidate = path / "metrics.csv"
        if candidate.exists():
            return candidate
    if path.suffix == "":
        candidate = Path("runs") / path / "metrics.csv"
        if candidate.exists():
            return candidate
    if path.suffix != ".csv":
        candidate = path / "metrics.csv"
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    raise FileNotFoundError(f"Cannot locate metrics.csv for '{path_like}'")


def load_metrics(path_like: str) -> pd.DataFrame:
    csv_path = resolve_metrics_path(path_like)
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = (
            df.sort_values("epoch")
            .drop_duplicates(subset=["epoch"], keep="last")
            .reset_index(drop=True)
        )
    return df


def annotate_last(ax, x, y, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.5),
    )


def plot_metric_panel(
    ax: matplotlib.axes.Axes,
    runs: List[Tuple[str, pd.DataFrame]],
    column: str,
    ylabel: str,
    title: str,
    fmt: str,
) -> None:
    color_map = {"ResNet": "#1f77b4", "TinyViT": "#ff7f0e"}
    style_map = {"FT": "-", "Scratch": "--"}
    markers = {"FT": "o", "Scratch": "^"}

    for label, df in runs:
        if column not in df.columns:
            continue
        base = "ResNet" if "ResNet" in label else "TinyViT"
        mode = "FT" if "FT" in label or "Pre" in label else "Scratch"
        color = color_map[base]
        linestyle = style_map[mode]
        marker = markers[mode]
        x = df["epoch"].values
        y = df[column].values
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            alpha=0.9,
        )
        annotate_last(ax, x, y, fmt.format(y[-1]), color)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()


def main() -> None:
    args = parse_args()
    ensure_backend(args.safe)

    runs_data = [
        ("ResNet FT", load_metrics(args.resnet_ft)),
        ("ResNet Scratch", load_metrics(args.resnet_scratch)),
        ("TinyViT FT", load_metrics(args.tinyvit_ft)),
        ("TinyViT Scratch", load_metrics(args.tinyvit_scratch)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    plot_metric_panel(
        axes[0, 0],
        runs_data,
        "train_acc",
        "Accuracy (%)",
        "Train Accuracy",
        "{:.2f}",
    )
    plot_metric_panel(
        axes[0, 1],
        runs_data,
        "val_acc",
        "Accuracy (%)",
        "Validation Accuracy",
        "{:.2f}",
    )
    plot_metric_panel(
        axes[1, 0],
        runs_data,
        "train_loss",
        "Loss",
        "Train Loss",
        "{:.3f}",
    )
    plot_metric_panel(
        axes[1, 1],
        runs_data,
        "val_loss",
        "Loss",
        "Validation Loss",
        "{:.3f}",
    )

    fig.tight_layout()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved four-panel plot to {output_path}")
    if not args.safe:
        plt.show()


if __name__ == "__main__":
    main()
