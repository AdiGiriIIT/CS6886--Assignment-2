"""Shared reproducibility, configuration, and result-writing helpers."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use device: auto or cpu.")
    return device


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_csv(path: str | Path, row: dict) -> None:
    path = ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_history_plot(history_path: str | Path, plot_path: str | Path) -> None:
    """Save loss and accuracy curves without requiring an interactive display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="test")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [float(row["train_accuracy"]) for row in rows], label="train")
    axes[1].plot(epochs, [float(row["val_accuracy"]) for row in rows], label="test")
    axes[1].set(title="Top-1 accuracy", xlabel="Epoch", ylabel="Percent")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    ensure_parent(plot_path)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
