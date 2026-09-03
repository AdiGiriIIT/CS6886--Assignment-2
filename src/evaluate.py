"""Evaluate a saved CIFAR-10 MobileNetV2 checkpoint."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from .data import build_cifar10_loaders
from .models import build_model
from .train import run_epoch
from .utils import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    set_seed(config["seed"])
    device = resolve_device(args.device)
    _, test_loader = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers,
                                           device.type == "cuda", config["seed"])
    model = build_model(checkpoint["model_config"], load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loss, accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Checkpoint: {args.checkpoint}\nTest loss: {loss:.4f}\nTest top-1 accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
