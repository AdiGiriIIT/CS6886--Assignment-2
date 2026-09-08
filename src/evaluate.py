"""Evaluate a saved CIFAR-10 MobileNetV2 checkpoint."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from .compression import build_quantized_model
from .data import build_cifar10_test_loader
from .models import build_model
from .pruning import enforce_masks, validate_masks
from .quantization import (ActivationFakeQuantizer, apply_quantizer_bit_overrides,
                           apply_weight_bit_map)
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
    config = checkpoint.get("config", checkpoint.get("qat_config", {}))
    set_seed(config.get("seed", 6886))
    device = resolve_device(args.device)
    test_loader = build_cifar10_test_loader(args.data_dir, args.batch_size, args.num_workers,
                                            device.type == "cuda")
    if "qat_config" in checkpoint:
        quantization = checkpoint.get("quantization_state", checkpoint["qat_config"])
        model = build_quantized_model(
            build_model(checkpoint["model_config"], load_pretrained=False),
            quantization["weight_bits"], quantization["activation_bits"], quantization.get("edge_bits"),
        ).to(device)
        if quantization.get("realized_weight_bits"):
            apply_weight_bit_map(model, quantization["realized_weight_bits"])
        if quantization.get("activation_overrides"):
            apply_quantizer_bit_overrides(model, quantization["activation_overrides"],
                                          ActivationFakeQuantizer)
    else:
        model = build_model(checkpoint["model_config"], load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("pruning_masks"):
        enforce_masks(model, checkpoint["pruning_masks"])
        validate_masks(model, checkpoint["pruning_masks"])
    loss, accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Checkpoint: {args.checkpoint}\nTest loss: {loss:.4f}\nTest top-1 accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
