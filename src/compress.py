"""Inspect a checkpoint and record a requested custom-compression configuration."""

from __future__ import annotations

import argparse

import torch

from .compression import checkpoint_size_mb, count_parameter_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--weight-bits", "--weight_bits", dest="weight_bits", type=int, required=True)
    parser.add_argument("--activation-bits", "--activation_bits", dest="activation_bits", type=int, required=True)
    args = parser.parse_args()
    if not (1 <= args.weight_bits <= 32 and 1 <= args.activation_bits <= 32):
        parser.error("bit widths must be between 1 and 32")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    baseline_bits = count_parameter_bits(checkpoint["state_dict"], 32)
    requested_bits = count_parameter_bits(checkpoint["state_dict"], args.weight_bits)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Serialized baseline checkpoint: {checkpoint_size_mb(args.checkpoint):.2f} MB")
    print(f"Weight payload estimate: {baseline_bits / 8 / 1024 ** 2:.2f} MB at fp32")
    print(f"Requested payload estimate: {requested_bits / 8 / 1024 ** 2:.2f} MB at {args.weight_bits}-bit")
    print(f"Ideal weight-payload ratio (metadata excluded): {baseline_bits / requested_bits:.2f}x")
    print(f"Activation bit width requested: {args.activation_bits}")
    print("No model was modified: custom quantization will be implemented in src/compression.py.")


if __name__ == "__main__":
    main()
