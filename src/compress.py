"""Run Day-1 quantization coverage/accounting diagnostics on a checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from .compression import build_quantized_model, checkpoint_size_mb, coverage_rows, weight_size_breakdown
from .data import build_cifar10_loaders
from .models import build_model
from .train import run_epoch
from .utils import resolve_device, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--weight-bits", type=int, required=True); parser.add_argument("--activation-bits", type=int, required=True)
    parser.add_argument("--coverage-file", default="results/tables/quantization_coverage.json")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the fake-quantized PTQ diagnostic.")
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = build_model(checkpoint["model_config"], load_pretrained=False); base.load_state_dict(checkpoint["state_dict"])
    rows = coverage_rows(base, args.weight_bits, args.activation_bits)
    Path(args.coverage_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.coverage_file).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    size = weight_size_breakdown(base, args.weight_bits)
    print(f"Checkpoint: {args.checkpoint}\nSerialized training artifact: {checkpoint_size_mb(args.checkpoint):.2f} MiB")
    print(f"Coverage: {len(rows)} Conv2d/Linear tensors; written to {args.coverage_file}")
    print(f"Conservative deployable weight estimate: {size.compressed_bytes} bytes ({size.compressed_bytes / 2**20:.3f} MiB)")
    print(f"FP32 deployable weight state: {size.fp32_weight_bytes} bytes; ratio: {size.ratio:.3f}x")
    print("Known Day-1 limitation: residual-add/input/projection boundaries are not yet requantized; do not use this as a final deployment result.")
    if args.evaluate:
        set_seed(checkpoint["config"]["seed"]); device = resolve_device(args.device)
        model = build_quantized_model(base, args.weight_bits, args.activation_bits).to(device).eval()
        _, _, test_loader = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", checkpoint["config"]["seed"])
        loss, accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"PTQ diagnostic test loss: {loss:.4f}; top-1 accuracy: {accuracy:.2f}%")

if __name__ == "__main__": main()
