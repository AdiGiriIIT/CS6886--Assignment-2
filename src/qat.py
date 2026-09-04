"""Short QAT pilots, always initialized from the immutable FP32 checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from .compression import build_quantized_model, weight_size_breakdown
from .data import build_cifar10_loaders
from .models import build_model
from .train import run_epoch
from .utils import resolve_device, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--weight-bits", type=int, required=True); parser.add_argument("--activation-bits", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--metrics-file", default="results/tables/day1_qat_pilots.jsonl")
    args = parser.parse_args(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    set_seed(checkpoint["config"]["seed"]); device = resolve_device(args.device)
    base = build_model(checkpoint["model_config"], load_pretrained=False); base.load_state_dict(checkpoint["state_dict"])
    model = build_quantized_model(base, args.weight_bits, args.activation_bits).to(device)
    train_loader, val_loader = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", checkpoint["config"]["seed"])
    scales, weights = [], []
    for name, parameter in model.named_parameters(): (scales if name.endswith(".scale") else weights).append(parameter)
    optimizer = torch.optim.SGD([{"params": weights, "lr": args.learning_rate, "weight_decay": 4e-5}, {"params": scales, "lr": args.learning_rate, "weight_decay": 0.0}], momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    val_loss = val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.max_train_batches)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, max_batches=args.max_val_batches)
        print(f"epoch={epoch} train_accuracy={train_acc:.2f}% val_accuracy={val_acc:.2f}% train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    size = weight_size_breakdown(base, args.weight_bits)
    print(f"conservative_deployable_weight_bytes={size.compressed_bytes} fp32_weight_bytes={size.fp32_weight_bytes} ratio={size.ratio:.3f}x")
    metrics_path = Path(args.metrics_file); metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"checkpoint": args.checkpoint, "weight_bits": args.weight_bits, "activation_bits": args.activation_bits, "epochs": args.epochs, "learning_rate": args.learning_rate, "final_validation_loss": val_loss, "final_validation_accuracy": val_acc, "conservative_weight_bytes": size.compressed_bytes, "fp32_weight_bytes": size.fp32_weight_bytes}) + "\n")

if __name__ == "__main__": main()
