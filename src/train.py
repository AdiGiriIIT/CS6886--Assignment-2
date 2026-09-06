"""Train the CIFAR-10 MobileNetV2 baseline from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from .data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from .models import build_model, freeze_backbone
from .utils import append_csv, ensure_parent, load_config, resolve_device, save_history_plot, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches: int | None = None,
              keep_batch_norm_eval: bool = False) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    if is_training and keep_batch_norm_eval:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = total_correct = total_examples = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)
    if total_examples == 0:
        raise ValueError("No batches were processed; increase --max-*-batches.")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        help="Override config device; use cuda to fail fast without a GPU.")
    parser.add_argument("--epochs", type=int, help="Override YAML epoch count.")
    parser.add_argument("--run-name", help="Override output.run_name.")
    parser.add_argument("--data-dir", help="Override data_dir from the YAML configuration.")
    parser.add_argument("--max-train-batches", type=int, help="Limit batches for a smoke test.")
    parser.add_argument("--max-val-batches", type=int, help="Limit test batches for a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.run_name is not None:
        config["output"]["run_name"] = args.run_name
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    train_loader, val_loader, _ = build_cifar10_loaders(
        config["data_dir"], config["training"]["batch_size"], config["num_workers"],
        config["pin_memory"] and device.type == "cuda", config["seed"], config["training"].get("validation_size", 5_000),
    )
    model = build_model(config["model"]).to(device)
    train_cfg = config["training"]
    optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["learning_rate"],
                                momentum=train_cfg["momentum"], weight_decay=train_cfg["weight_decay"],
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 0) if config["model"].get("pretrained", False) else 0
    if freeze_epochs:
        freeze_backbone(model, freeze=True)
        print(f"Using ImageNet-pretrained MobileNetV2: backbone frozen for {freeze_epochs} warm-up epoch(s).")
    output = config["output"]
    run_name = output["run_name"]
    history_path = Path(output["curve_dir"]) / f"{run_name}.csv"
    checkpoint_path = Path(output["checkpoint_dir"]) / f"{run_name}.pt"
    history_path.unlink(missing_ok=True)
    best_accuracy = -1.0
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, train_cfg["epochs"] + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs:
            freeze_backbone(model, freeze=False)
            print("Backbone unfrozen; fine-tuning all MobileNetV2 layers.")
        start = perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, args.max_train_batches)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, max_batches=args.max_val_batches)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "train_accuracy": f"{train_accuracy:.3f}",
               "val_loss": f"{val_loss:.6f}", "val_accuracy": f"{val_accuracy:.3f}",
               "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}"}
        append_csv(history_path, row)
        print(f"Epoch {epoch:03d}/{train_cfg['epochs']} | train {train_accuracy:.2f}% | validation {val_accuracy:.2f}% | {perf_counter() - start:.1f}s")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            ensure_parent(checkpoint_path)
            torch.save({"state_dict": model.state_dict(), "model_config": config["model"],
                        "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}, "config": config,
                        "epoch": epoch, "validation_accuracy": val_accuracy,
                        "validation_size": train_cfg.get("validation_size", 5_000)}, checkpoint_path)
    plot_path = Path(output["curve_dir"]) / f"{run_name}.png"
    save_history_plot(history_path, plot_path)
    append_csv(output["metrics_file"], {"run_name": run_name, "best_validation_accuracy": f"{best_accuracy:.3f}",
                                         "checkpoint": str(checkpoint_path), "epochs": train_cfg["epochs"], "seed": config["seed"]})
    print(f"Best validation accuracy: {best_accuracy:.2f}%\nCheckpoint: {checkpoint_path}\nCurves: {plot_path}")


if __name__ == "__main__":
    main()
