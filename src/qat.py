"""Day-2 QAT sweep runner; every run starts from an immutable FP32 baseline."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
from .compression import build_quantized_model, weight_size_breakdown
from .data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from .models import build_model
from .quantization import set_quantizer_bits
from .train import run_epoch
from .utils import append_csv, resolve_device, save_history_plot, set_seed

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def git_commit() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): return "unavailable"

def freeze_batch_norm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval(); child.weight.requires_grad_(False); child.bias.requires_grad_(False)

def precision_for_epoch(epoch: int, target_w: int, target_a: int, transition_epochs: int) -> tuple[int, int]:
    """8 -> 6 -> target schedule, leaving most epochs at claimed precision."""
    if target_w >= 8 and target_a >= 8 or transition_epochs <= 0: return target_w, target_a
    if epoch <= transition_epochs: return 8, 8
    if epoch <= 2 * transition_epochs and (target_w < 6 or target_a < 6): return max(6, target_w), max(6, target_a)
    return target_w, target_a

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--weight-bits", type=int, required=True); parser.add_argument("--activation-bits", type=int, required=True)
    parser.add_argument("--edge-bits", type=int, help="Optional 8-bit input/logit exception.")
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--transition-epochs", type=int, default=1)
    parser.add_argument("--freeze-bn-epoch", type=int, default=10); parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=6886)
    parser.add_argument("--run-name", help="Default: w{W}a{A}-seed{seed}."); parser.add_argument("--runs-dir", default="experiments/sweeps")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints"); parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not 2 <= args.weight_bits <= 16 or not 2 <= args.activation_bits <= 16: raise ValueError("bit widths must be between 2 and 16")
    checkpoint_path, run_name = Path(args.checkpoint), args.run_name or f"w{args.weight_bits}a{args.activation_bits}-seed{args.seed}"
    run_dir = Path(args.runs_dir) / run_name
    if run_dir.exists(): raise FileExistsError(f"Refusing to overwrite immutable run record: {run_dir}")
    run_dir.mkdir(parents=True)
    resolved = vars(args) | {"run_name": run_name, "baseline_sha256": sha256(checkpoint_path), "git_commit": git_commit(),
                            "started_at_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
                            "python": platform.python_version(), "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text("python -m src.qat " + " ".join(__import__("sys").argv[1:]) + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(resolved["git_commit"] + "\n", encoding="utf-8"); (run_dir / "gpu.txt").write_text(resolved["gpu"] + "\n", encoding="utf-8")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False); set_seed(args.seed); device = resolve_device(args.device)
    base = build_model(source["model_config"], load_pretrained=False); base.load_state_dict(source["state_dict"])
    model = build_quantized_model(base, args.weight_bits, args.activation_bits, args.edge_bits).to(device)
    validation_size = source.get("validation_size", source.get("config", {}).get("training", {}).get("validation_size", 5_000))
    train_loader, val_loader, _ = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", args.seed, validation_size)
    scales, weights = [], []
    for name, parameter in model.named_parameters(): (scales if name.endswith(".scale") else weights).append(parameter)
    optimizer = torch.optim.SGD([{"params": weights, "lr": args.learning_rate, "weight_decay": 4e-5}, {"params": scales, "lr": args.learning_rate, "weight_decay": 0.0}], momentum=0.9, nesterov=True)
    scheduler, criterion = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs), nn.CrossEntropyLoss()
    history_path, best_target_accuracy = run_dir / "history.csv", -1.0
    # Warm-up precision is useful for optimization but must never be eligible
    # for selection as a checkpoint advertised at the requested target bits.
    best_target_checkpoint = Path(args.checkpoint_dir) / f"qat-{run_name}-best-target.pt"
    latest_checkpoint = Path(args.checkpoint_dir) / f"qat-{run_name}-latest.pt"
    best_target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        current_w, current_a = precision_for_epoch(epoch, args.weight_bits, args.activation_bits, args.transition_epochs); set_quantizer_bits(model, current_w, current_a)
        if args.freeze_bn_epoch and epoch >= args.freeze_bn_epoch: freeze_batch_norm(model)
        started = perf_counter(); train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.max_train_batches, keep_batch_norm_eval=bool(args.freeze_bn_epoch and epoch >= args.freeze_bn_epoch)); val_loss, val_acc = run_epoch(model, val_loader, criterion, device, max_batches=args.max_val_batches); scheduler.step()
        append_csv(history_path, {"epoch": epoch, "weight_bits": current_w, "activation_bits": current_a, "train_loss": f"{train_loss:.6f}", "train_accuracy": f"{train_acc:.3f}", "val_loss": f"{val_loss:.6f}", "val_accuracy": f"{val_acc:.3f}", "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}"})
        at_target_precision = current_w == args.weight_bits and current_a == args.activation_bits
        state = {"state_dict": model.state_dict(), "model_config": source["model_config"], "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}, "qat_config": resolved, "quantization_state": {"weight_bits": current_w, "activation_bits": current_a, "edge_bits": args.edge_bits}, "epoch": epoch, "validation_accuracy": val_acc, "validation_size": validation_size, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_target_validation_accuracy": max(best_target_accuracy, val_acc) if at_target_precision else best_target_accuracy}
        torch.save(state, latest_checkpoint)
        if at_target_precision and val_acc > best_target_accuracy:
            best_target_accuracy = val_acc
            shutil.copy2(latest_checkpoint, best_target_checkpoint)
        print(f"epoch={epoch:02d}/{args.epochs} W{current_w}A{current_a} train={train_acc:.2f}% validation={val_acc:.2f}% loss={val_loss:.4f} elapsed={perf_counter() - started:.1f}s")
    if best_target_accuracy < 0:
        raise RuntimeError("The requested target precision was never reached; increase --epochs or shorten --transition-epochs.")
    size = weight_size_breakdown(base, args.weight_bits)
    plot_path = run_dir / "history.png"
    save_history_plot(history_path, plot_path)
    metrics = {"run_name": run_name, "baseline_sha256": resolved["baseline_sha256"], "weight_bits": args.weight_bits, "activation_bits": args.activation_bits, "edge_bits": args.edge_bits, "epochs": args.epochs, "validation_size": validation_size, "best_target_validation_accuracy": best_target_accuracy, "packed_weight_bytes": size.packed_weight_bytes, "weight_scale_bytes": size.weight_scale_bytes, "bias_bytes": size.bias_bytes, "descriptor_bytes": size.descriptor_bytes, "compressed_weight_bytes": size.compressed_bytes, "fp32_weight_bytes": size.fp32_weight_bytes, "weight_compression_ratio": size.ratio, "best_target_checkpoint": str(best_target_checkpoint), "latest_checkpoint": str(latest_checkpoint), "history_plot": str(plot_path)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    artifacts = [{"logical_name": label, "location": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for label, path in (("best_target_qat_checkpoint", best_target_checkpoint), ("latest_qat_checkpoint", latest_checkpoint))]
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(f"best_target_validation_accuracy={best_target_accuracy:.2f}% weight_ratio={size.ratio:.3f}x weight_bytes={size.compressed_bytes} run_record={run_dir}")

if __name__ == "__main__": main()
