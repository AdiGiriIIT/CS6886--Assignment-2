"""One-shot W4-pointwise pruning followed by fixed-mask QAT + KD recovery.

This runner never treats a zero-filled dense QPK as compressed.  Each selected
candidate writes a QPK2 bitmap/nonzero-code artifact and its exact byte count.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
from torch.nn import functional as F

from .compression import activation_liveness, build_quantized_model, export_sparse_packed_model
from .data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from .models import build_model
from .pruning import enforce_masks, global_magnitude_masks, validate_masks
from .quantization import ActivationFakeQuantizer, apply_quantizer_bit_overrides, apply_weight_bit_map
from .train import run_epoch
from .utils import append_csv, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Selected mixed-precision QAT resume checkpoint, never a .qpk")
    parser.add_argument("--teacher-checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.30, 0.40, 0.50])
    parser.add_argument("--max-recoveries", type=int, default=2, help="Recover this many best zero-shot candidates")
    parser.add_argument("--epochs", type=int, default=4); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0); parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=6886)
    parser.add_argument("--runs-dir", default="experiments/pruning_kd"); parser.add_argument("--run-name", default="pruned-mp-w4dw6edgew8-a6-seed6886")
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--evaluate-test", action="store_true", help="Use only after validation selects a final candidate.")
    return parser.parse_args()


def load_quantized_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    quant = checkpoint.get("quantization_state")
    if not quant: raise ValueError("--checkpoint must be a QAT resume checkpoint")
    model = build_quantized_model(build_model(checkpoint["model_config"], load_pretrained=False),
                                  quant["weight_bits"], quant["activation_bits"], quant.get("edge_bits"))
    if quant.get("realized_weight_bits"): apply_weight_bit_map(model, quant["realized_weight_bits"])
    if quant.get("activation_overrides"):
        apply_quantizer_bit_overrides(model, quant["activation_overrides"], ActivationFakeQuantizer)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint, quant


def kd_epoch(student, teacher, loader, device, optimizer, alpha, temperature, max_batches=None):
    training = optimizer is not None; student.train(training)
    # Recovery starts from a converged QAT model; keep its running BN statistics fixed.
    if training:
        for module in student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
    total_loss = total_correct = total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, (images, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches: break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            with torch.no_grad(): teacher_logits = teacher(images)
            hard = F.cross_entropy(logits, labels)
            soft = F.kl_div(F.log_softmax(logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean")
            loss = (1 - alpha) * hard + alpha * temperature * temperature * soft
            if training:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0); total_correct += (logits.argmax(1) == labels).sum().item(); total_examples += labels.size(0)
    if not total_examples: raise ValueError("no batches were processed")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha <= 1 or args.temperature <= 0: raise ValueError("alpha must be in [0,1] and temperature > 0")
    if args.max_recoveries < 1: raise ValueError("max-recoveries must be positive")
    if any(not 0 <= value < 1 for value in args.sparsities): raise ValueError("sparsities must be in [0,1)")
    set_seed(args.seed); device = resolve_device(args.device); checkpoint_path = Path(args.checkpoint)
    student, source, quant = load_quantized_checkpoint(checkpoint_path, device)
    teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = build_model(teacher_state["model_config"], load_pretrained=False).to(device)
    teacher.load_state_dict(teacher_state["state_dict"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    validation_size = source.get("validation_size", 5_000)
    train_loader, val_loader, test_loader = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", args.seed, validation_size)
    run_dir = Path(args.runs_dir) / args.run_name
    if run_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    baseline_loss, baseline_accuracy = run_epoch(student.eval(), val_loader, nn.CrossEntropyLoss(), device, max_batches=args.max_val_batches)
    zero_shot = []
    for sparsity in args.sparsities:
        candidate = copy.deepcopy(student); masks, summary = global_magnitude_masks(candidate, sparsity); enforce_masks(candidate, masks)
        loss, accuracy = run_epoch(candidate.eval(), val_loader, nn.CrossEntropyLoss(), device, max_batches=args.max_val_batches)
        zero_shot.append({"sparsity": sparsity, "validation_loss": loss, "validation_accuracy": accuracy,
                          "eligible_values": summary.eligible_values, "masked_values": summary.masked_values})
    (run_dir / "zero_shot.json").write_text(json.dumps({"unpruned_validation_loss": baseline_loss, "unpruned_validation_accuracy": baseline_accuracy, "candidates": zero_shot}, indent=2) + "\n", encoding="utf-8")
    selected = sorted(zero_shot, key=lambda row: row["validation_accuracy"], reverse=True)[:args.max_recoveries]
    results = []
    for rank, row in enumerate(selected, 1):
        sparsity = row["sparsity"]; candidate = copy.deepcopy(student); masks, _ = global_magnitude_masks(candidate, sparsity); enforce_masks(candidate, masks)
        scales, weights = [], []
        for name, parameter in candidate.named_parameters(): (scales if name.endswith(".scale") else weights).append(parameter)
        optimizer = torch.optim.SGD([{"params": weights, "lr": args.learning_rate, "weight_decay": 4e-5}, {"params": scales, "lr": args.learning_rate, "weight_decay": 0.0}], momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        candidate_dir = run_dir / f"s{int(sparsity * 100):02d}"; candidate_dir.mkdir(); history = candidate_dir / "history.csv"
        best_accuracy, best_state = -1.0, None
        for epoch in range(1, args.epochs + 1):
            started = perf_counter(); train_loss, train_acc = kd_epoch(candidate, teacher, train_loader, device, optimizer, args.alpha, args.temperature, args.max_train_batches)
            enforce_masks(candidate, masks)
            val_loss, val_acc = kd_epoch(candidate.eval(), teacher, val_loader, device, optimizer=None, alpha=args.alpha, temperature=args.temperature, max_batches=args.max_val_batches)
            scheduler.step(); append_csv(history, {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "train_accuracy": f"{train_acc:.3f}", "val_loss": f"{val_loss:.6f}", "val_accuracy": f"{val_acc:.3f}", "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}"})
            if val_acc > best_accuracy: best_accuracy, best_state = val_acc, copy.deepcopy(candidate.state_dict())
            print(f"s={sparsity:.2f} rank={rank} epoch={epoch}/{args.epochs} train={train_acc:.2f}% val={val_acc:.2f}% elapsed={perf_counter()-started:.1f}s")
        candidate.load_state_dict(best_state); enforce_masks(candidate, masks); summary = validate_masks(candidate, masks)
        checkpoint_out = candidate_dir / "best.pt"
        torch.save({"state_dict": candidate.state_dict(), "model_config": source["model_config"], "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}, "qat_config": source.get("qat_config", {}), "quantization_state": quant, "pruning_masks": {name: mask.cpu() for name, mask in masks.items()}, "pruning_summary": summary.__dict__, "validation_accuracy": best_accuracy, "validation_size": validation_size}, checkpoint_out)
        qpk_path = candidate_dir / "deployable_sparse.qpk"; storage = export_sparse_packed_model(candidate.eval(), qpk_path, masks)
        result = {"sparsity": sparsity, "best_validation_accuracy": best_accuracy, "checkpoint": str(checkpoint_out), "artifact": str(qpk_path), **storage.__dict__, "weight_compression_ratio": storage.ratio, "pruning_summary": summary.__dict__}
        results.append(result); (candidate_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    winner = max(results, key=lambda row: row["best_validation_accuracy"])
    if args.evaluate_test:
        winner_model, _, _ = load_quantized_checkpoint(Path(winner["checkpoint"]), device)
        masks = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)["pruning_masks"]; enforce_masks(winner_model, masks)
        _, winner["test_accuracy"] = run_epoch(winner_model.eval(), test_loader, nn.CrossEntropyLoss(), device)
    (run_dir / "results.json").write_text(json.dumps({"unpruned_validation_accuracy": baseline_accuracy, "candidates": results, "validation_selected": winner}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "validation_selected": winner}, indent=2))


if __name__ == "__main__": main()
