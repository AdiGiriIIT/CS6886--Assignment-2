"""Prune a selected QAT model and recover it with supervised fine-tuning.

Candidates use nested global-magnitude masks over W4 pointwise convolutions.
Masks can grow gradually, are enforced after every optimizer step, and each
candidate emits a byte-exact QPK2 bitmap/nonzero-code artifact.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from .compression import build_quantized_model, export_sparse_packed_model
from .data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from .models import build_model
from .pruning import enforce_masks, global_magnitude_masks, masked_optimizer_step, validate_masks
from .quantization import ActivationFakeQuantizer, apply_quantizer_bit_overrides, apply_weight_bit_map
from .train import run_epoch
from .utils import append_csv, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Selected mixed-precision QAT resume checkpoint")
    parser.add_argument("--sparsities", nargs="+", type=float, default=[.30, .35, .40, .45, .50])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=4)
    parser.add_argument("--initial-sparsity", type=float, default=.30)
    parser.add_argument("--max-validation-drop", type=float, default=.50,
                        help="Select smallest artifact no more than this many percentage points below the unpruned validation score")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6886); parser.add_argument("--runs-dir", default="experiments/pruning")
    parser.add_argument("--run-name", default="pruned-mp-w4dw6edgew8-a6-seed6886")
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def load_quantized_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    quant = checkpoint.get("quantization_state")
    if not quant:
        raise ValueError("--checkpoint must be a QAT resume checkpoint")
    model = build_quantized_model(build_model(checkpoint["model_config"], load_pretrained=False),
                                  quant["weight_bits"], quant["activation_bits"], quant.get("edge_bits"))
    if quant.get("realized_weight_bits"):
        apply_weight_bit_map(model, quant["realized_weight_bits"])
    if quant.get("activation_overrides"):
        apply_quantizer_bit_overrides(model, quant["activation_overrides"], ActivationFakeQuantizer)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint, quant


def recovery_epoch(model: nn.Module, loader, criterion, device: torch.device,
                   optimizer: torch.optim.Optimizer, masks: dict[str, torch.Tensor],
                   max_batches: int | None = None) -> tuple[float, float]:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    total_loss = total_correct = total_examples = 0
    for index, (images, labels) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images); loss = criterion(logits, labels)
        loss.backward(); masked_optimizer_step(optimizer, model, masks)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item(); total_examples += labels.size(0)
    if not total_examples:
        raise ValueError("no batches were processed")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def scheduled_sparsity(target: float, initial: float, epoch: int, warmup_epochs: int) -> float:
    """Cubic ramp from the initial mask to target; target is reached at warmup."""
    start = min(initial, target)
    if warmup_epochs <= 0:
        return target
    progress = min(1.0, epoch / warmup_epochs)
    return start + (target - start) * progress ** 3


def main() -> None:
    args = parse_args()
    if any(not 0 <= value < 1 for value in args.sparsities): raise ValueError("sparsities must be in [0,1)")
    if not 0 <= args.initial_sparsity < 1: raise ValueError("initial-sparsity must be in [0,1)")
    if args.epochs < 1 or args.pruning_warmup_epochs < 0: raise ValueError("invalid epoch count")
    if args.pruning_warmup_epochs > args.epochs: raise ValueError("pruning warmup cannot exceed epochs")
    if args.max_validation_drop < 0: raise ValueError("max-validation-drop must be nonnegative")
    if len(set(args.sparsities)) != len(args.sparsities): raise ValueError("duplicate sparsity candidates")

    set_seed(args.seed); device = resolve_device(args.device)
    source_model, source, quant = load_quantized_checkpoint(Path(args.checkpoint), device)
    validation_size = source.get("validation_size", 5_000)
    train_loader, val_loader, _ = build_cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", args.seed, validation_size)
    run_dir = Path(args.runs_dir) / args.run_name
    if run_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    baseline_loss, baseline_accuracy = run_epoch(source_model.eval(), val_loader, criterion, device,
                                                 max_batches=args.max_val_batches)
    # One immutable CPU copy defines every nested mask across the full sweep.
    mask_reference = copy.deepcopy(source_model).cpu()
    results = []
    for target in sorted(args.sparsities):
        candidate = copy.deepcopy(source_model)
        # All scheduled masks are derived from one immutable reference, making
        # them nested: a weight pruned at a lower sparsity never becomes active.
        weights, scales = [], []
        for name, parameter in candidate.named_parameters():
            (scales if name.endswith(".scale") else weights).append(parameter)
        optimizer = torch.optim.SGD(
            [{"params": weights, "lr": args.learning_rate, "weight_decay": 4e-5},
             {"params": scales, "lr": args.learning_rate, "weight_decay": 0.0}],
            momentum=.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        candidate_dir = run_dir / f"s{int(round(target * 100)):02d}"; candidate_dir.mkdir()
        history = candidate_dir / "history.csv"; best_accuracy = -1.0; best_state = best_masks = None
        for epoch in range(1, args.epochs + 1):
            current = scheduled_sparsity(target, args.initial_sparsity, epoch, args.pruning_warmup_epochs)
            masks, _ = global_magnitude_masks(mask_reference, current); enforce_masks(candidate, masks)
            started = perf_counter()
            train_loss, train_acc = recovery_epoch(candidate, train_loader, criterion, device, optimizer,
                                                   masks, args.max_train_batches)
            val_loss, val_acc = run_epoch(candidate.eval(), val_loader, criterion, device,
                                          max_batches=args.max_val_batches)
            scheduler.step(); at_target = abs(current - target) < 1e-12
            append_csv(history, {"epoch": epoch, "sparsity": f"{current:.6f}",
                       "at_target": at_target, "train_loss": f"{train_loss:.6f}",
                       "train_accuracy": f"{train_acc:.3f}", "val_loss": f"{val_loss:.6f}",
                       "val_accuracy": f"{val_acc:.3f}", "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}"})
            if at_target and val_acc > best_accuracy:
                best_accuracy = val_acc
                best_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
                best_masks = {name: mask.cpu().clone() for name, mask in masks.items()}
            print(f"target={target:.2f} epoch={epoch}/{args.epochs} sparsity={current:.3f} "
                  f"train={train_acc:.2f}% val={val_acc:.2f}% elapsed={perf_counter()-started:.1f}s")
        if best_state is None or best_masks is None: raise RuntimeError("candidate never reached target sparsity")
        candidate.load_state_dict(best_state); enforce_masks(candidate, best_masks)
        summary = validate_masks(candidate, best_masks)
        checkpoint_out = candidate_dir / "best.pt"
        torch.save({"state_dict": candidate.state_dict(), "model_config": source["model_config"],
                    "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD},
                    "qat_config": source.get("qat_config", {}), "quantization_state": quant,
                    "pruning_masks": best_masks, "pruning_summary": summary.__dict__,
                    "validation_accuracy": best_accuracy, "validation_size": validation_size}, checkpoint_out)
        qpk_path = candidate_dir / "deployable_sparse.qpk"
        storage = export_sparse_packed_model(candidate.eval(), qpk_path, best_masks)
        result = {"sparsity": target, "best_validation_accuracy": best_accuracy,
                  "validation_drop": baseline_accuracy - best_accuracy,
                  "checkpoint": str(checkpoint_out), "artifact": str(qpk_path), **storage.__dict__,
                  "weight_compression_ratio": storage.ratio, "pruning_summary": summary.__dict__}
        results.append(result)
        (candidate_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    feasible = [row for row in results if row["validation_drop"] <= args.max_validation_drop]
    selection_pool = feasible or results
    winner = min(selection_pool, key=lambda row: (row["total_bytes"], -row["best_validation_accuracy"]))
    selection = "smallest artifact within validation-drop constraint" if feasible else "highest compression fallback; no candidate met constraint"
    payload = {"unpruned_validation_loss": baseline_loss, "unpruned_validation_accuracy": baseline_accuracy,
               "max_validation_drop": args.max_validation_drop, "selection_rule": selection,
               "candidates": results, "validation_selected": winner}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "selection_rule": selection,
                      "validation_selected": winner}, indent=2))


if __name__ == "__main__":
    main()
