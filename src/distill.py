"""Train a width-scaled MobileNetV2 student with KD, then optionally QAT + KD."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
from torch.nn import functional as F

from .compression import activation_liveness, build_quantized_model, export_packed_model
from .data import CIFAR10_MEAN, CIFAR10_STD, build_cifar10_loaders
from .models import build_model
from .qat import freeze_batch_norm, precision_for_epoch
from .quantization import apply_mixed_weight_policy, set_quantizer_bits
from .train import run_epoch
from .utils import append_csv, resolve_device, set_seed


def kd_epoch(student, teacher, loader, device, optimizer, alpha, temperature, max_batches=None, freeze_bn=False):
    training = optimizer is not None; student.train(training)
    if training and freeze_bn: freeze_batch_norm(student)
    total_loss = total_correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for index, (images, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches: break
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            with torch.no_grad(): teacher_logits = teacher(images)
            loss = ((1 - alpha) * F.cross_entropy(logits, labels) + alpha * temperature ** 2 *
                    F.kl_div(F.log_softmax(logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean"))
            if training: loss.backward(); optimizer.step()
            total_loss += loss.item() * labels.size(0); total_correct += (logits.argmax(1) == labels).sum().item(); count += labels.size(0)
    if not count: raise ValueError("no batches were processed")
    return total_loss / count, 100 * total_correct / count


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fp32", "qat"), required=True)
    parser.add_argument("--teacher-checkpoint", default="results/checkpoints/baseline.pt")
    parser.add_argument("--student-checkpoint", help="Required QAT input; best FP32 KD student checkpoint")
    parser.add_argument("--width-mult", type=float, default=.75)
    parser.add_argument("--epochs", type=int, default=None); parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=2); parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--weight-bits", type=int, default=4); parser.add_argument("--activation-bits", type=int, default=6)
    parser.add_argument("--depthwise-weight-bits", type=int, default=6); parser.add_argument("--first-last-weight-bits", type=int, default=8)
    parser.add_argument("--transition-epochs", type=int, default=1); parser.add_argument("--freeze-bn-epoch", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--data-dir", default="data"); parser.add_argument("--device", default="auto"); parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=6886)
    parser.add_argument("--runs-dir", default="experiments/student_distillation"); parser.add_argument("--run-name")
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    return parser


def main():
    args = args_parser().parse_args()
    if args.stage == "qat" and not args.student_checkpoint: raise ValueError("--student-checkpoint is required for QAT")
    if args.width_mult <= 0 or args.temperature <= 0 or not 0 <= args.alpha <= 1: raise ValueError("invalid width/KD settings")
    args.epochs = args.epochs or (30 if args.stage == "fp32" else 10); args.learning_rate = args.learning_rate or (3e-4 if args.stage == "fp32" else 3e-4)
    set_seed(args.seed); device = resolve_device(args.device)
    teacher_data = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = build_model(teacher_data["model_config"], load_pretrained=False).to(device); teacher.load_state_dict(teacher_data["state_dict"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    student_config = {"width_mult": args.width_mult, "dropout": .2, "num_classes": 10, "pretrained": False}
    if args.stage == "fp32": student = build_model(student_config, load_pretrained=False).to(device)
    else:
        source = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
        student_config = source["model_config"]; base = build_model(student_config, load_pretrained=False); base.load_state_dict(source["state_dict"])
        student = build_quantized_model(base, 8, 8).to(device)
    validation_size = teacher_data.get("validation_size", 5_000)
    train_loader, val_loader, _ = build_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers, device.type == "cuda", args.seed, validation_size)
    run_name = args.run_name or f"mobilenetv2-{args.width_mult:g}-{args.stage}-kd-seed{args.seed}"; run_dir = Path(args.runs_dir) / run_name
    if run_dir.exists(): raise FileExistsError(f"refusing to overwrite immutable run directory: {run_dir}")
    run_dir.mkdir(parents=True); (run_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    scales, weights = [], []
    for name, parameter in student.named_parameters(): (scales if name.endswith(".scale") else weights).append(parameter)
    optimizer = torch.optim.SGD([{"params": weights, "lr": args.learning_rate, "weight_decay": 4e-5}, {"params": scales, "lr": args.learning_rate, "weight_decay": 0}], momentum=.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs); best_acc = -1.; best_state = None; history = run_dir / "history.csv"
    realized = None
    for epoch in range(1, args.epochs + 1):
        at_target = True
        if args.stage == "qat":
            weight_bits, activation_bits = precision_for_epoch(epoch, args.weight_bits, args.activation_bits, args.transition_epochs)
            set_quantizer_bits(student, weight_bits, activation_bits); at_target = (weight_bits, activation_bits) == (args.weight_bits, args.activation_bits)
            if at_target: realized = apply_mixed_weight_policy(student, args.weight_bits, args.depthwise_weight_bits, args.first_last_weight_bits)
        started = perf_counter(); train_loss, train_acc = kd_epoch(student, teacher, train_loader, device, optimizer, args.alpha, args.temperature, args.max_train_batches, args.stage == "qat" and epoch >= args.freeze_bn_epoch)
        val_loss, val_acc = kd_epoch(student.eval(), teacher, val_loader, device, None, args.alpha, args.temperature, args.max_val_batches); scheduler.step()
        append_csv(history, {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "train_accuracy": f"{train_acc:.3f}", "val_loss": f"{val_loss:.6f}", "val_accuracy": f"{val_acc:.3f}", "learning_rate": f"{optimizer.param_groups[0]['lr']:.8f}"})
        if at_target and val_acc > best_acc: best_acc, best_state = val_acc, {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        print(f"{args.stage} epoch={epoch}/{args.epochs} train={train_acc:.2f}% val={val_acc:.2f}% elapsed={perf_counter()-started:.1f}s")
    if best_state is None: raise RuntimeError("target precision was never reached")
    student.load_state_dict(best_state); checkpoint = run_dir / "best.pt"
    state = {"state_dict": student.state_dict(), "model_config": student_config, "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD}, "validation_accuracy": best_acc, "validation_size": validation_size, "distillation_config": vars(args)}
    if args.stage == "qat": state["qat_config"] = vars(args); state["quantization_state"] = {"weight_bits": args.weight_bits, "activation_bits": args.activation_bits, "realized_weight_bits": realized}
    torch.save(state, checkpoint); metrics = {"best_validation_accuracy": best_acc, "checkpoint": str(checkpoint)}
    if args.stage == "qat":
        artifact = run_dir / "deployable_model.qpk"; storage = export_packed_model(student.eval(), artifact); activation = activation_liveness(student, torch.zeros((1, 3, 32, 32), device=device)); metrics |= {**storage.__dict__, "weight_compression_ratio": storage.ratio, **activation.__dict__, "activation_compression_ratio": activation.ratio, "artifact": str(artifact)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n"); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
