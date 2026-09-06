"""Export a QAT model to the custom packed inference artifact and verify bytes.

This deliberately does not benchmark PyTorch fake-QAT.  The resulting .qpk
contains integer weight codes and packed activation/storage metadata for a
future integer-capable deployment backend.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .compression import activation_liveness, build_quantized_model, export_packed_model
from .models import build_model
from .quantization import apply_weight_bit_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="QAT training-resume checkpoint")
    parser.add_argument("--output", default="results/deploy/model.qpk")
    parser.add_argument("--report", default="results/deploy/model_accounting.json")
    parser.add_argument("--input-shape", nargs=3, type=int, default=(3, 32, 32), metavar=("C", "H", "W"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    quant = checkpoint.get("quantization_state")
    if not quant: raise ValueError("checkpoint has no quantization_state; supply a QAT checkpoint")
    base = build_model(checkpoint["model_config"], load_pretrained=False)
    model = build_quantized_model(base, quant["weight_bits"], quant["activation_bits"], quant.get("edge_bits"))
    if quant.get("realized_weight_bits"):
        apply_weight_bit_map(model, quant["realized_weight_bits"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    storage = export_packed_model(model, args.output)
    activation = activation_liveness(model, torch.zeros((1, *args.input_shape)))
    report = {"artifact": str(args.output), "actual_artifact_bytes": Path(args.output).stat().st_size,
              **storage.__dict__, "weight_compression_ratio": storage.ratio,
              **activation.__dict__, "activation_compression_ratio": activation.ratio,
              "units": "bytes (MiB = bytes / 2^20)",
              "runtime_note": "No latency, energy, or throughput result is implied: PyTorch fake-QAT conv2d uses floating-point kernels."}
    if report["actual_artifact_bytes"] != report["total_bytes"]: raise AssertionError("serialized length mismatch")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
