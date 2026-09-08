"""Upload saved Q3 compression results to Weights & Biases.

Example:
    wandb login
    python -m src.upload_wandb_results --project cs6886-assignment2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TEST_ACCURACY = re.compile(r"Test top-1 accuracy:\s*([0-9.]+)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", help="Optional W&B user or team entity.")
    parser.add_argument("--artifact-root", action="append", default=None,
                        help="Directory to scan; repeat to add roots.")
    return parser.parse_args()


def test_accuracy(metrics_path: Path) -> float:
    run_name = metrics_path.parent.name
    log_path = metrics_path.parents[3] / "results" / "logs" / f"{run_name}-held-out-test.log"
    match = TEST_ACCURACY.search(log_path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Could not find held-out top-1 accuracy in {log_path}")
    return float(match.group(1))


def upload(metrics_path: Path, project: str, entity: str | None) -> None:
    import wandb  # Normal training does not require W&B.

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    nominal_w = metrics["weight_bits"]
    realized = metrics.get("realized_weight_bits", {})
    config = {
        "weight_bits": nominal_w,
        "activation_bits": metrics["activation_bits"],
        "depthwise_weight_bits": metrics.get("depthwise_weight_bits") or nominal_w,
        "first_last_weight_bits": metrics.get("first_last_weight_bits") or nominal_w,
        "weight_exception_layers": sum(bits != nominal_w for bits in realized.values()),
        "activation_exception_boundaries": len(metrics.get("activation_overrides", {})),
        "seed": int(metrics["run_name"].rsplit("seed", 1)[-1]),
        "baseline_sha256": metrics["baseline_sha256"],
    }
    summary = {
        "validation_accuracy": metrics["best_target_validation_accuracy"],
        "test_accuracy": test_accuracy(metrics_path),
        "model_size_mib": metrics["total_bytes"] / 2**20,
        "model_size_bytes": metrics["total_bytes"],
        "weight_compression_ratio": metrics["weight_compression_ratio"],
        "activation_traffic_compression_ratio": metrics["activation_compression_ratio"],
        "quantized_activation_traffic_bytes": metrics["quantized_traffic_bytes"],
    }
    run = wandb.init(project=project, entity=entity, group="q3-compression",
                     name=metrics["run_name"], config=config)
    run.summary.update(summary)
    run.finish()


def main() -> None:
    args = parse_args()
    roots = args.artifact_root or ["Single_Precision_artifacts", "Mixed_Precision_artifacts"]
    paths = sorted(path for root in roots
                   for path in Path(root).glob("**/experiments/sweeps/*/metrics.json"))
    if not paths:
        raise FileNotFoundError("No metrics.json files found under the supplied artifact roots.")
    for path in paths:
        upload(path, args.project, args.entity)
        print(f"Uploaded {path.parent.name}")


if __name__ == "__main__":
    main()
