# Quantization status and Q3/Q4 handoff

Date: 8 September 2026. This document is based only on the current artifacts
in `Single_Precision_artifacts/` and `Mixed_Precision_artifacts/`. All seven
QAT runs used the same improved FP32 baseline SHA-256:
`73a0bc35a65ebd049fc8b698f2816b3feb6ed1b53e41ec373c674c325a1f2837`.
Its held-out CIFAR-10 test top-1 is **93.69%**. Runs use seed 6886, a fixed
45,000/5,000 train/validation split, 12 QAT epochs, a 8→6→target transition,
and validation-selected checkpoints. The test accuracies below are the saved
held-out evaluations of those selected checkpoints.

## Current results

| Policy | Validation top-1 | Test top-1 | Drop vs 93.69 | QPK size | Weight ratio | Activation traffic ratio |
|---|---:|---:|---:|---:|---:|---:|
| W8A8 | 93.60% | 93.03% | 0.66 pp | 2.380 MiB | 3.558x | 3.998x |
| W6A6 | 93.46% | 92.81% | 0.88 pp | 1.854 MiB | 4.566x | 5.330x |
| W4A6 | 91.96% | 90.75% | 2.94 pp | 1.329 MiB | 6.370x | 5.330x |
| W4A4 | 86.92% | 85.68% | 8.01 pp | 1.329 MiB | 6.370x | 7.992x |
| W4A6 + W8 stem/classifier | 92.10% | 91.00% | 2.69 pp | 1.336 MiB | 6.338x | 5.330x |
| W4A6 + W6 depthwise + W8 stem/classifier | 92.52% | 91.62% | 2.07 pp | 1.351 MiB | 6.267x | 5.330x |
| Previous policy + A8 input/logits | 92.38% | **91.65%** | **2.04 pp** | 1.351 MiB | 6.267x | 5.321x |

MiB means bytes divided by 2^20. QPK size includes packed codes, per-channel
scales, activation scales, int32 biases, requantization fields, descriptors,
padding, and header. Activation ratio is forward boundary traffic at batch one,
input shape `1×3×32×32`: `fp32_traffic_bytes / quantized_traffic_bytes`.

## Interpretation

W6A6 is the accuracy-oriented point: 0.88 pp loss, 4.566x packed-weight
reduction, and 5.330x activation-traffic reduction. The compact mixed point is
W4A6 plus W6 depthwise and W8 stem/classifier: 1.351 MiB, 27.14% smaller than
W6A6, for an additional 1.24 pp test loss. A8 input/logits is not material:
it gains only 0.03 pp test accuracy and has lower validation accuracy than the
otherwise identical A6-edge run (92.38% versus 92.52%).

W8 stem/classifier weights recover 0.25 pp over W4A6. Raising all depthwise
weights to W6 then recovers another 0.62 pp, making depthwise weights the
strongest demonstrated exception. W4A4 is dominated by W4A6: both QPKs are
1.329 MiB, while W4A4 loses a further 5.07 pp on test.

The diagnostics support one narrow activation experiment. In the W4/W6/W8-A6
run, stem ReLU6 `model.features.0.2.activation_quantizer` is the clear internal
hotspot (4.243% saturation; next internal boundary 0.129%). In W4A6,
`model.features.1.conv.0.2.activation_quantizer` reaches 5.600% and stem is
2.772%. These are the only justified internal A8 candidates.

## Remaining targeted experiments

Do not launch another broad sweep. Run these in order, selecting on validation
and evaluating held-out test only after selection. `BASELINE_CKPT` must be the
improved-baseline copy with SHA-256
`73a0bc35a65ebd049fc8b698f2816b3feb6ed1b53e41ec373c674c325a1f2837`.

### 1. Depthwise-only W6

This tests whether W8 stem/classifier weights are required after upgrading
depthwise weights, and is the highest-value missing ablation.

```bash
BASELINE_CKPT=results/checkpoints/baseline.pt
DATA_DIR=/path/to/cifar10

python -m src.qat --checkpoint "$BASELINE_CKPT" --data-dir "$DATA_DIR" --device cuda \
  --weight-bits 4 --activation-bits 6 --depthwise-weight-bits 6 --epochs 12 \
  --run-name mp-w4dw6-a6-seed6886

python -m src.evaluate \
  --checkpoint results/checkpoints/qat-mp-w4dw6-a6-seed6886-best-target.pt \
  --data-dir "$DATA_DIR" --device cuda
```

### 2. Stem-only A8 on compact mixed precision

This targets the 4.243%-saturated internal stem boundary, rather than repeating
the unconvincing input/logit A8 exception.

```bash
python -m src.qat --checkpoint "$BASELINE_CKPT" --data-dir "$DATA_DIR" --device cuda \
  --weight-bits 4 --activation-bits 6 --depthwise-weight-bits 6 \
  --first-last-weight-bits 8 --epochs 12 \
  --activation-bit-override model.features.0.2.activation_quantizer=8 \
  --run-name mp-w4dw6edgew8-a6-stem-a8-seed6886

python -m src.evaluate \
  --checkpoint results/checkpoints/qat-mp-w4dw6edgew8-a6-stem-a8-seed6886-best-target.pt \
  --data-dir "$DATA_DIR" --device cuda
```

### 3. Only if stem A8 helps: two early A8 boundaries

Add the first inverted-residual ReLU6 boundary. Do not promote later boundaries
without comparable saturation in a newly produced diagnostic.

```bash
python -m src.qat --checkpoint "$BASELINE_CKPT" --data-dir "$DATA_DIR" --device cuda \
  --weight-bits 4 --activation-bits 6 --depthwise-weight-bits 6 \
  --first-last-weight-bits 8 --epochs 12 \
  --activation-bit-override model.features.0.2.activation_quantizer=8 \
  --activation-bit-override model.features.1.conv.0.2.activation_quantizer=8 \
  --run-name mp-w4dw6edgew8-a6-early-a8-seed6886
```

`--activation-bit-override QUANTIZER=BITS` is an exact named-boundary
exception, recorded in the run configuration/checkpoint and restored by
`src.evaluate` and `src.export_deploy`.

## Stop rule and reporting

Keep a new policy only if its validation gain exceeds normal seed variation
while retaining a preferred size/accuracy frontier. If none meaningfully beats
the compact mixed policy, stop. Rerun W6A6 and the final compact policy with
two additional seeds and report mean/range. Do not claim latency, throughput,
or energy gains: QPK is byte-exact packed-storage accounting, not a validated
integer-kernel deployment.

## Artifact locations

- Uniform metrics and test logs: `Single_Precision_artifacts/*/experiments/sweeps/*/metrics.json` and `Single_Precision_artifacts/*/results/logs/*-held-out-test.log`.
- Mixed metrics and test logs: `Mixed_Precision_artifacts/*/experiments/sweeps/*/metrics.json` and `Mixed_Precision_artifacts/*/results/logs/*-held-out-test.log`.
- Saturation evidence: each run's `activation_quantization_diagnostics.csv`.
