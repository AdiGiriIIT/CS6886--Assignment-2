# Quantization status and Q3/Q4 handoff

Date: 8 September 2026. This handoff uses the updated artifacts in
`Single_Precision_artifacts/` and `Mixed_Precision_artifacts/` only. Every QAT
run uses the same improved FP32 baseline SHA-256:
`73a0bc35a65ebd049fc8b698f2816b3feb6ed1b53e41ec373c674c325a1f2837`.
That baseline achieves **93.69%** held-out CIFAR-10 top-1. All runs use a
12-epoch 8→6→target QAT schedule and choose a checkpoint using the 5,000-image
validation split before held-out test evaluation.

## Complete current frontier

| Policy | Validation top-1 | Test top-1 | Drop vs 93.69 | QPK size | Weight ratio | Activation traffic ratio |
|---|---:|---:|---:|---:|---:|---:|
| W8A8 | 93.60% | 93.03% | 0.66 pp | 2.380 MiB | 3.558x | 3.998x |
| W6A6 | 93.46% | 92.81% | 0.88 pp | 1.854 MiB | 4.566x | 5.330x |
| W4A6 | 91.96% | 90.75% | 2.94 pp | 1.329 MiB | 6.370x | 5.330x |
| W4A4 | 86.92% | 85.68% | 8.01 pp | 1.329 MiB | 6.370x | 7.992x |
| W4A6 + W8 stem/classifier | 92.10% | 91.00% | 2.69 pp | 1.336 MiB | 6.338x | 5.330x |
| W4A6 + W6 depthwise | 92.14% | 91.40% | 2.29 pp | 1.345 MiB | 6.297x | 5.330x |
| W4A6 + W6 depthwise + W8 stem/classifier | **92.52%** | 91.62% | 2.07 pp | 1.351 MiB | 6.267x | **5.330x** |
| Previous policy + A8 input/logits | 92.38% | 91.65% | 2.04 pp | 1.351 MiB | 6.267x | 5.320x |
| Previous policy + A8 stem ReLU6 | 92.32% | 91.58% | 2.11 pp | 1.351 MiB | 6.267x | 5.232x |
| Previous policy + A8 stem and first-block ReLU6 | 92.30% | **91.76%** | **1.93 pp** | 1.351 MiB | 6.267x | 5.138x |

MiB means bytes / 2^20. The QPK total includes packed codes, scales, int32
biases, requantization fields, descriptors, padding, and header. Activation
traffic is batch-one forward boundary traffic for `1×3×32×32`, calculated as
`fp32_traffic_bytes / quantized_traffic_bytes`.

## Result of the additional sensitivity-guided experiments

The depthwise-only W6 test confirms that W8 stem/classifier weights are useful:
adding them costs just 6,832 bytes (0.48% of the compact QPK) but raises
validation accuracy by 0.38 pp and test accuracy by 0.22 pp. Retain both
weight exceptions.

The internal-A8 hypothesis is not validated under the correct selection
protocol. Stem-only A8 lowers validation and test accuracy relative to the
all-A6 compact policy. Two early A8 boundaries produce the highest observed
test result (91.76%), but have **lower validation accuracy** (92.30% versus
92.52%) and 3.74% more activation traffic. Since the test set must not select
the precision policy, the 0.14 pp test difference is treated as seed/test
variation rather than evidence for the A8 exceptions. A8 input/logits is also
validation-negative and only adds 0.03 pp test accuracy.

W4A4 remains dominated by W4A6: it has the same 1.329 MiB QPK but loses a
further 5.07 pp test accuracy.

## Robustness check: all-A6 versus early-A8

The robustness artifacts contain two additional paired QAT seeds for the two
policies. Raw validation values should be compared *within a seed*, not pooled
across seeds, because the QAT seed also defines the train/validation split.

| Seed | All-A6 validation | All-A6 test | Early-A8 validation | Early-A8 test | Early-A8 − all-A6 validation | Early-A8 − all-A6 test |
|---:|---:|---:|---:|---:|---:|---:|
| 6886 | 92.52% | 91.62% | 92.30% | 91.76% | -0.22 pp | +0.14 pp |
| 1234 | 96.84% | 91.59% | 97.04% | 91.69% | +0.20 pp | +0.10 pp |
| 2026 | 96.52% | 91.54% | 96.54% | 91.45% | +0.02 pp | -0.09 pp |
| Mean / range | — | **91.58%** / 91.54–91.62% | — | **91.63%** / 91.45–91.76% | 0.00 pp mean | +0.05 pp mean |

Early A8 has no stable validation advantage: its paired validation differences
average exactly 0.00 pp. Its +0.05 pp mean test difference is smaller than its
seed-to-seed variation and reverses at seed 2026. It also raises quantized
activation traffic from 437,968 to 454,352 bytes (+3.74%), reducing the
activation-traffic ratio from 5.330x to 5.138x. Its test range is wider than
all-A6 (0.31 pp versus 0.08 pp). Therefore it does not justify the additional
activation precision.

## Final selected candidate

Finalize this compact policy:

```text
Default:          W4 / A6
Depthwise weights: W6
Stem weights:      W8
Classifier weights: W8
All activations:   A6 (no input/logit or internal A8 exceptions)
```

This is `mp-w4dw6edgew8-a6`: **91.58% mean held-out test top-1** over seeds
6886, 1234, and 2026 (range 91.54–91.62%), **1.351 MiB**, **6.267x** weight
reduction, and **5.330x** activation-traffic reduction. It has the same packed
weight size as the early-A8 policy, materially better activation compression,
and the more stable test result. This is the best balanced, defensible compact
operating point.

W6A6 is the accuracy-oriented comparator, not the selected compact candidate:
it reaches 92.81% test but is 37.25% larger than the selected QPK (1.854 MiB
versus 1.351 MiB) and has the same activation-traffic ratio.

## Deployment scope

The artifacts establish fake-QAT accuracy and byte-exact packed-storage
accounting. They do not establish latency, throughput, energy, or numerical
equivalence of exported integer inference, because the QPK has no validated
integer convolution backend.

## Artifact locations

- Uniform policies: `Single_Precision_artifacts/*/experiments/sweeps/*/metrics.json` and associated held-out logs.
- Mixed policies: `Mixed_Precision_artifacts/*/experiments/sweeps/*/metrics.json` and associated held-out logs.
- Saturation diagnostics: each sweep's `activation_quantization_diagnostics.csv`.
