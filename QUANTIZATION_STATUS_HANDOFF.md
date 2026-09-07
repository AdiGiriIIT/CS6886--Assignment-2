# Quantization status and Q3/Q4 handoff

Date: 6 September 2026. All accuracy differences below use the held-out test
result actually present in this repository: **92.39%**
(`baseline_results/results/logs/baseline-held-out-test.log`). The 92.69% number
in `main.tex` and `COMPRESSION_DESIGN.md` is not supported by the supplied
checkpoint/log and must not be mixed into the final comparison.

## Executive status

The custom QAT implementation is credible for measuring the accuracy of the
PyTorch fake-quantized graph. It covers all Conv2d/Linear weights with learned
per-output-channel symmetric scales, quantizes ReLU6 outputs as unsigned, and
places signed quantizers at input, projection, residual-add, and logit
boundaries. Runs start from the same baseline SHA-256 and select checkpoints on
the 5,000-image validation set before one held-out test evaluation.

The `.qpk` files prove that weight codes can be bit-packed and give useful,
byte-exact **storage estimates**. They are not yet executable integer models.
Therefore there is no measured inference latency, throughput, or energy
improvement. The honest current efficiency claims are serialized storage
reduction and analytical activation storage/traffic reduction.

## Reconciled results

| Policy | Test top-1 | Drop vs 92.39 | QPK size | Weight ratio | Activation traffic ratio |
|---|---:|---:|---:|---:|---:|
| W8A8 | 92.22% | 0.17 pp | 2.380 MiB | 3.558x | 3.998x |
| W6A6 | 91.19% | 1.20 pp | 1.854 MiB | 4.566x | 5.330x |
| W4A6 | 88.76% | 3.63 pp | 1.329 MiB | 6.370x | 5.330x |
| W4A4 | 81.98% | 10.41 pp | 1.329 MiB | 6.370x | 7.992x |
| MP: W4/A6 + W8 stem/classifier | 88.67% | 3.72 pp | 1.336 MiB | 6.338x | 5.330x |
| MP: W4/A6 + W6 depthwise + W8 stem/classifier | 89.93% | 2.46 pp | 1.351 MiB | 6.267x | 5.330x |
| MP: previous policy + A8 input/logits | 89.94% | 2.45 pp | 1.351 MiB | 6.267x | 5.320x |

The ratio denominator is 8,878,504 bytes of BN-folded FP32 deployable weights.
The QPK numerator includes packed codes, FP32 per-channel weight scales, FP32
activation scales, int32 biases, requantization fields, JSON descriptors,
alignment, and the header. Use MiB (`bytes / 2^20`) consistently; optionally
give decimal MB in parentheses.

The last column is recomputed from the recorded traffic bytes
(`fp32_traffic_bytes / quantized_traffic_bytes`). The artifact JSON field named
`activation_compression_ratio` is instead a peak-live ratio, because of an
implementation naming error; do not use that field as a traffic ratio.

The mixed runs are complete and valid: each used the same baseline SHA-256,
seed, 12-epoch schedule, and best-validation checkpoint protocol as the fixed
runs. The best validation-selected mixed point is the A8-input/logit variant
(91.04% validation, 89.94% test). It recovers **1.18 pp** over W4A6 on test for
only 22,912 additional QPK bytes (1.64%; 1.351 MiB total), while remaining
527,760 bytes (27.14%) smaller than W6A6. This is a real, useful new
accuracy--storage knee, but it does **not** beat W6A6 on accuracy: it remains
1.25 pp lower on test at nearly the same modeled A6 traffic ratio.

The W8-stem/classifier-only ablation is dominated by W4A6: it is 6,848 bytes
larger and 0.09 pp less accurate. Raising depthwise weights to W6 is what
produces the recovery. The additional A8 input/logit exception changes the QPK
size by zero and adds only 770 modeled traffic bytes; its 0.01 pp test gain
over the otherwise identical A6-edge policy is far below what one seed can
establish. It was nevertheless selected by validation (91.04% versus 90.80%)
at the same QPK size.

W6A6 remains the recommended Q4 point when accuracy is the primary objective:
it retains 98.70% of baseline accuracy while reducing weight storage 4.566x
and modeled activation traffic 5.330x. The depthwise mixed policy is the
recommended aggressive alternative when its 27.14% smaller QPK than W6A6 is
worth its additional 1.25 pp loss. W4A4 is dominated for persistent
model size by W4A6: it saves only 96 QPK bytes while losing another 6.78
accuracy points.

## What “activation compression” means

State this explicitly in the report: batch size one, eval mode, CIFAR-10 input
shape `1x3x32x32`; count every explicit quantized activation-boundary tensor in
one forward schedule, bit-pack it at its realized width, and include one FP32
scale per boundary. “Traffic” is the sum of bytes written at those boundaries.
Peak-live memory attempts to retain residual operands until addition, but the
current tracer relies on Python tensor identities and produced inconsistent
FP32 peaks across runs. Use the traffic ratio in the final table until the
liveness tracer is replaced and regenerated.

## Correctness gaps before claiming deployment

1. Export folds BN after training but retains pre-fold learned weight scales;
   exported numerical accuracy has not been evaluated.
2. Every layer descriptor currently receives the model-input scale for bias
   quantization, rather than its actual input-boundary scale.
3. Requantization multiplier/shift pairs are unity placeholders, not the
   accumulator-to-output scale mapping for each layer.
4. The QPK has no loader/integer convolution backend, so it cannot substantiate
   runtime speedup.
5. Learned scales are clamped in forward but are not positively parameterized;
   a scale driven negative can lose useful gradient.
6. Existing tests could not be rerun on this machine because its Python lacks
   PyTorch. Run `python -m unittest discover -s tests -v` in Kaggle/Colab.

These gaps do not invalidate the recorded fake-QAT test accuracies. They mean
the final report must call QPK size “packed storage/accounting” rather than a
validated integer deployment artifact.

## Mixed-precision status and next step

The logs support keeping activations at A6. At fixed W4, A6 to A4 costs 6.78
points. The completed ablations show that W8 stem/classifier weights alone do
not help, whereas W6 depthwise weights recover 1.17--1.18 pp. Thus the
depthwise exception, not the edge-weight exception, is the actionable result.

This is a good time to stop broad precision-policy exploration. The current
set identifies the relevant Pareto frontier: W4A6 for smallest model, mixed
W4/W6/W8-A6 for the compact middle point, W6A6 for the accuracy-oriented
selection, and W8A8 for near-baseline accuracy. More combinations of the same
global exception classes have low expected value; the A8-edge ablation already
showed a practically negligible test change.

If compute time remains, spend it on confirmation or one targeted experiment,
not another broad sweep:

- Highest-value confirmation: rerun the selected depthwise mixed policy and
  W6A6 with two additional seeds, select on validation, and report mean/range.
  The present 1.25 pp W6A6-versus-mixed gap is likely meaningful, but its exact
  size needs seed variation.
- Highest-value new ablation: add a configurable stem/early-ReLU activation
  exception and test A8 only at the boundaries previously diagnosed as A4
  saturation hotspots. The current `--edge-bits` controls input and logits,
  not the stem boundary, so it did not test that hypothesis. Do this only if
  the implementation work is acceptable; no evidence supports broad A8
  activations.

Do not claim a runtime advantage from any of these policies until the export
correctness gaps below and an integer backend are validated.

## Q3/Q4 submission checklist

- Q3: include all four uniform points plus the three completed mixed candidates,
  accuracy curves,
  the consolidated table, and the mandatory W&B Parallel Coordinates chart.
  Axes should include policy, realized average weight bits, activation bits,
  exception bytes, validation/test accuracy, QPK MiB, weight ratio, and
  activation-traffic ratio.
- Q4: report exactly one selected operating point. Use W6A6 for the
  accuracy-oriented recommendation; describe the depthwise mixed policy as the
  compact Pareto alternative, not a replacement. Give the selected point's four
  required values: weight ratio, activation ratio/method, test accuracy, and
  final size.
- Add a stacked metadata/payload breakdown. For W6A6: 1,651,920 packed-weight
  bytes; 68,264 weight-scale bytes; 216 activation-scale bytes; 68,264 int32
  bias bytes; 136,528 requantization bytes; 19,293 descriptor bytes; 63 padding
  bytes; 12 header bytes.
- Do not claim speedup from bit width or BOPs. A normalized BOP count may be
  presented as a theoretical compute proxy, clearly labeled as such.
- Resolve the baseline discrepancy and correct `main.tex`: checkpoint selection
  must be described as validation-based, not test-based.
- Add Q2--Q5 to `main.tex`; it currently ends after Q1. Add the GitHub URL and
  exact environment/reproduction commands.

## Artifact integrity notes

The W6 QPK matches its Day-2 exported SHA. The standalone W8 QPK differs from
the Day-2 W8 export despite equal size, and its peak-live accounting also
differs; regenerate W8 before using peak memory. Day-2 W4 metrics use an older
accounting schema, so use the standalone `compressed_artifacts` accounting
numbers consistently. The two incorrect `artifact` fields in W6/W8 accounting
JSON were corrected in this handoff.
