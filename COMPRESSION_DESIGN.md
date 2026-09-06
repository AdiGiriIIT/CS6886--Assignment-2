# Compression Design for MobileNetV2 on CIFAR-10

## 1. Objective and recommendation

The current baseline is an ImageNet-pretrained MobileNetV2 adapted to 32 x 32
inputs and fine-tuned on CIFAR-10. Its best test top-1 accuracy is **92.69%**.
The compression work should preserve this checkpoint as the common starting
point for every experiment.

The recommended method is **quantization-aware training (QAT) with learned,
uniform quantizers**:

- per-output-channel, symmetric signed quantization for convolution and linear
  weights;
- per-tensor learned quantization for activations;
- unsigned activation quantization after ReLU6 and signed quantization at
  boundaries that can contain negative values;
- straight-through gradients for rounding and clipping; and
- explicit bit packing and metadata accounting when estimating the deployable
  model size.

The primary target is **W4A4**: 4-bit weights and 4-bit activations. The final
model may retain a small number of 6- or 8-bit boundaries only if experiments
show that they recover meaningful accuracy. This is the best balance for the
relative grading: W8A8 is low risk but only offers an ideal 4x payload
reduction, while W3A3 is likely to require substantially more tuning on
MobileNetV2. W4A4 offers an ideal 8x reduction for both weight payload and
activation payload before overheads.

Pruning should not be added during this assignment cycle. It creates a second
training and representation problem, does not directly improve the activation
bit-width ratio, and makes the reported storage format harder to defend. It can
consume the limited time without improving the final Pareto point.

## 2. Why this design

MobileNetV2 is built from inverted residual blocks containing expansion
pointwise convolutions, depthwise convolutions, and linear projection
convolutions. Its narrow bottlenecks and residual paths make it more sensitive
to poorly chosen activation ranges than many conventional CNNs. Depthwise
filters also have independent and potentially very different weight ranges.
Consequently, one scale for an entire weight tensor can waste most of the
available levels on a few channels.

Per-channel weight scales address that problem with little overhead. A
convolution with `C_out` output channels needs `C_out` FP32 scale values, while
the usually much larger weight tensor is stored at 3--8 bits per value. Learned
per-tensor activation scales keep the runtime representation and the accounting
simple while allowing QAT to find better clipping ranges than raw min/max
calibration.

The design is based on the following references:

- [Learned Step Size Quantization (LSQ)](https://openreview.net/pdf?id=rkgO66VKDS)
  learns quantizer step sizes and introduces gradient scaling that stabilizes
  their optimization.
- [PACT](https://arxiv.org/abs/1805.06085) demonstrates why learned activation
  clipping is effective for low-bit activations.
- [The official torchvision MobileNetV2 implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
  is the source of truth for the block and residual structure used by this
  repository.
- [Data-Free Quantization Through Weight Equalization and Bias Correction](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf)
  documents the particular range-balancing difficulty of MobileNet-family
  models. Cross-layer equalization is a useful fallback if initial calibration
  is unstable, but is not required for the first QAT implementation.
- [HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html) provides useful
  context for mixed-precision and integer-only deployment, although its
  Hessian/ILP machinery is too large for the present deadline.

Post-training quantization (PTQ) should be implemented only as a diagnostic. It
can validate the quantizer, establish W8A8 behavior, and expose sensitive
layers. It should not be expected to be the final low-bit result.

## 3. Quantization specification

### 3.1 Weight quantizer

For a signed `b`-bit weight tensor, use the integer range

```text
Qn = -(2^(b-1) - 1),  Qp = 2^(b-1) - 1.
```

The deliberately symmetric range omits the extra negative two's-complement
level. For output channel `c`, the fake-quantized weight is

```text
q_c = clamp(round(w_c / s_c), Qn, Qp)
w_hat_c = q_c * s_c
```

where `s_c > 0` is one learned scale for each output channel. For Conv2d, the
scale broadcasts over input-channel and kernel dimensions; for Linear, it
broadcasts over the input dimension. Initialize it using LSQ's statistic-based
rule:

```text
s_c = 2 * mean(abs(w_c)) / sqrt(Qp)
```

with a small positive floor to handle an all-zero channel. Optimize an
unconstrained parameter through a positive mapping or clamp the effective scale
away from zero. Use a straight-through estimator (STE) for rounding. Apply
LSQ-style gradient scaling to each scale, proportional to
`1 / sqrt(number_of_values_in_channel * Qp)`.

This definition must remain identical in fake quantization, integer export, and
size reconstruction. Changing signed ranges between training and export would
invalidate the accuracy measurement.

### 3.2 Activation quantizer

Use one learned scale per activation boundary, shared across batch, channel,
and spatial dimensions.

- After ReLU6, use unsigned levels `[0, 2^b - 1]` and zero-point zero.
- At the normalized model input, projection outputs, residual operands/results,
  and logits, use the symmetric signed range above.
- Do not infer signedness only from observed calibration data. It must follow
  the semantic location in the graph so a later input cannot violate the
  assumed range.

For an activation `x`, use the same clamp-round-dequantize relation as for
weights. Initialize unsigned scales from a high percentile or the known ReLU6
range rather than a single observed maximum. Initialize signed scales from
mean absolute magnitude or a symmetric percentile. Learned scales then adapt
during QAT.

An alternative PACT-style learned upper clipping value may be used for ReLU6
outputs, with `scale = alpha / (2^b - 1)`. Do not simultaneously learn an
independent `alpha` and `scale`; they represent the same degree of freedom.

### 3.3 Training gradients

The forward pass must simulate the exact discrete levels intended for export,
but it may hold the fake-quantized values in FP32 tensors. Backpropagation uses
an STE through rounding. Values inside the clipping interval pass gradients to
the underlying tensor; clipped values follow the selected LSQ/PACT scale or
clipping gradient.

The master weights, optimizer state, BatchNorm statistics, and learned scales
remain FP32 during training. This is a training representation, not the claimed
compressed representation.

### 3.4 BatchNorm and bias

Keep BatchNorm in FP32 during QAT so its statistics can adapt to quantization
noise. Near the end of training, freeze its running statistics. For export and
storage measurement, fold each BatchNorm into the preceding convolution:

```text
scale_bn = gamma / sqrt(running_var + epsilon)
W_folded = W * scale_bn
b_folded = beta + (b - running_mean) * scale_bn
```

For convolutions without bias, treat `b` as zero. Quantize the folded weights
using the final weight scales. Store the effective bias as int32, with its scale
derived from the input-activation scale and corresponding weight scale. The
report must say explicitly if an implementation instead retains FP32 bias;
that choice must be charged in the size total.

## 4. Where quantization is applied

The coverage policy should be based on tensor boundaries, not merely on module
names.

| Component | Weight policy | Output/boundary policy |
|---|---:|---:|
| Normalized CIFAR-10 input | n/a | signed, initially 8-bit |
| Stem convolution | per-channel | initially 8-bit |
| Expansion 1 x 1 convolution | per-channel target bits | activation target bits after ReLU6 |
| Depthwise 3 x 3 convolution | per-channel target bits | activation target bits after ReLU6 |
| Linear projection 1 x 1 convolution | per-channel target bits | signed target bits |
| Residual add | n/a | initially 8-bit; target 4-bit after validation |
| Final 1 x 1 convolution | per-channel | initially 8-bit |
| Global average pooling | n/a | initially 8-bit |
| Classifier | per-channel | logits initially 8-bit or FP32 for softmax only |

All Conv2d and Linear weights are compressed, including depthwise layers. The
BatchNorm parameters disappear after folding. Dropout is inactive at inference
and contributes no storage.

Residual addition needs special care. Both operands must share an integer scale
or one operand must be requantized to the result scale. The simpler defensible
design is to assign one learned scale to the block output boundary and simulate
requantization before the addition. Count any requantization multiplier/shift
metadata in the export. Do not silently add two integer tensors with unrelated
scales.

The exception policy must be empirical and minimal:

1. Begin with first convolution, last convolution/classifier, and residual
   boundaries at 8 bits.
2. Test pure W4A4 by removing the exceptions as a group.
3. Restore one exception category at a time only if pure W4A4 loses meaningful
   accuracy.
4. Test 6-bit depthwise weights only if layer sensitivity identifies them as a
   major cause; do not assume all depthwise layers require an exception.

Every exception must appear in the run configuration, W&B record, compressed
size calculation, and final report table.

## 5. Correctness gates before a sweep

Do not spend GPU time on a large sweep until the following gates pass:

1. **FP32 reconstruction:** loading the immutable baseline through the
   compression wrapper reproduces 92.69% within evaluation noise.
2. **Identity/high-precision check:** bypassed quantizers reproduce the exact
   FP32 outputs; high-bit fake quantization is numerically close.
3. **W8A8 check:** calibrated or briefly fine-tuned W8A8 has negligible
   degradation. A large loss indicates incorrect boundaries, ranges, BN
   handling, or residual scaling.
4. **Coverage audit:** a generated table lists every Conv2d and Linear module,
   its tensor shape, bit width, scale shape, signedness, and exception reason.
   No eligible layer may be silently missed.
5. **Boundary tests:** verify zero tensors, all-equal tensors, values exactly on
   clipping limits, outliers, all-zero channels, 3/4/6/8-bit ranges, and scale
   positivity.
6. **Packing round trip:** pack integer values, unpack them, and verify exact
   equality for odd tensor lengths as well as 3-bit and 4-bit formats.
7. **Fold equivalence:** Conv-BN output before folding and FP32 folded-conv
   output agree within a tight numerical tolerance.
8. **Accounting audit:** the analytical payload plus metadata total matches
   the byte length of a serialized custom packed representation.

## 6. Training strategy

Every candidate starts from the same baseline checkpoint, never from a previous
candidate. This keeps the sweep comparable.

Recommended defaults:

- 10--15 QAT epochs;
- SGD with the baseline momentum and weight decay;
- initial learning rate in the approximate `1e-4` to `1e-3` range, selected
  using W8A8/W6A6 pilot runs rather than reusing the baseline's `0.01`;
- cosine decay to zero;
- scale parameters in a separate optimizer group with no weight decay;
- the existing seed 6886 for the main sweep;
- best-checkpoint selection on a validation split, with the official test set
  evaluated only for finalized candidates where possible; and
- BatchNorm adaptation initially, followed by frozen statistics for the final
  few epochs.

A short precision transition is safer than switching directly from FP32 to
W4A4. Start with initialized quantizers at 8 bits for approximately one epoch,
move to 6 bits briefly, and then train at the target precision. The final
checkpoint must spend most of its training at its claimed target precision.
Record the schedule as part of the resolved configuration.

Changing a bit width must also change the associated learned step size; merely
changing `Qp` can turn a previously safe activation range into severe clipping.
For signed LSQ quantizers, map `s_old` to
`s_new = s_old * sqrt(Qp_old / Qp_new)`, which is the corresponding LSQ
initialization scale under unchanged tensor statistics. For unsigned ReLU6
quantizers, map `s_new = s_old * Qp_old / Qp_new` so the learned clipping bound
`Qp * s` is preserved. Log each activation boundary's scale, clipping interval,
and saturation fraction for train and validation at every epoch.

If W4A4 training is unstable, debug in this order:

1. confirm activation signedness and residual scale handling;
2. inspect learned scales for collapse or explosion;
3. lower the learning rate for pretrained weights and/or scales;
4. extend the 8-to-6-to-4 transition;
5. freeze BatchNorm later or use a lower BatchNorm momentum;
6. try cross-layer equalization before QAT; and
7. introduce precision exceptions only after the above checks.

## 7. Experiment matrix and stopping rules

### 7.1 Primary sweep

Run these four experiments first:

| Candidate | Purpose |
|---|---|
| W8A8 | Correctness and near-lossless reference |
| W6A6 | Intermediate accuracy/compression reference |
| W4A6 | Separates activation sensitivity from weight sensitivity |
| W4A4 | Primary Pareto target |

Use identical initialization, data order, epoch count, and selection rule.

### 7.2 Exception sweep

Only if uniform W4A4 is materially worse than W4A6 or W6A6, run:

- W4A4 with first and last boundaries at 8 bits;
- W4A4 with residual boundaries at 8 bits;
- W4A4 with both categories of exceptions; and
- W4A4 with only sensitivity-identified depthwise weights at 6 bits.

One-factor changes make it possible to justify each overhead. Broadly assigning
all difficult layers to 8 bits may improve accuracy but weakens the compression
claim.

### 7.3 Stretch sweep

Run W3A4 and then W3A3 only when W4A4 is stable and the core accounting is
already complete. Stop a stretch experiment early if validation accuracy fails
to recover after several epochs or scale values collapse. W2 variants are not
a sensible use of the remaining time unless all required deliverables are
already complete.

### 7.4 Final model rule

The assignment requires one final compression setting for Question 4. Select it
using this rule:

1. Discard runs that fail correctness/accounting checks.
2. Find the best compressed-model validation accuracy.
3. Among runs within **1.0 percentage point** of that accuracy, choose the one
   with the smallest complete deployable model size.
4. If sizes are effectively tied, prefer fewer exceptions and the simpler
   uniform policy.
5. Re-run the selected configuration using the same seed to verify
   reproducibility; if time allows, add one secondary seed as a robustness
   check.

This rule targets the Pareto knee without choosing a compression ratio before
evidence exists. The report may show the full sweep for Question 3, but Question
4 must highlight only the selected model.

## 8. W&B logging and parallel-coordinates chart

Each run should log at least:

- run ID, Git commit, baseline checkpoint SHA-256, seed, GPU model;
- requested weight and activation bits;
- realized bit-width policy and number/list of exceptions;
- QAT epochs, learning rate, schedule, BatchNorm freeze epoch;
- best validation accuracy and final test accuracy;
- integer weight payload bytes;
- bias bytes, scale bytes, zero-point bytes, padding bytes, layer-descriptor
  bytes, and remaining FP32 bytes;
- complete deployable model bytes and weight compression ratio;
- FP32 and quantized peak-live activation bytes and activation ratio; and
- summed activation traffic as a separately named metric.

Suggested parallel-coordinate axes are `weight_bits`, `activation_bits`,
`exception_count`, `model_size_mib`, `weight_compression_ratio`,
`activation_compression_ratio`, and `test_accuracy`. Color the lines by test
accuracy or complete model size. Do not use only the nominal bit widths: mixed
precision means nominal W4A4 candidates can have different realized sizes.

## 9. Honest compression accounting

### 9.1 Baseline definition

Use a deployable FP32 state, not the training checkpoint file, as the baseline.
Training checkpoints can contain optimizer state, configuration text, and
serialization overhead. Define:

```text
baseline_weight_bytes = 4 * number_of_deployable_FP32_parameters
```

State whether this baseline is measured before or after BN folding. The fairest
comparison folds Conv-BN for both FP32 and quantized models, because both can be
deployed that way.

The existing `baseline.pt` file is about 8.77 MiB as a serialized training
artifact, but that file size must not be used as the denominator unless an
equivalent serialization container is used for the compressed model.

### 9.2 Packed weight payload

For a tensor containing `N` values at `b` bits:

```text
payload_bytes = ceil(N * b / 8)
```

Compute this separately per packed tensor so byte-alignment padding is not
hidden. For mixed precision, sum each tensor using its actual bit width. A
PyTorch tensor containing fake-quantized FP32 values is still an FP32 tensor and
does not demonstrate storage compression.

### 9.3 Metadata and remaining state

The final size must include:

- one FP32 scale per output channel for each weight tensor;
- one activation scale per quantized boundary;
- zero points if the selected representation uses them;
- int32 biases, or FP32 biases if retained;
- requantization multipliers/shifts for residual or next-layer boundaries;
- per-tensor shape, bit width, signedness, and byte-offset descriptors;
- packing/alignment padding; and
- any weights, biases, or constants deliberately retained in FP32/FP16/INT8.

Report a table with payload and each metadata category rather than one opaque
total. The realized weight compression is

```text
weight_ratio = baseline_FP32_weight_bytes / compressed_weight_and_metadata_bytes
```

Also provide the ideal nominal ratio (`32 / b`) only as context.

Use MiB (`bytes / 2^20`) throughout the report, and label it explicitly. If the
assignment wording expects MB, optionally show decimal MB in parentheses.

### 9.4 Activation compression

Use batch-size-one, evaluation-mode inference and report **peak live activation
memory** as the primary activation measure. This is more meaningful than the
largest individual tensor because MobileNetV2 residual connections require an
input to remain live while the branch output is computed.

Construct a liveness trace from the actual forward graph:

1. Treat model input, quantized layer outputs, residual saved operands, and
   pooling/classifier boundaries as stored tensors.
2. Allocate a tensor when produced and release it after its final consumer.
3. At each event, sum the packed bytes of all live tensors plus their scale and
   zero-point metadata.
4. Record the maximum sum for FP32 and for the candidate's actual mixed
   bit-width policy.

Then report:

```text
activation_ratio = FP32_peak_live_bytes / quantized_peak_live_bytes
```

Use the same boundary set and liveness schedule in numerator and denominator.
Do not count transient Python objects or CUDA allocator reservations. Report
summed bytes written/read across all activation boundaries as a secondary
“activation traffic” metric, clearly separated from peak memory.

### 9.5 What is and is not claimed

Fake quantization on the Tesla/Volta GPUs evaluates accuracy and enables an
honest packed-storage estimate. It does **not** by itself prove inference
latency, energy, or throughput improvements: ordinary PyTorch fake-quantized
operations still execute using floating-point kernels. Do not claim runtime
speedup without custom packed integer kernels and direct benchmarks.

## 10. Four-day execution plan

### Day 1: prove correctness

- Preserve and hash the baseline checkpoint.
- Finish unit checks for quantizer levels, scales, signedness, STE behavior,
  Conv-BN folding, packing, and size accounting.
- Reproduce FP32 accuracy through the compression wrapper.
- Run PTQ diagnostics and short W8A8/W6A6 pilots.
- Correct all coverage and residual-scale problems before continuing.

### Day 2: establish the Pareto candidates

- Launch W8A8, W6A6, and W4A6 on the available GPUs.
- Launch W4A4 as soon as its short pilot is stable.
- Use one process per GPU. Independent small CIFAR-10 runs are more efficient
  and less failure-prone than multi-GPU distributed training here.
- Synchronize metrics and checkpoints after every completed run.

### Day 3: targeted recovery

- Compare W4A4 against W4A6 to identify activation sensitivity.
- Run one-factor exception experiments only where evidence warrants them.
- Run W3A4 only after a defensible W4A4 result is safe.
- Generate the W&B parallel-coordinates view and choose the finalist using the
  fixed selection rule.

### Day 4: freeze and audit

- Repeat the finalist from the immutable baseline.
- Fold BatchNorm, construct the packed representation, and reconcile its byte
  length with the accounting table.
- Recompute the peak-live activation trace.
- Export all tables/plots, finish Questions 2--4 in `main.tex`, and test every
  README reproduction command.
- Create the final artifact manifest, back up the final checkpoint, tag the Git
  commit, and avoid starting risky new variants late in the day.

## 11. Repository organization

Keep implementation concerns separate:

```text
configs/
  baseline.yaml
  quantization.yaml
  sweeps/                 # one tracked YAML per sweep family
experiments/
  baseline/
  sweeps/<run-id>/        # immutable run record
results/
  checkpoints/            # ignored locally; backed up externally
  curves/
  tables/                 # compact tracked final summaries
  figures/                # tracked report-ready plots
src/
  train.py                # baseline/QAT training orchestration
  evaluate.py             # evaluation only
  quantization.py         # quantizer mathematics
  compression.py          # folding, packing, and accounting
  compress.py             # compression/export entry point
main.tex
COMPRESSION_DESIGN.md
```

Each immutable `experiments/sweeps/<run-id>/` record should contain:

```text
resolved_config.yaml
command.txt
environment.txt
git_commit.txt
gpu.txt
metrics.json
artifacts.json
```

The artifact manifest should include logical name, local/external location,
byte size, SHA-256 checksum, source Git commit, run ID, epoch, and measured
accuracy. Never overwrite a run directory or the baseline checkpoint.

## 12. Preventing loss of progress and data

The current `.gitignore` excludes `results/checkpoints/*.pt`, baseline curve
CSV/PNG files, and `results/metrics.csv`. Therefore, a Git push alone does not
protect the existing baseline or future QAT results.

Adopt the following policy immediately:

- Store source, configurations, resolved run metadata, compact metrics, final
  tables, plots, the report, and manifests in Git.
- Store checkpoints as versioned W&B artifacts, Git LFS objects, or in durable
  cloud storage. W&B artifacts are convenient because the sweep already needs
  W&B.
- Keep at least two independent copies of the baseline and final candidate,
  neither solely on an ephemeral GPU/Colab filesystem.
- Verify copied artifacts by SHA-256 before deleting or disconnecting a runtime.
- Save both `latest` and `best` QAT states. A resumable state includes model
  master weights, learned quantizer parameters, optimizer, scheduler, epoch,
  best metric, configuration, and Python/NumPy/PyTorch/CUDA RNG states.
- Download or sync a successful checkpoint as soon as a run completes; do not
  wait until the end of a multi-run session.

Use a dedicated compression branch. Make small commits after quantizer tests,
W8A8 validation, packed-accounting validation, sweep configuration, and final
report integration. Push each milestone. Tag the verified starting point and
the final submission, for example `baseline-92.69` and `submission-final`.

Do not commit CIFAR-10 data, Python caches, temporary checkpoints, W&B local
caches, or duplicate intermediate exports. Never use a mutable filename such as
`final.pt` as the only record of a result; encode the run ID and preserve its
manifest.

## 13. Required final tables

The Q3 sweep table should contain multiple rows:

| Run | W policy | A policy | Exceptions | Test accuracy | Weight ratio | Activation ratio | Model MiB |
|---|---|---|---|---:|---:|---:|---:|
| To be measured | | | | | | | |

The Q4 table should contain exactly the one chosen configuration:

| Baseline accuracy | Compressed accuracy | Weight ratio | Activation ratio | Payload MiB | Metadata MiB | Final MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 92.69% | To be measured | To be measured | To be measured | To be measured | To be measured | To be measured |

Additionally, include a metadata breakdown and a layer-policy appendix. This
makes it possible for a grader to reproduce the compression ratios rather than
accepting only nominal `32 / bits` claims.

## 14. Final acceptance checklist

- The FP32 wrapper reproduces the baseline.
- All eligible Conv2d and Linear layers appear in the coverage audit.
- Weight and activation signedness is correct at every boundary.
- Residual scale conversion is explicit and included in storage accounting.
- Quantizer boundary cases and odd-length packing round trips pass.
- Conv-BN folding is numerically validated.
- Exported packed bytes equal the reported payload plus metadata.
- Peak-live activation memory is measured at batch size one with residual
  liveness included.
- Q3 contains the sweep and W&B parallel-coordinates chart.
- Q4 reports one model chosen by the predeclared Pareto rule.
- Final accuracy is measured from the exported/folded quantization simulation,
  not only from an earlier training checkpoint.
- Baseline and finalist checkpoints have verified external backups and
  manifests.
- README commands, environment versions, seed, and GitHub link are complete.

Following these gates should produce a defensible W4A4 result while leaving a
controlled path to either recover accuracy with narrow exceptions or explore
W3A4 if the main target succeeds early.
