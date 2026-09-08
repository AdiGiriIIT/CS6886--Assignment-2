# Short-horizon pruning and knowledge-distillation plan

> **Implementation update:** Pruning and knowledge distillation are now
> independent experiments. `notebooks/kd-qat.ipynb` uses supervised-only,
> gradual fixed-mask recovery through `src.prune_qat`; KD is confined to
> `notebooks/knowledge-distillation-qat.ipynb` and `src.distill.py`. The older
> combined QAT+KD pruning proposal below is retained only as design history.

Date: 8 September 2026

## Decision

Yes, the selected QAT model can be the starting student for pruning and can be
fine-tuned with knowledge distillation (KD). Use the **resume checkpoint**, not
the packed `.qpk` file:

```text
Mixed_Precision_artifacts/
  mp-w4dw6edgew8-a6-seed6886-artifacts/results/checkpoints/
  qat-mp-w4dw6edgew8-a6-seed6886-best-target.pt
```

The starting student is the handoff's selected `mp-w4dw6edgew8-a6` policy:
W4 pointwise weights, W6 depthwise weights, W8 stem/classifier weights, and A6
activations. Its three-seed mean test top-1 is 91.58%, and its seed-6886
validation top-1 is 92.52%. The immutable 93.69%-test FP32 baseline should be
the teacher.

With less than a day available, the recommended experiment is **one-shot
unstructured magnitude pruning of only the W4 1x1 convolutions, followed by
3--4 epochs of QAT+KD recovery with a fixed mask**. First measure zero-shot
validation accuracy at 30%, 40%, and 50% sparsity; train at most the best two
viable levels. Do not attempt architecture search, iterative prune/retrain
cycles, or a new narrow student.

There is an important reporting constraint: the current QPK format is dense.
Merely setting weights to zero does **not** reduce its 1.351 MiB size. A smaller
claim requires either a new sparse representation (recommended: bitmap plus
packed nonzero codes) or a clearly labelled compressed-file result. KD itself
does not reduce size or inference cost; here it is a recovery loss for the
pruned, already-quantized student.

## Why this is the best deadline-compatible combination

PyTorch's pruning utilities support global unstructured pruning and implement
the mask as a parameter reparametrization. The reparametrization must be made
permanent before export, or an equivalent fixed mask must be maintained by our
custom QAT loop. The official tutorial also distinguishes unstructured from
structured pruning ([PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)).

KD trains a student from both labels and the teacher's softened class
distribution. Temperature exposes class similarities that a hard CIFAR-10
label cannot. This is the original formulation in
[Hinton, Vinyals, and Dean](https://arxiv.org/abs/1503.02531), and the official
[PyTorch KD tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
shows the same soft-target training pattern.

The existing artifact is unusually convenient for a conservative pilot:

| Weight group | Values | Packed payload | Treatment |
|---|---:|---:|---|
| W4 1x1 convolutions | 2,124,672 | 1,062,336 B | Globally prune |
| W6 depthwise convolutions | 64,224 | 48,168 B | Exclude |
| W8 stem/classifier | 13,664 | 13,664 B | Exclude |

Thus almost all weight values are in the W4 1x1 group, while the handoff's
empirically sensitive W6 depthwise and W8 edge exceptions stay intact.
Unstructured pruning also leaves tensor shapes and residual connections
unchanged, avoiding MobileNetV2 channel-surgery risk. Structured channel
pruning can produce an ordinary smaller dense model, but modern residual
connections need architectural adaptations; the network-slimming paper makes
that qualification explicitly
([Liu et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)).

## Proposed experiment

### 1. Reconstruct and verify the student

Rebuild the quantized wrapper exactly as the QAT runner does, load the selected
checkpoint, and reapply its realized mixed-precision policy. Do not load the
`.qpk`: it is a folded deployment/accounting artifact and contains neither the
training state nor a supported integer backend.

Before pruning, reproduce 92.52% seed-6886 validation accuracy (allow only
normal deterministic/evaluation tolerance). Keep the existing seed-6886
45,000/5,000 train/validation split. The official test set remains untouched
until one final candidate is selected.

### 2. Define one global, fixed pruning mask

Eligible tensors are `QuantizedConv2d.weight` tensors whose realized precision
is W4 and kernel is 1x1. This selects all 2,124,672 W4 values and excludes:

- the W8 stem and classifier;
- all W6 depthwise convolutions;
- bias, BatchNorm, and quantizer-scale parameters; and
- all activations (activation compression therefore remains 5.330x).

Rank eligible values globally by `abs(weight)` and mask the smallest fraction.
Use exact-count selection so reported sparsity equals the target. Global rather
than per-layer thresholds let parameter-heavy layers absorb more pruning, but
record per-layer sparsity to catch a collapsed layer.

Evaluate fixed masks at 30%, 40%, and 50% without training. Zero remains exactly
representable by the symmetric quantizers. Stop escalating sparsity when
zero-shot validation loss becomes clearly excessive; a practical gate is a
drop greater than 5 percentage points, since four recovery epochs are unlikely
to repair it reliably.

During recovery, enforce `weight *= mask` immediately after every optimizer
step (and before every evaluation/export), or mask the effective weight in the
forward pass and its gradient. A one-time call that zeroes weights is
insufficient because SGD can regrow them. Save the masks and verify both
`masked_nonzero == 0` and the exact global sparsity after reload.

### 3. Recover with QAT and response-based KD

Teacher: reconstruct `results/checkpoints/baseline.pt`, verify its recorded
SHA-256 is
`73a0bc35a65ebd049fc8b698f2816b3feb6ed1b53e41ec373c674c325a1f2837`,
put it in `eval()` mode, and run it under `torch.no_grad()`. The teacher stays
FP32 and is not exported.

Student: keep its final W4/W6/W8 and A6 precision from the first recovery
batch--do **not** repeat the 8-to-6 precision transition. Keep BatchNorm in eval
mode because the selected checkpoint is already trained and the recovery is
short. Continue optimizing master weights and learned quantizer scales, with
no weight decay on scales.

For student logits `z_s`, teacher logits `z_t`, labels `y`, temperature `T`,
and KD weight `alpha`, use:

```text
L_hard = cross_entropy(z_s, y)
L_soft = KL(log_softmax(z_s / T), softmax(z_t / T), reduction="batchmean")
L = (1 - alpha) * L_hard + alpha * T^2 * L_soft
```

Use one fixed setting to avoid a hyperparameter sweep: `T=2`, `alpha=0.5`,
SGD with the existing momentum/Nesterov settings, initial LR `1e-4`, cosine
decay, and 4 epochs. Select the epoch and sparsity only by validation accuracy.
The `T^2` factor preserves the soft-loss gradient scale as temperature changes,
following the original KD formulation.

Run order under the deadline:

1. unpruned checkpoint validation and 30/40/50% zero-shot validations;
2. four-epoch QAT+KD recovery for the best size/accuracy zero-shot candidate;
3. only if time remains, recover the next-lower sparsity candidate;
4. compare against an optional four-epoch **pruning + hard-label-only** control
   at the chosen sparsity; and
5. evaluate the official test set once for the validation-selected result.

Caching teacher logits could reduce repeated teacher work, but it is unsafe
with random crop/flip augmentation unless cached per augmented image. For only
one or two short runs, online teacher inference is simpler and defensible.

## Sparse storage accounting

The selected QPK is 1,416,800 bytes, of which 1,124,168 bytes are packed weight
codes and 1,062,336 bytes are the eligible W4 code stream. A simple sparse W4
format can store:

```text
one occupancy bit per original W4 position
+ four bits per retained nonzero W4 value
+ the unchanged W6/W8 payload and existing metadata
```

For W4, the bitmap alone costs 25% of the dense W4 payload, so sparsity must
exceed 25% merely to break even. The optimistic byte projections below retain
all current QPK metadata and add only the bitmap; they exclude any new section
descriptors/alignment and therefore are upper-bound goals, not measured
artifacts.

| W4 sparsity | Projected total | MiB | Ratio vs 8,878,504-B FP32 state | Change vs dense QPK |
|---:|---:|---:|---:|---:|
| 30% | about 1,363,700 B | 1.301 | 6.51x | 3.7% smaller |
| 40% | about 1,257,500 B | 1.199 | 7.06x | 11.2% smaller |
| 50% | about 1,151,300 B | 1.098 | 7.71x | 18.7% smaller |

This modest gain is the price of adding sparsity after already reaching W4.
CSR/COO indices would usually be worse for individual 4-bit values, so do not
claim their theoretical sparse parameter count as serialized size. Extend the
exporter, round-trip the bitmap/nonzero stream, and compare its exact file
length to accounting before reporting a new compression ratio. Also report
both overall sparsity and eligible-W4 sparsity.

Sparse storage does not establish latency. Unstructured masks need an actual
sparse low-bit kernel to skip work; the current fake-QAT path still calls dense
floating-point convolutions. Official TensorFlow guidance likewise separates
pruning's compression benefit from acceleration that depends on compatible
sparse inference support
([pruning overview](https://www.tensorflow.org/model_optimization/guide/pruning)).

## Alternatives considered

### KD without pruning

A 2--4 epoch KD fine-tune of the existing quantized student is very cheap and
may recover accuracy, but it produces the same architecture, precision, and
storage. It is useful only as a control or as preparation for a more aggressive
pruning level; it is not additional compression.

### Distil into a narrower MobileNetV2

A width-multiplier 0.75 or 0.5 student would yield real dense parameter and
compute reductions. However, its tensor shapes differ, so the selected QAT
weights cannot be loaded directly, and it needs training followed by a new QAT
stage. This is a sound longer-term experiment but too risky for the remaining
assignment window.

### Structured channel/filter pruning

This can yield a conventional smaller dense network and real dense-kernel
speedups. In MobileNetV2, though, expansion, depthwise, projection, BatchNorm,
and residual-compatible dimensions must be rewritten consistently, then
quantizer scales and packing descriptors regenerated. It is not a safe
last-day change.

### Iterative or automatic joint pruning/quantization

Joint optimization is a genuine research direction; recent work explicitly
targets joint structured pruning and QAT
([GETA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_Automatic_Joint_Structured_Pruning_and_Quantization_for_Efficient_Neural_Network_CVPR_2025_paper.html)).
Its multi-stage/search machinery is disproportionate here. A fixed one-shot
mask plus short QAT+KD recovery is easier to finish, audit, and explain.

## Acceptance and reporting checklist

A pruned result should replace neither the handoff result nor its artifact
unless all of these pass:

- validation-selected accuracy is measured at the claimed final sparsity and
  mixed precision;
- accuracy drop is reported against both 93.69% FP32 and 91.58% mean-QAT
  references (with the seed-6886 comparison identified separately);
- masks survive checkpoint reload, masked values remain zero after training,
  and per-layer/overall sparsity is logged;
- exact sparse encoding round-trips and its file byte length matches the size
  report;
- activation ratio remains separately reported as 5.330x; and
- no latency, throughput, or energy improvement is claimed without a sparse
  low-bit backend benchmark.

If sparse export cannot be completed and tested in time, report pruning as an
**accuracy-at-sparsity experiment** and retain 1.351 MiB / 6.267x as the only
validated deployable QPK result. That is more defensible than presenting a
zero-filled dense file as additional compression.
