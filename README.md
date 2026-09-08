# CS6886 Assignment 2 — MobileNetV2 on CIFAR-10

This repository fine-tunes torchvision's ImageNet-pretrained MobileNetV2 on CIFAR-10 and keeps training,
evaluation, and compression code separate. It is designed to run unchanged on a
CPU machine or a Colab GPU; CUDA is selected automatically when available.

## Setup

Use Python 3.10+ (Colab's current Python runtime is suitable).

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd assignment-2
# Colab already includes CUDA-enabled torch and torchvision.
python -m pip install PyYAML matplotlib
```

For a Colab GPU session, clone the repository in a cell, install the requirements,
then run the commands below. Dataset files are downloaded to `data/` by default;
generated checkpoints, curves, and metrics always go under `results/`.

## Reproducible commands

Run the short smoke test first. It downloads CIFAR-10 if needed, trains for one
epoch on a small fixed number of batches, writes a checkpoint, and verifies the
whole data/model/training path.

```bash
python -m src.train --config configs/baseline.yaml --device cuda --epochs 1 --max-train-batches 5 --max-val-batches 2 --run-name sanity
```

Train the full baseline configured in YAML:

```bash
python -m src.train --config configs/baseline.yaml --device cuda
python -m src.evaluate --checkpoint results/checkpoints/baseline.pt --device cuda
```

Day 1 compression diagnostics use from-scratch fake quantization (no framework
quantization APIs), write an all-layer coverage audit, and produce conservative
packed-weight accounting. `--evaluate` performs the PTQ diagnostic; it is not a
final deployment claim because residual requantization is deliberately still a
correctness gate for the next stage.

```bash
python -m src.compress --checkpoint results/checkpoints/baseline.pt --weight-bits 8 --activation-bits 8 --evaluate --device cuda

# Short correctness pilots, both initialized directly from baseline.pt.
python -m src.qat --checkpoint results/checkpoints/baseline.pt --weight-bits 8 --activation-bits 8 --epochs 2 --device cuda
python -m src.qat --checkpoint results/checkpoints/baseline.pt --weight-bits 6 --activation-bits 6 --epochs 2 --device cuda

# Mathematics, packing, folding, and accounting gates.
python -m unittest discover -s tests -v
```

## Day 2 QAT sweep

Use [the Day 2 notebook](notebooks/day2-quantize-sweep.ipynb) for Kaggle. Open one
copy per available GPU and assign one independent candidate to each: W8A8,
W6A6, W4A6, and W4A4. It runs the correctness gates first, uses `tee` to retain
the terminal logs, and creates immutable metadata under
`experiments/sweeps/<run-name>/`. Candidate commands are also listed in
`configs/sweeps/day2_candidates.md`.

The QAT wrapper now quantizes the signed model input/logit boundaries and each
MobileNetV2 residual addition with a shared learned scale for both operands.
It is still fake quantization: reported storage is an accounting estimate, not
a latency claim. Do not compare a training checkpoint file size to the packed
weight size.

After selecting a QAT checkpoint, make the actual deployable artifact and its
byte-verified accounting report (including batch-one peak-live activations):

```bash
python -m src.export_deploy --checkpoint results/checkpoints/qat-w4a4-seed6886-best-target.pt \
  --output results/deploy/w4a4.qpk --report results/deploy/w4a4_accounting.json
```

The exporter folds Conv--BN in both comparison representations, packs each
weight tensor at its realized bit width, and writes codes, scales, int32 biases,
descriptors, padding, and requantization fields. It intentionally reports no
PyTorch fake-QAT latency result: that path still dispatches floating-point
kernels. Integer-runtime speed needs a packed integer backend and a direct
benchmark there.

Mixed-precision weight follow-ups are supported without framework compression
APIs. For example, this keeps pointwise weights at W4, depthwise weights at W6,
stem/classifier weights at W8, and activations at A6:

```bash
python -m src.qat --checkpoint results/checkpoints/baseline.pt --device cuda \
  --weight-bits 4 --activation-bits 6 --depthwise-weight-bits 6 \
  --first-last-weight-bits 8 --epochs 12 \
  --run-name mp-w4dw6edgew8-a6-seed6886
```

See [the quantization status handoff](QUANTIZATION_STATUS_HANDOFF.md) for the
reconciled Q3/Q4 table, current QPK limitations, and recommended experiment
order. Packed size and activation-byte accounting are supported; an
integer-kernel latency/energy speedup is not yet measured.

## Distillation and sparse-pruning experiments

The Kaggle-ready [pruning/distillation notebook](notebooks/pruning_distillation_kaggle.ipynb)
implements the two follow-up documents.  Its default experiment reconstructs
the selected mixed-precision QAT **resume checkpoint**, measures 30/40/50%
global magnitude masks over only W4 1×1 convolutions, and recovers the two
best validation candidates with four epochs of fixed-mask QAT + KD.  It writes
a `QPK2` artifact with an occupancy bitmap and packed non-zero codes; this is
the only pruning result that may report a storage ratio above the dense QPK's
6.2×.  It makes no sparse-latency claim.

For the dense width-0.75 alternative in `DISTILLATION_QAT.md`:

```bash
python -m src.distill --stage fp32 --teacher-checkpoint results/checkpoints/baseline.pt --width-mult 0.75 --epochs 30 --device cuda
python -m src.distill --stage qat --teacher-checkpoint results/checkpoints/baseline.pt \
  --student-checkpoint experiments/student_distillation/mobilenetv2-0.75-fp32-kd-seed6886/best.pt \
  --width-mult 0.75 --epochs 10 --device cuda
```

`results/tables/baseline_manifest.json` records the required SHA-256 of the
immutable checkpoint. Verify it before any sweep with `sha256sum
results/checkpoints/baseline.pt`.

## Baseline design

* **Data:** a seed-6886 permutation makes a fixed 45,000/5,000
  train/validation split from CIFAR-10's official 50,000-example training set.
  The baseline's train transform uses `RandomCrop(32, padding=4)`, horizontal
  flip, RandAugment (2 operations, magnitude 7), tensor conversion, CIFAR-10
  channel normalization (mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470,
  0.2435, 0.2616)`), and Random Erasing (`p=0.10`). Validation and the official
  test set use only tensor conversion and the same normalization. Checkpoints
  are selected on validation; invoke `src.evaluate` once the checkpoint is
  selected to measure held-out test accuracy. The augmentation is configurable
  and opt-in at the data-loader level, so existing QAT commands retain their
  documented crop/flip protocol unless explicitly changed.
* **Model:** torchvision's official MobileNetV2 is initialized with public
  `IMAGENET1K_V2` weights, adapted for 32×32 inputs with a stride-1 stem, and
  given a newly initialized 10-class classifier. The YAML default uses width
  multiplier 1.0 and dropout 0.2.
* **Training:** the new classifier is warmed up for two epochs with the pretrained
  backbone frozen, then all layers are fine-tuned for 60 epochs using SGD with
  Nesterov momentum, `1e-4` weight decay, `0.1` label smoothing, cosine LR
  scheduling, and an explicit seed. See `configs/baseline.yaml` for all values.

Every checkpoint stores its model configuration and normalization metadata, so
`src.evaluate` can reconstruct the exact architecture. Each training run writes a
CSV history to `results/curves/<run-name>.csv`, appends final metrics to
`results/metrics.csv`, and saves its best validation checkpoint in
`results/checkpoints/`.

## Colab hand-off

After the run, download `results/` from the Colab file browser (or zip it in a
notebook cell), copy it into your local clone, review it, and commit/push locally.
Keep raw `data/` and generated `results/` files out of Git; the included `.gitignore`
does this while retaining the empty result directory structure.
