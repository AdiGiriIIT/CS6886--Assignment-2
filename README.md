# CS6886 Assignment 2 — MobileNetV2 on CIFAR-10

This repository trains a CIFAR-10-adapted MobileNetV2 baseline and keeps training,
evaluation, and compression code separate. It is designed to run unchanged on a
CPU machine or a Colab GPU; CUDA is selected automatically when available.

## Setup

Use Python 3.10+ (Colab's current Python runtime is suitable).

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd assignment-2
python -m pip install -r requirements.txt
```

For a Colab GPU session, clone the repository in a cell, install the requirements,
then run the commands below. Dataset files are downloaded to `data/` by default;
generated checkpoints, curves, and metrics always go under `results/`.

## Reproducible commands

Run the short smoke test first. It downloads CIFAR-10 if needed, trains for one
epoch on a small fixed number of batches, writes a checkpoint, and verifies the
whole data/model/training path.

```bash
python -m src.train --config configs/baseline.yaml --epochs 1 --max-train-batches 5 --max-val-batches 2 --run-name sanity
```

Train the full baseline configured in YAML:

```bash
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --checkpoint results/checkpoints/baseline.pt
```

The compression entry point is intentionally separate from training and evaluation.
It currently validates a saved checkpoint and reports the configuration that will be
used by the custom quantization implementation in the next assignment phase:

```bash
python -m src.compress --checkpoint results/checkpoints/baseline.pt --weight-bits 8 --activation-bits 8
```

## Baseline design

* **Data:** CIFAR-10 train transform is `RandomCrop(32, padding=4)`,
  `RandomHorizontalFlip()`, `ToTensor()`, and CIFAR-10 channel normalization
  (mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)`). Test data
  uses only tensor conversion and the same normalization.
* **Model:** MobileNetV2 is adapted for 32×32 inputs by using a 3×3, stride-1
  stem and omitting ImageNet's initial downsampling. The YAML default uses width
  multiplier 1.0, dropout 0.2, BatchNorm momentum 0.1, and a 10-class classifier.
* **Training:** SGD with Nesterov momentum, weight decay, cosine LR scheduling,
  and an explicit seed. See `configs/baseline.yaml` for all values.

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
