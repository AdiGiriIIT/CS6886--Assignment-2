# CS6886 Assignment 2 — CIFAR-10 Model Compression

This repository trains a MobileNetV2 CIFAR-10 baseline and applies quantization-aware training (QAT), sparse QAT, and knowledge distillation (KD). The selected Q.5 strategy is **45% sparse mixed-precision QAT**: W4 pointwise weights, W6 depthwise weights, W8 stem/classifier weights, A6 activations, and 45% magnitude pruning of eligible W4 pointwise convolutions.



## Setup

Use Python 3.10+; CUDA is recommended for full training. The first training or evaluation command downloads CIFAR-10 to `data/` when needed. To use an existing local CIFAR-10 copy instead, pass its parent dataset directory with `--data-dir`, for example `--data-dir /path/to/cifar-data`, to `src.train`, `src.qat`, `src.prune_qat`, `src.distill`, or `src.evaluate`; torchvision will use the files already present there rather than downloading them.

```bash
git clone https://github.com/AdiGiriIIT/CS6886--Assignment-2.git
cd Assignment-2
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Generated files go to `results/` and `experiments/`. Use a new `--run-name` when rerunning QAT, pruning, or KD, since these workflows refuse to overwrite an existing experiment directory.

## Selected strategy: 45% sparse mixed-precision QAT

Run these commands from the repository root. They use seed 6886 and the fixed 45,000/5,000 train/validation split.

### 1. Train and evaluate the FP32 baseline

```bash
python -m src.train --config configs/baseline.yaml --device cuda
python -m src.evaluate --checkpoint results/checkpoints/baseline.pt --device cuda
```

`configs/baseline.yaml` defines the reproducible baseline: ImageNet-pretrained width-1.0 MobileNetV2 with a stride-1 CIFAR-10 stem, 60 epochs of SGD, and the configured augmentation and optimization settings.

### 2. Train the mixed-precision QAT model

```bash
python -m src.qat \
  --checkpoint results/checkpoints/baseline.pt \
  --weight-bits 4 --activation-bits 6 \
  --depthwise-weight-bits 6 --first-last-weight-bits 8 \
  --epochs 12 --device cuda \
  --run-name mp-w4dw6edgew8-a6-seed6886
```

This produces the selected target-precision checkpoint:

```text
results/checkpoints/qat-mp-w4dw6edgew8-a6-seed6886-best-target.pt
```

It also saves immutable configuration, history, diagnostics, metrics, and a dense packed `.qpk` artifact under `experiments/sweeps/mp-w4dw6edgew8-a6-seed6886/`.

### 3. Recover the 45% sparse model

```bash
python -m src.prune_qat \
  --checkpoint results/checkpoints/qat-mp-w4dw6edgew8-a6-seed6886-best-target.pt \
  --sparsities 0.45 --initial-sparsity 0.30 \
  --epochs 10 --pruning-warmup-epochs 4 \
  --device cuda \
  --run-name pruned-mp-w4dw6edgew8-a6-s45-seed6886
```

The command gradually ramps the nested mask from 30% to 45%, enforces it after every optimizer step, and exports a byte-checked sparse QPK2 artifact:

```text
experiments/pruning/pruned-mp-w4dw6edgew8-a6-s45-seed6886/s45/best.pt
experiments/pruning/pruned-mp-w4dw6edgew8-a6-s45-seed6886/s45/deployable_sparse.qpk
```

Evaluate it on the official CIFAR-10 test set:

```bash
python -m src.evaluate \
  --checkpoint experiments/pruning/pruned-mp-w4dw6edgew8-a6-s45-seed6886/s45/best.pt \
  --device cuda
```

`results.json` records the candidate metrics and selection decision. With a single candidate, it retains the same selection and accounting path used for a pruning sweep.

## Other required results

### Baseline

The first selected-strategy step is the full baseline reproduction. For a quick end-to-end smoke test only:

```bash
python -m src.train --config configs/baseline.yaml --device auto \
  --epochs 1 --max-train-batches 5 --max-val-batches 2 --run-name sanity
```

### Knowledge distillation

KD trains a width-0.75 student from the baseline teacher, then applies the mixed-precision QAT policy to that student:

```bash
python -m src.distill --stage fp32 \
  --teacher-checkpoint results/checkpoints/baseline.pt \
  --width-mult 0.75 --student-init teacher-slice \
  --epochs 60 --learning-rate 0.01 --device cuda \
  --run-name mobilenetv2-0.75-fp32-kd-seed6886

python -m src.distill --stage qat \
  --teacher-checkpoint results/checkpoints/baseline.pt \
  --student-checkpoint experiments/student_distillation/mobilenetv2-0.75-fp32-kd-seed6886/best.pt \
  --width-mult 0.75 \
  --weight-bits 4 --activation-bits 6 \
  --depthwise-weight-bits 6 --first-last-weight-bits 8 \
  --epochs 12 --device cuda \
  --run-name mobilenetv2-0.75-qat-kd-seed6886

python -m src.evaluate \
  --checkpoint experiments/student_distillation/mobilenetv2-0.75-qat-kd-seed6886/best.pt \
  --device cuda
```

## Configurations and custom baselines

Every QAT run starts from a baseline checkpoint. A user can modify `configs/baseline.yaml` to train their own baseline, then compress it with any supported strategy. The supplied compression scripts expect a compatible MobileNetV2 checkpoint produced by this project.

- **Baseline:** change model width/dropout, augmentation, batch size, epochs, or optimizer settings in `configs/baseline.yaml`.
- **Uniform QAT:** choose any 2--16 bit `--weight-bits` and `--activation-bits` (for example W8A8, W6A6, W4A6, or W4A4).
- **Mixed-precision QAT:** combine base W/A widths with `--depthwise-weight-bits`, `--first-last-weight-bits`, optional `--edge-bits`, and repeatable `--weight-bit-override MODULE=BITS` or `--activation-bit-override QUANTIZER=BITS` options.
- **Sparse QAT:** pass one or more values to `--sparsities` after any QAT checkpoint. The script selects the smallest artifact within `--max-validation-drop` percentage points of the unpruned QAT validation accuracy.
- **Knowledge distillation:** vary `--width-mult`, KD `--alpha`, `--temperature`, initialization (`teacher-slice` or `random`), and QAT precision settings.

For the full 30--50% sparsity frontier on the selected QAT checkpoint:

```bash
python -m src.prune_qat \
  --checkpoint results/checkpoints/qat-mp-w4dw6edgew8-a6-seed6886-best-target.pt \
  --sparsities 0.30 0.35 0.40 0.45 0.50 \
  --epochs 10 --device cuda \
  --run-name pruning-frontier-mp-w4dw6edgew8-a6-seed6886
```

## Outputs and interpretation

`src.evaluate` reports held-out CIFAR-10 test accuracy. QAT and pruning write packed artifacts and byte-exact storage accounting. Fake QAT still uses floating-point PyTorch kernels during training: packed size is storage accounting, not a latency, throughput, or energy measurement. Sparse QPK2 artifacts use bitmap plus nonzero codes and do not alone establish sparse-kernel speedup.

- `results/checkpoints/`: baseline and QAT checkpoints.
- `results/curves/` and `results/metrics.csv`: baseline curves and summary metrics.
- `experiments/sweeps/<run-name>/`: QAT configuration, history, diagnostics, metrics, and dense packed artifact.
- `experiments/pruning/<run-name>/`: per-sparsity checkpoints, sparse artifacts, metrics, and selection record.
- `experiments/student_distillation/<run-name>/`: KD checkpoints, metrics, and QAT packed artifact.
