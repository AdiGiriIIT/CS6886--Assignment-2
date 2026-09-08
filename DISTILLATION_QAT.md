Yes—this is feasible within 24 hours if we test only one student, preferably MobileNetV2 `width_mult=0.75`.

The clean sequence is:

```text
FP32 width-1.0 teacher
        ↓ knowledge distillation
FP32 width-0.75 student
        ↓ initialize QAT from trained student
Quantized width-0.75 student
```

The selected width-1.0 QAT weights cannot directly initialize the narrower model because most convolution shapes differ. They can still be used as a teacher, but the 93.69%-accurate FP32 baseline is probably the stronger and simpler teacher.

## Recommended two-stage approach

### Stage 1: train the narrow FP32 student with KD

Construct:

```python
mobilenet_v2(weights=None, width_mult=0.75)
```

Adapt its stem and classifier exactly like the current CIFAR-10 model. Train it using:

```text
L = (1 - α) CE(student_logits, labels)
    + α T² KL(student_logits / T, teacher_logits / T)
```

Recommended starting configuration:

- Teacher: current FP32 width-1.0 baseline, frozen in `eval()` mode.
- Student: FP32 MobileNetV2-0.75.
- Temperature: `T=2`.
- KD weight: `α=0.5`.
- Epochs: initially 30.
- Validate every epoch and retain the best checkpoint.
- Use the existing augmentation and seed-6886 split.
- Do not freeze the randomly initialized student backbone.

This follows the standard teacher/student setup described in the [official PyTorch KD tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).

### Stage 2: QAT on the trained student

Load the best FP32 student checkpoint, construct its quantized wrapper, and perform QAT exactly as before:

```text
FP32 student weights
    → W8A8 warm-up
    → W6A6 transition
    → target mixed precision
```

A reasonable short QAT schedule would be:

- 1 epoch W8A8
- 1 epoch W6A6
- 6–8 epochs at target precision
- Total: 8–10 epochs
- Freeze BatchNorm for the final 2–3 epochs

The existing policy can initially be reused structurally:

```text
Default weights:      W4
Depthwise weights:    W6
Stem/classifier:      W8
Activations:          A6
```

However, this becomes a new precision policy for a different architecture. It must be validated again rather than inheriting the width-1.0 result automatically.

Continuing KD during QAT is worthwhile:

```text
FP32 width-1.0 teacher
        ↓
quantized width-0.75 student
```

This lets the teacher help the student cope with both reduced width and quantization noise.

## Why not train the narrow student directly under QAT?

It is possible, but not my first choice. Starting a randomly initialized narrow network with quantization introduces three simultaneous difficulties:

- learning CIFAR-10 representations;
- compensating for reduced model capacity;
- adapting to low-bit weight and activation noise.

Training the FP32 student first separates architecture compression from quantization. It also gives us a checkpoint that answers an important question: how much accuracy was lost because of width reduction, before quantization was added?

That produces a much cleaner comparison:

| Model | What it measures |
|---|---|
| FP32 width 1.0 | Original reference |
| QAT width 1.0 | Quantization cost |
| FP32 width 0.75 + KD | Width-reduction cost |
| QAT width 0.75 + KD | Combined compression cost |

## Can the existing QAT weights be reused?

There are three meanings of “reuse”:

1. **Direct initialization:** generally no. Width scaling changes input and output channel dimensions.

2. **As a teacher:** yes. The selected QAT model can produce soft targets for the narrower student.

3. **Partial channel transfer:** technically yes, but risky. We could select channels from the width-1.0 checkpoint and slice adjacent convolution and BatchNorm tensors consistently. In MobileNetV2, expansion, depthwise, projection, and residual-connected channels must all agree. This becomes structured pruning/model surgery and is the main implementation risk.

For the deadline, I would use the FP32 baseline teacher throughout. If we want the final student to mimic quantization-specific behaviour, an optional experiment could use the selected QAT model as teacher during Stage 2—but I would not introduce dual teachers initially.

## How long should it take?

The recorded QAT runs take approximately 48–56 seconds per epoch on the previous Kaggle GPU. Therefore:

- 30 FP32 KD epochs: likely around 20–40 minutes;
- 8–10 QAT+KD epochs: likely around 8–15 minutes;
- validation, exports, tests, and one retry: allow another 30–60 minutes.

KD adds a frozen teacher forward pass to every training batch. The narrower student is cheaper than the current model, so an epoch may still be reasonably close to the existing QAT time.

Thus raw training time is probably not the bottleneck. The bottlenecks are:

- The current code deliberately rejects pretrained torchvision initialization when `width_mult != 1.0`, so the student starts randomly initialized.
- Choosing enough FP32 epochs for convergence.
- Ensuring checkpoint reconstruction supports the new width.
- Revalidating mixed-precision exceptions.
- Producing and auditing a new QPK artifact.
- Avoiding a large sweep of widths, KD temperatures, and bit policies.

The lack of pretrained initialization is the biggest accuracy risk. Thirty epochs may be sufficient with KD, but the existing baseline used 60 epochs with ImageNet initialization. I would permit extension to 45 epochs only if validation is still clearly improving at epoch 30.

## Minimum viable experiment

Given the deadline, I recommend exactly this:

1. Train one `width_mult=0.75` FP32 student with KD for 30 epochs.
2. If validation is promising—approximately 91% or better—run 8–10 epochs of QAT+KD.
3. Export and measure its packed representation.
4. Do not try width 0.5 unless the 0.75 pipeline finishes early.
5. Keep the existing width-1.0 QAT result as the safe submission result until the narrow quantized student passes validation and export checks.

A width-0.5 student has greater compression potential but substantially less capacity and is more likely to require learning-rate, augmentation, and KD tuning.

## Suggested isolation

If implemented, I would place it under:

```text
student_distillation/
├── README.md
├── train_student.py
├── qat_student.py
├── distillation.py
├── configs/
│   ├── mobilenetv2_075_kd.yaml
│   └── mobilenetv2_075_qat_kd.yaml
└── tests/
    └── test_student_pipeline.py
```

It can import stable model, data, quantization, packing, and evaluation utilities from `src/` while keeping new training logic and artifacts separate.

So: training time itself is manageable. The real risk is getting a randomly initialized narrow student to converge well and then correctly reproducing the entire QAT/export protocol. A single width-0.75 run is realistic; a width/KD/precision sweep is not.