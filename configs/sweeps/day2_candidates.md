# Day 2 candidate commands

All commands reload `results/checkpoints/baseline.pt`; do not use a QAT
checkpoint as the next candidate's input.  `--run-name` is deliberately unique
because the runner refuses to overwrite an existing immutable sweep record.

```bash
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 8 --activation-bits 8 --epochs 12 --run-name w8a8-seed6886
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 6 --activation-bits 6 --epochs 12 --run-name w6a6-seed6886
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 4 --activation-bits 6 --epochs 12 --run-name w4a6-seed6886
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 4 --activation-bits 4 --epochs 12 --run-name w4a4-seed6886
```

The 4-bit runs use one epoch at W8A8 and one at W6A6 before W4A4/W4A6; W6A6
uses one W8A8 transition epoch. BatchNorm running statistics are frozen at
epoch 10. The selected checkpoint is
`results/checkpoints/qat-<run>-best-target.pt`: only epochs that have reached
the requested W/A precision are eligible. Each run writes console output to
`results/logs/<run>.log` when run from the Day 2 notebook and writes its
immutable record to `experiments/sweeps/<run>/`.

## Mixed-precision follow-up

The current data says A4 is the main accuracy bottleneck, so keep A6 and test
small W6/W8 exceptions on top of W4. Run these from the immutable FP32 baseline:

```bash
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 4 --activation-bits 6 --first-last-weight-bits 8 --epochs 12 --run-name mp-w4a6-edgew8-seed6886
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 4 --activation-bits 6 --depthwise-weight-bits 6 --first-last-weight-bits 8 --epochs 12 --run-name mp-w4dw6edgew8-a6-seed6886
python -m src.qat --checkpoint results/checkpoints/baseline.pt --data-dir "$DATA_DIR" --device cuda --weight-bits 4 --activation-bits 6 --edge-bits 8 --depthwise-weight-bits 6 --first-last-weight-bits 8 --epochs 12 --run-name mp-w4dw6edgew8-a6edge8-seed6886
```

Select on validation accuracy versus the *realized* packed byte count. Evaluate
the held-out test set only after choosing the policy. The third command's
`--edge-bits` applies to normalized input and logits; it does not yet raise the
stem activation boundary, so treat it as a separate ablation rather than a
complete early-activation exception policy.
