"""Fixed-mask magnitude pruning primitives for the mixed-precision QAT model."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from .compression import is_sparse_eligible
from .quantization import QuantizedConv2d


@dataclass(frozen=True)
class PruningSummary:
    target_sparsity: float
    eligible_values: int
    masked_values: int
    per_layer: list[dict[str, object]]

    @property
    def actual_sparsity(self) -> float:
        return 0.0 if not self.eligible_values else self.masked_values / self.eligible_values


def eligible_pruning_modules(model: nn.Module) -> list[tuple[str, QuantizedConv2d]]:
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, QuantizedConv2d) and is_sparse_eligible(module)]


def global_magnitude_masks(model: nn.Module, sparsity: float) -> tuple[dict[str, torch.Tensor], PruningSummary]:
    """Create exact-count global masks over W4 pointwise-convolution master weights."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    eligible = eligible_pruning_modules(model)
    if not eligible:
        raise ValueError("no W4 1x1 quantized convolutions are eligible for pruning")
    values = torch.cat([module.weight.detach().abs().reshape(-1).cpu() for _, module in eligible])
    count = int(math.floor(values.numel() * sparsity))
    # Stable order makes tied zero/small values reproducible across runs.
    selected = torch.argsort(values, stable=True)[:count]
    global_mask = torch.ones(values.numel(), dtype=torch.bool)
    global_mask[selected] = False
    masks: dict[str, torch.Tensor] = {}; offset = 0; rows = []
    for name, module in eligible:
        size = module.weight.numel(); mask = global_mask[offset:offset + size].reshape_as(module.weight).clone()
        masks[name] = mask; pruned = int((~mask).sum())
        rows.append({"name": name, "values": size, "masked_values": pruned,
                     "sparsity": 100.0 * pruned / size})
        offset += size
    return masks, PruningSummary(sparsity, values.numel(), count, rows)


@torch.no_grad()
def enforce_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Prevent optimizer updates from regrowing pruned master weights."""
    modules = dict(model.named_modules())
    if set(masks) - set(modules):
        raise ValueError("mask names do not match this model")
    for name, mask in masks.items():
        module = modules[name]
        if not isinstance(module, QuantizedConv2d):
            raise ValueError(f"mask target {name!r} is not a QuantizedConv2d")
        module.weight.mul_(mask.to(device=module.weight.device, dtype=module.weight.dtype))


@torch.no_grad()
def masked_optimizer_step(optimizer: torch.optim.Optimizer, model: nn.Module,
                          masks: dict[str, torch.Tensor]) -> None:
    """Take one optimizer step and immediately restore every permanent zero.

    Enforcing only at epoch boundaries lets pruned weights regrow and influence
    later minibatches.  Keeping this operation next to the pruning primitives
    makes the fixed-mask invariant explicit and easy to test.
    """
    optimizer.step()
    enforce_masks(model, masks)


@torch.no_grad()
def validate_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> PruningSummary:
    """Assert permanent zeros and return exact per-layer/global sparsity."""
    modules = dict(model.named_modules()); rows = []; eligible_values = masked_values = 0
    for name, mask in masks.items():
        module = modules.get(name)
        if not isinstance(module, QuantizedConv2d): raise ValueError(f"missing mask target {name!r}")
        mask = mask.to(device=module.weight.device, dtype=torch.bool)
        if bool((module.weight[~mask] != 0).any()): raise AssertionError(f"masked weights regrew in {name}")
        count, pruned = mask.numel(), int((~mask).sum())
        eligible_values += count; masked_values += pruned
        rows.append({"name": name, "values": count, "masked_values": pruned, "sparsity": 100.0 * pruned / count})
    return PruningSummary(masked_values / eligible_values if eligible_values else 0.0,
                          eligible_values, masked_values, rows)
