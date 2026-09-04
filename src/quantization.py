"""From-scratch fake quantizers used by PTQ diagnostics and QAT pilots."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def integer_range(bits: int, signed: bool) -> tuple[int, int]:
    if not 2 <= bits <= 16:
        raise ValueError("bits must be between 2 and 16")
    return (-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value): return value.round()
    @staticmethod
    def backward(ctx, gradient): return gradient

def ste_round(value: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(value)

def lsq_scale_init(value: torch.Tensor, bits: int, signed: bool, per_channel: bool = False) -> torch.Tensor:
    _, qp = integer_range(bits, signed)
    mean = value.detach().abs().mean(dim=tuple(range(1, value.ndim)), keepdim=True) if per_channel else value.detach().abs().mean()
    return (2 * mean / math.sqrt(qp)).clamp_min(1e-8)

def fake_quantize(value: torch.Tensor, scale: torch.Tensor, bits: int, signed: bool) -> torch.Tensor:
    qn, qp = integer_range(bits, signed)
    clipped = (value / scale.clamp_min(1e-8)).clamp(qn, qp)
    return ste_round(clipped) * scale.clamp_min(1e-8)

class WeightFakeQuantizer(nn.Module):
    """Learned per-output-channel symmetric signed weight quantizer."""
    def __init__(self, weight: torch.Tensor, bits: int):
        super().__init__(); self.bits = bits
        self.scale = nn.Parameter(lsq_scale_init(weight, bits, True, per_channel=True))
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        _, qp = integer_range(self.bits, True)
        gradient_scale = 1.0 / math.sqrt(weight[0].numel() * qp)
        scale = self.scale * gradient_scale + self.scale.detach() * (1 - gradient_scale)
        return fake_quantize(weight, scale, self.bits, True)

class ActivationFakeQuantizer(nn.Module):
    """Learned per-tensor activation quantizer; unsigned only after ReLU6."""
    def __init__(self, bits: int, signed: bool, initial_scale: float | None = None):
        super().__init__(); self.bits, self.signed = bits, signed
        _, qp = integer_range(bits, signed)
        initial = initial_scale if initial_scale is not None else (6.0 / qp if not signed else 1.0)
        self.scale = nn.Parameter(torch.tensor(float(initial)))
        self.register_buffer("initialized", torch.tensor(initial_scale is not None))
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized):
            with torch.no_grad():
                self.scale.copy_(lsq_scale_init(value, self.bits, self.signed)); self.initialized.fill_(True)
        return fake_quantize(value, self.scale, self.bits, self.signed)

class QuantizedConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, bits: int):
        super().__init__(); self.weight = nn.Parameter(module.weight.detach().clone())
        self.bias = nn.Parameter(module.bias.detach().clone()) if module.bias is not None else None
        self.weight_quantizer = WeightFakeQuantizer(self.weight, bits)
        self.stride, self.padding, self.dilation, self.groups = module.stride, module.padding, module.dilation, module.groups
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value, self.weight_quantizer(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantizedLinear(nn.Module):
    def __init__(self, module: nn.Linear, bits: int):
        super().__init__(); self.weight = nn.Parameter(module.weight.detach().clone())
        self.bias = nn.Parameter(module.bias.detach().clone()) if module.bias is not None else None
        self.weight_quantizer = WeightFakeQuantizer(self.weight, bits)
    def forward(self, value: torch.Tensor) -> torch.Tensor: return F.linear(value, self.weight_quantizer(self.weight), self.bias)

class QuantizedReLU6(nn.Module):
    def __init__(self, bits: int):
        super().__init__(); self.activation_quantizer = ActivationFakeQuantizer(bits, False, 6.0 / (2 ** bits - 1))
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.activation_quantizer(F.relu6(value))


class QuantizedResidualAdd(nn.Module):
    """Fake-quantize both operands to one learned signed boundary scale.

    Keeping one scale for the two operands makes the simulated add faithful to
    the simple integer deployment policy: both integer tensors can be added
    directly, then requantized at the same block-output boundary.
    """
    def __init__(self, bits: int):
        super().__init__()
        self.activation_quantizer = ActivationFakeQuantizer(bits, signed=True)

    def forward(self, skip: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        quantizer = self.activation_quantizer
        return quantizer(quantizer(skip) + quantizer(branch))


def set_quantizer_bits(module: nn.Module, weight_bits: int, activation_bits: int) -> None:
    """Apply a precision-transition step without replacing learned scales."""
    for child in module.modules():
        if isinstance(child, WeightFakeQuantizer):
            child.bits = weight_bits
        elif isinstance(child, ActivationFakeQuantizer):
            if not getattr(child, "fixed_bits", False):
                child.bits = activation_bits
