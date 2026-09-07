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
        # Non-persistent observer state is deliberately excluded from QAT
        # checkpoints. It is reset and summarized once per train/validation pass.
        self._observed_values: torch.Tensor | None = None
        self._saturated_values: torch.Tensor | None = None

    def reset_observer(self) -> None:
        self._observed_values = self._saturated_values = None

    def observe(self, value: torch.Tensor) -> None:
        """Accumulate clipping statistics without retaining model activations."""
        qn, qp = integer_range(self.bits, self.signed)
        with torch.no_grad():
            codes = value.detach() / self.scale.detach().clamp_min(1e-8)
            saturated = ((codes < qn) | (codes > qp)).sum(dtype=torch.int64)
            if self._observed_values is None:
                self._observed_values = torch.zeros((), device=value.device, dtype=torch.int64)
                self._saturated_values = torch.zeros((), device=value.device, dtype=torch.int64)
            self._observed_values.add_(value.numel())
            self._saturated_values.add_(saturated)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized):
            with torch.no_grad():
                self.scale.copy_(lsq_scale_init(value, self.bits, self.signed)); self.initialized.fill_(True)
        self.observe(value)
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


def _transition_scale_multiplier(old_bits: int, new_bits: int, signed: bool) -> float:
    """Map a learned step size to the statistically equivalent new precision.

    Signed scales use the LSQ initialization relation ``s ∝ 1/sqrt(Qp)``.
    ReLU6's unsigned scale represents a learned clipping bound ``Qp * s``;
    preserving that bound is the corresponding PACT-style transition policy.
    """
    _, old_qp = integer_range(old_bits, signed)
    _, new_qp = integer_range(new_bits, signed)
    return math.sqrt(old_qp / new_qp) if signed else old_qp / new_qp


def set_quantizer_bits(module: nn.Module, weight_bits: int, activation_bits: int) -> None:
    """Transition precision while rescaling learned steps for the new range.

    Leaving an 8-bit signed scale unchanged at 4 bits shrinks its clipping
    bound from ``127*s`` to ``7*s``.  This function instead maps scales using
    the documented LSQ/PACT policy above. Uninitialized activation quantizers
    are left alone so their first observed tensor initializes at active bits.
    """
    for child in module.modules():
        if isinstance(child, WeightFakeQuantizer):
            if child.bits != weight_bits:
                with torch.no_grad():
                    child.scale.mul_(_transition_scale_multiplier(child.bits, weight_bits, signed=True))
            child.bits = weight_bits
        elif isinstance(child, ActivationFakeQuantizer):
            if not getattr(child, "fixed_bits", False):
                if child.bits != activation_bits and bool(child.initialized):
                    with torch.no_grad():
                        child.scale.mul_(_transition_scale_multiplier(child.bits, activation_bits, child.signed))
                child.bits = activation_bits


def apply_mixed_weight_policy(module: nn.Module, default_bits: int,
                              depthwise_bits: int | None = None,
                              first_last_bits: int | None = None) -> dict[str, int]:
    """Apply a reproducible mixed-precision policy and return realized bits.

    ``first_last_bits`` covers the stem and the final weighted layer (the
    classifier), while ``depthwise_bits`` covers depthwise convolutions.  The
    first/last rule takes precedence if a category ever overlaps.  Calling
    this after the progressive precision transition makes exceptions active
    only at the target stage, keeping the 8 -> 6 -> target warm-up intact.
    """
    weighted = [(name, child) for name, child in module.named_modules()
                if isinstance(child, (QuantizedConv2d, QuantizedLinear))]
    realized: dict[str, int] = {}
    for index, (name, child) in enumerate(weighted):
        bits = default_bits
        if (depthwise_bits is not None and isinstance(child, QuantizedConv2d)
                and child.groups == child.weight.shape[0]
                and child.weight.shape[1] == 1):
            bits = depthwise_bits
        if first_last_bits is not None and index in (0, len(weighted) - 1):
            bits = first_last_bits
        if child.weight_quantizer.bits != bits:
            with torch.no_grad():
                child.weight_quantizer.scale.mul_(_transition_scale_multiplier(
                    child.weight_quantizer.bits, bits, signed=True))
        child.weight_quantizer.bits = bits
        realized[name] = bits
    return realized


def apply_quantizer_bit_overrides(module: nn.Module, overrides: dict[str, int],
                                  quantizer_type: type[nn.Module]) -> dict[str, int]:
    """Apply named target-stage precision exceptions with transition-safe scales.

    Names are the ``named_modules`` paths written to the activation diagnostics
    (for activation quantizers) or returned by the coverage audit (for weight
    wrappers).  This intentionally accepts only exact names: a misspelled
    exception must fail rather than silently quantizing a different layer.
    """
    modules = dict(module.named_modules())
    missing = sorted(set(overrides) - set(modules))
    if missing:
        raise ValueError(f"precision overrides contain unknown modules: {missing}")
    applied: dict[str, int] = {}
    for name, bits in overrides.items():
        child = modules[name]
        if not isinstance(child, quantizer_type):
            raise ValueError(f"precision override target {name!r} is not a {quantizer_type.__name__}")
        if isinstance(child, WeightFakeQuantizer):
            if child.bits != bits:
                with torch.no_grad():
                    child.scale.mul_(_transition_scale_multiplier(child.bits, bits, signed=True))
            child.bits = bits
        elif isinstance(child, ActivationFakeQuantizer):
            if child.bits != bits and bool(child.initialized):
                with torch.no_grad():
                    child.scale.mul_(_transition_scale_multiplier(child.bits, bits, child.signed))
            child.bits = bits
            # Keep this target-stage exception from being reset by a later
            # global precision update (for example when resuming QAT).
            child.fixed_bits = True
        applied[name] = bits
    return applied


def apply_weight_wrapper_overrides(module: nn.Module, overrides: dict[str, int]) -> dict[str, int]:
    """Apply exact named W-bit exceptions after category-level policies."""
    wrappers = dict(module.named_modules())
    quantizer_overrides: dict[str, int] = {}
    for name, bits in overrides.items():
        if name not in wrappers or not isinstance(wrappers[name], (QuantizedConv2d, QuantizedLinear)):
            raise ValueError(f"weight override target {name!r} is not a quantized Conv2d or Linear")
        quantizer_overrides[f"{name}.weight_quantizer"] = bits
    return apply_quantizer_bit_overrides(module, quantizer_overrides, WeightFakeQuantizer)


def apply_weight_bit_map(module: nn.Module, realized: dict[str, int]) -> None:
    """Restore the exact per-layer bit policy recorded in a checkpoint."""
    modules = dict(module.named_modules())
    missing = sorted(set(realized) - set(modules))
    if missing:
        raise ValueError(f"checkpoint weight policy contains unknown modules: {missing}")
    for name, bits in realized.items():
        child = modules[name]
        if not isinstance(child, (QuantizedConv2d, QuantizedLinear)):
            raise ValueError(f"weight policy target is not quantized: {name}")
        child.weight_quantizer.bits = int(bits)


def reset_activation_observers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, ActivationFakeQuantizer):
            child.reset_observer()


def activation_quantizer_diagnostics(module: nn.Module) -> list[dict[str, object]]:
    """Return scale and clipping data accumulated since the last observer reset."""
    rows = []
    for name, child in module.named_modules():
        if not isinstance(child, ActivationFakeQuantizer):
            continue
        qn, qp = integer_range(child.bits, child.signed)
        scale = float(child.scale.detach().clamp_min(1e-8).cpu())
        observed = 0 if child._observed_values is None else int(child._observed_values.cpu())
        saturated = 0 if child._saturated_values is None else int(child._saturated_values.cpu())
        rows.append({"quantizer": name, "bits": child.bits, "signed": child.signed,
                     "scale": scale, "clip_min": qn * scale, "clip_max": qp * scale,
                     "observed_values": observed, "saturated_values": saturated,
                     "saturation_percent": 0.0 if not observed else 100.0 * saturated / observed})
    return rows
