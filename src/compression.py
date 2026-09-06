"""Coverage, folding, packing, and accounting for custom quantization."""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from .quantization import (ActivationFakeQuantizer, QuantizedConv2d,
                           QuantizedLinear, QuantizedReLU6, QuantizedResidualAdd,
                           integer_range)


class QuantizedInvertedResidual(nn.Module):
    """MobileNetV2 block with quantized signed projection boundaries.

    A residual block obtains this boundary quantization from ``residual_add``:
    both operands and the result are quantized using its shared signed scale.
    A non-residual block has no add, so its Conv--BN projection output needs an
    explicit signed quantizer before it becomes the next block's input.
    """
    def __init__(self, module: nn.Module, activation_bits: int):
        super().__init__()
        self.conv = module.conv
        self.use_res_connect = module.use_res_connect
        self.residual_add = QuantizedResidualAdd(activation_bits) if self.use_res_connect else None
        self.output_quantizer = (None if self.use_res_connect
                                 else ActivationFakeQuantizer(activation_bits, signed=True))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        branch = self.conv(value)
        return self.residual_add(value, branch) if self.use_res_connect else self.output_quantizer(branch)


class QuantizedMobileNet(nn.Module):
    """Model-level boundaries which are not represented by a torchvision module."""
    def __init__(self, model: nn.Module, activation_bits: int, edge_bits: int | None = None):
        super().__init__()
        self.model = model
        edge = activation_bits if edge_bits is None else edge_bits
        self.input_quantizer = ActivationFakeQuantizer(edge, signed=True)
        self.logit_quantizer = ActivationFakeQuantizer(edge, signed=True)
        if edge_bits is not None:
            self.input_quantizer.fixed_bits = True
            self.logit_quantizer.fixed_bits = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.logit_quantizer(self.model(self.input_quantizer(value)))

def checkpoint_size_mb(path: str | Path) -> float: return Path(path).stat().st_size / (1024 ** 2)
def count_parameter_bits(state_dict: dict[str, torch.Tensor], bits_per_value: int = 32) -> int:
    return sum(tensor.numel() * bits_per_value for tensor in state_dict.values())

def pack_signed(values: torch.Tensor, bits: int) -> bytes:
    """Pack signed levels exactly, including 3-bit values and odd tensor sizes."""
    qn, qp = integer_range(bits, True); flat = values.detach().cpu().reshape(-1).to(torch.int64).tolist()
    if any(v < qn or v > qp for v in flat): raise ValueError("integer value outside requested signed range")
    accumulator = used = 0; output = bytearray()
    for value in flat:
        accumulator |= (value - qn) << used; used += bits
        while used >= 8: output.append(accumulator & 255); accumulator >>= 8; used -= 8
    if used: output.append(accumulator)
    return bytes(output)

def unpack_signed(payload: bytes, count: int, bits: int) -> torch.Tensor:
    qn, _ = integer_range(bits, True); accumulator = used = offset = 0; output = []
    while len(output) < count:
        while used < bits:
            if offset >= len(payload): raise ValueError("payload ended before all values were unpacked")
            accumulator |= payload[offset] << used; offset += 1; used += 8
        output.append((accumulator & ((1 << bits) - 1)) + qn); accumulator >>= bits; used -= bits
    return torch.tensor(output, dtype=torch.int64)

def fold_conv_bn(conv: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Return an eval-mode Conv2d equivalent to ``batch_norm(conv(x))``."""
    folded = copy.deepcopy(conv)
    if folded.bias is None: folded.bias = nn.Parameter(torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype))
    scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    folded.weight.data.mul_(scale.reshape(-1, 1, 1, 1))
    folded.bias.data.copy_(batch_norm.bias + (folded.bias - batch_norm.running_mean) * scale)
    return folded

def build_quantized_model(model: nn.Module, weight_bits: int, activation_bits: int,
                          edge_bits: int | None = None) -> nn.Module:
    """Clone MobileNetV2 and simulate all weight, ReLU6, and residual boundaries."""
    model = copy.deepcopy(model)
    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            # Importing torchvision's private class is unnecessary: this public
            # structural contract identifies its MobileNetV2 residual blocks.
            if hasattr(child, "use_res_connect") and hasattr(child, "conv"):
                replace(child.conv)
                setattr(parent, name, QuantizedInvertedResidual(child, activation_bits))
            elif isinstance(child, nn.Conv2d): setattr(parent, name, QuantizedConv2d(child, weight_bits))
            elif isinstance(child, nn.Linear): setattr(parent, name, QuantizedLinear(child, weight_bits))
            elif isinstance(child, nn.ReLU6): setattr(parent, name, QuantizedReLU6(activation_bits))
            else: replace(child)
    replace(model)
    return QuantizedMobileNet(model, activation_bits, edge_bits)

def coverage_rows(model: nn.Module, weight_bits: int, activation_bits: int) -> list[dict[str, object]]:
    return [{"name": name, "kind": type(module).__name__, "weight_shape": list(module.weight.shape), "weight_bits": weight_bits, "scale_shape": [module.weight.shape[0]], "weight_signed": True, "activation_bits": activation_bits, "exception_reason": "none"}
            for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]

@dataclass(frozen=True)
class SizeBreakdown:
    fp32_weight_bytes: int; packed_weight_bytes: int; weight_scale_bytes: int; bias_bytes: int; descriptor_bytes: int
    @property
    def compressed_bytes(self) -> int: return self.packed_weight_bytes + self.weight_scale_bytes + self.bias_bytes + self.descriptor_bytes
    @property
    def ratio(self) -> float: return self.fp32_weight_bytes / self.compressed_bytes

def weight_size_breakdown(model: nn.Module, bits: int, descriptor_bytes_per_tensor: int = 32) -> SizeBreakdown:
    """Conservative deployable weight estimate; un-fused BN remains charged FP32."""
    fp32 = packed = scales = biases = descriptors = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            count = module.weight.numel(); fp32 += 4 * count; packed += math.ceil(count * bits / 8)
            scales += 4 * module.weight.shape[0]; descriptors += descriptor_bytes_per_tensor
            if module.bias is not None: fp32 += 4 * module.bias.numel(); biases += 4 * module.bias.numel()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # Until folding, affine parameters and running mean/variance are all
            # required at inference and therefore remain explicitly charged FP32.
            for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
                if tensor.is_floating_point(): fp32 += 4 * tensor.numel(); biases += 4 * tensor.numel()
    return SizeBreakdown(fp32, packed, scales, biases, descriptors)
