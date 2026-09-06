"""Coverage, folding, packing, and accounting for custom quantization."""
from __future__ import annotations
import copy
import json
import math
import struct
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

def fold_batch_norms(model: nn.Module) -> nn.Module:
    """Fold adjacent Conv--BN pairs in an eval model, replacing BN with Identity.

    This accepts both the FP32 modules and the QAT ``QuantizedConv2d`` wrappers.
    It deliberately operates on a copy: a QAT resume checkpoint must retain its
    original trainable BatchNorm state.
    """
    model = copy.deepcopy(model).eval()
    def fold(parent: nn.Module) -> None:
        children = list(parent.named_children())
        for index, (name, child) in enumerate(children[:-1]):
            next_name, next_child = children[index + 1]
            source = child
            if isinstance(child, QuantizedConv2d):
                source = nn.Conv2d(child.weight.shape[1] * child.groups, child.weight.shape[0],
                                   child.weight.shape[2:], child.stride, child.padding,
                                   child.dilation, child.groups, child.bias is not None).to(
                                       device=child.weight.device, dtype=child.weight.dtype)
                source.weight.data.copy_(child.weight.data)
                if child.bias is not None: source.bias.data.copy_(child.bias.data)
            if isinstance(source, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d):
                fused = fold_conv_bn(source, next_child)
                if isinstance(child, QuantizedConv2d):
                    child.weight.data.copy_(fused.weight.data)
                    if child.bias is None:
                        child.bias = nn.Parameter(fused.bias.detach().clone())
                    else:
                        child.bias.data.copy_(fused.bias.data)
                else:
                    setattr(parent, name, fused)
                setattr(parent, next_name, nn.Identity())
        for _, child in parent.named_children(): fold(child)
    fold(model)
    return model

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


@dataclass(frozen=True)
class DeploymentSize:
    """Exact custom-container accounting, in bytes."""
    fp32_weight_bytes: int
    packed_weight_bytes: int
    weight_scale_bytes: int
    activation_scale_bytes: int
    int32_bias_bytes: int
    requantization_bytes: int
    descriptor_bytes: int
    padding_bytes: int
    header_bytes: int
    total_bytes: int
    @property
    def metadata_bytes(self) -> int:
        return (self.weight_scale_bytes + self.activation_scale_bytes + self.int32_bias_bytes +
                self.requantization_bytes + self.descriptor_bytes + self.padding_bytes + self.header_bytes)
    @property
    def ratio(self) -> float: return self.fp32_weight_bytes / self.total_bytes


def _quantized_weight_codes(module: QuantizedConv2d | QuantizedLinear) -> torch.Tensor:
    bits = module.weight_quantizer.bits
    qn, qp = integer_range(bits, True)
    return torch.round(module.weight.detach() / module.weight_quantizer.scale.detach().clamp_min(1e-8)).clamp(qn, qp).to(torch.int64)


def _align(blob: bytearray, alignment: int = 16) -> int:
    padding = (-len(blob)) % alignment
    blob.extend(b"\0" * padding)
    return padding


def _fp32_deployable_bytes(model: nn.Module) -> int:
    return sum(4 * parameter.numel() for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))
               for parameter in module.parameters(recurse=False))


def export_packed_model(model: nn.Module, path: str | Path) -> DeploymentSize:
    """Write an inspectable, byte-exact packed inference representation.

    The format is intentionally small and backend-neutral: ``QPK1`` header,
    UTF-8 JSON descriptors, then aligned sections for packed signed codes,
    FP32 scales, int32 biases, activation scales, and int32 requant pairs.
    It is an artifact for an integer runtime, not a PyTorch checkpoint.
    """
    model = fold_batch_norms(model)
    weights = [(name, m) for name, m in model.named_modules()
               if isinstance(m, (QuantizedConv2d, QuantizedLinear))]
    activations = [(name, m) for name, m in model.named_modules()
                   if isinstance(m, ActivationFakeQuantizer)]
    payloads: list[tuple[str, bytes]] = []
    scales = bytearray(); biases = bytearray(); requant = bytearray(); descriptors = []
    packed_weight_bytes = weight_scale_bytes = int32_bias_bytes = requantization_bytes = 0
    # Input-scale selection is explicit in each descriptor.  A backend may
    # replace the conservative first-boundary fallback with graph wiring.
    default_input_scale = float(activations[0][1].scale.detach().clamp_min(1e-8)) if activations else 1.0
    for name, module in weights:
        codes, bits = _quantized_weight_codes(module), module.weight_quantizer.bits
        payload = pack_signed(codes, bits); payloads.append((name, payload))
        scale = module.weight_quantizer.scale.detach().reshape(-1).float().cpu()
        scale_offset = len(scales); scales.extend(scale.numpy().tobytes())
        bias_offset = len(biases)
        if module.bias is not None:
            # Per-output-channel integer bias using the recorded input scale.
            denominator = (scale * default_input_scale).clamp_min(1e-12)
            codes_bias = torch.round(module.bias.detach().cpu() / denominator).clamp(-(2**31), 2**31 - 1).to(torch.int32)
            biases.extend(codes_bias.numpy().tobytes())
        rq_offset = len(requant)
        # One multiplier/shift pair per output channel.  The current reference
        # policy keeps a unity mapping; a target backend consumes these fields.
        requant.extend(struct.pack("<ii", 1 << 30, 30) * module.weight.shape[0])
        descriptors.append({"name": name, "shape": list(module.weight.shape), "bits": bits,
                            "signed": True, "count": module.weight.numel(), "payload_bytes": len(payload),
                            "weight_scale_offset": scale_offset, "weight_scale_count": module.weight.shape[0],
                            "bias_offset": bias_offset, "bias_count": 0 if module.bias is None else module.bias.numel(),
                            "requant_offset": rq_offset, "requant_count": module.weight.shape[0],
                            "input_scale": default_input_scale})
        packed_weight_bytes += len(payload); weight_scale_bytes += scale.numel() * 4
        int32_bias_bytes += 0 if module.bias is None else module.bias.numel() * 4
        requantization_bytes += module.weight.shape[0] * 8
    activation_blob = bytearray()
    for name, quantizer in activations:
        activation_blob.extend(struct.pack("<f", float(quantizer.scale.detach().clamp_min(1e-8))))
    descriptor_doc = {"format": "QPK1", "version": 1, "endianness": "little", "alignment": 16,
                      "zero_points": "implicit zero (symmetric / ReLU6 unsigned)",
                      "weights": descriptors,
                      "activation_boundaries": [{"name": n, "bits": q.bits, "signed": q.signed}
                                                for n, q in activations]}
    descriptor_blob = json.dumps(descriptor_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    blob = bytearray(struct.pack("<4sII", b"QPK1", 1, len(descriptor_blob)))
    blob.extend(descriptor_blob); header_bytes = 12
    padding = _align(blob)
    for _, payload in payloads:
        blob.extend(payload); padding += _align(blob)
    blob.extend(scales); padding += _align(blob)
    blob.extend(biases); padding += _align(blob)
    blob.extend(activation_blob); padding += _align(blob)
    blob.extend(requant); padding += _align(blob)
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(blob)
    result = DeploymentSize(_fp32_deployable_bytes(fold_batch_norms(_unwrap_quantized_model(model))),
                            packed_weight_bytes, weight_scale_bytes, len(activation_blob), int32_bias_bytes,
                            requantization_bytes, len(descriptor_blob), padding, header_bytes, len(blob))
    if result.total_bytes != Path(path).stat().st_size:
        raise AssertionError("packed artifact byte count disagrees with accounting")
    return result


def _unwrap_quantized_model(model: nn.Module) -> nn.Module:
    """Build an FP32 structural view solely to compute the folded baseline."""
    view = copy.deepcopy(model)
    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, QuantizedConv2d):
                conv = nn.Conv2d(child.weight.shape[1] * child.groups, child.weight.shape[0], child.weight.shape[2:],
                                 child.stride, child.padding, child.dilation, child.groups, child.bias is not None).to(
                                     device=child.weight.device, dtype=child.weight.dtype)
                conv.weight.data.copy_(child.weight.data)
                if child.bias is not None: conv.bias.data.copy_(child.bias.data)
                setattr(parent, name, conv)
            elif isinstance(child, QuantizedLinear):
                linear = nn.Linear(child.weight.shape[1], child.weight.shape[0], child.bias is not None).to(
                    device=child.weight.device, dtype=child.weight.dtype)
                linear.weight.data.copy_(child.weight.data)
                if child.bias is not None: linear.bias.data.copy_(child.bias.data)
                setattr(parent, name, linear)
            else: replace(child)
    replace(view)
    return view


@dataclass(frozen=True)
class ActivationMemory:
    fp32_peak_live_bytes: int
    quantized_peak_live_bytes: int
    fp32_traffic_bytes: int
    quantized_traffic_bytes: int
    event_count: int
    @property
    def ratio(self) -> float: return self.fp32_peak_live_bytes / self.quantized_peak_live_bytes


def activation_liveness(model: nn.Module, example: torch.Tensor) -> ActivationMemory:
    """Measure batch-one boundary storage from one real eval forward schedule.

    Each fake-quantizer invocation is a deployment boundary.  Tensor identities
    connect producer and consumer calls, so residual operands remain live until
    their add boundary.  The identical event/liveness schedule is costed twice:
    FP32 (4 bytes/value) and the boundary's actual activation bit width.
    """
    if example.shape[0] != 1: raise ValueError("activation accounting requires batch size one")
    model = model.eval(); events: list[dict[str, object]] = []; producer: dict[int, int] = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, ActivationFakeQuantizer): continue
        def before(_module, args, event_name=name):
            inputs = [id(value) for value in args if isinstance(value, torch.Tensor)]
            events.append({"name": event_name, "inputs": inputs, "output": None,
                           "count": 0, "bits": _module.bits})
        def after(_module, _args, output):
            event = events[-1]
            if isinstance(output, torch.Tensor):
                event["output"], event["count"] = id(output), output.numel()
                producer[id(output)] = len(events) - 1
        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    with torch.no_grad(): model(example)
    for handle in handles: handle.remove()
    # A residual quantizer is invoked on skip, branch, then their sum.  Its two
    # operand codes are consumed by the third call's integer-add boundary.
    groups: dict[str, list[int]] = {}
    for index, event in enumerate(events): groups.setdefault(str(event["name"]), []).append(index)
    consumers = [index for index in range(len(events))]
    for index, event in enumerate(events):
        for tensor_id in event["inputs"]:
            if tensor_id in producer: consumers[producer[tensor_id]] = max(consumers[producer[tensor_id]], index)
    for name, indices in groups.items():
        if name.endswith("residual_add.activation_quantizer"):
            for start in range(0, len(indices) - 2, 3):
                consumers[indices[start]] = max(consumers[indices[start]], indices[start + 2])
                consumers[indices[start + 1]] = max(consumers[indices[start + 1]], indices[start + 2])
    def byte_cost(event: dict[str, object], fp32: bool) -> int:
        payload = int(event["count"]) * (4 if fp32 else math.ceil(int(event["bits"]) / 8))
        # Packed tensors use ceil(N*b/8), not ceil(b/8) per element.
        if not fp32: payload = math.ceil(int(event["count"]) * int(event["bits"]) / 8)
        return payload + (0 if fp32 else 4)  # quantized boundary has one FP32 scale; zero point is implicit.
    def schedule(fp32: bool) -> tuple[int, int]:
        live: dict[int, int] = {}; peak = traffic = 0
        for index, event in enumerate(events):
            cost = byte_cost(event, fp32); live[index] = cost; traffic += cost
            peak = max(peak, sum(live.values()))
            for produced, last_consumer in enumerate(consumers):
                if last_consumer == index: live.pop(produced, None)
        return peak, traffic
    fp_peak, fp_traffic = schedule(True); q_peak, q_traffic = schedule(False)
    return ActivationMemory(fp_peak, q_peak, fp_traffic, q_traffic, len(events))
