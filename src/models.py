"""MobileNetV2 implementation adapted for CIFAR-10's 32x32 images."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channels to hardware-friendly multiples without reducing by >10%."""
    min_value = divisor if min_value is None else min_value
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, bn_momentum: float = 0.1) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int,
                 bn_momentum: float) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        hidden_channels = in_channels * expand_ratio
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1,
                                     bn_momentum=bn_momentum))
        layers.extend((
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride, groups=hidden_channels,
                       bn_momentum=bn_momentum),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        ))
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x) if self.use_residual else self.block(x)


class MobileNetV2CIFAR10(nn.Module):
    """MobileNetV2 with a stride-1 stem to preserve resolution for CIFAR-10."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 dropout: float = 0.2, bn_momentum: float = 0.1) -> None:
        super().__init__()
        settings: Sequence[tuple[int, int, int, int]] = (
            (1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
        )
        input_channels = _make_divisible(32 * width_mult)
        last_channels = _make_divisible(1280 * max(1.0, width_mult))
        features: list[nn.Module] = [ConvBNReLU(3, input_channels, stride=1, bn_momentum=bn_momentum)]
        for expand, channels, repeats, stride in settings:
            output_channels = _make_divisible(channels * width_mult)
            for repeat in range(repeats):
                features.append(InvertedResidual(input_channels, output_channels,
                                                 stride if repeat == 0 else 1, expand, bn_momentum))
                input_channels = output_channels
        features.append(ConvBNReLU(input_channels, last_channels, kernel_size=1, bn_momentum=bn_momentum))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_channels, num_classes))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


def build_model(model_config: dict) -> MobileNetV2CIFAR10:
    return MobileNetV2CIFAR10(**model_config)
