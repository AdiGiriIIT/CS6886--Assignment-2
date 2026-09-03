"""Shared utilities for forthcoming custom model compression experiments.

This module deliberately does not use framework quantization APIs, as prohibited
by the assignment. Quantization operators will be implemented here in the next
phase; the current helpers provide transparent baseline size accounting.
"""

from __future__ import annotations

from pathlib import Path

import torch


def count_parameter_bits(state_dict: dict[str, torch.Tensor], bits_per_value: int = 32) -> int:
    """Return payload bits for floating-point model weights (metadata excluded)."""
    return sum(tensor.numel() * bits_per_value for tensor in state_dict.values())


def checkpoint_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 ** 2)
