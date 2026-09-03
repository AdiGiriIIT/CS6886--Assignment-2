"""Home for assignment-compliant, from-scratch quantization operators.

Kept separate from orchestration in :mod:`src.compress` so experiment logic and
compression math remain independently testable.
"""

from __future__ import annotations


class QuantizationNotImplementedError(NotImplementedError):
    """Raised if a future compression command is invoked before implementation."""
