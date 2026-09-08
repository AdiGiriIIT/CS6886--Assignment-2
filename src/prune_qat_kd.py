"""Deprecated entry point for the former combined pruning/KD experiment.

Pruning recovery is now intentionally supervised-only. Use ``src.prune_qat``;
knowledge distillation is implemented independently in ``src.distill``.
"""

from .prune_qat import main


if __name__ == "__main__":
    main()
