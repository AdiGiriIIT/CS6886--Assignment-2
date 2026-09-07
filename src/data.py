"""CIFAR-10 data loading and documented preprocessing."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar10_transforms(augmentation: dict | None = None) -> tuple[transforms.Compose, transforms.Compose]:
    """Return CIFAR-10 transforms, optionally enabling stronger train-only augmentation.

    ``augmentation`` is deliberately opt-in so an existing QAT command keeps the
    original crop/flip protocol.  RandAugment operates on PIL images and Random
    Erasing operates on normalized tensors, hence their positions in the list.
    """
    augmentation = augmentation or {}
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_steps: list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if augmentation.get("randaugment", False):
        train_steps.append(transforms.RandAugment(
            num_ops=int(augmentation.get("randaugment_num_ops", 2)),
            magnitude=int(augmentation.get("randaugment_magnitude", 7)),
        ))
    train_steps.extend([transforms.ToTensor(), normalize])
    random_erasing_probability = float(augmentation.get("random_erasing_probability", 0.0))
    if not 0.0 <= random_erasing_probability <= 1.0:
        raise ValueError("random_erasing_probability must be in [0, 1]")
    if random_erasing_probability:
        train_steps.append(transforms.RandomErasing(p=random_erasing_probability))
    train = transforms.Compose(train_steps)
    test = transforms.Compose([transforms.ToTensor(), normalize])
    return train, test


def build_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int,
                          pin_memory: bool, seed: int, validation_size: int = 5_000,
                          augmentation: dict | None = None):
    """Build deterministic 45k/5k train/validation loaders plus the untouched test loader.

    The validation indices are sampled once from the official CIFAR-10 training
    split using ``seed``.  A separate dataset instance gives validation examples
    evaluation transforms rather than stochastic training augmentation.
    """
    train_transform, test_transform = cifar10_transforms(augmentation)
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    validation_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    if not 0 < validation_size < len(train_set):
        raise ValueError(f"validation_size must be between 1 and {len(train_set) - 1}, got {validation_size}")
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_indices, train_indices = indices[:validation_size], indices[validation_size:]
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(Subset(train_set, train_indices), shuffle=True, generator=generator, **common),
        DataLoader(Subset(validation_set, validation_indices), shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


def build_cifar10_test_loader(data_dir: str, batch_size: int, num_workers: int,
                              pin_memory: bool) -> DataLoader:
    """Build only the official CIFAR-10 test loader for final evaluation."""
    _, test_transform = cifar10_transforms()
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0)
