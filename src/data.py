"""CIFAR-10 data loading and documented preprocessing."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test = transforms.Compose([transforms.ToTensor(), normalize])
    return train, test


def build_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int,
                          pin_memory: bool, seed: int):
    train_transform, test_transform = cifar10_transforms()
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    generator = __import__("torch").Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **common),
        DataLoader(test_set, shuffle=False, **common),
    )
