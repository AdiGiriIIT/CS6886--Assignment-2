"""Official torchvision MobileNetV2 adapted and fine-tuned for CIFAR-10."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(model_config: dict, load_pretrained: bool = True) -> nn.Module:
    """Build MobileNetV2, optionally initializing from public ImageNet weights.

    The weights are used only for training initialization. Evaluation reconstructs
    the model without downloading them, then loads the saved CIFAR-10 checkpoint.
    """
    width_mult = model_config.get("width_mult", 1.0)
    pretrained = model_config.get("pretrained", False) and load_pretrained
    if pretrained and width_mult != 1.0:
        raise ValueError("Official ImageNet MobileNetV2 weights require width_mult: 1.0.")
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult,
                         dropout=model_config.get("dropout", 0.2))
    # CIFAR-10 images are 32x32; retain their spatial resolution at the stem.
    model.features[0][0].stride = (1, 1)
    model.classifier[1] = nn.Linear(model.last_channel, model_config.get("num_classes", 10))
    return model


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """Freeze/unfreeze pretrained feature weights while retaining the classifier."""
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze
