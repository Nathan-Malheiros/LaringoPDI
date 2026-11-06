"""Modelo fixo do protocolo: ResNet-18 pré-treinada na ImageNet."""
from __future__ import annotations

from torch import nn


def resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 com pesos ImageNet; troca só a camada final (entrada: cinza replicado 3×)."""
    from torchvision.models import ResNet18_Weights, resnet18 as tv_resnet18

    m = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build(name: str, n_classes: int, cfg: dict, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        return resnet18(n_classes, pretrained)
    raise ValueError(f"modelo desconhecido: {name}")


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Última camada convolucional para o Grad-CAM."""
    return model.layer4[-1]
