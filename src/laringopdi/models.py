"""Modelos fixos do protocolo: ResNet-18 (ImageNet) e SimpleCNN treinada do zero."""
from __future__ import annotations

import torch
from torch import nn


def resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 com pesos ImageNet; troca só a camada final (entrada: cinza replicado 3×)."""
    from torchvision.models import ResNet18_Weights, resnet18 as tv_resnet18

    m = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


class SimpleCNN(nn.Module):
    """4 blocos [conv 3×3 → BN → ReLU → max-pool 2] com 32/64/128/256 filtros → GAP → dropout → linear."""

    def __init__(self, n_classes: int, filters=(32, 64, 128, 256), dropout: float = 0.3, in_ch: int = 3):
        super().__init__()
        layers, c = [], in_ch
        for f in filters:
            layers += [nn.Conv2d(c, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = f
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build(name: str, n_classes: int, cfg: dict, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        return resnet18(n_classes, pretrained)
    if name in ("simplecnn", "simplecnn_fast"):
        return SimpleCNN(n_classes, cfg["simplecnn"]["filters"], cfg["simplecnn"]["dropout"])
    raise ValueError(f"modelo desconhecido: {name}")


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Última camada convolucional para o Grad-CAM."""
    return model.layer4[-1] if hasattr(model, "layer4") else model.features[-4]
