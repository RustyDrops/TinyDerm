"""Tiny Student Model module for TinyDerm Lab."""

import os
from pathlib import Path
from typing import Dict, Tuple
import timm
import torch
import torch.nn as nn

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


class TinyDermModel(nn.Module):
    """MobileNetV3-Small deployable baseline student classifier."""

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        num_classes: int = 7,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self):
        """Stage 1: Freeze backbone, train only the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier / head
        head = self.model.get_classifier()
        if isinstance(head, nn.Module):
            for param in head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Stage 2: Unfreeze complete network for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def get_model_size_mb(self) -> float:
        """Estimate in-memory parameter footprint in MB (FP32)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return round((total_params * 4) / (1024 * 1024), 2)
