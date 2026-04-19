from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEModel


class ASLVideoMAEClassifier(nn.Module):
    """VideoMAE backbone with a lightweight classifier head.

    Expected input shape: (B, T, C, H, W)
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape={tuple(x.shape)}")

        outputs = self.backbone(pixel_values=x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)
