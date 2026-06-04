from __future__ import annotations

import torch
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError("lengths must be a 1D tensor")
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def masked_mean_std(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape (batch, frames, dim)")
    if mask.shape != sequence.shape[:2]:
        raise ValueError("mask must have shape (batch, frames)")
    if not mask.any(dim=1).all():
        raise ValueError("each sequence must contain at least one valid frame")

    weights = mask.to(sequence.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    mean = (sequence * weights).sum(dim=1) / counts
    variance = ((sequence - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1) / counts
    std = torch.sqrt(variance.clamp_min(1e-6))
    return torch.cat([mean, std], dim=-1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LayerWeightedStatsHead(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        stats_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.classifier = MLPClassifier(stats_dim, hidden_dim, dropout)

    def forward(self, layer_stats: torch.Tensor) -> torch.Tensor:
        if layer_stats.ndim != 3:
            raise ValueError("layer_stats must have shape (batch, layers, dim)")
        if layer_stats.shape[1] != self.layer_logits.numel():
            raise ValueError(
                f"expected {self.layer_logits.numel()} layers, got {layer_stats.shape[1]}"
            )
        weights = torch.softmax(self.layer_logits, dim=0).to(layer_stats.dtype)
        features = (layer_stats * weights.view(1, -1, 1)).sum(dim=1)
        return self.classifier(features)


class EmbeddingHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = MLPClassifier(input_dim, hidden_dim, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, dim)")
        return self.classifier(features)
