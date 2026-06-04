from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .specs import MethodSpec


class ModelAccessError(RuntimeError):
    """Raised when a required frozen backbone cannot be loaded exactly."""


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def save_detector_artifacts(
    *,
    output_dir: Path,
    spec: MethodSpec,
    head: nn.Module,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output_dir / "detector_head.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "method": spec.key,
        "display_name": spec.display_name,
        "frozen_backbone": spec.backbone_id,
        **config,
    }
    (output_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
