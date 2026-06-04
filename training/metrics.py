from __future__ import annotations

import math

import torch


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    return float((predictions == labels).float().mean().item())


def equal_error_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().flatten().cpu().float()
    labels = labels.detach().flatten().cpu().long()
    if logits.numel() != labels.numel():
        raise ValueError("logits and labels must have the same number of elements")
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires at least one positive and one negative sample")

    order = torch.argsort(logits, descending=True)
    sorted_labels = labels[order]
    true_positives = torch.cumsum(sorted_labels, dim=0).float()
    false_positives = torch.cumsum(1 - sorted_labels, dim=0).float()

    false_negative_rate = (positives - true_positives) / positives
    false_positive_rate = false_positives / negatives

    start_fnr = torch.tensor([1.0])
    start_fpr = torch.tensor([0.0])
    fnr = torch.cat([start_fnr, false_negative_rate])
    fpr = torch.cat([start_fpr, false_positive_rate])
    idx = torch.argmin(torch.abs(fnr - fpr))
    eer = float(((fnr[idx] + fpr[idx]) / 2.0).item())
    if not math.isfinite(eer):
        raise ValueError("computed EER is not finite")
    return eer

