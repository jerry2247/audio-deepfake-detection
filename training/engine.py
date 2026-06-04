from __future__ import annotations

import importlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import AudioManifestDataset, collate_audio, load_manifest, records_for_split
from .metrics import binary_accuracy, equal_error_rate
from .modeling import save_detector_artifacts
from .specs import MethodSpec, final_output_dir, get_method_spec


class FeatureExtractor(Protocol):
    spec: MethodSpec

    def to(self, device: torch.device) -> Any: ...
    def eval(self) -> Any: ...
    def extract_features(self, batch: dict[str, Any]) -> torch.Tensor: ...
    def artifact_config(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str = "dataset"
    cache_root: str = "training"
    output_root: str = "final_models"
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 0
    seed: int = 153
    use_amp: bool = True
    print_progress: bool = True
    device: str = "auto"


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device. "auto" prefers CUDA, then CPU. Apple MPS is
    honored only on explicit request: a verification run measured 35 percent
    relative drift between MPS and CPU Whisper-encoder features, so MPS output
    is not treated as equivalent and never enters a cache silently."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    if preference not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"invalid device preference {preference!r}")
    return torch.device(preference)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def method_module(method: str) -> Any:
    spec = get_method_spec(method)
    return importlib.import_module(spec.module)


def build_feature_extractor(method: str, device: torch.device) -> FeatureExtractor:
    module = method_module(method)
    detector = module.build_detector(device=device)
    detector.eval()
    return detector


def build_head(method: str) -> nn.Module:
    module = method_module(method)
    return module.build_head()


def method_feature_config(method: str) -> dict[str, Any]:
    module = method_module(method)
    return module.feature_config()


def build_split_loader(config: TrainConfig, split: str, shuffle: bool = False) -> DataLoader:
    manifest = load_manifest(config.dataset_root)
    records = records_for_split(manifest, split)
    if not records:
        raise ValueError(f"manifest has no {split} records")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        AudioManifestDataset(config.dataset_root, records),
        shuffle=shuffle,
        generator=generator,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_audio,
        pin_memory=torch.cuda.is_available(),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def method_cache_dir(config: TrainConfig, method: str) -> Path:
    spec = get_method_spec(method)
    return Path(config.cache_root) / spec.final_subdir / "feature_cache"


def split_cache_dir(config: TrainConfig, method: str, split: str) -> Path:
    return method_cache_dir(config, method) / split


def split_feature_path(config: TrainConfig, method: str, split: str) -> Path:
    return split_cache_dir(config, method, split) / "features.pt"


def split_metadata_path(config: TrainConfig, method: str, split: str) -> Path:
    return split_cache_dir(config, method, split) / "metadata.json"


def require_cache(config: TrainConfig, method: str, split: str) -> None:
    feature_path = split_feature_path(config, method, split)
    metadata_path = split_metadata_path(config, method, split)
    if not feature_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature cache missing for method={method} split={split}. "
            f"Run: python -m training.extract_features --method {method}"
        )


def extract_feature_cache(
    method: str,
    config: TrainConfig,
    splits: tuple[str, ...] = ("train", "val", "test"),
    overwrite: bool = False,
) -> dict[str, Any]:
    set_seed(config.seed)
    loaders = {split: build_split_loader(config, split, shuffle=False) for split in splits}
    device = resolve_device(config.device)
    extractor = build_feature_extractor(method, device).to(device)
    spec = get_method_spec(method)
    extraction_summary: dict[str, Any] = {
        "method": method,
        "backbone": spec.backbone_id,
        "splits": {},
        "feature_config": method_feature_config(method),
    }

    for split, loader in loaders.items():
        output_dir = split_cache_dir(config, method, split)
        feature_path = split_feature_path(config, method, split)
        metadata_path = split_metadata_path(config, method, split)
        if feature_path.exists() and metadata_path.exists() and not overwrite:
            raise FileExistsError(
                f"Feature cache already exists for method={method} split={split}: {output_dir}"
            )

        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        clip_ids: list[str] = []
        paths: list[str] = []
        extractor.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                batch_features = extractor.extract_features(batch)
                features.append(batch_features.detach().cpu().to(torch.float16))
                labels.append(batch["labels"].detach().cpu().float())
                clip_ids.extend(batch["clip_ids"])
                paths.extend(batch["paths"])

        feature_tensor = torch.cat(features, dim=0).contiguous()
        label_tensor = torch.cat(labels, dim=0).contiguous()
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": feature_tensor, "labels": label_tensor}, feature_path)
        metadata = {
            "method": method,
            "split": split,
            "backbone": spec.backbone_id,
            "num_examples": int(feature_tensor.shape[0]),
            "feature_shape": list(feature_tensor.shape[1:]),
            "feature_dtype": "float16",
            "label_dtype": "float32",
            "clip_ids": clip_ids,
            "paths": paths,
            "feature_config": method_feature_config(method),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        extraction_summary["splits"][split] = {
            "num_examples": metadata["num_examples"],
            "feature_shape": metadata["feature_shape"],
            "cache_dir": str(output_dir),
        }
        if config.print_progress:
            print(
                f"method={method} split={split} cached_examples={feature_tensor.shape[0]} "
                f"feature_shape={list(feature_tensor.shape[1:])}",
                file=sys.stderr,
                flush=True,
            )
    return extraction_summary


class CachedFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, config: TrainConfig, method: str, split: str) -> None:
        require_cache(config, method, split)
        payload = torch.load(split_feature_path(config, method, split), weights_only=True)
        features = payload["features"]
        labels = payload["labels"]
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"cache row mismatch for method={method} split={split}")
        self.features = features.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "labels": self.labels[index]}


def collate_features(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty cached-feature batch")
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_cache_loader(config: TrainConfig, method: str, split: str, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        CachedFeatureDataset(config, method, split),
        shuffle=shuffle,
        generator=generator,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_features,
        pin_memory=torch.cuda.is_available(),
    )


def build_cache_loaders(config: TrainConfig, method: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        build_cache_loader(config, method, "train", shuffle=True),
        build_cache_loader(config, method, "val", shuffle=False),
        build_cache_loader(config, method, "test", shuffle=False),
    )


def evaluate_head(head: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    head.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = head(features)
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits_all.float(), labels_all.float()
    )
    return {
        "loss": float(loss.item()),
        "eer": equal_error_rate(logits_all, labels_all),
        "accuracy_at_0_5": binary_accuracy(logits_all, labels_all),
    }


def fit_detector(method: str, config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader = build_cache_loaders(config, method)
    head = build_head(method).to(device)

    params = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError(f"{method} exposes no trainable detector-head parameters")
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(config.use_amp and device.type == "cuda")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_eer = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        head.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = head(features)
                loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            batch_size = int(labels.shape[0])
            running_loss += float(loss.detach().cpu().item()) * batch_size
            seen += batch_size

        val_metrics = evaluate_head(head, val_loader, device)
        train_loss = running_loss / max(seen, 1)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_eer": val_metrics["eer"],
            "val_accuracy_at_0_5": val_metrics["accuracy_at_0_5"],
        }
        history.append(row)
        if config.print_progress:
            print(
                " ".join(
                    [
                        f"method={method}",
                        f"epoch={epoch}",
                        f"train_loss={train_loss:.6f}",
                        f"val_loss={val_metrics['loss']:.6f}",
                        f"val_eer={val_metrics['eer']:.6f}",
                        f"val_accuracy_at_0_5={val_metrics['accuracy_at_0_5']:.6f}",
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )

        if val_metrics["eer"] < best_val_eer:
            best_val_eer = val_metrics["eer"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation-selected head")
    head.load_state_dict(best_state)
    # Final report: the validation-selected head is evaluated once on every
    # split (train without shuffling), so loss, EER, and accuracy are usable
    # side by side. Test plays no role in selection.
    selected_head_metrics = {
        "train": evaluate_head(
            head, build_cache_loader(config, method, "train", shuffle=False), device
        ),
        "val": evaluate_head(head, val_loader, device),
        "test": evaluate_head(head, test_loader, device),
    }

    spec = get_method_spec(method)
    output_dir = final_output_dir(config.output_root, spec)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_eer": best_val_eer,
        "selected_head": selected_head_metrics,
        "history": history,
    }
    save_detector_artifacts(
        output_dir=output_dir,
        spec=spec,
        head=head,
        metrics=metrics,
        config={
            "train_config": asdict(config),
            "feature_config": method_feature_config(method),
            "cache_dir": str(method_cache_dir(config, method)),
        },
    )
    return metrics


def metrics_to_json(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


def evaluate_saved_detector(
    method: str,
    config: TrainConfig,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    set_seed(config.seed)
    device = resolve_device(config.device)
    loader = build_cache_loader(config, method, split, shuffle=False)
    head = build_head(method).to(device)

    spec = get_method_spec(method)
    output_dir = final_output_dir(config.output_root, spec)
    head_path = output_dir / "detector_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Saved detector head not found: {head_path}")
    state_dict = torch.load(head_path, map_location=device, weights_only=True)
    head.load_state_dict(state_dict)
    split_metrics = evaluate_head(head, loader, device)
    payload = {
        "method": method,
        "split": split,
        "metrics": split_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"evaluation_{split}.json").write_text(
        metrics_to_json(payload) + "\n",
        encoding="utf-8",
    )
    return payload

