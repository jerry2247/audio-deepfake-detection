from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


CLIP_ID_RE = re.compile(r"^clip_[0-9a-f]{32}$")
REQUIRED_COLUMNS = {
    "clip_id",
    "path",
    "label",
    "split",
    "eval_condition",
    "split_group_id",
    "is_heldout_generator",
    "is_in_the_wild",
    "source_dataset",
    "generator",
    "generator_family",
    "duration_s",
    "final_sample_rate",
    "final_channels",
    "final_format",
    "bit_depth",
    "sha256",
}


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    path: str
    label: str
    split: str
    duration_s: float

    @property
    def target(self) -> float:
        return 1.0 if self.label == "fake" else 0.0


def load_manifest(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    manifest_path = root / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}. "
            "Training and evaluation require the frozen dataset build output."
        )
    manifest = pd.read_parquet(manifest_path)
    validate_manifest(manifest, root)
    return manifest


def validate_manifest(manifest: pd.DataFrame, dataset_root: str | Path) -> None:
    root = Path(dataset_root)
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    if manifest["clip_id"].duplicated().any():
        raise ValueError("manifest clip_id values must be unique")

    for row in manifest.itertuples(index=False):
        clip_id = str(getattr(row, "clip_id"))
        rel_path = str(getattr(row, "path"))
        label = str(getattr(row, "label"))
        split = str(getattr(row, "split"))

        if not CLIP_ID_RE.fullmatch(clip_id):
            raise ValueError(f"invalid clip_id {clip_id!r}")
        if label not in {"real", "fake"}:
            raise ValueError(f"invalid label {label!r} for {clip_id}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {split!r} for {clip_id}")
        expected_path = f"audio/{split}/{label}/{clip_id}.wav"
        if rel_path != expected_path:
            raise ValueError(f"path for {clip_id} must be {expected_path!r}, got {rel_path!r}")
        resolved = (root / rel_path).resolve()
        if not str(resolved).startswith(str(root.resolve())):
            raise ValueError(f"path escapes dataset root for {clip_id}: {rel_path}")

        duration = float(getattr(row, "duration_s"))
        if not (0.0 < duration <= 30.0):
            raise ValueError(f"duration_s out of contract for {clip_id}: {duration}")
        if int(getattr(row, "final_sample_rate")) != 16000:
            raise ValueError(f"final_sample_rate must be 16000 for {clip_id}")
        if int(getattr(row, "final_channels")) != 1:
            raise ValueError(f"final_channels must be 1 for {clip_id}")
        if str(getattr(row, "final_format")) != "wav_pcm16":
            raise ValueError(f"final_format must be wav_pcm16 for {clip_id}")
        if int(getattr(row, "bit_depth")) != 16:
            raise ValueError(f"bit_depth must be 16 for {clip_id}")


def records_for_split(manifest: pd.DataFrame, split: str) -> list[ManifestRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"invalid split {split!r}")
    rows = manifest[manifest["split"] == split]
    return [
        ManifestRecord(
            clip_id=str(row.clip_id),
            path=str(row.path),
            label=str(row.label),
            split=str(row.split),
            duration_s=float(row.duration_s),
        )
        for row in rows.itertuples(index=False)
    ]


def load_waveform(dataset_root: str | Path, record: ManifestRecord) -> tuple[torch.Tensor, int]:
    audio_path = Path(dataset_root) / record.path
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file missing for {record.clip_id}: {audio_path}")
    info = sf.info(str(audio_path))
    if info.format != "WAV":
        raise ValueError(f"{record.clip_id} must be WAV, got {info.format}")
    if info.subtype != "PCM_16":
        raise ValueError(f"{record.clip_id} must be PCM_16, got {info.subtype}")
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)
    if sample_rate != 16000:
        raise ValueError(f"{record.clip_id} has sample rate {sample_rate}, expected 16000")
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"{record.clip_id} must be mono audio with shape (1, samples)")
    waveform = waveform.squeeze(0).float()
    if waveform.numel() == 0:
        raise ValueError(f"{record.clip_id} has empty audio")
    if waveform.numel() > 30 * 16000:
        raise ValueError(f"{record.clip_id} exceeds the 30 second dataset limit")
    return waveform, sample_rate


class AudioManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset_root: str | Path, records: list[ManifestRecord]) -> None:
        self.dataset_root = Path(dataset_root)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, sample_rate = load_waveform(self.dataset_root, record)
        return {
            "waveform": waveform,
            "length": waveform.numel(),
            "label": torch.tensor(record.target, dtype=torch.float32),
            "clip_id": record.clip_id,
            "path": record.path,
            "sample_rate": sample_rate,
        }


def collate_audio(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)
    labels = torch.stack([item["label"] for item in batch]).float()
    return {
        "waveforms": padded,
        "lengths": lengths,
        "labels": labels,
        "clip_ids": [item["clip_id"] for item in batch],
        "paths": [item["path"] for item in batch],
    }
