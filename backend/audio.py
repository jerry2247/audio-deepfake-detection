from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F


TARGET_SAMPLE_RATE = 16000
MAX_DURATION_S = 30.0
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class DecodedAudio:
    waveform: torch.Tensor
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return float(self.waveform.numel()) / float(self.sample_rate)

    @property
    def num_samples(self) -> int:
        return int(self.waveform.numel())


def decode_audio_upload(payload: bytes) -> DecodedAudio:
    if not payload:
        raise ValueError("Uploaded audio file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded audio exceeds {MAX_UPLOAD_BYTES} bytes")

    waveform, sample_rate = _decode_with_soundfile(payload)
    waveform = _to_mono(waveform)
    waveform = _resample_if_needed(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = waveform.contiguous().float()

    if waveform.numel() == 0:
        raise ValueError("Decoded audio contains no samples")
    duration = waveform.numel() / TARGET_SAMPLE_RATE
    if duration > MAX_DURATION_S:
        raise ValueError(f"Audio must be at most {MAX_DURATION_S:g} seconds")
    if not torch.isfinite(waveform).all():
        raise ValueError("Decoded audio contains non-finite samples")
    return DecodedAudio(waveform=waveform.clamp(-1.0, 1.0), sample_rate=TARGET_SAMPLE_RATE)


def _decode_with_soundfile(payload: bytes) -> tuple[torch.Tensor, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(
            "Could not decode audio. Upload a WAV, FLAC, OGG, or another format "
            "supported by the server audio backend."
        ) from exc
    if audio.ndim != 2:
        raise ValueError("Decoded audio must have channels")
    tensor = torch.from_numpy(np.asarray(audio).T)
    return tensor, int(sample_rate)


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError("Audio tensor must have shape (channels, samples)")
    if waveform.shape[0] == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def _resample_if_needed(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate <= 0:
        raise ValueError(f"Invalid sample rate: {source_rate}")
    if source_rate == target_rate:
        return waveform
    return F.resample(waveform, source_rate, target_rate)


def make_model_batch(audio: DecodedAudio, device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    waveform = audio.waveform.to(device)
    return {
        "waveforms": waveform.unsqueeze(0),
        "lengths": torch.tensor([waveform.numel()], dtype=torch.long, device=device),
        "labels": torch.zeros(1, dtype=torch.float32, device=device),
        "clip_ids": ["uploaded_audio"],
        "paths": ["uploaded_audio"],
    }
