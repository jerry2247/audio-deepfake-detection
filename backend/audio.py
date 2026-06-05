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
    truncated: bool = False
    original_duration_s: float | None = None

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

    try:
        waveform, sample_rate = _decode_with_soundfile(payload)
    except ValueError:
        # libsndfile covers WAV, MP3, FLAC, OPUS, and AIFF; M4A/AAC and some
        # OGG encodings need the ffmpeg fallback.
        waveform, sample_rate = _decode_with_ffmpeg(payload)
    waveform = _to_mono(waveform)
    waveform = _resample_if_needed(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = waveform.contiguous().float()

    if waveform.numel() == 0:
        raise ValueError("Decoded audio contains no samples")
    # The detectors accept at most 30 seconds in one pass, so longer uploads
    # are analyzed from their first 30 seconds and the response says so,
    # rather than rejecting the file.
    original_duration = waveform.numel() / TARGET_SAMPLE_RATE
    truncated = original_duration > MAX_DURATION_S
    if truncated:
        waveform = waveform[: int(MAX_DURATION_S * TARGET_SAMPLE_RATE)]
    if not torch.isfinite(waveform).all():
        raise ValueError("Decoded audio contains non-finite samples")
    return DecodedAudio(
        waveform=waveform.clamp(-1.0, 1.0),
        sample_rate=TARGET_SAMPLE_RATE,
        truncated=truncated,
        original_duration_s=round(original_duration, 4) if truncated else None,
    )


def _decode_with_soundfile(payload: bytes) -> tuple[torch.Tensor, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(
            "Could not decode audio. Supported formats include WAV, MP3, M4A, "
            "FLAC, OGG, and OPUS."
        ) from exc
    if audio.ndim != 2:
        raise ValueError("Decoded audio must have channels")
    tensor = torch.from_numpy(np.asarray(audio).T)
    return tensor, int(sample_rate)


def _decode_with_ffmpeg(payload: bytes) -> tuple[torch.Tensor, int]:
    """Decode formats libsndfile cannot read by converting to WAV with ffmpeg.

    The input is written to a temporary file because container formats such
    as M4A require seekable input. Raises the same ValueError as the primary
    decoder so callers see one consistent error."""
    import subprocess
    import tempfile
    from pathlib import Path

    error = ValueError(
        "Could not decode audio. Supported formats include WAV, MP3, M4A, "
        "FLAC, OGG, and OPUS."
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = Path(tmp_dir) / "upload.audio"
        dst = Path(tmp_dir) / "decoded.wav"
        src.write_bytes(payload)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(src), "-map", "a:0", "-acodec", "pcm_s16le", str(dst)],
                capture_output=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise error from exc
        if proc.returncode != 0 or not dst.exists():
            raise error
        return _decode_with_soundfile(dst.read_bytes())


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
