"""ffmpeg-based audio helpers used by source preps (decode/probe only — the uniform
conditioning DSP lives in dataset/build/condition.py, not here)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioError(Exception):
    pass


def ffprobe_info(path: Path) -> dict:
    """Return {duration_s, sample_rate, channels, codec} for an audio file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioError(f"ffprobe failed on {path}: {r.stderr[:200]}")
    d = json.loads(r.stdout)
    streams = d.get("streams") or []
    if not streams:
        raise AudioError(f"no audio stream in {path}")
    s = streams[0]
    dur = float(d.get("format", {}).get("duration") or 0.0)
    return {
        "duration_s": dur,
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": int(s.get("channels") or 0),
        "codec": s.get("codec_name") or "unknown",
    }


def decode_to_staged_wav(src: Path, dst: Path, start_s: float | None = None,
                         duration_s: float | None = None) -> dict:
    """Decode any audio file (or a segment of it) to mono PCM16 WAV at the NATIVE
    sample rate. Returns ffprobe info of the written file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-ac", "1", "-c:a", "pcm_s16le", "-map", "a:0", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        raise AudioError(f"ffmpeg decode failed on {src}: {r.stderr[:200]}")
    return ffprobe_info(dst)


def decode_bytes_to_staged_wav(data: bytes, suffix: str, dst: Path) -> dict:
    """Decode an in-memory audio payload (e.g. HF Audio(decode=False) bytes) to a
    mono PCM16 WAV at native rate, via a temp file + ffmpeg."""
    import tempfile

    suffix = suffix if suffix.startswith(".") else "." + (suffix or "bin")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return decode_to_staged_wav(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_pcm_array_to_wav(array, sample_rate: int, dst: Path) -> dict:
    """Write a float/int numpy array (as produced by HF datasets Audio casts or TTS
    models) to mono PCM16 WAV at the given rate."""
    import numpy as np
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(array)
    if a.ndim == 2:  # (channels, n) or (n, channels) -> mono
        a = a.mean(axis=0 if a.shape[0] < a.shape[1] else 1)
    if a.dtype.kind == "f":
        a = np.clip(a, -1.0, 1.0)
    sf.write(str(dst), a, sample_rate, subtype="PCM_16")
    return ffprobe_info(dst)


def codec_history_label(codec: str, bitrate_kbps: int | None = None) -> str:
    """Map an origin codec to the DATASHEET codec_history vocabulary."""
    codec = (codec or "").lower()
    table = {"mp3": "mp3", "aac": "aac", "opus": "opus", "vorbis": "vorbis",
             "flac": "flac", "pcm_s16le": "wav", "pcm_f32le": "wav", "wav": "wav"}
    base = table.get(codec, "unknown")
    if base in ("mp3", "aac", "opus", "vorbis") and bitrate_kbps:
        return f"{base}_{bitrate_kbps}k"
    return base
