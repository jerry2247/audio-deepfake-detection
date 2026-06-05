from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

os.environ.setdefault("AUDIO_LAB_LOAD_MODE", "inspect")

from backend.app import create_app
from backend.audio import decode_audio_upload
from backend.service import AudioLabService


def _wav_bytes(duration_s: float = 0.25, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_decode_audio_upload_resamples_to_project_rate() -> None:
    payload = _wav_bytes(sample_rate=8000)
    decoded = decode_audio_upload(payload)
    assert decoded.sample_rate == 16000
    assert decoded.waveform.ndim == 1
    assert decoded.num_samples == 4000


def _encode(payload_wav: bytes, suffix: str) -> bytes | None:
    """Encode WAV bytes into another container with ffmpeg; None if unavailable."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / f"out{suffix}"
        src.write_bytes(payload_wav)
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), str(dst)],
            capture_output=True, timeout=60,
        )
        if proc.returncode != 0 or not dst.exists():
            return None
        return dst.read_bytes()


def test_decode_audio_upload_truncates_long_clips_with_flag() -> None:
    payload = _wav_bytes(duration_s=42.0)
    decoded = decode_audio_upload(payload)
    assert decoded.truncated is True
    assert decoded.num_samples == 30 * 16000
    assert decoded.original_duration_s is not None
    assert 41.5 <= decoded.original_duration_s <= 42.5
    short = decode_audio_upload(_wav_bytes(duration_s=1.0))
    assert short.truncated is False and short.original_duration_s is None


def test_decode_audio_upload_supports_common_compressed_formats() -> None:
    wav = _wav_bytes(duration_s=1.0)
    for suffix in (".mp3", ".m4a", ".flac", ".ogg"):
        payload = _encode(wav, suffix)
        if payload is None:
            continue  # encoder not available on this machine
        decoded = decode_audio_upload(payload)
        assert decoded.sample_rate == 16000, suffix
        assert 0.8 <= decoded.duration_s <= 1.3, (suffix, decoded.duration_s)


def test_api_health_methods_and_analyze_without_model_loading() -> None:
    service = AudioLabService(model_root="final_models", load_models=False)
    client = TestClient(create_app(service))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["total_methods"] == 4

    methods = client.get("/methods")
    assert methods.status_code == 200
    method_payload = methods.json()["methods"]
    assert len(method_payload) == 4
    assert any(item["status"] == "not_loaded" for item in method_payload)

    response = client.post(
        "/analyze",
        files={"file": ("tone.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["audio"]["sample_rate"] == 16000
    assert set(payload["methods"]) == {
        "waveform_ssl",
        "asr_logmel_encoder",
        "audio_spectrogram_vit",
        "vision_spectrogram_vit",
    }
    assert payload["visualizations"]["waveform"]["points"]
    assert payload["visualizations"]["sslam_fbank"]["image_png_base64"]
    assert payload["visualizations"]["dinov3_spectrogram"]["image_png_base64"]
    assert any(result["status"] == "not_loaded" for result in payload["methods"].values())


def test_local_preview_origin_is_allowed() -> None:
    service = AudioLabService(model_root="final_models", load_models=False)
    client = TestClient(create_app(service))

    response = client.options(
        "/analyze",
        headers={
            "Origin": "http://127.0.0.1:4173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4173"
