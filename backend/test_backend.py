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
