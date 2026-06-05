from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MethodKey = Literal[
    "waveform_ssl",
    "asr_logmel_encoder",
    "audio_spectrogram_vit",
    "vision_spectrogram_vit",
]

MethodStatus = Literal[
    "ready",
    "not_trained",
    "not_loaded",
    "model_access_error",
    "runtime_error",
]

Prediction = Literal["real", "fake"]


class AudioSummary(BaseModel):
    duration_s: float
    sample_rate: int = 16000
    channels: int = 1
    num_samples: int
    truncated: bool = False
    original_duration_s: float | None = None


class WaveformVisualization(BaseModel):
    points: list[float]
    sample_rate: int = 16000


class ImageVisualization(BaseModel):
    image_png_base64: str
    width: int
    height: int


class SSLAMVisualization(ImageVisualization):
    mel_bins: int
    frames: int


class DINOv3Visualization(ImageVisualization):
    image_size: int
    patch_grid: tuple[int, int]


class VisualizationBlock(BaseModel):
    waveform: WaveformVisualization
    sslam_fbank: SSLAMVisualization
    dinov3_spectrogram: DINOv3Visualization


class MethodResult(BaseModel):
    status: MethodStatus
    display_name: str
    probability_fake: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction: Prediction | None = None
    elapsed_ms: float | None = Field(default=None, ge=0.0)
    error: str | None = None


class AnalyzeResponse(BaseModel):
    request_id: str
    audio: AudioSummary
    visualizations: VisualizationBlock
    methods: dict[MethodKey, MethodResult]


class MethodInfo(BaseModel):
    method: MethodKey
    display_name: str
    status: MethodStatus
    backbone: str
    has_artifacts: bool
    error: str | None = None


class MethodsResponse(BaseModel):
    methods: list[MethodInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ready_methods: int
    total_methods: int = 4
