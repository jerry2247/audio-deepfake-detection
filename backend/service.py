from __future__ import annotations

import importlib
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.modeling import ModelAccessError
from training.specs import METHODS, MethodSpec

from .audio import DecodedAudio, make_model_batch
from .schemas import (
    AnalyzeResponse,
    AudioSummary,
    HealthResponse,
    MethodInfo,
    MethodKey,
    MethodResult,
    MethodsResponse,
)
from .visuals import build_visualizations


DEFAULT_MODEL_ROOT = Path(os.environ.get("AUDIO_LAB_MODEL_ROOT", "final_models"))


@dataclass
class LoadedMethod:
    spec: MethodSpec
    status: str
    detector: Any | None = None
    error: str | None = None

    @property
    def display_name(self) -> str:
        return self.spec.display_name


class AudioLabService:
    def __init__(
        self,
        *,
        model_root: str | Path = DEFAULT_MODEL_ROOT,
        device: str = "auto",
        load_models: bool = True,
    ) -> None:
        self.model_root = Path(model_root)
        self.device = self._resolve_device(device)
        self.methods: dict[str, LoadedMethod] = {}
        for spec in METHODS.values():
            self.methods[spec.key] = self._load_method(spec) if load_models else self._inspect_method(spec)

    def health(self) -> HealthResponse:
        ready = sum(1 for method in self.methods.values() if method.status == "ready")
        return HealthResponse(status="ok", ready_methods=ready, total_methods=len(self.methods))

    def method_info(self) -> MethodsResponse:
        return MethodsResponse(
            methods=[
                MethodInfo(
                    method=method.spec.key,
                    display_name=method.spec.display_name,
                    status=method.status,
                    backbone=method.spec.backbone_id,
                    has_artifacts=self._has_artifacts(method.spec),
                    error=method.error,
                )
                for method in self.methods.values()
            ]
        )

    def analyze(self, audio: DecodedAudio) -> AnalyzeResponse:
        batch = make_model_batch(audio, self.device)
        results: dict[MethodKey, MethodResult] = {}
        for key, method in self.methods.items():
            results[key] = self._run_method(method, batch)
        return AnalyzeResponse(
            request_id=str(uuid.uuid4()),
            audio=AudioSummary(
                duration_s=round(audio.duration_s, 4),
                sample_rate=audio.sample_rate,
                channels=1,
                num_samples=audio.num_samples,
            ),
            visualizations=build_visualizations(audio),
            methods=results,
        )

    def _inspect_method(self, spec: MethodSpec) -> LoadedMethod:
        if not self._has_artifacts(spec):
            return LoadedMethod(spec=spec, status="not_trained")
        return LoadedMethod(
            spec=spec,
            status="not_loaded",
            error="Model artifacts are present, but model loading is disabled.",
        )

    def _load_method(self, spec: MethodSpec) -> LoadedMethod:
        if not self._has_artifacts(spec):
            return LoadedMethod(spec=spec, status="not_trained")
        try:
            module = importlib.import_module(spec.module)
            detector = module.build_detector(device=self.device)
            state = torch.load(
                self._artifact_dir(spec) / "detector_head.pt",
                map_location=self.device,
                weights_only=True,
            )
            detector.head.load_state_dict(state)
            detector.eval()
            return LoadedMethod(spec=spec, status="ready", detector=detector)
        except ModelAccessError as exc:
            # Surface the full chain in server logs; the API keeps the short message.
            traceback.print_exc()
            return LoadedMethod(spec=spec, status="model_access_error", error=str(exc))
        except Exception as exc:
            traceback.print_exc()
            return LoadedMethod(spec=spec, status="runtime_error", error=str(exc))

    def _run_method(self, method: LoadedMethod, batch: dict[str, Any]) -> MethodResult:
        if method.status != "ready" or method.detector is None:
            return MethodResult(
                status=method.status,
                display_name=method.display_name,
                error=method.error,
            )
        started = time.perf_counter()
        try:
            with torch.no_grad():
                logits = method.detector(batch)
                probability = torch.sigmoid(logits).detach().cpu().view(-1)[0].item()
            return MethodResult(
                status="ready",
                display_name=method.display_name,
                probability_fake=round(float(probability), 6),
                prediction="fake" if probability >= 0.5 else "real",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
        except Exception as exc:
            return MethodResult(
                status="runtime_error",
                display_name=method.display_name,
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2),
                error=str(exc),
            )

    def _artifact_dir(self, spec: MethodSpec) -> Path:
        return self.model_root / spec.final_subdir

    def _has_artifacts(self, spec: MethodSpec) -> bool:
        root = self._artifact_dir(spec)
        return all(
            (root / name).exists()
            for name in ("detector_head.pt", "config.json", "metrics.json")
        )

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        if preference == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if preference not in {"cuda", "cpu"}:
            raise ValueError("backend device must be auto, cuda, or cpu")
        return torch.device(preference)
