from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio import decode_audio_upload
from .schemas import AnalyzeResponse, HealthResponse, MethodsResponse
from .service import AudioLabService


def create_app(service: AudioLabService | None = None) -> FastAPI:
    app = FastAPI(
        title="AI Audio Lab API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    allowed_origins = [
        origin.strip()
        for origin in os.environ.get(
            "AUDIO_LAB_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:4173,http://127.0.0.1:4173,"
            "https://www.jerrygu.com,https://jerrygu.com",
        ).split(",")
        if origin.strip()
    ]
    # Vercel preview deployments use per-deploy subdomains, so they are
    # matched by pattern rather than enumerated.
    allowed_origin_regex = os.environ.get(
        "AUDIO_LAB_CORS_ORIGIN_REGEX",
        r"https://.*\.vercel\.app",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=allowed_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.audio_lab_service = service or AudioLabService(
        model_root=os.environ.get("AUDIO_LAB_MODEL_ROOT", "final_models"),
        device=os.environ.get("AUDIO_LAB_DEVICE", "auto"),
        load_models=os.environ.get("AUDIO_LAB_LOAD_MODE", "load") != "inspect",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return app.state.audio_lab_service.health()

    @app.get("/methods", response_model=MethodsResponse)
    def methods() -> MethodsResponse:
        return app.state.audio_lab_service.method_info()

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
        try:
            payload = await file.read()
            audio = decode_audio_upload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return app.state.audio_lab_service.analyze(audio)

    return app


app = create_app()
