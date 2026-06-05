from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    module: str
    builder: str
    final_subdir: str
    backbone_id: str
    expected_backbone_parameters: int


METHODS: dict[str, MethodSpec] = {
    "waveform_ssl": MethodSpec(
        key="waveform_ssl",
        display_name="WavLM-Base+ waveform SSL",
        module="training.waveform_ssl.wavlm_base_plus_detector",
        builder="build_detector",
        final_subdir="waveform_ssl",
        backbone_id="microsoft/wavlm-base-plus",
        # exact count verified against the loaded checkpoint
        expected_backbone_parameters=94_381_936,
    ),
    "asr_logmel_encoder": MethodSpec(
        key="asr_logmel_encoder",
        display_name="Whisper-base encoder",
        module="training.asr_logmel_encoder.whisper_base_encoder_detector",
        builder="build_detector",
        final_subdir="asr_logmel_encoder",
        backbone_id="openai/whisper-base",
        # encoder only; the 74M figure for whisper-base includes the unused
        # decoder. Exact count verified against the loaded checkpoint.
        expected_backbone_parameters=20_590_592,
    ),
    "audio_spectrogram_vit": MethodSpec(
        key="audio_spectrogram_vit",
        display_name="SSLAM ViT-Base audio spectrogram",
        module="training.audio_spectrogram_vit.sslam_vit_base_detector",
        builder="build_detector",
        final_subdir="audio_spectrogram_vit",
        backbone_id="ta012/SSLAM_pretrain",
        # exact count verified against the loaded checkpoint
        expected_backbone_parameters=89_972_736,
    ),
    "vision_spectrogram_vit": MethodSpec(
        key="vision_spectrogram_vit",
        display_name="DINOv3 ViT-S/16 vision spectrogram",
        module="training.vision_spectrogram_vit.dinov3_vit_s16_detector",
        builder="build_detector",
        final_subdir="vision_spectrogram_vit",
        backbone_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        expected_backbone_parameters=21_600_000,
    ),
}


def get_method_spec(method: str) -> MethodSpec:
    try:
        return METHODS[method]
    except KeyError as exc:
        valid = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method {method!r}. Valid methods: {valid}") from exc


def final_output_dir(output_root: str | Path, spec: MethodSpec) -> Path:
    return Path(output_root) / spec.final_subdir
