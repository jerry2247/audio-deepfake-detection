from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from training.heads import EmbeddingHead
from training.modeling import (
    ModelAccessError,
    count_parameters,
    freeze_module,
)
from training.specs import get_method_spec


MODEL_ID = "ta012/SSLAM_pretrain"
SAMPLE_RATE = 16000
TARGET_FRAMES = 1024
NORM_MEAN = -4.268
NORM_STD = 4.569
MEL_BINS = 128
FEATURE_DIM = 768


def build_head() -> EmbeddingHead:
    return EmbeddingHead(input_dim=FEATURE_DIM, hidden_dim=256, dropout=0.2)


def feature_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "target_frames": TARGET_FRAMES,
        "mel_bins": MEL_BINS,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "feature": "SSLAM extract_features embedding",
        "feature_shape": [FEATURE_DIM],
        "head": "MLP logit over cached SSLAM embedding",
    }


def _extract_embedding(raw_output: Any) -> torch.Tensor:
    output = raw_output
    if isinstance(output, dict):
        for key in ("pooler_output", "features", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
    elif isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise RuntimeError("SSLAM extract_features did not return a tensor-like output")
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output[:, 0, :]
    raise RuntimeError(f"Unexpected SSLAM feature shape: {tuple(output.shape)}")


class SSLAMViTBaseDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = get_method_spec("audio_spectrogram_vit")
        try:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        except Exception as exc:
            raise ModelAccessError(
                f"Could not load required backbone {MODEL_ID!r} with trust_remote_code=True. "
                "Do not substitute another checkpoint. If this is an access, network, "
                "or Hugging Face cache problem, resolve it before continuing."
            ) from exc

        self.backbone_parameter_count = count_parameters(self.backbone)
        freeze_module(self.backbone)
        self.head = build_head()

    def _fbank_one(self, waveform: torch.Tensor) -> torch.Tensor:
        # Kaldi-compatible fbank is CPU-only in torchaudio; compute on CPU and
        # let _prepare_mel move the stacked batch back to the model device.
        waveform = waveform.detach().cpu()
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=MEL_BINS,
            dither=0.0,
            frame_shift=10,
        )
        frames = mel.shape[0]
        if frames < TARGET_FRAMES:
            mel = F.pad(mel, (0, 0, 0, TARGET_FRAMES - frames))
        else:
            mel = mel[:TARGET_FRAMES, :]
        mel = (mel - NORM_MEAN) / (NORM_STD * 2.0)
        return mel

    def _prepare_mel(self, batch: dict[str, Any]) -> torch.Tensor:
        waveforms = batch["waveforms"]
        lengths = batch["lengths"]
        mels = [
            self._fbank_one(waveforms[index, : int(lengths[index].item())])
            for index in range(waveforms.shape[0])
        ]
        return torch.stack(mels, dim=0).unsqueeze(1).to(waveforms.device)

    def extract_features(self, batch: dict[str, Any]) -> torch.Tensor:
        mel = self._prepare_mel(batch)
        with torch.no_grad():
            if hasattr(self.backbone, "extract_features"):
                raw = self.backbone.extract_features(mel)
            else:
                raw = self.backbone(mel)
        return _extract_embedding(raw)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.head(self.extract_features(batch))

    def artifact_config(self) -> dict[str, Any]:
        config = feature_config()
        config.update(
            {
            "backbone_parameters": self.backbone_parameter_count,
            }
        )
        return config


def build_detector(device: torch.device | None = None) -> SSLAMViTBaseDetector:
    detector = SSLAMViTBaseDetector()
    if device is not None:
        detector.to(device)
    return detector
