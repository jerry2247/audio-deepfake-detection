from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch import nn

from training.heads import EmbeddingHead
from training.modeling import (
    ModelAccessError,
    count_parameters,
    freeze_module,
)
from training.specs import get_method_spec


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
SAMPLE_RATE = 16000
IMAGE_SIZE = 224
MEL_BINS = 128
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
TOP_DB = 80.0
REGISTER_TOKENS = 4
EMBED_DIM = 384
FEATURE_DIM = EMBED_DIM * 3


def build_head() -> EmbeddingHead:
    return EmbeddingHead(input_dim=FEATURE_DIM, hidden_dim=256, dropout=0.2)


def feature_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "image_size": IMAGE_SIZE,
        "mel_bins": MEL_BINS,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "top_db": TOP_DB,
        "feature": "pooler_output concatenated with patch-token mean and std",
        "feature_shape": [FEATURE_DIM],
        "head": "MLP logit over cached DINOv3 spectrogram feature",
    }


class DINOv3ViTS16Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = get_method_spec("vision_spectrogram_vit")
        try:
            from transformers import AutoImageProcessor, AutoModel

            self.image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.backbone = AutoModel.from_pretrained(MODEL_ID)
        except Exception as exc:
            raise ModelAccessError(
                f"Could not load required gated backbone {MODEL_ID!r}. "
                "Accept the Hugging Face license and authenticate before continuing. "
                "Do not substitute another checkpoint."
            ) from exc

        self.backbone_parameter_count = count_parameters(self.backbone)
        freeze_module(self.backbone)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=MEL_BINS,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=TOP_DB)
        self.head = build_head()

    def _spectrogram_image_one(self, waveform: torch.Tensor) -> Image.Image:
        mel = self.mel(waveform.unsqueeze(0))
        db = self.to_db(mel).squeeze(0)
        db_min = db.amin()
        db_max = db.amax()
        scaled = (db - db_min) / (db_max - db_min).clamp_min(1e-6)
        image = (scaled * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        image = np.stack([image, image, image], axis=-1)
        return Image.fromarray(image, mode="RGB")

    def _prepare_pixels(self, batch: dict[str, Any]) -> torch.Tensor:
        waveforms = batch["waveforms"]
        lengths = batch["lengths"]
        images = [
            self._spectrogram_image_one(waveforms[index, : int(lengths[index].item())])
            for index in range(waveforms.shape[0])
        ]
        encoded = self.image_processor(
            images=images,
            return_tensors="pt",
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        )
        return encoded["pixel_values"].to(waveforms.device)

    def _pooled_patch_statistics(self, outputs: Any) -> torch.Tensor:
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("DINOv3 output is missing last_hidden_state")
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"Unexpected DINOv3 hidden shape: {tuple(hidden.shape)}")

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = hidden[:, 0, :]
        patch_start = 1 + REGISTER_TOKENS
        if hidden.shape[1] <= patch_start:
            raise RuntimeError("DINOv3 output does not contain patch tokens")
        patches = hidden[:, patch_start:, :]
        patch_mean = patches.mean(dim=1)
        patch_std = patches.std(dim=1, unbiased=False)
        return torch.cat([pooled, patch_mean, patch_std], dim=-1)

    def extract_features(self, batch: dict[str, Any]) -> torch.Tensor:
        pixels = self._prepare_pixels(batch)
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixels, return_dict=True)
        return self._pooled_patch_statistics(outputs)

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


def build_detector(device: torch.device | None = None) -> DINOv3ViTS16Detector:
    detector = DINOv3ViTS16Detector()
    if device is not None:
        detector.to(device)
    return detector
