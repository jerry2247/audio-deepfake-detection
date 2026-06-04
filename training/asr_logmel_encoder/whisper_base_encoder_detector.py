from __future__ import annotations

from typing import Any

import torch
from torch import nn

from training.heads import LayerWeightedStatsHead, lengths_to_mask, masked_mean_std
from training.modeling import (
    ModelAccessError,
    count_parameters,
    freeze_module,
)
from training.specs import get_method_spec


MODEL_ID = "openai/whisper-base"
SAMPLE_RATE = 16000
ENCODER_FRAMES_PER_SECOND = 50
# Whisper-base encoder exposes 7 hidden states: the post-convolution embedding
# sequence plus the 6 Transformer layers. Synthetic-speech artifacts are not
# strongest at the top layer, so the cache keeps every layer and the head
# learns the layer weighting (the standard layer-probing protocol, matching
# the waveform_ssl method for a controlled comparison).
NUM_FEATURE_LAYERS = 7
D_MODEL = 512
FEATURE_DIM = D_MODEL * 2


def build_head() -> LayerWeightedStatsHead:
    return LayerWeightedStatsHead(
        num_layers=NUM_FEATURE_LAYERS,
        stats_dim=FEATURE_DIM,
        hidden_dim=256,
        dropout=0.2,
    )


def feature_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "feature": "per-layer padding-aware encoder mean and standard deviation",
        "feature_shape": [NUM_FEATURE_LAYERS, FEATURE_DIM],
        "head": "learned softmax layer weighting over cached per-layer statistics, MLP logit",
    }


class WhisperBaseEncoderDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = get_method_spec("asr_logmel_encoder")
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel

            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
            model = WhisperModel.from_pretrained(MODEL_ID)
            self.config = model.config
            self.backbone = model.encoder
            del model
        except Exception as exc:
            raise ModelAccessError(
                f"Could not load required backbone {MODEL_ID!r}. "
                "Do not substitute another checkpoint. If this is an access, network, "
                "or Hugging Face cache problem, resolve it before continuing."
            ) from exc

        self.backbone_parameter_count = count_parameters(self.backbone)
        freeze_module(self.backbone)
        self.head = build_head()

    def _prepare_inputs(self, batch: dict[str, Any]) -> torch.Tensor:
        waveforms = batch["waveforms"]
        lengths = batch["lengths"]
        arrays = [
            waveforms[index, : int(lengths[index].item())].detach().cpu().numpy()
            for index in range(waveforms.shape[0])
        ]
        encoded = self.feature_extractor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        return encoded.input_features.to(waveforms.device)

    def _encoder_mask(self, batch: dict[str, Any], frames: int) -> torch.Tensor:
        seconds = batch["lengths"].to(torch.float32) / float(SAMPLE_RATE)
        frame_lengths = torch.ceil(seconds * ENCODER_FRAMES_PER_SECOND).to(torch.long)
        frame_lengths = frame_lengths.clamp_min(1).clamp_max(frames)
        return lengths_to_mask(frame_lengths, frames)

    def extract_features(self, batch: dict[str, Any]) -> torch.Tensor:
        input_features = self._prepare_inputs(batch)
        with torch.no_grad():
            outputs = self.backbone(
                input_features,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != NUM_FEATURE_LAYERS:
            count = "none" if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"Whisper encoder returned {count} hidden states, "
                f"expected {NUM_FEATURE_LAYERS}"
            )
        mask = self._encoder_mask(batch, hidden_states[0].shape[1])
        stats = [masked_mean_std(hidden, mask) for hidden in hidden_states]
        return torch.stack(stats, dim=1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.head(self.extract_features(batch))

    def artifact_config(self) -> dict[str, Any]:
        config = feature_config()
        config.update(
            {
            "encoder_parameters": self.backbone_parameter_count,
            "encoder_layers": int(self.config.encoder_layers),
            "encoder_attention_heads": int(self.config.encoder_attention_heads),
            "d_model": int(self.config.d_model),
            "num_mel_bins": int(self.config.num_mel_bins),
            "max_source_positions": int(self.config.max_source_positions),
            }
        )
        return config


def build_detector(device: torch.device | None = None) -> WhisperBaseEncoderDetector:
    detector = WhisperBaseEncoderDetector()
    if device is not None:
        detector.to(device)
    return detector
