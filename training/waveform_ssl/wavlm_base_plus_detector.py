from __future__ import annotations

from typing import Any

import torch
from torch import nn

from training.heads import LayerWeightedStatsHead, lengths_to_mask, masked_mean_std
from training.modeling import ModelAccessError, count_parameters, freeze_module
from training.specs import get_method_spec


MODEL_ID = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
CONV_STRIDE_SAMPLES = 320
NUM_FEATURE_LAYERS = 13
HIDDEN_SIZE = 768


def build_head() -> LayerWeightedStatsHead:
    return LayerWeightedStatsHead(
        num_layers=NUM_FEATURE_LAYERS,
        stats_dim=HIDDEN_SIZE * 2,
        hidden_dim=256,
        dropout=0.2,
    )


def feature_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "feature": "per-layer masked mean and standard deviation",
        "feature_shape": [NUM_FEATURE_LAYERS, HIDDEN_SIZE * 2],
        "head": "learned layer weighting over cached per-layer statistics, MLP logit",
    }


class WavLMBasePlusDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = get_method_spec("waveform_ssl")
        try:
            from transformers import AutoFeatureExtractor, WavLMModel

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self.backbone = WavLMModel.from_pretrained(MODEL_ID)
        except Exception as exc:
            raise ModelAccessError(
                f"Could not load required backbone {MODEL_ID!r}. "
                "Do not substitute another checkpoint. If this is an access, network, "
                "or Hugging Face cache problem, resolve it before continuing."
            ) from exc

        self.backbone_parameter_count = count_parameters(self.backbone)
        freeze_module(self.backbone)
        self.head = build_head()

    def _prepare_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        waveforms = batch["waveforms"]
        lengths = batch["lengths"]
        arrays = [
            waveforms[index, : int(lengths[index].item())].detach().cpu().numpy()
            for index in range(waveforms.shape[0])
        ]
        encoded = self.feature_extractor(
            arrays,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        device = waveforms.device
        return {key: value.to(device) for key, value in encoded.items()}

    def _frame_mask(self, hidden_frames: int, input_attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "_get_feature_vector_attention_mask"):
            return self.backbone._get_feature_vector_attention_mask(
                hidden_frames,
                input_attention_mask,
            ).bool()
        lengths = input_attention_mask.sum(dim=1)
        frame_lengths = torch.div(
            lengths + CONV_STRIDE_SAMPLES - 1,
            CONV_STRIDE_SAMPLES,
            rounding_mode="floor",
        )
        frame_lengths = frame_lengths.clamp_min(1).clamp_max(hidden_frames)
        return lengths_to_mask(frame_lengths, hidden_frames)

    def extract_features(self, batch: dict[str, Any]) -> torch.Tensor:
        encoded = self._prepare_inputs(batch)
        with torch.no_grad():
            outputs = self.backbone(
                input_values=encoded["input_values"],
                attention_mask=encoded.get("attention_mask"),
                output_hidden_states=True,
                return_dict=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")
        frame_mask = self._frame_mask(
            outputs.hidden_states[0].shape[1],
            encoded["attention_mask"],
        )
        stats = [masked_mean_std(hidden, frame_mask) for hidden in outputs.hidden_states]
        return torch.stack(stats, dim=1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.head(self.extract_features(batch))

    def artifact_config(self) -> dict[str, Any]:
        config = feature_config()
        config.update(
            {
                "backbone_parameters": self.backbone_parameter_count,
                "hidden_size": int(self.backbone.config.hidden_size),
                "num_hidden_layers": int(self.backbone.config.num_hidden_layers),
                "num_attention_heads": int(self.backbone.config.num_attention_heads),
            }
        )
        return config


def build_detector(device: torch.device | None = None) -> WavLMBasePlusDetector:
    detector = WavLMBasePlusDetector()
    if device is not None:
        detector.to(device)
    return detector
