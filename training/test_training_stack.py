from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.data import AudioManifestDataset, collate_audio, load_manifest, records_for_split
from training.engine import TrainConfig, evaluate_head, evaluate_saved_detector, fit_detector
from training.heads import (
    EmbeddingHead,
    LayerWeightedStatsHead,
    lengths_to_mask,
    masked_mean_std,
)
from training.metrics import equal_error_rate


def write_pcm16_wav(path: Path, samples: int, sample_rate: int = 16000) -> None:
    audio = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def write_tiny_manifest(root: Path) -> None:
    rows = []
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            clip_id = f"clip_{len(rows):032x}"
            rel = f"audio/{split}/{label}/{clip_id}.wav"
            audio_path = root / rel
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            write_pcm16_wav(audio_path, samples=1600)
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": rel,
                    "label": label,
                    "split": split,
                    "eval_condition": "train_pool" if split == "train" else "validation_pool",
                    "split_group_id": f"group_{clip_id}",
                    "is_heldout_generator": False,
                    "is_in_the_wild": False,
                    "source_dataset": "unit",
                    "generator": "human" if label == "real" else "unit_fake",
                    "generator_family": "human" if label == "real" else "unit_fake",
                    "duration_s": 0.1,
                    "final_sample_rate": 16000,
                    "final_channels": 1,
                    "final_format": "wav_pcm16",
                    "bit_depth": 16,
                    "sha256": "0" * 64,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)


class TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def write_tiny_cache(cache_root: Path, method: str = "waveform_ssl") -> None:
    for split in ("train", "val", "test"):
        split_dir = cache_root / method / "feature_cache" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        features = torch.tensor([[0.0], [1.0]], dtype=torch.float16)
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
        torch.save({"features": features, "labels": labels}, split_dir / "features.pt")
        (split_dir / "metadata.json").write_text(
            """{"feature_shape":[1],"num_examples":2}""",
            encoding="utf-8",
        )


class TrainingStackTests(unittest.TestCase):
    def test_equal_error_rate_prefers_separated_scores(self) -> None:
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        good_logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        bad_logits = torch.tensor([3.0, 1.0, -1.0, -3.0])
        self.assertEqual(equal_error_rate(good_logits, labels), 0.0)
        self.assertGreater(equal_error_rate(bad_logits, labels), 0.5)

    def test_heads_accept_masked_sequences_and_embeddings(self) -> None:
        mask = lengths_to_mask(torch.tensor([5, 3]), 5)
        stats = masked_mean_std(torch.randn(2, 5, 768), mask)
        self.assertEqual(tuple(stats.shape), (2, 1536))

        wavlm_cached_head = LayerWeightedStatsHead(num_layers=13, stats_dim=1536)
        self.assertEqual(tuple(wavlm_cached_head(torch.randn(2, 13, 1536)).shape), (2,))

        embedding_head = EmbeddingHead(input_dim=1152)
        self.assertEqual(tuple(embedding_head(torch.randn(2, 1152)).shape), (2,))

    def test_manifest_audio_dataset_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tiny_manifest(root)

            manifest = load_manifest(root)
            train_records = records_for_split(manifest, "train")
            self.assertEqual(len(train_records), 2)
            dataset = AudioManifestDataset(root, train_records)
            batch = collate_audio([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["waveforms"].shape), (2, 1600))
            self.assertEqual(tuple(batch["labels"].shape), (2,))

    def test_evaluate_head_with_cached_features(self) -> None:
        class TinyDataset(Dataset[dict[str, Any]]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict[str, Any]:
                label = float(index >= 2)
                return {"features": torch.tensor([label]), "labels": torch.tensor(label)}

        def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "features": torch.stack([item["features"] for item in batch]),
                "labels": torch.stack([item["labels"] for item in batch]),
            }

        head = TinyHead()
        with torch.no_grad():
            head.linear.weight.fill_(4.0)
            head.linear.bias.fill_(-2.0)
        loader = DataLoader(TinyDataset(), batch_size=2, collate_fn=collate)
        metrics = evaluate_head(head, loader, torch.device("cpu"))
        self.assertEqual(metrics["eer"], 0.0)
        self.assertIn("loss", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)

    def test_evaluate_saved_detector_writes_split_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "training"
            output = Path(tmp) / "final_models"
            write_tiny_cache(cache_root)
            method_output = output / "waveform_ssl"
            method_output.mkdir(parents=True)
            head = TinyHead()
            with torch.no_grad():
                head.linear.weight.fill_(4.0)
                head.linear.bias.fill_(-2.0)
            torch.save(head.state_dict(), method_output / "detector_head.pt")

            config = TrainConfig(
                cache_root=str(cache_root),
                output_root=str(output),
                batch_size=2,
                print_progress=False,
            )
            with patch("training.engine.build_head", return_value=TinyHead()):
                payload = evaluate_saved_detector("waveform_ssl", config, split="test")

            self.assertEqual(payload["metrics"]["eer"], 0.0)
            self.assertTrue((method_output / "evaluation_test.json").exists())

    def test_fit_detector_trains_from_cache_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "training"
            output = Path(tmp) / "final_models"
            write_tiny_cache(cache_root)
            config = TrainConfig(
                cache_root=str(cache_root),
                output_root=str(output),
                batch_size=2,
                epochs=10,
                learning_rate=0.1,
                patience=10,
                print_progress=False,
            )
            with patch("training.engine.build_head", return_value=TinyHead()):
                metrics = fit_detector("waveform_ssl", config)

            self.assertIn("selected_head", metrics)
            for split in ("train", "val", "test"):
                split_metrics = metrics["selected_head"][split]
                for key in ("loss", "eer", "accuracy_at_0_5"):
                    self.assertIn(key, split_metrics)
            self.assertTrue((output / "waveform_ssl" / "detector_head.pt").exists())
            self.assertTrue((output / "waveform_ssl" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
