# CS153 AI Audio Detection

Binary detection of real human speech versus AI-generated speech using frozen
pretrained backbones and lightweight detector heads.

The project compares four detector families on one frozen dataset. Each method
derives its own features from the same canonical waveform clips, freezes the
pretrained backbone, and trains only a small detector head. The main evaluation
metric is Equal Error Rate (EER) on the dataset test split. Splits are supplied
by the dataset and are leakage safe: whole groups sharing a speaker, a source
recording, or a transcript appearing in both classes are assigned to a single
split (seed 301), and every split is balanced exactly 1:1 real to fake.

## AI Usage Disclosure

I used Claude Code and OpenAI Codex in setting up the experimental and data pipelines. Gemini Deep Research was used for researching AI architectures and literature in the area of audio deepfake detection, and Antigravity CLI (a new Google tool I wanted to explore and learn) was used for frontend UI design.

I also used Claude Code in helping with polishing documentation to be professional.

## Results

Dataset v1.0: 9,084 clips (4,542 real, 4,542 fake; train 3,184, validation
452, test 906 per class), about 17 hours, 73 distinct fake generators of 2025
to 2026 vintage. Heads are selected by validation EER; the test split plays no
role in selection. Metrics below are the selected head evaluated once per
split. Lower EER is better.

| method | backbone | split | loss | EER | accuracy at 0.5 |
|---|---|---|---|---|---|
| waveform_ssl | WavLM-Base+ (94.4M, frozen) | train | 0.257 | 3.80% | 89.0% |
| | | val | 0.317 | 5.31% | 88.1% |
| | | test | 0.337 | **5.08%** | 85.4% |
| vision_spectrogram_vit | DINOv3 ViT-S/16 (21.6M, frozen) | train | 0.058 | 1.88% | 97.9% |
| | | val | 0.107 | 3.54% | 96.8% |
| | | test | 0.141 | **5.41%** | 94.6% |
| asr_logmel_encoder | Whisper-base encoder (20.6M, frozen) | train | 0.168 | 4.93% | 93.9% |
| | | val | 0.181 | 4.20% | 93.0% |
| | | test | 0.177 | **5.63%** | 93.7% |
| audio_spectrogram_vit | SSLAM ViT-Base (90.0M, frozen) | train | 0.175 | 6.31% | 93.3% |
| | | val | 0.173 | 5.31% | 93.1% |
| | | test | 0.232 | **8.72%** | 90.3% |

Observations. The waveform self-supervised model gives the best test EER, and
its learned layer attention concentrates on early Transformer layers (2 and 3),
supporting the hypothesis that synthesis artifacts live in low-level acoustic
representations rather than top-layer semantics. A pure vision backbone applied
to spectrogram images is nearly as strong while being the smallest model. The
general-audio spectrogram model trails the speech-specialized representations.
Accuracy at the fixed 0.5 threshold is reported for completeness; EER is the
threshold-free headline metric.

## Repository Layout

```text
README.md
dataset/
  DATASHEET.md            frozen dataset contract
  SOURCES.md              composition record
  manifest.parquet        one row per clip, produced by the dataset build
  dataset_card.json       build metadata, counts, manifest hash
  audio/{train,val,test}/{real,fake}/<clip_id>.wav
  sources/  build/  common/   dataset-side machinery
training/
  README.md               training protocol, workflow, and results
  specs.py  data.py  engine.py  heads.py  metrics.py  modeling.py
  extract_features.py  train_detector.py  evaluate_detector.py
  waveform_ssl/            WavLM-Base+ method
  asr_logmel_encoder/      Whisper-base encoder method
  audio_spectrogram_vit/   SSLAM ViT-Base method
  vision_spectrogram_vit/  DINOv3 ViT-S/16 method
final_models/
  <method>/detector_head.pt  config.json  metrics.json  evaluation_test.json
backend/
  README.md
```

## Scope Boundaries

`dataset/` owns the frozen dataset contract, manifest, metadata, and audio. The
dataset contract is documented in `dataset/DATASHEET.md`.

`training/` owns the four detector methods and the shared training engine. It
reads `dataset/` and never writes back into it.

`final_models/` stores the final trained model deliverables: the
validation-selected detector head, its configuration, training metrics with
train, validation, and test results, and saved-head evaluation records.

`backend/` owns the inference API for the public demo. It reads trained
artifacts from `final_models/`, reuses the inference and preprocessing code in
`training/`, and exposes results for the frontend. It does not train models or
modify the dataset.

## Dataset Handoff

Training consumes:

- `dataset/manifest.parquet`
- `dataset/dataset_card.json`
- WAV files referenced by the manifest under `dataset/audio/`

Every delivered clip is mono 16 kHz WAV PCM16, greater than 0 seconds and at
most 30 seconds. Splits are supplied by the dataset side and are never rebuilt
by training code. The dataset build is complete at version v1.0.

## Methods

| Method folder | Frozen backbone | Cached feature per clip | Detector head |
|---|---|---|---|
| `training/waveform_ssl/` | WavLM-Base+ | [13, 1536] per-layer stats | layer attention + MLP |
| `training/asr_logmel_encoder/` | Whisper-base encoder | [7, 1024] per-layer stats | layer attention + MLP |
| `training/audio_spectrogram_vit/` | SSLAM ViT-Base | [768] embedding | MLP |
| `training/vision_spectrogram_vit/` | DINOv3 ViT-S/16 | [1152] CLS + patch stats | MLP |

Each method trains only the detector head (about 0.2M to 0.5M parameters). No
backbone fine-tuning, LoRA, adapters, or backbone updates are part of this
project scope. `training/README.md` documents the full protocol, the feature
cache policy, and reproduction commands.
