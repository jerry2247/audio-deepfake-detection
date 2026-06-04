# CS153 AI Audio Detection

Binary detection of real human speech versus AI-generated speech using frozen
pretrained backbones and lightweight detector heads.

The project compares four detector families on one frozen dataset. Each method
derives its own features from the same canonical waveform clips, freezes the
pretrained backbone, and trains only a small detector head. The main evaluation
metric is Equal Error Rate (EER) on the dataset-provided speaker-disjoint and
generator-disjoint splits.

## Repository Layout

```text
README.md
dataset/
  DATASHEET.md
  manifest.parquet      produced by the dataset build
  dataset_card.json     produced by the dataset build
  audio/
    train/{real,fake}/
    val/{real,fake}/
    test/{real,fake}/
training/
  waveform_ssl/
    WAVLM_BASE_PLUS_BLUEPRINT.md
  asr_logmel_encoder/
    WHISPER_BASE_ENCODER_BLUEPRINT.md
  audio_spectrogram_vit/
    SSLAM_VIT_BASE_BLUEPRINT.md
  vision_spectrogram_vit/
    DINOV3_VIT_S16_BLUEPRINT.md
final_models/
  waveform_ssl/
  asr_logmel_encoder/
  audio_spectrogram_vit/
  vision_spectrogram_vit/
```

## Scope Boundaries

`dataset/` owns the frozen dataset contract, manifest, metadata, and audio. The
dataset contract is documented in `dataset/DATASHEET.md`.

`training/` owns the four method blueprints and, later, the method-specific
training code. It contains exactly four method folders.

`final_models/` stores the final trained model deliverables. Training reads
`dataset/` and writes final selected artifacts to `final_models/`; it does not
write back into `dataset/`.

## Dataset Handoff

Training consumes:

- `dataset/manifest.parquet`
- `dataset/dataset_card.json`
- WAV files referenced by the manifest under `dataset/audio/`

Every delivered clip is expected to be mono 16 kHz WAV PCM16, greater than 0
seconds and at most 30 seconds. Splits are supplied by the dataset side and must
not be rebuilt by training code.

The repository currently contains only the dataset folder scaffold and
`dataset/DATASHEET.md`; the manifest, card, and audio files are produced when
the dataset build is run.

## Methods

| Method folder | Blueprint | Frozen backbone |
|---|---|---|
| `training/waveform_ssl/` | `WAVLM_BASE_PLUS_BLUEPRINT.md` | WavLM-Base+ |
| `training/asr_logmel_encoder/` | `WHISPER_BASE_ENCODER_BLUEPRINT.md` | Whisper-base encoder |
| `training/audio_spectrogram_vit/` | `SSLAM_VIT_BASE_BLUEPRINT.md` | SSLAM ViT-Base |
| `training/vision_spectrogram_vit/` | `DINOV3_VIT_S16_BLUEPRINT.md` | DINOv3 ViT-S/16 |

Each method trains only the detector head. No backbone fine-tuning, LoRA,
adapters, or backbone updates are part of this project scope.

## Final Model Deliverables

Final trained model outputs are stored here:

```text
final_models/waveform_ssl/
final_models/asr_logmel_encoder/
final_models/audio_spectrogram_vit/
final_models/vision_spectrogram_vit/
```

These folders are the canonical locations for selected trained heads,
validation-selected configuration, calibration metadata if used, and final EER
results.
