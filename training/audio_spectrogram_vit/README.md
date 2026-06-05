# SSLAM ViT-Base Audio Spectrogram Detector

## Research Purpose

This approach uses a frozen audio-pretrained spectrogram Transformer. It tests
whether a general audio representation trained on mixture-heavy soundscapes
captures synthetic-speech artifacts after the waveform is converted into the
SSLAM input representation.

## Frozen Backbone

Model: SSLAM ViT-Base, weights `ta012/SSLAM_pretrain`.

SSLAM is an audio spectrogram ViT-Base from Alex et al., ICLR 2025. The
Hugging Face feature-extraction checkpoint is `ta012/SSLAM_pretrain`. The model
uses embedding dimension 768 and 16x16 spectrogram patches, with a listed model
size of 90M parameters. The checkpoint is MIT licensed and is loaded with
`trust_remote_code=True` because the repository supplies the EAT-compatible
model code.

The backbone is frozen. No SSLAM weights, adapters, LoRA weights, or fine-tuning
parameters are trained.

The repository's EAT remote code targets transformers 4.x and does not load
under transformers 5. Feature extraction for this method therefore runs in a
dedicated virtual environment pinned to `transformers>=4.49,<5` (with `timm`
installed), created at `.venv_extract/` inside this folder:

```text
python3 -m venv --system-site-packages .venv_extract
.venv_extract/bin/pip install "transformers>=4.49,<5" timm
.venv_extract/bin/python -m training.extract_features --method audio_spectrogram_vit
```

Head training and evaluation are cache-facing and run in the main environment.
A linear-probe check on the cached features measured 4.9 percent validation
EER, confirming the small-magnitude embeddings carry strong class signal.

## Feature Extraction

Input is the SSLAM model-card feature representation: the waveform is mean
centered, converted with `torchaudio.compliance.kaldi.fbank` using 16 kHz audio,
128 mel bins, hanning window, no energy term, zero dither, and a 10 ms frame
shift. The fbank is padded or truncated to 1024 frames, then normalized as
`(mel - -4.268) / (4.569 * 2)`. These values are the checkpoint input contract.

## Backbone Output

The frozen encoder is called through `extract_features`. The cache stores one
768-dimensional SSLAM feature per clip.

## Detector Head

The detector head maps the 768-dimensional SSLAM feature through a LayerNorm,
GELU, dropout MLP from 768 to 256 to one logit. Only the detector head is
trained. The implementation is `sslam_vit_base_detector.py`.

## Training And Evaluation

Feature extraction writes:

```text
training/audio_spectrogram_vit/feature_cache/{train,val,test}/
```

Head training reads only the cached features, selects the best head by
validation EER, evaluates the selected head on the test split, and prints final
metrics JSON to stdout.

Saved-head evaluation reads the cached features and writes
`evaluation_<split>.json`.

## Final Output

Final selected artifacts for this method are written to:

```text
final_models/audio_spectrogram_vit/
```

The final folder is expected to contain the selected detector head, the
validation-selected configuration, training metrics, and saved-head evaluation
metrics.
