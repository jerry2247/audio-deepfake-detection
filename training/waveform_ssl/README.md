# WavLM-Base+ Waveform SSL Detector

## Research Purpose

This approach uses a frozen speech self-supervised model as the acoustic
representation. The detector tests whether broad speech SSL features expose
synthetic-speech artifacts when the only trained component is a small binary
head.

## Frozen Backbone

Model: `microsoft/wavlm-base-plus`.

WavLM-Base+ is a 12-layer Transformer speech encoder with hidden size 768 and 12
attention heads. It is preceded by a convolutional feature extractor with an
approximately 20 ms frame stride. The model has 94.70M parameters, uses an MIT
license, and was pretrained on 94k hours of 16 kHz speech: 60k hours
LibriLight, 10k hours GigaSpeech, and 24k hours VoxPopuli.

The backbone is frozen. No backbone weights, adapters, LoRA weights, or
fine-tuning parameters are trained.

## Feature Extraction

Input is the canonical dataset waveform: mono 16 kHz audio loaded from the
manifest path. WavLM consumes the raw waveform directly. No spectrogram,
tokenizer, or ASR frontend is used.

For an audio clip of length `L` seconds, the convolutional frontend produces
approximately `50 * L` frame positions.

With hidden states exposed, WavLM-Base+ returns 13 sequence tensors: the
feature-extractor projection plus the 12 Transformer encoder layers. Each tensor
has shape `(frames, 768)` after removing the batch dimension.

The cache stores compact per-clip features, not full frame sequences. For each
of the 13 hidden-state tensors, the extractor computes the masked temporal mean
and masked temporal standard deviation. The cached feature shape per clip is
`(13, 1536)`.

## Detector Head

The detector head learns a softmax-normalized weighting over the 13 WavLM
layer-statistic vectors, producing one 1536-dimensional clip representation. A
LayerNorm, GELU, dropout MLP maps 1536 to 256 to one logit.

Only the layer weights and classifier are trained. The
implementation is `wavlm_base_plus_detector.py`.

## Training And Evaluation

Feature extraction writes:

```text
training/waveform_ssl/feature_cache/{train,val,test}/
```

Head training reads only the cached features, selects the best head by
validation EER, evaluates the selected head on the test split, and prints final
metrics JSON to stdout.

Saved-head evaluation reads the cached features and writes
`evaluation_<split>.json`.

## Final Output

Final selected artifacts for this method are written to:

```text
final_models/waveform_ssl/
```

The final folder is expected to contain the selected detector head, the
validation-selected configuration, training metrics, and saved-head evaluation
metrics.
