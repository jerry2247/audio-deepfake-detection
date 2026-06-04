# Whisper-Base Encoder Detector

## Research Purpose

This approach uses a frozen supervised ASR encoder as the representation. It
tests whether Whisper's multilingual, noisy-web speech features provide a
different synthetic-speech signal from self-supervised speech models.

## Frozen Backbone

Model: encoder of `openai/whisper-base`.

Whisper-base is an encoder-decoder speech recognition model. This method loads
only the encoder and discards the decoder. The encoder has 6 Transformer layers
and model dimension 512. The full base model has 74M parameters; only the
encoder is used for this detector. Whisper is MIT licensed.

The backbone is frozen. No encoder weights, adapters, LoRA weights, or
fine-tuning parameters are trained.

## Feature Extraction

Whisper's frontend converts the waveform to an 80-bin log-mel spectrogram over
a mandatory 30 second window, producing a feature tensor of shape `(80, 3000)`.
Audio shorter than 30 seconds is padded by the Whisper frontend. That padding is
a fixed property of Whisper's input format, not a dataset preprocessing step.

For a clip with `L` seconds of speech content, the real content occupies roughly
the first `100 * L` mel frames out of 3000.

## Backbone Output

The frozen encoder returns a fixed-length sequence of shape `(1500, 512)` after
removing the batch dimension. The length is fixed because Whisper always sees a
30 second input window.

The cache stores compact per-clip features. The extractor computes the temporal
mean and standard deviation over the speech-content region and excludes padded
encoder positions. The cached feature shape per clip is `(1024)`.

## Detector Head

The detector head maps the cached 1024-dimensional Whisper statistic vector
through a LayerNorm, GELU, dropout MLP from 1024 to 256 to one logit.

Only the classifier is trained. The implementation is
`whisper_base_encoder_detector.py`.

## Training And Evaluation

Feature extraction writes:

```text
training/asr_logmel_encoder/feature_cache/{train,val,test}/
```

Head training reads only the cached features, selects the best head by
validation EER, evaluates the selected head on the test split, and prints final
metrics JSON to stdout.

Saved-head evaluation reads the cached features and writes
`evaluation_<split>.json`.

## Final Output

Final selected artifacts for this method are written to:

```text
final_models/asr_logmel_encoder/
```

The final folder is expected to contain the selected detector head, the
validation-selected configuration, training metrics, and saved-head evaluation
metrics.
