# Whisper-Base Encoder Detector

## Research Purpose

This approach uses a frozen supervised ASR encoder as the representation. It
tests whether Whisper's multilingual, noisy-web speech features provide a
different synthetic-speech signal from self-supervised speech models.

## Frozen Backbone

Model: encoder of `openai/whisper-base`.

Whisper-base is an encoder-decoder speech recognition model. This method loads
only the encoder and discards the decoder. The encoder has 6 Transformer layers,
model dimension 512, and 20.59M parameters (the commonly quoted 74M figure for
whisper-base includes the unused decoder). Whisper is MIT licensed.

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

With hidden states exposed, the frozen encoder returns 7 fixed-length sequence
tensors: the post-convolution embedding sequence plus the 6 Transformer
layers, each of shape `(1500, 512)` after removing the batch dimension. The
length is fixed because Whisper always sees a 30 second input window.

The cache stores compact per-clip features, not full frame sequences. For each
of the 7 hidden-state tensors, the extractor computes the temporal mean and
standard deviation over the speech-content region, excluding padded encoder
positions. The cached feature shape per clip is `(7, 1024)`. Keeping every
layer matters because synthetic-speech artifacts are not strongest at the top
layer of a speech encoder; which layers carry the signal is learned, and the
protocol matches the waveform_ssl method for a controlled comparison.

This cache is committed to the repository by project decision, so the
detector-head experiments below are reproducible without re-running the
backbone.

## Detector Head

The detector head learns a softmax-normalized weighting over the 7 cached
layer-statistic vectors, an attention over layers with one learned logit per
layer, producing a single 1024-dimensional clip representation. A LayerNorm,
GELU, dropout MLP maps 1024 to 256 to one logit. The head has 264,712
trainable parameters. The learned layer weights are themselves reportable:
they show which encoder depths carry the synthetic-speech signal.

Only the layer weights and classifier are trained. The implementation is
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
