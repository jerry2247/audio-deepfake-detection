# Training

This folder contains the four model-training approaches for AI speech
detection. Each approach uses the same research protocol: extract compact
features from a frozen backbone, train a small detector head from cached
features, select the head by validation EER, and report final EER on the
dataset test split.

## Boundaries

- No dataset files are modified.
- No backbone weights are trained.
- No LoRA, adapters, or fine-tuning are used.
- No substitute checkpoint is used if a required Hugging Face model cannot be
  loaded.
- `final_models/` is populated only by an actual head-training or saved-head
  evaluation run.

## Data Contract

The loader expects exactly the dataset format documented in
`dataset/DATASHEET.md`: `manifest.parquet` at the dataset root, paths relative
to that root, labels `real` and `fake`, splits `train`, `val`, and `test`, and
WAV PCM16 mono 16 kHz audio under `audio/<split>/<label>/<clip_id>.wav`.

The loader rejects wrong paths, wrong sample rates, non-mono audio, non-WAV
files, non-PCM16 WAV files, empty clips, and clips longer than 30 seconds.

## Methods

```text
waveform_ssl             microsoft/wavlm-base-plus
asr_logmel_encoder       openai/whisper-base
audio_spectrogram_vit    ta012/SSLAM_pretrain
vision_spectrogram_vit   facebook/dinov3-vits16-pretrain-lvd1689m
```

## Workflow

Feature extraction is the only stage that runs a large frozen backbone:

```text
python -m training.extract_features --method <method_name>
```

This writes compact fp16 features under:

```text
training/<method_name>/feature_cache/{train,val,test}/
```

Approximate compact-cache sizes for the 9,084-clip dataset are:

```text
waveform_ssl             about 360 MB    (13 layers x 1536 per clip)
asr_logmel_encoder       about 130 MB    (7 layers x 1024 per clip)
audio_spectrogram_vit    about 14 MB     (768 per clip)
vision_spectrogram_vit   about 21 MB     (1152 per clip)
```

Feature caches are local working data and are not tracked, with one exception:
the `asr_logmel_encoder` cache is committed by project decision, so its
detector-head experiments are reproducible directly from the repository.

Every runner accepts `--device {auto, cuda, mps, cpu}`. `auto` selects CUDA
when available and otherwise CPU. MPS is honored only on explicit request: a
verification run measured large numerical drift between MPS and CPU encoder
features (35 percent relative on Whisper-base), so MPS output is not treated
as equivalent and never enters a cache silently. Measured on an Apple M-series
CPU, the full Whisper-base extraction over all 9,084 clips takes about 11
minutes; head training afterward takes about 2 minutes.

Head training is local and uses only cached tensors:

```text
python -m training.train_detector --method <method_name>
```

The default training configuration is AdamW, learning rate `3e-4`, weight decay
`1e-4`, binary cross-entropy with logits, gradient clipping at norm `5.0`, batch
size `8`, maximum `20` epochs, and early stopping after `5` validation epochs
without lower EER. These values are encoded in `TrainConfig`.

At every epoch, the runner prints a progress line to stderr containing method,
epoch, training loss, validation loss, validation EER, and validation accuracy
at threshold 0.5. At the end it prints the final metrics JSON to stdout and
writes the same selected-run record to `final_models/<method>/metrics.json`.

`metrics.json` reports, under `selected_head`, the loss, EER, and accuracy of
the validation-selected head on each of the train, validation, and test
splits, alongside `best_epoch`, `best_val_eer`, and the full per-epoch
history. The test split is evaluated only after the validation-selected head
has been restored. Test metrics are not used for head selection.

A saved detector head is evaluated through:

```text
python -m training.evaluate_detector --method <method_name> --split test
```

The evaluation runner loads `final_models/<method>/detector_head.pt`, reads
cached features for the requested split, prints metrics JSON to stdout, and writes
`final_models/<method>/evaluation_<split>.json`.

The Python environment must already provide PyTorch, TorchAudio, Transformers,
Pandas, PyArrow, NumPy, Pillow, and SoundFile.

## Outputs

Each successful run writes exactly three selected artifacts into its method
folder under `final_models/`:

```text
detector_head.pt
config.json
metrics.json
```

The saved head is the validation-selected head. `metrics.json` records the best
validation EER, test EER, accuracy at threshold 0.5, and the training history.
Saved-head evaluation additionally writes `evaluation_train.json`,
`evaluation_val.json`, or `evaluation_test.json`, depending on the requested
split.

## Failure Policy

Model access failures are terminal during feature extraction. This matters for
gated DINOv3 access and for SSLAM remote-code loading. The runner prints
`MODEL ACCESS ERROR` and exits with code 2. It does not continue with another
checkpoint.

Feature extraction is the dataset-facing stage. It reads `manifest.parquet`,
loads the referenced WAV files, and verifies the frozen audio contract before
writing cached features.

Head training and saved-head evaluation are cache-facing stages. They do not
read raw audio. They require the cached features produced by feature extraction.
