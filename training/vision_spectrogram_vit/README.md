# DINOv3 ViT-S/16 Vision Spectrogram Detector

## Research Purpose

This approach treats a speech spectrogram as an image and uses a frozen
vision-pretrained DINOv3 encoder. It tests whether visual texture statistics in
spectrograms expose synthetic-speech artifacts using a compact image
representation.

## Frozen Backbone

Model: `facebook/dinov3-vits16-pretrain-lvd1689m`.

DINOv3 ViT-S/16 is a distilled vision Transformer from Simeoni et al., 2025,
arXiv 2508.10104. It was distilled from the DINOv3 ViT-7B teacher on the
LVD-1689M image set. The model uses patch size 16, embedding dimension 384, 6
attention heads, 4 register tokens, RoPE, an MLP feed-forward network, and 21.6M
parameters. The Hugging Face model is gated and requires accepting the model
license.

This is a vision model. It was not trained on audio. In this method, it is
applied to a spectrogram treated as an image.

The backbone is frozen. No DINOv3 weights, adapters, LoRA weights, or
fine-tuning parameters are trained.

## Feature Extraction

Input is a 128-bin log-mel spectrogram image computed from 16 kHz audio with
`n_fft=400`, `win_length=400`, `hop_length=160`, power spectrograms, and an
80 dB dynamic range cap. The spectrogram is min-max scaled to an 8-bit
3-channel RGB image and passed through the DINOv3 image processor at 224x224.
The patch size is 16, so 224x224 gives a 14x14 patch grid.

For a 224x224 processed image, the frozen model returns 1 CLS token, 4 register
tokens, and 196 patch tokens. The detector feature is the concatenation of
`pooler_output`, the mean of the 196 patch tokens, and the standard deviation
of the 196 patch tokens. This produces one 1152-dimensional feature vector.

The cache stores one 1152-dimensional DINOv3 spectrogram feature per clip.

## Detector Head

The detector head maps the 1152-dimensional DINOv3 spectrogram feature through
a LayerNorm, GELU, dropout MLP from 1152 to 256 to one logit. Only the head is
trained. The implementation is `dinov3_vit_s16_detector.py`.

## Training And Evaluation

Feature extraction writes:

```text
training/vision_spectrogram_vit/feature_cache/{train,val,test}/
```

Head training reads only the cached features, selects the best head by
validation EER, evaluates the selected head on the test split, and prints final
metrics JSON to stdout.

Saved-head evaluation reads the cached features and writes
`evaluation_<split>.json`.

## Final Output

Final selected artifacts for this method are written to:

```text
final_models/vision_spectrogram_vit/
```

The final folder is expected to contain the selected detector head, the
validation-selected configuration, training metrics, and saved-head evaluation
metrics.
