# Backend

This folder owns the inference API for the project demo.

The backend belongs in this repository because it depends on the trained model
artifacts in `final_models/` and the preprocessing definitions in `training/`.
The personal website should only call this API. It should not contain model
weights, detector code, training code, or dataset code.

## Role

The backend will expose one audio-upload endpoint that returns side-by-side
results for the four detector methods:

- `waveform_ssl`
- `asr_logmel_encoder`
- `audio_spectrogram_vit`
- `vision_spectrogram_vit`

Each method response should include the model score, the predicted label, and
the method status. The spectrogram-based methods should also return the
visualization data needed by the demo page.

## Boundaries

- Reads trained artifacts from `final_models/`.
- Reuses inference and preprocessing code from `training/`.
- Does not train models.
- Does not modify `dataset/`.
- Does not write new model artifacts.
- Does not belong in the personal website repository.

## Deployment

The API is deployed as a Hugging Face Docker Space on the free CPU tier:

```text
https://jerry2247-ai-audio-lab.hf.space
```

A Docker Space was chosen over Inference Endpoints because Endpoints are
billed infrastructure while Spaces provide a free public container with
sufficient CPU for this demo (all four methods answer in about one to two
seconds each). The Space image bundles the backend code, the reusable
`training/` inference modules, and the four trained heads from
`final_models/` (about 5 MB). It contains no dataset, no feature caches, and
no raw audio. Backbones are downloaded by model ID at container startup; the
gated DINOv3 backbone uses the `HF_TOKEN` space secret from an account that
has accepted the model license. Free Spaces sleep after sustained inactivity
and wake on the next request.

Dependency versions in `requirements.txt` are pinned to the combination
validated end to end, including `transformers` 4.x (the SSLAM remote code
does not load under transformers 5) and a CPU `torchvision` build matched to
the CPU `torch` build (a mismatched pair fails at import with a
`torchvision::nms` operator error and takes every method down with it).

## API Shape

The public demo should call one backend API, not four separate APIs.

The backend should expose one primary endpoint:

```text
POST /analyze
```

The request is a single uploaded audio file. The response contains one shared
audio summary, one shared visualization block, and four method result blocks.
Each method result is independent, so an unavailable model can be reported
without breaking the full response.

Recommended response structure:

```json
{
  "request_id": "string",
  "audio": {
    "duration_s": 0.0,
    "sample_rate": 16000,
    "channels": 1
  },
  "visualizations": {
    "waveform": {
      "points": []
    },
    "sslam_fbank": {
      "image_png_base64": "string",
      "mel_bins": 128,
      "frames": 1024
    },
    "dinov3_spectrogram": {
      "image_png_base64": "string",
      "image_size": 224,
      "patch_grid": [14, 14]
    }
  },
  "methods": {
    "waveform_ssl": {
      "status": "ready",
      "probability_fake": 0.0,
      "prediction": "real",
      "elapsed_ms": 0.0
    },
    "asr_logmel_encoder": {
      "status": "ready",
      "probability_fake": 0.0,
      "prediction": "real",
      "elapsed_ms": 0.0
    },
    "audio_spectrogram_vit": {
      "status": "ready",
      "probability_fake": 0.0,
      "prediction": "real",
      "elapsed_ms": 0.0
    },
    "vision_spectrogram_vit": {
      "status": "ready",
      "probability_fake": 0.0,
      "prediction": "real",
      "elapsed_ms": 0.0
    }
  }
}
```

The backend should also expose lightweight support endpoints:

```text
GET /health
GET /methods
```

`/health` reports whether the service is alive. `/methods` reports which of
the four trained heads are present in `final_models/` and which gated
backbones are currently loadable.

## Model Loading

The Hugging Face endpoint container should load all available methods once at
startup. A method is available only when its final artifacts exist:

```text
final_models/<method>/detector_head.pt
final_models/<method>/config.json
final_models/<method>/metrics.json
```

If a trained head is missing, the endpoint should return that method with
`status: "not_trained"`. If a required Hugging Face backbone cannot be loaded,
the endpoint should return that method with `status: "model_access_error"` and
should not substitute another checkpoint.

During local API tests, model loading can be disabled with
`AUDIO_LAB_LOAD_MODE=inspect`. In that mode, artifact-present methods report
`status: "not_loaded"` so the response is not confused with real inference.

The detector code should be reused from `training/`. The backend should not
copy the four feature extractors or detector-head definitions.

## Hugging Face Artifact Layout

The Hugging Face model repository used by the endpoint should contain only the
small serving artifacts:

```text
final_models/
  waveform_ssl/
  asr_logmel_encoder/
  audio_spectrogram_vit/
  vision_spectrogram_vit/
```

The repository should not contain:

```text
dataset/
training/*/feature_cache/
raw audio
staged audio
conditioned audio
```

The container holds the backend code, the reusable `training/` inference
modules, and the heads at `/app/final_models/<method>/`, which is where
`AUDIO_LAB_MODEL_ROOT` points in the Space image.

Backbone weights are loaded by their Hugging Face model IDs at container
startup. Gated models, including DINOv3, require the `HF_TOKEN` space secret
with a token from an account that has accepted the model license.

## Visualization Contract

The visualizations must be generated from the same preprocessing tensors used
for inference.

For `audio_spectrogram_vit`, the visualization is the SSLAM input fbank:

```text
16 kHz waveform
128 mel bins
10 ms frame shift
1024 frames after pad or truncate
normalization from the SSLAM detector implementation
```

For `vision_spectrogram_vit`, the visualization is the DINOv3 input image:

```text
128-bin log-mel spectrogram
80 dB dynamic range cap
3-channel image
224x224 processed image
14x14 patch grid
```

The frontend can render these as clean research figures. The backend should
return compact PNG images for the spectrogram views and a downsampled waveform
trace for the raw-audio view.

## Deployment Configuration

Deployed configuration:

```text
Host: Hugging Face Spaces (Docker), free CPU tier
Space: jerry2247/ai-audio-lab
API base: https://jerry2247-ai-audio-lab.hf.space
Runtime: Python FastAPI app served by uvicorn on port 7860
Artifacts: final_models heads baked into the Space image
Secret: HF_TOKEN for the gated DINOv3 backbone
Public website: calls the Space URL directly from the browser
```

The personal website stores only the endpoint URL in its build environment.
It does not store model weights or duplicate inference code.
