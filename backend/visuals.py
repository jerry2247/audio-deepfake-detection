from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image

from training.audio_spectrogram_vit import sslam_vit_base_detector as sslam
from training.vision_spectrogram_vit import dinov3_vit_s16_detector as dinov3

from .audio import DecodedAudio
from .schemas import DINOv3Visualization, SSLAMVisualization, VisualizationBlock, WaveformVisualization


WAVEFORM_POINTS = 700


def build_visualizations(audio: DecodedAudio) -> VisualizationBlock:
    waveform = audio.waveform.detach().cpu().float()
    return VisualizationBlock(
        waveform=WaveformVisualization(
            points=downsample_waveform(waveform, WAVEFORM_POINTS),
            sample_rate=audio.sample_rate,
        ),
        sslam_fbank=sslam_fbank_visualization(waveform),
        dinov3_spectrogram=dinov3_spectrogram_visualization(waveform),
    )


def downsample_waveform(waveform: torch.Tensor, points: int) -> list[float]:
    if waveform.numel() <= points:
        return [round(float(value), 5) for value in waveform.tolist()]
    pooled = F.adaptive_avg_pool1d(waveform.view(1, 1, -1), points).view(-1)
    return [round(float(value), 5) for value in pooled.tolist()]


def sslam_fbank_visualization(waveform: torch.Tensor) -> SSLAMVisualization:
    mel = _sslam_fbank(waveform)
    image = _matrix_to_image(mel)
    return SSLAMVisualization(
        image_png_base64=_png_base64(image),
        width=image.width,
        height=image.height,
        mel_bins=sslam.MEL_BINS,
        frames=sslam.TARGET_FRAMES,
    )


def dinov3_spectrogram_visualization(waveform: torch.Tensor) -> DINOv3Visualization:
    image = _dinov3_spectrogram_image(waveform).resize(
        (dinov3.IMAGE_SIZE, dinov3.IMAGE_SIZE),
        resample=Image.Resampling.BICUBIC,
    )
    patch_grid = dinov3.IMAGE_SIZE // 16
    return DINOv3Visualization(
        image_png_base64=_png_base64(image),
        width=image.width,
        height=image.height,
        image_size=dinov3.IMAGE_SIZE,
        patch_grid=(patch_grid, patch_grid),
    )


def _sslam_fbank(waveform: torch.Tensor) -> torch.Tensor:
    centered = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        centered.unsqueeze(0),
        htk_compat=True,
        sample_frequency=sslam.SAMPLE_RATE,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=sslam.MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )
    frames = mel.shape[0]
    if frames < sslam.TARGET_FRAMES:
        mel = F.pad(mel, (0, 0, 0, sslam.TARGET_FRAMES - frames))
    else:
        mel = mel[: sslam.TARGET_FRAMES, :]
    return (mel - sslam.NORM_MEAN) / (sslam.NORM_STD * 2.0)


def _dinov3_spectrogram_image(waveform: torch.Tensor) -> Image.Image:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=dinov3.SAMPLE_RATE,
        n_fft=dinov3.N_FFT,
        win_length=dinov3.WIN_LENGTH,
        hop_length=dinov3.HOP_LENGTH,
        n_mels=dinov3.MEL_BINS,
        power=2.0,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=dinov3.TOP_DB)
    mel = mel_transform(waveform.unsqueeze(0))
    db = to_db(mel).squeeze(0)
    return _matrix_to_image(db)


def _matrix_to_image(matrix: torch.Tensor) -> Image.Image:
    data = matrix.detach().cpu().float()
    data = data.T.flip(0)
    minimum = data.amin()
    maximum = data.amax()
    scaled = (data - minimum) / (maximum - minimum).clamp_min(1e-6)
    array = (scaled * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    rgb = np.stack([array, array, array], axis=-1)
    return Image.fromarray(rgb)


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
