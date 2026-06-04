#!/usr/bin/env python3
"""mlaad_v9 — MLAAD (Multi-Language Audio Anti-Spoofing Dataset), current head
(v9-era, 2026). The single richest modern-generator source: its English tree holds
100+ systems, including 2026 releases and live commercial APIs sampled at MLAAD
build time (2025/26).

We take fake/en/<model>/ for a curated list of MODERN systems only (2025-2026
model releases + current commercial APIs; legacy academic TTS like tacotron/vits/
xtts/bark excluded). Chatterbox/Resemble are excluded because Resemble's PerTh
watermark could act as a hidden label shortcut. Per-model clip cap keeps any one
system from dominating. CC-BY-NC-4.0 (fine: private educational use).

Usage: .venv/bin/python prep.py all [--per-model N]   (default 50)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_to_staged_wav  # noqa: E402
from common.hf import list_files, download, sample_deterministic  # noqa: E402

SOURCE = "mlaad_v9"
REPO = "mueller91/MLAAD"
RAW = SRC_DIR / "raw"

# model-dir name -> (family, paradigm, vintage). Curated 2026-06-04 from the live
# fake/en tree. Names must match the repo EXACTLY (spaces and case included).
MODELS = {
    # ---- 2025/2026 open-weight releases ----
    "kokoro":                      ("kokoro", "vocoder", "2025-01"),
    "zonosTTS-v0.1":               ("zonos", "ssm_hybrid", "2025-02"),
    "sesame_csm":                  ("sesame_csm", "ar_codec_lm", "2025-03"),
    "Spark-TTS-0.5B":              ("spark_tts", "ar_codec_lm", "2025-03"),
    "orpheus-tts-0.1-finetune":    ("orpheus", "ar_codec_lm", "2025-03"),
    "Nari Dia-1.6B":               ("dia", "ar_codec_lm", "2025-04"),
    "Nari Dia2":                   ("dia", "ar_codec_lm", "2026"),
    "Llasa-1B":                    ("llasa", "ar_codec_lm", "2025-01"),
    "Llasa-3B":                    ("llasa", "ar_codec_lm", "2025-01"),
    "Openaudio-S1-Mini":           ("openaudio", "ar_codec_lm", "2025-06"),
    "Fish-S2-Pro":                 ("openaudio", "ar_codec_lm", "2026"),
    "Higgs-Audio-V2":              ("higgs_audio", "ar_codec_lm", "2025-07"),
    "Microsoft VibeVoice 1.5B":    ("vibevoice", "ar_codec_lm", "2025-08"),
    "Microsoft VibeVoice Large":   ("vibevoice", "ar_codec_lm", "2025-08"),
    "Index-TTS-1.5":               ("index_tts", "ar_codec_lm", "2025"),
    "Index-TTS-2.0":               ("index_tts", "ar_codec_lm", "2025-09"),
    "FireRedTTS-2.0":              ("fireredtts", "ar_codec_lm", "2025"),
    "Kyutai-TTS":                  ("kyutai", "ar_codec_lm", "2025"),
    "Step-Audio-EditX":            ("step_audio", "ar_codec_lm", "2025"),
    "MiniCPM-o-2.6":               ("minicpm", "ar_codec_lm", "2025-01"),
    "Qwen2.5-Omni":                ("qwen_tts", "ar_codec_lm", "2025-03"),
    "Qwen3-TTS-12Hz-0.6B-Base":    ("qwen_tts", "ar_codec_lm", "2026-01"),
    "Qwen3-TTS-12Hz-1.7B-Base":    ("qwen_tts", "ar_codec_lm", "2026-01"),
    "MegaTTS3":                    ("megatts", "style_diffusion_gan", "2025-03"),
    "VoxCPM-0.5B":                 ("voxcpm", "style_diffusion_gan", "2025-09"),
    "VoxCPM2":                     ("voxcpm", "style_diffusion_gan", "2026"),
    "Maya1 TTS":                   ("maya", "ar_codec_lm", "2025-11"),
    "SoulX-Podcast":               ("soulx", "ar_codec_lm", "2025-10"),
    "NeuTTS-Air":                  ("neutts", "ar_codec_lm", "2025-10"),
    "NeuTTS-Nano":                 ("neutts", "ar_codec_lm", "2026"),
    "Kani-TTS-370M":               ("kani", "ar_codec_lm", "2025"),
    "Kitten-TTS-Nano-0.2":         ("kitten", "vocoder", "2025"),
    "Supertonic":                  ("supertonic", "vocoder", "2025"),
    "OmniVoice":                   ("omnivoice", "masked_generative", "2026-03"),
    "MOSS-TTS-1.7B":               ("moss", "ar_codec_lm", "2026-02"),
    "MOSS-TTS-8B":                 ("moss", "ar_codec_lm", "2026-02"),
    "Voxtral":                     ("voxtral", "ar_codec_lm", "2026-03"),
    "Marvis-TTS":                  ("marvis", "ar_codec_lm", "2025"),
    "ZipVoice":                    ("zipvoice", "flow_matching", "2025"),
    "f5-tts":                      ("f5_tts", "flow_matching", "2024-10"),
    "e2-tts":                      ("e2_tts", "flow_matching", "2024"),
    "GLM-TTS":                     ("glm", "ar_codec_lm", "2025"),
    "LFM2.5-Audio":                ("lfm_audio", "ar_codec_lm", "2026"),
    "LongCat-AudioDiT":            ("longcat", "style_diffusion_gan", "2026"),
    "KugelAudio":                  ("kugelaudio", "unknown", "2026"),
    "VoXtream2":                   ("voxtream", "ar_codec_lm", "2026"),
    "MiraTTS":                     ("mira", "unknown", "2025"),
    "LuxTTS":                      ("lux", "unknown", "2025"),
    # ---- current commercial APIs, generated by MLAAD in 2025/26 ----
    "ElevenLabs-v3":               ("elevenlabs", "unknown", "2025-06"),
    "ElevenLabs-Turbo-v2.5":       ("elevenlabs", "unknown", "2024-11"),
    "ElevenLabs-v2-Multilingual":  ("elevenlabs", "unknown", "2023"),
    "OpenAI TTS-1 HD":             ("openai_tts", "unknown", "2024"),
    "Gemini-3.1-Flash-TTS":        ("gemini_tts", "unknown", "2026"),
    "Cartesia.ai (Sonic-3)":       ("cartesia", "ssm_hybrid", "2025-10"),
    "DeepGram":                    ("deepgram", "unknown", "2025"),
    "Hume TADA-3B-ML":             ("hume", "ar_codec_lm", "2025"),
    "MiniMax-Speech-2.8-Turbo":    ("minimax", "ar_codec_lm", "2026"),
    "minimax_speech-2.6-hd":       ("minimax", "ar_codec_lm", "2025"),
    "minimax_speech-02-turbo":     ("minimax", "ar_codec_lm", "2025"),
    "Edge-TTS":                    ("azure_neural", "unknown", "2025"),
    "Resemble.ai (April 12th, 2025)": None,   # EXCLUDED: PerTh watermark
    "Chatterbox":                  None,      # EXCLUDED: PerTh watermark
    "Chatterbox-Turbo":            None,      # EXCLUDED: PerTh watermark
}


def do_download(per_model: int):
    listing = {}
    for model, spec in MODELS.items():
        if spec is None:
            continue
        files = list_files(REPO, f"fake/en/{model}", suffixes=(".wav", ".mp3", ".flac"))
        if not files:
            print(f"[{SOURCE}] WARNING: no audio listed for {model!r} — verify name")
            continue
        keep = sample_deterministic(files, per_model, SEED)
        # meta.csv carries transcripts + architecture info
        meta = [f for f in list_files(REPO, f"fake/en/{model}") if f.endswith("meta.csv")]
        listing[model] = (keep, meta[:1])
    total = sum(len(k) for k, _ in listing.values())
    print(f"[{SOURCE}] downloading {total} clips across {len(listing)} models")
    done = 0
    for model, (keep, meta) in listing.items():
        for f in meta + keep:
            download(REPO, f, RAW)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{total + len(listing)}")
    print(f"[{SOURCE}] download done")


def _load_meta(model: str) -> dict[str, dict]:
    meta_path = RAW / f"fake/en/{model}/meta.csv"
    out: dict[str, dict] = {}
    if not meta_path.exists():
        return out
    with open(meta_path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.readline()
        delim = "|" if "|" in sample else ","
        f.seek(0)
        for r in csv.DictReader(f, delimiter=delim):
            key = Path(r.get("path", "") or r.get("file", "") or "").name
            if key:
                out[key] = r
    return out


def do_stage():
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    for model, spec in sorted(MODELS.items()):
        if spec is None:
            continue
        family, paradigm, vintage = spec
        model_dir = RAW / f"fake/en/{model}"
        if not model_dir.exists():
            continue
        meta = _load_meta(model)
        for src in sorted(model_dir.rglob("*")):
            if src.suffix.lower() not in (".wav", ".mp3", ".flac"):
                continue
            m = meta.get(src.name, {})
            dst = w.next_clip_path(i)
            try:
                info = decode_to_staged_wav(src, dst)
            except Exception:
                w.skip("decode_error")
                continue
            ok = w.add(StagedClip(
                staged_path=str(dst.relative_to(SRC_DIR)),
                source=SOURCE, label="fake", language="en", domain="read_speech",
                generator=model, generator_family=family,
                generator_version=str(m.get("model_name", "") or "")[:60],
                synthesis_paradigm=paradigm,
                generation_date="2025" if vintage < "2026" else "2026",
                vintage=vintage,
                utterance_id=f"{model}/{src.stem}"[:120],
                transcript=str(m.get("transcript", "") or m.get("text", "") or ""),
                source_uri=f"hf://datasets/{REPO}/fake/en/{model}",
                source_license="cc_by_nc_4.0",
                codec_history="wav" if src.suffix.lower() == ".wav" else "mp3_unknown",
                native_sample_rate_hz=info["sample_rate"],
                duration_s=round(info["duration_s"], 3),
            ))
            if ok:
                i += 1
    stats = w.finish({"models_staged": len([m for m, s in MODELS.items()
                                            if s and (RAW / f'fake/en/{m}').exists()])})
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) "
          f"models={stats['models_staged']} skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "stage", "all"])
    ap.add_argument("--per-model", type=int, default=32)
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        do_download(a.per_model)
    if a.cmd in ("stage", "all"):
        do_stage()
