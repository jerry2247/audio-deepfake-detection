# mlaad_v9: MLAAD modern-systems slice (fake)

| | |
|---|---|
| Label | fake |
| Staged | **1,881 clips** (3.67 h) across 60 modern systems, up to 32 clips each |
| Language | English (`fake/en/` tree) |
| Source | `mueller91/MLAAD` (Hugging Face), current head (v9-era, 2026) |
| License | CC-BY-NC 4.0 (fine: private educational project) |
| Access | granted (gate accepted 2026-06-04) |

## What it is
The single richest modern-generator source in existence: the English tree holds
100+ systems including **2026 releases** (Nari Dia2, VoxCPM2, Qwen3-TTS, MOSS-TTS,
Voxtral, OmniVoice, Gemini-3.1-Flash-TTS, MiniMax-2.8, Fish-S2-Pro, NeuTTS-Nano)
and **live commercial APIs sampled at MLAAD build time** (ElevenLabs v3/Turbo-2.5/
v2, OpenAI TTS-1-HD, Cartesia Sonic-3, DeepGram, Hume, Edge/Azure).

## Exact selection (the curated MODELS table in prep.py)
- **Included (about 58 systems, 32 clips each)**: every 2025/2026 open-weight release
  + every current commercial API folder. Each entry carries family, paradigm
  (per the DATASHEET enum), and release vintage.
- **Excluded**: legacy academic systems (tacotron/vits/ljspeech/xtts/bark/
  tortoise/parler/melo/speecht5 and similar), excluded by the pre-2025 vintage rule; and
  **Chatterbox + Resemble.ai folders**; Resemble's PerTh audio watermark could
  act as a hidden "fake" label shortcut for the detector.
- A per-model cap (32) keeps any single system from dominating; downloads are
  per-file (≈1,800 files), NOT the 159 GB repo.

## Run
```bash
.venv/bin/python prep.py all          # 1,881 clips across 60 modern systems
```
