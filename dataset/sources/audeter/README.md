# audeter: AUDETER modern-TTS slice (fake)

| | |
|---|---|
| Label | fake |
| Staged | see STATS.json (target about 800: at most 40 per domain by model cell) |
| Language | English |
| Source | `wqz995/AUDETER` (Hugging Face, ungated; Sep 2025, updated 2026) |
| License | CC-BY-NC-ND 4.0 (fine: private educational use, audio never redistributed) |
| Vintage | generators late-2024 to 2025 |

## What it is
The same modern TTS systems rendered across four different acoustic domains;
audiobook, celebrity, crowdsource, us_congress; so the fake class doesn't equate
"synthetic" with one recording condition.

## Exact selection
- TTS systems: `cosyvoice`, `f5_tts`, `sparktts`, `fish_speech`, `zonos`
  (modern vintage only).
- All four domains, at most 40 clips per domain by model cell, deterministic sampling.
- **Excluded**: legacy TTS (bark, chattts, vits, openvoice, etc.) and the entire
  `vocoders/` tree (2020-23 vocoder resynthesis; fails the modern-vintage rule).

## Run
```bash
.venv/bin/python prep.py all
```
