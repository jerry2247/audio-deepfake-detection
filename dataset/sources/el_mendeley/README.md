# el_mendeley: ElevenLabs and Respeecher synthetic voices (fake)

| | |
|---|---|
| Label | fake |
| Staged | **600 clips / 1.52 h**: 335 ElevenLabs plus 265 Respeecher (see STATS.json) |
| Language | English |
| Generator families | `elevenlabs` (282 V2V + 53 TTS), `respeecher` (V2V) |
| Source | Mendeley Data `79g59sp69z` v1 (direct public zip, 522 MB) |
| License | CC-BY 4.0 |
| Vintage | **2025 (H1)** |

## What it is
2025-vintage commercial fakes from two production systems: ElevenLabs (voice-to-
voice + TTS) and Respeecher (voice conversion), with per-file metadata
(tool, type, gender, age group) in `metadata.xlsx`.

## Attribution correctness
Classification comes **only** from `metadata.xlsx` (`Audio`/`Tool`/`Type` columns)
; never from filenames (all files are flat `audioN.wav`; an early path-based
heuristic misattributed everything to Respeecher and was replaced; staged counts
now match the published composition exactly: 335 EL / 265 Respeecher).

## Run
```bash
.venv/bin/python prep.py all
```
