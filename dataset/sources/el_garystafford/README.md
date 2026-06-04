# el_garystafford: ElevenLabs slice of garystafford/deepfake-audio-detection (fake)

| | |
|---|---|
| Label | fake |
| Staged | **173 clips / 0.16 h** (see STATS.json) |
| Language | English |
| Generator family | `elevenlabs` (model version undocumented by upstream) |
| Source | `garystafford/deepfake-audio-detection` (Hugging Face, ungated), `fake/el_*.flac` only |
| License | CC-BY 4.0 |
| Vintage | **2024-12** (slightly pre-2025; included deliberately for ElevenLabs family volume; flagged here, in staged.csv `vintage`, and in SOURCES.md) |

## What it is
ElevenLabs-generated English clips published Dec 2024, including `_c_` codec-
compressed variants of the same generations (recorded in `notes`; grouped by
generation via `source_recording_id` so variants never straddle splits).

## What we deliberately did NOT take
The repo's other fakes (Kokoro, Hume, Polly, Speechify, Luvvoice); Kokoro is
covered at scale by MLAAD; Polly/Speechify/Luvvoice are legacy-vintage; Hume's
clips predate Octave. Its real clips are unneeded (we have richer real sources).

## Run
```bash
.venv/bin/python prep.py all
```
