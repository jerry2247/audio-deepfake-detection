# voxpopuli: Meta VoxPopuli (real)

| | |
|---|---|
| Label | real |
| Staged | **700 clips / 1.88 h**: 480 `en` plus 220 `en_accented` (see STATS.json) |
| Language | English (incl. 16 L2 accents) |
| Source | `facebook/voxpopuli` (Hugging Face, ungated), `test` shards |
| License | CC0 |
| Vintage | recordings 2009-2020 |

## What it is
European Parliament plenary recordings; formal oratory, real hall acoustics,
interpreter-grade microphones. **Recorded 2009–2020, i.e. before modern TTS
existed: zero AI-contamination risk by construction.** The `en_accented` subset
adds non-native English accents (16 L2 accents), broadening speaker variety
without leaving English.

## Exactly what prep.py does
1. Streams `en/test-*.parquet` (target 480) and `en_accented/test-*.parquet`
   (target 220) with deterministic shuffle (seed 20260604).
2. Decodes raw audio bytes to mono PCM16 WAV at native rate.
3. `speaker_id = vp_<speaker_id>`; accent recorded in `notes`.

## Run
```bash
.venv/bin/python prep.py all
```
