# emilia: Emilia-YODAS EN (real, conversational)

| | |
|---|---|
| Label | real |
| Staged | **350 clips / 1.0 h** (81 low-DNSMOS + 47 out-of-window candidates rejected; see STATS.json) |
| Language | English |
| Source | `amphion/Emilia-Dataset` (Hugging Face), `Emilia-YODAS/EN` tars |
| License | CC-BY 4.0 (the Emilia-YODAS portion) |
| Access | granted (gate accepted 2026-06-04) |

## What it is
In-the-wild spontaneous English from YouTube; podcasts, talk shows, interviews:
natural turn-taking, fillers, room tone. The conversational register the other
real sources don't cover.

## Contamination control (web-crawled 2023-2024, AI narration possible)
1. Keep only clips with DNSMOS at least 2.8 and duration 4-30 s.
2. `source_recording_id = emilia:<video id>`; a suspect video excises wholesale.
3. Volume capped at 350 (vs. thousands from provably-human corpora).

## Exactly what prep.py does
Downloads 2 deterministic tars (seed 20260604) from `Emilia-YODAS/EN`, pairs each
mp3 with its JSON metadata (text, speaker, dnsmos), applies the filters above,
stages 350.

## Run (after access is granted)
```bash
.venv/bin/python prep.py all
```
