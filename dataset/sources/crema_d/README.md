# crema_d: CREMA-D (real, acted emotion)

| | |
|---|---|
| Label | real |
| Staged | **631 clips / 0.45 h** (see STATS.json) |
| Language | English |
| Source | `confit/cremad` (Hugging Face mirror of the original, ungated) |
| License | ODbL |
| Vintage | recorded 2014, before modern TTS; zero contamination risk |

## What it is
91 actors (diverse in age/ethnicity) delivering 12 fixed sentences across 6 acted
emotions (anger, disgust, fear, happy, neutral, sad) at multiple intensities;
shouted and distressed real speech that read-speech corpora completely lack.

## Selection
Balanced round-robin over (emotion × actor) cells with deterministic shuffling
(seed 20260604), target 650 to 631 staged (19 clips under the 1.8 s staging
minimum skipped; CREMA-D clips average ~2.5 s, which is why the project's final
minimum clip duration is 2.0 s).

## Leakage note
Only 12 distinct sentences exist, so `content_id` groups span many actors. The
split graph handles this: each sentence's clips form components that stay within
one split (transcripts recorded from the official sentence table).

## Run
```bash
.venv/bin/python prep.py all
```
