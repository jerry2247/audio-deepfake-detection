# peoples_speech: MLCommons People's Speech (real)

| | |
|---|---|
| Label | real |
| Staged | **755 clips / 1.58 h from 31 distinct recordings, at most 25 clips each** (see STATS.json) |
| Language | English |
| Source | `MLCommons/peoples_speech` (Hugging Face, ungated), `clean/test` shards |
| License | CC-BY-SA 4.0 |
| Vintage | human speech, uncritical |

## What it is
Real-world English from government meetings, local radio, interviews, lectures:
far-field microphones, room noise, crosstalk, codecs. The messy-real-world
counterweight to studio/read corpora; exactly the conditions an in-the-wild
detector must not mistake for synthesis artifacts.

## Exactly what prep.py does
1. Streams `clean/test-*.parquet` shards in seeded-shuffled order (no bulk
   download; audio kept as raw bytes via `Audio(decode=False)`, decoded by ffmpeg).
2. Shuffle-buffer sampling (seed 20260604) with an **at most 25 clips per source
   recording cap**; without it, the recording-contiguous shard layout yields
   1,000 clips from just 12 recordings; with it, 755 clips from 31 recordings
   (the cap exhausts the scan budget before 1,000, and more clips from the same
   recordings would add volume, not diversity).
3. `source_recording_id` strips the segment suffix from the People's Speech row id
   so all segments of one source recording share a leakage group.

## Run
```bash
.venv/bin/python prep.py all
```
