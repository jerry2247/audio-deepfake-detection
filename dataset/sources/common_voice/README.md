# common_voice: Mozilla Common Voice v22.0 (real)

| | |
|---|---|
| Label | real |
| Staged | **1,100 clips / 1.81 h** (see STATS.json) |
| Language | English |
| Source | `fsicoli/common_voice_22_0` (Hugging Face, ungated) |
| Split used | `test` (validated: every clip has at least 2 positive human reviews) |
| License | CC0 |
| Vintage | corpus v22.0 (2025); human speech, vintage uncritical |

## What it is
Crowdsourced read speech: thousands of volunteer speakers, phones/laptops/headsets,
home acoustics, broad accent coverage. Human-reviewed before acceptance; the
trusted-by-construction clean-speech backbone of the real class.

## Why this mirror / version
The newest **ungated** mirror is v22.0; Mozilla's official v25 requires a Mozilla
Data Collective account for zero material benefit to this project (real-speech
character is unchanged across versions). Decision: v22.0 mirror.

## Exactly what prep.py does
1. Downloads `transcript/en/test.tsv` + all `audio/en/test/*.tar` shards (~687 MB).
2. Deterministic (seed 20260604) shuffle of clip names; takes 1,100 clips.
3. Decodes each MP3 to mono PCM16 WAV at native rate into `staged/clips/`.
4. Writes `staged/staged.csv`: `speaker_id = cv_<client_id>`,
   `source_recording_id = cv:<clip stem>`; the SAME namespace `sources/echofake`
   uses for its Common Voice source material, so a CV clip and any EchoFake spoof
   derived from it are forced into one split group by `build/split.py`.

## Run
```bash
.venv/bin/python prep.py all        # download + stage
```
