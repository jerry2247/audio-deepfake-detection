# expresso: Meta Expresso (real, expressive)

| | |
|---|---|
| Label | real |
| Staged | **392 clips / 0.40 h**: all 4 speakers (115/115/115/47) (see STATS.json) |
| Language | English |
| Source | `ylacombe/expresso` (Hugging Face, ungated), `read` config |
| License | CC-BY-NC 4.0 (fine: private educational project) |
| Vintage | human speech, uncritical |

## What it is
Studio recordings of 4 professional voice actors in explicit expressive styles;
happy, sad, angry, whispering, laughing, confused, enunciated, etc. Covers the
"not just humans talking normally" requirement on the clean-studio end: whisper
and laughter are exactly the registers TTS systems now imitate, so the real class
must contain them too.

## Exactly what prep.py does
1. Streams `read/train-*.parquet` shards in seeded-shuffled order with a
   per-speaker quota (Expresso's shards are speaker-contiguous; without the
   quota only 2 of the 4 speakers are reached).
2. Keeps 392 clips in the staging duration window across all 4 speakers.
3. `speaker_id = expresso_<id>`; style recorded in `notes`.

## Run
```bash
.venv/bin/python prep.py all
```
