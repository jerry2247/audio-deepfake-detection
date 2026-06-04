# yt_real: in-the-wild real speech from YouTube

| | |
|---|---|
| Label | real |
| Staged | 817 clips, 2.14 hours, from 2 source recordings |
| Language | English |
| Domains | conversational (Comp1), podcast (Comp2) |
| License bucket | research_only (private educational use, never redistributed) |

## Source material

Two long YouTube recordings were supplied to the project as MP3 files and stored
as `raw/Comp1.mp3` and `raw/Comp2.mp3` (raw audio is local working data and is
not tracked in git). Per-recording provenance, including the video URL, the
channel, the assigned domain, and confirmation that each video carries its
original human audio track rather than an automatic dub, is recorded in
`videos.csv`, which is tracked.

Comp1 is 62.5 minutes of scripted television dialogue from an official network
channel: many speakers, professional production, location sound. Comp2 is 80.0
minutes of a comedy podcast compilation with three recurring hosts, loud
podcast mastering, expressive and overlapping speech, and occasional music
transitions between excerpts. Both predate modern speech synthesis and were
verified to play their original English audio tracks.

## Processing

`prep.py` stages candidate clips from the raw recordings as follows.

1. Each recording is decoded once to a 16 kHz mono analysis track. A 30 ms
   frame energy detector with a loudness-adaptive threshold (the -45 dBFS at
   -20 LUFS convention of `build/condition.py`, transposed to the recording's
   measured integrated loudness) marks speech and silence. The first 60
   seconds and the last 30 seconds of every recording are excluded outright,
   because branded intro and outro music lives there; no clip can extend into
   these zones.
2. Pauses are silence runs of at least 0.30 s. Cut points sit at pause
   midpoints, and continuous speech longer than 12 s is split at its quietest
   frame. Clips span 4.0 to 12.0 s. The 12 s ceiling is deliberate: the
   conditioned corpus averages roughly 6.5 s per class, and staging longer
   real clips would have made clip duration itself a usable class shortcut.
3. Candidates must contain at least 55 percent speech frames and clearly
   voiced energy. Two music filters then apply, calibrated against the
   all-speech recording as ground truth: a candidate is rejected if its
   envelope autocorrelation in the 0.3 to 1.2 s lag range exceeds 0.55
   (rhythmic music) or if more than 10 percent of its frames sit inside a
   sustained spectral peak run of 0.4 s or longer (held musical notes).
   Rejection counts per recording are recorded in `STATS.json`. Laughter and
   speech over quiet background sound are retained deliberately; they are
   real human audio in real scenes.
4. Selected clips are cut sample-exactly from a single native-rate decode of
   each recording (never by seeking into the MP3, which drifts on
   variable-bitrate files) and written as mono PCM16 WAV at the native rate.
   Selection spreads uniformly across each recording's timeline.

Every clip carries `source_recording_id` of the form `yt_real:CompN`, so each
recording forms a single leakage group and is assigned to exactly one split.
Clips from this source carry the dataset's in-the-wild flag.

## Yield

| recording | minutes | candidates | rejected by music filters | staged |
|---|---|---|---|---|
| Comp1 | 62.5 | 344 | 2 | 342 |
| Comp2 | 80.0 | 486 | 11 | 475 |

## Reproduction

```bash
.venv/bin/python prep.py template   # create videos.csv rows for files in raw/
.venv/bin/python prep.py all        # stage; re-runs are byte-identical
```

## Limitations

The speech detector is energy based and does not separate speech over loud
background music from clean speech; the beat and tone filters remove clips
dominated by music, and the residual risk is clips with faint background
sound, which is accepted as in-the-wild character. Speaker identities inside
each recording are not annotated, so `speaker_id` is unknown and split safety
relies on recording-level grouping. All 817 clips come from two recordings,
a concentration that is intentional and documented: these are the only two
recordings approved for this source.
