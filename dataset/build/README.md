# build/: central pipeline from staged sources to the final dataset

Four gates, run in order from `dataset/`. Gates 1 to 3 write only to
`build/work/` (git-ignored, regenerated). Gate 4 is the only writer of
`dataset/audio/`, `manifest.parquet` and `dataset_card.json`, and refuses to
run without explicit authorization.

```bash
.venv/bin/python build/validate.py     # gate 1: staging contract checks
.venv/bin/python build/condition.py    # gate 2: uniform DSP
.venv/bin/python build/split.py        # gate 3: split assignment + balancing

# gate 4, only when finalization is authorized:
DATASET_FINALIZE_AUTHORIZED=YES .venv/bin/python build/finalize.py \
    --i-am-authorized-to-write-audio --dataset-version v1.0
```

## Gate 1: validation

Checks every staged source against the staging contract: schema completeness,
label and metadata consistency, duration bounds, and randomized ffprobe spot
checks (PCM16, mono). Output: `work/validation_report.md`.

## Gate 2: conditioning (cond_v1)

Applied identically to every clip of both classes, which is the central
anti-leakage measure: resample to 16 kHz mono, trim edge silence below -45
dBFS keeping 150 ms, apply linear gain to -20 LUFS integrated with the true
peak capped at -1.5 dBFS (no compression), run a shared MP3 64 kbps round
trip on both classes, segment to at most 30 s dropping pieces under 2 s, and
store WAV PCM16. Per segment it measures duration, loudness, peak, edge
silences, speech fraction, spectral bandwidth, and the SHA-256 of the file.
Output: `work/conditioned/` and `work/conditioned.csv`.

## Gate 3: split and balancing (seed 301)

Builds an identity graph over conditioned segments using union-find. Edges:
shared speaker identity across both `speaker_id` and
`cloned_source_speaker_id` (a cloned voice stays with its source speaker),
shared `source_recording_id`, and shared `content_id` where the content
group contains both labels (preventing real-to-fake transcript leakage
without chain-merging unrelated same-label clips that read the same stock
sentences). Whole connected components are assigned to train, validation and
test at 70/10/20; components of at least 100 segments are packed greedily by
remaining need, the rest by seeded weighted-random choice. Each split is then
balanced to exactly 1:1 real to fake by downsampling the surplus class,
stratified proportionally by generator family for fakes and by source for
reals, so no generator or corpus is eliminated. All randomness in this gate
uses seed 301. Output: `work/assignment.csv` and `work/split_report.md`.

## Gate 4: finalize

Guards: the environment variable `DATASET_FINALIZE_AUTHORIZED=YES`, the
explicit flag, and an `audio/` tree containing nothing but placeholder files
(a built dataset is never overwritten; a new build requires a new version and
an intentional clear). It mints stable clip identifiers, derives
`content_id` and `matched_pair_id`, copies the selected conditioned segments
to `audio/<split>/<label>/<clip_id>.wav`, and writes `manifest.parquet`
(exact 43-column frozen schema per `DATASHEET.md`) and `dataset_card.json`
(version, conditioning parameters, per-split counts, manifest SHA-256).
