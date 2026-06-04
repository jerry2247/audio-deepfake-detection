# Dataset

Real human speech versus AI-generated speech, as one frozen, self-contained dataset. This folder is the dataset scope: the frozen dataset (this contract, the manifest, the metadata, and the audio) and the build pipeline that produces it. The model side reads this folder and never writes to it; this folder never depends on the model side.

This document is the single source of truth for the dataset: what it contains, how it is built, the exact format, and how to use it. The format below is frozen and will not change; future versions may only append new nullable manifest columns at the end.

Status: the format, source composition, and build pipeline are frozen for v1.0.
The selected final dataset shape is 9,084 clips: 4,542 real and 4,542 fake,
balanced exactly 1:1 inside every split (train 3,184 per class, validation 452
per class, test 906 per class; a 70/10/20 division assigned at split-group
level with seed 301). Source material and build machinery remain under
`sources/` and `build/`; `SOURCES.md` is the composition record. The manifest,
the card, and the `audio/` tree are produced only by the authorized final build
(`build/finalize.py`) and may be absent before that step is run.

## Files in this folder

```
DATASHEET.md       this document (the frozen contract)
SOURCES.md         master composition record: every source, counts, status, rejections
manifest.parquet   source of truth, one row per clip (built during the build)
dataset_card.json  machine-readable build metadata: version, conditioning settings, counts, manifest hash
audio/
  train/{real,fake}/<clip_id>.wav
  val/{real,fake}/<clip_id>.wav
  test/{real,fake}/<clip_id>.wav
sources/           one folder per datasource: README.md + prep.py + STATS.json
                   (raw/ and staged/ are local working data, git-ignored)
build/             the four build gates (validate, condition, split, guarded finalize)
common/            shared staging/audio/HF helpers used by the preps
```

To use the dataset, read `manifest.parquet` and load the audio it points to. `dataset_card.json` carries the build metadata. Nothing else is required; `sources/`, `build/`, and `common/` are dataset-side machinery, not part of the consumer contract.

## Canonical audio format (frozen)

Every audio file is, without exception: WAV signed 16-bit PCM; mono; 16000 Hz; duration greater than 0 and at most 30.000 seconds; no zero-padding baked in; no clipped or full-scale samples. One type only, so there are no decoding surprises. No precomputed spectrograms or embeddings are shipped; features are derived from the waveform. The 16 kHz mono, 30 second ceiling is the shared substrate that every backbone in the project accepts.

## clip_id and paths (frozen)

- `clip_id` matches `^clip_[0-9a-f]{32}$`. It is a stable, opaque, metadata-derived identifier, not the audio content hash. The content hash is the separate `sha256` column.
- `path` is relative to this folder and is fully determined by split, label, and clip_id: `audio/<split>/<label>/<clip_id>.wav`.
- Filenames derive from `clip_id` only.

## Manifest schema (frozen)

`manifest.parquet` is authoritative (UTF-8, one row per clip). Column order below is normative. Types are exactly string, int64, double, bool. Dates are strings in `YYYY-MM-DD`. There are no nested or list columns. "req" means never null; "null" means nullable (use the sentinels below).

| # | Column | Type | Req | Notes |
|---|---|---|---|---|
| 1 | clip_id | string | req | unique, `^clip_[0-9a-f]{32}$` |
| 2 | path | string | req | `audio/<split>/<label>/<clip_id>.wav`, relative to this folder |
| 3 | label | string | req | {real, fake} |
| 4 | split | string | req | {train, val, test} |
| 5 | eval_condition | string | req | {train_pool, validation_pool, heldout_generator, in_the_wild, optional_probe} |
| 6 | split_group_id | string | req | atomic leakage key (see Splits) |
| 7 | is_heldout_generator | bool | req | true only for the held-out generator family |
| 8 | is_in_the_wild | bool | req | true only for In-the-Wild clips |
| 9 | source_dataset | string | req | origin corpus or generator collection |
| 10 | generator | string | req | "human" for real; else the model id |
| 11 | generator_family | string | req | grouping for generator-disjoint splits; "human" for real |
| 12 | generator_version | string | null | model or provider version if known |
| 13 | synthesis_paradigm | string | req | enum below; "n/a" for real |
| 14 | generation_date | string | null | YYYY-MM-DD for self-generated fakes if known |
| 15 | voice_id | string | null | TTS voice, cloned voice, or human voice label |
| 16 | speaker_id | string | null | human speaker identity where known |
| 17 | cloned_source_speaker_id | string | null | source speaker for voice conversion or cloning |
| 18 | source_recording_id | string | null | original long recording, session, book, or video id |
| 19 | utterance_id | string | null | pre-segmentation utterance id if available |
| 20 | source_uri_or_dataset_ref | string | null | provenance trace if legally safe |
| 21 | source_license | string | req | usage bucket (see controlled vocabularies below) |
| 22 | language | string | req | ISO 639-1 where possible (en, uk, de, ...), "und" if unknown |
| 23 | domain | string | req | enum below |
| 24 | transcript | string | null | verbatim text if available |
| 25 | content_id | string | null | stable hash of the normalized transcript |
| 26 | matched_pair_id | string | null | links a fake to the real clip with the same transcript; never crosses splits |
| 27 | duration_s | double | req | 0 < duration_s <= 30.0 |
| 28 | final_sample_rate | int64 | req | always 16000 |
| 29 | final_channels | int64 | req | always 1 |
| 30 | final_format | string | req | always wav_pcm16 |
| 31 | bit_depth | int64 | req | always 16 |
| 32 | file_size_bytes | int64 | req | size of the final file |
| 33 | sha256 | string | req | 64 hex chars, hash of the final audio file |
| 34 | native_sample_rate | int64 | null | sample rate before conditioning |
| 35 | codec_history | string | req | origin compression (see controlled vocabularies below) |
| 36 | loudness_lufs | double | req | integrated loudness after conditioning |
| 37 | peak_dbfs | double | req | peak level after conditioning, < 0.0 (conditioning targets a -1.5 dBFS true-peak ceiling) |
| 38 | leading_silence_ms | double | req | after conditioning |
| 39 | trailing_silence_ms | double | req | after conditioning |
| 40 | vad_speech_fraction | double | req | fraction of frames with speech, 0.0 to 1.0 |
| 41 | measured_bandwidth_hz | double | null | highest frequency carrying significant energy |
| 42 | bandwidth_flag | string | req | {full_band, band_limited, unknown} |
| 43 | conditioning_version | string | req | matches the value in dataset_card.json |

Controlled vocabularies:

- synthesis_paradigm: n/a, ar_codec_lm, flow_matching, masked_generative, style_diffusion_gan, vits_gan, ssm_hybrid, vocoder, codec_resynthesis, unknown
- domain: read_speech, audiobook, podcast, interview, parliament, conversational, celebrity, phone, studio, noisy_web, other
- codec_history: `<codec>` or `<codec>_<bitrate>k` or `<codec>_unknown` with codec in {mp3, aac, opus, vorbis, flac, wav}, plus `unknown` (e.g. mp3_32k, mp3_unknown, opus_160k, wav)
- source_license buckets: cc0, cc_by_4.0, cc_by_sa_4.0, cc_by_nc_4.0, cc_by_nc_nd_4.0, public_domain, apache_2.0, mit, odbl, research_only, grok_tos, openai_tos

Sentinels: a missing nullable string is `unknown`; a logically not-applicable value is `n/a`; a missing numeric is a real null; booleans are never null. Sentinels are never treated as identities.

## Splits and leakage guarantees

- `split` is exactly one of train, val, test, divided 70/10/20 with real:fake balanced exactly 1:1 inside every split. All sources are pooled and whole split groups are assigned randomly (seed 301). The surplus class inside each split is then downsampled to exact balance, stratified proportionally by generator family for fake clips and by source for real clips, so that no generator and no corpus is eliminated (project decision 2026-06-04: there is no held-out generator family and no test-only source).
- `split_group_id` is the connected components of a graph whose nodes are clips and whose edges are:
  1. shared speaker identity: any non-null, non-sentinel value appearing in `speaker_id` OR `cloned_source_speaker_id`, matched ACROSS the two columns, so a cloned voice always stays in the same split as its source speaker's real clips;
  2. shared `source_recording_id`;
  3. shared `content_id`, applied only when the content group contains BOTH labels (the transcript exists as real speech and as a fake rendition); this prevents real-to-fake transcript leakage (including all matched pairs) without chain-merging same-label clips that merely read the same stock sentences (real-real or fake-fake same-text pairs carry no label shortcut; linking them collapses fully-crossed corpora like CREMA-D into one mega-component for zero leakage benefit).
  Every clip in a component shares one `split_group_id`, and the whole component is in exactly one split. Sentinels never create an edge.
- Validation is for head selection only; never train or tune on test.
- `eval_condition` mapping: train to `train_pool`; val to `validation_pool`; test to `in_the_wild` for clips of in-the-wild provenance, else `optional_probe`. `heldout_generator` is retained in the enum for schema stability but is currently unused, and `is_heldout_generator` is false on every row.
- `is_in_the_wild` marks clips (real or fake) harvested from real-world circulating media rather than corpora or controlled generation.
- Evaluation protocol: the headline metric is EER on the pooled test split; report alongside it the EER restricted to `is_in_the_wild` test clips (vs all test reals) when in-the-wild clips are present in test. In version v1.0 the group-level assignment placed every in-the-wild clip (the yt_real source, two whole recordings) in train, so the test split contains no in-the-wild clips and the headline metric is the pooled test EER alone.

## Integrity and immutability

- Per-row `sha256` covers each final audio file.
- `dataset_card.json` records the `dataset_version`, the conditioning settings tied to `conditioning_version`, per-split and per-class counts, and a hash of `manifest.parquet`.
- The dataset is immutable once a `dataset_version` is tagged. Any change is a new version. Audio is git-ignored (a built artifact); this document, the manifest, and the card are tracked.
- Consume as read-only: do not modify files, do not re-split, do not train or tune on val or test.

## How the dataset is built

Composition follows one principle: generator diversity in the fake class matters more than raw clip count. The dataset is English-dominant by decision, and both classes span clean-studio through noisy real-world conditions so the label cannot be inferred from recording quality. The authoritative, continuously-updated composition record (per-source counts, status, rejection log) is `SOURCES.md`; per-source acquisition details are in `sources/<name>/README.md`.

Final shape (v1.0): 9,084 clips, exactly 4,542 real and 4,542 fake, about 17 hours.

Real (human): two long YouTube recordings forming the in-the-wild source (yt_real: scripted television dialogue and a comedy podcast compilation, 817 clips cut at natural pauses with music filtering; see `sources/yt_real/README.md`), Common Voice v22 (CC0, crowdsourced read speech), People's Speech (real-world meetings, radio, interviews), VoxPopuli (2009-2020 European Parliament, recorded before modern speech synthesis existed, including L2-accented English), CREMA-D (acted shouted and emotional speech), Expresso (studio expressive styles including whisper and laughter), and Emilia-YODAS EN (podcasts, DNSMOS-filtered).

Fake (synthetic): every fake generator is a 2025-2026 release or a current commercial system sampled in 2025-2026, drawn only from published datasets and two project-executed API generation batches; no local model generation, and no ElevenLabs API use (ElevenLabs audio comes exclusively from already-published free datasets). Sources: MLAAD current head filtered to 60 modern systems (2026 releases such as Dia2, VoxCPM2, Qwen3-TTS, MOSS, Voxtral, OmniVoice, Gemini-3.1-Flash-TTS, plus 2025 releases and live commercial APIs including ElevenLabs v3), EchoFake's modern non-replayed slice, AUDETER's modern TTS across four acoustic domains, the Mendeley ElevenLabs and Respeecher 2025 set, the garystafford ElevenLabs slice (2024-12, flagged per row in the manifest), and the two generated sources openai_tts (gpt-4o-mini-tts, 408 clips) and grok_tts (Grok TTS, 702 clips). Chatterbox and Resemble outputs are excluded everywhere: their PerTh audio watermark could act as a hidden label shortcut.

Conditioning is applied uniformly to every clip, real and fake (`build/condition.py`, parameters versioned as `conditioning_version` in `dataset_card.json`): decode, resample to 16 kHz mono, edge silence trim (-45 dBFS, 150 ms kept), LINEAR gain to -20 LUFS integrated capped at a -1.5 dBFS true peak (no compression), a shared MP3 64k codec round-trip on both classes, segmentation to at most 30 s with segments under 2.0 s dropped, store WAV PCM16. This neutralizes the documented leakage paths (silence duration, sample-rate and bandwidth nulls, codec mismatch, loudness) so the detector learns synthesis rather than a collection artifact.

Build order (gates, in `build/`): per-source staging (`sources/*/prep.py`), then `validate.py` (staging contract checks), then `condition.py` (uniform DSP, per-file measurements, `sha256`), then `split.py` (identity graph, `split_group_id`, random 70/10/20 group-level assignment and exact per-split class balancing, seed 301), then `finalize.py` (assign `clip_id`, derive `content_id` and `matched_pair_id`, write `audio/`, `manifest.parquet`, `dataset_card.json`, freeze). `finalize.py` is guarded and runs only with explicit authorization.
