# Dataset Sources: Composition Record

This document records what the dataset contains and why each source was chosen
or rejected. Per-source acquisition and processing details live in
`sources/<name>/README.md`; the frozen format contract and split design live in
`DATASHEET.md`. Composition was settled on 2026-06-04.

## Shape

| | real | fake | total |
|---|---|---|---|
| final (v1.0) | 4,542 | 4,542 | 9,084 clips, about 17 hours |

Clips are mono 16 kHz WAV PCM16, 2.0 to 30.0 seconds, English. Every source is
pooled; train, validation and test are divided 70/10/20 at split-group level
(speaker-, recording- and content-disjoint) with seed 301, and each split is
balanced to exactly 1:1 real to fake by stratified downsampling of the surplus
class. There is no held-out generator family and no test-only source. Every
fake generator is of 2025 or 2026 vintage with one flagged exception
(el_garystafford, December 2024), recorded per row in the manifest. Staging
deliberately overshoots; the final build performs the deterministic selection.

## Sources

| source | label | staged clips | hours | content |
|---|---|---|---|---|
| common_voice | real | 1,100 | 1.81 | crowdsourced read speech, about 1,085 speakers |
| peoples_speech | real | 755 | 1.58 | meetings, radio, interviews; 31 recordings |
| voxpopuli | real | 700 | 1.88 | European Parliament, including accented English |
| crema_d | real | 631 | 0.45 | acted emotional speech, 91 actors, 6 emotions |
| expresso | real | 392 | 0.40 | studio expressive styles, whisper and laughter |
| emilia | real | 350 | 1.00 | podcasts, DNSMOS-filtered |
| yt_real | real | 817 | 2.14 | two long YouTube recordings, in the wild |
| echofake | fake | 1,300 | 1.86 | CosyVoice2, FireRedTTS, IndexTTS, MaskGCT |
| audeter | fake | 705 | 1.59 | five modern generators across four domains |
| mlaad_v9 | fake | 1,881 | 3.67 | 60 modern systems, up to 32 clips each |
| el_mendeley | fake | 600 | 1.52 | ElevenLabs and Respeecher, 2025 |
| el_garystafford | fake | 173 | 0.16 | ElevenLabs, December 2024, flagged |
| openai_tts | fake | 408 | 0.82 | gpt-4o-mini-tts, 13 voices, 8 styles |
| grok_tts | fake | 702 | 1.19 | Grok TTS, 5 voices, 9 styles with speech tags |

Staged totals: 10,514 clips, 20.1 hours (real 4,745, fake 5,769), validated
with zero row errors across all 14 sources. Conditioning produced 10,219
segments (real 4,542, fake 5,677) in 100 percent contract format. The split
and balancing stage keeps 9,084 of these. Files under `build/work/` are
regenerated artifacts of the latest gate run, not tracked documents.

## Fake generator coverage

About 70 distinct systems across all synthesis paradigms:

- 2026 releases: Dia2, VoxCPM2, Qwen3-TTS, MOSS-TTS, Voxtral, OmniVoice,
  Gemini-3.1-Flash-TTS, MiniMax-2.8, Fish-S2-Pro, NeuTTS-Nano, LFM2.5-Audio
  and others.
- 2025 releases: Kokoro, Zonos, CSM, Spark, Orpheus, Dia, Llasa, OpenAudio-S1,
  Higgs v2, VibeVoice, IndexTTS 1.5 and 2.0, FireRedTTS-2, Kyutai,
  Step-Audio-EditX, MiniCPM-o, Qwen2.5-Omni, MegaTTS3, VoxCPM, Maya1, SoulX,
  NeuTTS-Air, Kani, Kitten, Supertonic, Marvis, ZipVoice, MiraTTS, LuxTTS,
  GLM-TTS and others.
- Commercial systems: ElevenLabs (v3, Turbo-2.5, v2, and 2025 voice
  conversion), Respeecher, OpenAI TTS, Cartesia Sonic-3, DeepGram, Hume,
  Azure and Edge, MiniMax, Grok (via MLAAD, el_mendeley, el_garystafford,
  and the two generated sources).
- Watermark exclusion: Chatterbox and Resemble.ai outputs are excluded
  everywhere because the PerTh audio watermark could act as a hidden label
  shortcut.

## Candidates considered and rejected

| candidate | reason |
|---|---|
| Local or self-hosted TTS generation | forbidden by project decision; no local model generation under any circumstances |
| ElevenLabs API generation | forbidden by project decision; free published samples and datasets only |
| skypro1111/elevenlabs (Ukrainian) | single-language non-English dataset |
| Sh1man/elevenlabs | Russian-only voices, same rule |
| AhmedAshrafMarzouk/arabic-deepfake-audio | Arabic-only, same rule |
| CodecFake+ CoSG | its 17 systems are of 2023-24 vintage; MLAAD covers the codec paradigm with modern systems |
| DFBench_Speech25 | challenge set without public ground-truth labels |
| UniDataPro/real-vs-fake | undocumented generators, restrictive license, low value |
| mueller91/human-perception-2026 | perception-judgment table, not an audio corpus |
| SpeechFake, ASVspoof 5, SpoofCeleb, DFADD, In-the-Wild-2022 | pre-2025 generators |
| EchoFake replayed half | physical replay is a channel artifact, not synthesis |
| AUDETER vocoder tree and legacy TTS | 2020-23 vintage |
| MLS, Omnilingual-ASR, GigaSpeech2, YODAS, MSP-Podcast, RAVDESS | real side consolidated to fewer sources covering the same domain spread |
| Deepfake-Eval-2024 | removed by project decision 2026-06-04; its manual access gate was never approved, and in-the-wild coverage comes from yt_real |

## Build pipeline

```
sources/<name>/prep.py all        per-source download or cut, then stage
build/validate.py                 staging contract checks over all sources
build/condition.py                uniform DSP, identical for both classes
build/split.py                    leakage-safe 70/10/20 assignment and exact
                                  per-split class balancing (seed 301)
build/finalize.py                 guarded final build; writes audio/, the
                                  manifest and the dataset card only with
                                  explicit authorization
```

`build/README.md` documents each gate. Development validation covered the
staging contract, conditioning invariants, split leakage rules and finalize
guards, most recently with 18 of 18 checks passing on 2026-06-04. The
verification batteries for the `yt_real` source are described in
`sources/yt_real/README.md`.

## Acquisition status

All acquisition is closed. MLAAD and Emilia access gates were granted and both
sources staged. The two TTS batches ran against approved frozen request
matrices (hashes recorded in `TTS_PLAN.md`) for a combined spend of about two
dollars. The yt_real source was staged from the two recordings supplied to the
project. Nothing remains pending.
