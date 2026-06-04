# echofake: EchoFake modern-generator slice (fake)

| | |
|---|---|
| Label | fake |
| Staged | **1,300 clips / 1.86 h**: CosyVoice2 347, FireRedTTS 331, IndexTTS 325, MaskGCT 297 (see STATS.json) |
| Language | English |
| Source | `EchoFake/EchoFake` (Hugging Face, ungated; Oct 2025, MIT) |
| License | MIT |
| Vintage | generators late-2024 to 2025 |

## What it is
Zero-shot TTS fakes generated over Common Voice EN source speakers/texts by the
EchoFake authors (2025). The modern-system filter accepts IndexTTS, MaskGCT,
CosyVoice2, OpenAudio-S1, LLaSA, F5-TTS, FireRedTTS; in the deterministic stream
sample, four landed (CosyVoice2 / FireRedTTS / IndexTTS / MaskGCT, about 325 each).
LLaSA / F5-TTS / OpenAudio coverage is supplied by mlaad_v9 instead, so nothing
is lost at the dataset level.

## What we deliberately exclude
- `bonafide` rows (they are Common Voice clips; we source CV directly)
- **replayed** clips (50% of EchoFake is re-recorded through physical speakers;
  that's a channel artifact, not synthesis; out of scope)
- legacy generators (XTTSv2, SpeechT5, StyleTTS2, OpenVoice-V2)

## Leakage linkage (important)
EchoFake fakes clone CV speakers and texts. We record
`cloned_source_speaker_id = cv_<hash>` and `source_recording_id = cv:<stem>` in
the same namespace as `sources/common_voice`, so `build/split.py` keeps a real CV
clip and fakes derived from it in one split group; voice-clone leakage is
structurally impossible across splits.

## Run
```bash
.venv/bin/python prep.py all --target 1300
```
