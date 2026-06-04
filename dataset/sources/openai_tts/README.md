# openai_tts: OpenAI TTS generated fake speech

| | |
|---|---|
| Status | complete; **408 clips staged** (batch plus smoke, 0 flagged; signal battery clean) |
| Label | fake |
| Achieved | 400-request approved matrix executed; 13 voices balanced (30-32 clips each), 8 styles, English |
| Generator | `gpt-4o-mini-tts` (verified newest speech-endpoint model 2026-06-04; runner aborts if a newer unknown TTS model appears) |
| Voices / modes | 13 built-in voices × 8 instruction-steered modes, register-compatible |
| Cost | about $0.66 estimated; $20 hard cap enforced in code |

Files: `build_requests.py` (deterministic frozen matrix), `requests.csv` (the
matrix, reviewed before batch), `generate.py` (`--dry-run/--smoke/--batch`),
`ledger/` (one receipt per completed request, tracked), `raw/` (WAVs,
git-ignored), and `prep.py` (offline staging from ledger+raw).
Full safety model: `dataset/TTS_PLAN.md` and `common/ttsgen.py`.
