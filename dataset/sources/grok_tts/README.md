# grok_tts: Grok TTS generated fake speech

| | |
|---|---|
| Status | complete; **702 clips staged** (batch plus smoke, 0 flagged; signal battery clean) |
| Label | fake |
| Achieved | 700-request approved matrix executed; 5 voices balanced (140-141 clips each), 9 styles including native tags, English |
| Generator | xAI Grok TTS via `POST api.x.ai/v1/tts` (model resolved server-side, recorded per clip) |
| Voices / modes | 5 voices (eve/ara/rex/sal/leo) × 9 modes incl. native tags `[laugh] [sigh] <whisper> <singing>` and `speed` 0.8/1.3 |
| Cost | about $1.11 estimated; $20 hard cap enforced in code |

Files: `build_requests.py` (deterministic frozen matrix), `requests.csv` (the
matrix, reviewed before batch), `generate.py` (`--dry-run/--smoke/--batch`),
`ledger/` (one receipt per completed request, tracked), `raw/` (WAVs,
git-ignored), and `prep.py` (offline staging from ledger+raw).
Full safety model: `dataset/TTS_PLAN.md` and `common/ttsgen.py`.
