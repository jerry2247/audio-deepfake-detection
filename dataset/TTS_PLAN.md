# TTS Generation Record (openai_tts, grok_tts)

The two generated fake sources were produced on 2026-06-04 by querying the
OpenAI and xAI speech APIs against frozen, reviewed request matrices. This
document records the methodology; per-source details and achieved counts live
in `sources/openai_tts/README.md` and `sources/grok_tts/README.md`.

## Outcome

| source | clips | hours | voices | styles | model |
|---|---|---|---|---|---|
| openai_tts | 408 | 0.82 | all 13 built-in voices, balanced 30 to 32 each | 8 instruction-steered styles | gpt-4o-mini-tts |
| grok_tts | 702 | 1.19 | all 5 voices, balanced 140 to 141 each | 9 styles including native speech tags | Grok TTS |

No clip failed automated verification. Each provider's output passed a
25-clip signal battery (speech-like energy distribution, PCM16 mono decode,
healthy levels, plausible durations) and conditioned spot checks at 100
percent contract format. Combined spend was about 2 dollars against a 40
dollar cap. Model identifiers were resolved at generation time from the live
model lists and recorded per clip; no voice cloning was used.

## Text pool

A shared text pool feeds both providers, so the same text rendered by both
gives a controlled cross-generator comparison. About 65 percent of the texts
were harvested from transcripts of staged real clips (Common Voice, Emilia,
VoxPopuli), producing genuine real-to-fake matched content; the remainder are
curated lines covering expressive speech, numbers, dates, entities, and
long-form paragraphs. The pool is built deterministically by
`sources/_tts_texts/build_pool.py` and tracked in git. Clip length mix:
roughly half short (3 to 8 s), a third medium (8 to 20 s), and the rest
long-form passages segmented to at most 30 s by the conditioner.

## Spend safety and reproducibility

Generation was designed so that no API credit could be wasted and no crash
could lose or double-bill work:

1. Frozen matrices. `requests.csv` for each provider was generated
   deterministically, reviewed, and frozen by SHA-256; the runner refuses any
   matrix whose hash differs. The approved hashes are recorded below.
2. Dry run and smoke gate. A dry run printed request counts, character
   totals, and cost estimates with zero network calls. A smoke pass sent one
   shortest-text request per provider and voice (18 calls) and ran automated
   audio checks; the batch refused to start without a passing smoke receipt.
3. Hard caps. The runner aborts before any call if total characters exceed
   the cap or any single text exceeds 1,200 characters.
4. Write-ahead receipts. Each completed request produced an atomic audio
   file plus a JSON receipt (voice, style, text id, model id, characters,
   HTTP request id, duration, timestamp). Completed requests are never
   re-billed; a crash loses at most the single in-flight request. Receipts
   are tracked in git as the billing and provenance audit trail; raw audio
   is git-ignored like all source audio.
5. Failure discipline. Server errors retried with exponential backoff up to
   three times; any client error aborted the entire run.

A nine-test safety suite covered crash resume, double-billing prevention, abort
semantics, cap enforcement, hash freeze, deterministic matrix build, smoke
selection, and markup stripping. It passed 9 of 9 checks before the batch ran.

Both sources then flowed through the same staging, validation, conditioning
and split gates as every other source.

## Approved matrix hashes

| source | requests.csv sha256 | approved |
|---|---|---|
| openai_tts | b63b5940ea276e114713f6fc6d7f0147df499d4e7815ebb099d839bf389fa747 | 2026-06-04 |
| grok_tts | 73aab5b9c4fca305d5a2e962ee927bd35d2b7e1120a43bd84b8012f5c5c4e7c6 | 2026-06-04 |
