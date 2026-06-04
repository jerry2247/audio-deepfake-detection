#!/usr/bin/env python3
"""Build the FROZEN request matrix for grok_tts (700 requests).

Endpoint: POST https://api.x.ai/v1/tts (verified against docs.x.ai 2026-06-04).
5 voices (eve, ara, rex, sal, leo) x 9 modes using Grok's native speech tags
(<whisper>, <loud>, <soft>, <slow>, <fast>, <pitch_*>, inline [laugh]/[sigh]/...)
and the speed parameter; register-compatible assignment; English; deterministic
(seed 20260604).

Run:  ../../.venv/bin/python build_requests.py
Then have the project lead review requests.csv; record its SHA-256 in
TTS_PLAN.md before generate.py will run.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import SEED  # noqa: E402
from common.ttsgen import Mode, build_matrix, write_matrix, summarize  # noqa: E402

TARGET = 700
VOICES = ["eve", "ara", "rex", "sal", "leo"]

MODES = [
    Mode("neutral", ("read", "conversational", "formal", "procedural",
                     "numbers", "question", "dense", "emotional", "longform"),
         weight=3),
    Mode("whisper", ("whisper", "read"), wrap="whisper"),
    Mode("soft", ("emotional", "whisper"), wrap="soft"),
    Mode("loud", ("shout", "emotional"), wrap="loud"),
    Mode("slow", ("longform", "read", "dense"), speed="0.8"),
    Mode("fast", ("conversational", "disfluent", "question"), speed="1.3"),
    Mode("pitch_high", ("dense", "question"), wrap="pitch_high"),
    # texts in these registers already carry their tags inline — send verbatim:
    Mode("tags_native", ("grok_tags", "disfluent")),
    Mode("singing", ("singing",)),
]

if __name__ == "__main__":
    rows = build_matrix("grok", VOICES, MODES, TARGET, SEED)
    digest = write_matrix(SRC_DIR / "requests.csv", rows)
    summarize(rows, "grok")
    print(f"requests.csv sha256: {digest}")
