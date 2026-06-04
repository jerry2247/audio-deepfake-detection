#!/usr/bin/env python3
"""Build the FROZEN request matrix for openai_tts (400 requests).

Model: gpt-4o-mini-tts (verified the newest speech-endpoint model 2026-06-04;
generate.py aborts if the live models list contains anything newer/unknown).
13 built-in voices x 8 instruction-steered modes, register-compatible
assignment, English, deterministic (seed 20260604).

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

TARGET = 400
VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova",
          "onyx", "sage", "shimmer", "verse", "marin", "cedar"]

MODES = [
    Mode("neutral", ("read", "conversational", "formal", "procedural",
                     "numbers", "question", "dense", "longform"), weight=2),
    Mode("excited", ("emotional", "shout", "conversational", "question"),
         instructions="Speak with excited, fast-paced energy, like sharing thrilling news."),
    Mode("whisper", ("whisper",),
         instructions="Whisper very softly, as if telling a secret late at night."),
    Mode("tired", ("conversational", "disfluent", "read"),
         instructions="Sound exhausted and flat, with low energy and frequent small pauses."),
    Mode("anchor", ("formal", "numbers", "longform", "procedural"),
         instructions="Deliver this as a composed, professional news anchor."),
    Mode("storyteller", ("longform", "read", "dense"),
         instructions="Use a warm, slow, expressive storytelling voice."),
    Mode("angry", ("emotional",),
         instructions="Speak with barely contained anger, tense and clipped."),
    Mode("support", ("numbers", "question", "procedural"),
         instructions="Sound like a cheerful, friendly customer support agent."),
]

# grok_tags/singing texts contain Grok-specific markup; never sent to OpenAI.
EXCLUDE = ("grok_tags", "singing")

if __name__ == "__main__":
    rows = build_matrix("openai", VOICES, MODES, TARGET, SEED,
                        exclude_registers=EXCLUDE)
    digest = write_matrix(SRC_DIR / "requests.csv", rows)
    summarize(rows, "openai")
    print(f"requests.csv sha256: {digest}")
