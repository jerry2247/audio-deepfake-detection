#!/usr/bin/env python3
"""grok_tts generation runner — see common/ttsgen.py for the safety model.

Usage: ../../.venv/bin/python generate.py --dry-run | --smoke | --batch

APPROVED_HASH is the SHA-256 of requests.csv recorded after project-lead review;
--batch refuses to run while it is None or if the file has changed since review.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.ttsgen import cli, GrokProvider  # noqa: E402

APPROVED_HASH: str | None = (
    "73aab5b9c4fca305d5a2e962ee927bd35d2b7e1120a43bd84b8012f5c5c4e7c6"
)  # approved 2026-06-04

if __name__ == "__main__":
    cli(SRC_DIR, "grok", lambda key: GrokProvider(key), APPROVED_HASH,
        default_concurrency=6)
