#!/usr/bin/env python3
"""grok_tts staging — offline, no network: ledger/ + raw/ -> staged/.

Usage: ../../.venv/bin/python prep.py stage
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.ttsgen import stage_generated  # noqa: E402

if __name__ == "__main__":
    stage_generated(SRC_DIR, "grok_tts", family="grok",
                    license_bucket="grok_tos")
