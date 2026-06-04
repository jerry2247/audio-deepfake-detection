#!/usr/bin/env python3
"""voxpopuli — Meta VoxPopuli (real): European Parliament recordings 2009-2020.
Entirely pre-modern-TTS-era => zero AI-contamination risk by construction. We take
English plus the L2 accented-English subset (16 non-native accents) for speaker/
accent diversity; formal register, real room acoustics. CC0.

Usage: .venv/bin/python prep.py all [--target-en N] [--target-accented N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_bytes_to_staged_wav  # noqa: E402
from common.hf import stream_parquet_rows  # noqa: E402

SOURCE = "voxpopuli"
REPO = "facebook/voxpopuli"


def stage_config(w: StagingWriter, glob: str, target: int, start_idx: int,
                 note: str) -> int:
    i = start_idx
    kept = 0
    for row in stream_parquet_rows(REPO, glob, limit=int(target * 1.8),
                                   shuffle_buffer=2000, seed=SEED):
        audio = row.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            w.skip("no_audio")
            continue
        suffix = Path(audio.get("path") or "x.wav").suffix or ".wav"
        dst = w.next_clip_path(i)
        try:
            info = decode_bytes_to_staged_wav(data, suffix, dst)
        except Exception:
            w.skip("decode_error")
            continue
        spk = str(row.get("speaker_id", "") or "")
        acc = str(row.get("accent", "") or "")
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="real", language="en", domain="parliament",
            generator="human", generator_family="human",
            speaker_id=f"vp_{spk}" if spk and spk.lower() != "none" else "",
            utterance_id=str(row.get("audio_id", i)),
            transcript=str(row.get("normalized_text", "") or row.get("raw_text", "") or ""),
            source_uri=f"hf://datasets/{REPO}",
            source_license="cc0",
            codec_history="wav",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes=(note + (f";accent={acc}" if acc and acc.lower() != "none" else "")),
        ))
        if ok:
            i += 1
            kept += 1
        if kept >= target:
            break
    return i


def do_stage(target_en: int, target_accented: int):
    w = StagingWriter(SRC_DIR, SOURCE)
    i = stage_config(w, "en/test-*.parquet", target_en, 0, "en_parliament")
    stage_config(w, "en_accented/test-*.parquet", target_accented, i, "accented_l2_english")
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "all"])
    ap.add_argument("--target-en", type=int, default=480)
    ap.add_argument("--target-accented", type=int, default=220)
    a = ap.parse_args()
    do_stage(a.target_en, a.target_accented)
