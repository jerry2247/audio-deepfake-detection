#!/usr/bin/env python3
"""peoples_speech — MLCommons People's Speech (real, English): 30k+ hours of
diverse real-world English (government meetings, radio, interviews, lectures) with
natural noise and far-field conditions — the messy-real-world counterweight to the
clean studio corpora. We stream the `clean/test` shards only.

Usage: .venv/bin/python prep.py all [--target N]   (default 1000)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_bytes_to_staged_wav  # noqa: E402
from common.hf import stream_parquet_rows, shuffled_shards  # noqa: E402

SOURCE = "peoples_speech"
REPO = "MLCommons/peoples_speech"
PER_RECORDING_CAP = 25  # the clean/test shards hold whole recordings contiguously;
                        # without a cap a 1,000-clip sample covers only ~12 recordings


def recording_id(row_id: str) -> str:
    """The PS row id embeds the source recording path; strip the segment suffix so
    all segments of one recording share a leakage group."""
    base = row_id.split("_SLASH_")[-1] if "_SLASH_" in row_id else row_id
    base = re.sub(r"[-_]?\d+\.(flac|wav|mp3)$", "", base)
    return base or row_id


def do_stage(target: int):
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    per_rec: dict[str, int] = {}
    shards = shuffled_shards(REPO, "clean", SEED)
    shards = [s for s in shards if "/test-" in s] or shards
    for row in stream_parquet_rows(REPO, shards, limit=int(target * 8),
                                   shuffle_buffer=5000, seed=SEED):
        if i >= target:
            break
        audio = row.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            w.skip("no_audio")
            continue
        rid = str(row.get("id", i))
        rec_key = recording_id(rid)
        if per_rec.get(rec_key, 0) >= PER_RECORDING_CAP:
            w.skip("recording_cap")
            continue
        suffix = Path(audio.get("path") or rid or "x.flac").suffix or ".flac"
        dst = w.next_clip_path(i)
        try:
            info = decode_bytes_to_staged_wav(data, suffix, dst)
        except Exception:
            w.skip("decode_error")
            continue
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="real", language="en", domain="other",
            generator="human", generator_family="human",
            source_recording_id=f"{SOURCE}:{rec_key}",
            utterance_id=rid[:120],
            transcript=str(row.get("text", "") or ""),
            source_uri=f"hf://datasets/{REPO}",
            source_license="cc_by_sa_4.0",
            codec_history="flac",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes="clean_test;real-world recordings (gov meetings, radio, interviews)",
        ))
        if ok:
            i += 1
            per_rec[rec_key] = per_rec.get(rec_key, 0) + 1
    stats = w.finish({"distinct_recordings": len(per_rec)})
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "all"])
    ap.add_argument("--target", type=int, default=1000)
    a = ap.parse_args()
    do_stage(a.target)
