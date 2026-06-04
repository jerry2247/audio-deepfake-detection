#!/usr/bin/env python3
"""expresso — Meta's Expresso corpus (real, English): 4 professional speakers in
studio reading + expressive styles (happy, sad, whisper, laughing, confused, ...).
Adds expressive non-neutral real speech. Mirror: ylacombe/expresso (parquet).

Streams parquet shards with Audio(decode=False) — raw bytes decoded by our ffmpeg
path; only reads what we keep.

Usage: .venv/bin/python prep.py all [--target N]   (default 420)
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
from common.hf import stream_parquet_rows, shuffled_shards  # noqa: E402

SOURCE = "expresso"
REPO = "ylacombe/expresso"


def do_stage(target: int):
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    # Expresso has exactly 4 speakers and its shards are speaker-contiguous:
    # shuffle the shard order and cap per speaker so all 4 are represented.
    per_speaker_cap = target // 4 + 10
    per_spk: dict[str, int] = {}
    for row in stream_parquet_rows(REPO, shuffled_shards(REPO, "read", SEED),
                                   limit=int(target * 8),
                                   shuffle_buffer=3000, seed=SEED):
        if i >= target:
            break
        audio = row.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            w.skip("no_audio")
            continue
        spk = str(row.get("speaker_id", "") or "unk")
        if per_spk.get(spk, 0) >= per_speaker_cap:
            w.skip("speaker_cap")
            continue
        suffix = Path(audio.get("path") or "x.wav").suffix or ".wav"
        dst = w.next_clip_path(i)
        try:
            info = decode_bytes_to_staged_wav(data, suffix, dst)
        except Exception:
            w.skip("decode_error")
            continue
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="real", language="en", domain="studio",
            generator="human", generator_family="human",
            speaker_id=f"expresso_{spk}",
            transcript=str(row.get("text", "") or ""),
            utterance_id=str(row.get("id", i)),
            source_uri=f"hf://datasets/{REPO}",
            source_license="cc_by_nc_4.0",
            codec_history="wav",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes=f"style={row.get('style','')}",
        ))
        if ok:
            i += 1
            per_spk[spk] = per_spk.get(spk, 0) + 1
    stats = w.finish({"by_speaker": per_spk})
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "all"])
    ap.add_argument("--target", type=int, default=420)
    a = ap.parse_args()
    do_stage(a.target)
