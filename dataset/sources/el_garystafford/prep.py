#!/usr/bin/env python3
"""el_garystafford — ElevenLabs slice of garystafford/deepfake-audio-detection (HF).

We take ONLY the `fake/el_*.flac` files, which the dataset card documents as
ElevenLabs-generated (Dec 2024). Everything else in that repo (Kokoro/Hume/Polly/
Speechify/Luvvoice fakes, real clips) is deliberately not used — see README.md.
Pooled like every other source: train/val/test sampled randomly at build time.

Usage: .venv/bin/python prep.py all          (download + stage)
       .venv/bin/python prep.py download|stage [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_to_staged_wav, ffprobe_info  # noqa: E402
from common.hf import list_files, download, sample_deterministic  # noqa: E402

REPO = "garystafford/deepfake-audio-detection"
SOURCE = "el_garystafford"
RAW = SRC_DIR / "raw"


def do_download(limit: int | None):
    files = [f for f in list_files(REPO, "fake", suffixes=(".flac",))
             if Path(f).name.startswith("el_")]
    files = sorted(files)
    if limit:
        files = sample_deterministic(files, limit, SEED)
    print(f"[{SOURCE}] downloading {len(files)} ElevenLabs flac files")
    for i, f in enumerate(files):
        download(REPO, f, RAW)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"[{SOURCE}] download done -> {RAW}")


def do_stage(limit: int | None):
    flacs = sorted((RAW / "fake").glob("el_*.flac"))
    if limit:
        flacs = flacs[:limit]
    if not flacs:
        sys.exit(f"[{SOURCE}] no raw files; run download first")
    w = StagingWriter(SRC_DIR, SOURCE)
    for i, src in enumerate(flacs):
        dst = w.next_clip_path(i)
        try:
            info = decode_to_staged_wav(src, dst)
        except Exception as e:
            w.skip(f"decode_error")
            continue
        stem = src.stem  # e.g. el_0001_c_part_002 ; "_c_" marks a codec-compressed variant
        rec_id = "el_" + stem.split("_")[1]  # groups parts of the same generation
        w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="fake", language="en", domain="studio",
            generator="elevenlabs", generator_family="elevenlabs",
            generator_version="unknown_2024", synthesis_paradigm="unknown",
            generation_date="2024-12", vintage="2024-12",
            source_recording_id=f"{SOURCE}:{rec_id}",
            utterance_id=stem,
            source_uri=f"hf://datasets/{REPO}/fake/{src.name}",
            source_license="cc_by_4.0",
            codec_history="flac",
            native_sample_rate_hz=info["sample_rate"], duration_s=round(info["duration_s"], 3),
            test_only="0",
            notes="codec-variant" if "_c_" in stem else "",
        ))
    stats = w.finish({"repo": REPO})
    print(f"[{SOURCE}] staged {stats['clips']} clips ({stats['hours']} h); skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "stage", "all"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        do_download(a.limit)
    if a.cmd in ("stage", "all"):
        do_stage(a.limit)
