#!/usr/bin/env python3
"""build/finalize.py — gate 4: THE ONLY SCRIPT THAT WRITES dataset/audio/.

HARD GUARD — refuses to run unless ALL of:
  1. env  DATASET_FINALIZE_AUTHORIZED=YES
  2. flag --i-am-authorized-to-write-audio
  3. dataset/audio/ contains nothing but .gitkeep files (immutability: a populated
     audio/ tree is never overwritten; a new dataset_version starts clean)

This script is intentionally NOT run during development. It exists so the final
move is a single reviewed command once the project lead gives the word:

    DATASET_FINALIZE_AUTHORIZED=YES .venv/bin/python build/finalize.py \
        --i-am-authorized-to-write-audio --dataset-version v1.0

What it does:
  - reads build/work/conditioned.csv + build/work/assignment.csv
  - mints clip_id = "clip_" + md5(source|utterance_id|segment_index) (metadata-
    derived, stable, NOT the content hash — per DATASHEET)
  - derives matched_pair_id where a real and fake share a source_recording_id
  - emits audio/<split>/<label>/<clip_id>.wav (copy of conditioned segment)
  - writes manifest.parquet with the EXACT 43-column frozen schema (order,
    names, types per DATASHEET.md) and dataset_card.json (version, conditioning
    params, per-split/class counts, manifest sha256)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BUILD_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

WORK = BUILD_DIR / "work"
AUDIO = DATASET_ROOT / "audio"

# DATASHEET.md manifest schema: (name, pyarrow type, required)
SCHEMA = [
    ("clip_id", "string"), ("path", "string"), ("label", "string"),
    ("split", "string"), ("eval_condition", "string"), ("split_group_id", "string"),
    ("is_heldout_generator", "bool"), ("is_in_the_wild", "bool"),
    ("source_dataset", "string"), ("generator", "string"),
    ("generator_family", "string"), ("generator_version", "string"),
    ("synthesis_paradigm", "string"), ("generation_date", "string"),
    ("voice_id", "string"), ("speaker_id", "string"),
    ("cloned_source_speaker_id", "string"), ("source_recording_id", "string"),
    ("utterance_id", "string"), ("source_uri_or_dataset_ref", "string"),
    ("source_license", "string"), ("language", "string"), ("domain", "string"),
    ("transcript", "string"), ("content_id", "string"), ("matched_pair_id", "string"),
    ("duration_s", "double"), ("final_sample_rate", "int64"),
    ("final_channels", "int64"), ("final_format", "string"), ("bit_depth", "int64"),
    ("file_size_bytes", "int64"), ("sha256", "string"),
    ("native_sample_rate", "int64"), ("codec_history", "string"),
    ("loudness_lufs", "double"), ("peak_dbfs", "double"),
    ("leading_silence_ms", "double"), ("trailing_silence_ms", "double"),
    ("vad_speech_fraction", "double"), ("measured_bandwidth_hz", "double"),
    ("bandwidth_flag", "string"), ("conditioning_version", "string"),
]


def guard(args) -> None:
    if os.environ.get("DATASET_FINALIZE_AUTHORIZED") != "YES":
        sys.exit("REFUSED: env DATASET_FINALIZE_AUTHORIZED=YES not set. "
                 "audio/ may only be written with explicit authorization.")
    if not args.i_am_authorized_to_write_audio:
        sys.exit("REFUSED: missing --i-am-authorized-to-write-audio.")
    stray = [p for p in AUDIO.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if stray:
        sys.exit(f"REFUSED: audio/ is not empty ({len(stray)} files). The dataset is "
                 "immutable once built — bump dataset_version and clear intentionally.")


def sentinel(v: str) -> str:
    return v if (v or "").strip() else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-authorized-to-write-audio", action="store_true")
    ap.add_argument("--dataset-version", default="v1.0")
    a = ap.parse_args()
    guard(a)
    import pandas as pd

    cond = {r["cond_path"]: r for r in
            csv.DictReader(open(WORK / "conditioned.csv", encoding="utf-8"))}
    assign = list(csv.DictReader(open(WORK / "assignment.csv", encoding="utf-8")))
    # matched pairs: real & fake sharing a source_recording_id
    rec_groups: dict[str, set[str]] = defaultdict(set)
    for r in cond.values():
        if (r.get("source_recording_id") or "").strip():
            rec_groups[r["source_recording_id"]].add(r["label"])
    out_rows = []
    for arow in assign:
        r = cond.get(arow["cond_path"])
        if r is None:
            continue
        split = arow["split"]
        # cond_path is unique per conditioned segment by construction, so the
        # minted id cannot collide. Utterance ids are NOT collision-safe:
        # peoples_speech truncates long recording ids at staging, which made
        # distinct clips share an id in the first v1.0 build attempt.
        clip_id = "clip_" + hashlib.md5(arow["cond_path"].encode()).hexdigest()
        label = r["label"]
        rel = f"audio/{split}/{label}/{clip_id}.wav"
        is_itw = "in_the_wild" in (r.get("notes") or "")
        eval_condition = {"train": "train_pool", "val": "validation_pool"}.get(
            split, "in_the_wild" if is_itw else "optional_probe")
        rec = (r.get("source_recording_id") or "").strip()
        matched = ("mp_" + hashlib.md5(rec.encode()).hexdigest()[:24]
                   if rec and rec_groups.get(rec) == {"real", "fake"} else "")
        bw = float(r["measured_bandwidth_hz"] or 0)
        out_rows.append({
            "clip_id": clip_id, "path": rel, "label": label, "split": split,
            "eval_condition": eval_condition,
            "split_group_id": arow["split_group_id"],
            "is_heldout_generator": False,  # no held-out family (decision 2026-06-04)
            "is_in_the_wild": is_itw,
            "source_dataset": r["source"], "generator": r["generator"],
            "generator_family": r["generator_family"],
            "generator_version": sentinel(r["generator_version"]),
            "synthesis_paradigm": r["synthesis_paradigm"] or "n/a",
            "generation_date": sentinel(r["generation_date"]),
            "voice_id": sentinel(r["voice_id"]),
            "speaker_id": sentinel(r["speaker_id"]),
            "cloned_source_speaker_id": sentinel(r["cloned_source_speaker_id"]),
            "source_recording_id": sentinel(r["source_recording_id"]),
            "utterance_id": sentinel(r["utterance_id"]),
            "source_uri_or_dataset_ref": sentinel(r["source_uri"]),
            "source_license": r["source_license"], "language": r["language"],
            "domain": r["domain"], "transcript": sentinel(r["transcript"]),
            "content_id": arow.get("content_id") or "unknown",
            "matched_pair_id": matched or "n/a",
            "duration_s": float(r["final_duration_s"]),
            "final_sample_rate": 16000, "final_channels": 1,
            "final_format": "wav_pcm16", "bit_depth": 16,
            "file_size_bytes": (WORK / "conditioned" / r["cond_path"]).stat().st_size,
            "sha256": r["sha256"],
            "native_sample_rate": int(float(r["native_sample_rate_hz"] or 0)),
            "codec_history": r["codec_history"],
            "loudness_lufs": float(r["loudness_lufs"] or "nan"),
            "peak_dbfs": float(r["peak_dbfs"]),
            "leading_silence_ms": float(r["leading_silence_ms"]),
            "trailing_silence_ms": float(r["trailing_silence_ms"]),
            "vad_speech_fraction": float(r["vad_speech_fraction"]),
            "measured_bandwidth_hz": bw,
            "bandwidth_flag": ("full_band" if bw >= 6800 else "band_limited"),
            "conditioning_version": json.loads(
                (WORK / "conditioning_summary.json").read_text())["conditioning_version"],
            "_src": str(WORK / "conditioned" / r["cond_path"]),
        })
    # hard guard: refuse to write anything if minted ids or paths collide
    ids = [row["clip_id"] for row in out_rows]
    paths = [row["path"] for row in out_rows]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        sys.exit("REFUSED: minted clip_ids or paths are not unique; "
                 "nothing was written")
    # write audio tree
    for row in out_rows:
        dst = DATASET_ROOT / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.pop("_src"), dst)
    df = pd.DataFrame(out_rows, columns=[c for c, _ in SCHEMA])
    df.to_parquet(DATASET_ROOT / "manifest.parquet", index=False)
    manifest_sha = hashlib.sha256((DATASET_ROOT / "manifest.parquet").read_bytes()).hexdigest()
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    card = {
        "dataset_version": a.dataset_version,
        "conditioning": json.loads((WORK / "conditioning_summary.json").read_text()),
        "counts": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                   for k, v in counts.items()},
        "n_clips": int(len(df)),
        "manifest_sha256": manifest_sha,
    }
    (DATASET_ROOT / "dataset_card.json").write_text(json.dumps(card, indent=2))
    print(f"FINALIZED {len(df)} clips -> audio/ ; manifest.parquet ; dataset_card.json")


if __name__ == "__main__":
    main()
