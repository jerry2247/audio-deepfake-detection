#!/usr/bin/env python3
"""common_voice — Mozilla Common Voice v22.0 (real, English): crowdsourced,
human-reviewed read speech, thousands of speakers/devices/accents. CC0.
Mirror: fsicoli/common_voice_22_0 (newest ungated mirror; the official v25
requires a Mozilla Data Collective account for zero material benefit).

We use the validated `test` split (every clip ≥2 positive reviews). Speaker ids
are namespaced cv_<client_id> and recording ids cv:<clip stem> — the SAME
namespace sources/echofake uses for its Common Voice source material, so the
split graph links a CV clip with any EchoFake spoof derived from it.

Usage: .venv/bin/python prep.py all [--target N]   (default 1100)
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tarfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_bytes_to_staged_wav  # noqa: E402
from common.hf import download, list_files  # noqa: E402

SOURCE = "common_voice"
REPO = "fsicoli/common_voice_22_0"
LANG = "en"
RAW = SRC_DIR / "raw"


def do_download():
    tsvs = [f for f in list_files(REPO, f"transcript/{LANG}", suffixes=(".tsv",))
            if Path(f).stem == "test"]
    if not tsvs:
        sys.exit(f"[{SOURCE}] no test.tsv found for {LANG}")
    download(REPO, tsvs[0], RAW)
    tars = list_files(REPO, f"audio/{LANG}/test", suffixes=(".tar",))
    print(f"[{SOURCE}] downloading {len(tars)} test tar shard(s)")
    for t in tars:
        download(REPO, t, RAW)
    print(f"[{SOURCE}] download done")


def do_stage(target: int):
    tsv_path = RAW / f"transcript/{LANG}/test.tsv"
    rows_by_clip: dict[str, dict] = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows_by_clip[Path(r["path"]).name] = r
    tars = sorted((RAW / f"audio/{LANG}/test").glob("*.tar"))
    if not tars:
        sys.exit(f"[{SOURCE}] no tars; run download first")
    names = sorted(rows_by_clip.keys())
    rng = random.Random(SEED)
    rng.shuffle(names)
    wanted = set(names[: int(target * 1.4)])
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    for tar_path in tars:
        if i >= target:
            break
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if i >= target:
                    break
                fname = Path(m.name).name
                if fname not in wanted:
                    continue
                r = rows_by_clip[fname]
                data = tf.extractfile(m)
                if data is None:
                    continue
                dst = w.next_clip_path(i)
                try:
                    info = decode_bytes_to_staged_wav(data.read(), Path(fname).suffix or ".mp3", dst)
                except Exception:
                    w.skip("decode_error")
                    continue
                ok = w.add(StagedClip(
                    staged_path=str(dst.relative_to(SRC_DIR)),
                    source=SOURCE, label="real", language="en", domain="read_speech",
                    generator="human", generator_family="human",
                    speaker_id=f"cv_{r.get('client_id','')}",
                    source_recording_id=f"cv:{Path(fname).stem}",
                    utterance_id=Path(fname).stem,
                    transcript=r.get("sentence", "") or "",
                    source_uri=f"hf://datasets/{REPO}",
                    source_license="cc0",
                    codec_history="mp3_unknown",
                    native_sample_rate_hz=info["sample_rate"],
                    duration_s=round(info["duration_s"], 3),
                    notes=f"accents={r.get('accents','')};gender={r.get('gender','')}"[:80],
                ))
                if ok:
                    i += 1
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "stage", "all"])
    ap.add_argument("--target", type=int, default=1100)
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        do_download()
    if a.cmd in ("stage", "all"):
        do_stage(a.target)
