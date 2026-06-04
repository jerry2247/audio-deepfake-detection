#!/usr/bin/env python3
"""emilia — Emilia-YODAS EN (real): in-the-wild spontaneous English speech from
YouTube (podcasts, talk shows, interviews) — the conversational counterweight to
read/parliament corpora. CC-BY portion (Emilia-YODAS), gated-auto on HF (token
already authorized).

Contamination note: Emilia was crawled 2023-24, when AI narration existed on
YouTube. Mitigations: (1) we keep only clips with DNSMOS >= 2.8 and duration
4-30 s, (2) the per-video grouping (source_recording_id = video id) means any
suspect video can be excised wholesale, (3) volume from this source is capped low
relative to provably-human corpora.

Usage: .venv/bin/python prep.py all [--target N]   (default 350)
"""
from __future__ import annotations

import argparse
import io
import json
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

SOURCE = "emilia"
REPO = "amphion/Emilia-Dataset"
RAW = SRC_DIR / "raw"
N_TARS = 2  # each EN tar holds ~3-4k clips; 2 is ample for our target


def pick_tars() -> list[str]:
    tars = list_files(REPO, "Emilia-YODAS/EN", suffixes=(".tar",))
    rng = random.Random(SEED)
    rng.shuffle(tars)
    return sorted(tars[:N_TARS])


def do_download():
    for t in pick_tars():
        print(f"[{SOURCE}] downloading {t}")
        download(REPO, t, RAW)
    print(f"[{SOURCE}] download done")


def do_stage(target: int):
    tars = sorted((RAW / "Emilia-YODAS/EN").glob("*.tar"))
    if not tars:
        sys.exit(f"[{SOURCE}] no tars; run download first")
    w = StagingWriter(SRC_DIR, SOURCE)
    # collect (json, mp3) member pairs streamingly; sample deterministically
    entries: list[tuple[str, dict]] = []
    payloads: dict[str, bytes] = {}
    for tar_path in tars:
        with tarfile.open(tar_path) as tf:
            metas: dict[str, dict] = {}
            for m in tf:
                stem = Path(m.name).stem
                if m.name.endswith(".json"):
                    try:
                        metas[stem] = json.load(tf.extractfile(m))
                    except Exception:
                        continue
                elif m.name.endswith((".mp3", ".flac", ".wav")):
                    f = tf.extractfile(m)
                    if f is not None:
                        payloads[stem] = f.read()
            for stem, meta in metas.items():
                if stem in payloads:
                    entries.append((stem, meta))
    rng = random.Random(SEED)
    rng.shuffle(entries)
    i = 0
    for stem, meta in entries:
        if i >= target:
            break
        dnsmos = float(meta.get("dnsmos", 0) or 0)
        dur = float(meta.get("duration", 0) or 0)
        if dnsmos and dnsmos < 2.8:
            w.skip("low_dnsmos")
            continue
        if not (4.0 <= dur <= 30.0):
            w.skip("duration_window")
            continue
        video_id = stem.rsplit("_", 1)[0]  # e.g. <video>_<segment>
        dst = w.next_clip_path(i)
        try:
            info = decode_bytes_to_staged_wav(payloads[stem], ".mp3", dst)
        except Exception:
            w.skip("decode_error")
            continue
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="real", language="en", domain="podcast",
            generator="human", generator_family="human",
            speaker_id=f"emilia_{meta.get('speaker','')}" if meta.get("speaker") else "",
            source_recording_id=f"emilia:{video_id}",
            utterance_id=stem,
            transcript=str(meta.get("text", "") or ""),
            source_uri=f"hf://datasets/{REPO}",
            source_license="cc_by_4.0",
            codec_history="mp3_unknown",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes=f"dnsmos={dnsmos}",
        ))
        if ok:
            i += 1
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "stage", "all"])
    ap.add_argument("--target", type=int, default=350)
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        do_download()
    if a.cmd in ("stage", "all"):
        do_stage(a.target)
