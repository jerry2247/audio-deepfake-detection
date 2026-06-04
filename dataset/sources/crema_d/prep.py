#!/usr/bin/env python3
"""crema_d — CREMA-D acted emotional speech (real). 91 actors x 12 sentences x 6
emotions (anger, disgust, fear, happy, neutral, sad) — covers shouted/emotional real
speech that read-speech corpora lack. Mirror: confit/cremad (single zip).

We sample a balanced subset across actors and emotions.

Usage: .venv/bin/python prep.py all [--target N]   (default 650)
"""
from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_to_staged_wav  # noqa: E402
from common.hf import download  # noqa: E402

SOURCE = "crema_d"
REPO = "confit/cremad"
RAW = SRC_DIR / "raw"

SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
            "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def do_download():
    z = download(REPO, "crema-d.zip", RAW)
    marker = RAW / "crema-d.extracted"
    if not marker.exists():
        print(f"[{SOURCE}] extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW / "extracted")
        marker.touch()
    print(f"[{SOURCE}] download done")


def do_stage(target: int):
    wavs = sorted((RAW / "extracted").rglob("*.wav"))
    if not wavs:
        sys.exit(f"[{SOURCE}] nothing extracted; run download first")
    # balanced sample: round-robin over (emotion, actor) cells
    by_cell: dict[tuple[str, str], list[Path]] = {}
    for p in wavs:
        parts = p.stem.split("_")  # ActorID_Sentence_Emotion_Level
        if len(parts) < 4 or parts[2] not in EMOTIONS:
            continue
        by_cell.setdefault((parts[2], parts[0]), []).append(p)
    rng = random.Random(SEED)
    for cell in by_cell.values():
        rng.shuffle(cell)
    picked: list[Path] = []
    cells = sorted(by_cell.keys())
    while len(picked) < target and cells:
        nxt = []
        for c in cells:
            if by_cell[c]:
                picked.append(by_cell[c].pop())
                if len(picked) >= target:
                    break
            if by_cell[c]:
                nxt.append(c)
        cells = nxt
    w = StagingWriter(SRC_DIR, SOURCE)
    for i, src in enumerate(sorted(picked)):
        actor, sent, emo, level = (src.stem.split("_") + [""])[:4]
        dst = w.next_clip_path(i)
        try:
            info = decode_to_staged_wav(src, dst)
        except Exception:
            w.skip("decode_error")
            continue
        w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="real", language="en", domain="studio",
            generator="human", generator_family="human",
            speaker_id=f"cremad_{actor}",
            transcript=SENTENCES.get(sent, ""),
            utterance_id=src.stem,
            source_uri=f"hf://datasets/{REPO}",
            source_license="odbl",
            codec_history="wav",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes=f"emotion={EMOTIONS.get(emo,'?')};level={level}",
        ))
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "stage", "all"])
    ap.add_argument("--target", type=int, default=650)
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        do_download()
    if a.cmd in ("stage", "all"):
        do_stage(a.target)
