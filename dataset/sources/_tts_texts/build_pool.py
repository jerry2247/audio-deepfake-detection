#!/usr/bin/env python3
"""Build the shared TTS text pool (pool.csv) from two inputs:

  1. HARVESTED (~65%): verbatim transcripts of REAL clips already staged in this
     dataset (common_voice, emilia, voxpopuli, peoples_speech). Every harvested
     text creates a true real<->fake matched pair once synthesized — the split
     graph keeps the pair in one split via content_id, and finalize derives
     matched_pair_id.
  2. CURATED (~35%): curated_lines.csv — expressive/emotional lines, numbers and
     entities, disfluencies, questions, tongue twisters, long-form paragraphs,
     and Grok-tag-native lines. Reviewed by the project lead before freezing.

Output columns: text_id, register, language, origin, chars, text
  - text_id: cur_* (curated) or h_<source>_<md5_12> (harvested, stable)
  - register: read | conversational | formal | procedural | emotional | whisper |
              shout | numbers | disfluent | question | dense | longform |
              grok_tags | singing
  - origin: curated | <source name>

Deterministic (seed 20260604). Run: ../../.venv/bin/python build_pool.py
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import SEED, load_staged_csv  # noqa: E402

# harvested quotas per source (register, count, min/max chars)
HARVEST = {
    "common_voice":   ("read", 420, 40, 180),   # incl. all usable es/de/fr transcripts
    "emilia":         ("conversational", 230, 40, 260),
    "voxpopuli":      ("formal", 150, 60, 300),
    "peoples_speech": ("procedural", 140, 60, 300),
}
BAD_CHARS = re.compile(r"[<>\[\]{}|\\^~]")  # avoid accidental tag/markup injection


def clean(text: str) -> str | None:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if BAD_CHARS.search(t):
        return None
    if not re.search(r"[a-zA-ZÀ-ÿ]", t):
        return None
    return t


def harvest() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    for source, (register, count, lo, hi) in HARVEST.items():
        staged = DATASET_ROOT / "sources" / source / "staged" / "staged.csv"
        if not staged.exists():
            sys.exit(f"staged.csv missing for {source} — stage it before building the pool")
        candidates = []
        seen: set[str] = set()
        for r in load_staged_csv(staged):
            t = clean(r.get("transcript", ""))
            if not t or not (lo <= len(t) <= hi):
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((t, r.get("language", "en")))
        candidates.sort()
        rng.shuffle(candidates)
        if len(candidates) < count:
            print(f"WARNING: {source} has only {len(candidates)} usable transcripts "
                  f"(wanted {count}) — taking all")
        for t, lang in candidates[:count]:
            tid = f"h_{source}_{hashlib.md5(t.lower().encode()).hexdigest()[:12]}"
            rows.append({"text_id": tid, "register": register, "language": lang,
                         "origin": source, "chars": len(t), "text": t})
    return rows


def curated() -> list[dict]:
    rows = []
    with open(HERE / "curated_lines.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = re.sub(r"\s+", " ", r["text"]).strip()
            rows.append({"text_id": r["text_id"], "register": r["register"],
                         "language": r["language"], "origin": "curated",
                         "chars": len(t), "text": t})
    ids = [r["text_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate text_id in curated_lines.csv")
    return rows


def main():
    rows = harvest() + curated()
    with open(HERE / "pool.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_id", "register", "language",
                                          "origin", "chars", "text"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    reg = Counter(r["register"] for r in rows)
    lang = Counter(r["language"] for r in rows)
    total_chars = sum(r["chars"] for r in rows)
    print(f"pool.csv: {len(rows)} texts, {total_chars:,} chars")
    print("registers:", dict(reg.most_common()))
    print("languages:", dict(lang.most_common()))


if __name__ == "__main__":
    main()
