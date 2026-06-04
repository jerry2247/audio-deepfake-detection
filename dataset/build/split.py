#!/usr/bin/env python3
"""build/split.py — gate 3: leakage-safe random split assignment + balancing.

Builds the identity graph over conditioned segments, assigns whole connected
components to train/val/test at 70/10/20, then balances each split to exactly
1:1 real:fake by deterministic stratified downsampling of the surplus class
(fakes stratified by generator_family, reals by source, so no generator or
corpus is eliminated). Per the project decision (2026-06-04): there is NO
held-out generator family — every source is pooled and sampled randomly at
the group level. All randomized decisions in this gate use SPLIT_SEED = 301.

Edges (union-find):
  - shared speaker identity: any non-empty value appearing in speaker_id OR
    cloned_source_speaker_id (cross-column: a clone of speaker X stays with X)
  - shared source_recording_id (segments of one recording / video / generation)
  - shared content_id (normalized-transcript hash) where the content group
    contains BOTH labels — i.e. the transcript exists as real speech AND as a
    fake rendition (matched pairs / TTS-from-corpus-text). Real-real or
    fake-fake same-text pairs carry no label shortcut and do not link; linking
    them collapses fully-crossed corpora (e.g. CREMA-D's 91 actors x 12 shared
    sentences) into one mega-component for zero leakage benefit. Documented in
    DATASHEET.md.

Outputs build/work/assignment.csv (cond_path -> split) + split_report.md.
audio/ is NOT touched.

Usage: .venv/bin/python build/split.py
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BUILD_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

WORK = BUILD_DIR / "work"
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
# Seed for split assignment and class balancing (project decision: 301).
# Staging-time sampling inside the per-source preps used their own recorded
# constants; this gate's randomness is fully governed by SPLIT_SEED.
SPLIT_SEED = 301


def content_id(transcript: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]+", "", (transcript or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if len(norm) < 12:  # too short/generic to be meaningful identity
        return ""
    return hashlib.md5(norm.encode()).hexdigest()


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    rows = list(csv.DictReader(open(WORK / "conditioned.csv", encoding="utf-8")))
    if not rows:
        sys.exit("no conditioned.csv — run condition.py first")
    uf = UF()
    by_speaker: dict[str, list[int]] = defaultdict(list)
    by_recording: dict[str, list[int]] = defaultdict(list)
    by_content: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        node = f"row:{idx}"
        for col in ("speaker_id", "cloned_source_speaker_id"):
            v = (r.get(col) or "").strip()
            if v:
                by_speaker[v].append(idx)
        rec = (r.get("source_recording_id") or "").strip()
        if rec:
            by_recording[rec].append(idx)
        cid = content_id(r.get("transcript", ""))
        r["_content_id"] = cid
        if cid:
            by_content[cid].append(idx)
    for ids in list(by_speaker.values()) + list(by_recording.values()):
        for i2 in ids[1:]:
            uf.union(f"row:{ids[0]}", f"row:{i2}")
    for cid, ids in by_content.items():
        labels = {rows[i]["label"] for i in ids}
        if labels == {"real", "fake"}:  # transcript exists in BOTH classes
            for i2 in ids[1:]:
                uf.union(f"row:{ids[0]}", f"row:{i2}")
    comps: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        comps[uf.find(f"row:{idx}")].append(idx)
    comp_list = list(comps.values())
    rng = random.Random(SPLIT_SEED)
    # Large components (>=100 segments) are packed first, greedily, into the
    # split with the largest remaining need — they are few and lumpy.
    # Everything else is assigned by SEEDED WEIGHTED-RANDOM choice proportional
    # to remaining per-label need: this keeps the 70/10/20 ratios while
    # decorrelating split assignment from source/row order (a pure argmax
    # greedy assigns long runs of singletons to one split, dumping whole
    # sources into a single split).
    big = sorted([c for c in comp_list if len(c) >= 100], key=len, reverse=True)
    small = [c for c in comp_list if len(c) < 100]
    rng.shuffle(small)
    need = {s: {"real": RATIOS[s] * sum(r["label"] == "real" for r in rows),
                "fake": RATIOS[s] * sum(r["label"] == "fake" for r in rows)}
            for s in RATIOS}
    assignment: dict[int, str] = {}

    def assign(comp: list[int], split: str):
        for i in comp:
            assignment[i] = split
        for i in comp:
            need[split][rows[i]["label"]] -= 1

    for comp in big:
        weight = {"real": sum(rows[i]["label"] == "real" for i in comp),
                  "fake": sum(rows[i]["label"] == "fake" for i in comp)}
        best = max(RATIOS, key=lambda s: need[s]["real"] * weight["real"]
                   + need[s]["fake"] * weight["fake"])
        assign(comp, best)
    splits = list(RATIOS)
    for comp in small:
        weight = {"real": sum(rows[i]["label"] == "real" for i in comp),
                  "fake": sum(rows[i]["label"] == "fake" for i in comp)}
        w = [max(sum(need[s][lab] * weight[lab] for lab in ("real", "fake")), 0.0)
             + 1e-9 for s in splits]
        assign(comp, rng.choices(splits, weights=w, k=1)[0])
    # Class balancing: the dataset ships exactly 1:1 real:fake INSIDE every
    # split (DATASHEET contract). The surplus class is downsampled per split,
    # proportionally stratified (fakes by generator_family, reals by source)
    # so no generator or corpus is eliminated. Dropping segments never creates
    # leakage: identities only ever disappear from a split, never cross one.
    bal_rng = random.Random(SPLIT_SEED)
    dropped: set[int] = set()
    for s in sorted(RATIOS):
        members = {"real": [], "fake": []}
        for i in range(len(rows)):
            if assignment[i] == s:
                members[rows[i]["label"]].append(i)
        n_keep = min(len(members["real"]), len(members["fake"]))
        for lab in ("real", "fake"):
            pool = members[lab]
            if len(pool) <= n_keep:
                continue
            key = "generator_family" if lab == "fake" else "source"
            strata: dict[str, list[int]] = defaultdict(list)
            for i in pool:
                strata[(rows[i].get(key) or "?").strip()].append(i)
            # largest-remainder proportional allocation of n_keep across strata
            quota = {k: n_keep * len(v) / len(pool) for k, v in strata.items()}
            keep_n = {k: int(quota[k]) for k in strata}
            rem = n_keep - sum(keep_n.values())
            for k in sorted(strata, key=lambda k: (-(quota[k] - keep_n[k]), k)):
                if rem == 0:
                    break
                if keep_n[k] < len(strata[k]):
                    keep_n[k] += 1
                    rem -= 1
            while rem > 0:  # spill if some strata saturated
                progressed = False
                for k in sorted(strata):
                    if rem > 0 and keep_n[k] < len(strata[k]):
                        keep_n[k] += 1
                        rem -= 1
                        progressed = True
                if not progressed:
                    break
            for k in sorted(strata):
                m = sorted(strata[k], key=lambda i: rows[i]["cond_path"])
                keep = set(bal_rng.sample(m, keep_n[k]))
                dropped.update(i for i in m if i not in keep)
    # write assignment (balanced selection only)
    with open(WORK / "assignment.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["cond_path", "split", "split_group_id", "content_id"])
        for idx, r in enumerate(rows):
            if idx in dropped:
                continue
            gid = "grp_" + hashlib.md5(uf.find(f"row:{idx}").encode()).hexdigest()[:16]
            wr.writerow([r["cond_path"], assignment[idx], gid, r.get("_content_id", "")])
    # report
    kept = [i for i in range(len(rows)) if i not in dropped]
    lines = ["# Split report", "",
             f"seed: {SPLIT_SEED}; conditioned segments: {len(rows)}; "
             f"components: {len(comp_list)}; "
             f"largest component: {max(len(c) for c in comp_list)}; "
             f"balanced dataset: {len(kept)} clips "
             f"({len(rows) - len(kept)} surplus downsampled)", ""]
    tab = Counter((assignment[i], rows[i]["label"]) for i in kept)
    lines.append("| split | real | fake |")
    lines.append("|---|---|---|")
    for s in ("train", "val", "test"):
        lines.append(f"| {s} | {tab[(s,'real')]} | {tab[(s,'fake')]} |")
    lines += ["", "## By source x split (after balancing)", "",
              "| source | train | val | test |", "|---|---|---|---|"]
    src_tab = Counter((rows[i]["source"], assignment[i]) for i in kept)
    for src in sorted({r["source"] for r in rows}):
        lines.append(f"| {src} | {src_tab[(src,'train')]} | {src_tab[(src,'val')]} "
                     f"| {src_tab[(src,'test')]} |")
    big = [c for c in comp_list if len(c) > 0.05 * len(rows)]
    if big:
        lines += ["", f"WARNING: {len(big)} component(s) exceed 5% of data "
                  f"(sizes: {[len(c) for c in big]}) — check identity-key hygiene"]
    (WORK / "split_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
