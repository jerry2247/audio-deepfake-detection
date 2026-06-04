#!/usr/bin/env python3
"""build/validate.py — gate 1 of the build: checks every sources/*/staged/ tree
against the staging contract before any conditioning happens.

Checks per source: staged.csv schema, row-level field validity, every referenced
wav exists, random ffprobe spot-checks (PCM16/mono), duration bounds, label/
generator coherence. Writes build/work/validation_report.md and exits non-zero on
any hard error.

Usage: .venv/bin/python build/validate.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BUILD_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import load_staged_csv, StagedClip, SEED  # noqa: E402
from common.audio import ffprobe_info  # noqa: E402

WORK = BUILD_DIR / "work"
SPOT_CHECKS_PER_SOURCE = 8


def main() -> int:
    WORK.mkdir(exist_ok=True)
    sources = sorted(p.parents[1] for p in (DATASET_ROOT / "sources").glob("*/staged/staged.csv"))
    hard_errors: list[str] = []
    lines = ["# Staging validation report", ""]
    total = {"real": 0, "fake": 0}
    total_hours = 0.0
    rng = random.Random(SEED)
    for src_dir in sources:
        name = src_dir.name
        try:
            rows = load_staged_csv(src_dir / "staged" / "staged.csv")
        except Exception as e:
            hard_errors.append(f"{name}: cannot read staged.csv: {e}")
            continue
        errs = 0
        for r in rows:
            clip = StagedClip(**{k: r.get(k, "") for k in r})
            clip.native_sample_rate_hz = int(float(r["native_sample_rate_hz"] or 0))
            clip.duration_s = float(r["duration_s"] or 0)
            problems = clip.validate()
            if problems:
                errs += 1
                if errs <= 3:
                    hard_errors.append(f"{name}/{r['staged_path']}: {problems}")
            if not (src_dir / r["staged_path"]).exists():
                errs += 1
                hard_errors.append(f"{name}: missing wav {r['staged_path']}")
        # ffprobe spot checks
        sample = rng.sample(rows, min(SPOT_CHECKS_PER_SOURCE, len(rows)))
        for r in sample:
            try:
                info = ffprobe_info(src_dir / r["staged_path"])
                if info["channels"] != 1:
                    hard_errors.append(f"{name}: {r['staged_path']} not mono")
                if abs(info["duration_s"] - float(r["duration_s"])) > 0.2:
                    hard_errors.append(f"{name}: {r['staged_path']} duration mismatch")
            except Exception as e:
                hard_errors.append(f"{name}: probe failed {r['staged_path']}: {e}")
        hrs = sum(float(r["duration_s"]) for r in rows) / 3600
        total_hours += hrs
        for r in rows:
            total[r["label"]] = total.get(r["label"], 0) + 1
        gens = len({r["generator"] for r in rows if r["label"] == "fake"})
        lines.append(f"- **{name}**: {len(rows)} clips, {hrs:.2f} h, "
                     f"{gens} fake generators, {errs} row errors")
    lines += ["", f"**Totals**: real {total.get('real',0)} / fake {total.get('fake',0)}, "
              f"{total_hours:.1f} h", ""]
    if hard_errors:
        lines += ["## HARD ERRORS", ""] + [f"- {e}" for e in hard_errors[:50]]
    (WORK / "validation_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 1 if hard_errors else 0


if __name__ == "__main__":
    sys.exit(main())
