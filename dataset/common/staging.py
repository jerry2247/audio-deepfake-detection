"""Staging contract shared by every source prep.

Each source folder produces:
    sources/<name>/staged/clips/*.wav     decoded PCM16 WAV, mono, NATIVE sample rate
    sources/<name>/staged/staged.csv      one row per staged clip (columns below)
    sources/<name>/STATS.json             tracked summary written by finish_staging()

Staged clips are *candidates*: the central build (dataset/build/) applies the uniform
conditioning (16 kHz mono, VAD edge trim, loudness, codec round-trip, <=30 s
segmentation) and the final selection. Staging preserves native fidelity and may
exceed final per-source targets; the build samples down deterministically.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, asdict, fields
from pathlib import Path

# Deterministic seed used by every prep for sampling decisions.
SEED = 20260604

# Staged candidate clips must be playable speech in this duration window.
# (Final contract is 0 < dur <= 30 s; we stage up to 35 s and let the build
#  segment. Final minimum after VAD edge-trim is 2.0 s — chosen so short acted
#  CREMA-D utterances survive; all four backbones accept 2 s input.)
MIN_STAGED_S = 1.8
MAX_STAGED_S = 35.0

STAGED_COLUMNS = [
    "staged_path",            # relative to the source folder, e.g. staged/clips/x.wav
    "source",                 # source folder name, e.g. "echofake"
    "label",                  # real | fake
    "language",               # ISO 639-1 lowercase, or "und"
    "domain",                 # DATASHEET domain enum value
    "generator",              # "human" for real, else model identifier
    "generator_family",       # "human" for real, else family (e.g. "elevenlabs")
    "generator_version",      # version string or ""
    "synthesis_paradigm",     # DATASHEET enum; "n/a" for real
    "generation_date",        # YYYY-MM-DD or YYYY-MM or YYYY or ""
    "vintage",                # generator release vintage: e.g. "2025", "2026", "2024-12"
    "speaker_id",             # human speaker identity where known, else ""
    "voice_id",               # TTS voice / preset where known, else ""
    "cloned_source_speaker_id",
    "source_recording_id",    # original long recording / video / session id
    "utterance_id",           # pre-segmentation utterance id
    "transcript",             # verbatim text if available, else ""
    "source_uri",             # provenance pointer (URL / repo path)
    "source_license",         # DATASHEET license bucket string
    "codec_history",          # origin compression: e.g. mp3_128k, opus_160k, wav, flac
    "native_sample_rate_hz",  # int
    "duration_s",             # float, duration of the staged wav
    "test_only",              # "1" if the clip may only ever appear in test, else "0"
    "notes",                  # free text (kept short)
]


@dataclass
class StagedClip:
    staged_path: str
    source: str
    label: str
    language: str
    domain: str
    generator: str
    generator_family: str
    generator_version: str = ""
    synthesis_paradigm: str = "n/a"
    generation_date: str = ""
    vintage: str = ""
    speaker_id: str = ""
    voice_id: str = ""
    cloned_source_speaker_id: str = ""
    source_recording_id: str = ""
    utterance_id: str = ""
    transcript: str = ""
    source_uri: str = ""
    source_license: str = "research_only"
    codec_history: str = "unknown"
    native_sample_rate_hz: int = 0
    duration_s: float = 0.0
    test_only: str = "0"
    notes: str = ""

    def validate(self) -> list[str]:
        errs = []
        if self.label not in ("real", "fake"):
            errs.append(f"bad label {self.label!r}")
        if self.label == "real" and self.generator != "human":
            errs.append("real clip must have generator='human'")
        if self.label == "fake" and self.generator in ("", "human"):
            errs.append("fake clip must name its generator")
        if self.label == "fake" and self.synthesis_paradigm == "n/a":
            errs.append("fake clip needs a synthesis_paradigm")
        if not (MIN_STAGED_S <= float(self.duration_s) <= MAX_STAGED_S):
            errs.append(f"duration {self.duration_s} outside [{MIN_STAGED_S},{MAX_STAGED_S}]")
        if int(self.native_sample_rate_hz) <= 0:
            errs.append("native_sample_rate_hz missing")
        if self.test_only not in ("0", "1"):
            errs.append("test_only must be '0' or '1'")
        return errs


class StagingWriter:
    """Accumulates staged clips for one source and writes staged.csv + STATS.json."""

    def __init__(self, source_dir: Path, source_name: str, fresh: bool = True):
        self.source_dir = Path(source_dir)
        self.source_name = source_name
        self.clips_dir = self.source_dir / "staged" / "clips"
        self.csv_path = self.source_dir / "staged" / "staged.csv"
        if fresh and (self.source_dir / "staged").exists():
            shutil.rmtree(self.source_dir / "staged")
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[StagedClip] = []
        self.skipped: dict[str, int] = {}

    def skip(self, reason: str):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def add(self, clip: StagedClip) -> bool:
        errs = clip.validate()
        if errs:
            self.skip("invalid:" + errs[0].split(" ")[0])
            # remove the orphaned wav so staged/ stays consistent
            p = self.source_dir / clip.staged_path
            if p.exists():
                p.unlink()
            return False
        self.rows.append(clip)
        return True

    def next_clip_path(self, idx: int) -> Path:
        return self.clips_dir / f"{self.source_name}_{idx:06d}.wav"

    def finish(self, extra_stats: dict | None = None) -> dict:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STAGED_COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow(asdict(r))
        by_label: dict[str, int] = {}
        by_lang: dict[str, int] = {}
        by_gen: dict[str, int] = {}
        total_s = 0.0
        for r in self.rows:
            by_label[r.label] = by_label.get(r.label, 0) + 1
            by_lang[r.language] = by_lang.get(r.language, 0) + 1
            by_gen[r.generator] = by_gen.get(r.generator, 0) + 1
            total_s += float(r.duration_s)
        stats = {
            "source": self.source_name,
            "clips": len(self.rows),
            "hours": round(total_s / 3600.0, 2),
            "by_label": by_label,
            "by_language": dict(sorted(by_lang.items(), key=lambda kv: -kv[1])),
            "by_generator": dict(sorted(by_gen.items(), key=lambda kv: -kv[1])),
            "skipped": self.skipped,
        }
        if extra_stats:
            stats.update(extra_stats)
        with open(self.source_dir / "STATS.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return stats


def load_staged_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = set(STAGED_COLUMNS) - set(rows[0].keys() if rows else STAGED_COLUMNS)
    if missing:
        raise ValueError(f"{path}: missing staged columns {sorted(missing)}")
    return rows
