#!/usr/bin/env python3
"""build/condition.py — gate 2: the UNIFORM conditioning pass.

Applied byte-identically to every staged clip of both classes so that no
processing artifact separates real from fake (the central anti-leakage measure of
the dataset contract):

    1. resample to 16 kHz mono
    2. VAD-style edge trim (leading/trailing silence below -45 dBFS, 150 ms kept)
    3. linear loudness normalization to -20 LUFS integrated (gain capped so
       true peak <= -1.5 dBFS; linear gain only — no dynamic compression)
    4. shared codec round-trip: MP3 64 kbps mono (both classes, identically)
    5. segmentation to <= 30.0 s (equal chunks; chunks < 2.0 s dropped)
    6. final encode WAV PCM16 16 kHz mono

Outputs build/work/conditioned/<source>/<clip>__sNN.wav plus conditioned.csv with
every staged column + measured fields (duration_s, loudness_lufs, peak_dbfs,
leading/trailing_silence_ms, vad_speech_fraction, measured_bandwidth_hz, sha256).
audio/ is NOT touched by this script.

CONDITIONING_VERSION identifies these exact parameters in dataset_card.json.

Usage: .venv/bin/python build/condition.py [--sources a,b,c] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

BUILD_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BUILD_DIR.parent
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import load_staged_csv, STAGED_COLUMNS  # noqa: E402

WORK = BUILD_DIR / "work"
OUT = WORK / "conditioned"

CONDITIONING_VERSION = "cond_v1"
PARAMS = {
    "sample_rate": 16000, "channels": 1, "format": "wav_pcm16",
    "edge_trim_threshold_db": -45.0, "edge_trim_keep_ms": 150,
    "loudness_target_lufs": -20.0, "true_peak_max_dbfs": -1.5,
    "codec_roundtrip": "libmp3lame_64k_mono_16k",
    "max_segment_s": 30.0, "min_segment_s": 2.0,
}

MEASURED_COLUMNS = ["cond_path", "segment_index", "final_duration_s", "loudness_lufs",
                    "peak_dbfs", "leading_silence_ms", "trailing_silence_ms",
                    "vad_speech_fraction", "measured_bandwidth_hz", "sha256"]


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}...: {r.stderr[-300:]}")
    return r.stderr  # ffmpeg writes reports to stderr


def measure_lufs(path: Path) -> float:
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-af", "ebur128=framelog=quiet", "-f", "null", "-"])
    m = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", err)
    return float(m[-1]) if m else float("nan")


def edge_silence_and_vad(x: np.ndarray, sr: int, thr_db: float = -45.0):
    """Frame-based (30 ms) energy analysis: leading/trailing silence + speech frac."""
    frame = int(sr * 0.03)
    if len(x) < frame:
        return 0.0, 0.0, 0.0
    n = len(x) // frame
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    active = db > thr_db
    if not active.any():
        return len(x) / sr * 1000, len(x) / sr * 1000, 0.0
    first, last = int(np.argmax(active)), int(n - 1 - np.argmax(active[::-1]))
    lead_ms = first * 30.0
    trail_ms = (n - 1 - last) * 30.0
    return lead_ms, trail_ms, float(active.mean())


def measured_bandwidth(x: np.ndarray, sr: int) -> float:
    """Highest frequency whose long-term spectrum is within 50 dB of the peak band."""
    if len(x) < sr // 2:
        return float(sr / 2)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    smooth = np.convolve(spec, np.ones(32) / 32, mode="same")
    db = 20 * np.log10(smooth + 1e-12)
    mask = db > (db.max() - 50.0)
    return float(freqs[mask][-1]) if mask.any() else float(sr / 2)


def condition_one(task: tuple[dict, str, str]) -> list[dict] | str:
    """staged row -> 1..N conditioned segment rows (or error string)."""
    row, src_dir_s, out_dir_s = task
    src_dir, out_dir = Path(src_dir_s), Path(out_dir_s)
    in_path = src_dir / row["staged_path"]
    stem = Path(row["staged_path"]).stem
    t = PARAMS
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # 1+2: resample 16k mono + edge trim
            trimmed = td / "trim.wav"
            thr = f"{t['edge_trim_threshold_db']}dB"
            keep = t["edge_trim_keep_ms"] / 1000.0
            af = (f"aresample={t['sample_rate']},"
                  f"silenceremove=start_periods=1:start_threshold={thr}:start_silence={keep},"
                  f"areverse,"
                  f"silenceremove=start_periods=1:start_threshold={thr}:start_silence={keep},"
                  f"areverse")
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path),
                  "-ac", "1", "-af", af, "-c:a", "pcm_s16le", str(trimmed)])
            x, sr = sf.read(trimmed, dtype="float64")
            if len(x) < sr * t["min_segment_s"]:
                return f"too_short_after_trim:{row['source']}/{stem}"
            # 3: linear loudness gain
            lufs = measure_lufs(trimmed)
            peak = float(np.max(np.abs(x)) + 1e-12)
            gain_db = (t["loudness_target_lufs"] - lufs) if np.isfinite(lufs) else 0.0
            headroom_db = t["true_peak_max_dbfs"] - 20 * np.log10(peak)
            gain_db = min(gain_db, headroom_db)
            x = np.clip(x * (10 ** (gain_db / 20.0)), -1.0, 1.0)
            gained = td / "gain.wav"
            sf.write(gained, x, sr, subtype="PCM_16")
            # 4: shared codec round-trip
            mp3 = td / "rt.mp3"
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(gained),
                  "-c:a", "libmp3lame", "-b:a", "64k", "-ar", str(sr), "-ac", "1", str(mp3)])
            back = td / "back.wav"
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
                  "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(back)])
            x, sr = sf.read(back, dtype="float64")
            # 5: segmentation
            max_n = int(t["max_segment_s"] * sr)
            n_seg = max(1, int(np.ceil(len(x) / max_n)))
            seg_len = int(np.ceil(len(x) / n_seg))
            out_rows = []
            for k in range(n_seg):
                seg = x[k * seg_len: (k + 1) * seg_len]
                if len(seg) < sr * t["min_segment_s"]:
                    continue
                out_path = out_dir / row["source"] / f"{stem}__s{k:02d}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                seg16 = np.clip(seg, -1.0, 1.0)
                sf.write(out_path, seg16, sr, subtype="PCM_16")
                lead, trail, vadf = edge_silence_and_vad(seg16, sr,
                                                         t["edge_trim_threshold_db"])
                pk = float(np.max(np.abs(seg16)) + 1e-12)
                seg_lufs = measure_lufs(out_path)
                digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
                out = dict(row)
                out.update({
                    "cond_path": str(out_path.relative_to(out_dir)),
                    "segment_index": k,
                    "final_duration_s": round(len(seg16) / sr, 3),
                    "loudness_lufs": round(seg_lufs, 2) if np.isfinite(seg_lufs) else "",
                    "peak_dbfs": round(20 * np.log10(pk), 2),
                    "leading_silence_ms": round(lead, 1),
                    "trailing_silence_ms": round(trail, 1),
                    "vad_speech_fraction": round(vadf, 3),
                    "measured_bandwidth_hz": round(measured_bandwidth(seg16, sr), 0),
                    "sha256": digest,
                })
                out_rows.append(out)
            return out_rows if out_rows else f"no_valid_segments:{row['source']}/{stem}"
    except Exception as e:
        return f"error:{row['source']}/{stem}:{str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="", help="comma list; default = all staged")
    ap.add_argument("--merge", action="store_true",
                    help="keep existing conditioned.csv rows for OTHER sources and "
                         "replace only the rows of the sources conditioned now")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = set(s for s in a.sources.split(",") if s)
    tasks = []
    conditioned_now: set[str] = set()
    for csv_path in sorted((DATASET_ROOT / "sources").glob("*/staged/staged.csv")):
        src_dir = csv_path.parents[1]
        if chosen and src_dir.name not in chosen:
            continue
        conditioned_now.add(src_dir.name)
        for row in load_staged_csv(csv_path):
            tasks.append((row, str(src_dir), str(OUT)))
    kept_rows: list[dict] = []
    if a.merge and (WORK / "conditioned.csv").exists():
        with open(WORK / "conditioned.csv", newline="", encoding="utf-8") as f:
            kept_rows = [r for r in csv.DictReader(f)
                         if r["source"] not in conditioned_now]
        print(f"[condition] merge: keeping {len(kept_rows)} rows from other sources")
    print(f"[condition] {len(tasks)} staged clips, version={CONDITIONING_VERSION}")
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(condition_one, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs)):
            r = fut.result()
            (results.extend(r) if isinstance(r, list) else failures.append(r))
            if (n + 1) % 500 == 0:
                print(f"  {n+1}/{len(tasks)} ({len(failures)} dropped)")
    cols = STAGED_COLUMNS + MEASURED_COLUMNS
    all_rows = kept_rows + results
    with open(WORK / "conditioned.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(all_rows)
    summary = {"conditioning_version": CONDITIONING_VERSION, "params": PARAMS,
               "input_clips": len(tasks), "output_segments": len(all_rows),
               "new_segments": len(results), "dropped": len(failures)}
    (WORK / "conditioning_summary.json").write_text(json.dumps(summary, indent=2))
    (WORK / "conditioning_drops.txt").write_text("\n".join(failures))
    print(f"[condition] done: {len(results)} segments, {len(failures)} dropped "
          f"-> {WORK/'conditioned.csv'}")


if __name__ == "__main__":
    main()
