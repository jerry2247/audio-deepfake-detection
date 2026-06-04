#!/usr/bin/env python3
"""yt_real — manually supplied YouTube originals (REAL, human).

The project lead downloads verified-human YouTube videos as audio files into
raw/ (flat, one file per video) and fills in videos.csv with per-video
provenance (`prep.py template` creates the sheet). This prep cuts each long
recording into staged candidate clips at natural pauses; the central build
then conditions/segments/splits them like every other source.

Per video:
    1. one full decode to a temp mono WAV at NATIVE rate — the cutting
       master. Clips are cut from it by sample index, never by seeking into
       the MP3 (VBR seek drift would desync cuts from the analysis).
    2. 30 ms frame energy VAD (same frame size as build/condition.py) with a
       loudness-adaptive threshold: the conditioning convention of -45 dBFS
       at -20 LUFS, transposed to the video's measured integrated loudness.
    3. the first HEAD_SKIP_S and last TAIL_SKIP_S of every video are
       excluded from the analysis window entirely, so no clip can touch
       them: branded intro/outro jingles live there (confirmed by
       calibration on the actual 2026-06-04 source files).
    4. pauses = silence runs >= 0.30 s; cut points at pause midpoints;
       continuous speech longer than MAX_CLIP_S splits at its quietest frame.
    5. greedy assembly into 4.0-12.0 s candidates. The 12 s ceiling is
       deliberate: the conditioned corpus averages 6.4 s (real) / 6.5 s
       (fake) with <1% fake mass above 20 s — staging long clips here would
       make clip DURATION itself a real-vs-fake shortcut. A candidate must
       be >= 55% speech frames and contain clearly voiced energy.
    6. music QC per candidate (calibrated 2026-06-04 against the all-talk
       file as ground-truth speech): reject envelope beat periodicity
       > MAX_BEAT_STRENGTH (rhythmic music) or sustained-FFT-peak fraction
       > MAX_TONE_FRACTION (held musical notes). Laughter and speech over
       background sound PASS — they are real human vocal in real scenes.
    7. fair per-video quota (waterfill of --target across videos), selected
       by uniform stride across each video's timeline (start-to-end spread).

Leakage: every clip carries source_recording_id=yt_real:<stem>, so a whole
video lands in exactly one split. If two videos feature the same person, give
both rows the same `speaker` value in videos.csv — split.py then groups them
through speaker_id as well.

Known limitation (accepted, documented): the VAD is energy-based like the
rest of the pipeline — it cannot tell music from speech. Choose talky videos
and spot-check staged output before the build.

Usage:
    .venv/bin/python prep.py template            # create videos.csv from raw/
    .venv/bin/python prep.py all [--target N]    # validate sheet + stage
                                                 # (default target 1500)
    # `stage` is an alias for `all` (this source has no download step);
    # --allow-non-english overrides the English-only check deliberately.

No network access. raw/ is supplied manually and is never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip  # noqa: E402
from common.audio import ffprobe_info, codec_history_label  # noqa: E402

SOURCE = "yt_real"
RAW = SRC_DIR / "raw"
SHEET = SRC_DIR / "videos.csv"

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".opus", ".ogg", ".oga", ".wav",
              ".flac", ".webm"}

# --- segmentation parameters (recorded in STATS.json with every run) --------
ANALYSIS_SR = 16000          # analysis decode only; staged clips keep native rate
FRAME_S = 0.03               # frame size, matches build/condition.py VAD
BASE_SILENCE_DB = -45.0      # silence threshold at -20 LUFS (conditioning convention)
PAUSE_MIN_S = 0.30           # a silence run this long is a cuttable pause
MIN_CLIP_S = 4.0             # staged floor (final floor is 2.0 s after edge trim)
MAX_CLIP_S = 12.0            # staged ceiling — NOT the conditioner's 30 s: the
                             # corpus averages ~6.5 s per class, so longer real
                             # clips would make duration a label shortcut
                             # (measured 2026-06-04: fake >20 s is 0.5%)
SPLIT_SEARCH_FROM = 0.6      # run-on speech splits at the quietest frame in
                             # the window [0.6*MAX, MAX] from the current start
MIN_SPEECH_FRACTION = 0.55   # candidate must be at least this much speech frames
MIN_VOICED_MARGIN_DB = 15.0  # ...and contain a frame this far above threshold
HEAD_SKIP_S = 60.0           # blanked: branded intro jingles (calibrated)
TAIL_SKIP_S = 30.0           # blanked: outro/credits music (calibrated)
MAX_BEAT_STRENGTH = 0.55     # envelope autocorr peak @ 0.3-1.2 s lag; above
                             # this = rhythmic music (all-talk file maxes 0.54
                             # mid-file; jingles measured 0.59-0.86)
MAX_TONE_FRACTION = 0.10     # frames inside sustained-FFT-peak runs (>=0.4 s
                             # stable pitch = held notes; speech maxes 0.07)

SHEET_COLUMNS = ["file", "url", "channel", "speaker", "language", "domain",
                 "original_audio_verified", "notes"]
DOMAINS = {"read_speech", "audiobook", "podcast", "interview", "parliament",
           "conversational", "celebrity", "phone", "studio", "noisy_web",
           "other"}


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


def _bitrate_kbps(path: Path, duration_s: float) -> int | None:
    """Container-reported bitrate, falling back to size/duration."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=bit_rate", "-of", "json", str(path)],
                       capture_output=True, text=True)
    try:
        return max(1, round(int(json.loads(r.stdout)["format"]["bit_rate"]) / 1000))
    except Exception:
        pass
    if duration_s > 0:
        return max(1, round(path.stat().st_size * 8 / duration_s / 1000))
    return None


def discover_raw_files() -> list[Path]:
    if not RAW.is_dir():
        sys.exit(f"[{SOURCE}] {RAW} does not exist — create it and drop the MP3s there")
    return sorted(p for p in RAW.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


# --- videos.csv (provenance sheet) -------------------------------------------

def do_template():
    files = discover_raw_files()
    if not files:
        sys.exit(f"[{SOURCE}] no audio files in {RAW} — drop the MP3s there first")
    if SHEET.exists():
        sys.exit(f"[{SOURCE}] {SHEET.name} already exists — edit it, or delete it to regenerate")
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for p in files:
            w.writerow({"file": p.name, "url": "", "channel": "", "speaker": "",
                        "language": "en", "domain": "",
                        "original_audio_verified": "", "notes": ""})
    print(f"[{SOURCE}] wrote {SHEET.name} with {len(files)} rows. Fill in per row:\n"
          f"  url      — the video URL (provenance, required)\n"
          f"  channel  — channel name (required)\n"
          f"  speaker  — optional; SAME value on two rows = same person (split grouping)\n"
          f"  domain   — one of: {', '.join(sorted(DOMAINS))}\n"
          f"  original_audio_verified — set to 'yes' only after confirming the\n"
          f"      video plays its ORIGINAL human audio track (no YouTube AI auto-dub)")


def load_sheet(allow_non_english: bool) -> list[dict]:
    if not SHEET.exists():
        sys.exit(f"[{SOURCE}] {SHEET.name} missing — run `prep.py template` first")
    files = {p.name: p for p in discover_raw_files()}
    with open(SHEET, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"[{SOURCE}] {SHEET.name} has no rows")
    missing_cols = [c for c in SHEET_COLUMNS if c not in rows[0]]
    if missing_cols:
        sys.exit(f"[{SOURCE}] {SHEET.name} missing columns {missing_cols}")
    errs: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(rows, start=2):  # 1-based + header line
        name = (r.get("file") or "").strip()
        where = f"row {i} ({name or '?'})"
        if not name:
            errs.append(f"{where}: empty file");  continue
        if name in seen:
            errs.append(f"{where}: duplicate file entry")
        seen.add(name)
        if name not in files:
            errs.append(f"{where}: not found in raw/")
        if not (r.get("url") or "").strip().startswith(("http://", "https://")):
            errs.append(f"{where}: url must be the video URL")
        if not (r.get("channel") or "").strip():
            errs.append(f"{where}: channel is required")
        lang = (r.get("language") or "").strip().lower()
        if not re.fullmatch(r"[a-z]{2}|und", lang):
            errs.append(f"{where}: language must be ISO 639-1 (e.g. en)")
        elif lang != "en" and not allow_non_english:
            errs.append(f"{where}: language={lang} but the dataset is English by "
                        f"decision — pass --allow-non-english to override deliberately")
        if (r.get("domain") or "").strip() not in DOMAINS:
            errs.append(f"{where}: domain must be one of {sorted(DOMAINS)}")
        if (r.get("original_audio_verified") or "").strip().lower() != "yes":
            errs.append(f"{where}: original_audio_verified must be 'yes' — confirm the "
                        f"ORIGINAL human audio track (no AI auto-dub) before staging")
    for name in sorted(set(files) - seen):
        errs.append(f"raw/{name}: present in raw/ but has no row in {SHEET.name}")
    if errs:
        sys.exit(f"[{SOURCE}] {SHEET.name} failed validation:\n  - "
                 + "\n  - ".join(errs))
    return rows


# --- segmentation -------------------------------------------------------------

def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame RMS in dBFS over 30 ms frames (same scheme as condition.py)."""
    frame = int(sr * FRAME_S)
    n = len(x) // frame
    if n == 0:
        return np.empty(0)
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame).astype(np.float64) ** 2,
                          axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def adaptive_threshold(lufs: float) -> float:
    """-45 dBFS at -20 LUFS, transposed to this video's loudness: a frame that
    would sit below -45 after the conditioner's gain to -20 LUFS is silence
    here too. Clamped to a sane window for broken loudness measurements."""
    if not np.isfinite(lufs):
        return BASE_SILENCE_DB
    return float(np.clip(lufs - 25.0, -70.0, -35.0))


def find_candidates(db: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """Frame-domain segmentation -> [(start_frame, end_frame, speech_fraction)].

    Cut points sit at pause midpoints (silence runs >= PAUSE_MIN_S); run-on
    speech splits at its quietest frame; dead-air blocks are dropped whole;
    contiguous pieces merge greedily up to MAX_CLIP_S; candidates outside
    [MIN_CLIP_S, MAX_CLIP_S] or below the speech-quality floor are rejected."""
    pause_f = int(round(PAUSE_MIN_S / FRAME_S))
    max_f = int(round(MAX_CLIP_S / FRAME_S))
    min_f = int(np.ceil(MIN_CLIP_S / FRAME_S))
    n = len(db)
    if n < min_f:
        return []
    active = db > thr
    # silence runs -> cut points at their midpoints
    a = active.astype(np.int8)
    d = np.diff(a)
    sil_starts = np.flatnonzero(d == -1) + 1
    sil_ends = np.flatnonzero(d == 1) + 1
    if not active[0]:
        sil_starts = np.r_[0, sil_starts]
    if not active[-1]:
        sil_ends = np.r_[sil_ends, n]
    cuts = [int((s + e) // 2) for s, e in zip(sil_starts, sil_ends)
            if e - s >= pause_f]
    edges = [0] + cuts + [n]
    pieces: list[tuple[int, int]] = []
    for s, e in zip(edges, edges[1:]):
        if e <= s or not active[s:e].any():
            continue  # dead-air block — drop entirely
        while e - s > max_f:  # run-on speech: cut at the quietest frame
            w0 = s + int(max_f * SPLIT_SEARCH_FROM)
            cut = w0 + int(np.argmin(db[w0: s + max_f]))
            pieces.append((s, cut))
            s = cut
        pieces.append((s, e))
    if not pieces:
        return []
    merged: list[tuple[int, int]] = []
    cs, ce = pieces[0]
    for s, e in pieces[1:]:
        if s == ce and e - cs <= max_f:  # contiguous and still fits
            ce = e
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    out: list[tuple[int, int, float]] = []
    for s, e in merged:
        if not (min_f <= e - s <= max_f):
            continue
        frac = float(active[s:e].mean())
        if frac < MIN_SPEECH_FRACTION:
            continue
        if float(db[s:e].max()) < thr + MIN_VOICED_MARGIN_DB:
            continue
        out.append((s, e, frac))
    return out


# --- music QC (calibrated 2026-06-04 on the actual source files) ---------------

def beat_strength(env_seg: np.ndarray) -> float:
    """Envelope autocorrelation peak in the 0.3-1.2 s lag range. Rhythmic
    music shows a strong tempo peak; speech does not (syllables are not
    periodic at these lags)."""
    e = env_seg - env_seg.mean()
    if len(e) < 80 or not e.any():
        return 0.0
    # near-constant envelope: autocorr ratio is numerically unstable AND such
    # drones are the tone filter's job — beat only applies to modulated audio
    if float(e.std()) < 0.05 * float(env_seg.mean() + 1e-12):
        return 0.0
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac /= (ac[0] + 1e-12)
    lo = int(0.3 / FRAME_S)
    hi = min(int(1.2 / FRAME_S), len(ac) - 1)
    return float(ac[lo:hi].max()) if hi > lo else 0.0


def tonal_run_fraction(x: np.ndarray, sr: int) -> float:
    """Fraction of 30 ms frames sitting inside a sustained-pitch run: the
    dominant FFT peak stays within +/-1 bin for >= 0.4 s with >= 8x median
    prominence. Held musical notes do this; speech pitch glides do not."""
    frame = int(sr * FRAME_S)
    n = len(x) // frame
    if n < 14:
        return 0.0
    fr = x[: n * frame].reshape(n, frame) * np.hanning(frame)
    spec = np.abs(np.fft.rfft(fr, n=2048, axis=1)) + 1e-12
    spec = spec[:, 5:]  # drop DC/rumble bins
    peak_bin = spec.argmax(axis=1)
    prominent = spec.max(axis=1) > 8 * np.median(spec, axis=1)
    run, runs_frames = 0, 0
    for i in range(1, n):
        if prominent[i] and abs(int(peak_bin[i]) - int(peak_bin[i - 1])) <= 1:
            run += 1
            if run >= int(0.4 / FRAME_S):
                runs_frames += 1
        else:
            run = 0
    return float(runs_frames / n)


def music_qc(x: np.ndarray, sr: int, env: np.ndarray,
             cands: list[tuple[int, int, float]]):
    """Split candidates into (kept, {reject_reason: count})."""
    kept: list[tuple[int, int, float]] = []
    rejected = {"qc_beat": 0, "qc_tone": 0}
    frame = int(sr * FRAME_S)
    for s, e, frac in cands:
        if beat_strength(env[s:e]) > MAX_BEAT_STRENGTH:
            rejected["qc_beat"] += 1
            continue
        if tonal_run_fraction(x[s * frame: e * frame], sr) > MAX_TONE_FRACTION:
            rejected["qc_tone"] += 1
            continue
        kept.append((s, e, frac))
    return kept, rejected


# --- quota and selection --------------------------------------------------------

def fair_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    """Waterfill: every video gets as close to target/n as its yield allows;
    spare capacity in rich videos absorbs the shortfall of thin ones.
    Deterministic (name tie-breaks); never allocates more than available."""
    alloc = {k: 0 for k in counts}
    remaining = target
    pool = sorted(counts, key=lambda k: (counts[k], k))  # thinnest first
    for i, k in enumerate(pool):
        share = remaining // (len(pool) - i)
        take = min(counts[k], share)
        alloc[k] = take
        remaining -= take
    if remaining > 0:  # integer-division leftovers -> most spare first
        for k in sorted(counts, key=lambda k: (alloc[k] - counts[k], k)):
            extra = min(counts[k] - alloc[k], remaining)
            alloc[k] += extra
            remaining -= extra
            if remaining == 0:
                break
    return alloc


def stride_select(items: list, k: int) -> list:
    """k items spread uniformly across a time-ordered list (deterministic)."""
    if k >= len(items):
        return list(items)
    return [items[(i * len(items)) // k] for i in range(k)]


# --- staging --------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def do_stage(rows: list[dict], target: int):
    if target <= 0:
        sys.exit(f"[{SOURCE}] --target must be positive")
    rows = sorted(rows, key=lambda r: r["file"].strip())

    # pass 1 — analyze every video (cheap 16 kHz decode), collect candidates
    videos = []
    for r in rows:
        src = RAW / r["file"].strip()
        try:
            info = ffprobe_info(src)
        except Exception as e:
            sys.exit(f"[{SOURCE}] {src.name}: unreadable ({e}) — fix or remove it")
        with tempfile.TemporaryDirectory() as td:
            aw = Path(td) / "analysis.wav"
            _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                  "-i", str(src), "-ac", "1", "-ar", str(ANALYSIS_SR),
                  "-map", "a:0", "-c:a", "pcm_s16le", str(aw)])
            lufs = measure_lufs(aw)
            x, _ = sf.read(aw, dtype="float32")
        db = frame_db(x, ANALYSIS_SR)
        thr = adaptive_threshold(lufs)
        # exclude intro/outro jingle zones BY CONSTRUCTION: the analysis
        # window is trimmed, so no candidate (and no audio the music QC has
        # not seen) can extend into them. Blanking instead of trimming lets
        # midpoint cuts reach back into the zone — found by the staging
        # verification battery, hence this approach.
        head_f = int(HEAD_SKIP_S / FRAME_S)
        tail_f = int(TAIL_SKIP_S / FRAME_S)
        if len(db) <= head_f + tail_f + int(MIN_CLIP_S / FRAME_S):
            head_f, tail_f = 0, 0  # degenerate short input: no skip
        frame_a = int(ANALYSIS_SR * FRAME_S)
        end_f = len(db) - tail_f
        db_use = db[head_f:end_f]
        x_use = x[head_f * frame_a: end_f * frame_a]
        speech_pct = round(100 * float((db_use > thr).mean()), 1) if len(db_use) else 0.0
        raw_cands = find_candidates(db_use, thr)
        env = 10.0 ** (db_use / 20.0)
        cands, qc_rej = music_qc(x_use, ANALYSIS_SR, env, raw_cands)
        videos.append({"row": r, "src": src, "stem": src.stem, "info": info,
                       "lufs": lufs, "speech_pct": speech_pct, "cands": cands,
                       "qc_rejected": qc_rej, "off_f": head_f})
        print(f"[{SOURCE}] {src.name}: {info['duration_s'] / 60:.1f} min, "
              f"{lufs:.1f} LUFS, speech {speech_pct:.0f}%, "
              f"{len(raw_cands)} candidates -> {len(cands)} after music QC "
              f"(rejected {qc_rej['qc_beat']} beat / {qc_rej['qc_tone']} tone)")
        if not cands:
            print(f"[{SOURCE}]   WARNING: {src.name} yields nothing — too quiet, "
                  f"too short, or not speech-dense enough")

    alloc = fair_allocation({v["stem"]: len(v["cands"]) for v in videos}, target)

    # pass 2 — cut selected candidates from a native-rate master decode
    w = StagingWriter(SRC_DIR, SOURCE)
    idx = 0
    per_video: dict[str, dict] = {}
    for v in videos:
        r, src, stem = v["row"], v["src"], v["stem"]
        selected = stride_select(v["cands"], alloc[stem])
        staged_n = 0
        if selected:
            kbps = _bitrate_kbps(src, v["info"]["duration_s"])
            ch = codec_history_label(v["info"]["codec"], kbps)
            spk = (r.get("speaker") or "").strip()
            speaker_id = f"yt:{_slug(spk)}" if spk else ""
            with tempfile.TemporaryDirectory() as td:
                mw = Path(td) / "master.wav"
                _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(src), "-ac", "1", "-map", "a:0",
                      "-c:a", "pcm_s16le", str(mw)])
                with sf.SoundFile(str(mw)) as mf:
                    sr, total = mf.samplerate, len(mf)
                    off_f = v["off_f"]  # head-skip offset back to file time
                    for fs, fe, _frac in selected:
                        start = int(round((fs + off_f) * FRAME_S * sr))
                        end = min(int(round((fe + off_f) * FRAME_S * sr)), total)
                        if end - start < int((MIN_CLIP_S - 0.5) * sr):
                            w.skip("eof_truncated")
                            continue
                        mf.seek(start)
                        seg = mf.read(end - start, dtype="int16",
                                      always_2d=False)
                        dst = w.next_clip_path(idx)
                        sf.write(str(dst), seg, sr, subtype="PCM_16")
                        ok = w.add(StagedClip(
                            staged_path=str(dst.relative_to(SRC_DIR)),
                            source=SOURCE, label="real",
                            language=(r["language"] or "en").strip().lower(),
                            domain=r["domain"].strip(),
                            generator="human", generator_family="human",
                            synthesis_paradigm="n/a",
                            speaker_id=speaker_id,
                            source_recording_id=f"{SOURCE}:{stem}",
                            utterance_id=f"{stem}:{int((fs + off_f) * FRAME_S * 1000):08d}",
                            source_uri=(r["url"] or "").strip(),
                            source_license="research_only",
                            codec_history=ch,
                            native_sample_rate_hz=sr,
                            duration_s=round(len(seg) / sr, 3),
                            test_only="0",
                            # in_the_wild: finalize.py maps this marker to the
                            # manifest is_in_the_wild flag — raw circulating
                            # media, unlike the corpus-distributed sources
                            notes=f"in_the_wild;channel={(r['channel'] or '').strip()}"[:120],
                        ))
                        if ok:
                            idx += 1
                            staged_n += 1
        per_video[stem] = {"minutes": round(v["info"]["duration_s"] / 60, 1),
                           "speech_pct": v["speech_pct"],
                           "candidates": len(v["cands"]),
                           "qc_rejected": v["qc_rejected"],
                           "quota": alloc[stem], "staged": staged_n}
        print(f"[{SOURCE}] {src.name}: quota {alloc[stem]} -> staged {staged_n}")

    stats = w.finish({
        "target": target, "videos": len(videos), "per_video": per_video,
        "segmentation": {"analysis_sr": ANALYSIS_SR, "frame_s": FRAME_S,
                         "base_silence_db": BASE_SILENCE_DB,
                         "pause_min_s": PAUSE_MIN_S,
                         "min_clip_s": MIN_CLIP_S, "max_clip_s": MAX_CLIP_S,
                         "split_search_from": SPLIT_SEARCH_FROM,
                         "min_speech_fraction": MIN_SPEECH_FRACTION,
                         "min_voiced_margin_db": MIN_VOICED_MARGIN_DB,
                         "head_skip_s": HEAD_SKIP_S, "tail_skip_s": TAIL_SKIP_S,
                         "max_beat_strength": MAX_BEAT_STRENGTH,
                         "max_tone_fraction": MAX_TONE_FRACTION},
    })
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) from "
          f"{len(videos)} videos skipped={stats['skipped']}")
    if stats["clips"] < target:
        print(f"[{SOURCE}] SHORTFALL: staged {stats['clips']} of target {target} — "
              f"the raw material tops out here (per-video yields above); add "
              f"longer/more videos or accept a smaller yt_real share")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["template", "stage", "all"])
    ap.add_argument("--target", type=int, default=1500,
                    help="total staged-clip target across all videos (default 1500)")
    ap.add_argument("--allow-non-english", action="store_true",
                    help="permit language != en rows (dataset is English by decision)")
    a = ap.parse_args()
    if a.cmd == "template":
        do_template()
    else:  # stage / all are the same — this source has no download step
        do_stage(load_sheet(a.allow_non_english), a.target)
