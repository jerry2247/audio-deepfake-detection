#!/usr/bin/env python3
"""echofake — EchoFake (2025, MIT, 16 kHz mono): fake speech from modern zero-shot
TTS systems over Common Voice EN source material. We keep ONLY:
  - label = spoof (their bona fide clips are Common Voice, which we already source
    directly via sources/common_voice/)
  - non-replayed clips (replay_details empty) — physical replay is a channel
    artifact, not synthesis; out of scope for this dataset
  - modern generators (2025-era open-set + late-2024/2025 closed-set systems)

Leakage links: EchoFake spoof clips clone Common Voice speakers/utterances. We tag
cloned_source_speaker_id = cv_<hash> and source_recording_id = cv:<clip stem> in the
same namespace as sources/common_voice so the split graph keeps them together.

Usage: .venv/bin/python prep.py all [--target N]   (default 1000)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip, SEED  # noqa: E402
from common.audio import decode_bytes_to_staged_wav  # noqa: E402
from common.hf import stream_parquet_rows  # noqa: E402

SOURCE = "echofake"
REPO = "EchoFake/EchoFake"

# generator -> (paradigm, vintage). Modern systems only; everything else skipped.
MODERN = {
    "indextts":   ("ar_codec_lm", "2025"),
    "index-tts":  ("ar_codec_lm", "2025"),
    "maskgct":    ("masked_generative", "2024-10"),
    "cosyvoice2": ("flow_matching", "2024-12"),
    "cosyvoice 2": ("flow_matching", "2024-12"),
    "openaudio-s1": ("ar_codec_lm", "2025"),
    "openaudio s1": ("ar_codec_lm", "2025"),
    "fireredtts": ("ar_codec_lm", "2024-09"),
    "llasa":      ("ar_codec_lm", "2025-01"),
    "f5-tts":     ("flow_matching", "2024-10"),
    "f5tts":      ("flow_matching", "2024-10"),
}


def classify(model: str):
    m = (model or "").strip().lower()
    for key, (paradigm, vintage) in MODERN.items():
        if key in m:
            return paradigm, vintage
    return None, None


def do_stage(target: int):
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    shards = ["data/train-*.parquet", "data/open_set_eval-*.parquet"]
    for row in stream_parquet_rows(REPO, shards, limit=int(target * 30),
                                   shuffle_buffer=5000, seed=SEED):
        if i >= target:
            break
        if (row.get("label") or "").lower() == "bonafide":
            w.skip("bonafide")
            continue
        rep = row.get("replay_details") or {}
        if any(v not in (None, "", "None") for v in
               (rep.values() if isinstance(rep, dict) else [])):
            w.skip("replayed")
            continue
        syn = row.get("synthesis_details") or {}
        model = (syn.get("model") if isinstance(syn, dict) else "") or ""
        paradigm, vintage = classify(model)
        if paradigm is None:
            w.skip("legacy_or_unknown_generator")
            continue
        audio = row.get("path") or {}
        data = audio.get("bytes") if isinstance(audio, dict) else None
        if not data:
            w.skip("no_audio")
            continue
        src_name = str(row.get("source", "") or "")
        src_spk = str(row.get("source_speaker_id", "") or "")
        ref_spk = str((syn.get("reference_speaker_id") if isinstance(syn, dict) else "") or "")
        dst = w.next_clip_path(i)
        try:
            info = decode_bytes_to_staged_wav(data, ".mp3", dst)
        except Exception:
            w.skip("decode_error")
            continue
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(SRC_DIR)),
            source=SOURCE, label="fake", language="en", domain="read_speech",
            generator=model.strip(), generator_family=model.strip().lower().split()[0].split("-")[0],
            synthesis_paradigm=paradigm,
            generation_date="2025", vintage=vintage,
            cloned_source_speaker_id=(f"cv_{ref_spk or src_spk}" if (ref_spk or src_spk) else ""),
            source_recording_id=(f"cv:{Path(src_name).stem}" if src_name else ""),
            utterance_id=str(row.get("utt_id", i)),
            transcript=str(row.get("source_text", "") or ""),
            source_uri=f"hf://datasets/{REPO}",
            source_license="mit",
            codec_history="mp3_unknown",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
        ))
        if ok:
            i += 1
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) "
          f"by_gen={stats['by_generator']} skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "all"])
    ap.add_argument("--target", type=int, default=1000)
    a = ap.parse_args()
    do_stage(a.target)
