#!/usr/bin/env python3
"""audeter — AUDETER (2025, updated 2026): large-scale deepfake audio over four
real-corpus domains (audiobook, celebrity, crowdsource, us_congress). We take ONLY
its modern TTS systems (cosyvoice, f5_tts, sparktts, fish_speech, zonos where
present) across all four domains — same generators, four acoustic domains. Legacy
systems (bark, chattts, vits, openvoice, ...) and the vocoder-resynthesis tree
(2020-23 vocoders) are excluded by the modern-vintage rule.

Audio lives ONLY inside multi-GB tar shards, so this prep STREAMS each remote tar
over HTTP and stops after extracting the first N audio members per (domain x
model) cell (~tens of MB per cell instead of 1.5-3.7 GB). If a tar was already
fully downloaded into raw/, it is read locally instead.

Usage: .venv/bin/python prep.py all [--per-cell N]   (default 40)
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SRC_DIR.parents[1]
sys.path.insert(0, str(DATASET_ROOT))

from common.staging import StagingWriter, StagedClip  # noqa: E402
from common.audio import decode_bytes_to_staged_wav  # noqa: E402
from common.hf import list_files  # noqa: E402

SOURCE = "audeter"
REPO = "wqz995/AUDETER"
RAW = SRC_DIR / "raw"

DOMAINS = {"audiobook": "audiobook", "celebrity": "celebrity",
           "crowdsource": "read_speech", "us_congress": "parliament"}
MODELS = {
    "cosyvoice":   ("cosyvoice", "flow_matching", "2024-12"),
    "f5_tts":      ("f5_tts", "flow_matching", "2024-10"),
    "sparktts":    ("spark_tts", "ar_codec_lm", "2025-03"),
    "fish_speech": ("openaudio", "ar_codec_lm", "2024-09"),
    "zonos":       ("zonos", "ssm_hybrid", "2025-02"),
}
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")


def first_tar_for_cell(dom: str, model: str) -> str | None:
    try:
        tars = [f for f in list_files(REPO, f"{dom}/tts/{model}", suffixes=(".tar",))
                if "/test/" in f or f.endswith(".tar")]
    except Exception:
        return None
    return sorted(tars)[0] if tars else None


def iter_tar_members(repo_file: str, n_wanted: int):
    """Yield (name, bytes) for the first n audio members, reading the tar as a
    stream — local copy if present, else HTTP streaming with early close."""
    local = RAW / repo_file
    if local.exists():
        with tarfile.open(local) as tf:
            got = 0
            for m in tf:
                if got >= n_wanted:
                    break
                if m.isfile() and m.name.lower().endswith(AUDIO_SUFFIXES):
                    f = tf.extractfile(m)
                    if f:
                        yield m.name, f.read()
                        got += 1
        return
    import requests
    from huggingface_hub import hf_hub_url, get_token
    url = hf_hub_url(REPO, repo_file, repo_type="dataset")
    headers = {}
    tok = get_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with requests.get(url, stream=True, timeout=120, headers=headers,
                      allow_redirects=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        got = 0
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for m in tf:
                if got >= n_wanted:
                    break
                if m.isfile() and m.name.lower().endswith(AUDIO_SUFFIXES):
                    f = tf.extractfile(m)
                    if f:
                        yield m.name, f.read()
                        got += 1
        # leaving the context closes the connection — only ~N members downloaded


def do_stage(per_cell: int):
    w = StagingWriter(SRC_DIR, SOURCE)
    i = 0
    for dom in DOMAINS:
        for model, (family, paradigm, vintage) in MODELS.items():
            repo_file = first_tar_for_cell(dom, model)
            if repo_file is None:
                print(f"[{SOURCE}] cell {dom}/{model}: absent, skipping")
                continue
            print(f"[{SOURCE}] cell {dom}/{model}: streaming {repo_file}", flush=True)
            try:
                members = iter_tar_members(repo_file, per_cell)
                for name, data in members:
                    dst = w.next_clip_path(i)
                    try:
                        info = decode_bytes_to_staged_wav(data, Path(name).suffix, dst)
                    except Exception:
                        w.skip("decode_error")
                        continue
                    ok = w.add(StagedClip(
                        staged_path=str(dst.relative_to(SRC_DIR)),
                        source=SOURCE, label="fake", language="en",
                        domain=DOMAINS[dom],
                        generator=f"audeter_{model}", generator_family=family,
                        synthesis_paradigm=paradigm,
                        generation_date="2025", vintage=vintage,
                        utterance_id=f"{dom}/{model}/{Path(name).stem}"[:120],
                        source_uri=f"hf://datasets/{REPO}/{repo_file}",
                        source_license="cc_by_nc_nd_4.0",
                        codec_history="wav" if name.lower().endswith(".wav") else "flac",
                        native_sample_rate_hz=info["sample_rate"],
                        duration_s=round(info["duration_s"], 3),
                        notes=f"audeter_domain={dom}",
                    ))
                    if ok:
                        i += 1
            except Exception as e:
                print(f"[{SOURCE}] cell {dom}/{model} FAILED: {str(e)[:140]}", flush=True)
    stats = w.finish()
    print(f"[{SOURCE}] staged {stats['clips']} ({stats['hours']} h) "
          f"by_gen={stats['by_generator']} skipped={stats['skipped']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "stage"])
    ap.add_argument("--per-cell", type=int, default=40)
    a = ap.parse_args()
    do_stage(a.per_cell)
