"""Hugging Face download helpers shared by source preps.

All downloads use the user's stored HF token automatically (huggingface_hub reads
~/.cache/huggingface/token). Gated repos used by this project (mueller91/MLAAD,
amphion/Emilia-Dataset) were verified accessible with the stored token on
2026-06-04."""

from __future__ import annotations

import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

_api = HfApi()


def list_dir(repo_id: str, path: str, recursive: bool = False) -> list:
    """List entries under a path of a dataset repo."""
    return list(_api.list_repo_tree(repo_id, path_in_repo=path,
                                    repo_type="dataset", recursive=recursive))


def list_files(repo_id: str, path: str, suffixes: tuple[str, ...] = (),
               recursive: bool = True) -> list[str]:
    out = []
    for e in list_dir(repo_id, path, recursive=recursive):
        p = e.path
        if getattr(e, "size", None) is None:  # directory
            continue
        if not suffixes or p.lower().endswith(suffixes):
            out.append(p)
    return sorted(out)


def download(repo_id: str, filename: str, raw_dir: Path) -> Path:
    """Download one file into raw_dir, preserving repo-relative layout. Resumable
    and idempotent (huggingface_hub caches + we keep a local copy)."""
    local = raw_dir / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    cached = hf_hub_download(repo_id, filename, repo_type="dataset")
    local.parent.mkdir(parents=True, exist_ok=True)
    # hard-link when possible to avoid duplicating disk; fall back to copy
    import os, shutil
    try:
        os.link(cached, local)
    except OSError:
        shutil.copy2(cached, local)
    return local


def sample_deterministic(items: list, n: int, seed: int) -> list:
    """Stable shuffle-and-take used by every prep (sorted input -> seeded sample)."""
    items = sorted(items)
    rng = random.Random(seed)
    if n >= len(items):
        return items
    return rng.sample(items, n)


def shuffled_shards(repo_id: str, path: str, seed: int) -> list[str]:
    """Resolve parquet shards under a repo path and return them in seeded-shuffled
    order — defeats recording/speaker-contiguous shard layouts that a streaming
    shuffle buffer cannot escape."""
    shards = list_files(repo_id, path, suffixes=(".parquet",))
    rng = random.Random(seed)
    rng.shuffle(shards)
    return shards


def stream_parquet_rows(repo_id: str, data_files, limit: int | None = None,
                        shuffle_buffer: int | None = None, seed: int = 0,
                        audio_columns: tuple[str, ...] = ("audio",)):
    """Stream rows from parquet shards of a dataset repo without downloading
    everything. data_files: glob(s) relative to the repo, e.g. 'clean/test-*.parquet'.

    Audio columns are cast to decode=False so rows carry raw {'bytes', 'path'}
    dicts — we decode with ffmpeg ourselves (avoids the datasets-4.x torchcodec
    dependency entirely)."""
    from datasets import load_dataset, Audio

    ds = load_dataset("parquet",
                      data_files={"x": [f"hf://datasets/{repo_id}/{g}" for g in
                                        ([data_files] if isinstance(data_files, str) else data_files)]},
                      split="x", streaming=True)
    # Cast EVERY Audio-typed feature to decode=False (raw bytes). Detect from the
    # features schema; fall back to the provided column names if features are
    # unresolved in streaming mode.
    feat = getattr(ds, "features", None)
    audio_like = ([name for name, f in feat.items() if type(f).__name__ == "Audio"]
                  if feat else list(audio_columns))
    for col in audio_like:
        try:
            ds = ds.cast_column(col, Audio(decode=False))
        except Exception:
            pass
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    count = 0
    for row in ds:
        yield row
        count += 1
        if limit is not None and count >= limit:
            break
