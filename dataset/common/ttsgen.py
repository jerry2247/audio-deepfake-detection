"""Shared TTS generation engine for sources/openai_tts and sources/grok_tts.

Safety model (the reason this file exists):
  CREDIT SAFETY — a frozen, reviewed request matrix (requests.csv) whose SHA-256
  must match the hash approved in TTS_PLAN.md; hard per-provider character and
  dollar caps asserted before any network call; dry-run mode; smoke gate
  (1 minimal clip per voice) required before batch; any HTTP 4xx aborts the
  whole run (a 4xx means OUR request is malformed — iterating would burn the
  matrix); 429/5xx retry with exponential backoff (a failed request returns no
  audio and bills nothing), max 3 attempts, then the row is marked failed and
  skipped (never silently re-billed).
  CRASH SAFETY — write-ahead receipts: audio lands as raw/<request_id>.wav via
  tmp-file + fsync + atomic rename, THEN ledger/<request_id>.json is written the
  same way. On any restart, rows with a ledger receipt are skipped entirely. A
  crash can cost at most the single in-flight request.

No code here writes to staged/, build/, or audio/ — staging is a separate
offline step (each source's prep.py) that reads raw/ + ledger/ only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[1]

REQUEST_COLUMNS = ["request_id", "provider", "voice", "mode", "language",
                   "text_id", "register", "speed", "instructions", "text",
                   "est_chars"]

# Hard ceilings (project decision 2026-06-04): $20 per provider.
# Pricing basis: OpenAI gpt-4o-mini-tts ~$12/1M input chars equivalent;
# Grok TTS $15/1M chars. Caps below are chars at which spend would still be
# far under $20 even if pricing doubled.
PROVIDER_CAPS = {
    "openai": {"max_chars": 600_000, "max_requests": 600},
    "grok":   {"max_chars": 600_000, "max_requests": 1000},
}
PER_TEXT_CHAR_CAP = 1_200
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


def load_env() -> dict[str, str]:
    """Read dataset/.env KEY=VALUE pairs. Values are never printed or logged."""
    env: dict[str, str] = {}
    p = DATASET_ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


@dataclass
class RequestRow:
    request_id: str
    provider: str
    voice: str
    mode: str
    language: str
    text_id: str
    register: str
    speed: str          # "" or float-as-string (grok only)
    instructions: str   # openai only
    text: str           # final text sent to the API (may contain grok tags)
    est_chars: int


def request_id_for(provider: str, voice: str, mode: str, text_id: str) -> str:
    return f"{provider}_{hashlib.md5(f'{provider}|{voice}|{mode}|{text_id}'.encode()).hexdigest()[:16]}"


# ---------------------------------------------------------------- matrix I/O

def write_matrix(path: Path, rows: list[RequestRow]) -> str:
    ids = [r.request_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request_id in matrix")
    for r in rows:
        if r.est_chars > PER_TEXT_CHAR_CAP:
            raise ValueError(f"{r.request_id}: text exceeds {PER_TEXT_CHAR_CAP} chars")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    return matrix_hash(path)


def matrix_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matrix(path: Path, approved_hash: str | None) -> list[RequestRow]:
    actual = matrix_hash(path)
    if approved_hash is not None and actual != approved_hash:
        sys.exit(f"REFUSED: {path.name} hash {actual[:16]}… does not match the "
                 f"approved hash {approved_hash[:16]}…. Re-review the matrix.")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            d["est_chars"] = int(d["est_chars"])
            rows.append(RequestRow(**d))
    return rows


# ---------------------------------------------------------------- providers

class ProviderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message[:200]}")
        self.status = status


class OpenAIProvider:
    """Speech endpoint. Model verified at runtime: gpt-4o-mini-tts is the newest
    speech-endpoint model (verified against official docs 2026-06-04); if the
    live models list ever contains an unknown *-tts model, we ABORT so a human
    consciously chooses the frontier model rather than silently using an old one."""

    name = "openai"
    KNOWN_TTS_MODELS = {"gpt-4o-mini-tts", "tts-1", "tts-1-hd"}
    MODEL = "gpt-4o-mini-tts"

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def preflight(self) -> str:
        ids = {m.id for m in self.client.models.list()}
        tts_like = {i for i in ids if "tts" in i and not i.startswith("whisper")}
        unknown = {i for i in tts_like
                   if not any(i.startswith(k) for k in self.KNOWN_TTS_MODELS)}
        if unknown:
            sys.exit(f"ABORT: unknown TTS model(s) on the account: {sorted(unknown)}. "
                     "A newer model may exist — update OpenAIProvider.MODEL deliberately.")
        return self.MODEL

    def synthesize(self, row: RequestRow) -> bytes:
        kwargs = dict(model=self.MODEL, voice=row.voice, input=row.text,
                      response_format="wav")
        if row.instructions:
            kwargs["instructions"] = row.instructions
        try:
            resp = self.client.audio.speech.create(**kwargs)
            return resp.read() if hasattr(resp, "read") else resp.content
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "http_status", 0) or 0
            raise ProviderError(int(status or 0), str(exc)) from exc


class GrokProvider:
    """REST endpoint POST https://api.x.ai/v1/tts (verified against docs.x.ai
    2026-06-04): fields text, language, voice_id, speed, output_format."""

    name = "grok"
    ENDPOINT = "https://api.x.ai/v1/tts"
    VOICES_ENDPOINT = "https://api.x.ai/v1/tts/voices"
    EXPECTED_VOICES = {"eve", "ara", "rex", "sal", "leo"}

    def __init__(self, api_key: str):
        import requests
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {api_key}"

    def preflight(self) -> str:
        r = self.http.get(self.VOICES_ENDPOINT, timeout=30)
        if r.status_code != 200:
            sys.exit(f"ABORT: voice listing failed (HTTP {r.status_code}).")
        listed = {str(v.get("voice_id", v.get("id", ""))).lower()
                  for v in (r.json() if isinstance(r.json(), list)
                            else r.json().get("voices", []))}
        missing = self.EXPECTED_VOICES - listed
        if missing:
            sys.exit(f"ABORT: expected Grok voices missing: {sorted(missing)} "
                     f"(listed: {sorted(listed)}). Update the matrix deliberately.")
        return "grok-tts (resolved server-side)"

    def synthesize(self, row: RequestRow) -> bytes:
        payload = {
            "text": row.text,
            "language": row.language or "en",
            "voice_id": row.voice,
            "output_format": {"codec": "wav", "sample_rate": 24000},
        }
        if row.speed:
            payload["speed"] = float(row.speed)
        r = self.http.post(self.ENDPOINT, json=payload, timeout=120)
        if r.status_code != 200:
            raise ProviderError(r.status_code, r.text)
        return r.content


# ---------------------------------------------------------------- audio checks

def audio_sanity(path: Path, est_chars: int) -> str | None:
    """Return None if sane, else a reason string. ~15 chars/s of speech is the
    loose physical envelope; we accept 2x slack on both sides."""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=sample_rate,channels:format=duration",
               "-of", "json", str(path)]
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        dur = float(d["format"]["duration"])
        if not (0.5 <= dur <= 120):
            return f"duration_out_of_range:{dur:.1f}s"
        expected = max(est_chars / 15.0, 1.0)
        if dur < expected / 4 or dur > expected * 6:
            return f"duration_implausible:{dur:.1f}s_for_{est_chars}chars"
        # silence check
        r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect",
                            "-f", "null", "-"], capture_output=True, text=True)
        import re
        m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
        if m and float(m.group(1)) < -55:
            return f"near_silent:{m.group(1)}dB"
        return None
    except Exception as e:
        return f"probe_failed:{str(e)[:60]}"


# ---------------------------------------------------------------- runner

def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Runner:
    def __init__(self, source_dir: Path, provider, concurrency: int):
        self.source_dir = source_dir
        self.provider = provider
        self.concurrency = concurrency
        self.raw = source_dir / "raw"
        self.ledger = source_dir / "ledger"
        self.ledger.mkdir(parents=True, exist_ok=True)

    def done_ids(self) -> set[str]:
        return {p.stem for p in self.ledger.glob("*.json")}

    def pending(self, rows: list[RequestRow]) -> list[RequestRow]:
        done = self.done_ids()
        return [r for r in rows if r.request_id not in done]

    def assert_caps(self, rows: list[RequestRow]):
        caps = PROVIDER_CAPS[self.provider.name]
        chars = sum(r.est_chars for r in rows)
        if chars > caps["max_chars"]:
            sys.exit(f"ABORT: pending chars {chars:,} exceed cap {caps['max_chars']:,}")
        if len(rows) > caps["max_requests"]:
            sys.exit(f"ABORT: pending requests {len(rows)} exceed cap {caps['max_requests']}")

    def _do_one(self, row: RequestRow, model_id: str) -> tuple[str, str | None]:
        wav_path = self.raw / f"{row.request_id}.wav"
        attempt = 0
        while True:
            attempt += 1
            try:
                data = self.provider.synthesize(row)
                break
            except ProviderError as e:
                if 400 <= e.status < 500 and e.status != 429:
                    raise  # our bug — abort the whole run upstream
                if attempt >= MAX_ATTEMPTS:
                    return row.request_id, f"failed_after_{attempt}:{e.status}"
                time.sleep(min(2 ** attempt, 20))
        _atomic_write(wav_path, data)
        problem = audio_sanity(wav_path, row.est_chars)
        receipt = {
            **asdict(row),
            "model": model_id,
            "bytes": len(data),
            "audio_check": problem or "ok",
            "attempts": attempt,
            "completed_unix": int(time.time()),
        }
        _atomic_write(self.ledger / f"{row.request_id}.json",
                      json.dumps(receipt, ensure_ascii=False, indent=1).encode())
        return row.request_id, problem

    def run(self, rows: list[RequestRow], label: str) -> dict:
        pending = self.pending(rows)
        self.assert_caps(pending)
        skipped = len(rows) - len(pending)
        print(f"[{self.provider.name}] {label}: {len(pending)} to do "
              f"({skipped} already in ledger), "
              f"{sum(r.est_chars for r in pending):,} chars")
        if not pending:
            return {"done": 0, "skipped": skipped, "problems": []}
        model_id = self.provider.preflight()
        problems, fatal = [], None
        completed = 0
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(self._do_one, r, model_id): r for r in pending}
            for fut in as_completed(futs):
                try:
                    rid, problem = fut.result()
                except ProviderError as e:
                    fatal = e
                    for other in futs:
                        other.cancel()
                    break
                completed += 1
                if problem:
                    problems.append((rid, problem))
                if completed % 50 == 0:
                    print(f"  {completed}/{len(pending)}")
        if fatal is not None:
            sys.exit(f"ABORT (4xx — request malformed, nothing further sent): {fatal}")
        print(f"[{self.provider.name}] {label} complete: {completed} done, "
              f"{len(problems)} flagged: {problems[:10]}")
        return {"done": completed, "skipped": skipped, "problems": problems}


# ---------------------------------------------------------------- CLI helpers

def summarize(rows: list[RequestRow], provider: str):
    from collections import Counter
    chars = sum(r.est_chars for r in rows)
    est_cost = chars / 1_000_000 * 15.0  # conservative $15/1M chars for both
    print(f"[{provider}] {len(rows)} requests, {chars:,} chars, "
          f"≈${est_cost:.2f} (conservative)")
    print("  voices:", dict(Counter(r.voice for r in rows)))
    print("  modes:", dict(Counter(r.mode for r in rows)))
    print("  languages:", dict(Counter(r.language for r in rows)))
    print("  registers:", dict(Counter(r.register for r in rows)))


def smoke_rows(rows: list[RequestRow]) -> list[RequestRow]:
    """One shortest-text request per voice."""
    by_voice: dict[str, RequestRow] = {}
    for r in sorted(rows, key=lambda r: r.est_chars):
        by_voice.setdefault(r.voice, r)
    return list(by_voice.values())


# ---------------------------------------------------------------- matrix builder

@dataclass
class Mode:
    name: str
    registers: tuple[str, ...]          # compatible registers
    instructions: str = ""              # openai style steering
    wrap: str = ""                      # grok wrapping tag, e.g. "whisper"
    speed: str = ""                     # grok speed multiplier as string
    weight: int = 1                     # relative share among compatible modes


def load_pool() -> list[dict]:
    path = DATASET_ROOT / "sources" / "_tts_texts" / "pool.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- CLI + staging

def cli(source_dir: Path, provider_name: str, make_provider, approved_hash: str | None,
        default_concurrency: int):
    """Shared command-line entry for generate.py:
       --dry-run  cost/summary printout, zero network calls (no key needed)
       --smoke    one shortest request per voice; writes smoke_receipt.json
       --batch    full pending matrix; REQUIRES approved_hash set AND a passing
                  smoke receipt."""
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--batch", action="store_true")
    ap.add_argument("--concurrency", type=int, default=default_concurrency)
    a = ap.parse_args()

    rows = load_matrix(source_dir / "requests.csv", approved_hash)
    if a.dry_run:
        summarize(rows, provider_name)
        print(f"matrix sha256: {matrix_hash(source_dir / 'requests.csv')}")
        print("dry-run only — no network calls were made.")
        return

    env = load_env()
    key_name = {"openai": "OPENAI_API_KEY", "grok": "XAI_API_KEY"}[provider_name]
    key = env.get(key_name, "")
    if not key or key.startswith("PASTE_"):
        sys.exit(f"ABORT: {key_name} not set in dataset/.env")
    provider = make_provider(key)
    runner = Runner(source_dir, provider, a.concurrency)
    receipt_path = source_dir / "smoke_receipt.json"

    if a.smoke:
        srows = smoke_rows(rows)
        result = runner.run(srows, "smoke")
        passed = (result["done"] + result["skipped"] >= len(srows)
                  and not result["problems"])
        _atomic_write(receipt_path, json.dumps({
            "passed": passed, "voices_tested": len(srows),
            "problems": result["problems"], "unix": int(time.time()),
        }, indent=1).encode())
        print(f"smoke {'PASSED' if passed else 'FAILED'} -> {receipt_path.name}")
        if not passed:
            sys.exit(1)
        return

    # batch
    if approved_hash is None:
        sys.exit("ABORT: requests.csv has not been approved (APPROVED_HASH is None). "
                 "Review the matrix, record its sha256, then retry.")
    if not receipt_path.exists() or not json.loads(receipt_path.read_text()).get("passed"):
        sys.exit("ABORT: no passing smoke receipt. Run --smoke first.")
    runner.run(rows, "batch")


def strip_markup(text: str) -> str:
    """Remove Grok speech tags for the manifest transcript field."""
    import re
    t = re.sub(r"\[[a-z\-]+\]", " ", text)
    t = re.sub(r"</?[a-z_]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stage_generated(source_dir: Path, source_name: str, family: str,
                    license_bucket: str):
    """Offline staging: ledger receipts + raw wavs -> staged/ (no network).
    Only receipts with audio_check == 'ok' are staged; flagged ones are listed."""
    from common.staging import StagingWriter, StagedClip
    from common.audio import decode_to_staged_wav

    w = StagingWriter(source_dir, source_name)
    receipts = sorted((source_dir / "ledger").glob("*.json"))
    if not receipts:
        sys.exit(f"[{source_name}] empty ledger — run generate.py first")
    flagged = []
    i = 0
    for rp in receipts:
        r = json.loads(rp.read_text())
        if r.get("audio_check") != "ok":
            flagged.append((r["request_id"], r.get("audio_check")))
            continue
        src = source_dir / "raw" / f"{r['request_id']}.wav"
        dst = w.next_clip_path(i)
        try:
            info = decode_to_staged_wav(src, dst)
        except Exception:
            w.skip("decode_error")
            continue
        ok = w.add(StagedClip(
            staged_path=str(dst.relative_to(source_dir)),
            source=source_name, label="fake",
            language=r["language"] or "en", domain="studio",
            generator=r["model"], generator_family=family,
            generator_version=r["model"], synthesis_paradigm="unknown",
            generation_date="2026-06", vintage="2026",
            voice_id=f"{family}:{r['voice']}",
            source_recording_id=f"{source_name}:{r['request_id']}",
            utterance_id=r["request_id"],
            transcript=strip_markup(r["text"]),
            source_uri="api_generation",
            source_license=license_bucket,
            codec_history="wav",
            native_sample_rate_hz=info["sample_rate"],
            duration_s=round(info["duration_s"], 3),
            notes=f"mode={r['mode']};register={r['register']}",
        ))
        if ok:
            i += 1
    stats = w.finish({"flagged_not_staged": flagged})
    print(f"[{source_name}] staged {stats['clips']} ({stats['hours']} h) "
          f"by_lang={stats['by_language']} flagged={len(flagged)} "
          f"skipped={stats['skipped']}")


def build_matrix(provider: str, voices: list[str], modes: list[Mode],
                 target: int, seed: int,
                 exclude_registers: tuple[str, ...] = ()) -> list[RequestRow]:
    """Deterministic (seeded) assignment of (text, mode, voice):
       - every compatible text is considered in seeded-shuffled order, cycling
         the pool if target > pool size (a text may appear under 2+ modes/voices)
       - modes rotate per register, weighted; voices rotate globally so every
         voice covers every register over the run."""
    import random as _random
    rng = _random.Random(seed)
    pool = [t for t in load_pool() if t["register"] not in exclude_registers]
    by_register: dict[str, list[Mode]] = {}
    for m in modes:
        for reg in m.registers:
            by_register.setdefault(reg, []).extend([m] * m.weight)
    pool = [t for t in pool if t["register"] in by_register]
    rng.shuffle(pool)
    rows: list[RequestRow] = []
    seen_ids: set[str] = set()
    mode_counter: dict[str, int] = {}
    vi = 0
    k = 0
    while len(rows) < target and k < target * 4:
        t = pool[k % len(pool)]
        k += 1
        reg = t["register"]
        midx = mode_counter.get(reg, 0)
        mode = by_register[reg][midx % len(by_register[reg])]
        mode_counter[reg] = midx + 1
        voice = voices[vi % len(voices)]
        vi += 1
        rid = request_id_for(provider, voice, mode.name, t["text_id"])
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        text = t["text"]
        if mode.wrap:
            text = f"<{mode.wrap}>{text}</{mode.wrap}>"
        rows.append(RequestRow(
            request_id=rid, provider=provider, voice=voice, mode=mode.name,
            language=t["language"], text_id=t["text_id"], register=reg,
            speed=mode.speed, instructions=mode.instructions,
            text=text, est_chars=len(text),
        ))
    if len(rows) < target:
        raise ValueError(f"matrix underfilled: {len(rows)}/{target}")
    return rows
