"""Download a Commons video, pick a speech-dense 35-60 s window, normalise it, verify with Whisper."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vwatch import media
from vwatch.dataset._commons import CommonsClient, FileInfo
from vwatch.dataset._common import FPS, SR, WIDTH
from vwatch.dataset._source import close_mask
from vwatch.transcribe import Transcript

MIN_SEG, MAX_SEG = 35.0, 60.0
MIN_COVERAGE = 0.60  # fraction of the segment covered by Whisper speech segments
MAX_TRIES = 3  # Whisper passes per source (CPU-bound: a few seconds to a minute each)
LEAD_IN = 2.0  # skip the first seconds of a file (titles, jingles)
ENERGY_SR = 16000


class SegmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommonsSource:
    short: str  # used in video ids and as DefectTruth.source
    title: str  # "File:..." on Commons
    language: str  # human label, e.g. "Hindi"
    seg_len: float = 50.0  # wanted segment length in seconds (clamped to the file)


@dataclass(frozen=True)
class Prepared:
    source: CommonsSource
    info: FileInfo
    segment_path: Path
    start: float
    duration: float
    coverage: float
    language_detected: str


def frames_for(seconds: float) -> int:
    return int(round(seconds * FPS))


def rough_speech(audio: np.ndarray, sr: int = ENERGY_SR) -> np.ndarray:
    """Bool per 10 ms: cheap energy gate (same idea as `_source.speech_mask`, at low sample rate)."""
    hop = sr // 100
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, bool)
    rms = np.sqrt((audio[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(1))
    thr = max(0.005, 0.2 * float(np.percentile(rms, 90)))
    return close_mask(rms > thr, 10)


def rank_windows(mask: np.ndarray, total_s: float, seg_len: float, *, step: float = 0.5, top: int = 6) -> list[float]:
    """Start times (s) of the best non-overlapping windows of `seg_len`, densest speech first."""
    lead = LEAD_IN if total_s - LEAD_IN >= MIN_SEG else 0.0  # a barely-long-enough clip keeps its intro
    seg_len = min(seg_len, total_s - lead)
    if seg_len < MIN_SEG:
        return []
    w = int(seg_len * 100)
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    starts = [s for s in np.arange(lead, total_s - seg_len + 1e-9, step) if int(s * 100) + w <= len(mask)]
    if not starts:
        return []
    score = {float(s): (c[int(s * 100) + w] - c[int(s * 100)]) / w for s in starts}
    chosen: list[float] = []
    for s in sorted(score, key=lambda k: -score[k]):
        if all(abs(s - o) >= seg_len for o in chosen):
            chosen.append(s)
        if len(chosen) >= top:
            break
    return chosen


def coverage(tr: Transcript, duration: float) -> float:
    """Fraction of [0, duration] covered by the union of Whisper speech segments."""
    spans = sorted((max(0.0, s.t0), min(duration, s.t1)) for s in tr.segments if s.t1 > s.t0)
    covered, cur_end = 0.0, 0.0
    for a, b in spans:
        a = max(a, cur_end)
        if b > a:
            covered += b - a
            cur_end = b
    return covered / duration if duration > 0 else 0.0


def cut_segment(raw: Path, out: Path, start: float, duration: float) -> Path:
    """540 px wide (even height), 30 fps, H.264 + AAC 44.1 kHz mono: like the existing T1 videos."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = frames_for(duration)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [media.FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(raw), "-t", f"{n / FPS:.4f}",
           "-vf", f"scale={WIDTH}:-2:flags=area,setsar=1,fps={FPS}", "-frames:v", str(n),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ar", str(SR), "-ac", "1", "-b:a", "128k", str(tmp)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise SegmentError(f"ffmpeg cut failed for {raw.name}: {r.stderr.decode('utf-8', 'replace')[-400:]}")
    tmp.replace(out)
    return out


def _meta_path(cache: Path, short: str) -> Path:
    return cache / f"{short}.segment.json"


def prepare_source(client: CommonsClient, src: CommonsSource, cache: Path,
                   transcriber: Callable[[Path], Transcript]) -> Prepared:
    """Idempotent: info + raw download + segment all cached under `cache` (= eval/data/_cache/commons)."""
    cache.mkdir(parents=True, exist_ok=True)
    info_p = cache / f"{src.short}.info.json"
    if info_p.exists():
        info = FileInfo.from_json(json.loads(info_p.read_text(encoding="utf-8")))
    else:
        info = client.info(src.title)
        info_p.write_text(json.dumps(info.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    seg_p, meta_p = cache / f"{src.short}.segment.mp4", _meta_path(cache, src.short)
    if seg_p.exists() and meta_p.exists():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        return Prepared(src, info, seg_p, m["start"], m["duration"], m["coverage"], m["language_detected"])

    raw = cache / f"{src.short}{Path(info.url).suffix.lower()}"
    if not raw.exists():
        client.download(info.url, raw)
    probe = media.probe(raw)
    audio = media.extract_audio(raw, sr=ENERGY_SR)
    seg_len = max(MIN_SEG, min(MAX_SEG, src.seg_len))
    cands = rank_windows(rough_speech(audio), probe.duration, seg_len)
    if not cands:
        raise SegmentError(f"{src.short}: file too short ({probe.duration:.1f}s) for a {MIN_SEG:.0f}+ s segment")
    best: tuple[float, float, str] = (-1.0, 0.0, "")
    for start in cands[:MAX_TRIES]:
        dur = min(seg_len, probe.duration - start - 0.1)
        dur = frames_for(dur) / FPS
        cut_segment(raw, seg_p, start, dur)
        tr = transcriber(seg_p)
        cov = coverage(tr, dur)
        if cov > best[0]:
            best = (cov, start, tr.language)
        if cov >= MIN_COVERAGE:
            meta = {"start": start, "duration": dur, "coverage": round(cov, 4), "language_detected": tr.language}
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return Prepared(src, info, seg_p, start, dur, cov, tr.language)
    seg_p.unlink(missing_ok=True)
    raise SegmentError(f"{src.short}: no window reached {MIN_COVERAGE:.0%} speech coverage (best {best[0]:.0%} "
                       f"at {best[1]:.1f}s)")
