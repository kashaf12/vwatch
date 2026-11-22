"""Cheap, model-free perception: the "retina" layer.

Computes frame-difference and audio-loudness tracks and turns them into candidate moments
worth a closer look. These are hints for the agent, not verdicts: a natural scene cut and a
jump cut can look alike, and so do a sentence pause and an audio dropout. The freeze and jump-cut
detectors, and the extra tracks they need, live in `vwatch.detectors`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from vwatch import detectors, media

SIGNAL_WIDTH = 64
HOP_S = 0.02  # audio loudness hop
SPEECH_DB = -38.0  # above this = sound present
SILENCE_DB = -55.0  # below this = (near) digital silence
KINDS = ("flash", "freeze", "jump_cut", "cut", "cut_in_speech", "silence_in_speech", "local_change")


@dataclass(frozen=True)
class Candidate:
    t0: float
    t1: float
    kind: str  # flash | freeze | jump_cut | cut | cut_in_speech | silence_in_speech | local_change
    score: float  # 0..1, how unusual
    note: str = ""

    @property
    def center(self) -> float:
        return (self.t0 + self.t1) / 2


@dataclass(frozen=True)
class SignalTrack:
    fps: float
    duration: float
    diff: np.ndarray  # diff[i] = mean |frame i - frame i-1| (0..255); diff[0] = 0
    local: np.ndarray  # local[i] = max cell diff / (global diff + 1): >1 means a localized change
    rms_db: np.ndarray  # loudness per HOP_S window
    frames: np.ndarray  # tiny gray frames (N, h, w), kept for flash checks
    # Richer tracks for vwatch.detectors; None = not computed (the freeze check then trusts the 64 px
    # diff alone, and no jump cuts are reported without the audio seams).
    cell_max: np.ndarray | None = None  # mid-res: largest 8x8-cell mean abs diff to the previous frame
    seam_ratio: np.ndarray | None = None  # per frame boundary: audio transient vs its +-80 ms median
    seam_peak: np.ndarray | None = None  # per frame boundary: audio transient vs the strongest within +-40 ms

    def speech_at(self, t: float) -> bool:
        i = int(t / HOP_S)
        return 0 <= i < len(self.rms_db) and self.rms_db[i] > SPEECH_DB


def frame_diffs(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(gray) < 2:
        z = np.zeros(len(gray), dtype=np.float32)
        return z, z.copy()
    g = gray.astype(np.int16)
    d = np.abs(np.diff(g, axis=0)).astype(np.float32)  # (N-1, h, w)
    glob = np.concatenate([[0.0], d.mean(axis=(1, 2))])
    n, h, w = d.shape
    ch, cw = max(1, h // 8), max(1, w // 4)
    cells = d[:, : ch * 8, : cw * 4].reshape(n, 8, ch, 4, cw).mean(axis=(2, 4)).reshape(n, -1)
    local = np.concatenate([[0.0], cells.max(axis=1) / (d.mean(axis=(1, 2)) + 1.0)])
    return glob.astype(np.float32), local.astype(np.float32)


def loudness_db(audio: np.ndarray, sr: int, hop_s: float = HOP_S) -> np.ndarray:
    hop = max(1, int(sr * hop_s))
    n = len(audio) // hop
    if n == 0:
        return np.full(0, -120.0, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return (20 * np.log10(rms + 1e-6)).astype(np.float32)


def compute(path: str | Path) -> SignalTrack:
    gray, info = media.decode_all(path, SIGNAL_WIDTH, gray=True)
    diff, local = frame_diffs(gray)
    mid, _ = media.decode_all(path, detectors.MID_WIDTH, gray=True)
    cell_max = detectors.cell_motion(mid)[: len(diff)]
    del mid
    sr = 16000
    audio = media.extract_audio(path, sr=sr)
    hi = media.extract_audio(path, sr=detectors.SEAM_SR)
    seam_ratio, seam_peak = detectors.audio_seams(hi, detectors.SEAM_SR, info.fps, len(diff))
    return SignalTrack(fps=info.fps, duration=info.duration, diff=diff, local=local,
                       rms_db=loudness_db(audio, sr), frames=gray, cell_max=cell_max,
                       seam_ratio=seam_ratio, seam_peak=seam_peak)


# ------------------------------------------------------------------ detectors


def _spike_threshold(diff: np.ndarray) -> float:
    active = diff[1:]
    if active.size == 0:
        return float("inf")
    return max(12.0, float(np.median(active)) * 6, float(np.percentile(active, 99)) * 0.5)


def detect_flashes(track: SignalTrack, max_len: int = 4) -> list[Candidate]:
    """Short excursions: picture jumps away then returns to (almost) where it was."""
    thr = _spike_threshold(track.diff)
    out: list[Candidate] = []
    n = len(track.diff)
    i = 1
    while i < n:
        if track.diff[i] >= thr:
            for k in range(1, max_len + 1):
                j = i + k
                if j >= n:
                    break
                before, after = track.frames[i - 1].astype(np.int16), track.frames[j].astype(np.int16)
                back = float(np.abs(after - before).mean())
                if track.diff[j] >= thr * 0.5 and back < thr * 0.5:
                    t0, t1 = i / track.fps, j / track.fps
                    out.append(Candidate(t0, t1, "flash", min(1.0, 0.7 + 0.3 * (k <= 3)),
                                         f"{k} frame(s) differ from both neighbours"))
                    i = j
                    break
        i += 1
    return out


def detect_cuts(track: SignalTrack, flashes: list[Candidate]) -> list[Candidate]:
    thr = _spike_threshold(track.diff)
    flash_frames = {int(round(c.t0 * track.fps)) for c in flashes} | {
        int(round(c.t1 * track.fps)) for c in flashes}
    out = []
    for i in np.nonzero(track.diff >= thr)[0]:
        if int(i) in flash_frames or i == 0:
            continue
        t = i / track.fps
        in_speech = track.speech_at(t - 0.1) and track.speech_at(t + 0.1)
        kind = "cut_in_speech" if in_speech else "cut"
        out.append(Candidate(t, t, kind, 0.75 if in_speech else 0.35,
                             "picture changes abruptly" + (" while sound is on" if in_speech else "")))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def detect_freezes(track: SignalTrack, min_s: float = 0.4) -> list[Candidate]:
    """Legacy freeze rule: picture perfectly still while sound continues, between stretches of motion.

    Kept for reference and its tests; `find_candidates` uses `vwatch.detectors.detect_freezes`, which
    also checks stillness at mid resolution and how the hold ends (a gentle restart is a natural pause).
    """
    still = track.diff < 0.05
    still[0] = False
    moving_level = float(np.median(track.diff[1:])) if len(track.diff) > 1 else 0.0
    out = []
    for a, b in _runs(still):
        dur = (b - a) / track.fps
        if dur < min_s:
            continue
        t0, t1 = a / track.fps, b / track.fps
        talking = np.mean([track.speech_at(t) for t in np.arange(t0, t1, 0.1)]) > 0.6
        pre = track.diff[max(1, a - 5):a]
        post = track.diff[b:b + 5]
        floor = max(0.5, 0.5 * moving_level)
        motion_around = (pre.size and pre.mean() > floor) or (post.size and post.mean() > floor)
        if talking and motion_around:
            out.append(Candidate(t0, t1, "freeze", 0.8, f"picture still for {dur:.2f}s while sound continues"))
    return out


def detect_silences(track: SignalTrack, min_s: float = 0.3) -> list[Candidate]:
    """Near-silence with sound on both sides."""
    silent = track.rms_db < SILENCE_DB
    out = []
    hops_ctx = int(0.4 / HOP_S)
    for a, b in _runs(silent):
        dur = (b - a) * HOP_S
        if dur < min_s:
            continue
        before = track.rms_db[max(0, a - hops_ctx):a]
        after = track.rms_db[b:b + hops_ctx]
        if before.size and after.size and before.max() > SPEECH_DB and after.max() > SPEECH_DB:
            out.append(Candidate(a * HOP_S, b * HOP_S, "silence_in_speech", min(0.9, 0.4 + dur / 2),
                                 f"{dur:.2f}s of silence between sounds"))
    return out


def detect_local_changes(track: SignalTrack, ratio: float = 4.0) -> list[Candidate]:
    """Something changes in one region while the rest of the picture stays put (overlays, captions)."""
    mask = (track.local > ratio) & (track.diff > 0.5)
    out = []
    for a, b in _runs(mask):
        t0 = a / track.fps
        out.append(Candidate(t0, b / track.fps, "local_change", 0.3, "change confined to one region"))
    return out


def merge(cands: list[Candidate], gap: float = 0.25) -> list[Candidate]:
    """Merge same-kind candidates closer than `gap`; keep the higher score."""
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.kind, c.t0)):
        if out and out[-1].kind == c.kind and c.t0 - out[-1].t1 <= gap:
            last = out[-1]
            out[-1] = replace(last, t1=max(last.t1, c.t1), score=max(last.score, c.score))
        else:
            out.append(c)
    return sorted(out, key=lambda c: c.t0)


def find_candidates(track: SignalTrack) -> list[Candidate]:
    flashes = detect_flashes(track)
    cands = flashes + detect_cuts(track, flashes) + detectors.detect_freezes(track) + detect_silences(track)
    cands += detect_local_changes(track) + detectors.detect_jump_cuts(track)
    return merge(cands)


def describe(cands: list[Candidate]) -> str:
    from vwatch.imaging import fmt_time

    if not cands:
        return "(no unusual moments detected by the low-level signals)"
    return "\n".join(f"- {fmt_time(c.t0)}–{fmt_time(c.t1)} {c.kind} (score {c.score:.2f}): {c.note}"
                     for c in cands)
