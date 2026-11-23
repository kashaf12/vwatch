"""Post-render sanity checks for T1x videos: re-decode the written file and compare against the
untouched source clip and the truth, the same way the original T1 set was checked."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import subprocess

import numpy as np

from vwatch import media
from vwatch.dataset._common import FPS, SR
from vwatch.dataset._source import SourceClip
from vwatch.schemas import DefectTruth, DefectType

TOL = 3.0  # mean abs diff (0-255) between a re-encoded frame and its source frame


@dataclass(frozen=True)
class Check:
    video_id: str
    ok: bool
    detail: str


def small_rgb(frames: np.ndarray, step: int = 4, chunk: int = 64) -> np.ndarray:
    """Block-mean downscale (N, H/step, W/step, 3), uint8, matching ffmpeg's `scale=...:area`."""
    n, h, w = frames.shape[:3]
    h4, w4 = h // step, w // step
    out = np.empty((n, h4, w4, 3), np.uint8)
    for i in range(0, n, chunk):
        b = frames[i:i + chunk, :h4 * step, :w4 * step].astype(np.float32)
        out[i:i + chunk] = b.reshape(len(b), h4, step, w4, step, 3).mean((2, 4)).round().astype(np.uint8)
    return out


def decode_small(path: Path, info: media.VideoInfo, step: int = 4) -> np.ndarray:
    """Stream-decode `path` and block-mean each frame, on the same grid as `small_rgb` (never holds full frames)."""
    w, h = info.width, info.height
    h4, w4 = h // step, w // step
    out: list[np.ndarray] = []
    proc = subprocess.Popen([media.ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    assert proc.stdout is not None
    nbytes = w * h * 3
    while len(buf := proc.stdout.read(nbytes)) == nbytes:
        f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)[: h4 * step, : w4 * step].astype(np.float32)
        out.append(f.reshape(h4, step, w4, step, 3).mean((1, 3)).round().astype(np.uint8))
    proc.wait()
    return np.stack(out) if out else np.zeros((0, h4, w4, 3), np.uint8)


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _env(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // 441
    return np.sqrt((audio[: n * 441].reshape(n, 441).astype(np.float64) ** 2).mean(1)) if n else np.zeros(0)


def source_small(src: SourceClip) -> np.ndarray:
    return small_rgb(src.frames)


def verify_video(path: Path, truth: DefectTruth, src: SourceClip, src_small: np.ndarray) -> Check:
    vid = truth.video_id
    try:
        info = media.probe(path)
        out = decode_small(path, info)
        audio = media.extract_audio(path, sr=SR)
    except Exception as exc:  # corrupt output is a failed check, not a crash
        return Check(vid, False, f"decode failed: {exc}")
    errs: list[str] = []
    h, w = src.frames.shape[1:3]
    if (info.width, info.height) != (w, h) or info.width % 2 or info.height % 2:
        errs.append(f"size {info.width}x{info.height} != {w}x{h}")
    if abs(info.fps - FPS) > 0.01:
        errs.append(f"fps {info.fps}")
    if abs(len(out) / FPS - truth.duration) > 1.5 / FPS:
        errs.append(f"{len(out)} frames vs duration {truth.duration}")
    if abs(len(audio) / SR - len(out) / FPS) > 0.1:
        errs.append(f"audio {len(audio) / SR:.2f}s vs video {len(out) / FPS:.2f}s")
    if len(out) == 0 or src_small.shape[1:] != out.shape[1:]:
        return Check(vid, False, "; ".join(errs + ["frame geometry mismatch"]))
    note = "clean"
    if not truth.defects:
        trim = src.n - len(out)
        if not 0 <= trim <= 3 * FPS + 1:
            errs.append(f"clean length off by {trim} frames")
        else:
            d = np.mean([_d(out[i], src_small[i + trim]) for i in range(0, len(out), 15)])
            note = f"clean trim {trim}f, mean diff {d:.2f}"
            if d > TOL:
                errs.append(f"clean frames differ from source ({d:.2f})")
    else:
        d0 = truth.defects[0]
        s, e = round(d0.start * FPS), round(d0.end * FPS)
        env, senv = _env(audio), _env(src.audio)
        if d0.type is DefectType.FLASH_FRAME:
            spikes = [_d(out[i], out[i - 1]) for i in (s, e)]
            same = _d(out[s - 3], src_small[s - 3]) < TOL and _d(out[e + 3], src_small[e + 3]) < TOL
            inside = [_d(out[i], src_small[i]) for i in range(s, e)]
            note = f"flash {e - s}f spike in/out {spikes[0]:.0f}/{spikes[1]:.0f}, vs src {min(inside):.0f}"
            if min(inside) < 8 or min(spikes) < 8 or not same:
                errs.append(f"flash not isolated: {note}")
        elif d0.type is DefectType.FROZEN_FRAME:
            run = max(_d(out[i], out[i - 1]) for i in range(s + 1, e))
            hidden = _d(out[e - 1], src_small[e - 1])
            back = _d(out[e + 2], src_small[e + 2])
            note = f"frozen {e - s}f max consecutive diff {run:.2f}, hides motion {hidden:.1f}, resumes {back:.1f}"
            if run > 1.0 or hidden < 0.8 or back > TOL:
                errs.append(f"freeze check failed: {note}")
            if len(audio) and len(src.audio) and abs(_env(audio[: len(src.audio)])[: len(senv)].mean() - senv.mean()) > 0.02:
                errs.append("audio changed under freeze")
        elif d0.type is DefectType.JUMP_CUT:
            k = src.n - len(out)
            before, after = _d(out[s - 1], src_small[s - 1]), _d(out[s], src_small[s + k])
            note = f"jump {k}f ({k / FPS:.2f}s) shorter; seam frames match source {before:.1f}/{after:.1f}"
            if not 9 <= k <= 18 or max(before, after) > TOL:
                errs.append(f"jump cut mismatch: {note}")
            if abs((src.n - len(out)) / FPS - (len(src.audio) - len(audio)) / SR) > 0.1:
                errs.append("audio not shortened by the removed length")
        elif d0.type is DefectType.AUDIO_DROPOUT:
            a, b = round(d0.start * 100), round(d0.end * 100)  # round: 12.34*100 floors to 1233
            inside = env[a + 3:b - 3].max() if b - a > 8 else 1.0
            sp_in = float(src.speech[a:b].mean())  # source speech mask: speech where the hole is
            sides = {}
            for name, sl in (("before", slice(a - 30, a - 3)), ("after", slice(b + 3, b + 30))):
                kept = float(env[sl].mean() / max(senv[sl].mean(), 1e-9))  # output loudness vs source
                sides[name] = (bool(src.speech[sl].all()), kept)
            note = (f"dropout {(b - a) / 100:.2f}s: inside max {inside:.4f}, source speech in hole {sp_in:.0%}, "
                    + ", ".join(f"{k} speech={v[0]} kept={v[1]:.2f}" for k, v in sides.items()))
            if inside > 0.003 or sp_in < 0.7 or any(not v[0] or v[1] < 0.5 for v in sides.values()):
                errs.append(f"dropout check failed: {note}")
        else:
            errs.append(f"unexpected defect type {d0.type}")
    return Check(vid, not errs, "; ".join(errs) if errs else note)
