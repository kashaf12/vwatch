"""Real-video sources for T1/T4: decode once at 540 px, analyse (motion, cuts, speech, captions),
and describe edits as a `Plan` so a defect video is streamed to ffmpeg without copying the source."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vwatch.dataset._common import FPS, HEIGHT, SPF, SR, WIDTH
from vwatch.media import decode_all, extract_audio

# Folder holding the four source Shorts (not redistributed). Point VWATCH_SHORTS_DIR at your own copy.
STREET_OUT = Path(os.environ.get("VWATCH_SHORTS_DIR", "eval/sources/shorts"))
SPEECH_FRAME = 441  # samples per 10 ms at 44.1 kHz


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    # Caption fields default to "no burned-in captions" (generic sources, e.g. T1x Commons clips).
    caption_band: tuple[float, float] = (0.0, 0.0)  # y range (fraction of height) covered by a caption_blocked box
    detect_band: tuple[float, float] = (0.0, 0.0)  # y range scanned for "is a caption visible"
    caption_kind: str = "none"  # "navy_pill" | "light_text" | "none"
    caption_thr: float = 1.0
    props_path: Path | None = None

    @property
    def has_captions(self) -> bool:
        return self.caption_kind != "none"


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("dosa-reel-DEMO", STREET_OUT / "dosa-reel-DEMO.mp4", (0.765, 0.915), (0.78, 0.85), "navy_pill", 0.12),
    SourceSpec("dosa-demo-stick-story", STREET_OUT / "dosa-demo-stick-story.mp4", (0.835, 0.925), (0.845, 0.915),
               "light_text", 0.03),
    SourceSpec("dosa-stall-bengaluru-2026-11-13T18-39-54",
               STREET_OUT / "dosa-stall-bengaluru-2026-11-13T18-39-54" / "video.mp4", (0.80, 0.885), (0.80, 0.885),
               "light_text", 0.012, STREET_OUT / "dosa-stall-bengaluru-2026-11-13T18-39-54" / "props.json"),
    SourceSpec("dosa-stall-bengaluru-2026-11-13T18-15-12",
               STREET_OUT / "dosa-stall-bengaluru-2026-11-13T18-15-12" / "video.mp4", (0.80, 0.885), (0.80, 0.885),
               "light_text", 0.012),
)


# ------------------------------------------------------------------ plans


@dataclass(frozen=True)
class Overlay:
    start: int  # output frame range [start, end)
    end: int
    sprite: np.ndarray  # RGBA
    cx: float
    cy: float


@dataclass(frozen=True)
class Plan:
    """Output = frames[src_idx[i]] (or a solid colour / foreign frame), with overlays, plus audio."""

    src_idx: np.ndarray
    audio: np.ndarray
    solid: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    overlays: tuple[Overlay, ...] = ()

    @property
    def n_frames(self) -> int:
        return int(len(self.src_idx))


def identity_plan(n_frames: int, audio: np.ndarray, start: int = 0) -> Plan:
    return Plan(np.arange(start, n_frames), audio[start * SPF:])


def iter_plan_frames(frames: np.ndarray, plan: Plan) -> Iterator[np.ndarray]:
    from vwatch.dataset._render import paste  # local: keeps _render optional for analysis-only users

    h, w = frames.shape[1:3]
    for i, j in enumerate(plan.src_idx):
        if i in plan.solid:
            yield np.broadcast_to(np.asarray(plan.solid[i], np.uint8), (h, w, 3)).copy()
            continue
        f = frames[j]
        active = [o for o in plan.overlays if o.start <= i < o.end]
        if active:
            f = f.copy()
            for o in active:
                paste(f, o.sprite, o.cx, o.cy)
        yield f


# ------------------------------------------------------------------ analysis


@dataclass(frozen=True)
class SourceClip:
    spec: SourceSpec
    frames: np.ndarray  # (N, H, W, 3) uint8, never mutated
    audio: np.ndarray  # float32 mono at SR, exactly N * SPF samples
    gray: np.ndarray  # (N, H/4, W/4) uint8
    diff: np.ndarray  # (N,) mean abs gray diff to previous frame (0 for frame 0)
    cuts: np.ndarray  # frame indices of scene cuts
    speech: np.ndarray  # bool per 10 ms
    cap_on: np.ndarray  # (N,) bool caption visible
    cap_diff: np.ndarray  # (N,) mean abs gray diff inside the caption band

    @property
    def n(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        return self.n / FPS

    @property
    def near_cut(self) -> np.ndarray:
        """Bool per frame: within 1 s of a scene cut (used by jump_cut); flash uses 0.5 s."""
        return cut_mask(self.n, self.cuts, FPS)

    def window_speech(self, t0: float, t1: float) -> float:
        """Fraction of [t0, t1) seconds with speech."""
        a, b = max(0, int(round(t0 * 100))), min(len(self.speech), int(round(t1 * 100)))
        if b <= a:
            return 0.0
        return float(self.speech[a:b].mean())


def cut_mask(n: int, cuts: np.ndarray, radius: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for c in cuts:
        m[max(0, int(c) - radius): min(n, int(c) + radius + 1)] = True
    return m


def gray_small(frames: np.ndarray, step: int = 4, chunk: int = 64) -> np.ndarray:
    out = np.empty((len(frames), -(-frames.shape[1] // step), -(-frames.shape[2] // step)), np.uint8)  # ceil: odd sizes
    for i in range(0, len(frames), chunk):
        out[i:i + chunk] = frames[i:i + chunk, ::step, ::step].astype(np.float32).mean(-1).round().astype(np.uint8)
    return out


def frame_diffs(gray: np.ndarray) -> np.ndarray:
    d = np.zeros(len(gray), np.float32)
    for i in range(1, len(gray)):
        d[i] = np.abs(gray[i].astype(np.int16) - gray[i - 1].astype(np.int16)).mean()
    return d


def detect_cuts(diff: np.ndarray) -> np.ndarray:
    """Large frame-diff spikes (consecutive spike frames collapse to the first)."""
    if len(diff) < 3:
        return np.zeros(0, int)
    thr = max(12.0, 2.2 * float(np.percentile(diff, 90)))
    idx = np.flatnonzero(diff > thr)
    keep = [int(i) for k, i in enumerate(idx) if k == 0 or i - idx[k - 1] > 2]
    return np.asarray(keep, dtype=int)


def close_mask(mask: np.ndarray, gap: int) -> np.ndarray:
    """Binary closing: bridge False gaps of up to 2*gap samples."""
    k = np.ones(2 * gap + 1)
    dil = np.convolve(mask.astype(float), k, mode="same") > 0
    return np.convolve(dil.astype(float), k, mode="same") >= len(k) - 1e-9


def speech_mask(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // SPEECH_FRAME
    if n == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((audio[: n * SPEECH_FRAME].reshape(n, SPEECH_FRAME) ** 2).mean(1))
    thr = max(0.01, 0.2 * float(np.percentile(rms, 90)))
    return close_mask(rms > thr, 7)


def rolling_all(mask: np.ndarray, w: int) -> np.ndarray:
    """out[i] = mask[i:i+w].all(); length len(mask)-w+1 (empty if w > len)."""
    if w > len(mask):
        return np.zeros(0, dtype=bool)
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    return (c[w:] - c[:-w]) == w


def caption_presence(frames: np.ndarray, spec: SourceSpec, chunk: int = 32) -> np.ndarray:
    if not spec.has_captions:
        return np.zeros(len(frames), bool)
    h = frames.shape[1]
    y0, y1 = int(spec.detect_band[0] * h), int(spec.detect_band[1] * h)
    out = np.zeros(len(frames), bool)
    for i in range(0, len(frames), chunk):
        b = frames[i:i + chunk, y0:y1].astype(np.int16)
        if spec.caption_kind == "navy_pill":
            hit = (abs(b[..., 0] - 48) < 35) & (abs(b[..., 1] - 72) < 35) & (abs(b[..., 2] - 148) < 40)
        elif spec.caption_kind == "light_text":
            hit = (b.min(-1) > 225) | ((b[..., 0] > 220) & (b[..., 1] > 140) & (b[..., 1] < 205) & (b[..., 2] < 110))
        else:
            raise ValueError(f"unknown caption_kind {spec.caption_kind!r}")
        out[i:i + chunk] = hit.mean((1, 2)) > spec.caption_thr
    return out


def band_diff(gray: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    h = gray.shape[1]
    y0, y1 = int(band[0] * h), max(int(band[0] * h) + 1, int(band[1] * h))
    d = np.zeros(len(gray), np.float32)
    for i in range(1, len(gray)):
        d[i] = np.abs(gray[i, y0:y1].astype(np.int16) - gray[i - 1, y0:y1].astype(np.int16)).mean()
    return d


def fit_audio(audio: np.ndarray, n_frames: int) -> np.ndarray:
    want = n_frames * SPF
    if len(audio) >= want:
        return audio[:want].astype(np.float32)
    return np.concatenate([audio, np.zeros(want - len(audio), np.float32)]).astype(np.float32)


def load_source(spec: SourceSpec) -> SourceClip:
    if not spec.path.exists():
        raise FileNotFoundError(f"source video missing: {spec.path}")
    frames, info = decode_all(spec.path, WIDTH)
    if abs(info.fps - FPS) > 0.01:
        raise ValueError(f"{spec.path}: fps {info.fps}, expected {FPS}")
    if not spec.has_captions:  # generic source: any aspect ratio, but 540 px wide with even dimensions
        h, w = frames.shape[1:3]
        if w != WIDTH or h % 2:
            raise ValueError(f"{spec.path}: decoded {w}x{h}, expected {WIDTH} px wide with even height")
    elif frames.shape[1:3] != (HEIGHT, WIDTH):
        raise ValueError(f"{spec.path}: decoded {frames.shape[2]}x{frames.shape[1]}, expected {WIDTH}x{HEIGHT} (9:16)")
    audio = fit_audio(extract_audio(spec.path, sr=SR), len(frames))
    return analyse(spec, frames, audio)


def analyse(spec: SourceSpec, frames: np.ndarray, audio: np.ndarray) -> SourceClip:
    gray = gray_small(frames)
    diff = frame_diffs(gray)
    return SourceClip(
        spec=spec, frames=frames, audio=audio, gray=gray, diff=diff, cuts=detect_cuts(diff),
        speech=speech_mask(audio), cap_on=caption_presence(frames, spec),
        cap_diff=band_diff(gray, spec.detect_band) if spec.has_captions else np.zeros(len(gray), np.float32),
    )


def pick_nearest(cands: list[tuple[int, float]], target: int) -> int:
    """cands = [(index, extra_cost)]; returns the index minimising |index-target| + cost."""
    if not cands:
        raise ValueError("no valid placement found for this defect in this source")
    return min(cands, key=lambda c: abs(c[0] - target) + c[1])[0]
