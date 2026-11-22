"""T1: inject exactly one defect into a real Short (or leave it clean) and write exact ground truth.

Each injector takes (source analysis, rng, target frame) and returns an `Injection`: an edit `Plan`
plus the `TruthDefect` in the OUTPUT timeline. Nothing mutates the decoded source.
"""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from vwatch.dataset import _render as R
from vwatch.dataset._common import FPS, SPF, assign_splits, finish_item, item_rng, scale_spec, video_out
from vwatch.dataset._encode import write_video_stream
from vwatch.dataset._source import (
    DEFAULT_SOURCES, Overlay, Plan, SourceClip, SourceSpec, cut_mask, identity_plan, iter_plan_frames,
    load_source, pick_nearest, rolling_all,
)
from vwatch.schemas import DatasetItem, DefectTruth, DefectType, Task, TruthDefect

MARGIN = int(1.5 * FPS)  # keep defects out of the first/last 1.5 s
SATURATED = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255))
BLOCK_COLOURS = ((230, 40, 60), (40, 120, 230), (250, 200, 30), (40, 170, 90), (150, 70, 200), (255, 255, 255),
                 (20, 20, 20), (255, 120, 20))
STICKERS = ("SALE", "HOT", "NEW", "OFFER", "WOW")


@dataclass(frozen=True)
class Injection:
    plan: Plan
    defect: TruthDefect


def _span(src: SourceClip) -> tuple[int, int]:
    return MARGIN, src.n - MARGIN


def _frame_change(src: SourceClip, i: int, j: int) -> float:
    return float(np.abs(src.gray[i].astype(np.int16) - src.gray[j].astype(np.int16)).mean())


def _defect(kind: DefectType, start: float, end: float, detail: str) -> TruthDefect:
    return TruthDefect(type=kind, start=round(start, 4), end=round(end, 4), detail=detail)


# ------------------------------------------------------------------ 1. flash_frame


def _foreign_frame(src: SourceClip, rng: np.random.Generator, s: int, n: int) -> int | None:
    """A frame >= 5 s away that looks clearly different from the frames it replaces."""
    far = np.flatnonzero((np.abs(np.arange(src.n) - s) >= 5 * FPS) & (np.arange(src.n) + n <= src.n))
    if len(far) == 0:
        return None
    for j in rng.choice(far, size=min(40, len(far)), replace=False):
        if all(_frame_change(src, int(j) + i, s + i) >= 10.0 for i in range(n)):
            return int(j)
    return None


def inject_flash_frame(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    n = int(rng.integers(1, 4))
    mode = str(rng.choice(["black", "colour", "foreign"]))
    near = cut_mask(src.n, src.cuts, FPS // 2)
    lo, hi = _span(src)
    s = pick_nearest([(i, 0.0) for i in range(lo, hi - n) if not near[i:i + n].any()], target)
    idx = np.arange(src.n)
    solid: dict[int, tuple[int, int, int]] = {}
    if mode == "foreign":
        j = _foreign_frame(src, rng, s, n)
        if j is None:
            mode = "colour"
        else:
            idx[s:s + n] = np.arange(j, j + n)
    if mode == "black":
        solid = {s + i: (0, 0, 0) for i in range(n)}
    elif mode == "colour":
        rgb = SATURATED[int(rng.integers(len(SATURATED)))]
        solid = {s + i: rgb for i in range(n)}
    plan = Plan(idx, src.audio, solid)
    return Injection(plan, _defect(DefectType.FLASH_FRAME, s / FPS, (s + n) / FPS, f"{n} frame(s) of {mode}"))


# ------------------------------------------------------------------ 2. frozen_frame


def inject_frozen_frame(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    dur = int(rng.integers(15, 61))  # 0.5 - 2.0 s
    floor = max(float(np.median(src.diff)), 0.05)
    moving = np.concatenate([[0], np.cumsum(src.diff > floor)])
    capchg = np.concatenate([[0], np.cumsum(src.cap_diff > 1.0)])
    lo, hi = _span(src)
    cands: list[tuple[int, float]] = []
    for s in range(lo, hi - dur, 2):
        if _frame_change(src, s, s + dur - 1) < 1.0:  # freezing would be invisible
            continue
        frac = (moving[s + dur] - moving[s + 1]) / (dur - 1)
        cost = (0.0 if frac >= 0.6 else 300.0) + (0.0 if capchg[s + dur] - capchg[s] > 0 else 90.0)
        cands.append((s, cost))
    s = pick_nearest(cands, target)
    idx = np.arange(src.n)
    idx[s:s + dur] = s
    return Injection(Plan(idx, src.audio), _defect(DefectType.FROZEN_FRAME, s / FPS, (s + dur) / FPS,
                                                    f"first frame held for {dur / FPS:.2f}s"))


# ------------------------------------------------------------------ 3. jump_cut


def inject_jump_cut(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    k = int(rng.integers(9, 19))  # 0.3 - 0.6 s
    near = cut_mask(src.n, src.cuts, FPS)
    lo, hi = _span(src)
    cands: list[tuple[int, float]] = []
    for a in range(lo, hi - k):
        if near[a] or near[a + k]:
            continue
        if src.window_speech(a / FPS - 0.3, a / FPS) < 1.0 or src.window_speech((a + k) / FPS, (a + k) / FPS + 0.3) < 1.0:
            continue
        cands.append((a, 0.0 if _frame_change(src, a - 1, a + k) >= 0.5 else 150.0))
    a = pick_nearest(cands, target)
    idx = np.concatenate([np.arange(a), np.arange(a + k, src.n)])
    audio = np.concatenate([src.audio[: a * SPF], src.audio[(a + k) * SPF:]])
    return Injection(Plan(idx, audio), _defect(DefectType.JUMP_CUT, a / FPS, a / FPS, f"removed {k / FPS:.2f}s"))


# ------------------------------------------------------------------ 4. audio_dropout


def inject_audio_dropout(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    dur = int(rng.integers(40, 201))  # 10 ms units: 0.4 - 2.0 s
    sp = src.speech
    ctx = rolling_all(sp, 30)  # ctx[i]: speech for 0.3 s starting at i
    csum = np.concatenate([[0], np.cumsum(sp)])
    lo, hi = MARGIN * 100 // FPS, len(sp) - MARGIN * 100 // FPS - dur
    cands = [(i, 0.0) for i in range(max(lo, 30), hi)
             if i + dur < len(ctx) and ctx[i - 30] and ctx[i + dur] and (csum[i + dur] - csum[i]) >= 0.7 * dur]
    i = pick_nearest(cands, target * 100 // FPS)
    a = src.audio.copy()
    s, e, f = i * 441, (i + dur) * 441, 441  # 10 ms fades
    a[s - f:s] *= np.linspace(1, 0, f, dtype=np.float32)
    a[s:e] = 0
    a[e:e + f] *= np.linspace(0, 1, f, dtype=np.float32)
    return Injection(Plan(np.arange(src.n), a), _defect(DefectType.AUDIO_DROPOUT, i / 100, (i + dur) / 100,
                                                        f"silent for {dur / 100:.2f}s"))


# ------------------------------------------------------------------ 5. caption_blocked


def _block_sprite(w: int, h: int, rgb: tuple[int, int, int], radius: int, label: str | None) -> np.ndarray:
    img = R.rounded_box(w, h, rgb, radius)
    if label:
        f = R.font(max(8, int(h * 0.55)))
        d = ImageDraw.Draw(img)
        l, t, r, b = d.textbbox((0, 0), label, font=f)
        fg = (255, 255, 255) if sum(rgb) < 380 else (20, 20, 20)
        d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), label, font=f, fill=fg + (255,))
    return np.asarray(img).copy()


def inject_caption_blocked(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    dur = int(rng.integers(30, 91))  # 1 - 3 s
    on = np.concatenate([[0], np.cumsum(src.cap_on)])
    lo, hi = _span(src)
    s = pick_nearest([(i, 0.0) for i in range(lo, hi - dur) if (on[i + dur] - on[i]) >= 0.85 * dur], target)
    h, w = src.frames.shape[1:3]
    y0, y1 = src.spec.caption_band
    bw, bh = int(0.94 * w), max(4, int((y1 - y0) * h))
    rgb = BLOCK_COLOURS[int(rng.integers(len(BLOCK_COLOURS)))]
    radius = int(rng.choice([0, 0.2, 0.4]) * bh)
    label = STICKERS[int(rng.integers(len(STICKERS)))] if rng.random() < 0.5 else None
    sprite = _block_sprite(bw, bh, rgb, radius, label)
    ov = Overlay(s, s + dur, sprite, w / 2, (y0 + y1) / 2 * h)
    detail = f"opaque rgb{rgb} box{' labelled ' + label if label else ''} over caption band y={y0:.2f}-{y1:.2f}"
    return Injection(Plan(np.arange(src.n), src.audio, overlays=(ov,)),
                     _defect(DefectType.CAPTION_BLOCKED, s / FPS, (s + dur) / FPS, detail))


# ------------------------------------------------------------------ 6. wrong_number


def caption_numbers(props_path: Path | None, t0: float, t1: float) -> set[str]:
    """Digit tokens captioned in [t0, t1] according to a Shorts-pipeline props.json (empty if none)."""
    if props_path is None or not props_path.exists():
        return set()
    out: set[str] = set()
    base = 0.0
    for scene in json.loads(props_path.read_text(encoding="utf-8")).get("scenes", []):
        for cap in scene.get("captions", []):
            if base + cap["endSec"] >= t0 - 1.5 and base + cap["startSec"] <= t1 + 1.5:
                out.update(re.findall(r"\d+", cap["text"]))
        base += float(scene["durationSec"])
    return out


def inject_wrong_number(src: SourceClip, rng: np.random.Generator, target: int) -> Injection:
    dur = int(rng.integers(60, 121))  # 2 - 4 s
    lo, hi = _span(src)
    cands = [(i, 0.0) for i in range(lo, hi - dur) if src.window_speech(i / FPS, (i + dur) / FPS) >= 0.7]
    s = pick_nearest(cands, target)
    taken = caption_numbers(src.spec.props_path, s / FPS, (s + dur) / FPS)
    number = next(n for n in (int(rng.integers(50, 500)) * 2 + 1 for _ in range(100)) if str(n) not in taken)
    h, w = src.frames.shape[1:3]
    text, fpath = R.rupee_label(number)
    sprite = R.text_sprite(text, max(10, int(w * 0.17)), (255, 255, 255), (200, 30, 50), font_path=fpath,
                           pad=max(4, int(w * 0.03)), radius=max(4, int(w * 0.04)))
    cx, cy = w * float(rng.uniform(0.35, 0.65)), h * float(rng.uniform(0.30, 0.50))
    plan = Plan(np.arange(src.n), src.audio, overlays=(Overlay(s, s + dur, sprite, cx, cy),))
    return Injection(plan, _defect(DefectType.WRONG_NUMBER, s / FPS, (s + dur) / FPS, f"shown {number}"))


INJECTORS: dict[DefectType, Callable[[SourceClip, np.random.Generator, int], Injection]] = {
    DefectType.FLASH_FRAME: inject_flash_frame,
    DefectType.FROZEN_FRAME: inject_frozen_frame,
    DefectType.JUMP_CUT: inject_jump_cut,
    DefectType.AUDIO_DROPOUT: inject_audio_dropout,
    DefectType.CAPTION_BLOCKED: inject_caption_blocked,
    DefectType.WRONG_NUMBER: inject_wrong_number,
}
DEFECT_ORDER = tuple(DefectType)


# ------------------------------------------------------------------ building


@dataclass(frozen=True)
class T1Job:
    video_id: str
    source: int
    defect: DefectType | None
    k: int
    frac: float  # target position within the usable range, 0..1 (stratified)


def plan_t1_jobs(scale: str, seed: int, n_sources: int) -> list[T1Job]:
    spec = scale_spec(scale)
    jobs: list[T1Job] = []
    for ti, kind in enumerate(DEFECT_ORDER):
        n = spec.t1_per_type
        rng = item_rng(seed, 1, ti, 777)
        fracs = (rng.permutation(n) + rng.uniform(0.15, 0.85, n)) / n
        for k in range(n):
            jobs.append(T1Job(f"t1_{kind.value}_{k:03d}", (k + ti) % n_sources, kind, k, float(fracs[k])))
    for k in range(spec.t1_clean):
        jobs.append(T1Job(f"t1_clean_{k:03d}", k % n_sources, None, k, 0.0))
    return jobs


def make_clean(src: SourceClip, rng: np.random.Generator) -> Plan:
    """Untouched re-encode, with a random 0-3 s trimmed off the start so clean copies differ."""
    trim = int(rng.integers(0, min(3 * FPS, src.n // 4) + 1))
    return identity_plan(src.n, src.audio, trim)


def render_job(src: SourceClip, job: T1Job, seed: int) -> tuple[Plan, tuple[TruthDefect, ...]]:
    if job.defect is None:
        return make_clean(src, item_rng(seed, 1, 99, job.k)), ()
    rng = item_rng(seed, 1, DEFECT_ORDER.index(job.defect), job.k)
    lo, hi = _span(src)
    target = int(lo + job.frac * (hi - lo - 1))
    inj = INJECTORS[job.defect](src, rng, target)
    return inj.plan, (inj.defect,)


def write_job(root: Path, src: SourceClip, job: T1Job, seed: int) -> DatasetItem:
    plan, defects = render_job(src, job, seed)
    h, w = src.frames.shape[1:3]
    out = video_out(root, Task.DEFECTS, job.video_id)
    write_video_stream(iter_plan_frames(src.frames, plan), FPS, out, size=(w, h), audio=plan.audio)
    truth = DefectTruth(video_id=job.video_id, source=src.spec.name, duration=plan.n_frames / FPS, defects=defects)
    return finish_item(root, Task.DEFECTS, job.video_id, truth, job.defect.value if job.defect else "clean")


def build_t1(root: Path, scale: str, seed: int, sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
             loader: Callable[[SourceSpec], SourceClip] = load_source) -> list[DatasetItem]:
    """Build all T1 videos + truth files; returns manifest items in job order with splits assigned."""
    jobs = plan_t1_jobs(scale, seed, len(sources))
    done: dict[str, DatasetItem] = {}
    for si, spec in enumerate(sources):
        mine = [j for j in jobs if j.source == si]
        if not mine:
            continue
        src = loader(spec)
        for job in mine:
            done[job.video_id] = write_job(root, src, job, seed)
            print(f"[t1] {job.video_id} ({spec.name})", flush=True)
        del src
        gc.collect()
    splits = assign_splits([j.video_id for j in jobs], seed, Task.DEFECTS)
    return [done[j.video_id].model_copy(update={"split": splits[j.video_id]}) for j in jobs]
