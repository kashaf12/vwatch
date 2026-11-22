"""The watcher's tools: overview, seek (dense frames), crop (foveation), listen (audio meter + words).

Each tool renders evidence into an Evidence log and returns its token cost, so the agent loop
can enforce the budget before sending anything.
"""

from __future__ import annotations

import numpy as np

from vwatch import media
from vwatch.arms.common import Evidence, VideoContext
from vwatch.imaging import (Box, crop, estimate_image_tokens, estimate_text_tokens, fmt_time, grid,
                            resize_width, stamp)
from vwatch.signals import HOP_S

OVERVIEW_TILE_W = 135
SEEK_TILE_W = 180
CROP_SOURCE_W = 540
CROP_MAX_W = 480
GRID_COLS = 4
MAX_SEEK_FRAMES = 16
OVERVIEW_PER_GRID = 12


def _tile_h(ctx: VideoContext, w: int) -> int:
    return int(round(ctx.info.height * w / ctx.info.width))


def grid_tokens(ctx: VideoContext, n: int, tile_w: int, cols: int = GRID_COLS) -> int:
    if n <= 0:
        return 0
    cols = min(cols, n)
    rows = -(-n // cols)
    return estimate_image_tokens(cols * tile_w + (cols - 1) * 4, rows * _tile_h(ctx, tile_w) + (rows - 1) * 4)


def seek_times(t0: float, t1: float, fps: float, duration: float) -> list[float]:
    t0, t1 = max(0.0, t0), min(duration - 1e-3, t1)
    if t1 <= t0:
        return [round(t0, 3)]
    fps = max(0.5, min(fps, 30.0))
    n = int((t1 - t0) * fps) + 1
    if n > MAX_SEEK_FRAMES:
        n = MAX_SEEK_FRAMES
    return [round(t0 + k * (t1 - t0) / max(1, n - 1), 3) for k in range(n)] if n > 1 else [round(t0, 3)]


def seek_cost(ctx: VideoContext, t0: float, t1: float, fps: float) -> int:
    times = seek_times(t0, t1, fps, ctx.info.duration)
    n = len({ctx.info.frame_index(t) for t in times})  # frames_at deduplicates
    return sum(grid_tokens(ctx, min(OVERVIEW_PER_GRID, n - k), SEEK_TILE_W)
               for k in range(0, n, OVERVIEW_PER_GRID)) + 20


def _render_grids(ev: Evidence, frames: list[tuple[float, np.ndarray]], tile_w: int, per_grid: int,
                  kind: str) -> None:
    for k in range(0, len(frames), per_grid):
        chunk = frames[k:k + per_grid]
        tiles = [stamp(resize_width(f, tile_w), fmt_time(t)) for t, f in chunk]
        ev.image(grid(tiles, GRID_COLS), kind, chunk[0][0], chunk[-1][0])
        for t, _ in chunk:
            ev.observe("frame", t, t)


FULL_VIEW_OPTIONS = ((360, 8.0), (360, 6.0), (270, 6.0), (270, 4.0))  # (tile width, fps), best first
FULL_VIEW_SHARE = 0.4  # a short clip is watched whole if that fits in this share of the budget


def overview(ctx: VideoContext, ev: Evidence, fps: float, tile_w: int = OVERVIEW_TILE_W,
             *, full: bool = False) -> int:
    """Tiles of the whole video: a low-res skim, or (full=True) the whole short clip at watchable size."""
    before = ev.tokens
    n = max(2, int(ctx.info.duration * fps))
    times = [round((k + 0.5) * ctx.info.duration / n, 3) for k in range(n)]
    frames = media.frames_at(ctx.path, times, min(tile_w * 2, ctx.info.width))
    if full:
        ev.text(f"FULL VIEW: this clip is short enough to watch whole. {len(frames)} frames, one every "
                f"{ctx.info.duration / n:.2f} s, tiled left-to-right, top-to-bottom.")
    else:
        ev.text(f"OVERVIEW: {len(frames)} low-resolution frames, one every {ctx.info.duration / n:.2f} s, "
                f"tiled left-to-right, top-to-bottom.")
    _render_grids(ev, frames, tile_w, OVERVIEW_PER_GRID, "frame")
    return ev.tokens - before


def overview_cost(ctx: VideoContext, fps: float, tile_w: int = OVERVIEW_TILE_W) -> int:
    n = max(2, int(ctx.info.duration * fps))
    return sum(grid_tokens(ctx, min(OVERVIEW_PER_GRID, n - k), tile_w)
               for k in range(0, n, OVERVIEW_PER_GRID)) + 40


def full_view(ctx: VideoContext, budget: int) -> tuple[int, float] | None:
    """(tile width, fps) to watch a short clip whole, or None when it is too long for that budget.

    A person watches a five-second clip in full instead of skimming it; so does the watcher.
    """
    for tile_w, fps in FULL_VIEW_OPTIONS:
        if overview_cost(ctx, fps, tile_w) <= budget * FULL_VIEW_SHARE:
            return tile_w, fps
    return None


def seek(ctx: VideoContext, ev: Evidence, t0: float, t1: float, fps: float, why: str = "") -> int:
    before = ev.tokens
    times = seek_times(t0, t1, fps, ctx.info.duration)
    frames = media.frames_at(ctx.path, times, SEEK_TILE_W * 2)
    label = f" ({why})" if why else ""
    ev.text(f"SEEK {fmt_time(times[0])}–{fmt_time(times[-1])}{label}: {len(frames)} consecutive frames "
            f"at ~{fps:.0f} fps, tiled left-to-right, top-to-bottom.")
    _render_grids(ev, frames, SEEK_TILE_W, OVERVIEW_PER_GRID, "frame")
    return ev.tokens - before


def crop_cost(ctx: VideoContext, box: Box) -> int:
    w = int(CROP_SOURCE_W * (box.x1 - box.x0))
    h = int(_tile_h(ctx, CROP_SOURCE_W) * (box.y1 - box.y0))
    if w > CROP_MAX_W:
        h, w = int(h * CROP_MAX_W / w), CROP_MAX_W
    return estimate_image_tokens(max(1, w), max(1, h)) + 15


def crops(ctx: VideoContext, ev: Evidence, times: list[float], box: Box, name: str = "") -> int:
    before = ev.tokens
    frames = media.frames_at(ctx.path, times, CROP_SOURCE_W)
    for t, f in frames:
        c = crop(f, box)
        if c.shape[1] > CROP_MAX_W:
            c = resize_width(c, CROP_MAX_W)
        label = name or f"box {box.x0:.2f},{box.y0:.2f},{box.x1:.2f},{box.y1:.2f}"
        ev.text(f"CROP {label} at {fmt_time(t)} (full resolution):")
        ev.image(stamp(c, fmt_time(t)), "crop", t, t)
    return ev.tokens - before


def loudness_line(ctx: VideoContext, t0: float, t1: float, step: float = 0.1) -> str:
    db = ctx.track.rms_db
    if db.size == 0:
        return "(no audio)"
    vals = []
    for t in np.arange(max(0.0, t0), min(ctx.info.duration, t1) + 1e-6, step):
        i = int(t / HOP_S)
        seg = db[i:i + max(1, int(step / HOP_S))]
        vals.append("sil" if seg.size == 0 or seg.max() < -55 else f"{int(seg.max())}")
    return " ".join(vals)


def listen_text(ctx: VideoContext, t0: float, t1: float) -> str:
    return (f"LISTEN {fmt_time(t0)}–{fmt_time(t1)}: loudness in dB every 0.1 s ('sil' = silence): "
            f"{loudness_line(ctx, t0, t1)}\n  words heard (word@seconds): "
            f"{ctx.transcript.words_text(t0, t1)}")


def listen(ctx: VideoContext, ev: Evidence, t0: float, t1: float) -> int:
    before = ev.tokens
    ev.text(listen_text(ctx, t0, t1))
    ev.observe("audio", t0, t1)
    return ev.tokens - before


def listen_cost(ctx: VideoContext, t0: float, t1: float) -> int:
    return estimate_text_tokens(listen_text(ctx, t0, t1))
