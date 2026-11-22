"""T2: procedural event clips. A calm scene (gradient, static "furniture", one slowly drifting
element) with 8-12 non-overlapping events whose durations are spread over the recall buckets."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vwatch.dataset import _render as R
from vwatch.dataset._common import FPS, HEIGHT, SR, WIDTH, finish_item, item_rng, silent_audio, video_out
from vwatch.dataset._encode import write_video_stream
from vwatch.schemas import DatasetItem, EventTruth, Task, TruthEvent

BUCKET_FRAMES = {"tiny": (2, 6), "short": (6, 14), "mid": (16, 29), "long": (32, 88), "xlong": (93, 135)}
GAP_S = 1.55  # minimum gap between events and from the clip edges (sampled gap >= 1.5 s after rounding)
FURNITURE_SLOTS = ((120, 250), (420, 330), (140, 700), (410, 780))
DRIFT_Y = (440, 540)


@dataclass(frozen=True)
class Furniture:
    shape: str
    colour: str
    cx: int
    cy: int
    size: int


@dataclass(frozen=True)
class EventSpec:
    kind: str  # appear | color_change | move | flash
    start: int  # first frame the event is visible
    n: int  # frames
    colour: str  # appear/move/flash colour, or the NEW colour for color_change
    shape: str  # shape (appear/move), furniture shape (color_change), "flash"
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    size: int = 0
    furniture: int = -1  # index for color_change


@dataclass(frozen=True)
class EventScene:
    n_frames: int
    furniture: tuple[Furniture, ...]
    drift_shape: str
    drift_colour: str
    drift_phase: tuple[float, float]
    events: tuple[EventSpec, ...]

    @property
    def duration(self) -> float:
        return self.n_frames / FPS


def sample_buckets(rng: np.random.Generator, n: int) -> list[str]:
    base = ["tiny", "tiny", "short", "mid", "mid", "long", "long", "xlong"]
    extra_pool = ["short", "mid", "long", "mid", "tiny", "xlong"]
    out = base + [extra_pool[int(rng.integers(len(extra_pool)))] for _ in range(n - len(base))]
    out = out[:n] if n < len(base) else out
    return [out[i] for i in rng.permutation(len(out))]


def _free(x: float, y: float, size: int, furniture: tuple[Furniture, ...]) -> bool:
    if not (size / 2 + 30 <= x <= WIDTH - size / 2 - 30 and size / 2 + 60 <= y <= HEIGHT - size / 2 - 60):
        return False
    if DRIFT_Y[0] - size / 2 - 30 < y < DRIFT_Y[1] + size / 2 + 30:
        return False
    return all(math.hypot(x - f.cx, y - f.cy) > (f.size + size) / 2 + 20 for f in furniture)


def _free_point(rng: np.random.Generator, size: int, furniture: tuple[Furniture, ...]) -> tuple[float, float] | None:
    for _ in range(400):
        x, y = float(rng.uniform(80, WIDTH - 80)), float(rng.uniform(100, HEIGHT - 100))
        if _free(x, y, size, furniture):
            return x, y
    return None


def _place(rng: np.random.Generator, size: int, furniture: tuple[Furniture, ...]) -> tuple[float, float, int]:
    """A free spot for a shape of about `size` px; shrinks the shape (never below 120 px) if the canvas is crowded."""
    for s in (size, 150, 135, 120):
        if s > size:
            continue
        pt = _free_point(rng, s, furniture)
        if pt is not None:
            return pt[0], pt[1], s
    raise RuntimeError("could not place an event shape on the canvas")


def plan_scene(rng: np.random.Generator) -> EventScene:
    shapes = [R.SHAPES[i] for i in rng.permutation(len(R.SHAPES))]
    n_furn = int(rng.integers(3, 5))
    furn_colours = [R.COLOUR_NAMES[i] for i in rng.permutation(len(R.COLOUR_NAMES))[:n_furn]]
    slots = [FURNITURE_SLOTS[i] for i in rng.permutation(len(FURNITURE_SLOTS))[:n_furn]]
    furniture = tuple(Furniture(shapes[i], furn_colours[i], slots[i][0], slots[i][1], int(rng.integers(140, 171)))
                      for i in range(n_furn))
    drift_shape = shapes[n_furn]
    event_shapes = shapes[n_furn + 1:]  # never collide with a furniture or drifting shape
    n = int(rng.integers(8, 13))
    for _ in range(200):
        buckets = sample_buckets(rng, n)
        durs = [int(rng.integers(BUCKET_FRAMES[b][0], BUCKET_FRAMES[b][1] + 1)) for b in buckets]
        need = GAP_S * (n + 1) + sum(durs) / FPS
        if need <= 44.5:
            break
    else:
        raise RuntimeError("could not fit events in 45 s")
    total = float(rng.uniform(max(30.0, need + 0.5), 45.0))
    slack = total - need
    gaps = np.full(n + 1, GAP_S) + rng.dirichlet(np.ones(n + 1)) * slack
    n_frames = int(round(total * FPS))
    colours = [R.COLOUR_NAMES[i] for i in rng.permutation(len(R.COLOUR_NAMES))]
    n_change = 0
    n_flash = 0
    events: list[EventSpec] = []
    t = 0.0
    used: set[tuple[str, str]] = set()
    uses = {c: 0 for c in colours}

    def pick(ok) -> str:  # least-used colour that satisfies `ok`, so colours spread over the clip
        c = min((c for c in colours if ok(c)), key=lambda c: uses[c])
        uses[c] += 1
        return c
    for i, (b, d) in enumerate(zip(buckets, durs, strict=True)):
        t += float(gaps[i])
        start = int(round(t * FPS))
        t += d / FPS
        kind = str(rng.choice(["appear", "appear", "move", "color_change", "flash"]))
        if kind == "flash" and (b not in ("tiny", "short") or n_flash >= 3):
            kind = "appear"
        if kind == "color_change" and n_change >= n_furn:
            kind = "move"
        if kind == "flash":
            n_flash += 1
            colour = pick(lambda c: (c, "flash") not in used)
            used.add((colour, "flash"))
            events.append(EventSpec("flash", start, int(rng.integers(1, 4)), colour, "flash"))
        elif kind == "color_change":
            fi = n_change
            n_change += 1
            f = furniture[fi]
            colour = pick(lambda c: c != f.colour and (c, f.shape) not in used)
            used.add((colour, f.shape))
            events.append(EventSpec("color_change", start, d, colour, f.shape, furniture=fi))
        else:
            shape = next(s for s in (event_shapes[int(rng.integers(len(event_shapes)))] for _ in range(200))
                         if any((c, s) not in used for c in colours))
            colour = pick(lambda c: (c, shape) not in used)
            used.add((colour, shape))
            x0, y0, size = _place(rng, int(rng.integers(130, 171)), furniture)
            x1, y1 = x0, y0
            if kind == "move":
                for _ in range(300):
                    cand = _free_point(rng, size, furniture)
                    if cand is not None and math.hypot(cand[0] - x0, cand[1] - y0) >= 220:
                        x1, y1 = cand
                        break
            events.append(EventSpec(kind, start, d, colour, shape, x0, y0, x1, y1, size))
    order = tuple(sorted(events, key=lambda e: e.start))
    return EventScene(n_frames, furniture, drift_shape, colours[-1], (float(rng.uniform(0, 6.28)), float(rng.uniform(0, 6.28))),
                      order)


def render_frames(scene: EventScene) -> Iterator[np.ndarray]:
    bg = R.gradient(HEIGHT, WIDTH, (92, 104, 128), (138, 148, 170))
    for f in range(scene.n_frames):
        t = f / FPS
        active = [e for e in scene.events if e.start <= f < e.start + e.n]
        flash = next((e for e in active if e.kind == "flash"), None)
        if flash is not None:
            yield np.broadcast_to(np.asarray(R.COLOURS[flash.colour], np.uint8), (HEIGHT, WIDTH, 3)).copy()
            continue
        canvas = bg.copy()
        recoloured = {e.furniture: e.colour for e in active if e.kind == "color_change"}
        for i, fu in enumerate(scene.furniture):
            R.paste(canvas, R.shape_sprite(fu.shape, R.COLOURS[recoloured.get(i, fu.colour)], fu.size), fu.cx, fu.cy)
        dx = 270 + 170 * math.sin(2 * math.pi * t / 17 + scene.drift_phase[0])
        dy = 490 + 45 * math.sin(2 * math.pi * t / 11 + scene.drift_phase[1])
        R.paste(canvas, R.shape_sprite(scene.drift_shape, R.COLOURS[scene.drift_colour], 64), dx, dy)
        for e in active:
            if e.kind == "color_change":
                continue
            u = (f - e.start + 0.5) / e.n
            R.paste(canvas, R.shape_sprite(e.shape, R.COLOURS[e.colour], e.size),
                    e.x0 + (e.x1 - e.x0) * u, e.y0 + (e.y1 - e.y0) * u)
        yield canvas


def scene_truth(video_id: str, scene: EventScene) -> EventTruth:
    events = tuple(
        TruthEvent(id=f"e{i + 1}", start=round(e.start / FPS, 4), end=round((e.start + e.n) / FPS, 4), kind=e.kind,
                   subject=f"{e.colour} {e.shape}", keywords=(e.colour, e.shape))
        for i, e in enumerate(scene.events)
    )
    return EventTruth(video_id=video_id, duration=scene.duration, events=events)


def build_t2(root: Path, count: int, seed: int) -> list[DatasetItem]:
    items = []
    for k in range(count):
        scene = plan_scene(item_rng(seed, 2, k))
        vid = f"t2_{k:03d}"
        write_video_stream(render_frames(scene), FPS, video_out(root, Task.EVENTS, vid), size=(WIDTH, HEIGHT),
                           audio=silent_audio(scene.duration), sr=SR)
        items.append(finish_item(root, Task.EVENTS, vid, scene_truth(vid, scene), "synthetic"))
        print(f"[t2] {vid} {len(scene.events)} events {scene.duration:.1f}s", flush=True)
    return items
