"""Mini scene graph + sprites for the T3 Rube-Goldberg clips (PIL, antialiased by supersampling)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from vwatch.dataset import _render as R
from vwatch.dataset._common import HEIGHT, WIDTH

SS = 3
FLOOR_Y = 785
Rgb = tuple[int, int, int]
ImgFn = Callable[[float], Image.Image | None]
PosFn = Callable[[float], tuple[float, float]]
AngFn = Callable[[float], float]


def _zero(_t: float) -> float:
    return 0.0


@dataclass(frozen=True)
class Actor:
    """z-ordered sprite whose image/position/angle (degrees, CCW) are functions of time in seconds."""

    z: int
    image: ImgFn
    pos: PosFn
    angle: AngFn = _zero

    def retimed(self, scale: float, shift: float) -> Actor:
        """Local time u -> global time t = shift + scale * u."""
        f = lambda t: (t - shift) / scale  # noqa: E731
        return Actor(self.z, lambda t: self.image(f(t)), lambda t: self.pos(f(t)), lambda t: self.angle(f(t)))


def const(img: Image.Image) -> ImgFn:
    return lambda _t: img


def switch(states: Sequence[tuple[float, Image.Image | None]]) -> ImgFn:
    """Image changes at the given local times; before the first entry the first image is shown."""
    ordered = sorted(states, key=lambda s: s[0])

    def fn(t: float) -> Image.Image | None:
        cur = ordered[0][1]
        for t0, img in ordered:
            if t >= t0:
                cur = img
        return cur
    return fn


def at(x: float, y: float) -> PosFn:
    return lambda _t: (x, y)


def ease_out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 2


def ease_in(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def render(actors: Sequence[Actor], bg: Image.Image, n_frames: int, fps: int, flip: bool) -> Iterator[np.ndarray]:
    ordered = sorted(actors, key=lambda a: a.z)
    for f in range(n_frames):
        t = f / fps
        canvas = bg.copy()
        for a in ordered:
            img = a.image(t)
            if img is None:
                continue
            ang = a.angle(t)
            if ang:
                img = img.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
            x, y = a.pos(t)
            canvas.paste(img, (int(round(x - img.width / 2)), int(round(y - img.height / 2))), img)
        arr = np.asarray(canvas)
        yield np.ascontiguousarray(arr[:, ::-1]) if flip else arr


# ------------------------------------------------------------------ background


def sky_bg() -> Image.Image:
    arr = R.gradient(HEIGHT, WIDTH, (186, 218, 244), (244, 238, 226))
    img = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle((0, FLOOR_Y, WIDTH, HEIGHT), fill=(168, 170, 178))
    d.rectangle((0, FLOOR_Y, WIDTH, FLOOR_Y + 5), fill=(110, 112, 122))
    return img


# ------------------------------------------------------------------ sprites


def _new(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _s(*v: float) -> tuple[float, ...]:
    return tuple(x * SS for x in v)


def _done(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.Resampling.LANCZOS)


def _a(rgb: Rgb, alpha: int = 255) -> tuple[int, int, int, int]:
    return rgb + (alpha,)


def rect_img(w: int, h: int, rgb: Rgb, radius: int = 4) -> Image.Image:
    img, d = _new(w, h)
    d.rounded_rectangle(_s(0, 0, w - 1, h - 1), radius=radius * SS, fill=_a(R.outline_for(rgb)))
    d.rounded_rectangle(_s(2, 2, w - 3, h - 3), radius=max(0, radius - 1) * SS, fill=_a(rgb))
    return _done(img, w, h)


def poly_img(w: int, h: int, pts: Sequence[tuple[float, float]], rgb: Rgb) -> Image.Image:
    img, d = _new(w, h)
    d.polygon([(x * SS, y * SS) for x, y in pts], fill=_a(R.outline_for(rgb)))
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    d.polygon([((cx + (x - cx) * 0.86) * SS, (cy + (y - cy) * 0.86) * SS) for x, y in pts], fill=_a(rgb))
    return _done(img, w, h)


def ball_img(rgb: Rgb, r: int) -> Image.Image:
    w = 2 * r
    img, d = _new(w, w)
    d.ellipse(_s(0, 0, w - 1, w - 1), fill=_a(R.outline_for(rgb)))
    d.ellipse(_s(2, 2, w - 3, w - 3), fill=_a(rgb))
    d.line(_s(r * 0.4, r * 0.4, w - r * 0.4, w - r * 0.4), fill=(255, 255, 255, 190), width=int(r * 0.3 * SS))
    return _done(img, w, w)


def box_img(rgb: Rgb, s: int) -> Image.Image:
    img, d = _new(s, s)
    d.rounded_rectangle(_s(0, 0, s - 1, s - 1), radius=8 * SS, fill=_a(R.outline_for(rgb)))
    d.rounded_rectangle(_s(3, 3, s - 4, s - 4), radius=6 * SS, fill=_a(rgb))
    d.line(_s(s * 0.5, s * 0.22, s * 0.5, s * 0.78), fill=(255, 255, 255, 170), width=4 * SS)
    d.line(_s(s * 0.22, s * 0.5, s * 0.78, s * 0.5), fill=(255, 255, 255, 170), width=4 * SS)
    return _done(img, s, s)


def button_img(rgb: Rgb, pressed: bool) -> Image.Image:
    w, h = 110, 36
    img, d = _new(w, h)
    d.rounded_rectangle(_s(0, 26, w - 1, h - 1), radius=5 * SS, fill=(70, 70, 80, 255))
    top = 16 if pressed else 4
    d.rounded_rectangle(_s(6, top, w - 7, 30), radius=7 * SS, fill=_a(R.outline_for(rgb)))
    d.rounded_rectangle(_s(9, top + 3, w - 10, 28), radius=6 * SS, fill=_a(rgb))
    return _done(img, w, h)


def lamp_img(on: bool) -> Image.Image:
    w, h = 150, 200
    img, d = _new(w, h)
    if on:
        for r, al in ((72, 50), (58, 70), (44, 90)):
            d.ellipse(_s(75 - r, 62 - r, 75 + r, 62 + r), fill=(255, 238, 120, al))
    d.rectangle(_s(71, 92, 79, 182), fill=(70, 70, 78, 255))
    d.ellipse(_s(45, 172, 105, 198), fill=(70, 70, 78, 255))
    shade = (255, 226, 80) if on else (150, 150, 156)
    pts = [(48, 30), (102, 30), (122, 94), (28, 94)]
    d.polygon([(x * SS, y * SS) for x, y in pts], fill=_a(R.outline_for(shade)))
    d.polygon([(x * SS, y * SS) for x, y in [(52, 34), (98, 34), (116, 90), (34, 90)]], fill=_a(shade))
    return _done(img, w, h)


def cannon_img(rgb: Rgb) -> Image.Image:
    w, h = 120, 70
    img, d = _new(w, h)
    d.rounded_rectangle(_s(30, 18, 118, 48), radius=6 * SS, fill=_a(R.outline_for(rgb)))
    d.rounded_rectangle(_s(33, 21, 115, 45), radius=5 * SS, fill=_a(rgb))
    d.ellipse(_s(4, 22, 58, 68), fill=(40, 40, 46, 255))
    d.ellipse(_s(18, 36, 44, 56), fill=(120, 120, 128, 255))
    return _done(img, w, h)


def muzzle_img() -> Image.Image:
    w = 80
    img, d = _new(w, w)
    pts = []
    for i in range(16):
        r = 38 if i % 2 == 0 else 15
        a = i * math.pi / 8
        pts.append(((40 + r * math.cos(a)) * SS, (40 + r * math.sin(a)) * SS))
    d.polygon(pts, fill=(255, 150, 20, 255))
    d.ellipse(_s(22, 22, 58, 58), fill=(255, 235, 120, 255))
    return _done(img, w, w)


def balloon_img(rgb: Rgb, r: int) -> Image.Image:
    w, h = 2 * r + 6, 2 * r + 46
    img, d = _new(w, h)
    d.line(_s(w / 2, 2 * r + 6, w / 2 + 5, h - 2), fill=(90, 90, 90, 255), width=2 * SS)
    d.ellipse(_s(3, 3, w - 3, 2 * r + 3), fill=_a(R.outline_for(rgb)))
    d.ellipse(_s(6, 6, w - 6, 2 * r), fill=_a(rgb))
    d.polygon([(w / 2 * SS, (2 * r + 3) * SS), ((w / 2 - 6) * SS, (2 * r + 12) * SS), ((w / 2 + 6) * SS, (2 * r + 12) * SS)],
              fill=_a(R.outline_for(rgb)))
    d.ellipse(_s(r * 0.5, r * 0.4, r * 0.9, r * 0.8), fill=(255, 255, 255, 150))
    return _done(img, w, h)


def burst_img(rgb: Rgb, r: int) -> Image.Image:
    w = int(r * 3)
    img, d = _new(w, w)
    pts = []
    for i in range(20):
        rr = r * 1.4 if i % 2 == 0 else r * 0.6
        a = i * math.pi / 10
        pts.append(((w / 2 + rr * math.cos(a)) * SS, (w / 2 + rr * math.sin(a)) * SS))
    d.polygon(pts, fill=_a(R.outline_for(rgb)))
    d.polygon([((w / 2 + (x / SS - w / 2) * 0.82) * SS, (w / 2 + (y / SS - w / 2) * 0.82) * SS) for x, y in pts], fill=_a(rgb))
    return _done(img, w, w)


def bird_img(rgb: Rgb, flap: bool) -> Image.Image:
    w, h = 84, 56
    img, d = _new(w, h)
    wing = [(30, 30), (46, 30), (38, 6 if flap else 52)]
    d.ellipse(_s(18, 22, 60, 42), fill=_a(R.outline_for(rgb)))
    d.ellipse(_s(21, 25, 57, 39), fill=_a(rgb))
    d.ellipse(_s(52, 16, 70, 34), fill=_a(R.outline_for(rgb)))
    d.ellipse(_s(54, 18, 68, 32), fill=_a(rgb))
    d.polygon([(70 * SS, 22 * SS), (82 * SS, 26 * SS), (70 * SS, 30 * SS)], fill=(255, 150, 30, 255))
    d.polygon([(x * SS, y * SS) for x, y in wing], fill=_a(R.outline_for(rgb)))
    d.polygon([((38 + (x - 38) * 0.7) * SS, (30 + (y - 30) * 0.8) * SS) for x, y in wing], fill=_a(rgb))
    d.ellipse(_s(61, 21, 65, 25), fill=(20, 20, 20, 255))
    return _done(img, w, h)


def cloud_img() -> Image.Image:
    w, h = 170, 80
    img, d = _new(w, h)
    blobs = ((15, 30, 75, 75), (45, 8, 110, 70), (85, 22, 150, 75), (30, 38, 140, 78))
    for b in blobs:
        d.ellipse(_s(*b), fill=(150, 160, 175, 255))
    for b in blobs:
        d.ellipse(_s(b[0] + 3, b[1] + 3, b[2] - 3, b[3] - 3), fill=(252, 252, 255, 255))
    return _done(img, w, h)


def kite_img(rgb: Rgb) -> Image.Image:
    w, h = 70, 120
    img, d = _new(w, h)
    pts = [(35, 2), (66, 38), (35, 74), (4, 38)]
    d.polygon([(x * SS, y * SS) for x, y in pts], fill=_a(R.outline_for(rgb)))
    d.polygon([((35 + (x - 35) * 0.85) * SS, (38 + (y - 38) * 0.85) * SS) for x, y in pts], fill=_a(rgb))
    d.line(_s(35, 74, 28, 90, 42, 104, 35, 118), fill=(80, 80, 80, 255), width=2 * SS)
    return _done(img, w, h)
