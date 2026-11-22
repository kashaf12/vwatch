"""PIL/numpy drawing primitives for the procedural clips: coloured shape sprites, gradients, text."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 165, 70),
    "blue": (45, 95, 225),
    "yellow": (245, 210, 30),
    "orange": (245, 130, 20),
    "purple": (140, 60, 190),
    "pink": (245, 120, 175),
    "white": (250, 250, 250),
    "black": (20, 20, 20),
    "brown": (130, 80, 40),
}
COLOUR_NAMES = tuple(COLOURS)
SHAPES = ("circle", "square", "triangle", "star", "diamond", "heart", "ring", "cross", "arrow")

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_UNICODE = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_KOHINOOR = "/System/Library/Fonts/Kohinoor.ttc"
_SS = 4  # supersampling factor for sprites


# ------------------------------------------------------------------ fonts


@lru_cache(maxsize=64)
def font(size: int, path: str = FONT_BOLD) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _has_glyph(path: str, ch: str) -> bool:
    try:
        f = ImageFont.truetype(path, 40)
    except OSError:
        return False
    # Unassigned code points render the font's .notdef box; a real glyph differs from it.
    return bytes(f.getmask(ch)) != bytes(f.getmask("\U0010ffff"))


@lru_cache(maxsize=1)
def rupee_font_path() -> str | None:
    """First font that really has the rupee sign U+20B9, or None."""
    for path in (FONT_KOHINOOR, FONT_UNICODE, FONT_BOLD):
        if _has_glyph(path, "₹"):
            return path
    return None


def rupee_label(amount: int | str) -> tuple[str, str]:
    """(text, font path) for a rupee amount; falls back to 'Rs ' when no font has the glyph."""
    path = rupee_font_path()
    if path is None:
        return f"Rs {amount}", FONT_BOLD
    return f"₹{amount}", path


# ------------------------------------------------------------------ canvases


def gradient(h: int, w: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    col = np.asarray(top, np.float32) * (1 - t) + np.asarray(bottom, np.float32) * t
    return np.broadcast_to(col, (h, w, 3)).round().astype(np.uint8).copy()


def paste(canvas: np.ndarray, sprite: np.ndarray, cx: float, cy: float) -> None:
    """Alpha-blend an RGBA sprite centred at (cx, cy) onto `canvas` IN PLACE (renderer-private buffer)."""
    sh, sw = sprite.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    ch, cw = canvas.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(cw, x0 + sw), min(ch, y0 + sh)
    if X0 >= X1 or Y0 >= Y1:
        return
    s = sprite[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = s[..., 3:4].astype(np.float32) / 255.0
    dst = canvas[Y0:Y1, X0:X1].astype(np.float32)
    canvas[Y0:Y1, X0:X1] = (s[..., :3] * a + dst * (1 - a)).round().astype(np.uint8)


# ------------------------------------------------------------------ shapes


def _star(n: int = 5, r_out: float = 0.48, r_in: float = 0.2) -> list[tuple[float, float]]:
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / n
        pts.append((0.5 + r * math.cos(a), 0.53 + r * math.sin(a)))
    return pts


def _heart() -> list[tuple[float, float]]:
    pts = []
    for k in range(80):
        t = 2 * math.pi * k / 80
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((0.5 + x / 36, 0.47 + y / 36))
    return pts


_POLYS: dict[str, list[tuple[float, float]]] = {
    "square": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
    "triangle": [(0.5, 0.06), (0.95, 0.9), (0.05, 0.9)],
    "diamond": [(0.5, 0.03), (0.97, 0.5), (0.5, 0.97), (0.03, 0.5)],
    "cross": [(0.36, 0.05), (0.64, 0.05), (0.64, 0.36), (0.95, 0.36), (0.95, 0.64), (0.64, 0.64),
              (0.64, 0.95), (0.36, 0.95), (0.36, 0.64), (0.05, 0.64), (0.05, 0.36), (0.36, 0.36)],
    "arrow": [(0.04, 0.34), (0.54, 0.34), (0.54, 0.08), (0.96, 0.5), (0.54, 0.92), (0.54, 0.66), (0.04, 0.66)],
    "star": _star(),
    "heart": _heart(),
}


def _scaled(poly: list[tuple[float, float]], s: float) -> list[tuple[float, float]]:
    return [(0.5 + (x - 0.5) * s, 0.5 + (y - 0.5) * s) for x, y in poly]


def outline_for(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (25, 25, 25) if lum > 110 else (240, 240, 240)


@lru_cache(maxsize=512)
def shape_sprite(shape: str, rgb: tuple[int, int, int], size: int) -> np.ndarray:
    """RGBA (size, size, 4) sprite of a named shape with a contrasting outline, antialiased."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    big = size * _SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    line = outline_for(rgb) + (255,)
    fill = rgb + (255,)

    def px(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * big, y * big) for x, y in poly]

    if shape == "circle":
        d.ellipse((0.04 * big, 0.04 * big, 0.96 * big, 0.96 * big), fill=line)
        d.ellipse((0.1 * big, 0.1 * big, 0.9 * big, 0.9 * big), fill=fill)
    elif shape == "ring":
        d.ellipse((0.04 * big, 0.04 * big, 0.96 * big, 0.96 * big), fill=line)
        d.ellipse((0.1 * big, 0.1 * big, 0.9 * big, 0.9 * big), fill=fill)
        d.ellipse((0.28 * big, 0.28 * big, 0.72 * big, 0.72 * big), fill=line)
        d.ellipse((0.34 * big, 0.34 * big, 0.66 * big, 0.66 * big), fill=(0, 0, 0, 0))
    else:
        poly = _POLYS[shape]
        d.polygon(px(poly), fill=line)
        d.polygon(px(_scaled(poly, 0.86)), fill=fill)
    return np.asarray(img.resize((size, size), Image.Resampling.LANCZOS)).copy()


def rounded_box(w: int, h: int, fill: tuple[int, int, int], radius: int, outline: tuple[int, int, int] | None = None,
                width: int = 0) -> Image.Image:
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), radius=radius * 2, fill=fill + (255,),
                                          outline=(outline + (255,)) if outline else None, width=width * 2)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def text_sprite(text: str, size: int, fg: tuple[int, int, int], bg: tuple[int, int, int] | None = None,
                *, font_path: str = FONT_BOLD, pad: int = 14, radius: int = 16,
                min_w: int = 0) -> np.ndarray:
    """RGBA sprite of `text`; with `bg` it sits on an opaque rounded box."""
    f = font(size, font_path)
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    l, t, r, b = probe.textbbox((0, 0), text, font=f)
    tw, th = r - l, b - t
    w, h = max(min_w, tw + 2 * pad), th + 2 * pad
    img = rounded_box(w, h, bg, radius) if bg is not None else Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(((w - tw) / 2 - l, (h - th) / 2 - t), text, font=f, fill=fg + (255,))
    return np.asarray(img).copy()


def bell_sprite(size: int, rgb: tuple[int, int, int] = (245, 200, 40)) -> np.ndarray:
    """A simple bell icon."""
    big = size * _SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    line = outline_for(rgb) + (255,)
    for col, inset in ((line, 0.0), (rgb + (255,), 0.04)):
        i = inset * big
        d.pieslice((0.18 * big + i, 0.12 * big + i, 0.82 * big - i, 0.8 * big - i), 180, 360, fill=col)
        d.polygon([(0.18 * big + i, 0.46 * big), (0.82 * big - i, 0.46 * big), (0.92 * big - i, 0.78 * big),
                   (0.08 * big + i, 0.78 * big)], fill=col)
    d.ellipse((0.4 * big, 0.76 * big, 0.6 * big, 0.96 * big), fill=line)
    d.ellipse((0.44 * big, 0.8 * big, 0.56 * big, 0.92 * big), fill=rgb + (255,))
    d.ellipse((0.45 * big, 0.04 * big, 0.55 * big, 0.14 * big), fill=line)
    return np.asarray(img.resize((size, size), Image.Resampling.LANCZOS)).copy()
