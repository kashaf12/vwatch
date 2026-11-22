"""Image helpers: timestamps, grids, crops, JPEG encoding, token estimates."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Claude resizes images whose long edge exceeds this; tokens ~= w*h/750 after resize.
MAX_LONG_EDGE = 1568
TOKENS_PER_PIXEL = 1 / 750


@dataclass(frozen=True)
class Box:
    """Normalized box: 0..1 in both axes, (x0, y0) top-left."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError(f"invalid normalized box {self}")

    @classmethod
    def clamp(cls, x0: float, y0: float, x1: float, y1: float) -> Box:
        c = lambda v: min(1.0, max(0.0, float(v)))  # noqa: E731
        x0, x1 = sorted((c(x0), c(x1)))
        y0, y1 = sorted((c(y0), c(y1)))
        if x1 - x0 < 0.02:
            x0, x1 = (x0, x0 + 0.02) if x0 <= 0.98 else (0.98, 1.0)
        if y1 - y0 < 0.02:
            y0, y1 = (y0, y0 + 0.02) if y0 <= 0.98 else (0.98, 1.0)
        return cls(x0, y0, x1, y1)


FULL = Box(0, 0, 1, 1)


def estimate_image_tokens(width: int, height: int) -> int:
    long_edge = max(width, height)
    if long_edge > MAX_LONG_EDGE:
        s = MAX_LONG_EDGE / long_edge
        width, height = int(width * s), int(height * s)
    return max(1, math.ceil(width * height * TOKENS_PER_PIXEL))


def estimate_text_tokens(text: str) -> int:
    # Conservative for mixed Hindi/English: ~3 chars per token.
    return max(1, math.ceil(len(text) / 3))


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, x1 = int(box.x0 * w), max(int(box.x0 * w) + 1, int(box.x1 * w))
    y0, y1 = int(box.y0 * h), max(int(box.y0 * h) + 1, int(box.y1 * h))
    return frame[y0:y1, x0:x1].copy()


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = max(1, int(round(h * width / w)))
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp(frame: np.ndarray, label: str) -> np.ndarray:
    """Burn a label (usually a timestamp) into the top-left corner. Returns a new array."""
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    size = max(10, img.width // 18)
    font = _font(size)
    pad = 3
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad), fill=(255, 255, 0))
    draw.text((pad, pad), label, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def grid(frames: Sequence[np.ndarray], cols: int, *, gap: int = 4) -> np.ndarray:
    """Tile same-size frames into a grid (row-major). Pads the last row with grey."""
    if not frames:
        raise ValueError("grid needs at least one frame")
    h, w = frames[0].shape[:2]
    cols = max(1, min(cols, len(frames)))
    rows = math.ceil(len(frames) / cols)
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 90, dtype=np.uint8)
    for k, f in enumerate(frames):
        if f.shape[:2] != (h, w):
            f = np.asarray(Image.fromarray(f).resize((w, h)))
        r, c = divmod(k, cols)
        out[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = f[..., :3]
    return out


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def to_jpeg_b64(frame: np.ndarray, quality: int = 82) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def image_block(frame: np.ndarray, quality: int = 82) -> dict:
    """Anthropic Messages API image content block."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": to_jpeg_b64(frame, quality)}}
