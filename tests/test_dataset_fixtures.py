"""Tiny synthetic source for dataset tests: 8 s, 108x192, 30 fps, moving content, a caption band with
changing light text, one hard scene cut and a tone 'voice' with one pause."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vwatch.dataset._common import FPS, SPF, SR
from vwatch.dataset._source import SourceClip, SourceSpec, analyse

W, H, N = 108, 192, 240
CUT = 120
SPEC = SourceSpec("synthetic", Path("synthetic.mp4"), (0.78, 0.93), (0.78, 0.93), "light_text", 0.01,
                  props_path=None)


def make_frames() -> np.ndarray:
    fr = np.zeros((N, H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(N):
        base = (60, 90, 140) if i < CUT else (220, 160, 60)
        f = np.empty((H, W, 3), np.uint8)
        f[:] = base
        f[..., 0] = np.clip(base[0] + (yy * 0.3) + 10 * np.sin(i / 7), 0, 255)
        x0 = (i * 2) % (W - 20)
        f[40:60, x0:x0 + 20] = (230, 200, 40)  # moving block
        if 12 <= i < N - 12:  # caption: light text blocks that change every second
            seg = i // FPS
            f[int(0.82 * H):int(0.9 * H), 10:W - 10] = (30, 30, 30)
            for k in range(3 + seg % 3):
                f[int(0.84 * H):int(0.88 * H), 14 + k * 28 + seg % 4:14 + k * 28 + 20 + seg % 4] = (250, 250, 250)
        fr[i] = f
    return fr


def make_audio() -> np.ndarray:
    t = np.arange(N * SPF) / SR
    a = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    a[int(4.0 * SR):int(4.3 * SR)] = 0  # a pause
    return a


def make_source() -> SourceClip:
    return analyse(SPEC, make_frames(), make_audio())
