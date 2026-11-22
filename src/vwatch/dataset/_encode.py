"""Streaming H.264/AAC writer: same encoder settings as `vwatch.media.write_video`, but frames
come from an iterator so a 55 s 540x960 clip never has to sit in memory twice."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from vwatch.media import FFMPEG, MediaError


def write_video_stream(
    frames: Iterable[np.ndarray],
    fps: float,
    out_path: str | Path,
    *,
    size: tuple[int, int],
    audio: np.ndarray | None = None,
    sr: int = 44100,
    crf: int = 20,
) -> Path:
    """Encode (H, W, 3) uint8 frames (+ optional mono float32 audio). `size` is (width, height)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    tmp_wav: Path | None = None
    if audio is not None and audio.size:
        tmp_wav = out.with_suffix(".tmp.f32")
        tmp_wav.write_bytes(np.clip(audio, -1, 1).astype(np.float32).tobytes())
        cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(tmp_wav)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if tmp_wav is not None:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += [str(out)]
    n = 0
    try:
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
            assert proc.stdin is not None
            try:
                for f in frames:
                    if f.shape != (h, w, 3) or f.dtype != np.uint8:
                        raise ValueError(f"frame {n} has shape {f.shape} dtype {f.dtype}, expected {(h, w, 3)} uint8")
                    proc.stdin.write(np.ascontiguousarray(f).tobytes())
                    n += 1
            except BrokenPipeError:
                pass  # ffmpeg died; its stderr explains why below
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                code = proc.wait()
            if code != 0:
                err.seek(0)
                raise MediaError(f"ffmpeg failed ({code}) after {n} frames: {err.read().decode('utf-8', 'replace')[-800:]}")
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)
    if n == 0:
        raise ValueError("no frames to encode")
    return out
