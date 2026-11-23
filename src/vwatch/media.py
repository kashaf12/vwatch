"""ffmpeg/ffprobe wrappers. All video I/O goes through here.

Frames are numpy uint8 arrays shaped (H, W, 3) RGB, or (H, W) for gray.
Audio is float32 mono in [-1, 1].
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


class MediaError(RuntimeError):
    """ffmpeg/ffprobe failed. The message carries ffmpeg's stderr tail."""


class FFmpegNotFoundError(FileNotFoundError):
    """No usable ffmpeg/ffprobe: not configured, not on PATH, and no static build could be fetched."""


# ------------------------------------------------------------------ binary resolution
#
# ffmpeg and ffprobe are resolved once per process, in this order:
#   1. $VWATCH_FFMPEG / $VWATCH_FFPROBE (a path, or a name looked up on PATH)
#   2. `ffmpeg` / `ffprobe` on PATH (the system install; returned as the bare name, as before)
#   3. a static build from the optional `static-ffmpeg` package (in the `mcp` extra), downloaded once
#      to $VWATCH_CACHE/ffmpeg/<platform> (default ~/.cache/vwatch/ffmpeg/<platform>)
# `media.FFMPEG` / `media.FFPROBE` still work: they resolve lazily through the same functions.

_ENV_VARS = {"ffmpeg": "VWATCH_FFMPEG", "ffprobe": "VWATCH_FFPROBE"}
_fetch_lock = threading.Lock()


def _cache_root() -> Path:
    return Path(os.environ.get("VWATCH_CACHE", Path.home() / ".cache" / "vwatch"))


@functools.cache
def _static_binaries() -> tuple[str, str]:
    """(ffmpeg, ffprobe) from `static-ffmpeg`, downloading them on first use. Raises FFmpegNotFoundError."""
    try:
        from static_ffmpeg import run
    except ImportError as exc:
        raise FFmpegNotFoundError(
            "ffmpeg/ffprobe not found. Install ffmpeg (macOS: `brew install ffmpeg`, Debian/Ubuntu: `sudo apt "
            "install ffmpeg`, Windows: `winget install ffmpeg`), set VWATCH_FFMPEG/VWATCH_FFPROBE to the "
            "binaries, or install the `static-ffmpeg` package (`pip install 'vwatch[mcp]'`) to fetch a static "
            "build automatically.") from exc
    dest = _cache_root() / "ffmpeg" / run.get_platform_key()
    with _fetch_lock:
        if not (dest / "installed.crumb").exists():
            print(f"vwatch: no ffmpeg on PATH; downloading a static ffmpeg build to {dest} (one time)...",
                  file=sys.stderr, flush=True)
        try:
            # static-ffmpeg prints progress to stdout; an MCP stdio server must keep stdout for the protocol.
            with contextlib.redirect_stdout(sys.stderr):
                ffmpeg_exe, ffprobe_exe = run.get_or_fetch_platform_executables_else_raise(download_dir=str(dest))
        except Exception as exc:
            raise FFmpegNotFoundError(
                f"ffmpeg/ffprobe not found on PATH and the static ffmpeg download into {dest} failed "
                f"({type(exc).__name__}: {exc}). Install ffmpeg yourself (macOS: `brew install ffmpeg`, "
                "Debian/Ubuntu: `sudo apt install ffmpeg`, Windows: `winget install ffmpeg`) or set "
                "VWATCH_FFMPEG/VWATCH_FFPROBE.") from exc
    for exe in (ffmpeg_exe, ffprobe_exe):
        if not os.access(exe, os.X_OK):
            raise FFmpegNotFoundError(f"static ffmpeg build is incomplete: {exe} is missing or not executable; "
                                      f"delete {dest} to re-download it")
    return ffmpeg_exe, ffprobe_exe


@functools.cache
def _resolve(name: str) -> str:
    var = _ENV_VARS[name]
    configured = os.environ.get(var, "").strip()
    if configured:
        found = shutil.which(configured)
        if found is None:
            raise FFmpegNotFoundError(f"{var}={configured!r} is not an executable file")
        return found
    if shutil.which(name) is not None:
        return name
    ffmpeg_exe, ffprobe_exe = _static_binaries()
    return ffmpeg_exe if name == "ffmpeg" else ffprobe_exe


def ffmpeg() -> str:
    """The ffmpeg executable to run (resolved once, then cached). Raises FFmpegNotFoundError."""
    return _resolve("ffmpeg")


def ffprobe() -> str:
    """The ffprobe executable to run (resolved once, then cached). Raises FFmpegNotFoundError."""
    return _resolve("ffprobe")


def reset_binaries() -> None:
    """Forget resolved binaries (after changing PATH or VWATCH_FFMPEG/VWATCH_FFPROBE)."""
    _resolve.cache_clear()
    _static_binaries.cache_clear()


def __getattr__(name: str) -> Any:  # PEP 562: keep `media.FFMPEG` / `media.FFPROBE` working, resolved lazily
    if name == "FFMPEG":
        return ffmpeg()
    if name == "FFPROBE":
        return ffprobe()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    nb_frames: int
    has_audio: bool

    def frame_time(self, index: int) -> float:
        return index / self.fps

    def frame_index(self, t: float) -> int:
        return max(0, min(self.nb_frames - 1, int(round(t * self.fps))))


def _name(exe: str) -> str:
    """'ffmpeg' for '/some/cache/ffmpeg' (error messages stay short whichever binary ran)."""
    base = os.path.basename(exe)
    return base[:-4] if base.lower().endswith(".exe") else base


def _run(cmd: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(list(cmd), input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise MediaError(f"{_name(cmd[0])} failed ({proc.returncode}): {tail}")
    return proc.stdout


def probe(path: str | Path) -> VideoInfo:
    out = _run(
        [ffprobe(), "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames",
         "-of", "json", str(path)]
    )
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    fps = float(Fraction(video["r_frame_rate"]))
    duration = float(data["format"]["duration"])
    nb = int(video.get("nb_frames") or 0) or int(round(duration * fps))
    return VideoInfo(
        path=str(path),
        duration=duration,
        fps=fps,
        width=int(video["width"]),
        height=int(video["height"]),
        nb_frames=nb,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def _scaled_size(info: VideoInfo, width: int) -> tuple[int, int]:
    w = width - width % 2
    h = int(round(info.height * w / info.width))
    return w, h - h % 2


def decode_all(path: str | Path, width: int, *, gray: bool = False) -> tuple[np.ndarray, VideoInfo]:
    """Decode every frame at `width`. Returns (N, H, W[, 3]) uint8. Use small widths."""
    info = probe(path)
    w, h = _scaled_size(info, width)
    pix = "gray" if gray else "rgb24"
    raw = _run([ffmpeg(), "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area",
                "-f", "rawvideo", "-pix_fmt", pix, "-"])
    ch = 1 if gray else 3
    n = len(raw) // (w * h * ch)
    arr = np.frombuffer(raw[: n * w * h * ch], dtype=np.uint8)
    arr = arr.reshape((n, h, w)) if gray else arr.reshape((n, h, w, 3))
    return arr, info


SELECT_MAX_TERMS = 40  # ffmpeg's expression parser fails on very long select expressions


def frames_by_index(path: str | Path, indices: Iterable[int], width: int) -> dict[int, np.ndarray]:
    """Exact frames by index, one decode pass. Returns {index: (H, W, 3) RGB}.

    Indices past the end of the video are dropped.
    """
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    info = probe(path)
    w, h = _scaled_size(info, width)
    if len(wanted) <= SELECT_MAX_TERMS:
        return _select_frames(path, wanted, w, h)
    return _stream_frames(path, wanted, w, h)


def _select_frames(path: str | Path, wanted: list[int], w: int, h: int) -> dict[int, np.ndarray]:
    expr = "+".join(f"eq(n\\,{i})" for i in wanted)
    raw = _run([ffmpeg(), "-v", "error", "-i", str(path),
                "-vf", f"select='{expr}',scale={w}:{h}:flags=area",
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    size = w * h * 3
    got = len(raw) // size
    frames = [np.frombuffer(raw[k * size:(k + 1) * size], dtype=np.uint8).reshape(h, w, 3) for k in range(got)]
    return dict(zip(wanted[:got], frames))


def _stream_frames(path: str | Path, wanted: list[int], w: int, h: int) -> dict[int, np.ndarray]:
    """Decode sequentially up to the last wanted frame, keeping only the wanted ones (bounded memory)."""
    size = w * h * 3
    keep = set(wanted)
    proc = subprocess.Popen([ffmpeg(), "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=area",
                             "-frames:v", str(wanted[-1] + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out: dict[int, np.ndarray] = {}
    assert proc.stdout is not None and proc.stderr is not None
    n = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        if n in keep:
            out[n] = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        n += 1
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise MediaError(f"ffmpeg failed ({proc.returncode}): {err[-800:]}")
    return out


def frames_at(path: str | Path, times: Iterable[float], width: int) -> list[tuple[float, np.ndarray]]:
    """Frames nearest to each time. Returns [(actual_time, frame)] sorted by time, deduplicated."""
    info = probe(path)
    idx = sorted({info.frame_index(t) for t in times})
    got = frames_by_index(path, idx, width)
    return [(info.frame_time(i), got[i]) for i in idx if i in got]


def extract_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Mono float32 audio. Empty array if the file has no audio."""
    info = probe(path)
    if not info.has_audio:
        return np.zeros(0, dtype=np.float32)
    raw = _run([ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                "-f", "f32le", "-"])
    audio = np.frombuffer(raw, dtype=np.float32)
    # Put sample 0 at the first video frame: some files start their audio stream later (or earlier)
    # than the video, and every audio timestamp (silences, words, loudness) would be off by that much.
    shift = int(round(audio_offset(path) * sr))
    if shift > 0:
        return np.concatenate([np.zeros(shift, dtype=np.float32), audio])
    return audio[-shift:].copy()


def audio_offset(path: str | Path) -> float:
    """Seconds from the first video frame to the first audio sample (negative: audio starts first)."""
    out = _run([ffprobe(), "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(path)])
    starts: dict[str, float] = {}
    for st in json.loads(out).get("streams", []):
        kind, start = st.get("codec_type"), st.get("start_time")
        if kind in ("video", "audio") and kind not in starts and start not in (None, "N/A"):
            starts[kind] = float(start)
    if "video" not in starts or "audio" not in starts:
        return 0.0
    return starts["audio"] - starts["video"]


def write_video(
    frames: np.ndarray,
    fps: float,
    out_path: str | Path,
    *,
    audio: np.ndarray | None = None,
    sr: int = 44100,
    crf: int = 20,
) -> Path:
    """Encode (N, H, W, 3) RGB uint8 frames (+ optional mono float32 audio) to H.264/AAC mp4."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    cmd = [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
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
    try:
        _run(cmd, input_bytes=np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)
    return out
