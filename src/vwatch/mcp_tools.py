"""vwatch's perception primitives as plain functions, for agents that bring their own model.

No model calls happen here: everything is ffmpeg, numpy and (for speech) a local whisper. Each tool
returns a fresh list of content blocks ({"type": "text"} or {"type": "image"} with base64 JPEG), and
the first block is always a short text summary with the timestamps shown and an estimated token
cost, so the calling agent can budget. `vwatch.mcp_server` exposes these over MCP.

Expected failures (missing file, not a video, bad arguments) raise `VideoToolError` with a message
meant for the agent to read.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from vwatch import media, signals, transcribe
from vwatch.arms import tools
from vwatch.arms.common import Evidence, VideoContext, select_hints
from vwatch.imaging import (
    Box,
    crop,
    estimate_image_tokens,
    estimate_text_tokens,
    fmt_time,
    image_block,
    resize_width,
    stamp,
)
from vwatch.llm.base import text_block

Block = dict

# Response caps: keep every tool result well under an MCP client's per-result limit (~25k tokens).
OVERVIEW_BUDGET = 25_000  # the watcher's default evidence budget; full view / skim density derive from it
OVERVIEW_MAX_TOKENS = 10_000
OVERVIEW_MAX_FRAMES = 96
TILE_W_RANGE = (90, 360)
SEEK_MAX_WINDOW_S = 60.0
CROP_DECODE_MAX_W = 1920
CROP_OUT_MAX_EDGE = 1024
LISTEN_MAX_WINDOW_S = 30.0
TEXT_MAX_CHARS = 48_000  # ~16k tokens at the repo's 3 chars/token estimate
SIGNALS_TOP_RANGE = (1, 200)
CACHE_SIZE = 8


class VideoToolError(ValueError):
    """A bad request or unreadable file. The message is written for the calling agent."""


# ------------------------------------------------------------------ per-video cache


@dataclass(frozen=True)
class _Entry:
    path: Path
    info: media.VideoInfo
    track: signals.SignalTrack | None = None
    speech_s: float | None = None  # seconds of speech found by VAD; None = not checked yet
    transcripts: tuple[tuple[str, transcribe.Transcript], ...] = ()  # (language or "auto", transcript)


_lock = threading.Lock()
_cache: OrderedDict[tuple[str, int, int], _Entry] = OrderedDict()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def resolve_path(path: str) -> Path:
    """Absolute path of an existing local file. Relative paths resolve against the server's cwd."""
    if not isinstance(path, str) or not path.strip():
        raise VideoToolError("path is required: a local video file (absolute, or relative to the server's cwd)")
    if "://" in path:
        raise VideoToolError(f"only local files are supported, not URLs: {path}")
    given = Path(os.path.expanduser(path.strip()))
    p = given if given.is_absolute() else Path.cwd() / given
    if not p.exists():
        hint = "" if given.is_absolute() else \
            f" (relative paths resolve against the server's working directory {Path.cwd()}; pass an absolute path)"
        raise VideoToolError(f"no such file: {p}{hint}")
    if not p.is_file():
        raise VideoToolError(f"not a file: {p}")
    return p.resolve()


def _key(p: Path) -> tuple[str, int, int]:
    st = p.stat()
    return str(p), st.st_mtime_ns, st.st_size


def _entry(path: str) -> _Entry:
    p = resolve_path(path)
    key = _key(p)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    try:
        info = media.probe(p)
    except (media.MediaError, ValueError, KeyError) as exc:
        raise VideoToolError(f"not a readable video: {p} ({str(exc).strip()[-300:]})") from exc
    if info.duration <= 0 or info.width <= 0 or info.height <= 0:
        raise VideoToolError(f"video has no frames or zero duration: {p}")
    return _store(key, _Entry(path=p, info=info))


def _store(key: tuple[str, int, int], entry: _Entry) -> _Entry:
    with _lock:
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return entry


def _update(entry: _Entry, compute: Callable[[_Entry], _Entry]) -> _Entry:
    """Compute a lazily-filled field (outside the lock) and store the new entry."""
    new = compute(entry)
    return _store(_key(entry.path), new)


def _with_track(entry: _Entry) -> _Entry:
    if entry.track is not None:
        return entry
    return _update(entry, lambda e: replace(e, track=signals.compute(e.path)))


def _with_transcript(entry: _Entry, language: str | None) -> tuple[_Entry, transcribe.Transcript]:
    lang = language or "auto"
    for k, tr in entry.transcripts:
        if k == lang:
            return entry, tr
    if not entry.info.has_audio:
        return entry, transcribe.EMPTY
    tr = transcribe.transcribe(entry.path, language=language)
    entry = _update(entry, lambda e: replace(e, transcripts=e.transcripts + ((lang, tr),)))
    return entry, tr


def _speech_seconds(path: Path) -> float:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = media.extract_audio(path, sr=16000)
    if audio.size == 0:
        return 0.0
    chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=100))
    return sum(c["end"] - c["start"] for c in chunks) / 16000


def _ctx(entry: _Entry, transcript: transcribe.Transcript = transcribe.EMPTY) -> VideoContext:
    """A VideoContext for the arms' rendering helpers. Image tools never read the signal track."""
    i = entry.info
    track = entry.track or signals.SignalTrack(
        fps=i.fps, duration=i.duration, diff=np.zeros(0, np.float32), local=np.zeros(0, np.float32),
        rms_db=np.zeros(0, np.float32), frames=np.zeros((0, 1, 1), np.uint8))
    return VideoContext(path=entry.path, info=i, transcript=transcript, track=track, candidates=())


# ------------------------------------------------------------------ helpers


def _num(name: str, v: float | None, lo: float | None = None, hi: float | None = None) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise VideoToolError(f"{name} must be a number, got {v!r}") from exc
    if not np.isfinite(f):
        raise VideoToolError(f"{name} must be finite, got {v!r}")
    if lo is not None and f < lo:
        raise VideoToolError(f"{name} must be >= {lo}, got {f}")
    if hi is not None and f > hi:
        raise VideoToolError(f"{name} must be <= {hi}, got {f}")
    return f


def _window(info: media.VideoInfo, t0: float | None, t1: float | None, max_len: float | None
            ) -> tuple[float, float, str]:
    """Validated [t0, t1] inside the video, capped to max_len. Returns (t0, t1, note)."""
    a = _num("t0", t0, 0.0)
    b = _num("t1", t1, 0.0)
    a = 0.0 if a is None else a
    b = info.duration if b is None else b
    if a >= info.duration:
        raise VideoToolError(f"t0={a} is past the end of the video ({info.duration:.2f} s)")
    if b <= a:
        raise VideoToolError(f"t1 must be greater than t0 (got t0={a}, t1={b})")
    b = min(b, info.duration)
    note = ""
    if max_len is not None and b - a > max_len:
        b = a + max_len
        note = f" (window capped at {max_len:.0f} s; call again for the rest)"
    return a, b, note


def _truncate(text: str, hint: str) -> str:
    if len(text) <= TEXT_MAX_CHARS:
        return text
    cut = text[:TEXT_MAX_CHARS].rsplit("\n", 1)[0]
    return f"{cut}\n... (truncated at {TEXT_MAX_CHARS} characters; {hint})"


def _frame_times(ev: Evidence) -> list[float]:
    """Distinct single-frame times the rendering helpers logged (grids also log their span)."""
    return sorted({o.t0 for o in ev.observations if o.kind == "frame" and o.t0 == o.t1})


def _with_summary(summary: str, ev: Evidence) -> list[Block]:
    body = list(ev.blocks)
    est = ev.tokens + estimate_text_tokens(summary) + 15
    return [text_block(f"{summary} [~{est} tokens]")] + body


# ------------------------------------------------------------------ tools


def video_info(path: str) -> list[Block]:
    e = _entry(path)
    i = e.info
    if e.speech_s is None:
        try:
            speech = _speech_seconds(e.path) if i.has_audio else 0.0
        except Exception:  # noqa: BLE001 - the VAD is a nicety; never fail video_info over it
            speech = -1.0
        e = _update(e, lambda x: replace(x, speech_s=speech))
    if not i.has_audio:
        speech_line = "audio: none"
    elif e.speech_s is not None and e.speech_s < 0:
        speech_line = "audio: yes; speech: unknown (voice-activity check failed; try video_transcript)"
    elif e.speech_s and e.speech_s >= 0.3:
        speech_line = f"audio: yes; speech: yes, ~{e.speech_s:.1f} s of voice (use video_transcript)"
    else:
        speech_line = "audio: yes; speech: none detected (music/effects/silence only)"
    ctx = _ctx(e)
    full = tools.full_view(ctx, OVERVIEW_BUDGET)
    ov = (f"video_overview default: whole clip at {full[1]:g} fps, ~{tools.overview_cost(ctx, full[1], full[0])} tokens"
          if full else f"video_overview default: low-res skim, up to ~{OVERVIEW_MAX_TOKENS} tokens")
    lines = [
        f"file: {e.path}",
        f"duration: {i.duration:.2f} s ({fmt_time(i.duration)}), {i.nb_frames} frames at {i.fps:.3g} fps",
        f"size: {i.width}x{i.height}",
        speech_line,
        ov,
        f"video_seek cost: ~{tools.seek_cost(ctx, 0, 1, 8)} tokens per 1 s at 8 fps (max 16 frames per call)",
    ]
    return [text_block("\n".join(lines))]


def video_transcript(path: str, language: str | None = None, t0: float | None = None,
                     t1: float | None = None, words: bool = False) -> list[Block]:
    e = _entry(path)
    a, b, _ = _window(e.info, t0, t1, None)
    if language is not None and (not isinstance(language, str) or not language.strip().isalpha()
                                 or len(language.strip()) > 8):
        raise VideoToolError(f"language must be a code like 'en' or 'hi', got {language!r}")
    lang = language.strip().lower() if language else None
    if not e.info.has_audio:
        return [text_block("TRANSCRIPT: the file has no audio track. [~15 tokens]")]
    try:
        e, tr = _with_transcript(e, lang)
    except (RuntimeError, OSError) as exc:
        raise VideoToolError(f"transcription failed: {exc}") from exc
    body = tr.to_text(a, b)
    if words:
        body += "\nWORDS (word@seconds): " + tr.words_text(a, b)
    body = _truncate(body, "narrow it with t0/t1")
    head = (f"TRANSCRIPT {fmt_time(a)}–{fmt_time(b)} (language {tr.language or 'n/a'}; "
            f"[start–end] mm:ss.ss)")
    est = estimate_text_tokens(body) + 20
    return [text_block(f"{head} [~{est} tokens]\n{body}")]


def video_signals(path: str, kinds: list[str] | None = None, top: int = 40) -> list[Block]:
    e = _entry(path)
    k = int(_num("top", top, SIGNALS_TOP_RANGE[0], SIGNALS_TOP_RANGE[1]))
    if kinds is not None and (not isinstance(kinds, list | tuple) or not all(isinstance(x, str) for x in kinds)):
        raise VideoToolError("kinds must be a list of strings, e.g. ['flash', 'freeze']")
    e = _with_track(e)
    cands = signals.find_candidates(e.track)
    wanted = tuple(x.strip() for x in kinds) if kinds else None
    hints = select_hints(cands, wanted, k)
    present = sorted({c.kind for c in cands})
    head = (f"SIGNALS: {len(hints)} of {len(cands)} candidate moments"
            f"{' of kinds ' + ', '.join(wanted) if wanted else ''} (top {k} by score, in time order). "
            "Model-free hints from every frame and the audio, not verdicts: look before you claim.")
    if wanted and not hints:
        head += f" Kinds present in this video: {', '.join(present) or 'none'}."
    body = signals.describe(hints)
    text = f"{head}\n{body}"
    return [text_block(f"{text}\n[~{estimate_text_tokens(text) + 10} tokens]")]


def video_overview(path: str, fps: float | None = None, tile_width: int | None = None) -> list[Block]:
    e = _entry(path)
    req_fps = _num("fps", fps, 0.001, 30.0)
    req_w = _num("tile_width", tile_width, TILE_W_RANGE[0], TILE_W_RANGE[1])
    ctx = _ctx(e)
    dur = e.info.duration
    full = tools.full_view(ctx, OVERVIEW_BUDGET) if req_fps is None and req_w is None else None
    notes = []
    if full:
        tile_w, rate = full
    else:
        tile_w = int(req_w) if req_w else tools.OVERVIEW_TILE_W
        rate = req_fps or 1.0
        if req_fps is None:
            while rate > 0.25 and tools.overview_cost(ctx, rate, tile_w) > OVERVIEW_BUDGET * 0.25:
                rate /= 2
        if dur * rate > OVERVIEW_MAX_FRAMES:
            rate = OVERVIEW_MAX_FRAMES / dur
            notes.append(f"capped at {OVERVIEW_MAX_FRAMES} frames")
        if tools.overview_cost(ctx, rate, tile_w) > OVERVIEW_MAX_TOKENS:
            while tools.overview_cost(ctx, rate, tile_w) > OVERVIEW_MAX_TOKENS and dur * rate > 2:
                rate *= 0.8
            notes.append(f"density reduced to stay under ~{OVERVIEW_MAX_TOKENS} tokens")
    ev = Evidence()
    tools.overview(ctx, ev, rate, tile_w, full=bool(full))
    shown = _frame_times(ev)
    n = len(shown)
    span = f"{fmt_time(min(shown))}–{fmt_time(max(shown))}" if shown else "-"
    mode = "FULL VIEW (whole clip)" if full else "OVERVIEW (skim)"
    summary = (f"video_overview: {mode}, {n} frames {span}, one every {dur / max(2, int(dur * rate)):.2f} s, "
               f"tile {tile_w}px, {sum(1 for b in ev.blocks if b['type'] == 'image')} image(s)"
               f"{'; ' + '; '.join(notes) if notes else ''}. Timestamps are burned into each tile.")
    return _with_summary(summary, ev)


def video_seek(path: str, t0: float, t1: float, fps: float = 8.0) -> list[Block]:
    e = _entry(path)
    if t0 is None or t1 is None:
        raise VideoToolError("t0 and t1 are required (seconds)")
    rate = _num("fps", fps, 0.5, 30.0)
    a, b, note = _window(e.info, t0, t1, SEEK_MAX_WINDOW_S)
    ctx = _ctx(e)
    ev = Evidence()
    tools.seek(ctx, ev, a, b, rate)
    shown = _frame_times(ev)
    capped = int((b - a) * rate) + 1 > tools.MAX_SEEK_FRAMES
    cap = (f" Capped at {tools.MAX_SEEK_FRAMES} frames spread evenly over the window; use a shorter window to see"
           " every frame." if capped else "")
    summary = (f"video_seek: {len(shown)} frames {fmt_time(min(shown))}–{fmt_time(max(shown))}{note}.{cap}"
               if shown else f"video_seek: no frames decoded in {fmt_time(a)}–{fmt_time(b)}")
    return _with_summary(summary, ev)


def video_crop(path: str, t: float, box: list[float]) -> list[Block]:
    e = _entry(path)
    tt = _num("t", t, 0.0)
    if tt is None:
        raise VideoToolError("t is required (seconds)")
    if tt >= e.info.duration:
        raise VideoToolError(f"t={tt} is past the end of the video ({e.info.duration:.2f} s)")
    if not isinstance(box, list | tuple) or len(box) != 4:
        raise VideoToolError("box must be [x0, y0, x1, y1], normalized 0..1 with (0,0) the top-left")
    vals = [_num(f"box[{k}]", v, 0.0, 1.0) for k, v in enumerate(box)]
    try:
        bx = Box(*vals)  # type: ignore[arg-type]
    except ValueError as exc:
        raise VideoToolError(f"box must satisfy x0 < x1 and y0 < y1 (got {list(box)})") from exc
    width = min(e.info.width, CROP_DECODE_MAX_W)
    frames = media.frames_at(e.path, [tt], width)
    if not frames:
        raise VideoToolError(f"could not decode a frame at t={tt}")
    at, frame = frames[0]
    c = crop(frame, bx)
    h, w = c.shape[:2]
    if max(h, w) > CROP_OUT_MAX_EDGE:
        c = resize_width(c, max(1, int(w * CROP_OUT_MAX_EDGE / max(h, w))))
    img = stamp(c, fmt_time(at))
    tokens = estimate_image_tokens(img.shape[1], img.shape[0])
    src = f"{int(bx.x0 * e.info.width)},{int(bx.y0 * e.info.height)}–{int(bx.x1 * e.info.width)},{int(bx.y1 * e.info.height)}"
    summary = (f"video_crop: frame at {fmt_time(at)}, box {bx.x0:.2f},{bx.y0:.2f},{bx.x1:.2f},{bx.y1:.2f} "
               f"(source pixels {src}, decoded at {width}px wide), shown {img.shape[1]}x{img.shape[0]} "
               f"[~{tokens + 40} tokens]")
    return [text_block(summary), {**image_block(img), "_tokens": tokens}]


def video_listen(path: str, t0: float, t1: float) -> list[Block]:
    e = _entry(path)
    if t0 is None or t1 is None:
        raise VideoToolError("t0 and t1 are required (seconds)")
    a, b, note = _window(e.info, t0, t1, LISTEN_MAX_WINDOW_S)
    if not e.info.has_audio:
        return [text_block(f"LISTEN {fmt_time(a)}–{fmt_time(b)}: the file has no audio track. [~20 tokens]")]
    e = _with_track(e)
    try:
        e, tr = _with_transcript(e, None)
    except (RuntimeError, OSError):
        tr = transcribe.EMPTY
        note += " (words unavailable: transcription failed)"
    text = tools.listen_text(_ctx(e, tr), a, b) + note
    return [text_block(f"{text}\n[~{estimate_text_tokens(text) + 10} tokens]")]


def strip_private(blocks: list[Block]) -> list[Block]:
    """Blocks without internal bookkeeping keys (e.g. '_tokens')."""
    return [{k: v for k, v in b.items() if not k.startswith("_")} for b in blocks]


TOOLS: dict[str, Callable[..., list[Block]]] = {
    "video_info": video_info,
    "video_transcript": video_transcript,
    "video_signals": video_signals,
    "video_overview": video_overview,
    "video_seek": video_seek,
    "video_crop": video_crop,
    "video_listen": video_listen,
}
