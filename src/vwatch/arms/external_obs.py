"""Best-effort reconstruction of which video frames an external agent actually looked at.

Sources, in stream order:
- `ffmpeg` commands the agent ran (CC): output file name -> timestamp, from -ss/-t/-to and fps=/-r.
  Montages (tile), select= filters and shell loops with variables are not mapped.
- `Read` of an image whose timestamp is known -> one frame observation.
- claude-video-vision results: every "### Frame at HH:MM:SS" header (whole seconds, so the
  observation spans [s, s+1)).
- popcorn_analyze results: the inline frames it returned (same even selection as popcorn) and the
  keyframe paths it wrote, which count once the agent Reads them.

These are frames the agent received, not frames it necessarily paid attention to.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from pathlib import Path

from vwatch.schemas import Observation

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_CVV_FRAME = re.compile(r"###\s*Frame at (\d{1,2}):(\d{2}):(\d{2})")
_SEGMENT_SPLIT = re.compile(r"&&|\|\|?|;|\n")
_UNMAPPABLE_FILTERS = ("select", "tile", "thumbnail", "framestep")
MAX_MAPPED_FRAMES = 5000


def parse_time(s: str) -> float | None:
    try:
        parts = [float(p) for p in s.strip().split(":")]
    except ValueError:
        return None
    if not parts or len(parts) > 3 or any(math.isnan(p) for p in parts):
        return None
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def _rate(s: str) -> float | None:
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _key(path: str, workdir: Path) -> str:
    return os.path.realpath(path if os.path.isabs(path) else os.path.join(str(workdir), path))


def _ffmpeg_outputs(argv: list[str], workdir: Path, duration: float) -> dict[str, float]:
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv) - 1:
        if argv[i].startswith("-"):
            opts.setdefault(argv[i], argv[i + 1])
        i += 1
    out = argv[-1] if argv else ""
    if not out.lower().endswith(_IMAGE_EXT):
        return {}
    vf = opts.get("-vf") or opts.get("-filter:v") or opts.get("-filter_complex") or ""
    if any(f in vf for f in _UNMAPPABLE_FILTERS):
        return {}
    ss = parse_time(opts.get("-ss", "0")) or 0.0
    t, to = opts.get("-t"), opts.get("-to")
    end = ss + (parse_time(t) or 0.0) if t else (parse_time(to) if to else None)
    end = duration if end is None else min(end, duration) if duration > 0 else end
    m = re.search(r"(?:^|[,;\s])fps=(?:fps=)?([0-9.]+(?:/[0-9.]+)?)", vf)
    rate = _rate(m.group(1)) if m else (_rate(opts["-r"]) if "-r" in opts else None)
    if "%" not in out:
        return {_key(out, workdir): round(ss, 3)}
    if not rate or rate <= 0 or end is None or end <= ss:
        return {}
    cap = opts.get("-frames:v") or opts.get("-vframes") or opts.get("-frames")
    n = int(math.floor((end - ss) * rate + 1e-6)) or 1
    if cap and cap.isdigit():
        n = min(n, int(cap))
    mapped: dict[str, float] = {}
    for k in range(1, min(n, MAX_MAPPED_FRAMES) + 1):
        try:
            name = out % k
        except (TypeError, ValueError):
            return {}
        mapped[_key(name, workdir)] = round(ss + (k - 1) / rate, 3)
    return mapped


def ffmpeg_frame_map(command: str, workdir: Path, duration: float) -> dict[str, float]:
    """Image file -> video timestamp for every ffmpeg invocation in a shell command we can read."""
    mapped: dict[str, float] = {}
    for seg in _SEGMENT_SPLIT.split(command):
        try:
            argv = shlex.split(seg)
        except ValueError:
            continue
        if argv and os.path.basename(argv[0]) == "ffmpeg":
            mapped.update(_ffmpeg_outputs(argv[1:], workdir, duration))
    return mapped


def message_blocks(event: dict) -> list[dict]:
    """Content blocks of an assistant/user event; [] for anything else (system events, odd shapes)."""
    if event.get("type") not in ("assistant", "user"):
        return []
    msg = event.get("message")
    content = msg.get("content") if isinstance(msg, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _result_texts(block: dict) -> list[str]:
    content = block.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
    return []


def _even_pick(n: int, count: int) -> list[int]:
    """popcorn's selectRepresentativeFrames, on indices."""
    if n <= count:
        return list(range(n))
    if count <= 1:
        return [0]
    step = (n - 1) / (count - 1)
    return [round(i * step) for i in range(count)]


def _popcorn(texts: list[str], workdir: Path, frame_map: dict[str, float]) -> list[float]:
    for text in texts:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("keyframes"), dict):
            continue
        frames = [f for f in obj["keyframes"].get("frames") or [] if isinstance(f, dict)]
        for f in frames:
            if isinstance(f.get("path"), str) and isinstance(f.get("timeSec"), (int, float)):
                frame_map[_key(f["path"], workdir)] = float(f["timeSec"])
        inline = obj.get("inlineFrames") if isinstance(obj.get("inlineFrames"), dict) else {}
        count = int(inline.get("count") or 0)
        return [float(frames[i].get("timeSec", 0.0)) for i in _even_pick(len(frames), count)] if count else []
    return []


def infer_observations(events: list[dict], workdir: Path, duration: float) -> tuple[Observation, ...]:
    frame_map: dict[str, float] = {}
    names: dict[str, str] = {}
    spans: set[tuple[float, float]] = set()
    for ev in events:
        for block in message_blocks(ev):
            if block.get("type") == "tool_use":
                names[str(block.get("id", ""))] = str(block.get("name", ""))
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                if block.get("name") == "Bash":
                    frame_map.update(ffmpeg_frame_map(str(inp.get("command", "")), workdir, duration))
                elif block.get("name") == "Read" and isinstance(inp.get("file_path"), str):
                    t = frame_map.get(_key(inp["file_path"], workdir))
                    if t is not None:
                        spans.add((t, t))
            elif block.get("type") == "tool_result" and not block.get("is_error"):
                texts = _result_texts(block)
                name = names.get(str(block.get("tool_use_id", "")), "")
                for m in (_CVV_FRAME.finditer(t) for t in texts):
                    for hit in m:
                        s = int(hit.group(1)) * 3600 + int(hit.group(2)) * 60 + int(hit.group(3))
                        spans.add((float(s), round(s + 0.999, 3)))
                if name.endswith("popcorn_analyze"):
                    spans.update((t, t) for t in _popcorn(texts, workdir, frame_map))
    return tuple(Observation(kind="frame", t0=a, t1=b) for a, b in sorted(spans))
