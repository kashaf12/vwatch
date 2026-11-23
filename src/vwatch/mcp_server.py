"""MCP server: vwatch's video tools for any agent (Claude Code, Claude Desktop, Cursor, ...).

    vwatch-mcp            stdio server (what MCP clients launch)
    vw mcp                same

The tools make no model calls and need no API key: they turn a local video file into what an agent
can look at (timestamped frame grids, crops, a transcript, a loudness meter, model-free signal hints).
The host agent does the thinking. Needs the `mcp` extra: `pip install 'vwatch[mcp]'`.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import Field

from vwatch import __version__, mcp_tools
from vwatch import media, signals
from vwatch.media import FFmpegNotFoundError, MediaError

log = logging.getLogger("vwatch.mcp")

INSTRUCTIONS = """vwatch: look at local video files. Tools return timestamped evidence; no model runs inside them.
Method: video_info -> video_transcript (if speech) and video_signals (free, model-free hints) -> video_overview
(skim) -> video_seek / video_crop / video_listen only on moments that surprise you or that the question
needs. Every image has its time burned in (mm:ss.ss); cite those times. Do not describe moments you did
not look at. Each result states its approximate token cost."""

Path_ = Annotated[str, Field(description="Local video file. Prefer an absolute path: relative paths resolve "
                                         "against the server's working directory, which may not be yours.")]
Seconds = Annotated[float, Field(description="Seconds from the start of the video.", ge=0)]

DESCRIPTIONS = {
    "video_info": "Duration, fps, size, whether there is audio and speech (cheap voice-activity check), and the "
                  "token cost of an overview and a seek. Call first.",
    "video_transcript": "Timestamped speech transcript (local whisper, cached; the first call on a file can take a "
                        "while, and the first ever call downloads the whisper model). Optional t0/t1 window and "
                        "word-level timestamps.",
    "video_signals": "Model-free candidate moments computed from every frame and the audio: flash, freeze, "
                     "jump_cut, cut, cut_in_speech, silence_in_speech, local_change (score 0..1). Hints, not "
                     "verdicts: they say where to look. Text only, cheap.",
    "video_overview": "Low-resolution tiled frame grids of the whole video with timestamps burned in. Short clips are "
                      "shown whole at watchable size; longer ones are skimmed (capped at 96 frames / ~10k tokens). "
                      "Optional fps and tile_width (90-360 px) override the automatic choice.",
    "video_seek": "Dense consecutive frames for a window (default 8 fps, at most 16 frames per call, window capped "
                  "at 60 s). Use on a moment you need to see frame by frame: a flash, freeze, jump cut, an action.",
    "video_crop": "One frame at time t, cropped to a normalized box [x0, y0, x1, y1] (0..1, (0,0) top-left) at "
                  "source resolution. Use to read small text or check a detail.",
    "video_listen": "Loudness meter for a window (dB every 0.1 s, 'sil' = silence; window capped at 30 s) plus the "
                    "words heard with their times. Use to check audio dropouts, pops, or what was said at a moment.",
}


def _to_content(blocks: list[dict]) -> list[Any]:
    from mcp_types import ImageContent, TextContent

    out: list[Any] = []
    for b in mcp_tools.strip_private(blocks):
        if b["type"] == "image":
            src = b["source"]
            out.append(ImageContent(type="image", data=src["data"], mime_type=src["media_type"]))
        else:
            out.append(TextContent(type="text", text=b["text"]))
    return out


def _call(name: str, fn: Callable[..., list[dict]], **kwargs: Any) -> list[Any]:
    """Run a tool; turn expected failures into a clean tool error instead of a crash."""
    from mcp.server.mcpserver.exceptions import ToolError

    try:
        return _to_content(fn(**kwargs))
    except mcp_tools.VideoToolError as exc:
        raise ToolError(str(exc)) from exc
    except MediaError as exc:
        raise ToolError(f"ffmpeg could not read the video: {str(exc)[-400:]}") from exc
    except FFmpegNotFoundError as exc:  # no ffmpeg/ffprobe anywhere; the message says how to fix it
        raise ToolError(str(exc)) from exc
    except FileNotFoundError as exc:  # a configured binary or the file vanished mid-call
        raise ToolError(f"{exc}. vwatch needs ffmpeg and ffprobe (on PATH, or via VWATCH_FFMPEG/VWATCH_FFPROBE).") from exc
    except Exception as exc:
        log.exception("%s failed", name)
        raise ToolError(f"{name} failed: {type(exc).__name__}: {exc}") from exc


def build_server():
    """The MCPServer with all vwatch tools registered."""
    from mcp.server.mcpserver import MCPServer
    from mcp_types import ToolAnnotations

    server = MCPServer(name="vwatch", version=__version__, instructions=INSTRUCTIONS,
                       website_url="https://github.com/kashaf12/vwatch")
    ro = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False, idempotent_hint=True)

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        server.add_tool(fn, name=fn.__name__, description=DESCRIPTIONS[fn.__name__], annotations=ro,
                        structured_output=False)
        return fn

    @tool
    def video_info(path: Path_) -> list[Any]:
        return _call("video_info", mcp_tools.video_info, path=path)

    @tool
    def video_transcript(
        path: Path_,
        language: Annotated[str | None, Field(description="Speech language code, e.g. 'en', 'hi'. Default: auto.")] = None,
        t0: Annotated[float | None, Field(description="Window start (s). Default: start.", ge=0)] = None,
        t1: Annotated[float | None, Field(description="Window end (s). Default: end.", ge=0)] = None,
        words: Annotated[bool, Field(description="Also list every word with its start time.")] = False,
    ) -> list[Any]:
        return _call("video_transcript", mcp_tools.video_transcript, path=path, language=language, t0=t0, t1=t1,
                     words=words)

    @tool
    def video_signals(
        path: Path_,
        kinds: Annotated[list[str] | None, Field(description=f"Only these kinds ({', '.join(signals.KINDS)}), "
                                                             "e.g. ['flash', 'jump_cut']. Default: all.")] = None,
        top: Annotated[int, Field(description="How many highest-scoring moments to return (1-200).", ge=1,
                                  le=200)] = 40,
    ) -> list[Any]:
        return _call("video_signals", mcp_tools.video_signals, path=path, kinds=kinds, top=top)

    @tool
    def video_overview(
        path: Path_,
        fps: Annotated[float | None, Field(description="Frames per second to sample. Default: automatic.",
                                           gt=0, le=30)] = None,
        tile_width: Annotated[int | None, Field(description="Tile width in px (90-360). Default: automatic.",
                                                ge=90, le=360)] = None,
    ) -> list[Any]:
        return _call("video_overview", mcp_tools.video_overview, path=path, fps=fps, tile_width=tile_width)

    @tool
    def video_seek(
        path: Path_, t0: Seconds, t1: Seconds,
        fps: Annotated[float, Field(description="Frames per second in the window (0.5-30).", ge=0.5, le=30)] = 8.0,
    ) -> list[Any]:
        return _call("video_seek", mcp_tools.video_seek, path=path, t0=t0, t1=t1, fps=fps)

    @tool
    def video_crop(
        path: Path_, t: Seconds,
        box: Annotated[list[float], Field(description="[x0, y0, x1, y1], normalized 0..1, (0,0) = top-left.",
                                          min_length=4, max_length=4)],
    ) -> list[Any]:
        return _call("video_crop", mcp_tools.video_crop, path=path, t=t, box=box)

    @tool
    def video_listen(path: Path_, t0: Seconds, t1: Seconds) -> list[Any]:
        return _call("video_listen", mcp_tools.video_listen, path=path, t0=t0, t1=t1)

    return server


def _prefetch_ffmpeg() -> threading.Thread:
    """Resolve ffmpeg/ffprobe in the background so a first-run static download overlaps the agent's setup
    instead of stalling its first tool call. Failures are only logged: the tool call reports them properly."""

    def work() -> None:
        try:
            media.ffmpeg()
            media.ffprobe()
        except Exception as exc:  # noqa: BLE001 - surfaced again, with advice, by the first tool call
            log.warning("ffmpeg not available yet: %s", exc)

    thread = threading.Thread(target=work, name="vwatch-ffmpeg-prefetch", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] in ("-V", "--version"):
        print(__version__)
        return 0
    try:
        server = build_server()
    except ImportError as exc:
        print(f"error: the MCP server needs the 'mcp' extra: pip install 'vwatch[mcp]' ({exc})", file=sys.stderr)
        return 1
    _prefetch_ffmpeg()
    from vwatch import mcp_tools
    mcp_tools.PREFETCH = os.environ.get("VWATCH_PREFETCH", "1") != "0"  # signals + voice check start on first touch
    # ... and the transcript too, but only when the speech model is already downloaded
    mcp_tools.PREFETCH_TRANSCRIPT = os.environ.get("VWATCH_PREFETCH_TRANSCRIPT", "1") != "0"
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
