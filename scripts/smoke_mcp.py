"""Fresh-install smoke test: start the vwatch MCP server exactly as the Claude Code plugin does, then call tools.

    uv run --no-project --with "vwatch[mcp] @ file://$PWD" python scripts/smoke_mcp.py

Reads `plugins/vwatch/.claude-plugin/plugin.json`, expands ${CLAUDE_PLUGIN_ROOT}, launches that command over
stdio with the MCP SDK client (environment passed through, so VWATCH_SPEC can point the launcher at a local
checkout), lists the tools, and calls video_info, video_overview and video_signals on a tiny generated video.

On a machine without ffmpeg the server fetches a static build at startup (vwatch.media); this script waits for
that download to finish and reuses it to render the test video, so ffmpeg is downloaded once. Exits 1 on any
failure. Progress goes to stderr, a JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import anyio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "vwatch"
EXPECTED_TOOLS = {"video_info", "video_transcript", "video_signals", "video_overview", "video_seek", "video_crop",
                  "video_listen"}


def say(msg: str) -> None:
    print(f"[smoke {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def plugin_command(plugin_dir: Path) -> tuple[str, list[str]]:
    manifest = json.loads((plugin_dir / ".claude-plugin" / "plugin.json").read_text())
    server = manifest["mcpServers"]["vwatch"]
    expand = lambda s: s.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_dir))  # noqa: E731
    return expand(server["command"]), [expand(a) for a in server["args"]]


def wait_for_static_ffmpeg(timeout_s: float) -> None:
    """Block until the server's one-time static ffmpeg download is complete (no-op if ffmpeg is configured)."""
    if os.environ.get("VWATCH_FFMPEG") or (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return
    from static_ffmpeg import run

    from vwatch.media import _cache_root

    crumb = _cache_root() / "ffmpeg" / run.get_platform_key() / "installed.crumb"
    say(f"no system ffmpeg: waiting for the server's static ffmpeg download ({crumb.parent})")
    deadline = time.monotonic() + timeout_s
    while not crumb.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"static ffmpeg not downloaded after {timeout_s:.0f} s ({crumb} missing); "
                               "see the server's 'ffmpeg not available yet' warning above for why")
        time.sleep(1)


def make_video(path: Path) -> Path:
    """3 s, 160x120, 30 fps: a moving square, a white flash frame at 1.5 s, and a 440 Hz tone."""
    from vwatch import media

    n, fps, h, w = 90, 30, 120, 160
    frames = np.full((n, h, w, 3), 60, dtype=np.uint8)
    for k in range(n):
        x = int(k / (n - 1) * (w - 24))
        frames[k, h // 2 - 12:h // 2 + 12, x:x + 24] = (230, 60, 60)
    frames[45] = 255
    sr = 44100
    t = np.arange(3 * sr) / sr
    return media.write_video(frames, fps, path, audio=(0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr=sr)


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


async def smoke(args: argparse.Namespace) -> dict:
    from mcp import Client, StdioServerParameters

    command, cmd_args = plugin_command(args.plugin_dir)
    say(f"launching: {command} {' '.join(cmd_args)}  (VWATCH_SPEC={os.environ.get('VWATCH_SPEC', '<unset>')})")
    params = StdioServerParameters(command=command, args=cmd_args, env=dict(os.environ), cwd=str(args.workdir))
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    with anyio.fail_after(args.timeout):
        async with Client(params) as client:
            timings["server_ready_s"] = round(time.monotonic() - t0, 1)
            names = {t.name for t in (await client.list_tools()).tools}
            say(f"server ready in {timings['server_ready_s']} s; tools: {sorted(names)}")
            if names != EXPECTED_TOOLS:
                raise AssertionError(f"tool list mismatch: {sorted(names)}")

            t = time.monotonic()
            await anyio.to_thread.run_sync(wait_for_static_ffmpeg, args.ffmpeg_wait)
            video = await anyio.to_thread.run_sync(make_video, args.workdir / "smoke.mp4")
            timings["ffmpeg_ready_and_video_s"] = round(time.monotonic() - t, 1)
            say(f"test video: {video} ({video.stat().st_size} bytes)")

            results = {}
            for name, tool_args in [("video_info", {"path": str(video)}),
                                    ("video_overview", {"path": str(video)}),
                                    ("video_signals", {"path": str(video)})]:
                t = time.monotonic()
                res = await client.call_tool(name, tool_args)
                timings[f"{name}_s"] = round(time.monotonic() - t, 1)
                body = text_of(res)
                if res.is_error:
                    raise AssertionError(f"{name} returned an error: {body}")
                images = sum(1 for c in res.content if c.type == "image")
                say(f"{name} ok in {timings[f'{name}_s']} s, {images} image(s):\n{body[:600]}")
                results[name] = {"images": images, "text": body[:2000]}

    info = results["video_info"]["text"]
    if "duration: 3.0" not in info or "160x120" not in info:
        raise AssertionError(f"unexpected video_info: {info}")
    if results["video_overview"]["images"] < 1:
        raise AssertionError("video_overview returned no images")
    if "flash" not in results["video_signals"]["text"]:
        raise AssertionError(f"video_signals missed the flash frame: {results['video_signals']['text']}")
    return {"ok": True, "tools": sorted(names), "timings": timings, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plugin-dir", type=Path, default=PLUGIN)
    ap.add_argument("--timeout", type=float, default=900, help="seconds for the whole run (first install is slow)")
    ap.add_argument("--ffmpeg-wait", type=float, default=300,
                    help="seconds to wait for the server's static ffmpeg download (normally ~25 s; if the server's "
                         "download failed for good, the file never appears, so don't wait the whole run)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="vwatch-smoke-") as tmp:
        args.workdir = Path(tmp)
        try:
            summary = anyio.run(smoke, args)
        except Exception as exc:  # noqa: BLE001 - report any failure as a failed smoke run
            while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]  # anyio wraps the real error in a TaskGroup ExceptionGroup
            say(f"FAILED: {type(exc).__name__}: {exc}")
            return 1
    print(json.dumps(summary, indent=2))
    say("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
