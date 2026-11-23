"""The MCP tools (vwatch.mcp_tools), the server wiring (vwatch.mcp_server) and the plugin manifests."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from vwatch import mcp_tools as mt
from vwatch import transcribe
from vwatch.imaging import estimate_image_tokens

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "vwatch"
TOOL_NAMES = ["video_info", "video_transcript", "video_signals", "video_overview", "video_seek", "video_crop",
              "video_listen"]


@pytest.fixture(autouse=True)
def fresh_cache():
    mt.clear_cache()
    yield
    mt.clear_cache()


@pytest.fixture
def fake_whisper(monkeypatch, tiny_transcript):
    """Replace whisper with the fixture transcript; count calls."""
    calls = []

    def fake(path, *, language=None, **_):
        calls.append((str(path), language))
        return tiny_transcript

    monkeypatch.setattr(mt.transcribe, "transcribe", fake)
    return calls


def _texts(blocks):
    return [b["text"] for b in blocks if b["type"] == "text"]


def _images(blocks):
    out = []
    for b in blocks:
        if b["type"] == "image":
            assert b["source"]["media_type"] == "image/jpeg"
            out.append(Image.open(io.BytesIO(base64.b64decode(b["source"]["data"]))))
    return out


def _est(text: str) -> int:
    m = re.search(r"\[~(\d+) tokens\]", text)
    assert m, f"no token estimate in: {text!r}"
    return int(m.group(1))


# ------------------------------------------------------------------ paths and cache


class TestPaths:
    def test_missing_file(self, tmp_path):
        with pytest.raises(mt.VideoToolError, match="no such file"):
            mt.video_info(str(tmp_path / "nope.mp4"))

    def test_directory_refused(self, tmp_path):
        with pytest.raises(mt.VideoToolError, match="not a file"):
            mt.video_info(str(tmp_path))

    @pytest.mark.parametrize("bad", ["", "   ", "https://example.com/a.mp4", "file:///tmp/a.mp4"])
    def test_empty_or_url_refused(self, bad):
        with pytest.raises(mt.VideoToolError):
            mt.video_info(bad)

    def test_not_a_video(self, tmp_path):
        p = tmp_path / "notes.mp4"
        p.write_text("not a video")
        with pytest.raises(mt.VideoToolError, match="not a readable video"):
            mt.video_info(str(p))

    def test_relative_to_cwd(self, tone_video, monkeypatch):
        monkeypatch.chdir(tone_video.parent)
        assert mt.resolve_path(tone_video.name) == tone_video.resolve()

    def test_cache_hit_and_invalidation(self, tone_video, tmp_path, monkeypatch):
        probes = []
        real = mt.media.probe
        monkeypatch.setattr(mt.media, "probe", lambda p: probes.append(p) or real(p))
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 0.0)
        copy = tmp_path / "copy.mp4"
        copy.write_bytes(tone_video.read_bytes())
        mt.video_info(str(copy))
        mt.video_info(str(copy))
        assert len(probes) == 1
        os.utime(copy, ns=(1, 1))  # new mtime -> new key
        mt.video_info(str(copy))
        assert len(probes) == 2

    def test_cache_is_bounded(self, tone_video, tmp_path, monkeypatch):
        monkeypatch.setattr(mt, "CACHE_SIZE", 2)
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 0.0)
        for k in range(3):
            p = tmp_path / f"v{k}.mp4"
            p.write_bytes(tone_video.read_bytes())
            mt.video_info(str(p))
        assert len(mt._cache) == 2


# ------------------------------------------------------------------ tools


class TestInfo:
    def test_tone_video(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 0.0)
        txt = _texts(mt.video_info(str(tone_video)))[0]
        assert "duration: 4.00 s" in txt and "size: 108x192" in txt and "30 fps" in txt
        assert "audio: yes; speech: none detected" in txt
        assert "whole clip" in txt  # a 4 s clip is shown whole by video_overview

    def test_speech_reported(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 3.2)
        assert "speech: yes, ~3.2 s" in _texts(mt.video_info(str(tone_video)))[0]

    def test_vad_failure_is_not_fatal(self, tone_video, monkeypatch):
        def boom(p):
            raise RuntimeError("no onnx")
        monkeypatch.setattr(mt, "_speech_seconds", boom)
        assert "speech: unknown" in _texts(mt.video_info(str(tone_video)))[0]

    def test_silent_video(self, silent_video):
        assert "audio: none" in _texts(mt.video_info(str(silent_video)))[0]


class TestTranscript:
    def test_window_and_cache(self, tone_video, fake_whisper):
        txt = _texts(mt.video_transcript(str(tone_video), t0=1.8, t1=4))[0]
        assert "second line here" in txt and "hello there" not in txt
        assert "language en" in txt and _est(txt) > 0
        mt.video_transcript(str(tone_video))
        assert len(fake_whisper) == 1  # cached in-process

    def test_words(self, tone_video, fake_whisper):
        txt = _texts(mt.video_transcript(str(tone_video), words=True, t0=0, t1=1.2))[0]
        assert "hello@0.20" in txt and "there@0.60" in txt

    def test_language_passed_and_validated(self, tone_video, fake_whisper):
        mt.video_transcript(str(tone_video), language="HI")
        assert fake_whisper[-1][1] == "hi"
        with pytest.raises(mt.VideoToolError, match="language"):
            mt.video_transcript(str(tone_video), language="en; rm -rf")

    def test_no_audio(self, silent_video, fake_whisper):
        assert "no audio track" in _texts(mt.video_transcript(str(silent_video)))[0]
        assert fake_whisper == []

    def test_long_text_truncated(self, tone_video, monkeypatch):
        segs = tuple(transcribe.Segment(k * 0.01, k * 0.01 + 0.01, "x" * 80) for k in range(400))
        tr = transcribe.Transcript("en", segs, ())
        monkeypatch.setattr(mt.transcribe, "transcribe", lambda *a, **k: tr)
        monkeypatch.setattr(mt, "TEXT_MAX_CHARS", 2000)
        txt = _texts(mt.video_transcript(str(tone_video)))[0]
        assert "truncated" in txt and len(txt) < 2300

    def test_bad_window(self, tone_video, fake_whisper):
        with pytest.raises(mt.VideoToolError, match="greater than t0"):
            mt.video_transcript(str(tone_video), t0=2, t1=1)
        with pytest.raises(mt.VideoToolError, match="past the end"):
            mt.video_transcript(str(tone_video), t0=9)


class TestSignals:
    def test_lists_candidates_in_time_order(self, tone_video, monkeypatch):
        from vwatch.signals import Candidate

        cands = [Candidate(2.0, 2.0, "cut", 0.35, "b"), Candidate(1.0, 1.1, "flash", 0.9, "a"),
                 Candidate(3.0, 3.0, "cut", 0.30, "c")]
        monkeypatch.setattr(mt.signals, "find_candidates", lambda track: cands)
        txt = _texts(mt.video_signals(str(tone_video), top=2))[0]
        assert "2 of 3 candidate moments" in txt
        assert txt.index("flash") < txt.index("00:02.00") and "00:03.00" not in txt
        assert _est(txt) > 0

    def test_kinds_filter_and_unknown_kind(self, tone_video, monkeypatch):
        from vwatch.signals import Candidate

        monkeypatch.setattr(mt.signals, "find_candidates",
                            lambda track: [Candidate(1.0, 1.0, "cut", 0.35, "b")])
        txt = _texts(mt.video_signals(str(tone_video), kinds=["freeze"]))[0]
        assert "0 of 1" in txt and "Kinds present in this video: cut" in txt
        assert "no unusual moments" in txt

    def test_real_track(self, tone_video):
        txt = _texts(mt.video_signals(str(tone_video)))[0]
        assert txt.startswith("SIGNALS:")

    def test_unknown_kind_lists_the_valid_ones(self, tone_video):
        with pytest.raises(mt.VideoToolError, match="unknown signal kind.*'jumpcut'.*valid kinds: .*jump_cut"):
            mt.video_signals(str(tone_video), kinds=["flash", "jumpcut"])

    def test_jump_cut_is_a_known_kind(self, tone_video, monkeypatch):
        from vwatch.signals import KINDS, Candidate

        assert "jump_cut" in KINDS
        monkeypatch.setattr(mt.signals, "find_candidates",
                            lambda track: [Candidate(1.0, 1.0, "jump_cut", 0.8, "splice"),
                                           Candidate(2.0, 2.0, "cut", 0.4, "b")])
        txt = _texts(mt.video_signals(str(tone_video), kinds=["jump_cut"]))[0]
        assert "1 of 2" in txt and "jump_cut" in txt and "00:02.00" not in txt

    def test_server_describes_every_kind(self):
        pytest.importorskip("mcp")
        import anyio

        from vwatch.mcp_server import DESCRIPTIONS, build_server
        from vwatch.signals import KINDS

        tool = next(t for t in anyio.run(build_server().list_tools) if t.name == "video_signals")
        kinds_doc = tool.input_schema["properties"]["kinds"]["description"]
        assert all(k in DESCRIPTIONS["video_signals"] and k in kinds_doc for k in KINDS)

    @pytest.mark.parametrize("bad", [{"top": 0}, {"top": 1000}, {"kinds": "flash"}])
    def test_bad_args(self, tone_video, bad):
        with pytest.raises(mt.VideoToolError):
            mt.video_signals(str(tone_video), **bad)


class TestOverview:
    def test_short_clip_shown_whole(self, tone_video):
        blocks = mt.video_overview(str(tone_video))
        head = _texts(blocks)[0]
        assert head.startswith("video_overview: FULL VIEW")
        imgs = _images(blocks)
        assert 1 <= len(imgs) <= 8
        est = _est(head)
        assert est <= mt.OVERVIEW_MAX_TOKENS
        assert est >= sum(estimate_image_tokens(*im.size) for im in imgs)

    def test_skim_with_explicit_density(self, tone_video):
        blocks = mt.video_overview(str(tone_video), fps=2, tile_width=90)
        head = _texts(blocks)[0]
        assert "OVERVIEW (skim), 8 frames" in head and "tile 90px" in head
        assert all(im.width <= 4 * 90 + 3 * 4 for im in _images(blocks))

    def test_frame_cap(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "OVERVIEW_MAX_FRAMES", 10)
        head = _texts(mt.video_overview(str(tone_video), fps=30))[0]
        assert "capped at 10 frames" in head
        assert int(re.search(r"(\d+) frames", head).group(1)) <= 10

    def test_token_cap(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "OVERVIEW_MAX_TOKENS", 800)
        head = _texts(mt.video_overview(str(tone_video), fps=8, tile_width=360))[0]
        assert "density reduced" in head

    @pytest.mark.parametrize("bad", [{"fps": 0}, {"fps": -1}, {"tile_width": 20}, {"tile_width": 4000},
                                     {"fps": "fast"}])
    def test_bad_args(self, tone_video, bad):
        with pytest.raises(mt.VideoToolError):
            mt.video_overview(str(tone_video), **bad)


class TestSeek:
    def test_window(self, tone_video):
        blocks = mt.video_seek(str(tone_video), 1.0, 2.0, fps=8)
        head = _texts(blocks)[0]
        assert head.startswith("video_seek: 9 frames 00:01.00–00:02.00")
        assert _images(blocks) and _est(head) > 0

    def test_frame_cap(self, tone_video):
        head = _texts(mt.video_seek(str(tone_video), 0, 3.9, fps=30))[0]
        assert "16 frames" in head and "Capped at 16 frames" in head

    def test_window_clipped_to_video_and_cap(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "SEEK_MAX_WINDOW_S", 1.0)
        head = _texts(mt.video_seek(str(tone_video), 2.0, 50.0, fps=4))[0]
        assert "window capped at 1 s" in head and "00:03.00" in head

    @pytest.mark.parametrize("t0,t1,fps,msg", [(5, 6, 8, "past the end"), (2, 1, 8, "greater than t0"),
                                                (1, 2, 0.1, "fps"), (None, 2, 8, "required")])
    def test_bad_args(self, tone_video, t0, t1, fps, msg):
        with pytest.raises(mt.VideoToolError, match=msg):
            mt.video_seek(str(tone_video), t0, t1, fps)


class TestCrop:
    def test_crop_at_source_resolution(self, tone_video):
        blocks = mt.video_crop(str(tone_video), 2.0, [0.0, 0.5, 1.0, 1.0])
        head = _texts(blocks)[0]
        assert "frame at 00:02.00" in head and "decoded at 108px" in head
        (img,) = _images(blocks)
        assert img.size == (108, 96)
        assert "_tokens" not in mt.strip_private(blocks)[1]

    def test_large_crop_downscaled(self, tone_video, monkeypatch):
        monkeypatch.setattr(mt, "CROP_OUT_MAX_EDGE", 64)
        (img,) = _images(mt.video_crop(str(tone_video), 1.0, [0, 0, 1, 1]))
        assert max(img.size) <= 64

    @pytest.mark.parametrize("t,box,msg", [(1, [0.5, 0.5, 0.2, 0.9], "x0 < x1"), (1, [0, 0, 1], r"\[x0"),
                                           (1, [0, 0, 1.5, 1], "<= 1.0"), (9, [0, 0, 1, 1], "past the end"),
                                           (1, "0,0,1,1", r"\[x0")])
    def test_bad_args(self, tone_video, t, box, msg):
        with pytest.raises(mt.VideoToolError, match=msg):
            mt.video_crop(str(tone_video), t, box)


class TestListen:
    def test_meter_and_words(self, tone_video, fake_whisper):
        txt = _texts(mt.video_listen(str(tone_video), 0.0, 1.2))[0]
        assert txt.startswith("LISTEN 00:00.00–00:01.20")
        meter = txt.split("silence): ")[1].split("\n")[0].split()
        assert len(meter) == 13 and "sil" not in meter  # a steady tone is never silent
        assert "hello@0.20" in txt and _est(txt) > 0

    def test_window_cap(self, tone_video, fake_whisper, monkeypatch):
        monkeypatch.setattr(mt, "LISTEN_MAX_WINDOW_S", 1.0)
        assert "window capped" in _texts(mt.video_listen(str(tone_video), 0, 3))[0]

    def test_no_audio(self, silent_video, fake_whisper):
        assert "no audio track" in _texts(mt.video_listen(str(silent_video), 0, 1))[0]

    def test_whisper_failure_still_gives_meter(self, tone_video, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("could not load whisper")
        monkeypatch.setattr(mt.transcribe, "transcribe", boom)
        txt = _texts(mt.video_listen(str(tone_video), 0, 1))[0]
        assert "words unavailable" in txt and "dB" in txt


def test_tool_table_complete():
    assert sorted(mt.TOOLS) == sorted(TOOL_NAMES)


# ------------------------------------------------------------------ MCP server


class TestServer:
    @pytest.fixture(autouse=True)
    def _needs_mcp(self):
        pytest.importorskip("mcp")

    def test_lists_tools_with_schemas(self):
        import anyio

        from vwatch.mcp_server import build_server

        tools = anyio.run(build_server().list_tools)
        assert sorted(t.name for t in tools) == sorted(TOOL_NAMES)
        by = {t.name: t for t in tools}
        assert by["video_crop"].input_schema["required"] == ["path", "t", "box"]
        assert all(t.annotations.read_only_hint for t in tools)
        assert all(t.description for t in tools)

    def test_call_returns_image_content(self, tone_video):
        import anyio

        from vwatch.mcp_server import build_server

        res = anyio.run(build_server().call_tool, "video_seek", {"path": str(tone_video), "t0": 1, "t1": 1.5})
        kinds = [c.type for c in res.content]
        assert kinds[0] == "text" and "image" in kinds and not res.is_error
        img = next(c for c in res.content if c.type == "image")
        assert img.mime_type == "image/jpeg"
        Image.open(io.BytesIO(base64.b64decode(img.data)))

    def test_bad_input_is_a_tool_error_not_a_crash(self, tmp_path):
        from mcp.server.mcpserver.exceptions import ToolError

        from vwatch import mcp_server

        with pytest.raises(ToolError, match="no such file"):
            mcp_server._call("video_info", mt.video_info, path=str(tmp_path / "x.mp4"))

        def boom(**_):
            raise KeyError("weird")
        with pytest.raises(ToolError, match="video_info failed: KeyError"):
            mcp_server._call("video_info", boom)

    def test_missing_ffmpeg_is_a_tool_error_with_advice(self):
        from mcp.server.mcpserver.exceptions import ToolError

        from vwatch import mcp_server
        from vwatch.media import FFmpegNotFoundError

        def no_ffmpeg(**_):
            raise FFmpegNotFoundError("ffmpeg/ffprobe not found. Install ffmpeg (macOS: `brew install ffmpeg`)")
        with pytest.raises(ToolError, match=r"^ffmpeg/ffprobe not found\. Install ffmpeg"):
            mcp_server._call("video_info", no_ffmpeg)

    def test_prefetch_never_raises(self, monkeypatch):
        from vwatch import mcp_server
        from vwatch.media import FFmpegNotFoundError

        def missing():
            raise FFmpegNotFoundError("nothing here")
        monkeypatch.setattr(mcp_server.media, "ffmpeg", missing)
        thread = mcp_server._prefetch_ffmpeg()
        thread.join(5)
        assert not thread.is_alive() and thread.daemon

    def test_stdio_roundtrip(self, tmp_path):
        """Start the real server over stdio with the SDK client, list tools, call one with a bad path."""
        import anyio
        from mcp import Client, StdioServerParameters

        async def run():
            params = StdioServerParameters(command=sys.executable, args=["-m", "vwatch.mcp_server"],
                                           cwd=str(tmp_path))
            with anyio.fail_after(60):
                async with Client(params) as client:
                    names = [t.name for t in (await client.list_tools()).tools]
                    res = await client.call_tool("video_info", {"path": "missing.mp4"})
                    return names, res

        try:
            names, res = anyio.run(run)
        except OSError as exc:  # pragma: no cover - sandboxed CI without subprocess pipes
            pytest.skip(f"cannot spawn the stdio server here: {exc}")
        assert sorted(names) == sorted(TOOL_NAMES)
        assert res.is_error and "no such file" in res.content[0].text

    def test_cli_help_and_version(self, capsys):
        from vwatch import __version__
        from vwatch.cli import main

        assert main(["mcp", "--version"]) == 0
        assert capsys.readouterr().out.strip() == __version__


# ------------------------------------------------------------------ plugin + marketplace


class TestPlugin:
    def test_versions_agree(self):
        """One version everywhere: pyproject, package, plugin.json and the git tag the plugin launches."""
        from vwatch import __version__

        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        args = " ".join(manifest["mcpServers"]["vwatch"]["args"])
        assert __version__ == version == manifest["version"]
        assert f"github.com/kashaf12/vwatch@v{version}" in args and "vwatch[mcp]" in args
        assert args.endswith("vwatch-mcp")

    def test_marketplace_points_at_plugin(self):
        market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
        (entry,) = market["plugins"]
        manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        assert market["name"] == "vwatch" and entry["name"] == manifest["name"] == "vwatch"
        assert (ROOT / entry["source"]).resolve() == PLUGIN.resolve()

    def test_skill_and_command(self):
        skill = (PLUGIN / "skills" / "watch-video" / "SKILL.md").read_text()
        front = skill.split("---")[1]
        assert "name: watch-video" in front and "description:" in front and "when_to_use:" in front
        for name in TOOL_NAMES:
            assert f"`{name}" in skill
        cmd = (PLUGIN / "commands" / "watch.md").read_text()
        assert "$ARGUMENTS" in cmd
        allowed = cmd.split("allowed-tools:")[1].split("\n")[0]
        assert all(f"mcp__plugin_vwatch_vwatch__{n}" in allowed for n in TOOL_NAMES)


# ------------------------------------------------------------------ plugin launcher (plugins/vwatch/bin/vwatch-mcp)

LAUNCHER = PLUGIN / "bin" / "vwatch-mcp"


@pytest.mark.skipif(sys.platform == "win32", reason="the launcher is a POSIX sh script")
class TestLauncher:
    """The launcher only finds/installs uv and forwards plugin.json's args to uvx; a fake uvx records them."""

    @staticmethod
    def manifest_args() -> list[str]:
        manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        return manifest["mcpServers"]["vwatch"]["args"]

    @staticmethod
    def fake_uvx(bindir: Path) -> Path:
        bindir.mkdir(parents=True, exist_ok=True)
        uvx = bindir / "uvx"
        uvx.write_text('#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done > "$UVX_LOG"\n')
        uvx.chmod(0o755)
        return bindir

    def run(self, tmp_path, args, *, path_dirs=(), **env):
        import subprocess

        log = tmp_path / "uvx-args.txt"
        base = {"HOME": str(tmp_path / "home"), "PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"]),
                "UVX_LOG": str(log)}
        proc = subprocess.run(["sh", str(LAUNCHER), *args], env={**base, **env}, capture_output=True, text=True,
                              timeout=30)
        got = log.read_text().splitlines() if log.exists() else None
        return proc, got

    def test_plugin_json_runs_the_launcher_via_sh(self):
        manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        server = manifest["mcpServers"]["vwatch"]
        assert server["command"] == "sh"  # no reliance on the exec bit surviving the plugin cache copy
        assert server["args"][0] == "${CLAUDE_PLUGIN_ROOT}/bin/vwatch-mcp"
        assert LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)

    def test_version_lives_only_in_plugin_json(self):
        """The launcher forwards plugin.json's pinned `--from` spec, so it carries no version that could drift."""
        assert not re.search(r"@v\d+\.\d+", LAUNCHER.read_text())

    def test_forwards_plugin_args_to_uvx_on_path(self, tmp_path):
        args = self.manifest_args()[1:]
        proc, got = self.run(tmp_path, args, path_dirs=[self.fake_uvx(tmp_path / "bin")])
        assert proc.returncode == 0, proc.stderr
        assert got == args and proc.stdout == ""

    def test_finds_uvx_in_home_local_bin(self, tmp_path):
        self.fake_uvx(tmp_path / "home" / ".local" / "bin")
        proc, got = self.run(tmp_path, ["--from", "x", "vwatch-mcp"])
        assert proc.returncode == 0, proc.stderr
        assert got == ["--from", "x", "vwatch-mcp"]

    def test_finds_uvx_in_cargo_bin(self, tmp_path):
        self.fake_uvx(tmp_path / "home" / ".cargo" / "bin")
        proc, got = self.run(tmp_path, ["--from", "x", "vwatch-mcp"])
        assert proc.returncode == 0, proc.stderr
        assert got == ["--from", "x", "vwatch-mcp"]

    def test_vwatch_spec_overrides_the_pinned_spec(self, tmp_path):
        spec = "vwatch[mcp] @ file:///src/vwatch"
        proc, got = self.run(tmp_path, self.manifest_args()[1:], path_dirs=[self.fake_uvx(tmp_path / "bin")],
                             VWATCH_SPEC=spec)
        assert proc.returncode == 0, proc.stderr
        expected = self.manifest_args()[1:]
        expected[expected.index("--from") + 1] = spec
        assert got == expected and "--python" in got  # only the spec changes; uvx options stay

    def test_vwatch_spec_without_from_is_added(self, tmp_path):
        proc, got = self.run(tmp_path, ["--python", "3.12", "vwatch-mcp", "--version"],
                             path_dirs=[self.fake_uvx(tmp_path / "bin")], VWATCH_SPEC="vwatch[mcp] @ file:///x y")
        assert proc.returncode == 0, proc.stderr
        assert got == ["--from", "vwatch[mcp] @ file:///x y", "--python", "3.12", "vwatch-mcp", "--version"]

    def test_plugin_pins_a_python_with_wheels_for_every_dependency(self):
        """uv would otherwise pick the newest CPython, which onnxruntime (faster-whisper) may not have wheels for."""
        args = self.manifest_args()
        python = args[args.index("--python") + 1]
        assert python == ">=3.11,<3.13"

    def test_vwatch_spec_without_args(self, tmp_path):
        proc, got = self.run(tmp_path, [], path_dirs=[self.fake_uvx(tmp_path / "bin")], VWATCH_SPEC="vwatch[mcp]")
        assert proc.returncode == 0, proc.stderr
        assert got == ["--from", "vwatch[mcp]", "vwatch-mcp"]

    def test_no_args_and_no_spec_is_a_usage_error(self, tmp_path):
        proc, got = self.run(tmp_path, [], path_dirs=[self.fake_uvx(tmp_path / "bin")])
        assert proc.returncode == 2 and "usage" in proc.stderr and got is None and proc.stdout == ""

    def test_no_uv_and_install_disabled_fails_cleanly(self, tmp_path):
        import shutil

        if shutil.which("uvx", path="/usr/bin:/bin"):
            pytest.skip("uvx is installed system-wide here")
        proc, got = self.run(tmp_path, ["--from", "x", "vwatch-mcp"], VWATCH_UV_INSTALL="0")
        assert proc.returncode == 127 and got is None
        assert proc.stdout == "" and "could not find or install uv" in proc.stderr


class TestConcurrencyAndPrefetch:
    """Parallel tool calls share one computation and never drop each other's results; the server prefetches."""

    @staticmethod
    def slow_track(monkeypatch, calls, delay=0.3):
        import time
        real = mt.signals.compute

        def fake(path):
            calls.append(str(path))
            time.sleep(delay)
            return real(path)
        monkeypatch.setattr(mt.signals, "compute", fake)

    def test_parallel_signal_calls_compute_once(self, tone_video, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        calls = []
        self.slow_track(monkeypatch, calls)
        with ThreadPoolExecutor(3) as ex:
            outs = list(ex.map(lambda _: mt.video_signals(str(tone_video)), range(3)))
        assert len(calls) == 1 and len({o[0]["text"] for o in outs}) == 1

    def test_parallel_info_does_not_drop_the_track(self, tone_video, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor
        import time
        calls = []
        self.slow_track(monkeypatch, calls, delay=0.1)
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: time.sleep(0.4) or 1.0)  # finishes after the track
        with ThreadPoolExecutor(2) as ex:
            a = ex.submit(mt.video_info, str(tone_video))
            b = ex.submit(mt.video_signals, str(tone_video))
            a.result(), b.result()
        mt.video_signals(str(tone_video))
        assert len(calls) == 1  # the info call's late update kept the track

    def test_prefetch_starts_track_and_transcript(self, tone_video, monkeypatch, fake_whisper):
        calls = []
        self.slow_track(monkeypatch, calls, delay=0.05)
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 1.0)
        monkeypatch.setattr(mt, "PREFETCH", True)
        mt.video_info(str(tone_video))
        mt.wait_prefetch(timeout=10)
        mt.video_signals(str(tone_video))
        mt.video_transcript(str(tone_video))
        assert len(calls) == 1 and len(fake_whisper) == 1

    def test_no_prefetch_by_default(self, tone_video, monkeypatch, fake_whisper):
        calls = []
        self.slow_track(monkeypatch, calls, delay=0.0)
        monkeypatch.setattr(mt, "_speech_seconds", lambda p: 1.0)
        mt.video_info(str(tone_video))
        mt.wait_prefetch(timeout=1)
        assert calls == [] and fake_whisper == []
