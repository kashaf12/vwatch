"""Where vwatch finds ffmpeg/ffprobe: env override -> system PATH -> static-ffmpeg download (all faked here)."""

from __future__ import annotations

import os
import stat
import sys
import types
from pathlib import Path

import pytest

from vwatch import media

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake binaries are POSIX shell scripts")


@pytest.fixture(autouse=True)
def fresh_resolution():
    media.reset_binaries()
    yield
    media.reset_binaries()


def fake_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def no_system_ffmpeg(monkeypatch, tmp_path):
    """PATH with no ffmpeg/ffprobe and no VWATCH_FFMPEG/VWATCH_FFPROBE."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("VWATCH_FFMPEG", raising=False)
    monkeypatch.delenv("VWATCH_FFPROBE", raising=False)
    monkeypatch.setenv("VWATCH_CACHE", str(tmp_path / "cache"))
    # never hit the network: tests that want the static fallback install a fake `static_ffmpeg`
    monkeypatch.setitem(sys.modules, "static_ffmpeg", None)
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", None)
    return tmp_path


def install_fake_static_ffmpeg(monkeypatch, fetch):
    """Register a fake `static_ffmpeg.run` whose fetch is `fetch(download_dir) -> (ffmpeg, ffprobe)`."""
    run = types.ModuleType("static_ffmpeg.run")
    run.get_platform_key = lambda: "testplat"
    run.get_or_fetch_platform_executables_else_raise = lambda fix_permissions=True, download_dir=None: fetch(
        download_dir)
    pkg = types.ModuleType("static_ffmpeg")
    pkg.run = run
    monkeypatch.setitem(sys.modules, "static_ffmpeg", pkg)
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", run)


class TestEnvOverride:
    def test_env_path_wins_over_path(self, no_system_ffmpeg, monkeypatch):
        custom = fake_exe(no_system_ffmpeg / "custom" / "my-ffmpeg")
        fake_exe(Path(os.environ["PATH"]) / "ffmpeg")
        monkeypatch.setenv("VWATCH_FFMPEG", str(custom))
        assert media.ffmpeg() == str(custom)

    def test_env_name_is_looked_up_on_path(self, no_system_ffmpeg, monkeypatch):
        found = fake_exe(Path(os.environ["PATH"]) / "ffprobe7")
        monkeypatch.setenv("VWATCH_FFPROBE", "ffprobe7")
        assert media.ffprobe() == str(found)

    def test_bad_env_value_is_a_clear_error(self, no_system_ffmpeg, monkeypatch):
        monkeypatch.setenv("VWATCH_FFMPEG", str(no_system_ffmpeg / "nope"))
        with pytest.raises(media.FFmpegNotFoundError, match="VWATCH_FFMPEG="):
            media.ffmpeg()


class TestSystemPath:
    def test_bare_name_when_on_path(self, no_system_ffmpeg):
        """Same command line as before the resolver existed: plain `ffmpeg` / `ffprobe`."""
        bindir = Path(os.environ["PATH"])
        fake_exe(bindir / "ffmpeg")
        fake_exe(bindir / "ffprobe")
        assert (media.ffmpeg(), media.ffprobe()) == ("ffmpeg", "ffprobe")

    def test_module_constants_resolve_lazily(self, no_system_ffmpeg):
        bindir = Path(os.environ["PATH"])
        fake_exe(bindir / "ffmpeg")
        fake_exe(bindir / "ffprobe")
        assert (media.FFMPEG, media.FFPROBE) == ("ffmpeg", "ffprobe")
        with pytest.raises(AttributeError):
            media.NOT_A_THING  # noqa: B018

    def test_resolution_is_cached_until_reset(self, no_system_ffmpeg):
        bindir = Path(os.environ["PATH"])
        fake_exe(bindir / "ffmpeg")
        assert media.ffmpeg() == "ffmpeg"
        (bindir / "ffmpeg").unlink()
        assert media.ffmpeg() == "ffmpeg"
        media.reset_binaries()
        with pytest.raises(media.FFmpegNotFoundError):
            media.ffmpeg()


class TestStaticFallback:
    def test_downloads_into_vwatch_cache_and_keeps_stdout_clean(self, no_system_ffmpeg, monkeypatch, capsys):
        calls = []

        def fetch(download_dir):
            calls.append(download_dir)
            print("Downloading https://example.invalid/ffmpeg.zip")  # static-ffmpeg really prints to stdout
            d = Path(download_dir)
            return str(fake_exe(d / "ffmpeg")), str(fake_exe(d / "ffprobe"))

        install_fake_static_ffmpeg(monkeypatch, fetch)
        dest = no_system_ffmpeg / "cache" / "ffmpeg" / "testplat"
        assert media.ffmpeg() == str(dest / "ffmpeg")
        assert media.ffprobe() == str(dest / "ffprobe")
        assert calls == [str(dest)]  # fetched once for both binaries
        out, err = capsys.readouterr()
        assert out == ""  # an MCP stdio server's stdout is the protocol
        assert "Downloading" in err and "downloading a static ffmpeg build" in err

    def test_system_ffprobe_with_static_ffmpeg(self, no_system_ffmpeg, monkeypatch):
        fake_exe(Path(os.environ["PATH"]) / "ffprobe")

        def fetch(download_dir):
            d = Path(download_dir)
            return str(fake_exe(d / "ffmpeg")), str(fake_exe(d / "ffprobe"))

        install_fake_static_ffmpeg(monkeypatch, fetch)
        assert media.ffprobe() == "ffprobe"
        assert media.ffmpeg().endswith(os.path.join("testplat", "ffmpeg"))

    def test_download_failure_is_a_clear_error_and_retried_later(self, no_system_ffmpeg, monkeypatch):
        attempts = []

        def fetch(download_dir):
            attempts.append(download_dir)
            raise OSError("network down")

        install_fake_static_ffmpeg(monkeypatch, fetch)
        with pytest.raises(media.FFmpegNotFoundError, match="download .* failed .*network down.*brew install"):
            media.ffmpeg()
        with pytest.raises(media.FFmpegNotFoundError):
            media.ffmpeg()
        assert len(attempts) == 2  # failures are not cached: a later call can succeed

    def test_incomplete_download_is_reported(self, no_system_ffmpeg, monkeypatch):
        install_fake_static_ffmpeg(monkeypatch, lambda d: (str(Path(d) / "ffmpeg"), str(Path(d) / "ffprobe")))
        with pytest.raises(media.FFmpegNotFoundError, match="incomplete"):
            media.ffmpeg()

    def test_without_static_ffmpeg_package(self, no_system_ffmpeg):
        # the fixture maps `static_ffmpeg` to None in sys.modules, so importing it raises ImportError
        with pytest.raises(media.FFmpegNotFoundError, match="ffmpeg/ffprobe not found.*VWATCH_FFMPEG"):
            media.ffprobe()

    def test_media_error_names_the_binary_not_its_path(self, no_system_ffmpeg, monkeypatch, tmp_path):
        failing = tmp_path / "bin" / "ffprobe"
        failing.parent.mkdir()
        failing.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
        failing.chmod(0o755)
        monkeypatch.setenv("VWATCH_FFPROBE", str(failing))
        with pytest.raises(media.MediaError, match=r"^ffprobe failed \(3\): broken"):
            media.probe(tmp_path / "x.mp4")
