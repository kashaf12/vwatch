"""The `vw` entry point: help, argument validation and friendly errors. No model calls."""

from __future__ import annotations

import pytest

from vwatch import cli
from vwatch.media import MediaError


def test_help_lists_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in ("vw watch", "vw signals", "vw dataset build", "vw eval run", "vw eval report"):
        assert cmd in out


def test_unknown_command_returns_2(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["signals", "missing.mp4"], ["watch", "missing.mp4", "-q", "x"]])
def test_missing_video_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "no such video file: missing.mp4" in capsys.readouterr().err


def test_watch_rejects_unknown_arm(tone_video, capsys):
    with pytest.raises(SystemExit):
        cli.main(["watch", str(tone_video), "-q", "x", "--arm", "Z"])
    assert "invalid choice" in capsys.readouterr().err


def test_signals_prints_hints(tone_video, capsys):
    assert cli.main(["signals", str(tone_video)]) == 0
    capsys.readouterr()  # output content is covered by test_signals; here it only has to run


def test_media_error_is_reported_without_traceback(monkeypatch, tone_video, capsys):
    from vwatch import signals

    def boom(_path):
        raise MediaError("ffprobe failed (1): broken file")

    monkeypatch.setattr(signals, "compute", boom)
    assert cli.main(["signals", str(tone_video)]) == 1
    assert capsys.readouterr().err.strip() == "error: ffprobe failed (1): broken file"
