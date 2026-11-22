"""Shared fixtures: a tiny synthetic video with a tone, and a VideoContext built without whisper."""

from __future__ import annotations

import numpy as np
import pytest

from vwatch import media, signals, transcribe
from vwatch.arms.common import VideoContext

W, H, FPS, SECS = 108, 192, 30, 4


def make_frames(n: int = FPS * SECS, w: int = W, h: int = H) -> np.ndarray:
    """Gradient background with a moving square, so every frame differs a little."""
    yy = np.linspace(40, 120, h, dtype=np.float32)[:, None, None]
    base = np.broadcast_to(yy, (h, w, 3)).astype(np.uint8)
    frames = np.repeat(base[None], n, axis=0).copy()
    for k in range(n):
        x = int((k / max(1, n - 1)) * (w - 24))
        frames[k, h // 2:h // 2 + 24, x:x + 24] = (230, 60, 60)
    return frames


def tone(seconds: float = SECS, sr: int = 44100, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("videos")


@pytest.fixture(scope="session")
def tone_video(video_dir):
    """108x192, 4 s, 30 fps, 440 Hz tone."""
    return media.write_video(make_frames(), FPS, video_dir / "tone.mp4", audio=tone())


@pytest.fixture(scope="session")
def silent_video(video_dir):
    return media.write_video(make_frames(), FPS, video_dir / "silent.mp4")


@pytest.fixture(scope="session")
def tiny_transcript():
    return transcribe.Transcript(
        language="en",
        segments=(transcribe.Segment(0.2, 1.5, " hello there world"),
                  transcribe.Segment(2.0, 3.5, " second line here")),
        words=(transcribe.Word(0.2, 0.6, " hello"), transcribe.Word(0.6, 1.0, " there"),
               transcribe.Word(1.0, 1.5, " world"), transcribe.Word(2.0, 2.6, " second"),
               transcribe.Word(2.6, 3.0, " line"), transcribe.Word(3.0, 3.5, " here")))


@pytest.fixture(scope="session")
def video_ctx(tone_video, tiny_transcript):
    info = media.probe(tone_video)
    track = signals.compute(tone_video)
    cands = (signals.Candidate(1.0, 1.0, "cut_in_speech", 0.75, "x"),
             signals.Candidate(2.5, 2.7, "flash", 0.8, "y"))
    return VideoContext(path=tone_video, info=info, transcript=tiny_transcript, track=track,
                        candidates=cands)


@pytest.fixture(scope="session")
def silent_ctx(silent_video):
    info = media.probe(silent_video)
    track = signals.compute(silent_video)
    return VideoContext(path=silent_video, info=info, transcript=transcribe.EMPTY, track=track,
                        candidates=())
