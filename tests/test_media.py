import subprocess
from pathlib import Path

import numpy as np
import pytest

from vwatch import media
from vwatch.media import MediaError, VideoInfo

N = 40


@pytest.fixture(scope="module")
def marked_video(tmp_path_factory):
    """40 frames, 64x96, frame k flat colour (6k, 255-6k, 200*(k%2)): each frame identifiable."""
    frames = np.zeros((N, 96, 64, 3), dtype=np.uint8)
    for k in range(N):
        frames[k] = (6 * k, 255 - 6 * k, 200 * (k % 2))
    return media.write_video(frames, 20, tmp_path_factory.mktemp("m") / "marked.mp4", crf=8)


def _which(frame):
    """Index decoded from red level (6k) and blue parity."""
    r = float(frame[..., 0].mean())
    odd = float(frame[..., 2].mean()) > 100
    return int(np.argmin([abs(r - 6 * k) + (0 if (k % 2 == 1) == odd else 1000) for k in range(N)]))


class TestProbe:
    def test_fields(self, tone_video):
        info = media.probe(tone_video)
        assert (info.width, info.height) == (108, 192)
        assert info.fps == pytest.approx(30)
        assert info.nb_frames == 120
        assert info.duration == pytest.approx(4.0, abs=0.1)
        assert info.has_audio

    def test_no_audio(self, silent_video):
        assert not media.probe(silent_video).has_audio

    def test_frame_time_index_clamp(self):
        info = VideoInfo("p", 4.0, 30.0, 10, 10, 120, False)
        assert info.frame_time(30) == 1.0
        assert info.frame_index(1.0) == 30
        assert info.frame_index(-5) == 0
        assert info.frame_index(99) == 119

    def test_bad_file(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"this is not a video")
        with pytest.raises(MediaError):
            media.probe(bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(MediaError):
            media.probe(tmp_path / "nope.mp4")

    def test_audio_only_has_no_video_stream(self, tmp_path):
        out = tmp_path / "a.wav"
        media._run([media.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=d=0.5", str(out)])
        with pytest.raises(MediaError, match="no video stream"):
            media.probe(out)

    def test_decode_bad_file(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"xx")
        with pytest.raises(MediaError):
            media.decode_all(bad, 32)


class TestDecodeAll:
    def test_gray_shape(self, tone_video):
        arr, info = media.decode_all(tone_video, 54, gray=True)
        assert arr.shape == (120, 96, 54) and arr.dtype == np.uint8
        assert info.nb_frames == 120

    def test_rgb_shape(self, tone_video):
        arr, _ = media.decode_all(tone_video, 54)
        assert arr.shape == (120, 96, 54, 3)

    def test_odd_width_rounded_even(self, tone_video):
        arr, _ = media.decode_all(tone_video, 55, gray=True)
        assert arr.shape[2] == 54 and arr.shape[1] % 2 == 0


class TestFramesByIndex:
    def test_exact_frames(self, marked_video):
        idx = [0, 3, 17, 18, 39]
        got = media.frames_by_index(marked_video, idx, 32)
        assert sorted(got) == idx
        for i, f in got.items():
            assert f.shape == (48, 32, 3)
            assert _which(f) == i
            assert abs(float(f[..., 1].mean()) - (255 - 6 * i)) < 10  # colour preserved

    def test_unsorted_duplicates(self, marked_video):
        got = media.frames_by_index(marked_video, [20, 5, 20, 5], 32)
        assert sorted(got) == [5, 20]
        assert _which(got[5]) == 5 and _which(got[20]) == 20

    def test_empty(self, marked_video):
        assert media.frames_by_index(marked_video, [], 32) == {}

    def test_past_end_dropped(self, marked_video):
        got = media.frames_by_index(marked_video, [2, 10_000], 32)
        assert list(got) == [2] and _which(got[2]) == 2


class TestFramesAt:
    def test_dedup_and_sorted(self, marked_video):
        # 20 fps: 0.0 and 0.01 -> frame 0; 1.0 -> frame 20
        out = media.frames_at(marked_video, [1.0, 0.0, 0.01], 32)
        assert [round(t, 3) for t, _ in out] == [0.0, 1.0]
        assert [_which(f) for _, f in out] == [0, 20]

    def test_beyond_end_clamps_to_last(self, marked_video):
        out = media.frames_at(marked_video, [100.0], 32)
        assert len(out) == 1 and _which(out[0][1]) == N - 1

    def test_returns_actual_time(self, marked_video):
        (t, f), = media.frames_at(marked_video, [0.52], 32)  # nearest frame 10 -> 0.5 s
        assert t == pytest.approx(0.5) and _which(f) == 10


class TestAudio:
    def test_with_audio(self, tone_video):
        a = media.extract_audio(tone_video, sr=16000)
        assert a.dtype == np.float32
        assert abs(len(a) - 4 * 16000) < 2000
        assert float(np.sqrt((a ** 2).mean())) == pytest.approx(0.3 / np.sqrt(2), abs=0.04)

    def test_without_audio(self, silent_video):
        a = media.extract_audio(silent_video)
        assert a.size == 0 and a.dtype == np.float32


def _onset(a: np.ndarray, sr: int) -> float:
    """Time of the first sample louder than half the peak."""
    return float(np.argmax(np.abs(a) > 0.5 * np.abs(a).max())) / sr


@pytest.fixture(scope="module")
def late_tone(tmp_path_factory):
    """4 s video; its audio is silent for 0.5 s, then a tone. Returns (writer, dir)."""
    d = tmp_path_factory.mktemp("offset")
    sr = 16000
    t = np.arange(4 * sr) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * (t >= 0.5)).astype(np.float32)
    frames = np.zeros((120, 64, 64, 3), dtype=np.uint8)
    src = media.write_video(frames, 30, d / "src.mp4", audio=audio, sr=sr)

    def shifted(name: str, audio_shift: float) -> Path:
        out = d / name
        a_in = ["-itsoffset", str(audio_shift)] if audio_shift > 0 else []
        v_in = ["-itsoffset", str(-audio_shift)] if audio_shift < 0 else []
        subprocess.run([media.ffmpeg(), "-v", "error", "-y", *v_in, "-i", str(src), *a_in, "-i", str(src),
                        "-map", "0:v", "-map", "1:a", "-c", "copy", str(out)], check=True)
        return out

    return shifted


class TestAudioTimeline:
    """Audio samples line up with the video's clock even when the audio stream starts at another time."""

    def test_audio_starting_late_is_padded(self, late_tone):
        path = late_tone("late.mp4", 1.0)  # audio stream starts 1 s after the first frame
        a = media.extract_audio(path, sr=16000)
        assert _onset(a, 16000) == pytest.approx(1.5, abs=0.05)

    def test_audio_starting_early_is_trimmed(self, late_tone):
        path = late_tone("early.mp4", -0.3)  # video starts 0.3 s after the audio
        a = media.extract_audio(path, sr=16000)
        assert _onset(a, 16000) == pytest.approx(0.2, abs=0.05)

    def test_aligned_audio_is_unchanged(self, late_tone):
        a = media.extract_audio(late_tone("aligned.mp4", 0.0), sr=16000)
        assert _onset(a, 16000) == pytest.approx(0.5, abs=0.05)


class TestWriteVideo:
    def test_round_trip(self, tmp_path):
        frames = np.full((12, 32, 48, 3), 100, dtype=np.uint8)
        out = media.write_video(frames, 12, tmp_path / "sub" / "x.mp4")
        assert out.exists()
        info = media.probe(out)
        assert (info.width, info.height, info.nb_frames) == (48, 32, 12)
        assert not info.has_audio
        arr, _ = media.decode_all(out, 48)
        assert abs(float(arr.mean()) - 100) < 4

    def test_audio_round_trip_and_no_temp_left(self, tmp_path):
        frames = np.zeros((24, 32, 32, 3), dtype=np.uint8)
        audio = (0.5 * np.sin(np.arange(44100) / 20)).astype(np.float32)
        out = media.write_video(frames, 24, tmp_path / "x.mp4", audio=audio)
        assert media.probe(out).has_audio
        assert not list(tmp_path.glob("*.tmp.f32"))

    def test_encode_failure_raises_and_cleans_up(self, tmp_path):
        frames = np.zeros((2, 31, 31, 3), dtype=np.uint8)  # odd dims: yuv420 libx264 rejects
        audio = np.zeros(100, dtype=np.float32) + 0.1
        with pytest.raises(MediaError):
            media.write_video(frames, 10, tmp_path / "x.mp4", audio=audio)
        assert not list(tmp_path.glob("*.tmp.f32"))
