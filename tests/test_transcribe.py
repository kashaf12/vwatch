import json

import pytest

from vwatch import transcribe
from vwatch.transcribe import EMPTY, Segment, Transcript, Word


@pytest.fixture
def tr():
    return Transcript(
        language="hi",
        segments=(Segment(0.0, 2.0, " namaste dosto"), Segment(5.0, 7.5, " yeh hai kahani")),
        words=(Word(0.0, 0.8, " namaste"), Word(0.9, 2.0, " dosto"), Word(5.0, 5.4, " yeh"),
               Word(5.5, 6.0, " hai")))


class TestTranscript:
    def test_to_text_all(self, tr):
        assert tr.to_text().splitlines() == ["[00:00.00–00:02.00] namaste dosto",
                                              "[00:05.00–00:07.50] yeh hai kahani"]

    def test_to_text_window(self, tr):
        assert tr.to_text(4.0, 9.0) == "[00:05.00–00:07.50] yeh hai kahani"
        assert tr.to_text(t1=1.0).count("\n") == 0

    def test_to_text_overlap_edges(self, tr):
        assert "namaste" in tr.to_text(2.0, 3.0)  # touching the end counts

    def test_to_text_empty(self, tr):
        assert tr.to_text(3.0, 4.0) == "(no speech)"
        assert EMPTY.to_text() == "(no speech)"

    def test_words_text(self, tr):
        assert tr.words_text(5.0, 6.0) == "yeh@5.00 hai@5.50"
        assert tr.words_text(3.0, 4.0) == "(no words)"

    def test_json_round_trip(self, tr):
        d = json.loads(json.dumps(tr.to_json(), ensure_ascii=False))
        assert Transcript.from_json(d) == tr

    def test_empty_round_trip(self):
        assert Transcript.from_json(EMPTY.to_json()) == EMPTY


class _Boom:
    def __getattr__(self, name):
        raise AssertionError("whisper must not be loaded")


@pytest.fixture
def no_whisper(monkeypatch):
    def boom(name):
        raise AssertionError("whisper must not be loaded")

    monkeypatch.setattr(transcribe, "_whisper", boom)


class TestTranscribe:
    def test_file_hash_stable(self, tone_video, silent_video):
        assert transcribe.file_hash(tone_video) == transcribe.file_hash(tone_video)
        assert transcribe.file_hash(tone_video) != transcribe.file_hash(silent_video)

    def test_cache_hit(self, tone_video, tr, tmp_path, no_whisper):
        key = f"{transcribe.file_hash(tone_video)}-small-hi"
        (tmp_path / f"{key}.json").write_text(json.dumps(tr.to_json()))
        out = transcribe.transcribe(tone_video, model_name="small", language="hi", cache_dir=tmp_path)
        assert out == tr

    def test_cache_key_includes_model_and_language(self, tone_video, tr, tmp_path, no_whisper,
                                                   monkeypatch):
        key = f"{transcribe.file_hash(tone_video)}-small-auto"
        (tmp_path / f"{key}.json").write_text(json.dumps(tr.to_json()))
        assert transcribe.transcribe(tone_video, model_name="small", cache_dir=tmp_path) == tr
        # different language -> miss -> would load whisper
        with pytest.raises(AssertionError, match="whisper"):
            transcribe.transcribe(tone_video, model_name="small", language="en", cache_dir=tmp_path)

    def test_silent_video_empty_and_cached(self, silent_video, tmp_path, no_whisper):
        cache = tmp_path / "c"
        out = transcribe.transcribe(silent_video, cache_dir=cache)
        assert out == EMPTY
        files = list(cache.glob("*.json"))
        assert len(files) == 1
        assert transcribe.transcribe(silent_video, cache_dir=cache) == EMPTY

    def test_silent_audio_track_is_empty(self, tmp_path, no_whisper):
        import numpy as np

        from vwatch import media

        v = media.write_video(np.zeros((12, 32, 32, 3), np.uint8), 12, tmp_path / "z.mp4",
                              audio=np.zeros(44100, dtype=np.float32) + 1e-9)
        assert transcribe.transcribe(v, cache_dir=tmp_path / "c") == EMPTY

    def test_fake_whisper_path(self, tone_video, tmp_path, monkeypatch):
        from types import SimpleNamespace as NS

        class Model:
            def transcribe(self, audio, **kw):
                assert kw["language"] == "en" and kw["word_timestamps"]
                seg = NS(start=0.123, end=1.456, text=" hi there",
                         words=[NS(start=0.123, end=0.5, word=" hi"), NS(start=0.6, end=1.0, word=" there")])
                return [seg, NS(start=2, end=3, text=" x", words=None)], NS(language="en")

        monkeypatch.setattr(transcribe, "_whisper", lambda name: Model())
        out = transcribe.transcribe(tone_video, language="en", cache_dir=tmp_path)
        assert out.language == "en" and len(out.segments) == 2
        assert out.segments[0] == Segment(0.12, 1.46, " hi there")
        assert [w.text for w in out.words] == [" hi", " there"]
        assert list(tmp_path.glob("*.json"))


class TestWhisperLoader:
    @pytest.fixture
    def fake_fw(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        calls = []

        class WhisperModel:
            def __init__(self, name, device, compute_type):
                calls.append((name, device, compute_type))
                if device == "mps":
                    raise ValueError("no mps")

        monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=WhisperModel))
        monkeypatch.setattr(transcribe, "_MODEL_CACHE", {})
        monkeypatch.delenv("VWATCH_WHISPER_DEVICE", raising=False)
        return calls

    def test_auto_falls_back_to_cpu_and_caches(self, fake_fw):
        m1 = transcribe._whisper("small")
        m2 = transcribe._whisper("small")
        assert m1 is m2
        assert fake_fw == [("small", "mps", "float16"), ("small", "cpu", "int8")]

    def test_explicit_device(self, fake_fw, monkeypatch):
        monkeypatch.setenv("VWATCH_WHISPER_DEVICE", "cpu")
        transcribe._whisper("tiny")
        assert fake_fw == [("tiny", "cpu", "default")]

    def test_all_backends_fail(self, fake_fw, monkeypatch):
        monkeypatch.setenv("VWATCH_WHISPER_DEVICE", "mps")
        with pytest.raises(RuntimeError, match="could not load whisper: mps: no mps"):
            transcribe._whisper("small")


class TestResegment:
    """Batched decoding returns ~30 s segments; we rebuild sentence-sized ones from the word times."""

    @staticmethod
    def words(spec):
        from vwatch.transcribe import Word
        return tuple(Word(a, b, t) for a, b, t in spec)

    def test_splits_at_sentence_end(self):
        from vwatch.transcribe import resegment
        segs = resegment(self.words([(0.0, 0.4, " Hello"), (0.4, 0.8, " there."), (0.9, 1.2, " Next"), (1.2, 1.5, " one.")]))
        assert [(s.t0, s.t1, s.text) for s in segs] == [(0.0, 0.8, " Hello there."), (0.9, 1.5, " Next one.")]

    def test_splits_at_long_pause(self):
        from vwatch.transcribe import resegment
        segs = resegment(self.words([(0.0, 0.5, " and"), (0.5, 1.0, " so"), (2.5, 3.0, " then")]))
        assert [s.text for s in segs] == [" and so", " then"]

    def test_caps_segment_length(self):
        from vwatch.transcribe import resegment
        ws = self.words([(k * 0.5, k * 0.5 + 0.4, " w") for k in range(60)])  # 30 s with no punctuation or pause
        segs = resegment(ws)
        assert len(segs) >= 3 and all(s.t1 - s.t0 <= 12.5 for s in segs)
        assert sum(len(s.text.split()) for s in segs) == 60

    def test_empty(self):
        from vwatch.transcribe import resegment
        assert resegment(()) == ()
