"""Timestamped speech transcript via faster-whisper, cached on disk by file content hash."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from vwatch import media
from vwatch.imaging import fmt_time

CACHE_DIR = Path(os.environ.get("VWATCH_CACHE", Path.home() / ".cache" / "vwatch")) / "transcripts"
DEFAULT_WHISPER = os.environ.get("VWATCH_WHISPER_MODEL", "small")


@dataclass(frozen=True)
class Word:
    t0: float
    t1: float
    text: str


@dataclass(frozen=True)
class Segment:
    t0: float
    t1: float
    text: str


@dataclass(frozen=True)
class Transcript:
    language: str
    segments: tuple[Segment, ...]
    words: tuple[Word, ...]

    def to_text(self, t0: float | None = None, t1: float | None = None) -> str:
        segs = [s for s in self.segments
                if (t0 is None or s.t1 >= t0) and (t1 is None or s.t0 <= t1)]
        if not segs:
            return "(no speech)"
        return "\n".join(f"[{fmt_time(s.t0)}–{fmt_time(s.t1)}] {s.text.strip()}" for s in segs)

    def words_text(self, t0: float, t1: float) -> str:
        ws = [w for w in self.words if w.t1 >= t0 and w.t0 <= t1]
        if not ws:
            return "(no words)"
        return " ".join(f"{w.text.strip()}@{w.t0:.2f}" for w in ws)

    def to_json(self) -> dict:
        return {"language": self.language,
                "segments": [s.__dict__ for s in self.segments],
                "words": [w.__dict__ for w in self.words]}

    @classmethod
    def from_json(cls, d: dict) -> Transcript:
        return cls(language=d["language"],
                   segments=tuple(Segment(**s) for s in d["segments"]),
                   words=tuple(Word(**w) for w in d["words"]))


EMPTY = Transcript(language="", segments=(), words=())


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_MODEL_CACHE: dict[str, object] = {}


def _whisper(model_name: str):
    if model_name not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        device = os.environ.get("VWATCH_WHISPER_DEVICE", "auto")
        attempts = [("mps", "float16"), ("cpu", "int8")] if device == "auto" else [(device, "default")]
        errors = []
        for dev, compute in attempts:
            try:
                _MODEL_CACHE[model_name] = WhisperModel(model_name, device=dev, compute_type=compute)
                break
            except (ValueError, RuntimeError) as exc:  # backend unavailable on this machine
                errors.append(f"{dev}: {exc}")
        else:
            raise RuntimeError("could not load whisper: " + "; ".join(errors))
    return _MODEL_CACHE[model_name]


def transcribe(path: str | Path, *, model_name: str = DEFAULT_WHISPER, language: str | None = None,
               cache_dir: Path = CACHE_DIR) -> Transcript:
    key = f"{file_hash(path)}-{model_name}-{language or 'auto'}"
    cached = cache_dir / f"{key}.json"
    if cached.exists():
        return Transcript.from_json(json.loads(cached.read_text()))
    audio = media.extract_audio(path, sr=16000)
    if audio.size == 0 or float(abs(audio).max()) < 1e-4:
        result = EMPTY
    else:
        model = _whisper(model_name)
        segs, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, beam_size=5,
                                     language=language)
        segments, words = [], []
        for s in segs:
            segments.append(Segment(round(s.start, 2), round(s.end, 2), s.text))
            for w in s.words or []:
                words.append(Word(round(w.start, 2), round(w.end, 2), w.word))
        result = Transcript(language=info.language, segments=tuple(segments), words=tuple(words))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result.to_json(), ensure_ascii=False))
    return result
