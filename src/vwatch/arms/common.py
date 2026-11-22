"""Pieces shared by every arm: the per-video context, the evidence log and the final answer step."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vwatch import media, signals, transcribe
from vwatch.imaging import estimate_image_tokens, estimate_text_tokens, image_block
from vwatch.llm.base import LLMError, Reply, Session, text_block
from vwatch.schemas import Observation, Task, Usage
from vwatch.tasks import REPAIR_PROMPT, parse_report


@dataclass(frozen=True)
class VideoContext:
    """Everything computed once per video and shared by all arms and repeats."""

    path: Path
    info: media.VideoInfo
    transcript: transcribe.Transcript
    track: signals.SignalTrack
    candidates: tuple[signals.Candidate, ...]

    @classmethod
    def load(cls, path: str | Path, *, language: str | None = None) -> VideoContext:
        p = Path(path)
        info = media.probe(p)
        tr = transcribe.transcribe(p, language=language) if info.has_audio else transcribe.EMPTY
        track = signals.compute(p)
        return cls(path=p, info=info, transcript=tr, track=track,
                   candidates=tuple(signals.find_candidates(track)))


@dataclass(frozen=True)
class ArmResult:
    raw_output: str
    parsed: dict | None
    parse_error: str | None
    content_tokens_est: int
    usage: Usage
    cost_usd: float
    wall_s: float
    turns: int
    observations: tuple[Observation, ...]
    error: str | None = None
    tool_calls: tuple[dict, ...] = ()  # external agent arms: {"name", "input"} per tool call, for human review


@dataclass
class Evidence:
    """Builds one user turn and tracks the evidence budget. Local to a single run."""

    blocks: list[dict] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    tokens: int = 0

    def text(self, s: str) -> None:
        self.blocks.append(text_block(s))
        self.tokens += estimate_text_tokens(s)

    def image(self, frame: np.ndarray, kind: str, t0: float, t1: float) -> None:
        h, w = frame.shape[:2]
        self.blocks.append(image_block(frame))
        self.tokens += estimate_image_tokens(w, h)
        self.observations.append(Observation(kind=kind, t0=round(t0, 3), t1=round(t1, 3)))

    def observe(self, kind: str, t0: float, t1: float) -> None:
        self.observations.append(Observation(kind=kind, t0=round(t0, 3), t1=round(t1, 3)))

    def take(self) -> list[dict]:
        """Return the pending blocks and start a new turn (observations and tokens keep accumulating)."""
        out, self.blocks = self.blocks, []
        return out


@dataclass
class Conversation:
    """Wraps a session; accumulates usage, cost, wall time and turns."""

    session: Session
    usage: Usage = Usage()
    cost: float = 0.0
    wall: float = 0.0
    turns: int = 0
    last_text: str = ""

    def send(self, blocks: list[dict]) -> Reply:
        reply = self.session.send(blocks)
        self.usage = self.usage.plus(reply.usage)
        self.cost += reply.cost_usd
        self.wall += reply.wall_s
        self.turns += 1
        self.last_text = reply.text
        return reply


def finish(conv: Conversation, task: Task, ev: Evidence, started: float, text: str) -> ArmResult:
    """Parse the final answer, with one repair turn on failure."""
    parsed, err = parse_report(task, text)
    if parsed is None:
        reply = conv.send([text_block(REPAIR_PROMPT.format(error=err))])
        text = reply.text
        parsed, err = parse_report(task, text)
    return ArmResult(raw_output=text, parsed=parsed, parse_error=err, content_tokens_est=ev.tokens,
                     usage=conv.usage, cost_usd=conv.cost, wall_s=time.monotonic() - started,
                     turns=conv.turns, observations=tuple(ev.observations))


def failed(conv: Conversation | None, ev: Evidence, started: float, exc: Exception) -> ArmResult:
    return ArmResult(raw_output=conv.last_text if conv else "", parsed=None, parse_error=None,
                     content_tokens_est=ev.tokens, usage=conv.usage if conv else Usage(),
                     cost_usd=conv.cost if conv else 0.0, wall_s=time.monotonic() - started,
                     turns=conv.turns if conv else 0, observations=tuple(ev.observations),
                     error=f"{type(exc).__name__}: {exc}")


RECOVERABLE = (LLMError, media.MediaError, OSError, ValueError)

MAX_HINTS = 40
HINTS_HEADER = "LOW-LEVEL SIGNAL HINTS (computed from every frame and the audio; hints, not verdicts):\n"


def select_hints(cands: tuple[signals.Candidate, ...] | list[signals.Candidate],
                 kinds: tuple[str, ...] | None = None, k: int = MAX_HINTS) -> list[signals.Candidate]:
    """The k highest-scoring candidates (optionally of given kinds), returned in time order."""
    pool = [c for c in cands if kinds is None or c.kind in kinds]
    top = sorted(pool, key=lambda c: (-c.score, c.t0))[:k]
    return sorted(top, key=lambda c: c.t0)


def add_hints(ev: Evidence, hints: list[signals.Candidate]) -> None:
    ev.text(HINTS_HEADER + signals.describe(hints))
    for c in hints:
        ev.observe("signal", c.t0, c.t1)


def video_header(ctx: VideoContext) -> str:
    i = ctx.info
    speech = "yes" if ctx.transcript.segments else "no speech detected"
    return (f"VIDEO: {i.duration:.2f} s long, {i.width}x{i.height}, {i.fps:.0f} fps. Speech: {speech}. "
            "Times are seconds from the start; every image has its timestamp burned in (mm:ss.ss).")
