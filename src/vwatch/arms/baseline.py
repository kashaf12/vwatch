"""Baselines B0/B1/B2: how Claude handles video by default — uniformly sampled frames, one shot.

B0: frames only. B1: frames + timestamped transcript. B2 is B1 run at twice the budget (runner's job).
"""

from __future__ import annotations

import time

from vwatch import media
from vwatch.arms.common import (HINTS_HEADER, RECOVERABLE, ArmResult, Conversation, Evidence, VideoContext,
                                add_hints, failed, finish, select_hints, video_header)
from vwatch.signals import describe
from vwatch.imaging import estimate_image_tokens, estimate_text_tokens, fmt_time, stamp
from vwatch.llm.base import Backend
from vwatch.schemas import Task

FRAME_WIDTH = 360
MIN_FRAMES = 4

SYSTEM = ("You analyse videos from the frames and transcript you are given. Be precise about times. "
          "Follow the requested JSON format exactly.")


def uniform_times(duration: float, n: int) -> list[float]:
    step = duration / n
    return [round((k + 0.5) * step, 3) for k in range(n)]


def frame_budget(ctx: VideoContext, budget: int, text_tokens: int) -> int:
    w = FRAME_WIDTH
    h = int(round(ctx.info.height * w / ctx.info.width))
    per_frame = estimate_image_tokens(w, h) + 6  # + the timestamp text block
    n = (budget - text_tokens) // per_frame
    return int(max(MIN_FRAMES, min(n, ctx.info.nb_frames)))


def run_baseline(ctx: VideoContext, task: Task, prompt: str, budget: int, backend: Backend,
                 *, with_transcript: bool, with_hints: bool = False) -> ArmResult:
    started = time.monotonic()
    ev = Evidence()
    conv: Conversation | None = None
    try:
        transcript = ctx.transcript.to_text() if with_transcript else ""
        intro = video_header(ctx)
        hints = select_hints(ctx.candidates) if with_hints else []
        hint_text = HINTS_HEADER + describe(hints) if with_hints else ""
        text_tokens = estimate_text_tokens(intro + transcript + prompt + hint_text) + 60
        n = frame_budget(ctx, budget, text_tokens)
        if n >= ctx.info.nb_frames:  # budget covers every frame: take them exactly
            times = [ctx.info.frame_time(i) for i in range(ctx.info.nb_frames)]
        else:
            times = uniform_times(ctx.info.duration, n)
        frames = media.frames_at(ctx.path, times, FRAME_WIDTH)
        ev.text(f"{intro}\nBelow are {len(frames)} frames sampled uniformly "
                f"(about one every {ctx.info.duration / max(1, len(frames)):.2f} s).")
        for t, f in frames:
            ev.text(f"t={fmt_time(t)}")
            ev.image(stamp(f, fmt_time(t)), "frame", t, t)
        if with_transcript:
            ev.text("TRANSCRIPT (automatic speech recognition, timestamped):\n" + transcript)
            for s in ctx.transcript.segments:
                ev.observe("transcript", s.t0, s.t1)
        if with_hints:
            add_hints(ev, hints)
        ev.text(prompt)
        conv = Conversation(backend.open(SYSTEM))
        reply = conv.send(ev.take())
        return finish(conv, task, ev, started, reply.text)
    except RECOVERABLE as exc:
        return failed(conv, ev, started, exc)
    finally:
        if conv is not None:
            conv.session.close()
