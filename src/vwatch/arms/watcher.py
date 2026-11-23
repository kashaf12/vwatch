"""The watcher (arm W) and its ablations.

One conversation, four stages:
  1. PLAN (question first): before any frame, the model reads the task, the video facts and the
     transcript, and says what it is looking for, which regions need a close look and how dense
     to sample.                                              [skipped in W-noQ]
  2. OVERVIEW + PREDICT: a cheap low-res pass over the whole video plus the low-level signal hints;
     the model predicts what should happen between frames and flags where its prediction is
     uncertain or broken.                                    [W-noP: no prediction, signals only]
  3. FOCUS: the harness serves the most surprising windows densely, with crops of the planned
     regions and an audio meter.                             [W-noCrop: no crops anywhere]
  4. ACT: the model may request more seeks / crops / listens within the budget, then answers.
W-noS removes the low-level signal hints entirely: attention then comes from the model alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vwatch.arms import tools
from vwatch.arms.common import (RECOVERABLE, ArmResult, Conversation, Evidence, VideoContext, add_hints,
                                failed, finish, select_hints, video_header)
from vwatch.imaging import Box
from vwatch.llm.base import Backend, extract_json, text_block
from vwatch.schemas import Arm, Task
from vwatch import signals
from vwatch.signals import Candidate

SYSTEM = ("You watch videos the way an attentive person does: decide what you are looking for, skim, "
          "notice where your expectations break, and look closely there. You only see what you ask "
          "for, so ask for what you need. Be precise about times. When asked for JSON, reply with JSON only.")

SIGNAL_KINDS = signals.KINDS
MAX_ACTION_ROUNDS = 3
MAX_ACTIONS_PER_ROUND = 6
FOCUS_SHARE = 0.7  # share of the budget the harness may spend before the model's own requests
WINDOW_PAD = 0.4
MAX_WINDOW = 2.0


@dataclass(frozen=True)
class Variant:
    plan_first: bool = True
    predict: bool = True
    use_crops: bool = True
    use_signals: bool = True


VARIANTS: dict[Arm, Variant] = {
    Arm.W: Variant(),
    Arm.W_NOQ: Variant(plan_first=False),
    Arm.W_NOP: Variant(predict=False),
    Arm.W_NOCROP: Variant(use_crops=False),
    Arm.W_NOS: Variant(use_signals=False),
}


# ------------------------------------------------------------------ model-facing schemas


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class PlanRegion(_Loose):
    name: str = "region"
    box: tuple[float, float, float, float]

    def to_box(self) -> Box:
        return Box.clamp(*self.box)


class WatchPlan(_Loose):
    looking_for: tuple[str, ...] = ()
    regions: tuple[PlanRegion, ...] = ()
    dense_fps: float = Field(default=8.0)
    overview_fps: float = Field(default=1.0)
    signal_kinds: tuple[str, ...] = SIGNAL_KINDS

    @field_validator("dense_fps")
    @classmethod
    def _dense(cls, v: float) -> float:
        return min(12.0, max(2.0, float(v)))

    @field_validator("overview_fps")
    @classmethod
    def _over(cls, v: float) -> float:
        return min(2.0, max(0.25, float(v)))

    @field_validator("signal_kinds")
    @classmethod
    def _kinds(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        kept = tuple(k for k in v if k in SIGNAL_KINDS)
        return kept or SIGNAL_KINDS


DEFAULT_PLAN = WatchPlan()


class Surprise(_Loose):
    t0: float
    t1: float | None = None
    score: float = 0.5
    why: str = ""


class Surprises(_Loose):
    surprises: tuple[Surprise, ...] = ()


class Action(_Loose):
    type: str
    t0: float | None = None
    t1: float | None = None
    t: float | None = None
    fps: float | None = None
    box: tuple[float, float, float, float] | None = None


# ------------------------------------------------------------------ prompts

PLAN_PROMPT = """Before you see a single frame, plan how to watch this video for the task above.
Decide: what exactly you are looking for; which screen regions need a close, full-resolution look
(normalized boxes [x0, y0, x1, y1], 0..1, origin top-left; e.g. a subtitle band near the bottom);
how many frames per second you need when inspecting a suspicious moment (2-12); how dense the first
skim should be (0.25-2 fps); and which low-level signal hints matter for this task, from:
flash, freeze, jump_cut, cut, cut_in_speech, silence_in_speech, local_change.
Reply with JSON only:
{"looking_for": ["..."], "regions": [{"name": "...", "box": [x0, y0, x1, y1]}], "dense_fps": 8,
 "overview_fps": 1, "signal_kinds": ["..."]}"""

PREDICT_PROMPT = """Now predict. Going through the overview in order, for each gap between consecutive frames say to
yourself what you expect to happen in between, given the picture so far and the narration. Then list the
moments where your expectation is uncertain or broken: something could have happened between frames,
the picture and the narration disagree, a region you care about changes unexpectedly, or a signal hint
points there. Score each 0..1 by how much it matters for the task. At most 12.
Reply with JSON only: {"surprises": [{"t0": <s>, "t1": <s>, "score": <0..1>, "why": "..."}]}"""

ACT_PROMPT = """You can request more evidence before answering. Remaining evidence budget: about {remaining} tokens
(a 1-second seek at 8 fps costs ~{seek_cost}, one crop ~{crop_cost}, a listen ~{listen_cost}).
Either request up to {max_actions} actions:
{{"actions": [{{"type": "seek", "t0": <s>, "t1": <s>, "fps": <2-12>}},
              {{"type": "crop", "t": <s>, "box": [x0, y0, x1, y1]}},
              {{"type": "listen", "t0": <s>, "t1": <s>}}]}}
or give your final answer in the task's JSON format wrapped as {{"final": <answer JSON>}}.
Reply with JSON only."""

FINAL_PROMPT = 'Evidence budget used up. Give your final answer now, wrapped as {"final": <answer JSON in the task format>}.'


# ------------------------------------------------------------------ helpers


def _ask(conv: Conversation, ev: Evidence, model: type[_Loose], default: _Loose) -> _Loose:
    reply = conv.send(ev.take())
    try:
        return model.model_validate(extract_json(reply.text))
    except (ValueError, ValidationError):
        return default


def rank_windows(cands: list[Candidate], surprises: tuple[Surprise, ...], duration: float,
                 kinds: tuple[str, ...]) -> list[tuple[float, float, float, str]]:
    """Merge signal candidates and model surprises into scored (t0, t1, score, why) windows.

    A window flagged by both the signals and the model gets a bonus: both eyes agree.
    """
    # (t0, t1, score, why, sources) where sources is a frozenset of {"signal", "model"}
    raw: list[tuple[float, float, float, str, frozenset[str]]] = []
    for c in cands:
        if c.kind in kinds:
            raw.append((c.t0, c.t1, c.score, f"signal: {c.kind}", frozenset({"signal"})))
    for s in surprises:
        t1 = s.t1 if s.t1 is not None else s.t0
        raw.append((min(s.t0, t1), max(s.t0, t1), min(1.0, max(0.0, s.score)) + 0.1,
                    f"you flagged: {s.why[:80]}", frozenset({"model"})))
    windows = []
    for t0, t1, score, why, src in raw:
        if t1 - t0 > MAX_WINDOW:  # long spans: look at the start and the end
            windows.append((t0, t0, score, why, src))
            windows.append((t1, t1, score * 0.9, why, src))
        else:
            windows.append((t0, t1, score, why, src))
    padded = sorted(((max(0.0, a - WINDOW_PAD), min(duration, b + WINDOW_PAD), s, w, src)
                     for a, b, s, w, src in windows), key=lambda x: x[0])
    merged: list[tuple[float, float, float, str, frozenset[str]]] = []
    for a, b, s, w, src in padded:
        if merged and a <= merged[-1][1] and b - merged[-1][0] <= MAX_WINDOW + 2 * WINDOW_PAD:
            pa, pb, ps, pw, psrc = merged[-1]
            both = psrc | src
            bonus = 0.15 if len(both) == 2 and psrc != both else 0.0  # second eye joins
            merged[-1] = (pa, max(pb, b), max(ps, s) + bonus, pw if ps >= s else w, both)
        else:
            merged.append((a, b, s, w, src))
    return sorted(((a, b, s, w) for a, b, s, w, _ in merged), key=lambda x: -x[2])


def _window_cost(ctx: VideoContext, plan: WatchPlan, use_crops: bool, a: float, b: float) -> int:
    cost = tools.seek_cost(ctx, a, b, plan.dense_fps)
    if use_crops:
        cost += sum(tools.crop_cost(ctx, r.to_box()) for r in plan.regions[:2]) * 2
    if ctx.info.has_audio:
        cost += tools.listen_cost(ctx, a, b)
    return cost


def _serve_window(ctx: VideoContext, ev: Evidence, plan: WatchPlan, use_crops: bool,
                  a: float, b: float, why: str) -> None:
    tools.seek(ctx, ev, a, b, plan.dense_fps, why)
    if use_crops:
        for r in plan.regions[:2]:
            tools.crops(ctx, ev, [a + (b - a) * 0.25, a + (b - a) * 0.75], r.to_box(), r.name)
    if ctx.info.has_audio:
        tools.listen(ctx, ev, a, b)


def _run_action(ctx: VideoContext, ev: Evidence, act: Action, plan: WatchPlan, use_crops: bool,
                remaining: int) -> tuple[int, str | None]:
    """Execute one requested action if it fits. Returns (tokens spent, refusal note)."""
    d = ctx.info.duration
    kind = act.type.lower()
    if kind == "seek" and act.t0 is not None:
        t1 = act.t1 if act.t1 is not None else act.t0 + 1.0
        fps = act.fps or plan.dense_fps
        cost = tools.seek_cost(ctx, act.t0, t1, fps)
        if cost > remaining:
            return 0, f"seek {act.t0:.2f}-{t1:.2f} skipped: needs {cost} tokens"
        return tools.seek(ctx, ev, act.t0, min(t1, act.t0 + 4.0), fps, "requested"), None
    if kind == "crop" and act.box is not None and (act.t is not None or act.t0 is not None):
        if not use_crops:
            return 0, "crop unavailable in this configuration"
        box = Box.clamp(*act.box)
        cost = tools.crop_cost(ctx, box)
        if cost > remaining:
            return 0, "crop skipped: over budget"
        t = act.t if act.t is not None else act.t0
        return tools.crops(ctx, ev, [min(max(0.0, t), d - 1e-3)], box), None
    if kind == "listen" and act.t0 is not None:
        t1 = act.t1 if act.t1 is not None else act.t0 + 1.0
        if not ctx.info.has_audio:
            return 0, "listen: this video has no audio"
        cost = tools.listen_cost(ctx, act.t0, t1)
        if cost > remaining:
            return 0, "listen skipped: over budget"
        return tools.listen(ctx, ev, act.t0, min(t1, act.t0 + 6.0)), None
    return 0, f"unrecognised action {act.model_dump()}"


# ------------------------------------------------------------------ main loop


def run_watcher(ctx: VideoContext, task: Task, prompt: str, budget: int, backend: Backend,
                *, arm: Arm = Arm.W) -> ArmResult:
    variant = VARIANTS[arm]
    started = time.monotonic()
    ev = Evidence()
    conv: Conversation | None = None
    try:
        conv = Conversation(backend.open(SYSTEM))
        ev.text(video_header(ctx))
        ev.text(prompt)
        if ctx.transcript.segments:
            ev.text("TRANSCRIPT (automatic speech recognition, timestamped):\n" + ctx.transcript.to_text())
            for s in ctx.transcript.segments:
                ev.observe("transcript", s.t0, s.t1)

        plan = DEFAULT_PLAN
        if variant.plan_first:
            ev.text(PLAN_PROMPT)
            plan = _ask(conv, ev, WatchPlan, DEFAULT_PLAN)
        if not variant.use_crops:
            plan = plan.model_copy(update={"regions": ()})

        # Short clip: watch it whole. Otherwise a low-res skim sized to at most ~25% of the budget.
        full = tools.full_view(ctx, budget)
        if full is not None:
            tools.overview(ctx, ev, full[1], full[0], full=True)
        else:
            fps = plan.overview_fps
            while fps > 0.25 and tools.overview_cost(ctx, fps) > budget * 0.25:
                fps /= 2
            tools.overview(ctx, ev, fps)
        cands: list[Candidate] = []
        if variant.use_signals:
            cands = [c for c in ctx.candidates if c.kind in plan.signal_kinds]
            add_hints(ev, select_hints(cands))

        surprises: tuple[Surprise, ...] = ()
        if variant.predict:
            ev.text(PREDICT_PROMPT)
            surprises = _ask(conv, ev, Surprises, Surprises()).surprises  # type: ignore[union-attr]

        # Focus: serve the most surprising windows densely.
        windows = rank_windows(cands, surprises, ctx.info.duration, plan.signal_kinds)
        served = 0
        for a, b, _score, why in windows:
            cost = _window_cost(ctx, plan, variant.use_crops, a, b)
            if ev.tokens + cost > budget * FOCUS_SHARE:
                continue
            _serve_window(ctx, ev, plan, variant.use_crops, a, b, why)
            served += 1
        if served == 0:
            ev.text("(no windows served: nothing stood out within budget)")

        # Act: the model asks for what it still needs.
        for _ in range(MAX_ACTION_ROUNDS):
            remaining = budget - ev.tokens
            if remaining < 300:
                break
            ev.text(ACT_PROMPT.format(remaining=remaining, max_actions=MAX_ACTIONS_PER_ROUND,
                                      seek_cost=tools.seek_cost(ctx, 0, 1, 8),
                                      crop_cost=tools.crop_cost(ctx, Box(0.05, 0.75, 0.95, 0.95)),
                                      listen_cost=60))
            reply = conv.send(ev.take())
            try:
                obj = extract_json(reply.text)
            except ValueError:
                obj = {}
            if "final" in obj:
                return finish(conv, task, ev, started, reply.text)
            actions = obj.get("actions") if isinstance(obj.get("actions"), list) else []
            notes = []
            for raw in actions[:MAX_ACTIONS_PER_ROUND]:
                try:
                    act = Action.model_validate(raw)
                except ValidationError:
                    notes.append(f"invalid action {raw}")
                    continue
                spent, note = _run_action(ctx, ev, act, plan, variant.use_crops, budget - ev.tokens)
                if note:
                    notes.append(note)
            if not actions:
                break
            if notes:
                ev.text("NOTES: " + "; ".join(notes))
        ev.text(FINAL_PROMPT)
        reply = conv.send(ev.take())
        return finish(conv, task, ev, started, reply.text)
    except RECOVERABLE as exc:
        return failed(conv, ev, started, exc)
    finally:
        if conv is not None:
            conv.session.close()

