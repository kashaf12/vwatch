"""T4: targeted-QA clips. 5 questions per video, each answerable only by looking at the right moment.

(a) procedural clips: a number sign and a word card that change between events, a bell icon, a
    yellow triangle, corner shapes and blinking squares;
(b) the real Short with a props.json: questions generated from the scene list.
"""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vwatch.dataset import _render as R
from vwatch.dataset._common import FPS, HEIGHT, SR, WIDTH, finish_item, item_rng, silent_audio, video_out
from vwatch.dataset._encode import write_video_stream
from vwatch.dataset._source import DEFAULT_SOURCES, SourceClip, SourceSpec, identity_plan, iter_plan_frames, load_source
from vwatch.schemas import DatasetItem, QAItem, QATruth, Task

WORDS = ("cat", "dog", "sun", "moon", "tree", "fish", "bird", "star", "rain", "milk", "book", "cake")
CORNER_COLOURS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "brown")
EVENT_KINDS = ("bell", "triangle", "corner", "red", "blue")
BLINK_ON, BLINK_PERIOD = 6, 15  # frames


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


@dataclass(frozen=True)
class QAScene:
    n_frames: int
    windows: dict[str, tuple[int, int]]  # event kind -> [start, end) frames
    sign_cuts: tuple[int, ...]  # frames at which the sign number changes
    sign_values: tuple[int, ...]  # len(sign_cuts) + 1
    card_cuts: tuple[int, ...]
    card_words: tuple[str, ...]
    red_n: int
    blue_n: int
    tl_colour: str
    tr_colour: str
    tl_shape: str
    tr_shape: str
    drift_phase: float

    @property
    def duration(self) -> float:
        return self.n_frames / FPS

    def sign_at(self, f: int) -> int:
        return self.sign_values[bisect.bisect_right(self.sign_cuts, f)]

    def card_at(self, f: int) -> str:
        return self.card_words[bisect.bisect_right(self.card_cuts, f)]


def plan_scene(rng: np.random.Generator) -> QAScene:
    order = [EVENT_KINDS[i] for i in rng.permutation(len(EVENT_KINDS))]
    red_n = int(rng.integers(3, 7))
    blue_n = int(rng.choice([n for n in range(2, 7) if n != red_n]))
    secs = {"bell": 2.0, "triangle": 2.0, "corner": 2.0, "red": red_n * 0.5, "blue": blue_n * 0.5}
    total = float(rng.uniform(30.0, 36.0))
    lead, tail, min_gap = 2.5, 2.0, 2.4
    slack = total - lead - tail - sum(secs.values()) - min_gap * 4
    gaps = min_gap + rng.dirichlet(np.ones(4)) * slack
    t = lead
    windows: dict[str, tuple[int, int]] = {}
    sign_cuts, card_cuts = [], []
    for i, kind in enumerate(order):
        windows[kind] = (int(round(t * FPS)), int(round((t + secs[kind]) * FPS)))
        t += secs[kind]
        if i < 4:
            card_cuts.append(int(round((t + 0.35 * gaps[i]) * FPS)))
            sign_cuts.append(int(round((t + 0.65 * gaps[i]) * FPS)))
            t += float(gaps[i])
    nums = rng.choice(np.arange(10, 1000), size=5, replace=False)
    words = [WORDS[i] for i in rng.permutation(len(WORDS))[:5]]
    cols = [CORNER_COLOURS[i] for i in rng.permutation(len(CORNER_COLOURS))[:2]]
    shapes = [R.SHAPES[i] for i in rng.permutation(len(R.SHAPES))[:2]]
    return QAScene(int(round(total * FPS)), windows, tuple(sign_cuts), tuple(int(n) for n in nums), tuple(card_cuts),
                   tuple(words), red_n, blue_n, cols[0], cols[1], shapes[0], shapes[1], float(rng.uniform(0, 6.28)))


def render_frames(scene: QAScene) -> Iterator[np.ndarray]:
    bg = R.gradient(HEIGHT, WIDTH, (92, 104, 128), (138, 148, 170))
    signs = {v: R.text_sprite(str(v), 110, (255, 255, 255), (30, 40, 60), pad=28, radius=20, min_w=340)
             for v in scene.sign_values}
    cards = {w: R.text_sprite(w, 70, (20, 20, 20), (250, 245, 225), pad=22, radius=18, min_w=380)
             for w in scene.card_words}
    bell = R.bell_sprite(120)
    tri = R.shape_sprite("triangle", R.COLOURS["yellow"], 120)
    tl = R.shape_sprite(scene.tl_shape, R.COLOURS[scene.tl_colour], 130)
    tr = R.shape_sprite(scene.tr_shape, R.COLOURS[scene.tr_colour], 130)
    red = R.shape_sprite("square", R.COLOURS["red"], 100)
    blue = R.shape_sprite("square", R.COLOURS["blue"], 100)
    ring = R.shape_sprite("ring", R.COLOURS["white"], 56)
    for f in range(scene.n_frames):
        canvas = bg.copy()
        R.paste(canvas, signs[scene.sign_at(f)], 270, 420)
        R.paste(canvas, cards[scene.card_at(f)], 270, 640)
        R.paste(canvas, ring, 270 + 150 * np.sin(2 * np.pi * f / FPS / 15 + scene.drift_phase), 790)
        w = scene.windows
        if w["bell"][0] <= f < w["bell"][1]:
            R.paste(canvas, bell, 270, 150)
        if w["triangle"][0] <= f < w["triangle"][1]:
            R.paste(canvas, tri, 270, 150)
        if w["corner"][0] <= f < w["corner"][1]:
            R.paste(canvas, tl, 105, 110)
            R.paste(canvas, tr, 435, 110)
        for kind, sprite, n, x in (("red", red, scene.red_n, 110), ("blue", blue, scene.blue_n, 430)):
            k = f - w[kind][0]
            if 0 <= k < n * BLINK_PERIOD and k % BLINK_PERIOD < BLINK_ON:
                R.paste(canvas, sprite, x, 870)
        yield canvas


def scene_items(scene: QAScene) -> tuple[QAItem, ...]:
    w = {k: v[0] / FPS for k, v in scene.windows.items()}
    f = lambda kind: scene.windows[kind][0] + 3  # noqa: E731  (a moment safely inside the window)
    corner_t = w["corner"] + 1.0
    return (
        QAItem(id="q1", question="What number is on the sign when the bell icon appears?",
               answer=str(scene.sign_at(f("bell"))), answer_type="number", time=round(w["bell"], 3)),
        QAItem(id="q2", question="How many times does the red square flash?", answer=str(scene.red_n),
               answer_type="number", time=round(w["red"], 3)),
        QAItem(id="q3", question=f"What colour is the shape that appears at the top-left around {mmss(corner_t)}?",
               answer=scene.tl_colour, answer_type="text", time=round(w["corner"], 3)),
        QAItem(id="q4", question="What word is on the card when the yellow triangle appears?",
               answer=scene.card_at(f("triangle")), answer_type="text", time=round(w["triangle"], 3)),
        QAItem(id="q5", question="What number is on the sign when the blue square starts flashing?",
               answer=str(scene.sign_at(f("blue"))), answer_type="number", time=round(w["blue"], 3)),
    )


# ------------------------------------------------------------------ real Short (props.json)


def scene_starts(props: dict, cuts: np.ndarray, video_duration: float) -> list[float]:
    """Scene start times in the video. props durations omit the transitions, so estimate the drift
    linearly and snap each scene to the nearest detected cut (fail loudly if none is close)."""
    scenes = props["scenes"]
    total = sum(float(s["durationSec"]) for s in scenes)
    drift = (video_duration - total) / len(scenes)
    cut_t = cuts / FPS
    starts, cum = [0.0], 0.0
    for k, s in enumerate(scenes[:-1], start=1):
        cum += float(s["durationSec"])
        est = cum + k * drift
        if len(cut_t) == 0 or np.min(np.abs(cut_t - est)) > 0.8:
            raise ValueError(f"no scene cut within 0.8 s of the estimated start of scene {k + 1} ({est:.2f}s)")
        starts.append(float(cut_t[np.argmin(np.abs(cut_t - est))]))
    return starts


def real_items(props: dict, starts: Sequence[float]) -> tuple[QAItem, ...]:
    scenes = props["scenes"]
    values: list[QAItem] = []
    captions: list[QAItem] = []
    for s, t0 in zip(scenes, starts, strict=True):
        m = re.fullmatch(r"\s*(₹)?\s*(\d+)\s*", str(s.get("value") or ""))  # plain card values only
        if m:
            what = "rupee amount" if m.group(1) else "number"
            values.append(QAItem(id="", question=f"What {what} is shown on screen in the scene whose headline is "
                                                 f"'{s['headline']}'?", answer=m.group(2), answer_type="number",
                                 time=round(t0 + 1.5, 3)))
        for cap in s["captions"]:  # first "A रुपये से B" caption of the scene
            m = re.search(r"(\d+)\s*रुपये\s*से\s*(\d+)", cap["text"])
            if m:
                lo = min(int(m.group(1)), int(m.group(2)))
                captions.append(QAItem(
                    id="", question=f"In the scene whose headline is '{s['headline']}', what is the smaller of the "
                                    "two numbers in the caption that shows two numbers?",
                    answer=str(lo), answer_type="number", time=round(t0 + cap["startSec"] + 0.4, 3)))
                break
    city = [QAItem(id="", question="Which city is named in the small text at the bottom of the screen?",
                   answer=str(props["city"]).lower(), answer_type="text", time=None)] if props.get("city") else []
    ordered = values[:3] + captions[:1] + city + captions[1:] + values[3:]  # mix kinds, then fill up
    if len(ordered) < 5:
        raise ValueError(f"only {len(ordered)} questions could be derived from props.json; need 5")
    return tuple(q.model_copy(update={"id": f"q{i + 1}"}) for i, q in enumerate(ordered[:5]))


def build_real(root: Path, spec: SourceSpec, loader: Callable[[SourceSpec], SourceClip], video_id: str) -> DatasetItem:
    if spec.props_path is None:
        raise ValueError(f"{spec.name} has no props.json")
    props = json.loads(spec.props_path.read_text(encoding="utf-8"))
    src = loader(spec)
    items = real_items(props, scene_starts(props, src.cuts, src.duration))
    plan = identity_plan(src.n, src.audio)
    h, w = src.frames.shape[1:3]
    write_video_stream(iter_plan_frames(src.frames, plan), FPS, video_out(root, Task.QA, video_id), size=(w, h),
                       audio=plan.audio)
    return finish_item(root, Task.QA, video_id, QATruth(video_id=video_id, duration=src.duration, items=items), "real")


def build_t4(root: Path, count: int, seed: int, real_source: SourceSpec | None = DEFAULT_SOURCES[2],
             loader: Callable[[SourceSpec], SourceClip] = load_source) -> list[DatasetItem]:
    n_real = 1 if (count >= 2 and real_source is not None) else 0
    items: list[DatasetItem] = []
    for k in range(count - n_real):
        scene = plan_scene(item_rng(seed, 4, k))
        vid = f"t4_proc_{k:03d}"
        write_video_stream(render_frames(scene), FPS, video_out(root, Task.QA, vid), size=(WIDTH, HEIGHT),
                           audio=silent_audio(scene.duration), sr=SR)
        truth = QATruth(video_id=vid, duration=scene.duration, items=scene_items(scene))
        items.append(finish_item(root, Task.QA, vid, truth, "synthetic"))
        print(f"[t4] {vid}", flush=True)
    if n_real:
        assert real_source is not None
        items.append(build_real(root, real_source, loader, "t4_real_000"))
        print("[t4] t4_real_000", flush=True)
    return items
