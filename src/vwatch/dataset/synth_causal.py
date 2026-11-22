"""T3: procedural cause-and-effect clips (12-25 s): a chain of 2-4 cause->effect links plus 1-2
distractor events that cause nothing. Five template variants, colours/objects/ordering/speed vary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from vwatch.dataset import _causal_scene as S
from vwatch.dataset import _causal_templates as T
from vwatch.dataset._common import FPS, HEIGHT, SR, WIDTH, finish_item, item_rng, silent_audio, video_out
from vwatch.dataset._encode import write_video_stream
from vwatch.schemas import CausalLink, CausalTruth, DatasetItem, Task

CHAIN_COLOURS = ("red", "green", "blue", "yellow", "orange", "purple", "pink")
VARIANTS = (("ledge", False), ("domino", False), ("seesaw", False), ("ledge", True), ("domino", True))
DISTRACTORS = ("bird", "cloud", "kite")


@dataclass(frozen=True)
class CausalClip:
    variant: str
    n_frames: int
    actors: tuple[S.Actor, ...]
    bg: Image.Image
    flip: bool
    truth_links: tuple[tuple[str, str, float], ...]  # (cause, effect, time in output seconds)
    distractors: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.n_frames / FPS


def plan_clip(rng: np.random.Generator, k: int) -> CausalClip:
    name, extra = VARIANTS[k % len(VARIANTS)]
    cols = [CHAIN_COLOURS[i] for i in rng.permutation(len(CHAIN_COLOURS))]
    if name == "ledge":
        built = T.ledge_template(rng, cols, extra)
    elif name == "domino":
        built = T.domino_template(rng, cols, extra)
    else:
        built = T.seesaw_template(rng, cols)
    speed = float(rng.uniform(1.0, 1.6))
    chain = built.chain_end * speed
    total = max(float(rng.uniform(12.0, 25.0)), chain + 4.5)
    if total > 25.0:
        raise RuntimeError(f"chain too long ({chain:.1f}s) for a 25 s clip")
    lead = float(rng.uniform(1.5, total - chain - 2.5))
    actors = [a.retimed(speed, lead) for a in built.actors]
    used = {c for link in built.links for c in CHAIN_COLOURS if c in link[0] or c in link[1]}
    spare = [c for c in cols if c not in used] or cols
    n_d = int(rng.integers(1, 3))
    phrases: list[str] = []
    for i, kind in enumerate(rng.permutation(DISTRACTORS)[:n_d]):
        phrase, actor = T.distractor(rng, str(kind), spare[i % len(spare)], total)
        phrases.append(phrase)
        actors.append(actor)
    links = tuple((c, e, round(lead + t * speed, 3)) for c, e, t in built.links)
    name_id = f"{name}{'_x' if extra else ''}"
    return CausalClip(name_id, int(round(total * FPS)), tuple(actors), built.bg, bool(rng.random() < 0.5), links,
                      tuple(phrases))


def clip_truth(video_id: str, clip: CausalClip) -> CausalTruth:
    links = tuple(CausalLink(id=f"l{i + 1}", cause=c, effect=e, time=t) for i, (c, e, t) in enumerate(clip.truth_links))
    return CausalTruth(video_id=video_id, duration=clip.duration, links=links, distractors=clip.distractors)


def render_frames(clip: CausalClip) -> Iterator[np.ndarray]:
    return S.render(clip.actors, clip.bg, clip.n_frames, FPS, clip.flip)


def build_t3(root: Path, count: int, seed: int) -> list[DatasetItem]:
    items = []
    for k in range(count):
        clip = plan_clip(item_rng(seed, 3, k), k)
        vid = f"t3_{clip.variant}_{k:03d}"
        write_video_stream(render_frames(clip), FPS, video_out(root, Task.CAUSAL, vid), size=(WIDTH, HEIGHT),
                           audio=silent_audio(clip.duration), sr=SR)
        items.append(finish_item(root, Task.CAUSAL, vid, clip_truth(vid, clip), clip.variant))
        print(f"[t3] {vid} {len(clip.truth_links)} links {clip.duration:.1f}s", flush=True)
    return items
