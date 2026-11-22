from __future__ import annotations

import numpy as np
import pytest

from vwatch.dataset import synth_causal as sc
from vwatch.dataset import synth_events as se
from vwatch.dataset._common import FPS, HEIGHT, WIDTH
from vwatch.dataset._render import COLOUR_NAMES, SHAPES, rupee_label, shape_sprite
from vwatch.media import probe
from vwatch.schemas import CausalTruth, EventTruth


def frame_at(frames_iter, n):
    for i, f in enumerate(frames_iter):
        if i == n:
            return f
    raise AssertionError("clip too short")


def bucket(d: float) -> str:
    return "<0.5" if d < 0.5 else "0.5-1" if d < 1 else "1-3" if d < 3 else ">3"


@pytest.mark.parametrize("seed", range(12))
def test_event_scene_constraints(seed):
    scene = se.plan_scene(np.random.default_rng(seed))
    truth = se.scene_truth("x", scene)
    assert 30 <= truth.duration <= 45 and 8 <= len(truth.events) <= 12
    ev = truth.events
    for a, b in zip(ev, ev[1:], strict=False):
        assert b.start - a.end >= 1.5 - 1e-6
    assert ev[0].start >= 1.5 - 1e-6 and truth.duration - ev[-1].end >= 1.5 - 1e-6
    assert len({e.keywords for e in ev}) == len(ev)  # unique subject per event
    assert len({e.subject for e in ev}) == len(ev)
    for e in ev:
        assert e.keywords[0] in COLOUR_NAMES
        assert e.keywords[1] in SHAPES or (e.kind == "flash" and e.keywords[1] == "flash")
        assert e.subject == " ".join(e.keywords)
    assert {bucket(e.end - e.start) for e in ev} >= {"<0.5", "0.5-1", "1-3", ">3"}
    assert any(0.06 <= e.end - e.start <= 0.21 for e in ev)
    assert all(e.end - e.start <= 0.1 + 1e-6 for e in ev if e.kind == "flash")


def test_event_render_matches_truth():
    scene = se.plan_scene(np.random.default_rng(3))
    ev = scene.events
    first = next(e for e in ev if e.kind in ("appear", "move") and e.n >= 6)
    mid, before = first.start + first.n // 2, first.start - 3
    frames = {i: f for i, f in enumerate(se.render_frames(scene)) if i in (mid, before, first.start + first.n + 3)}
    assert frames[mid].shape == (HEIGHT, WIDTH, 3)
    diff_in = np.abs(frames[mid].astype(int) - frames[before].astype(int)).sum(-1) > 30
    assert diff_in.sum() > 3000  # a large shape appeared
    # the shape is gone again after its duration (only the slow drift differs)
    after = np.abs(frames[first.start + first.n + 3].astype(int) - frames[before].astype(int)).sum(-1) > 30
    assert after.sum() < diff_in.sum() / 2
    flash = next((e for e in ev if e.kind == "flash"), None)
    if flash is not None:
        f = frame_at(se.render_frames(scene), flash.start)
        assert (f == f[0, 0]).all()  # whole frame one colour


def test_event_deterministic_and_truth_roundtrip():
    a = se.scene_truth("x", se.plan_scene(np.random.default_rng(5)))
    b = se.scene_truth("x", se.plan_scene(np.random.default_rng(5)))
    c = se.scene_truth("x", se.plan_scene(np.random.default_rng(6)))
    assert a == b and a != c
    assert EventTruth.model_validate_json(a.model_dump_json()) == a


def test_build_t2_writes_video(tmp_path):
    items = se.build_t2(tmp_path, 1, 7)
    it = items[0]
    truth = EventTruth.model_validate_json((tmp_path / it.truth_path).read_text())
    info = probe(tmp_path / it.video_path)
    assert (info.width, info.height, round(info.fps)) == (540, 960, 30) and info.has_audio
    assert abs(info.duration - truth.duration) < 0.2 and it.bucket == "synthetic"


def test_shape_sprites_and_rupee():
    for s in SHAPES:
        sp = shape_sprite(s, (200, 30, 30), 80)
        assert sp.shape == (80, 80, 4) and 0.1 < (sp[..., 3] > 128).mean() < 0.95
    text, path = rupee_label(437)
    assert "437" in text and path
    with pytest.raises(ValueError):
        shape_sprite("hexagon", (1, 2, 3), 10)


@pytest.mark.parametrize("k", range(5))
def test_causal_plan(k):
    clip = sc.plan_clip(np.random.default_rng(100 + k), k)
    truth = sc.clip_truth(f"t3_{k}", clip)
    assert 12 <= truth.duration <= 25 and 2 <= len(truth.links) <= 4 and 1 <= len(truth.distractors) <= 2
    times = [lk.time for lk in truth.links]
    assert times == sorted(times) and 0 < times[0] and times[-1] < truth.duration
    chained = truth.links[1:] if clip.variant.endswith("_x") and clip.variant.startswith("ledge") else truth.links
    if not clip.variant.startswith("seesaw"):
        for a, b in zip(chained, chained[1:], strict=False):
            assert a.effect == b.cause  # each effect triggers the next link
    assert CausalTruth.model_validate_json(truth.model_dump_json()) == truth


def test_causal_variants_cover_templates_and_lengths():
    clips = [sc.plan_clip(np.random.default_rng(7 + k), k) for k in range(5)]
    assert len({c.variant.split("_")[0] for c in clips}) >= 3
    assert {len(c.truth_links) for c in clips} == {2, 3, 4}


def test_causal_effect_visible_at_truth_time():
    clip = sc.plan_clip(np.random.default_rng(7), 0)  # ledge: last link = lamp turns on
    assert clip.truth_links[-1][1] == "the lamp turns on"
    t = clip.truth_links[-1][2]
    want = {int((t - 0.3) * FPS), int((t + 0.5) * FPS)}
    got = {i: f for i, f in enumerate(sc.render_frames(clip)) if i in want}
    a, b = got[int((t - 0.3) * FPS)], got[int((t + 0.5) * FPS)]
    # the lamp glow (bright yellow) is present after the effect and not before
    yellow = lambda f: ((f[..., 0] > 240) & (f[..., 1] > 215) & (f[..., 2] < 140)).sum()  # noqa: E731
    assert yellow(b) > yellow(a) + 200


def test_build_t3_writes_video(tmp_path):
    it = sc.build_t3(tmp_path, 1, 7)[0]
    truth = CausalTruth.model_validate_json((tmp_path / it.truth_path).read_text())
    info = probe(tmp_path / it.video_path)
    assert abs(info.duration - truth.duration) < 0.2 and info.has_audio
