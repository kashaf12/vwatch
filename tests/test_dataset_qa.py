from __future__ import annotations

import json

import numpy as np
import pytest
from test_dataset_fixtures import CUT, H, N, W, make_source

from vwatch.dataset import qa
from vwatch.dataset._common import FPS
from vwatch.dataset._source import SourceSpec
from vwatch.media import probe
from vwatch.schemas import QATruth

PROPS = {
    "city": "Bengaluru",
    "scenes": [
        {"headline": "Hook", "durationSec": 4.0, "captions": [{"text": "hi", "startSec": 0, "endSec": 4}]},
        {"headline": "Price", "value": "₹80", "durationSec": 3.9,
         "captions": [{"text": "20 रुपये से 80", "startSec": 1.0, "endSec": 3}]},
        {"headline": "Count", "value": "300", "durationSec": 0.1, "captions": []},
        {"headline": "Cost", "value": "₹25", "durationSec": 0.1,
         "captions": [{"text": "8 रुपये से 25", "startSec": 0.5, "endSec": 1}]},
        {"headline": "Odd", "value": "₹minus 78 - ₹6", "durationSec": 0.1, "captions": []},
    ],
}


def test_procedural_scene_answers_match_render():
    scene = qa.plan_scene(np.random.default_rng(4))
    items = qa.scene_items(scene)
    assert [i.id for i in items] == ["q1", "q2", "q3", "q4", "q5"]
    assert {i.answer_type for i in items} == {"number", "text"}
    for i in items:
        assert i.answer == i.answer.lower() and i.time is not None
        if i.answer_type == "number":
            assert i.answer.isdigit()
    assert 30 <= scene.duration <= 36 and scene.red_n != scene.blue_n
    # sign / card are constant across each question's event window
    for kind in ("bell", "triangle", "blue"):
        a, b = scene.windows[kind]
        assert len({scene.sign_at(f) for f in range(a, b)}) == 1
        assert len({scene.card_at(f) for f in range(a, b)}) == 1
    assert len(set(scene.sign_values)) == 5
    ws = sorted(scene.windows.values())
    assert all(b[0] - a[1] >= 2.4 * FPS - 1 for a, b in zip(ws, ws[1:], strict=False))


def test_procedural_render_shows_bell_and_blinks():
    scene = qa.plan_scene(np.random.default_rng(4))
    b0, b1 = scene.windows["bell"]
    r0, _ = scene.windows["red"]
    want = {b0 - 5, b0 + 5, r0 + 1, r0 + qa.BLINK_ON + 2}
    got = {i: f for i, f in enumerate(qa.render_frames(scene)) if i in want}
    bell_zone = (slice(90, 210), slice(210, 330))
    assert np.abs(got[b0 + 5][bell_zone].astype(int) - got[b0 - 5][bell_zone].astype(int)).sum() > 10000
    red_zone = (slice(820, 920), slice(60, 160))
    assert np.abs(got[r0 + 1][red_zone].astype(int) - got[r0 + qa.BLINK_ON + 2][red_zone].astype(int)).sum() > 10000


def test_mmss():
    assert qa.mmss(72.9) == "01:12" and qa.mmss(5) == "00:05"


def test_real_items_from_props():
    starts = [0.0, 4.2, 8.4, 8.5, 8.6]
    items = qa.real_items(PROPS, starts)
    assert len(items) == 5 and [i.answer for i in items[:3]] == ["80", "300", "25"]
    assert items[0].time == pytest.approx(5.7) and "'Price'" in items[0].question
    assert "rupee amount" in items[0].question and "What number" in items[1].question
    assert items[3].answer == "20" and items[3].time == pytest.approx(4.2 + 1.0 + 0.4)  # smaller of "20 se 80"
    assert items[4].answer == "bengaluru" and items[4].time is None


def test_real_items_needs_enough_questions():
    with pytest.raises(ValueError, match="need 5"):
        qa.real_items({"scenes": [PROPS["scenes"][0]]}, [0.0])


def test_scene_starts_snap_to_cuts_and_fail_loudly():
    props = {"scenes": [{"durationSec": 3.0}, {"durationSec": 3.0}, {"durationSec": 1.0}]}
    cuts = np.array([int(3.4 * FPS), int(6.8 * FPS)])
    assert qa.scene_starts(props, cuts, 7.6) == pytest.approx([0.0, 3.4, 6.8], abs=0.04)
    with pytest.raises(ValueError, match="no scene cut"):
        qa.scene_starts(props, np.array([], dtype=int), 7.6)


def test_build_t4_procedural_and_real(tmp_path):
    src = make_source()
    props_path = tmp_path / "props.json"
    props = {"city": "Pune", "scenes": [
        {"headline": "A", "value": "₹50", "durationSec": CUT / FPS - 0.2,
         "captions": [{"text": "5 रुपये से 9", "startSec": 1, "endSec": 2}]},
        {"headline": "B", "value": "300", "durationSec": 3.9,
         "captions": [{"text": "20 रुपये से 40", "startSec": 1, "endSec": 2}]}]}
    props_path.write_text(json.dumps(props))
    spec = SourceSpec("synthetic", tmp_path / "x.mp4", (0.78, 0.93), (0.78, 0.93), "light_text", 0.01, props_path)
    real_clip = type(src)(**{**src.__dict__, "spec": spec})
    items = qa.build_t4(tmp_path / "out", 2, 7, spec, loader=lambda _s: real_clip)
    assert [i.bucket for i in items] == ["synthetic", "real"]
    for it in items:
        truth = QATruth.model_validate_json((tmp_path / "out" / it.truth_path).read_text())
        assert len(truth.items) == 5
        info = probe(tmp_path / "out" / it.video_path)
        assert abs(info.duration - truth.duration) < 0.25
    real = QATruth.model_validate_json((tmp_path / "out" / items[1].truth_path).read_text())
    assert [i.answer for i in real.items] == ["50", "300", "5", "pune", "20"]
    assert real.items[0].time == pytest.approx(1.5)
    assert (W, H, N) == (108, 192, 240)


def test_build_t4_without_real_source_is_all_procedural(tmp_path):
    items = qa.build_t4(tmp_path, 1, 7, None)
    assert [i.video_id for i in items] == ["t4_proc_000"]
