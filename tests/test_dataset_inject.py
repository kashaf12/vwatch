from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_dataset_fixtures import CUT, H, N, SPEC, W, make_source

from vwatch.dataset import inject as inj
from vwatch.dataset._common import FPS, SPF, SR
from vwatch.dataset._source import iter_plan_frames
from vwatch.media import extract_audio, probe
from vwatch.schemas import DefectTruth, DefectType


@pytest.fixture(scope="module")
def src():
    return make_source()


def out_frames(src, plan):
    return np.stack(list(iter_plan_frames(src.frames, plan)))


def run(src, kind, seed=1, frac=0.5):
    rng = np.random.default_rng(seed)
    lo, hi = inj._span(src)
    return inj.INJECTORS[kind](src, rng, int(lo + frac * (hi - lo - 1)))


def test_source_analysis(src):
    assert src.n == N and len(src.audio) == N * SPF
    assert any(abs(int(c) - CUT) <= 1 for c in src.cuts)
    assert src.speech[100:300].all() and not src.speech[405:425].any()  # pause at 4.0-4.3 s
    assert src.cap_on[30:200].all() and not src.cap_on[:8].any()


@pytest.mark.parametrize("seed", range(8))
def test_flash_frame(src, seed):
    r = run(src, DefectType.FLASH_FRAME, seed, frac=seed / 8)
    d = r.defect
    s, e = round(d.start * FPS), round(d.end * FPS)
    assert 1 <= e - s <= 3 and d.type is DefectType.FLASH_FRAME
    assert all(abs(c - s) >= FPS // 2 and abs(c - e) >= FPS // 2 for c in src.cuts)
    out = out_frames(src, r.plan)
    assert len(out) == N
    for i in range(s, e):
        assert np.abs(out[i].astype(int) - src.frames[i].astype(int)).mean() > 8
    others = np.ones(N, bool)
    others[s:e] = False
    assert np.array_equal(out[others], src.frames[others])
    assert np.array_equal(r.plan.audio, src.audio)


def test_frozen_frame(src):
    r = run(src, DefectType.FROZEN_FRAME, 3)
    s, e = round(r.defect.start * FPS), round(r.defect.end * FPS)
    assert 15 <= e - s <= 60
    out = out_frames(src, r.plan)
    assert all(np.array_equal(out[i], src.frames[s]) for i in range(s, e))
    assert not np.array_equal(src.frames[s], src.frames[e - 1])  # freezing hides a real change
    assert np.array_equal(out[e:], src.frames[e:])
    assert np.array_equal(r.plan.audio, src.audio)


@pytest.mark.parametrize("seed", range(4))
def test_jump_cut(src, seed):
    r = run(src, DefectType.JUMP_CUT, seed, frac=0.2 + 0.2 * seed)
    d = r.defect
    k = N - r.plan.n_frames
    a = round(d.start * FPS)
    assert d.start == d.end and 9 <= k <= 18 and d.detail == f"removed {k / FPS:.2f}s"
    assert len(r.plan.audio) == (N - k) * SPF
    out = out_frames(src, r.plan)
    assert np.array_equal(out[a], src.frames[a + k]) and np.array_equal(out[a - 1], src.frames[a - 1])
    assert np.array_equal(r.plan.audio[a * SPF:], src.audio[(a + k) * SPF:])
    assert all(abs(c - a) >= FPS for c in src.cuts)
    assert src.window_speech(a / FPS - 0.3, a / FPS) == 1.0


def test_audio_dropout(src):
    r = run(src, DefectType.AUDIO_DROPOUT, 5)
    d = r.defect
    assert 0.4 <= d.end - d.start <= 2.0
    s, e = round(d.start * SR), round(d.end * SR)
    assert np.abs(r.plan.audio[s:e]).max() == 0
    assert np.abs(src.audio[s:e]).max() > 0.05  # it was speech
    assert np.array_equal(r.plan.audio[: s - 500], src.audio[: s - 500])  # outside the 10 ms fades
    assert np.array_equal(r.plan.audio[e + 500:], src.audio[e + 500:])
    assert abs(r.plan.audio[s - 5]) < abs(src.audio[s - 5]) + 1e-9
    assert np.array_equal(out_frames(src, r.plan), src.frames)


def test_caption_blocked(src):
    r = run(src, DefectType.CAPTION_BLOCKED, 2)
    d = r.defect
    s, e = round(d.start * FPS), round(d.end * FPS)
    assert 1.0 <= d.end - d.start <= 3.0 and src.cap_on[s:e].mean() >= 0.85
    out = out_frames(src, r.plan)
    y0, y1 = int(0.80 * H), int(0.91 * H)
    changed = np.abs(out[s + 1, y0:y1].astype(int) - src.frames[s + 1, y0:y1].astype(int)).sum(-1) > 0
    assert changed.mean() > 0.6  # an opaque box now covers most of the band
    # opaque: the covered pixels no longer depend on the original frame
    ov = r.plan.overlays[0]
    assert ov.sprite[ov.sprite.shape[0] // 2, ov.sprite.shape[1] // 2, 3] == 255
    assert np.array_equal(out[:s], src.frames[:s]) and np.array_equal(out[e:], src.frames[e:])


def test_wrong_number(src):
    r = run(src, DefectType.WRONG_NUMBER, 4)
    d = r.defect
    s, e = round(d.start * FPS), round(d.end * FPS)
    n = int(d.detail.split()[1])
    assert d.detail.startswith("shown ") and 100 < n < 1000 and n % 2 == 1
    assert 2.0 <= d.end - d.start <= 4.0
    out = out_frames(src, r.plan)
    top = slice(int(0.25 * H), int(0.55 * H))
    assert np.abs(out[s + 1, top].astype(int) - src.frames[s + 1, top].astype(int)).sum() > 0
    assert np.array_equal(out[:s], src.frames[:s]) and np.array_equal(out[e:], src.frames[e:])
    assert src.window_speech(d.start, d.end) >= 0.7


def test_caption_numbers(tmp_path):
    p = tmp_path / "props.json"
    p.write_text(json.dumps({"scenes": [
        {"durationSec": 5, "captions": [{"text": "20 rupees to 80", "startSec": 0, "endSec": 5}]},
        {"durationSec": 5, "captions": [{"text": "300 sold", "startSec": 0, "endSec": 5}]}]}))
    assert inj.caption_numbers(p, 1, 2) == {"20", "80"}
    assert inj.caption_numbers(p, 9, 10) == {"300", "20", "80"} or inj.caption_numbers(p, 9, 10) >= {"300"}
    assert inj.caption_numbers(None, 0, 1) == set()


def test_deterministic(src):
    a = run(src, DefectType.FROZEN_FRAME, 11)
    b = run(src, DefectType.FROZEN_FRAME, 11)
    assert a.defect == b.defect and np.array_equal(a.plan.src_idx, b.plan.src_idx)


def test_no_valid_placement_fails_loudly(src):
    with pytest.raises(ValueError, match="no valid placement"):
        inj.pick_nearest([], 5)


@pytest.mark.parametrize("scale,total", [("tiny", 7), ("pilot", 15), ("full", 240)])
def test_job_counts(scale, total):
    jobs = inj.plan_t1_jobs(scale, 7, 4)
    assert len(jobs) == total and len({j.video_id for j in jobs}) == total
    assert sum(j.defect is None for j in jobs) == {"tiny": 1, "pilot": 3, "full": 60}[scale]
    assert all(0 <= j.frac <= 1 for j in jobs)
    for kind in DefectType:  # stratified targets spread over the whole range
        f = sorted(j.frac for j in jobs if j.defect is kind)
        if scale == "full":
            assert f[0] < 0.1 and f[-1] > 0.9


def test_unknown_scale():
    with pytest.raises(ValueError, match="unknown scale"):
        inj.plan_t1_jobs("huge", 7, 4)


def test_build_t1_end_to_end(tmp_path, src):
    items = inj.build_t1(tmp_path, "tiny", 7, [SPEC], loader=lambda _s: src)
    assert len(items) == 7
    kinds = {i.bucket for i in items}
    assert kinds == {k.value for k in DefectType} | {"clean"}
    for it in items:
        truth = DefectTruth.model_validate_json((tmp_path / it.truth_path).read_text())
        info = probe(tmp_path / it.video_path)
        assert abs(info.duration - truth.duration) < 0.15 and (info.width, info.height) == (W, H)
        assert info.has_audio
        assert (len(truth.defects) == 0) == (it.bucket == "clean")
        for d in truth.defects:
            assert 0 <= d.start <= d.end <= truth.duration + 1e-6
    # the encoded jump-cut file really is shorter and the dropout really is silent
    by = {i.bucket: i for i in items}
    jc = DefectTruth.model_validate_json((tmp_path / by["jump_cut"].truth_path).read_text())
    assert jc.duration < N / FPS - 0.25
    ad = DefectTruth.model_validate_json((tmp_path / by["audio_dropout"].truth_path).read_text()).defects[0]
    aud = extract_audio(tmp_path / by["audio_dropout"].video_path, sr=44100)
    mid = slice(int((ad.start + 0.1) * 44100), int((ad.end - 0.1) * 44100))
    assert np.abs(aud[mid]).max() < 0.01
    # determinism
    again = inj.build_t1(tmp_path / "b", "tiny", 7, [SPEC], loader=lambda _s: src)
    assert [i.model_dump() for i in again] == [i.model_dump() for i in items]
    for it in items:
        assert (tmp_path / it.truth_path).read_text() == (tmp_path / "b" / it.truth_path).read_text()


def test_clean_differs_by_trim(src):
    plans = [inj.make_clean(src, np.random.default_rng(s)) for s in range(6)]
    assert len({p.n_frames for p in plans}) > 1 and all(p.n_frames > N - 91 for p in plans)
    assert isinstance(Path("x"), Path)
