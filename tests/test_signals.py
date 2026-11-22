import numpy as np
import pytest

from vwatch import media, signals
from vwatch.signals import Candidate, SignalTrack

FPS = 30.0


def make_track(frames, rms_db=None, fps=FPS):
    frames = np.asarray(frames, dtype=np.uint8)
    diff, local = signals.frame_diffs(frames)
    n_hops = int(len(frames) / fps / signals.HOP_S)
    rms = np.full(n_hops, -20.0, dtype=np.float32) if rms_db is None else np.asarray(rms_db, np.float32)
    return SignalTrack(fps=fps, duration=len(frames) / fps, diff=diff, local=local, rms_db=rms,
                       frames=frames)


def flat(n, level=100, size=16):
    return np.full((n, size, size), level, dtype=np.uint8)


class TestPrimitives:
    def test_frame_diffs_values(self):
        f = flat(5)
        f[3:] = 110
        diff, local = signals.frame_diffs(f)
        assert diff.shape == (5,) and local.shape == (5,)
        assert diff[0] == 0 and diff[3] == pytest.approx(10) and diff[1] == 0

    def test_local_ratio_high_for_localized_change(self):
        f = flat(3)
        f[2, :2, :4] = 255  # one corner cell
        _, local = signals.frame_diffs(f)
        assert local[2] > 4

    def test_loudness_db(self):
        sr = 16000
        a = np.full(sr, 0.1, dtype=np.float32)
        db = signals.loudness_db(a, sr)
        assert len(db) == 50 and db[0] == pytest.approx(-20, abs=0.1)

    def test_loudness_short_audio(self):
        assert signals.loudness_db(np.zeros(3, dtype=np.float32), 16000).size == 0

    def test_speech_at(self):
        t = make_track(flat(30), rms_db=[-20] * 25 + [-80] * 25)
        assert t.speech_at(0.1) and not t.speech_at(0.9)
        assert not t.speech_at(-1) and not t.speech_at(99)

    def test_candidate_center(self):
        assert Candidate(1, 3, "flash", 0.5).center == 2


class TestFlash:
    def test_two_frame_excursion(self):
        f = flat(60)
        f[30:32] = 250
        t = make_track(f)
        flashes = signals.detect_flashes(t)
        assert len(flashes) == 1
        c = flashes[0]
        assert c.kind == "flash" and c.t0 == pytest.approx(30 / FPS) and c.t1 == pytest.approx(32 / FPS)
        assert c.score == 1.0

    def test_single_frame_excursion(self):
        f = flat(60)
        f[20] = 250
        flashes = signals.detect_flashes(make_track(f))
        assert len(flashes) == 1 and flashes[0].t0 == pytest.approx(20 / FPS)

    def test_permanent_change_is_not_flash(self):
        f = flat(60)
        f[30:] = 250
        assert signals.detect_flashes(make_track(f)) == []

    def test_long_excursion_not_flash(self):
        f = flat(60)
        f[20:30] = 250  # 10 frames > max_len
        assert signals.detect_flashes(make_track(f)) == []

    def test_static_video_nothing(self):
        t = make_track(flat(30))
        assert signals.find_candidates(t) == []

    def test_single_frame_video(self):
        diff, local = signals.frame_diffs(flat(1))
        assert diff.shape == (1,) and local.shape == (1,)


class TestCuts:
    def _cut_frames(self):
        f = flat(60)
        f[30:] = 250
        return f

    def test_cut_without_sound(self):
        t = make_track(self._cut_frames(), rms_db=np.full(100, -90.0))
        cuts = signals.detect_cuts(t, [])
        assert [c.kind for c in cuts] == ["cut"] and cuts[0].t0 == pytest.approx(1.0)
        assert cuts[0].score == 0.35

    def test_cut_in_speech(self):
        t = make_track(self._cut_frames(), rms_db=np.full(100, -20.0))
        cuts = signals.detect_cuts(t, [])
        assert [c.kind for c in cuts] == ["cut_in_speech"] and cuts[0].score == 0.75
        assert "sound is on" in cuts[0].note

    def test_cut_needs_sound_on_both_sides(self):
        db = np.full(100, -20.0)
        db[50:] = -90.0  # sound only before the cut (cut at 1.0 s = hop 50)
        t = make_track(self._cut_frames(), rms_db=db)
        assert signals.detect_cuts(t, [])[0].kind == "cut"

    def test_flash_frames_not_reported_as_cuts(self):
        f = flat(60)
        f[30:32] = 250
        t = make_track(f)
        flashes = signals.detect_flashes(t)
        assert signals.detect_cuts(t, flashes) == []
        assert len(signals.detect_cuts(t, [])) == 2  # without exclusion both edges are cuts


class TestFreeze:
    def _ramp(self, n=90, step=2, freeze=(40, 55)):
        f = np.zeros((n, 16, 16), dtype=np.uint8)
        level = 20
        for k in range(n):
            if not (freeze[0] <= k < freeze[1]):
                level += step
            f[k] = level
        return f

    def test_freeze_with_speech_and_motion_before(self):
        # burst of faster motion right before the freeze, steady motion elsewhere
        lv, frames = 20, []
        for k in range(90):
            if 40 <= k < 55:
                pass
            elif 35 <= k < 40:
                lv += 6
            else:
                lv += 1
            frames.append(np.full((16, 16), lv, dtype=np.uint8))
        t = make_track(frames, rms_db=np.full(150, -20.0))
        fr = signals.detect_freezes(t)
        assert len(fr) == 1
        assert fr[0].kind == "freeze" and fr[0].t0 == pytest.approx(40 / FPS)
        assert fr[0].t1 == pytest.approx(55 / FPS) and fr[0].score == 0.8

    def test_no_freeze_when_silent(self):
        lv, frames = 20, []
        for k in range(90):
            if not 40 <= k < 55:
                lv += 6 if 35 <= k < 40 else 1
            frames.append(np.full((16, 16), lv, dtype=np.uint8))
        t = make_track(frames, rms_db=np.full(150, -90.0))
        assert signals.detect_freezes(t) == []

    def test_short_still_ignored(self):
        lv, frames = 20, []
        for k in range(90):
            if not 40 <= k < 44:  # 4 frames = 0.13 s < 0.4
                lv += 6 if 35 <= k < 40 else 1
            frames.append(np.full((16, 16), lv, dtype=np.uint8))
        t = make_track(frames, rms_db=np.full(150, -20.0))
        assert signals.detect_freezes(t) == []

    def test_fully_static_video_is_not_a_freeze(self):
        # no motion anywhere: a still talking-head slide, not a defect
        t = make_track(flat(90), rms_db=np.full(150, -20.0))
        assert signals.detect_freezes(t) == []

    def test_freeze_inside_steady_motion(self):
        t = make_track(self._ramp(), rms_db=np.full(150, -20.0))
        assert len(signals.detect_freezes(t)) == 1


class TestSilence:
    def _db(self, before=50, gap=25, after=50):
        return np.concatenate([np.full(before, -20.0), np.full(gap, -90.0), np.full(after, -20.0)])

    def test_silence_between_sounds(self):
        t = make_track(flat(60), rms_db=self._db())
        out = signals.detect_silences(t)
        assert len(out) == 1
        c = out[0]
        assert c.kind == "silence_in_speech"
        assert c.t0 == pytest.approx(1.0) and c.t1 == pytest.approx(1.5)
        assert c.score == pytest.approx(0.65)

    def test_no_sound_before(self):
        t = make_track(flat(60), rms_db=self._db(before=0))
        assert signals.detect_silences(t) == []

    def test_no_sound_after(self):
        t = make_track(flat(60), rms_db=self._db(after=0))
        assert signals.detect_silences(t) == []

    def test_too_short(self):
        t = make_track(flat(60), rms_db=self._db(gap=10))  # 0.2 s < 0.3
        assert signals.detect_silences(t) == []

    def test_score_capped(self):
        t = make_track(flat(300), rms_db=self._db(gap=400, before=50, after=50))
        assert signals.detect_silences(t)[0].score == 0.9


class TestLocalChange:
    def test_run_detected(self):
        t = make_track(flat(30))
        local = np.zeros(30, dtype=np.float32)
        diff = np.zeros(30, dtype=np.float32)
        local[10:14], diff[10:14] = 6.0, 1.0
        t2 = SignalTrack(t.fps, t.duration, diff, local, t.rms_db, t.frames)
        out = signals.detect_local_changes(t2)
        assert len(out) == 1 and out[0].kind == "local_change"
        assert out[0].t0 == pytest.approx(10 / FPS) and out[0].t1 == pytest.approx(14 / FPS)

    def test_needs_both_conditions(self):
        t = make_track(flat(30))
        local = np.full(30, 9.0, dtype=np.float32)
        t2 = SignalTrack(t.fps, t.duration, np.zeros(30, dtype=np.float32), local, t.rms_db, t.frames)
        assert signals.detect_local_changes(t2) == []

    def test_end_to_end_localized_overlay(self):
        f = flat(40)
        f[15:25, :2, :4] = 255  # small overlay appears for 10 frames
        out = signals.detect_local_changes(make_track(f))
        assert out and all(c.kind == "local_change" for c in out)


class TestMergeDescribe:
    def test_merge_same_kind(self):
        a = Candidate(1.0, 1.1, "flash", 0.5)
        b = Candidate(1.2, 1.4, "flash", 0.9)
        out = signals.merge([b, a])
        assert out == [Candidate(1.0, 1.4, "flash", 0.9)]

    def test_no_merge_far_or_other_kind(self):
        cs = [Candidate(1.0, 1.1, "flash", 0.5), Candidate(2.0, 2.1, "flash", 0.5),
              Candidate(1.05, 1.2, "cut", 0.3)]
        out = signals.merge(cs)
        assert len(out) == 3 and [c.t0 for c in out] == sorted(c.t0 for c in out)

    def test_merge_empty(self):
        assert signals.merge([]) == []

    def test_describe_empty(self):
        assert "no unusual moments" in signals.describe([])

    def test_describe_lines(self):
        s = signals.describe([Candidate(1.0, 2.5, "freeze", 0.8, "still"),
                              Candidate(3.0, 3.0, "cut", 0.35, "jump")])
        lines = s.splitlines()
        assert len(lines) == 2
        assert "00:01.00–00:02.50 freeze (score 0.80): still" in lines[0]
        assert lines[1].startswith("- 00:03.00") and "cut" in lines[1]


@pytest.fixture(scope="module")
def flashy(tmp_path_factory):
    frames = np.full((90, 96, 64, 3), 90, dtype=np.uint8)
    frames[45:47] = 250
    audio = (0.3 * np.sin(np.arange(3 * 44100) / 10)).astype(np.float32)
    return media.write_video(frames, 30, tmp_path_factory.mktemp("s") / "f.mp4", audio=audio, crf=16)


class TestRealVideo:

    def test_compute_shapes(self, tone_video):
        t = signals.compute(tone_video)
        assert t.fps == pytest.approx(30) and t.duration == pytest.approx(4.0, abs=0.1)
        assert len(t.diff) == 120 and t.frames.shape == (120, 114, 64)
        assert abs(len(t.rms_db) - 200) <= 2
        assert t.rms_db.mean() > signals.SPEECH_DB  # tone is loud
        assert t.diff[0] == 0

    def test_silent_video_has_empty_loudness(self, silent_video):
        assert signals.compute(silent_video).rms_db.size == 0

    def test_flash_found_in_encoded_video(self, flashy):
        t = signals.compute(flashy)
        cands = signals.find_candidates(t)
        flashes = [c for c in cands if c.kind == "flash"]
        assert len(flashes) == 1
        assert flashes[0].t0 == pytest.approx(45 / 30, abs=0.07)
        assert flashes[0].t1 == pytest.approx(47 / 30, abs=0.07)
        assert not [c for c in cands if c.kind in ("cut", "cut_in_speech")]


class TestEdges:
    def test_spike_threshold_single_frame(self):
        assert signals._spike_threshold(np.zeros(1, dtype=np.float32)) == float("inf")

    def test_excursion_cut_off_by_end_of_video(self):
        f = flat(40)
        f[38:] = 250  # spike with no room to return
        assert signals.detect_flashes(make_track(f)) == []
