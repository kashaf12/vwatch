import dataclasses

import numpy as np
import pytest

from vwatch import detectors, media, signals
from vwatch.signals import SignalTrack

FPS = 30.0
SR = 44100


def track(diff, *, rms_db=None, cell_max=None, seam_ratio=None, seam_peak=None, fps=FPS):
    diff = np.asarray(diff, dtype=np.float32)
    n = len(diff)
    hops = int(n / fps / signals.HOP_S)
    rms = np.full(hops, -20.0, dtype=np.float32) if rms_db is None else np.asarray(rms_db, np.float32)
    opt = [None if a is None else np.asarray(a, np.float32) for a in (cell_max, seam_ratio, seam_peak)]
    return SignalTrack(fps=fps, duration=n / fps, diff=diff, local=np.zeros(n, np.float32), rms_db=rms,
                       frames=np.zeros((n, 4, 4), np.uint8), cell_max=opt[0], seam_ratio=opt[1], seam_peak=opt[2])


def voice(seconds: float, sr: int = SR) -> np.ndarray:
    """Loud harmonic 'voice': two partials that do not line up with the frame rate."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 173 * t) + 0.1 * np.sin(2 * np.pi * 347 * t)).astype(np.float32)


# ------------------------------------------------------------------ tracks


class TestCellMotion:
    def test_localized_change_shows_in_one_cell(self):
        g = np.full((3, 64, 64), 100, np.uint8)
        g[2, :8, :8] = 180  # exactly one 8x8 cell changes by 80
        out = detectors.cell_motion(g)
        assert out[0] == 0 and out[1] == 0
        assert out[2] == pytest.approx(80)

    def test_whole_frame_change(self):
        g = np.full((2, 32, 32), 10, np.uint8)
        g[1] = 30
        assert detectors.cell_motion(g)[1] == pytest.approx(20)

    def test_short_inputs(self):
        assert detectors.cell_motion(np.zeros((1, 8, 8), np.uint8)).tolist() == [0.0]
        assert detectors.cell_motion(np.zeros((0, 8, 8), np.uint8)).size == 0


class TestAudioSeams:
    def _spliced(self, a=30, k=10, n_src=60):
        spf = SR // int(FPS)
        x = voice(n_src / FPS)
        y = np.concatenate([x[: a * spf], x[(a + k) * spf:]])
        return y, n_src - k

    def test_splice_click_on_its_frame_boundary(self):
        y, n = self._spliced()
        ratio, peak = detectors.audio_seams(y, SR, FPS, n)
        assert ratio.shape == peak.shape == (n,)
        assert ratio[30] > 50 and peak[30] > 2
        others = np.delete(ratio[2:-2], 28)  # all boundaries but the splice
        assert others.max() < 6  # a steady tone: peak vs median of a periodic signal

    def test_unspliced_audio_is_flat(self):
        ratio, peak = detectors.audio_seams(voice(2.0), SR, FPS, 60)
        assert ratio[1:-2].max() < 6 and peak[1:-2].max() < 1.5

    def test_no_audio(self):
        ratio, peak = detectors.audio_seams(np.zeros(0, np.float32), SR, FPS, 10)
        assert ratio.tolist() == [0.0] * 10 and peak.tolist() == [0.0] * 10

    def test_digital_silence_scores_zero(self):
        ratio, _ = detectors.audio_seams(np.zeros(SR, np.float32), SR, FPS, 30)
        assert not ratio.any()


class TestSpeechMask:
    def test_quiet_recording_counts_as_sound(self):
        db = np.full(50, -45.0)  # below the fixed -38 dB speech level, well above silence
        assert detectors.speech_mask(db, -38.0, -55.0).all()

    def test_silence_stays_silence(self):
        assert not detectors.speech_mask(np.full(50, -90.0), -38.0, -55.0).any()

    def test_pause_inside_loud_speech(self):
        db = np.r_[np.full(40, -20.0), np.full(10, -50.0)]
        m = detectors.speech_mask(db, -38.0, -55.0)
        assert m[:40].all() and not m[40:].any()

    def test_empty(self):
        assert detectors.speech_mask(np.zeros(0), -38.0, -55.0).size == 0


# ------------------------------------------------------------------ freeze


def held(resume=5.0, hold_cell=0.02, around=0.5, n=150, a=60, b=90):
    diff = np.full(n, around, np.float32)
    diff[0] = 0
    diff[a:b] = 0.0
    diff[b] = resume
    cells = np.full(n, 1.0, np.float32)
    cells[a:b] = hold_cell
    return diff, cells


class TestFreeze:
    def test_catch_up_after_hold(self):
        diff, cells = held()
        out = detectors.detect_freezes(track(diff, cell_max=cells))
        assert len(out) == 1
        c = out[0]
        assert c.kind == "freeze" and c.score == 0.9
        assert c.t0 == pytest.approx(2.0) and c.t1 == pytest.approx(3.0)
        assert "catch-up" in c.note and "1.00s" in c.note

    def test_gentle_restart_is_a_natural_pause(self):
        diff, cells = held(resume=0.5)
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_cut_after_hold_is_reported_low(self):
        diff, cells = held(resume=40.0)
        out = detectors.detect_freezes(track(diff, cell_max=cells))
        assert [c.score for c in out] == [0.55] and "scene cut" in out[0].note

    def test_mid_res_motion_rejects_hold(self):
        diff, cells = held(hold_cell=0.5)  # lips move: 64 px average still, cells are not
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_encoder_settle_frames_tolerated(self):
        diff, cells = held()
        cells[60:63] = 0.4  # first frames of a hold are still settling after re-encode
        cells[85] = 0.3  # and one refresh inside it
        assert len(detectors.detect_freezes(track(diff, cell_max=cells))) == 1

    def test_without_mid_res_track_uses_frame_diff(self):
        diff, _ = held()
        assert len(detectors.detect_freezes(track(diff))) == 1

    def test_silent_clip_gentle_restart_is_not_a_freeze(self):
        # Changed with the owner's approval (2026-11-23): a fully silent clip no longer rules freezes
        # out entirely, but a hold that ends gently is still a natural pause.
        diff, cells = held(resume=0.6)
        assert detectors.detect_freezes(track(diff, cell_max=cells, rms_db=np.full(250, -90.0))) == []

    def test_silent_clip_catch_up_is_a_freeze(self):
        diff, cells = held()
        out = detectors.detect_freezes(track(diff, cell_max=cells, rms_db=np.full(250, -90.0)))
        assert len(out) == 1 and out[0].score == 0.9 and "without sound" in out[0].note

    def test_silent_clip_hold_then_cut_is_reported_low(self):
        # A catch-up far into a fast shot looks like a new scene; reported low for the agent to check.
        diff, cells = held(resume=40.0)
        frames = np.zeros((150, 4, 4), np.uint8)
        frames[90:] = 200  # the picture after the hold looks like another shot
        t = dataclasses.replace(track(diff, cell_max=cells, rms_db=np.full(250, -90.0)), frames=frames)
        out = detectors.detect_freezes(t)
        assert [c.score for c in out] == [0.55] and "scene cut" in out[0].note

    def test_silent_clip_still_shot_then_cut_is_not_a_freeze(self):
        diff, cells = held(resume=40.0)
        diff[1:60] = 0.02  # the held shot never moved before the hold
        out = detectors.detect_freezes(track(diff, cell_max=cells, rms_db=np.full(250, -90.0)))
        assert out == []

    def test_quiet_sound_still_counts(self):
        diff, cells = held()
        assert len(detectors.detect_freezes(track(diff, cell_max=cells, rms_db=np.full(250, -46.0)))) == 1

    def test_static_scene_is_not_a_freeze(self):
        diff, cells = held(around=0.02)
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_short_hold_ignored(self):
        diff, cells = held(b=70)  # 10 frames = 0.33 s
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_hold_until_the_end_is_not_reported(self):
        diff = np.full(150, 0.5, np.float32)
        diff[100:] = 0.0
        assert detectors.detect_freezes(track(diff)) == []

    def test_tiny_track(self):
        assert detectors.detect_freezes(track([0.0, 0.0])) == []


class TestCellHolds:
    """Holds seen at mid resolution: a talking head whose 64 px average barely moves, but whose cells do."""

    @staticmethod
    def head(resume=3.0, hold_cell=0.01, around_cell=1.5, around_diff=0.1, n=150, a=60, b=90):
        diff = np.full(n, around_diff, np.float32)
        diff[0] = 0
        diff[a:b] = 0.0
        diff[b] = resume
        cells = np.full(n, around_cell, np.float32)
        cells[a:b] = hold_cell
        return diff, cells

    def test_low_motion_talking_head_catch_up_is_found(self):
        diff, cells = self.head()
        out = detectors.detect_freezes(track(diff, cell_max=cells))
        assert len(out) == 1 and out[0].score == 0.9
        assert out[0].t0 == pytest.approx(2.0) and out[0].t1 == pytest.approx(3.0)

    def test_exact_long_hold_with_gentle_restart_is_reported_lower(self):
        diff, cells = self.head(resume=0.1, b=100)  # 1.33 s, nothing jumps when it resumes
        out = detectors.detect_freezes(track(diff, cell_max=cells))
        assert [c.score for c in out] == [0.6] and "every cell" in out[0].note

    def test_short_exact_hold_with_gentle_restart_is_a_pause(self):
        diff, cells = self.head(resume=0.1, b=80)  # 0.67 s
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_moving_cells_inside_the_hold_are_not_a_freeze(self):
        diff, cells = self.head(hold_cell=0.2)  # lips still move at mid resolution
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_still_slide_is_not_a_freeze(self):
        diff, cells = self.head(around_cell=0.3)  # nothing moves around the hold either
        assert detectors.detect_freezes(track(diff, cell_max=cells)) == []

    def test_needs_speech_during_the_hold(self):
        diff, cells = self.head()
        rms = np.full(250, -20.0)
        rms[100:150] = -90.0  # the hold (2.0-3.0 s, 20 ms hops) is a quiet gap in an otherwise loud clip
        assert detectors.detect_freezes(track(diff, cell_max=cells, rms_db=rms)) == []

    def test_silent_clip_has_no_cell_holds(self):
        diff, cells = self.head()
        assert detectors.detect_freezes(track(diff, cell_max=cells, rms_db=np.full(250, -90.0))) == []

    def test_not_reported_twice(self):
        diff, cells = held()  # the classic case: both passes see it
        out = detectors.detect_freezes(track(diff, cell_max=cells))
        assert len(out) == 1


# ------------------------------------------------------------------ jump cut


def jump(spike=3.0, seam=100.0, peak=5.0, n=150, i=75):
    k = np.arange(n)
    diff = (0.5 + 0.05 * np.sin(k)).astype(np.float32)
    diff[0] = 0
    diff[i] = spike
    ratio = np.full(n, 2.0, np.float32)
    peaks = np.full(n, 0.3, np.float32)
    ratio[i], peaks[i] = seam, peak
    return diff, ratio, peaks


class TestJumpCut:
    def test_picture_jump_with_audio_click(self):
        diff, sr, sp = jump()
        out = detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp))
        assert len(out) == 1
        c = out[0]
        assert c.kind == "jump_cut" and c.t0 == c.t1 == pytest.approx(2.5)
        assert 0.6 <= c.score <= 0.95 and "clicks" in c.note

    def test_stronger_evidence_scores_higher(self):
        d, r, p = jump(spike=1.4, seam=60.0, peak=2.0)
        weak = detectors.detect_jump_cuts(track(d, seam_ratio=r, seam_peak=p))
        d, r, p = jump(spike=6.0, seam=1000.0, peak=50.0)
        strong = detectors.detect_jump_cuts(track(d, seam_ratio=r, seam_peak=p))
        assert len(weak) == 1 and len(strong) == 1 and strong[0].score > weak[0].score

    def test_picture_jump_without_click(self):
        diff, sr, sp = jump(seam=2.0, peak=0.3)
        assert detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp)) == []

    def test_click_without_picture_jump(self):
        diff, sr, sp = jump(spike=0.5)
        assert detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp)) == []

    def test_weak_jump_needs_strong_click(self):
        diff, sr, sp = jump(spike=1.2, seam=20.0, peak=0.6)  # each above its floor, jointly too weak
        assert detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp)) == []

    def test_scene_cut_is_not_a_jump_cut(self):
        diff, sr, sp = jump(spike=40.0, peak=1.0)  # some transient on the boundary, not the sharpest around
        assert detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp)) == []

    def test_cut_sized_jump_with_dominant_click(self):
        diff, sr, sp = jump(spike=40.0, peak=5.0)  # removed piece spanned a picture change; the splice still clicks
        assert len(detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp))) == 1

    def test_sustained_change_is_not_a_jump(self):
        diff, sr, sp = jump()
        diff[74:77] = 3.0  # a fast move over three frames, not a single-frame jump
        assert detectors.detect_jump_cuts(track(diff, seam_ratio=sr, seam_peak=sp)) == []

    def test_needs_speech_on_both_sides(self):
        diff, sr, sp = jump()
        db = np.full(250, -20.0)
        db[110:140] = -90.0  # 2.2-2.8 s silent: the cut at 2.5 s sits in a pause
        assert detectors.detect_jump_cuts(track(diff, rms_db=db, seam_ratio=sr, seam_peak=sp)) == []

    def test_without_seam_tracks_nothing(self):
        diff, _, _ = jump()
        assert detectors.detect_jump_cuts(track(diff)) == []

    def test_tiny_track(self):
        assert detectors.detect_jump_cuts(track([0, 1, 2], seam_ratio=[0, 0, 0], seam_peak=[0, 0, 0])) == []


# ------------------------------------------------------------------ end to end on encoded video


def moving_frames(n: int, w: int = 96, h: int = 128) -> np.ndarray:
    """A low-contrast square sliding 1 px per frame over a gradient, so every frame moves a little."""
    yy = np.linspace(60, 120, h, dtype=np.float32)[:, None, None]
    base = np.broadcast_to(yy, (h, w, 3)).astype(np.uint8)
    out = np.repeat(base[None], n, axis=0).copy()
    for k in range(n):
        x = 4 + (k % (w - 28))
        out[k, 50:70, x:x + 20] = (150, 150, 150)
    return out


@pytest.fixture(scope="module")
def edited_videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("edits")
    n, spf = 150, SR // int(FPS)
    src, audio = moving_frames(n), voice(n / FPS)
    cut = np.r_[0:60, 72:n]  # 0.4 s removed from picture and sound at 2.0 s
    jumped = media.write_video(src[cut], FPS, d / "jump.mp4", crf=16,
                               audio=np.concatenate([audio[: 60 * spf], audio[72 * spf:]]))
    idx = np.arange(n)
    idx[60:90] = 59  # frame 59 held for 1 s at 2.0 s, sound runs on
    frozen = media.write_video(src[idx], FPS, d / "freeze.mp4", crf=16, audio=audio)
    clean = media.write_video(src, FPS, d / "clean.mp4", crf=16, audio=audio)
    return jumped, frozen, clean


class TestEncoded:
    def test_compute_fills_rich_tracks(self, edited_videos):
        t = signals.compute(edited_videos[2])
        n = len(t.diff)
        assert t.cell_max is not None and t.cell_max.shape == (n,)
        assert t.seam_ratio is not None and t.seam_ratio.shape == (n,) and t.seam_peak.shape == (n,)

    def test_jump_cut_found(self, edited_videos):
        cands = signals.find_candidates(signals.compute(edited_videos[0]))
        jc = [c for c in cands if c.kind == "jump_cut"]
        assert len(jc) == 1 and jc[0].t0 == pytest.approx(2.0, abs=0.07)

    def test_freeze_found(self, edited_videos):
        cands = signals.find_candidates(signals.compute(edited_videos[1]))
        fr = [c for c in cands if c.kind == "freeze"]
        assert len(fr) == 1 and fr[0].score == 0.9
        assert fr[0].t0 == pytest.approx(2.0, abs=0.1) and fr[0].t1 == pytest.approx(3.0, abs=0.1)

    def test_clean_video_has_neither(self, edited_videos):
        cands = signals.find_candidates(signals.compute(edited_videos[2]))
        assert not [c for c in cands if c.kind in ("jump_cut", "freeze")]


class TestSilentAndCadenceFreezes:
    """Freezes in silent clips and in repeated-frame (low-fps upsampled) sources."""

    def test_has_dup_cadence_detects_alternating_duplicates(self):
        import numpy as np
        from vwatch import detectors as D

        alt = np.array([0.0] + [5.0, 0.0] * 50)
        smooth = np.array([0.0] + [5.0] * 100)
        assert D.has_dup_cadence(alt, D.STILL_DIFF, D.MOTION_MIN)
        assert not D.has_dup_cadence(smooth, D.STILL_DIFF, D.MOTION_MIN)

    def test_cadence_motion_ignores_duplicates(self):
        import numpy as np
        from vwatch import detectors as D

        alt = np.array([6.0, 0.0] * 8)
        assert D.plain_motion(alt) < 3.5 and D.cadence_motion(alt) == 6.0
        assert D.cadence_motion(np.array([])) == 0.0 and D.cadence_motion(np.array([2.0])) == 2.0

    def test_same_shot(self):
        import numpy as np
        from vwatch import detectors as D

        rng = np.random.default_rng(0)
        a = rng.integers(0, 120, (32, 32), dtype=np.uint8)
        moved = np.roll(a, 3, axis=1)
        other = rng.integers(140, 255, (32, 32), dtype=np.uint8)
        assert D.same_shot(a, moved) and not D.same_shot(a, other)
        flat = np.full((8, 8), 10, dtype=np.uint8)
        assert D.same_shot(flat, flat.copy())
