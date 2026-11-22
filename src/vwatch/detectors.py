"""Full-pass defect detectors for frozen frames and jump cuts (model-free).

`signals.py` keeps the cheap tracks and the simple detectors; this module adds the evidence those
were missing and the two detectors that use it:

- a mid-resolution motion track (largest per-cell change on an 8x8 grid), so "nothing moves" is
  checked where small things such as lips would show, not just in a 64 px average;
- an audio seam track: how sharp a high-frequency transient sits exactly on each video frame
  boundary. Cutting audio and video together at a frame boundary without a crossfade leaves a click
  there; natural speech transients are not tied to frame boundaries.

All thresholds below were tuned on the dev split only (t1x videos outside eval/bench/t1x_ids.txt);
see eval/bench/eval_signals.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vwatch.signals import Candidate, SignalTrack

MID_WIDTH = 128  # px, mid-resolution gray decode used for freeze confirmation
GRID = 8  # cells per side for the mid-resolution motion track
SEAM_SR = 44100  # the click of a hard splice lives above the 16 kHz track's band

# freeze
STILL_DIFF = 0.12  # 64 px mean abs diff below this = "no change" (re-encoding noise included)
HOLD_CELL = 0.1  # mid-res max-cell diff below this = this frame repeats the previous one (re-encode noise)
HOLD_SHARE = 0.8  # share of a hold's frames that must repeat exactly at mid resolution
SETTLE = 3  # frames the encoder needs to settle on a held picture
MOTION_MIN = 0.15  # median frame diff around a hold that counts as "the scene was moving"
CATCH_UP = 4.0  # resume jump vs the surrounding motion that marks a catch-up after a hold
SOUND_MIN = 0.3  # share of the hold with sound on
SILENT_CLIP_SHARE = 0.1  # below this share of sound over the whole clip, the clip counts as silent

# jump cut
JUMP_SPIKE = 2.2  # frame diff vs its +-10 frame median
JUMP_ISOLATED = 2.5  # neighbours of the spike must stay below this ratio (single-frame jump)
SEAM_RATIO = 15.0  # boundary transient vs the median over +-80 ms
SEAM_PEAK = 0.5  # boundary transient vs the strongest transient elsewhere within +-40 ms
SPEECH_SIDE = 0.5  # share of speech in the 0.3 s on each side of the cut
CUT_SEAM_PEAK = 2.0  # a cut-sized picture jump counts only if its click is the sharpest around
JUMP_EVIDENCE = 1.3  # joint gate: log10(spike) + log10(seam ratio)/2 + log10(min(seam peak, 4))/2


# ------------------------------------------------------------------ tracks


def cell_motion(gray: np.ndarray, grid: int = GRID) -> np.ndarray:
    """out[i] = largest mean abs change of any grid cell between frame i-1 and i; out[0] = 0."""
    n = len(gray)
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out
    h, w = gray.shape[1:3]
    ch, cw = max(1, h // grid), max(1, w // grid)
    gy, gx = min(grid, h), min(grid, w)
    for i in range(1, n, 256):  # chunked: bounded memory on long videos
        j = min(n, i + 256)
        d = np.abs(gray[i:j, : ch * gy, : cw * gx].astype(np.int16) - gray[i - 1:j - 1, : ch * gy, : cw * gx])
        cells = d.reshape(j - i, gy, ch, gx, cw).mean(axis=(2, 4), dtype=np.float32)
        out[i:j] = cells.reshape(j - i, -1).max(axis=1)
    return out


def _window_energy(x: np.ndarray, w: int) -> np.ndarray:
    """e[j] = mean of x[j:j+w] (len(x) - w + 1 values)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[w:] - c[:-w]) / w


def audio_seams(audio: np.ndarray, sr: int, fps: float, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per video frame boundary: (ratio, peak) of the high-frequency transient sitting on it.

    ratio: strongest 1 ms high-pass energy within [-2, +1] ms of the boundary / its median over
    +-80 ms. peak: the same energy / the strongest one elsewhere within +-40 ms (>1 = the sharpest
    transient around). Both are 0 where there is no audio.
    """
    ratio = np.zeros(n_frames, dtype=np.float32)
    peak = np.zeros(n_frames, dtype=np.float32)
    w = max(1, int(sr * 0.001))
    if audio.size < 4 * w or fps <= 0:
        return ratio, peak
    hp = np.diff(audio.astype(np.float64), 2, prepend=[0.0, 0.0])  # 2nd difference: a crude high-pass
    e = _window_energy(hp * hp, w)
    wide, near = int(0.08 * sr), int(0.04 * sr)
    eps = 1e-12
    for k in range(1, n_frames):
        i = int(round(k / fps * sr))
        if i >= len(e):
            break
        pk = float(e[max(0, i - 2 * w):i + w].max())
        if pk <= eps:
            continue
        med = float(np.median(e[max(0, i - wide):i + wide:4]))
        around = np.concatenate([e[max(0, i - near):max(0, i - 4 * w)], e[i + 3 * w:i + near]])
        ratio[k] = pk / (med + eps)
        peak[k] = pk / (float(around.max()) + eps) if around.size else 0.0
    return ratio, peak


def speech_mask(rms_db: np.ndarray, speech_db: float, silence_db: float) -> np.ndarray:
    """Loudness hops with sound on, relative to the clip's own level (quiet recordings count too)."""
    if rms_db.size == 0:
        return np.zeros(0, dtype=bool)
    thr = max(silence_db, min(speech_db, float(np.percentile(rms_db, 90)) - 14.0))
    return rms_db > thr


CADENCE_SHARE = 0.15  # share of moving-duplicate-moving frame triples that marks a repeated-frame cadence


def has_dup_cadence(diff: np.ndarray, still: float, moving: float) -> bool:
    """True for low-frame-rate sources stretched to a higher rate (every Nth frame an exact repeat)."""
    if diff.size < 4:
        return False
    d = diff[1:]
    triples = (d[1:-1] < still) & (d[:-2] > moving) & (d[2:] > moving)
    return float(triples.mean()) > CADENCE_SHARE


def plain_motion(diff: np.ndarray) -> float:
    return float(np.median(diff)) if diff.size else 0.0


def cadence_motion(diff: np.ndarray, window: int = 3) -> float:
    """Typical motion level, robust to repeated frames (15 fps stretched to 30 fps, 3:2 pulldown).

    Duplicate-frame cadences put zeros between real motion, which drags a plain median to ~0 and makes
    a moving scene look still; taking the max over a small sliding window first removes that.
    """
    if diff.size == 0:
        return 0.0
    if diff.size < window:
        return float(diff.max())
    peaks = np.lib.stride_tricks.sliding_window_view(diff, window).max(axis=1)
    return float(np.median(peaks))


SAME_SHOT_CORR = 0.9  # brightness-histogram correlation above which two frames are the same shot


def same_shot(f1: np.ndarray, f2: np.ndarray, bins: int = 32) -> bool:
    """True when two small gray frames look like the same shot (a catch-up, not a scene cut)."""
    h1 = np.histogram(f1, bins=bins, range=(0, 255))[0].astype(np.float64)
    h2 = np.histogram(f2, bins=bins, range=(0, 255))[0].astype(np.float64)
    if h1.std() == 0 or h2.std() == 0:
        return bool(np.array_equal(h1, h2))
    return float(np.corrcoef(h1, h2)[0, 1]) >= SAME_SHOT_CORR


def _share(mask: np.ndarray, hop_s: float, t0: float, t1: float) -> float:
    a, b = max(0, int(t0 / hop_s)), min(len(mask), int(np.ceil(t1 / hop_s)))
    return float(mask[a:b].mean()) if b > a else 0.0


# ------------------------------------------------------------------ freeze


def detect_freezes(track: SignalTrack, min_s: float = 0.4) -> list[Candidate]:
    """The picture holds (near) exactly while sound goes on, inside a moving scene.

    A hold is a run of near-zero frame difference. With the mid-resolution track it must also be
    still cell by cell (an exact repeat, not a calm shot with moving lips). It needs sound during
    the hold and motion around it. How it ends decides: a catch-up jump (the picture leaps to where
    the motion got to) is the freeze signature (score 0.9); a scene cut right after is ambiguous,
    since an editor's held frame looks the same (0.55); a gentle restart is a natural pause (dropped).
    """
    from vwatch.signals import HOP_S, SILENCE_DB, SPEECH_DB, Candidate, _runs, _spike_threshold

    diff = track.diff
    n = len(diff)
    if n < 3:
        return []
    still = diff < STILL_DIFF
    still[0] = False
    sound = speech_mask(track.rms_db, SPEECH_DB, SILENCE_DB)
    # A silent clip (music-less montage, no audio track) can't use "sound goes on" as context, so the
    # sound rule is dropped there and only the strongest evidence counts: a catch-up jump.
    silent_clip = sound.size == 0 or float(sound.mean()) < SILENT_CLIP_SHARE
    # Repeated-frame sources need a motion estimate that ignores the duplicates; others keep the median.
    motion = cadence_motion if has_dup_cadence(diff, STILL_DIFF, MOTION_MIN) else plain_motion
    cut_thr = _spike_threshold(diff)
    out: list[Candidate] = []
    for a, b in _runs(still):
        dur = (b - a) / track.fps
        if dur < min_s:
            continue
        if track.cell_max is not None and len(track.cell_max) == n:
            core = track.cell_max[min(a + SETTLE, b - 1):b]
            if float(np.mean(core < HOLD_CELL)) < HOLD_SHARE:
                continue  # something still moves at mid resolution
        t0, t1 = a / track.fps, b / track.fps
        if not silent_clip and _share(sound, HOP_S, t0, t1) < SOUND_MIN:
            continue
        pre, post = diff[max(1, a - 15):a], diff[b + 1:b + 16]
        m_pre = motion(pre)
        m_post = motion(post)
        if max(m_pre, m_post) < MOTION_MIN:
            continue  # nothing moves around it either: a still shot or slide
        resume = float(diff[b]) if b < n else 0.0
        base = max(0.05, m_post, 0.5 * m_pre)
        held = (f"picture held still for {dur:.2f}s in a clip without sound" if silent_clip
                else f"picture held still for {dur:.2f}s while sound continues")
        after = min(n - 1, b + 1)
        same = track.frames is not None and len(track.frames) == n and same_shot(track.frames[a], track.frames[after])
        if silent_clip and resume >= cut_thr and same and m_pre >= MOTION_MIN:
            # Silent clip: a cut-sized jump that stays in the same shot is the picture catching up after
            # a freeze. (With sound the existing rules apply unchanged; widening them there added false
            # alarms on the dev set.)
            out.append(Candidate(t0, t1, "freeze", 0.9, held + f", then jumps ahead {resume / base:.0f}x the "
                                                              "normal motion within the same shot (catch-up)"))
        elif resume >= cut_thr:
            if m_pre < MOTION_MIN or silent_clip:  # silent: an editor's hold before a cut is too likely
                continue  # the held shot itself never moved: a still shot, then a cut
            out.append(Candidate(t0, t1, "freeze", 0.55, held + ", then a scene cut (held frame before an edit?)"))
        elif resume >= CATCH_UP * base and m_post >= 0.05:
            out.append(Candidate(t0, t1, "freeze", 0.9, held + f", then jumps ahead {resume / base:.0f}x the "
                                                              "normal motion (catch-up after a frozen frame)"))
        # else: motion restarts gently (or the video ends): a natural pause, not reported
    return out


# ------------------------------------------------------------------ jump cut


def _local_ratio(diff: np.ndarray, k: int = 10) -> np.ndarray:
    """diff[i] / median of diff over +-k frames (excluding i and frame 0)."""
    n = len(diff)
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        w = np.concatenate([diff[max(1, i - k):i], diff[i + 1:i + k + 1]])
        if w.size:
            out[i] = diff[i] / max(0.05, float(np.median(w)))
    return out


def detect_jump_cuts(track: SignalTrack) -> list[Candidate]:
    """A small picture jump and an audio splice on the same frame boundary, inside speech.

    Visual: one frame differs from its predecessor several times more than the frames around it
    (but less than a scene cut). Audio: a sharp high-frequency transient sits exactly on that frame
    boundary, the click a hard audio splice leaves. Speech must be on at both sides: a natural edit
    tends to sit in a pause.
    """
    from vwatch.signals import HOP_S, SILENCE_DB, SPEECH_DB, Candidate, _spike_threshold

    if track.seam_ratio is None or track.seam_peak is None:
        return []
    diff = track.diff
    n = min(len(diff), len(track.seam_ratio), len(track.seam_peak))
    if n < 5:
        return []
    ratio = _local_ratio(diff[:n])
    nb = np.maximum(np.concatenate([[0.0], ratio[:-1]]), np.concatenate([ratio[1:], [0.0]]))
    cut_thr = _spike_threshold(diff)
    sr, sp = track.seam_ratio[:n], track.seam_peak[:n]
    small = (diff[:n] < cut_thr) | (sp >= CUT_SEAM_PEAK)
    hit = (ratio >= JUMP_SPIKE) & (nb < JUMP_ISOLATED) & small & (sr >= SEAM_RATIO) & (sp >= SEAM_PEAK)
    hit[:2] = False
    sound = speech_mask(track.rms_db, SPEECH_DB, SILENCE_DB)
    out: list[Candidate] = []
    for i in np.flatnonzero(hit):
        t = i / track.fps
        if min(_share(sound, HOP_S, t - 0.3, t), _share(sound, HOP_S, t, t + 0.3)) < SPEECH_SIDE:
            continue
        # a weak picture jump needs a stronger click, and the other way round
        evidence = float(np.log10(ratio[i]) + np.log10(sr[i]) / 2 + np.log10(min(float(sp[i]), 4.0)) / 2)
        if evidence < JUMP_EVIDENCE:
            continue
        score = float(np.clip(0.6 + 0.3 * (evidence - JUMP_EVIDENCE), 0.6, 0.95))
        out.append(Candidate(t, t, "jump_cut", round(score, 2),
                             f"picture jumps {ratio[i]:.1f}x the nearby motion and the audio clicks on the "
                             "same frame mid-speech (piece cut out?)"))
    return out
