"""Arms: baseline, watcher, tools and shared pieces. No whisper, no network: FakeBackend + scripted JSON."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from vwatch import transcribe
from vwatch.arms import ArmResult, VideoContext, baseline, common, run_arm, tools, watcher
from vwatch.imaging import Box, estimate_image_tokens
from vwatch.llm.base import LLMError, Reply
from vwatch.llm.fake import FakeBackend, content_text, count_images
from vwatch.schemas import Arm, Task, Usage
from vwatch.signals import Candidate

OK_DEFECTS = '{"defects": []}'
FINAL = '{"final": {"defects": [{"type": "flash_frame", "start": 1.0, "end": 1.1}]}}'
PLAN = json.dumps({"looking_for": ["flash"], "regions": [{"name": "cap", "box": [0, 0.8, 1, 1]}],
                   "dense_fps": 12, "overview_fps": 1})
PREDICT = json.dumps({"surprises": [{"t0": 1.0, "t1": 1.2, "score": 0.9, "why": "jump"}]})


def stage_of(content: list[dict]) -> str:
    txt = content_text(content)
    for marker, name in (("Your last reply was not valid JSON", "repair"),
                         ("Before you see a single frame", "plan"),
                         ("Now predict.", "predict"),
                         ("Evidence budget used up", "final"),
                         ("You can request more evidence", "act")):
        if marker in txt:
            return name
    return "other"


class Script:
    """Scripted responder: records the stage of every turn it answers."""

    def __init__(self, plan=PLAN, predict=PREDICT, acts=(), final=FINAL):
        self.plan, self.predict, self.acts, self.final = plan, predict, list(acts), final
        self.stages: list[str] = []
        self._act_i = 0

    def __call__(self, system, content, turn):
        st = stage_of(content)
        self.stages.append(st)
        if st == "plan":
            return self.plan
        if st == "predict":
            return self.predict
        if st == "act":
            i, self._act_i = self._act_i, self._act_i + 1
            return self.acts[i] if i < len(self.acts) else self.final
        return self.final


def run_w(ctx, script, budget=12000, arm=Arm.W, task=Task.DEFECTS):
    be = FakeBackend(script)
    res = run_arm(arm, ctx, task, "THE-TASK-PROMPT", budget, be)
    return res, be, be.sessions[0]


def all_text(session) -> str:
    return "\n".join(content_text(c) for c in session.log)


def n_images(session) -> int:
    return sum(count_images(c) for c in session.log)


def act(*actions):
    return json.dumps({"actions": list(actions)})


# ------------------------------------------------------------------ common


class TestCommon:
    def test_evidence_tokens_and_take(self):
        ev = common.Evidence()
        ev.text("x" * 30)
        ev.image(np.zeros((100, 100, 3), np.uint8), "frame", 1.23456, 2.0)
        assert ev.tokens == 10 + estimate_image_tokens(100, 100)
        blocks = ev.take()
        assert [b["type"] for b in blocks] == ["text", "image"] and ev.blocks == []
        assert ev.observations[0].t0 == 1.235 and ev.tokens > 0  # accumulates across takes
        ev.observe("audio", 0, 1)
        assert [o.kind for o in ev.observations] == ["frame", "audio"]

    def test_video_header(self, video_ctx, silent_ctx):
        h = common.video_header(video_ctx)
        assert "4.00 s" in h and "108x192" in h and "30 fps" in h and "Speech: yes" in h
        assert "no speech detected" in common.video_header(silent_ctx)

    def test_conversation_accumulates(self):
        class S:
            def send(self, c):
                return Reply("t", Usage(input_tokens=3, output_tokens=1), 0.5, 0.25)

        conv = common.Conversation(S())
        conv.send([])
        conv.send([])
        assert (conv.usage.input_tokens, conv.cost, conv.wall, conv.turns, conv.last_text) == (6, 1.0, 0.5, 2, "t")

    def test_finish_repair_turn(self):
        be = FakeBackend(lambda s, c, t: '{"defects": []}')
        conv = common.Conversation(be.open("s"))
        ev = common.Evidence()
        res = common.finish(conv, Task.DEFECTS, ev, 0.0, "garbage")
        assert res.parsed == {"defects": []} and res.turns == 1
        assert "not valid JSON" in content_text(be.sessions[0].log[0])

    def test_failed_without_conversation(self):
        res = common.failed(None, common.Evidence(), 0.0, ValueError("boom"))
        assert res.error == "ValueError: boom" and res.turns == 0 and res.usage == Usage()
        assert res.parsed is None and res.raw_output == ""

    def test_load_uses_transcribe_only_with_audio(self, tone_video, silent_video, monkeypatch):
        calls = []

        def fake_tr(path, language=None):
            calls.append((str(path), language))
            return transcribe.Transcript("en", (), ())

        monkeypatch.setattr(transcribe, "transcribe", fake_tr)
        ctx = VideoContext.load(tone_video, language="hi")
        assert calls == [(str(tone_video), "hi")] and ctx.info.has_audio
        assert len(ctx.track.diff) == 120
        ctx2 = VideoContext.load(silent_video)
        assert len(calls) == 1 and ctx2.transcript is transcribe.EMPTY


# ------------------------------------------------------------------ tools


class TestTools:
    def test_seek_times_single(self):
        assert tools.seek_times(1.0, 1.0, 8, 4.0) == [1.0]
        assert tools.seek_times(-5, -4, 8, 4.0) == [0.0]

    def test_seek_times_cap_and_clamp(self):
        ts = tools.seek_times(0, 4, 12, 4.0)
        assert len(ts) == tools.MAX_SEEK_FRAMES and ts[0] == 0 and ts[-1] <= 4.0 - 1e-3 + 1e-9

    def test_seek_times_fps_clamped(self):
        assert len(tools.seek_times(0, 0.5, 1000, 4.0)) == int(0.5 * 30) + 1
        assert len(tools.seek_times(0, 2.0, 0.0, 4.0)) == int(2.0 * 0.5) + 1

    def test_grid_tokens(self, video_ctx):
        assert tools.grid_tokens(video_ctx, 0, 100) == 0
        one = tools.grid_tokens(video_ctx, 1, 100)
        assert one == estimate_image_tokens(100, 178)
        assert tools.grid_tokens(video_ctx, 12, 100) > tools.grid_tokens(video_ctx, 4, 100) > one

    @pytest.mark.parametrize("a,b,fps", [(0, 1, 8), (0, 4, 12), (1, 1.1, 8), (0, 2, 2)])
    def test_seek_cost_tracks_actual(self, video_ctx, a, b, fps):
        ev = common.Evidence()
        spent = tools.seek(video_ctx, ev, a, b, fps, "why")
        assert spent == ev.tokens
        assert abs(spent - tools.seek_cost(video_ctx, a, b, fps)) <= 30
        assert "(why)" in ev.blocks[0]["text"]
        assert all(o.kind == "frame" for o in ev.observations)
        assert sum(1 for b_ in ev.blocks if b_["type"] == "image") == -(-len(ev.observations) // 12)

    @pytest.mark.parametrize("fps", [0.25, 1, 2])
    def test_overview_cost_is_upper_bound(self, video_ctx, fps):
        ev = common.Evidence()
        spent = tools.overview(video_ctx, ev, fps)
        assert 0 < spent <= tools.overview_cost(video_ctx, fps)
        assert "OVERVIEW" in ev.blocks[0]["text"]

    def test_crops(self, video_ctx):
        ev = common.Evidence()
        box = Box(0.05, 0.75, 0.95, 0.95)
        spent = tools.crops(video_ctx, ev, [1.0, 1.0, 2.0], box, "cap")
        assert spent == ev.tokens and abs(spent - 2 * tools.crop_cost(video_ctx, box)) <= 40
        assert [o.kind for o in ev.observations] == ["crop", "crop"]  # duplicate time deduped
        assert "CROP cap at 00:01.00" in ev.blocks[0]["text"]

    def test_crop_default_label(self, video_ctx):
        ev = common.Evidence()
        tools.crops(video_ctx, ev, [1.0], Box(0, 0, 0.5, 0.5))
        assert "box 0.00,0.00,0.50,0.50" in ev.blocks[0]["text"]

    def test_crop_cost_capped_width(self, video_ctx):
        assert tools.crop_cost(video_ctx, Box(0, 0, 1, 1)) <= estimate_image_tokens(480, 854) + 15

    def test_loudness_line(self, video_ctx, silent_ctx):
        line = tools.loudness_line(video_ctx, 0.5, 1.0)
        assert len(line.split()) == 6 and "sil" not in line
        assert tools.loudness_line(silent_ctx, 0, 1) == "(no audio)"

    def test_loudness_line_silence(self, video_ctx):
        quiet = dataclasses.replace(video_ctx.track, rms_db=np.full(200, -90.0, dtype=np.float32))
        ctx = dataclasses.replace(video_ctx, track=quiet)
        assert set(tools.loudness_line(ctx, 0, 1).split()) == {"sil"}

    def test_listen(self, video_ctx):
        ev = common.Evidence()
        spent = tools.listen(video_ctx, ev, 0.0, 1.0)
        txt = ev.blocks[0]["text"]
        assert "LISTEN 00:00.00–00:01.00" in txt and "hello@0.20" in txt
        assert spent == tools.listen_cost(video_ctx, 0.0, 1.0) == ev.tokens
        assert (ev.observations[0].kind, ev.observations[0].t1) == ("audio", 1.0)


# ------------------------------------------------------------------ baseline


class TestBaseline:
    def test_uniform_times(self):
        assert baseline.uniform_times(4.0, 4) == [0.5, 1.5, 2.5, 3.5]

    def test_frame_budget_math(self, video_ctx):
        per = estimate_image_tokens(360, 640) + 6
        assert baseline.frame_budget(video_ctx, per * 10 + 100, 100) == 10
        assert baseline.frame_budget(video_ctx, per * 10 + 99, 100) == 9
        assert baseline.frame_budget(video_ctx, 100, 90) == baseline.MIN_FRAMES
        assert baseline.frame_budget(video_ctx, 10**9, 0) == video_ctx.info.nb_frames

    def test_b0_has_no_transcript_b1_has(self, video_ctx):
        be0, be1 = FakeBackend(lambda *a: OK_DEFECTS), FakeBackend(lambda *a: OK_DEFECTS)
        r0 = run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", 6000, be0)
        r1 = run_arm(Arm.B1, video_ctx, Task.DEFECTS, "P", 6000, be1)
        t0, t1 = all_text(be0.sessions[0]), all_text(be1.sessions[0])
        assert "TRANSCRIPT" not in t0 and "hello there world" not in t0
        assert "TRANSCRIPT" in t1 and "hello there world" in t1
        assert not [o for o in r0.observations if o.kind == "transcript"]
        assert len([o for o in r1.observations if o.kind == "transcript"]) == 2
        # B0 has more room for frames than B1 at the same budget
        assert n_images(be0.sessions[0]) >= n_images(be1.sessions[0])

    def test_b2_runs_like_b1(self, video_ctx):
        be = FakeBackend(lambda *a: OK_DEFECTS)
        run_arm(Arm.B2, video_ctx, Task.DEFECTS, "P", 6000, be)
        assert "TRANSCRIPT" in all_text(be.sessions[0])

    @pytest.mark.parametrize("budget", [3000, 6000, 12000])
    def test_frame_count_matches_budget(self, video_ctx, budget):
        be = FakeBackend(lambda *a: OK_DEFECTS)
        res = run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", budget, be)
        n = count_images(be.sessions[0].log[0])
        per = estimate_image_tokens(360, 640) + 6
        assert n >= baseline.MIN_FRAMES and abs(n * per - budget) < 2 * per + 400
        assert res.content_tokens_est <= budget
        assert res.turns == 1 and res.parsed == {"defects": []} and res.error is None
        assert len([o for o in res.observations if o.kind == "frame"]) == n

    def test_prompt_and_timestamps_present(self, video_ctx):
        be = FakeBackend(lambda *a: OK_DEFECTS)
        run_arm(Arm.B0, video_ctx, Task.DEFECTS, "UNIQUE-PROMPT", 3000, be)
        blocks = be.sessions[0].log[0]
        assert content_text(blocks).rstrip().endswith("UNIQUE-PROMPT")
        assert "t=00:00.23" in content_text(blocks)

    def test_every_frame_served_when_budget_allows(self, video_ctx):
        be = FakeBackend(lambda *a: OK_DEFECTS)
        run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", 10**6, be)
        assert count_images(be.sessions[0].log[0]) == video_ctx.info.nb_frames

    def test_repair_turn_on_bad_json(self, video_ctx):
        replies = iter(["I could not decide", OK_DEFECTS])
        be = FakeBackend(lambda *a: next(replies))
        res = run_arm(Arm.B1, video_ctx, Task.DEFECTS, "P", 4000, be)
        assert res.turns == 2 and res.parsed == {"defects": []} and res.parse_error is None
        assert "not valid JSON" in content_text(be.sessions[0].log[1])
        assert count_images(be.sessions[0].log[1]) == 0
        assert res.usage.input_tokens > 0 and res.cost_usd == pytest.approx(0.002)

    def test_unrecoverable_bad_json(self, video_ctx):
        be = FakeBackend(lambda *a: "still nonsense")
        res = run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", 4000, be)
        assert res.turns == 2 and res.parsed is None and "no JSON" in res.parse_error and res.error is None

    def test_schema_mismatch_goes_to_repair(self, video_ctx):
        replies = iter(['{"defects": [{"type": "x"}]}', OK_DEFECTS])
        res = run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", 4000, FakeBackend(lambda *a: next(replies)))
        assert res.turns == 2 and res.parsed is not None

    def test_llm_error_returns_arm_result(self, video_ctx):
        def boom(*a):
            raise LLMError("rate limited")

        res = run_arm(Arm.B1, video_ctx, Task.DEFECTS, "P", 4000, FakeBackend(boom))
        assert isinstance(res, ArmResult)
        assert res.error == "LLMError: rate limited" and res.parsed is None
        assert res.content_tokens_est > 0 and len(res.observations) > 0

    def test_media_error_returns_arm_result(self, video_ctx, tmp_path):
        ctx = dataclasses.replace(video_ctx, path=tmp_path / "gone.mp4")
        res = run_arm(Arm.B0, ctx, Task.DEFECTS, "P", 4000, FakeBackend(lambda *a: OK_DEFECTS))
        assert res.error.startswith("MediaError") and res.turns == 0

    def test_session_closed_even_on_error(self, video_ctx):
        closed = []

        class Sess:
            def send(self, c):
                raise LLMError("x")

            def close(self):
                closed.append(1)

        class Be:
            model = "m"

            def open(self, system):
                return Sess()

        res = run_arm(Arm.B0, video_ctx, Task.DEFECTS, "P", 4000, Be())
        assert res.error and closed == [1]

    def test_unknown_arm(self, video_ctx):
        with pytest.raises(ValueError, match="unknown arm"):
            run_arm("nope", video_ctx, Task.DEFECTS, "P", 1000, FakeBackend(lambda *a: ""))


# ------------------------------------------------------------------ watcher: stage order


class TestWatcherStages:
    def test_full_order(self, video_ctx):
        script = Script(acts=[act({"type": "listen", "t0": 1, "t1": 2})] * 5)
        res, be, sess = run_w(video_ctx, script)
        assert script.stages == ["plan", "predict", "act", "act", "act", "final"]
        assert res.turns == 6 and res.error is None
        assert res.parsed["defects"][0]["type"] == "flash_frame"
        # plan turn carries no images; predict turn carries the overview
        assert count_images(sess.log[0]) == 0 and count_images(sess.log[1]) >= 1
        assert "TRANSCRIPT" in content_text(sess.log[0]) and "THE-TASK-PROMPT" in content_text(sess.log[0])

    def test_noq_skips_plan(self, video_ctx):
        script = Script()
        res, _, sess = run_w(video_ctx, script, arm=Arm.W_NOQ)
        assert "plan" not in script.stages and script.stages[0] == "predict"
        assert "Before you see a single frame" not in all_text(sess)
        assert res.error is None and res.parsed is not None

    def test_nop_skips_predict(self, video_ctx):
        script = Script()
        res, _, sess = run_w(video_ctx, script, arm=Arm.W_NOP)
        assert script.stages[0] == "plan" and "predict" not in script.stages
        assert "Now predict" not in all_text(sess)
        assert res.parsed is not None

    def test_nocrop_sends_no_crop_images(self, video_ctx):
        script = Script(acts=[act({"type": "crop", "t": 1.0, "box": [0, 0, 1, 0.5]})])
        res, _, sess = run_w(video_ctx, script, arm=Arm.W_NOCROP)
        assert "CROP" not in all_text(sess)
        assert not [o for o in res.observations if o.kind == "crop"]
        assert "crop unavailable in this configuration" in all_text(sess)

    def test_w_serves_crops_from_plan_regions(self, video_ctx):
        res, _, sess = run_w(video_ctx, Script(), arm=Arm.W)
        assert "CROP cap at" in all_text(sess)
        assert [o for o in res.observations if o.kind == "crop"]

    def test_bad_plan_and_predict_fall_back(self, video_ctx):
        script = Script(plan="no idea", predict="[]")
        res, _, _ = run_w(video_ctx, script)
        assert script.stages[:2] == ["plan", "predict"] and res.error is None and res.parsed

    def test_plan_values_clamped(self):
        p = watcher.WatchPlan(dense_fps=500, overview_fps=0.001, signal_kinds=("bogus",))
        assert (p.dense_fps, p.overview_fps) == (12.0, 0.25) and p.signal_kinds == watcher.SIGNAL_KINDS
        assert watcher.WatchPlan(signal_kinds=("flash", "x")).signal_kinds == ("flash",)

    def test_plan_region_box_clamped(self):
        r = watcher.PlanRegion(box=(-1, 0.9, 2, 0.1))
        assert r.to_box() == Box(0, 0.1, 1, 0.9)

    def test_variants_table(self):
        v = watcher.VARIANTS
        assert (v[Arm.W].plan_first, v[Arm.W].predict, v[Arm.W].use_crops) == (True, True, True)
        assert not v[Arm.W_NOQ].plan_first and not v[Arm.W_NOP].predict and not v[Arm.W_NOCROP].use_crops

    def test_system_prompt_set(self, video_ctx):
        _, _, sess = run_w(video_ctx, Script())
        assert sess.system == watcher.SYSTEM

    def test_final_early_exit(self, video_ctx):
        script = Script(acts=[FINAL])
        res, _, _ = run_w(video_ctx, script)
        assert script.stages == ["plan", "predict", "act"] and res.turns == 3
        assert res.parsed is not None

    def test_bad_final_gets_repair(self, video_ctx):
        script = Script(acts=['{"final": {"defects": [{"oops": 1}]}}'])
        script.final = OK_DEFECTS
        res, _, _ = run_w(video_ctx, script)
        assert script.stages == ["plan", "predict", "act", "repair"] and res.parsed == {"defects": []}

    def test_no_actions_goes_to_final(self, video_ctx):
        script = Script(acts=[act()])
        run_w(video_ctx, script)
        assert script.stages == ["plan", "predict", "act", "final"]

    def test_action_rounds_capped(self, video_ctx):
        script = Script(acts=[act({"type": "listen", "t0": 0, "t1": 1})] * 10)
        res, _, _ = run_w(video_ctx, script, budget=40000)
        assert script.stages.count("act") == watcher.MAX_ACTION_ROUNDS and script.stages[-1] == "final"

    def test_low_budget_skips_act_rounds(self, video_ctx):
        script = Script()
        run_w(video_ctx, script, budget=600)
        assert "act" not in script.stages and script.stages[-1] == "final"

    def test_error_path(self, video_ctx):
        def boom(system, content, turn):
            if turn == 1:
                raise LLMError("quota")
            return PLAN

        res = run_arm(Arm.W, video_ctx, Task.DEFECTS, "P", 8000, FakeBackend(boom))
        assert res.error == "LLMError: quota" and res.parsed is None and res.turns == 1
        assert res.content_tokens_est > 0


# ------------------------------------------------------------------ watcher: actions & budget


class TestWatcherActions:
    def test_seek_crop_listen_executed(self, video_ctx):
        script = Script(acts=[act({"type": "seek", "t0": 2.0, "t1": 2.5, "fps": 8},
                                  {"type": "crop", "t": 3.0, "box": [0, 0, 1, 0.5]},
                                  {"type": "listen", "t0": 2.0, "t1": 3.0})])
        res, _, sess = run_w(video_ctx, script, budget=40000)
        txt = all_text(sess)
        assert "SEEK 00:02.00" in txt and "(requested)" in txt
        assert "CROP box 0.00,0.00,1.00,0.50 at 00:03.00" in txt
        assert "LISTEN 00:02.00–00:03.00" in txt
        kinds = {(o.kind) for o in res.observations}
        assert {"frame", "crop", "audio", "transcript", "signal"} <= kinds
        crop_obs = [o for o in res.observations if o.kind == "crop" and o.t0 == 3.0]
        assert crop_obs
        assert any(o.kind == "audio" and (o.t0, o.t1) == (2.0, 3.0) for o in res.observations)
        assert "NOTES" not in txt

    def test_seek_window_capped_at_4s(self, video_ctx):
        ctx = dataclasses.replace(video_ctx, info=dataclasses.replace(video_ctx.info, duration=60.0))
        ev = common.Evidence()
        spent, note = watcher._run_action(ctx, ev, watcher.Action(type="seek", t0=0, t1=30, fps=2),
                                          watcher.DEFAULT_PLAN, True, 10**6)
        assert note is None and spent > 0
        assert "00:03.99" in ev.blocks[0]["text"] or "00:04.0" in ev.blocks[0]["text"]

    def test_listen_window_capped_at_6s(self, video_ctx):
        ev = common.Evidence()
        watcher._run_action(video_ctx, ev, watcher.Action(type="listen", t0=0, t1=100),
                            watcher.DEFAULT_PLAN, True, 10**6)
        assert "00:00.00–00:06.00" in ev.blocks[0]["text"]

    def test_over_budget_actions_skipped_with_note(self, video_ctx):
        ev = common.Evidence()
        plan = watcher.DEFAULT_PLAN
        s, n1 = watcher._run_action(video_ctx, ev, watcher.Action(type="seek", t0=0, t1=1, fps=8), plan, True, 50)
        c, n2 = watcher._run_action(video_ctx, ev, watcher.Action(type="crop", t=1, box=(0, 0, 1, 1)), plan, True, 5)
        l, n3 = watcher._run_action(video_ctx, ev, watcher.Action(type="listen", t0=0, t1=5), plan, True, 1)
        assert (s, c, l) == (0, 0, 0) and ev.blocks == [] and ev.tokens == 0
        assert "skipped: needs" in n1 and "over budget" in n2 and "over budget" in n3

    def test_unrecognised_and_incomplete_actions(self, video_ctx):
        ev = common.Evidence()
        plan = watcher.DEFAULT_PLAN
        for a in (watcher.Action(type="zoom"), watcher.Action(type="seek"),
                  watcher.Action(type="crop", t=1), watcher.Action(type="listen")):
            spent, note = watcher._run_action(video_ctx, ev, a, plan, True, 10**6)
            assert spent == 0 and note.startswith("unrecognised action")

    def test_listen_on_silent_video(self, silent_ctx):
        ev = common.Evidence()
        spent, note = watcher._run_action(silent_ctx, ev, watcher.Action(type="listen", t0=0, t1=1),
                                          watcher.DEFAULT_PLAN, True, 10**6)
        assert spent == 0 and "no audio" in note

    def test_crop_with_t0_and_clamped_time(self, video_ctx):
        ev = common.Evidence()
        spent, note = watcher._run_action(video_ctx, ev, watcher.Action(type="crop", t0=99, box=(0, 0, 1, 1)),
                                          watcher.DEFAULT_PLAN, True, 10**6)
        assert note is None and spent > 0 and ev.observations[0].t0 < 4.0

    def test_action_type_case_insensitive(self, video_ctx):
        ev = common.Evidence()
        spent, note = watcher._run_action(video_ctx, ev, watcher.Action(type="LISTEN", t0=0, t1=1),
                                          watcher.DEFAULT_PLAN, True, 10**6)
        assert note is None and spent > 0

    def test_over_budget_note_reaches_model(self, video_ctx):
        big = {"type": "seek", "t0": 0, "t1": 4, "fps": 12}
        script = Script(acts=[act(big, {"type": "seek", "t0": 1, "t1": 2, "fps": 12})] * 3)
        _, _, sess = run_w(video_ctx, script, budget=2000)
        assert "skipped" in all_text(sess)

    def test_invalid_action_note(self, video_ctx):
        script = Script(acts=[act({"type": 5}, {"type": "listen", "t0": 0, "t1": 1})])
        _, _, sess = run_w(video_ctx, script, budget=40000)
        txt = all_text(sess)
        assert "NOTES: invalid action" in txt and "LISTEN 00:00.00" in txt

    def test_max_actions_per_round(self, video_ctx):
        many = [{"type": "listen", "t0": i * 0.1, "t1": i * 0.1 + 0.1} for i in range(10)]
        _, _, base = run_w(video_ctx, Script(acts=[act()]), budget=40000)
        _, _, sess = run_w(video_ctx, Script(acts=[act(*many)]), budget=40000)
        assert all_text(sess).count("LISTEN ") - all_text(base).count("LISTEN ") == watcher.MAX_ACTIONS_PER_ROUND

    @pytest.mark.parametrize("budget,arm", [(b, a) for b in (800, 2000, 6000)
                                            for a in (Arm.W, Arm.W_NOQ, Arm.W_NOP, Arm.W_NOCROP)]
                             + [(12000, Arm.W)])
    def test_budget_not_exceeded(self, video_ctx, budget, arm):
        heavy = act({"type": "seek", "t0": 0, "t1": 4, "fps": 12},
                    {"type": "seek", "t0": 1, "t1": 2, "fps": 12},
                    {"type": "crop", "t": 1, "box": [0, 0, 1, 1]},
                    {"type": "listen", "t0": 0, "t1": 6})
        script = Script(acts=[heavy] * 3)
        res, _, _ = run_w(video_ctx, script, budget=budget, arm=arm)
        assert res.error is None
        assert res.content_tokens_est <= budget + 150

    def test_unused_budget_not_force_spent(self, video_ctx):
        res, _, _ = run_w(video_ctx, Script(acts=[FINAL]), budget=10**6)
        assert res.content_tokens_est < 20000

    def test_no_window_served_note(self, video_ctx):
        ctx = dataclasses.replace(video_ctx, candidates=())
        _, _, sess = run_w(ctx, Script(predict='{"surprises": []}'), budget=3000)
        assert "no windows served" in all_text(sess)

    def test_silent_video_has_no_listen_in_focus(self, silent_ctx):
        res, _, sess = run_w(silent_ctx, Script(), budget=12000)
        assert "LISTEN" not in all_text(sess) and res.error is None
        assert "TRANSCRIPT" not in all_text(sess)

    def test_signal_kinds_filter_candidates(self, video_ctx):
        plan = json.dumps({"signal_kinds": ["flash"]})
        res, _, sess = run_w(video_ctx, Script(plan=plan), budget=12000)
        hints = [b["text"] for c in sess.log for b in c if b.get("text", "").startswith("LOW-LEVEL")][0]
        assert "flash" in hints and "cut_in_speech" not in hints
        assert [o for o in res.observations if o.kind == "signal"]

    def test_observations_recorded(self, video_ctx):
        res, _, _ = run_w(video_ctx, Script(), budget=12000)
        kinds = [o.kind for o in res.observations]
        assert kinds.count("transcript") == 2 and "signal" in kinds and "frame" in kinds
        assert all(0 <= o.t0 <= o.t1 <= 4.0 + 1e-6 for o in res.observations)

    def test_budget_scales_evidence(self, video_ctx):
        small, _, _ = run_w(video_ctx, Script(), budget=3000)
        big, _, _ = run_w(video_ctx, Script(), budget=12000)
        assert big.content_tokens_est > small.content_tokens_est


# ------------------------------------------------------------------ rank_windows


def rw(cands=(), surprises=(), duration=10.0, kinds=watcher.SIGNAL_KINDS):
    return watcher.rank_windows(list(cands), tuple(surprises), duration, kinds)


class TestRankWindows:
    def test_empty(self):
        assert rw() == []

    def test_kind_filter(self):
        out = rw([Candidate(2, 2, "flash", 0.8), Candidate(5, 5, "cut", 0.4)], kinds=("flash",))
        assert len(out) == 1 and out[0][3] == "signal: flash"

    def test_padded_window_clamped_to_video(self):
        out = rw([Candidate(0.1, 0.1, "flash", 0.8), Candidate(9.9, 9.9, "flash", 0.7)], duration=10.0)
        by_score = {round(s, 2): (a, b) for a, b, s, _ in out}
        assert by_score[0.8] == (0.0, pytest.approx(0.5)) and by_score[0.7] == (pytest.approx(9.5), 10.0)

    def test_sorted_by_score_desc(self):
        out = rw([Candidate(1, 1, "flash", 0.3), Candidate(5, 5, "flash", 0.9), Candidate(8, 8, "cut", 0.6)])
        assert [round(s, 2) for _, _, s, _ in out] == [0.9, 0.6, 0.3]

    def test_signal_plus_surprise_bonus(self):
        out = rw([Candidate(1.0, 1.0, "cut_in_speech", 0.75)],
                 [watcher.Surprise(t0=1.1, t1=1.2, score=0.5, why="jumpy")])
        assert len(out) == 1
        a, b, s, why = out[0]
        assert s == pytest.approx(0.75 + 0.15) and why == "signal: cut_in_speech"
        assert a == pytest.approx(0.6) and b == pytest.approx(1.6)

    def test_surprise_wins_label_when_higher(self):
        out = rw([Candidate(1.0, 1.0, "cut", 0.35)], [watcher.Surprise(t0=1.0, score=0.9, why="odd")])
        assert out[0][2] == pytest.approx(1.0 + 0.15) and out[0][3].startswith("you flagged: odd")

    def test_same_source_no_bonus(self):
        out = rw([Candidate(1.0, 1.0, "flash", 0.6), Candidate(1.3, 1.3, "cut", 0.8)])
        assert len(out) == 1 and out[0][2] == pytest.approx(0.8)
        out = rw(surprises=[watcher.Surprise(t0=1.0, score=0.4), watcher.Surprise(t0=1.2, score=0.5)])
        assert len(out) == 1 and out[0][2] == pytest.approx(0.6)

    def test_surprise_score_clipped_and_t1_defaults(self):
        out = rw(surprises=[watcher.Surprise(t0=3.0, score=7.0), watcher.Surprise(t0=6.0, t1=None, score=-2)])
        scores = sorted(round(s, 2) for _, _, s, _ in out)
        assert scores == [0.1, 1.1]
        assert out[0][0] == pytest.approx(2.6) and out[0][1] == pytest.approx(3.4)

    def test_reversed_surprise_span(self):
        out = rw(surprises=[watcher.Surprise(t0=4.0, t1=3.5, score=0.5)])
        assert out[0][0] == pytest.approx(3.1) and out[0][1] == pytest.approx(4.4)

    def test_long_span_split_into_ends(self):
        out = rw([Candidate(1.0, 6.0, "freeze", 0.8)], duration=10.0)
        assert len(out) == 2
        (a1, b1, s1, _), (a2, b2, s2, _) = out
        assert (a1, b1) == (pytest.approx(0.6), pytest.approx(1.4)) and s1 == pytest.approx(0.8)
        assert (a2, b2) == (pytest.approx(5.6), pytest.approx(6.4)) and s2 == pytest.approx(0.72)

    def test_merge_limited_by_max_window(self):
        cands = [Candidate(t, t, "flash", 0.5) for t in (1.0, 1.7, 2.4, 3.1, 3.8)]  # chain of overlapping pads
        out = rw(cands, duration=10.0)
        assert len(out) >= 2
        assert all(b - a <= watcher.MAX_WINDOW + 2 * watcher.WINDOW_PAD + 1e-9 for a, b, _, _ in out)

    def test_far_windows_not_merged(self):
        assert len(rw([Candidate(1, 1, "flash", 0.5), Candidate(5, 5, "flash", 0.5)])) == 2

    def test_surprise_text_mentioning_signal_still_counts_as_model(self):
        out = rw([Candidate(1.0, 1.0, "cut_in_speech", 0.75)],
                 [watcher.Surprise(t0=1.0, score=0.5, why="audio signal drops")])
        assert out[0][2] == pytest.approx(0.75 + 0.15)


def test_non_json_act_reply_ends_loop(video_ctx):
    script = Script(acts=["I would like to see more, please"])
    res, _, _ = run_w(video_ctx, script)
    assert script.stages == ["plan", "predict", "act", "final"] and res.parsed


class TestSignalArms:
    """B1H (baseline + hints) and W-noS (watcher without hints) isolate the detectors' contribution."""

    def test_b1h_sends_hints_and_fewer_frames_than_b1(self, video_ctx):
        be1 = FakeBackend(lambda *a: OK_DEFECTS)
        beh = FakeBackend(lambda *a: OK_DEFECTS)
        run_arm(Arm.B1, video_ctx, Task.DEFECTS, "P", 3000, be1)
        res = run_arm(Arm.B1H, video_ctx, Task.DEFECTS, "P", 3000, beh)
        text_h = content_text(beh.sessions[0].log[0])
        assert "LOW-LEVEL SIGNAL HINTS" in text_h
        assert "LOW-LEVEL SIGNAL HINTS" not in content_text(be1.sessions[0].log[0])
        assert count_images(beh.sessions[0].log[0]) <= count_images(be1.sessions[0].log[0])
        assert any(o.kind == "signal" for o in res.observations)

    def test_w_nos_sends_no_hints(self, video_ctx):
        be = FakeBackend(lambda *a: '{"final": {"defects": []}}')
        res = run_arm(Arm.W_NOS, video_ctx, Task.DEFECTS, "P", 6000, be)
        sent = "\n".join(content_text(c) for c in be.sessions[0].log)
        assert "LOW-LEVEL SIGNAL HINTS" not in sent
        assert not any(o.kind == "signal" for o in res.observations)
        assert res.parsed == {"defects": []}

    def test_select_hints_top_k_by_score_in_time_order(self):
        from vwatch.arms.common import select_hints
        from vwatch.signals import Candidate

        cands = [Candidate(float(t), float(t), "cut", s) for t, s in [(5, 0.2), (1, 0.9), (3, 0.5), (9, 0.95)]]
        got = select_hints(cands, k=2)
        assert [c.t0 for c in got] == [1.0, 9.0]
        assert [c.t0 for c in select_hints(cands, kinds=("flash",))] == []


class TestFullView:
    """Short clips are watched whole instead of skimmed when the budget allows."""

    def test_full_view_picks_best_affordable_option(self, video_ctx):
        got = tools.full_view(video_ctx, 10**6)
        assert got == tools.FULL_VIEW_OPTIONS[0]
        assert tools.full_view(video_ctx, 50) is None

    def test_full_view_respects_budget_share(self, video_ctx):
        got = tools.full_view(video_ctx, 20000)
        assert got is not None
        w, fps = got
        assert tools.overview_cost(video_ctx, fps, w) <= 20000 * tools.FULL_VIEW_SHARE

    def test_watcher_uses_full_view_on_short_clip(self, video_ctx):
        be = FakeBackend(lambda *a: '{"final": {"defects": []}}')
        run_arm(Arm.W, video_ctx, Task.DEFECTS, "P", 20000, be)
        sent = "\n".join(content_text(c) for c in be.sessions[0].log)
        assert "FULL VIEW" in sent and "OVERVIEW:" not in sent

    def test_watcher_skims_when_budget_is_small(self, video_ctx):
        be = FakeBackend(lambda *a: '{"final": {"defects": []}}')
        run_arm(Arm.W, video_ctx, Task.DEFECTS, "P", 1500, be)
        sent = "\n".join(content_text(c) for c in be.sessions[0].log)
        assert "FULL VIEW" not in sent
