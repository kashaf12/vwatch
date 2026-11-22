"""External-tool arms (CC / CVV / POP), driven by fake `claude` binaries that speak stream-json."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from vwatch.arms import external as ex
from vwatch.arms import run_arm
from vwatch.arms.external_audit import ToolCall, audit, tool_calls
from vwatch.arms.external_obs import ffmpeg_frame_map, infer_observations, parse_time
from vwatch.schemas import Arm, Task

ANSWER = '{"defects": [{"type": "flash_frame", "start": 1.0, "end": 1.1, "evidence": "white frame"}]}'

FAKE = """#!{python}
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
scen = json.load(open(os.path.join(HERE, "scenario.json")))
rec = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "files": sorted(os.listdir(".")), "pid": os.getpid(),
       "env": sorted(os.environ), "stdin": []}}
print(json.dumps({{"type": "system", "subtype": "init", "cwd": os.getcwd()}}), flush=True)
for turn in scen:
    line = sys.stdin.readline()
    if not line:
        break
    rec["stdin"].append(json.loads(line))
    with open(os.path.join(HERE, "record.json"), "w") as f:
        json.dump(rec, f)
    time.sleep(turn.get("sleep", 0))
    for ev in turn["events"]:
        print(json.dumps(ev), flush=True)
with open(os.path.join(HERE, "record.json"), "w") as f:
    json.dump(rec, f)
if scen:  # stay alive until stdin closes, like claude; an empty scenario exits right away
    sys.stdin.read()
"""


def use(tid: str, name: str, inp: dict, mid: str) -> dict:
    return {"type": "assistant", "message": {"id": mid, "content": [
        {"type": "tool_use", "id": tid, "name": name, "input": inp}]}}


def res(tid: str, content: object) -> dict:
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]}}


def result(text: str, cost: float, inp: int = 100, out: int = 20, is_error: bool = False) -> dict:
    return {"type": "result", "subtype": "success", "is_error": is_error, "result": text,
            "total_cost_usd": cost, "usage": {"input_tokens": inp, "output_tokens": out,
                                              "cache_read_input_tokens": 1000}}


def say(text: str, mid: str) -> dict:
    return {"type": "assistant", "message": {"id": mid, "content": [{"type": "text", "text": text}]}}


def make_fake(tmp_path: Path, scenario: list[dict]) -> Path:
    d = tmp_path / "fakebin"
    d.mkdir(exist_ok=True)
    (d / "scenario.json").write_text(json.dumps(scenario))
    script = d / "claude"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def record(script: Path) -> dict:
    return json.loads((script.parent / "record.json").read_text())


def cfg_for(tmp_path: Path, script: Path, **kw) -> ex.ExternalConfig:
    kw.setdefault("timeout_s", 30.0)
    return ex.ExternalConfig(model="claude-opus-5-5", effort="medium", binary=str(script),
                             log_dir=tmp_path / "logs", temp_root=tmp_path / "runs", **kw)


def fake_vendor(tmp_path: Path) -> Path:
    v = tmp_path / "vendor"
    for rel in ("claude-video-vision/mcp-server/dist/index.js", "popcorn/dist/index.js",
                "whisper.cpp/build/bin/whisper-cli", "popcorn-venv/bin/python3"):
        (v / rel).parent.mkdir(parents=True, exist_ok=True)
        (v / rel).write_text("")
    for part in (".claude-plugin", "commands", "skills", "agents"):
        (v / "claude-video-vision" / part).mkdir()
        (v / "claude-video-vision" / part / "x.md").write_text("x")
    return v


def good_run(workdir_marker: str = "") -> list[dict]:
    return [{"events": [
        use("t1", "Bash", {"command": "mkdir -p frames && ffmpeg -i video.mp4 -vf fps=2 frames/f_%03d.jpg",
                           "description": "extract frames"}, "m1"),
        res("t1", "ok"),
        use("t2", "Read", {"file_path": f"{workdir_marker}frames/f_003.jpg"}, "m2"),
        res("t2", [{"type": "text", "text": "image"}]),
        say(ANSWER, "m3"),
        result(ANSWER, 0.42),
    ]}]


# ---------------------------------------------------------------- command construction


class TestCommand:
    def test_cc_flags(self):
        cfg = ex.ExternalConfig(model="m1", effort="high")
        cmd = ex.build_command(Arm.CC, cfg, "claude", [])
        assert cmd[:2] == ["claude", "-p"]
        joined = " ".join(cmd)
        for flag in ("--input-format stream-json", "--output-format stream-json", "--strict-mcp-config",
                     "--no-session-persistence", "--permission-mode dontAsk", "--permission-prompts none",
                     "--model m1", "--effort high", "--tools Bash,Read"):
            assert flag in joined
        assert cmd[cmd.index("--setting-sources") + 1] == ""
        assert "--dangerously-skip-permissions" not in cmd and "--system-prompt" not in cmd
        assert "--plugin-dir" not in cmd and "--mcp-config" not in cmd
        dis = cmd.index("--disallowedTools")
        assert cmd[dis + 1:dis + 3] == ["WebFetch", "WebSearch"]

    def test_no_effort(self):
        assert "--effort" not in ex.build_command(Arm.CC, ex.ExternalConfig(effort=None), "c", [])

    def test_allowed_tools(self):
        cc = ex.allowed_tools(Arm.CC)
        assert "Read(./**)" in cc and "Bash(ffmpeg *)" in cc and "Bash(ffprobe *)" in cc
        cmds = {t[5:-1].split(" ")[0] for t in cc if t.startswith("Bash(")}
        assert {"python3", "ffmpeg", "ls"} <= cmds and not cmds & {"curl", "wget", "rm", "mv", "sed", "awk", "xargs"}
        assert ex.CVV_MCP not in cc and ex.POP_MCP not in cc
        assert ex.CVV_MCP in ex.allowed_tools(Arm.CVV) and ex.POP_MCP not in ex.allowed_tools(Arm.CVV)
        assert ex.POP_MCP in ex.allowed_tools(Arm.POP) and ex.CVV_MCP not in ex.allowed_tools(Arm.POP)

    def test_prompts(self):
        assert ex.agent_prompt(Arm.CC, "TASK").startswith("The video is ./video.mp4. TASK")
        assert ex.agent_prompt(Arm.CVV, "TASK").startswith("/claude-video-vision:watch-video ./video.mp4 TASK")
        assert "popcorn" in ex.agent_prompt(Arm.POP, "TASK")

    def test_setup_cvv_and_pop(self, tmp_path, tone_video):
        vendor = fake_vendor(tmp_path)
        cfg = ex.ExternalConfig(vendor_dir=vendor, check_pins=False, temp_root=tmp_path / "r")
        env = {"PATH": os.environ["PATH"], "HOME": "/h"}
        dirs = ex.make_run_dirs(tone_video, cfg.temp_root)
        extra = ex.setup_cvv(cfg, dirs, env)
        assert extra[0] == "--plugin-dir" and "--mcp-config" in extra
        conf = json.loads(Path(extra[extra.index("--mcp-config") + 1]).read_text())
        server = conf["mcpServers"]["claude-video-vision"]
        assert server["args"][0].endswith("mcp-server/dist/index.js")
        assert server["env"]["HOME"].startswith(str(dirs.root))  # plugin config isolated from ~
        assert server["env"]["PATH"].startswith(str(vendor / "whisper.cpp/build/bin"))
        home_conf = json.loads((dirs.root / "home/.claude-video-vision/config.json").read_text())
        assert home_conf["backend"] == "local"
        assert json.loads((Path(extra[1]) / ".mcp.json").read_text()) == {}  # no `npx ...@latest`
        pop = ex.setup_pop(cfg, dirs, env)
        assert pop[0] == "--mcp-config" and "--plugin-dir" not in pop
        pconf = json.loads(Path(pop[1]).read_text())["mcpServers"]["popcorn"]
        assert pconf["env"]["PATH"].startswith(str(vendor / "popcorn-venv/bin"))

    def test_missing_vendor_fails_fast(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, [])
        cfg = cfg_for(tmp_path, script, vendor_dir=tmp_path / "nope")
        out = ex.run_external(Arm.POP, video_ctx, Task.DEFECTS, "TASK", None, cfg)
        assert out.error and "SetupError" in out.error and "setup_external.sh" in out.error
        assert not (script.parent / "record.json").exists()  # claude never started
        assert list((tmp_path / "runs").iterdir()) == []  # temp dir cleaned

    def test_pin_mismatch(self, tmp_path):
        repo = tmp_path / "popcorn"
        repo.mkdir()
        os.system(f"git -C {repo} init -q && git -C {repo} -c user.email=a@b -c user.name=a "
                  f"commit -q --allow-empty -m x")
        with pytest.raises(ex.SetupError, match="PINS"):
            ex._pinned(tmp_path, "popcorn", True)

    def test_agent_env_drops_repo_paths(self):
        repo = Path("/r/vwatch")
        env = ex.agent_env({"HOME": "/h", "PATH": f"/r/vwatch/.venv/bin:/usr/bin",
                            "VIRTUAL_ENV": "/r/vwatch/.venv", "PWD": "/r/vwatch",
                            "ANTHROPIC_API_KEY": "k", "SECRET": "s"}, repo)
        assert env["PATH"] == "/usr/bin" and env["HOME"] == "/h" and env["ANTHROPIC_API_KEY"] == "k"
        assert "VIRTUAL_ENV" not in env and "PWD" not in env and "SECRET" not in env
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


# ---------------------------------------------------------------- end to end with a fake claude


class TestRun:
    def test_happy_path(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, good_run())
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "FIND DEFECTS", None, cfg_for(tmp_path, script))
        assert out.error is None and out.parse_error is None
        assert out.parsed["defects"][0]["type"] == "flash_frame"
        assert out.cost_usd == pytest.approx(0.42)
        assert out.usage.input_tokens == 100 and out.usage.cache_read_input_tokens == 1000
        assert out.turns == 3 and out.content_tokens_est == 0
        assert [c["name"] for c in out.tool_calls] == ["Bash", "Read"]
        assert [(o.t0, o.kind) for o in out.observations] == [(1.0, "frame")]  # f_003 at fps=2
        rec = record(script)
        assert rec["files"] == ["video.mp4"]  # isolation: only the neutral-named copy
        assert os.path.realpath(rec["cwd"]).startswith(os.path.realpath(tmp_path / "runs"))
        assert Path(rec["cwd"]).name == "work"
        assert "tone" not in json.dumps(rec["stdin"])  # source file name never shown
        assert rec["stdin"][0]["message"]["content"][0]["text"].startswith("The video is ./video.mp4. FIND")
        assert "VIRTUAL_ENV" not in rec["env"]
        assert list((tmp_path / "runs").iterdir()) == []  # workdir deleted
        logs = list((tmp_path / "logs").glob("CC_tone_*.json"))
        assert len(logs) == 1 and json.loads(logs[0].read_text())["tool_calls"][0]["name"] == "Bash"

    def test_cvv_command_has_plugin(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, good_run())
        cfg = cfg_for(tmp_path, script, vendor_dir=fake_vendor(tmp_path), check_pins=False)
        out = ex.run_external(Arm.CVV, video_ctx, Task.DEFECTS, "Q", None, cfg)
        assert out.error is None
        rec = record(script)
        assert "--plugin-dir" in rec["argv"] and "--mcp-config" in rec["argv"]
        assert rec["files"] == ["video.mp4"]  # plugin + config live outside the workdir
        assert rec["stdin"][0]["message"]["content"][0]["text"].startswith("/claude-video-vision:watch-video")

    def test_audit_flags_outside_path(self, tmp_path, video_ctx):
        scen = [{"events": [use("t1", "Read", {"file_path": "/etc/hosts"}, "m1"), res("t1", "denied"),
                            use("t2", "Bash", {"command": "ls ../../"}, "m2"), result(ANSWER, 0.1)]}]
        script = make_fake(tmp_path, scen)
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error.startswith("audit:") and "/etc/hosts" in out.error and "../../" in out.error
        assert out.parsed is not None  # kept for review, but the error makes it a failure

    def test_audit_flags_truth_marker(self, tmp_path, video_ctx):
        scen = [{"events": [use("t1", "Bash", {"command": "ls *truth*"}, "m1"), result(ANSWER, 0.1)]}]
        script = make_fake(tmp_path, scen)
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error.startswith("audit:") and "truth" in out.error

    def test_timeout_kills(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, [{"sleep": 60, "events": [result(ANSWER, 0.1)]}])
        t0 = time.monotonic()
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script, timeout_s=1.5))
        assert time.monotonic() - t0 < 20
        assert out.error.startswith("timeout") and out.parsed is None
        pid = record(script)["pid"]
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert list((tmp_path / "runs").iterdir()) == []

    def test_repair_turn(self, tmp_path, video_ctx):
        scen = [{"events": [say("I think there is a flash.", "m1"), result("I think there is a flash.", 0.3, 50, 5)]},
                {"events": [say(ANSWER, "m2"), result(ANSWER, 0.35, 10, 7)]}]
        script = make_fake(tmp_path, scen)
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error is None and out.parsed is not None and out.parse_error is None
        assert out.cost_usd == pytest.approx(0.35)  # total_cost_usd is cumulative
        assert out.usage.input_tokens == 60 and out.usage.output_tokens == 12
        assert out.turns == 2
        assert "not valid JSON" in record(script)["stdin"][1]["message"]["content"][0]["text"]

    def test_parse_error_after_repair(self, tmp_path, video_ctx):
        scen = [{"events": [result("no json", 0.1)]}, {"events": [result("still none", 0.2)]}]
        script = make_fake(tmp_path, scen)
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.parsed is None and out.parse_error and out.error is None
        assert out.raw_output == "still none" and out.cost_usd == pytest.approx(0.2)

    def test_is_error_result(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, [{"events": [result("rate limited", 0.0, is_error=True)]}])
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error and "rate limited" in out.error

    def test_process_dies(self, tmp_path, video_ctx):
        script = make_fake(tmp_path, [])  # exits right after init
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error and "LLMError" in out.error

    def test_run_arm_dispatch(self, monkeypatch, video_ctx):
        seen = {}

        def fake(arm, ctx, task, prompt, backend):
            seen.update(arm=arm, prompt=prompt)
            return "ok"

        monkeypatch.setattr("vwatch.arms.run_external", fake)
        assert run_arm(Arm.POP, video_ctx, Task.DEFECTS, "P", 10, object()) == "ok"
        assert seen == {"arm": Arm.POP, "prompt": "P"}

    def test_rejects_internal_arm(self, video_ctx):
        with pytest.raises(ValueError):
            ex.run_external(Arm.B1, video_ctx, Task.DEFECTS, "T", None)


# ---------------------------------------------------------------- audit and observations


class TestAudit:
    W = Path("/var/folders/x/T/vwatch-ext/abc/work")

    def flags(self, name: str, inp: dict) -> tuple[str, ...]:
        return audit((ToolCall("i", name, inp),), self.W, repo_root=Path("/Users/u/vwatch"))

    @pytest.mark.parametrize("cmd", [
        "ffmpeg -i video.mp4 -vf fps=1/2,scale=iw/2:ih/2 frames/f_%03d.jpg 2>/dev/null",
        "ffprobe -v error -show_entries format=duration -of csv=p=0 video.mp4",
        f"ls {W}/frames", f"ls /private{W}", "ls frames/../video.mp4", "echo 'scale=2; 30/4' | bc",
    ])
    def test_clean(self, cmd):
        assert self.flags("Bash", {"command": cmd, "description": "look in /etc for truth"}) == ()

    @pytest.mark.parametrize("cmd,needle", [
        ("cat /etc/passwd", "/etc/passwd"), ("ls ~/", "home"), ("ls ..", "escapes"),
        ("ffmpeg -i frames/../../x.mp4 o.jpg", "escapes"), ("grep -r x /", "outside"),
        ("ls /Users/u/vwatch/eval", "repo"), ("cat manifest.jsonl", "manifest"),
        ("ffmpeg -i movie=/tmp/x.mp4 o.jpg", "/tmp/x.mp4"),
    ])
    def test_flagged(self, cmd, needle):
        out = self.flags("Bash", {"command": cmd})
        assert len(out) == 1 and needle in out[0]

    def test_follows_cd(self):
        calls = (ToolCall("a", "Bash", {"command": "mkdir -p an && cd an && ffmpeg -i ../video.mp4 x.png"}),
                 ToolCall("b", "Bash", {"command": "ls ../video.mp4 && cat ../../escape"}),
                 ToolCall("c", "Bash", {"command": "cd .. && ls"}),  # back to the workdir: fine
                 ToolCall("d", "Bash", {"command": "cd .. && ls"}))  # now above it
        out = audit(calls, self.W)
        assert len(out) == 2 and "../../escape" in out[0] and out[1].startswith("Bash(cd ..")

    def test_mcp_and_read(self):
        assert self.flags("Read", {"file_path": f"{self.W}/frames/a.jpg"}) == ()
        assert self.flags("mcp__popcorn__popcorn_read", {"path": "/Users/u/x.json"})
        assert self.flags("mcp__claude-video-vision__video_watch", {"path": "./video.mp4", "fps": "auto"}) == ()

    def test_tool_calls_dedupe_and_summary(self):
        ev = use("a", "Bash", {"command": "x" * 500}, "m")
        calls = tool_calls([ev, ev, {"type": "user"}])
        assert len(calls) == 1 and len(calls[0].summary) == 240
        assert calls[0].to_log() == {"name": "Bash", "input": calls[0].summary}


class TestObservations:
    W = Path("/tmp/w")

    def test_parse_time(self):
        assert parse_time("1:02.5") == 62.5 and parse_time("00:00:03") == 3 and parse_time("x") is None

    def test_ffmpeg_map(self):
        m = ffmpeg_frame_map("ffmpeg -ss 10 -t 2 -i video.mp4 -vf 'fps=2,scale=320:-1' f/%02d.jpg", self.W, 30)
        assert sorted(m.values()) == [10.0, 10.5, 11.0, 11.5]
        one = ffmpeg_frame_map("ffmpeg -y -ss 00:00:04.5 -i video.mp4 -frames:v 1 a.png", self.W, 30)
        assert list(one.values()) == [4.5]
        assert ffmpeg_frame_map("ffmpeg -i video.mp4 -vf fps=1,tile=4x4 sheet.jpg", self.W, 30) == {}
        assert ffmpeg_frame_map("ffprobe video.mp4", self.W, 30) == {}
        assert len(ffmpeg_frame_map("ffmpeg -i video.mp4 -vf fps=1 -frames:v 5 %03d.jpg", self.W, 30)) == 5

    def test_unread_frames_not_observed(self):
        ev = [use("a", "Bash", {"command": "ffmpeg -i video.mp4 -vf fps=1 %03d.jpg"}, "m")]
        assert infer_observations(ev, self.W, 10) == ()

    def test_cvv_frames(self):
        ev = [use("a", "mcp__claude-video-vision__video_watch", {"path": "./video.mp4"}, "m"),
              res("a", [{"type": "text", "text": "## Video Metadata"}, {"type": "text", "text": "### Frame at 00:00:07"},
                        {"type": "image", "data": "x"}, {"type": "text", "text": "### Frame at 00:01:02"}])]
        obs = infer_observations(ev, self.W, 90)
        assert [(o.t0, o.t1) for o in obs] == [(7.0, 7.999), (62.0, 62.999)]

    def test_popcorn_inline_and_read(self):
        frames = [{"path": f"/tmp/w/.popcorn/k/{i}.jpg", "timeSec": float(i * 2)} for i in range(5)]
        payload = json.dumps({"keyframes": {"frames": frames}, "inlineFrames": {"count": 2}})
        ev = [use("a", "mcp__popcorn__popcorn_analyze", {"path": "./video.mp4"}, "m"),
              res("a", [{"type": "text", "text": payload}]),
              use("b", "Read", {"file_path": "/tmp/w/.popcorn/k/2.jpg"}, "m2")]
        assert [o.t0 for o in infer_observations(ev, self.W, 10)] == [0.0, 4.0, 8.0]

    def test_odd_events_ignored(self, tmp_path, video_ctx):
        odd = [{"type": "system", "subtype": "api_retry", "message": "retrying"},
               {"type": "assistant", "message": "not a dict"}, {"type": "user", "message": {"content": "text"}}]
        assert infer_observations(odd, self.W, 10) == () and tool_calls(odd) == ()
        script = make_fake(tmp_path, [{"events": odd + [result(ANSWER, 0.1)]}])
        out = ex.run_external(Arm.CC, video_ctx, Task.DEFECTS, "T", None, cfg_for(tmp_path, script))
        assert out.error is None and out.parsed is not None


class TestAuditInlineCode:
    """Division in inline Python must not read as the filesystem root; real root access still must."""

    W = Path("/tmp/vwatch-ext/x/work")

    def _flags(self, command):
        from vwatch.arms.external_audit import audit
        from vwatch.arms.external_audit import ToolCall

        return audit((ToolCall("i", "Bash", {"command": command}),), self.W)

    def test_division_in_python_c_is_clean(self):
        assert self._flags('python3 -c "\nt=open(\'stats.txt\').read()\nprint(len(t) / 2)\n"') == ()

    def test_quoted_root_in_python_c_is_flagged(self):
        assert self._flags('python3 -c "import os; print(os.listdir(\'/\'))"')

    def test_bare_root_in_shell_is_still_flagged(self):
        assert self._flags("grep -r secret /")
