import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from vwatch.llm import DEFAULT_MODEL, default_backend, extract_json, text_block
from vwatch.llm import anthropic_api as api
from vwatch.llm import claude_cli as cli
from vwatch.llm.base import LLMError
from vwatch.llm.fake import FakeBackend, content_text, count_images
from vwatch.schemas import Usage


# ------------------------------------------------------------ extract_json
class TestExtractJson:
    def test_bare(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('Here:\n```json\n{"a": 2}\n```\nbye') == {"a": 2}

    def test_fenced_no_lang(self):
        assert extract_json('```\n{"a": 3}\n```') == {"a": 3}

    def test_multiple_objects_last_wins(self):
        assert extract_json('{"a": 1} then {"a": 2}') == {"a": 2}

    def test_multiple_fences_last_wins(self):
        txt = '```json\n{"a": 1}\n```\ntext\n```json\n{"a": 9}\n```'
        assert extract_json(txt) == {"a": 9}

    def test_nested_is_one_object(self):
        assert extract_json('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}

    def test_prose_braces_before(self):
        assert extract_json('use {curly} braces. {"ok": true}') == {"ok": True}

    def test_none_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            extract_json("nothing here")

    def test_array_only_raises(self):
        with pytest.raises(ValueError):
            extract_json("[1, 2, 3]")

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            extract_json('{"a": ')

    def test_text_block(self):
        assert text_block("hi") == {"type": "text", "text": "hi"}


# ------------------------------------------------------------ fake backend
class TestFake:
    def test_records_and_scripts(self):
        be = FakeBackend(responder=lambda s, c, t: f"{s}:{t}:{content_text(c)}")
        sess = be.open("sys")
        r0 = sess.send([text_block("a"), {"type": "image"}, {"type": "image"}])
        r1 = sess.send([text_block("b")])
        assert (r0.text, r1.text) == ("sys:0:a", "sys:1:b")
        assert r0.usage.input_tokens == 700 and r1.usage.input_tokens == 100
        assert count_images(sess.log[0]) == 2
        assert be.sessions == [sess] and len(sess.log) == 2
        assert sess.close() is None
        assert r0.cost_usd == 0.001


# ------------------------------------------------------------ default_backend
class TestDefaultBackend:
    def test_cli_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("VWATCH_BACKEND", raising=False)
        be = default_backend(model="m", effort="low")
        assert isinstance(be, cli.ClaudeCLIBackend) and be.model == "m" and be.effort == "low"

    def test_api_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("VWATCH_BACKEND", raising=False)
        assert isinstance(default_backend(), api.AnthropicBackend)

    def test_cli_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("VWATCH_BACKEND", "cli")
        assert isinstance(default_backend(), cli.ClaudeCLIBackend)

    def test_default_model_exported(self):
        assert default_backend().model == DEFAULT_MODEL


# ------------------------------------------------------------ claude_cli helpers
class TestCliHelpers:
    def test_build_command(self):
        cmd = cli.build_command("/bin/claude", "mod", "SYS", "high")
        assert cmd[:2] == ["/bin/claude", "-p"]
        assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
        assert cmd[cmd.index("--model") + 1] == "mod"
        assert cmd[cmd.index("--effort") + 1] == "high"
        assert cmd[cmd.index("--tools") + 1] == ""
        for flag in ("--strict-mcp-config", "--no-session-persistence", "--verbose"):
            assert flag in cmd

    def test_build_command_no_effort(self):
        assert "--effort" not in cli.build_command("c", "m", "s", None)

    def test_parse_usage(self):
        u = cli.parse_usage({"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 1,
                             "cache_creation_input_tokens": None})
        assert u == Usage(input_tokens=5, output_tokens=7, cache_read_input_tokens=1)

    def test_parse_usage_none(self):
        assert cli.parse_usage(None) == Usage()

    def test_find_binary_env(self, monkeypatch):
        monkeypatch.setenv("VWATCH_CLAUDE_BIN", "/x/claude")
        assert cli.find_claude_binary() == "/x/claude"

    def test_find_binary_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VWATCH_CLAUDE_BIN", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(LLMError, match="not found"):
            cli.find_claude_binary()

    def test_find_binary_on_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VWATCH_CLAUDE_BIN", raising=False)
        exe = tmp_path / "claude"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert cli.find_claude_binary() == str(exe)


# ------------------------------------------------------------ fake claude binary
FAKE_CLAUDE = '''#!{py}
import json, os, sys, time
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "die":
    sys.exit(3)
n = 0
for line in sys.stdin:
    msg = json.loads(line)
    n += 1
    if mode == "exit_on_msg":
        sys.stderr.write("boom happened\\n"); sys.stderr.flush()
        sys.exit(1)
    if mode == "hang":
        sys.stdin.read()
        sys.exit(0)
    print("not json noise", flush=True)
    print(json.dumps({{"type": "system", "subtype": "init"}}), flush=True)
    text = " ".join(b.get("text", "") for b in msg["message"]["content"])
    ev = {{"type": "result", "result": "reply%d: %s" % (n, text), "total_cost_usd": 0.01 * n * (n + 1) / 2,
          "usage": {{"input_tokens": 10 * n, "output_tokens": 2}}}}
    if mode == "error":
        ev.update(is_error=True, result="rate limited")
    print(json.dumps(ev), flush=True)
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    path = tmp_path / "fake_claude"
    path.write_text(FAKE_CLAUDE.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    return str(path)


class TestClaudeCLISession:
    def test_multi_turn_cost_is_per_turn(self, fake_claude):
        be = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20)
        sess = be.open("sys")
        try:
            r1 = sess.send([text_block("one")])
            r2 = sess.send([text_block("two")])
            r3 = sess.send([text_block("three")])
        finally:
            sess.close()
        assert r1.text == "reply1: one" and r2.text == "reply2: two" and r3.text == "reply3: three"
        # cumulative = 0.01, 0.03, 0.06 -> per-turn 0.01, 0.02, 0.03
        assert [round(r.cost_usd, 6) for r in (r1, r2, r3)] == [0.01, 0.02, 0.03]
        assert r2.usage.input_tokens == 20 and r2.usage.output_tokens == 2
        assert r1.wall_s >= 0

    def test_close_is_idempotent(self, fake_claude):
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20).open("s")
        sess.send([text_block("x")])
        sess.close()
        sess.close()
        assert sess.proc.poll() is not None

    def test_is_error(self, fake_claude, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20).open("s")
        try:
            with pytest.raises(LLMError, match="rate limited"):
                sess.send([text_block("x")])
        finally:
            sess.close()

    def test_process_exits_mid_turn(self, fake_claude, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "exit_on_msg")
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20).open("s")
        try:
            with pytest.raises(LLMError, match="closed its output|could not write"):
                sess.send([text_block("x")])
        finally:
            sess.close()

    def test_process_already_dead(self, fake_claude, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "die")
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20).open("s")
        sess.proc.wait(timeout=10)
        try:
            with pytest.raises(LLMError, match="exited"):
                sess.send([text_block("x")])
        finally:
            sess.close()

    def test_exit_message(self, fake_claude, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "exit_on_msg")
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=20).open("s")
        with pytest.raises(LLMError) as ei:
            sess.send([text_block("x")])
        assert "closed its output" in str(ei.value) or "could not write" in str(ei.value)

    def test_timeout(self, fake_claude, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "hang")
        sess = cli.ClaudeCLIBackend(binary=fake_claude, timeout_s=0.6).open("s")
        try:
            with pytest.raises(LLMError, match="timed out"):
                sess.send([text_block("x")])
        finally:
            sess.close()


# ------------------------------------------------------------ anthropic api
class TestAnthropicCost:
    def test_turn_cost_math(self):
        u = Usage(input_tokens=1_000_000, output_tokens=1_000_000,
                  cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
        # opus 4/20: 4 + 4*1.25 + 4*0.1 + 20
        assert api.turn_cost("claude-opus-5-5", u) == pytest.approx(4 + 5 + 0.4 + 20)

    def test_unknown_model_uses_default_price(self):
        u = Usage(input_tokens=1_000_000)
        assert api.turn_cost("who-knows", u) == pytest.approx(4.0)

    def test_sonnet(self):
        assert api.turn_cost("claude-sonnet-5-5", Usage(output_tokens=500_000)) == pytest.approx(5.0)


class _Block(SimpleNamespace):
    pass


def _resp(text="hello", stop="end_turn"):
    return SimpleNamespace(
        content=[_Block(type="text", text=text), _Block(type="thinking", text="ignored")],
        stop_reason=stop, stop_details="because",
        usage=SimpleNamespace(input_tokens=100, output_tokens=10,
                              cache_read_input_tokens=None, cache_creation_input_tokens=50))


class _Client:
    def __init__(self, resp=None, exc=None):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._resp, self._exc = resp, exc

    def _create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return self._resp


class TestAnthropicSession:
    def test_send_builds_request_and_accumulates_history(self):
        client = _Client(_resp("hi"))
        s = api.AnthropicSession(client=client, model="claude-opus-5-5", system="SYS", effort="low")
        r = s.send([text_block("q1")])
        assert r.text == "hi"  # non-text blocks are dropped
        assert r.usage == Usage(input_tokens=100, output_tokens=10, cache_creation_input_tokens=50)
        assert r.cost_usd == pytest.approx((100 * 4 + 50 * 4 * 1.25 + 10 * 20) / 1e6)
        kw = client.calls[0]
        assert kw["system"] == "SYS" and kw["output_config"] == {"effort": "low"}
        assert kw["messages"] == [{"role": "user", "content": [text_block("q1")]}]
        s.send([text_block("q2")])
        assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant", "user"]
        # first call's list must not have been mutated by the second send
        assert len(client.calls[0]["messages"]) == 1
        assert s.close() is None

    def test_no_effort_omits_output_config(self):
        client = _Client(_resp())
        api.AnthropicSession(client=client, model="m", system="s", effort=None).send([text_block("q")])
        assert "output_config" not in client.calls[0]

    def test_refusal(self):
        s = api.AnthropicSession(client=_Client(_resp(stop="refusal")), model="m", system="s", effort=None)
        with pytest.raises(LLMError, match="refused"):
            s.send([text_block("q")])

    def test_status_error(self):
        import anthropic

        resp = SimpleNamespace(request=None, status_code=429, headers={})
        exc = anthropic.APIStatusError("bad", response=resp, body=None)
        s = api.AnthropicSession(client=_Client(exc=exc), model="m", system="s", effort=None)
        with pytest.raises(LLMError, match="429"):
            s.send([text_block("q")])

    def test_connection_error(self):
        import anthropic

        exc = anthropic.APIConnectionError(request=None)
        s = api.AnthropicSession(client=_Client(exc=exc), model="m", system="s", effort=None)
        with pytest.raises(LLMError, match="connection"):
            s.send([text_block("q")])

    def test_backend_open(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        sess = api.AnthropicBackend(model="m", effort="low").open("sys")
        assert sess.system == "sys" and sess.model == "m"


# ------------------------------------------------------------ stubs for rarely hit paths
class _Stdin:
    def __init__(self, fail=False):
        self.fail, self.closed = fail, False

    def write(self, s):
        if self.fail:
            raise BrokenPipeError("pipe closed")

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _Proc:
    def __init__(self, stdin, wait_raises=False):
        import io

        self.stdin, self.stdout, self.stderr = stdin, io.StringIO(""), io.StringIO("some stderr text\n")
        self.returncode, self._wait_raises, self.killed = None, wait_raises, False

    def poll(self):
        return None if not self.killed else -9

    def wait(self, timeout=None):
        if self._wait_raises:
            import subprocess

            raise subprocess.TimeoutExpired("claude", timeout)

    def kill(self):
        self.killed = True


class TestCliStubs:
    def test_broken_pipe_becomes_llm_error(self):
        sess = cli.ClaudeCLISession(proc=_Proc(_Stdin(fail=True)), timeout_s=5)
        import time

        time.sleep(0.1)  # let the stderr pump read
        with pytest.raises(LLMError, match="could not write to claude"):
            sess.send([text_block("x")])

    def test_close_kills_stuck_process(self):
        proc = _Proc(_Stdin(), wait_raises=True)
        sess = cli.ClaudeCLISession(proc=proc, timeout_s=5)
        sess.close()
        assert proc.stdin.closed and proc.killed

    def test_stderr_tail_included(self):
        import time

        sess = cli.ClaudeCLISession(proc=_Proc(_Stdin()), timeout_s=5)
        time.sleep(0.2)
        with pytest.raises(LLMError, match="closed its output.*some stderr text"):
            sess.send([text_block("x")])


class TestJudge:
    def test_judge_uses_fresh_session_and_closes(self, monkeypatch):
        from vwatch.llm import judge

        be = FakeBackend(responder=lambda s, c, t: "verdict")
        seen = {}

        def fake_default(model, effort):
            seen.update(model=model, effort=effort)
            return be

        monkeypatch.setattr(judge, "default_backend", fake_default)
        j = judge.make_judge(model="m-judge")
        assert j("p1") == "verdict" and j("p2") == "verdict"
        assert seen == {"model": "m-judge", "effort": "low"}
        assert len(be.sessions) == 2 and be.sessions[0].system == judge.JUDGE_SYSTEM
