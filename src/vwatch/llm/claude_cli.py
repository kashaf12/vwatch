"""Backend that drives Claude through the Claude Code CLI in headless stream-json mode.

Used when no ANTHROPIC_API_KEY is available: it rides the logged-in Claude Code account.
The CLI runs with no tools, no MCP servers, no user settings and a replaced system prompt,
so the model sees only what we send.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

from vwatch.llm.base import DEFAULT_MODEL, LLMError, Reply
from vwatch.schemas import Usage


def find_claude_binary() -> str:
    env = os.environ.get("VWATCH_CLAUDE_BIN")
    if env:
        return env
    for cand in (shutil.which("claude"), os.path.expanduser("~/.local/bin/claude"),
                 os.path.expanduser("~/.claude/local/claude")):
        if cand and os.path.exists(cand):
            return cand
    raise LLMError("Claude Code CLI not found; set VWATCH_CLAUDE_BIN or install claude")


def build_command(binary: str, model: str, system: str, effort: str | None) -> list[str]:
    cmd = [binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--system-prompt", system, "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--no-session-persistence", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def parse_usage(raw: dict | None) -> Usage:
    raw = raw or {}
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
        cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
    )


@dataclass
class ClaudeCLISession:
    """One persistent `claude -p` process = one conversation."""

    proc: subprocess.Popen
    timeout_s: float
    _lines: queue.Queue = field(default_factory=queue.Queue)
    _cum_cost: float = 0.0
    _stderr: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    def _fail(self, msg: str) -> LLMError:
        tail = "".join(self._stderr)[-600:]
        return LLMError(f"{msg}; stderr: {tail}" if tail else msg)

    def send(self, content: list[dict]) -> Reply:
        if self.proc.poll() is not None:
            raise self._fail(f"claude process exited ({self.proc.returncode})")
        msg = {"type": "user", "message": {"role": "user", "content": content}}
        t0 = time.monotonic()
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._fail(f"could not write to claude: {exc}") from exc
        while True:
            remaining = self.timeout_s - (time.monotonic() - t0)
            if remaining <= 0:
                self.close()
                raise self._fail(f"claude turn timed out after {self.timeout_s:.0f}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise self._fail("claude closed its output before replying")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON noise
            if event.get("type") != "result":
                continue
            if event.get("is_error"):
                raise self._fail(f"claude returned an error: {str(event.get('result'))[:300]}")
            cum = float(event.get("total_cost_usd") or 0.0)
            turn_cost = max(0.0, cum - self._cum_cost)
            self._cum_cost = cum
            return Reply(text=str(event.get("result") or ""), usage=parse_usage(event.get("usage")),
                         cost_usd=turn_cost, wall_s=time.monotonic() - t0)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()


@dataclass(frozen=True)
class ClaudeCLIBackend:
    model: str = DEFAULT_MODEL
    effort: str | None = "medium"
    timeout_s: float = 600.0
    binary: str | None = None

    def open(self, system: str) -> ClaudeCLISession:
        cmd = build_command(self.binary or find_claude_binary(), self.model, system, self.effort)
        env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
                                cwd=os.path.expanduser("~"))
        return ClaudeCLISession(proc=proc, timeout_s=self.timeout_s)
