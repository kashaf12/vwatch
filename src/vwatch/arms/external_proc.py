"""Drive one headless `claude -p` agent session (stream-json in and out) with a hard deadline.

Unlike `vwatch.llm.claude_cli` (one model turn per send, no tools), here each user message starts a
whole agentic loop: the agent calls tools until it produces a `result` event. Every event is kept so
the caller can audit tool calls afterwards. The process runs in its own process group so a timeout
kills claude together with its MCP servers and any ffmpeg it started.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from vwatch.llm.base import LLMError


class AgentTimeout(LLMError):
    """The agent session ran past its deadline and was killed."""


@dataclass
class AgentProcess:
    """One `claude -p --input-format stream-json` process. Not shared between runs."""

    proc: subprocess.Popen
    deadline: float  # time.monotonic() value after which the process tree is killed
    events: list[dict] = field(default_factory=list)
    _lines: queue.Queue = field(default_factory=queue.Queue)
    _stderr: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, cmd: list[str], *, cwd: str, env: dict[str, str], timeout_s: float) -> AgentProcess:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, cwd=cwd, env=env, start_new_session=True)
        agent = cls(proc=proc, deadline=time.monotonic() + timeout_s)
        threading.Thread(target=agent._pump_stdout, daemon=True).start()
        threading.Thread(target=agent._pump_stderr, daemon=True).start()
        return agent

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    @property
    def stderr_tail(self) -> str:
        return "".join(self._stderr)[-800:]

    def _fail(self, msg: str, cls: type[LLMError] = LLMError) -> LLMError:
        tail = self.stderr_tail
        return cls(f"{msg}; stderr: {tail}" if tail.strip() else msg)

    def ask(self, text: str) -> dict:
        """Send one user message and block until its `result` event. Returns that event."""
        if self.proc.poll() is not None:
            raise self._fail(f"claude process exited ({self.proc.returncode})")
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._fail(f"could not write to claude: {exc}") from exc
        return self._read_until_result()

    def _read_until_result(self) -> dict:
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise self._fail("agent timed out and was killed", AgentTimeout)
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise self._fail(f"claude closed its output before a result (exit {self.proc.poll()})")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON noise on stdout
            if not isinstance(event, dict):
                continue
            self.events.append(event)
            if event.get("type") == "result":
                return event

    def kill(self) -> None:
        """Kill the whole process group (claude, MCP servers, ffmpeg children)."""
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=5)
        except ProcessLookupError:
            pass  # exited between poll() and killpg()

    def close(self) -> None:
        """End the session politely (close stdin), then make sure nothing is left running."""
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                pass  # falls through to kill() below
        self.kill()
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)  # stray MCP servers that outlived claude
        except (ProcessLookupError, PermissionError):
            pass
