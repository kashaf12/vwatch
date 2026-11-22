"""Scripted backend for tests: no network, records everything it was sent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vwatch.llm.base import Reply
from vwatch.schemas import Usage

Responder = Callable[[str, list[dict], int], str]  # (system, content, turn_index) -> reply text


def count_images(content: list[dict]) -> int:
    return sum(1 for b in content if b.get("type") == "image")


def content_text(content: list[dict]) -> str:
    return "\n".join(b["text"] for b in content if b.get("type") == "text")


@dataclass
class FakeSession:
    system: str
    responder: Responder
    log: list[list[dict]]
    turn: int = 0

    def send(self, content: list[dict]) -> Reply:
        self.log.append(content)
        text = self.responder(self.system, content, self.turn)
        self.turn += 1
        usage = Usage(input_tokens=100 + 300 * count_images(content), output_tokens=50)
        return Reply(text=text, usage=usage, cost_usd=0.001, wall_s=0.0)

    def close(self) -> None:
        return None


@dataclass
class FakeBackend:
    responder: Responder
    model: str = "fake-model"
    sessions: list[FakeSession] = field(default_factory=list)

    def open(self, system: str) -> FakeSession:
        s = FakeSession(system=system, responder=self.responder, log=[])
        self.sessions.append(s)
        return s
