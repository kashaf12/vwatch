"""Model backend contracts.

A Backend opens Sessions. A Session is one conversation: each `send` appends a user turn
(content blocks in Anthropic Messages format: text and base64 image blocks) and returns
the assistant's reply text plus usage for that turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from vwatch.schemas import Usage

DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    """The model call failed (transport, auth, refusal or timeout)."""


@dataclass(frozen=True)
class Reply:
    text: str
    usage: Usage
    cost_usd: float  # cost of this turn
    wall_s: float


class Session(Protocol):
    def send(self, content: list[dict]) -> Reply: ...

    def close(self) -> None: ...


class Backend(Protocol):
    model: str

    def open(self, system: str) -> Session: ...


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_LEADING_DOT = re.compile(r"([:\[,]\s*-?)\.(\d)")


def extract_json(text: str) -> dict:
    """Return the last top-level JSON object in `text` (fenced or bare). Raises ValueError."""
    text = _LEADING_DOT.sub(r"\g<1>0.\g<2>", text)  # models sometimes write .5 for 0.5
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    for chunk in reversed(candidates):
        obj = _last_object(chunk)
        if obj is not None:
            return obj
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


def _last_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    found: dict | None = None
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            return found
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            found = obj
        i = end
