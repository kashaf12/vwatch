"""Backend using the official Anthropic Python SDK (`pip install vwatch[api]`).

Picked automatically when ANTHROPIC_API_KEY is set. The session keeps the message
history client-side and re-sends it each turn, with automatic prompt caching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vwatch.llm.base import DEFAULT_MODEL, LLMError, Reply
from vwatch.schemas import Usage

# $ per million tokens: (input, output). Cache reads/writes priced from input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.0, 40.0),
}


def turn_cost(model: str, u: Usage) -> float:
    pin, pout = PRICES.get(model, (4.0, 20.0))
    return (u.input_tokens * pin + u.cache_creation_input_tokens * pin * 1.25
            + u.cache_read_input_tokens * pin * 0.1 + u.output_tokens * pout) / 1e6


@dataclass
class AnthropicSession:
    client: object
    model: str
    system: str
    effort: str | None
    messages: list[dict] = field(default_factory=list)

    def send(self, content: list[dict]) -> Reply:
        import anthropic

        history = [*self.messages, {"role": "user", "content": content}]
        kwargs: dict = {"model": self.model, "max_tokens": 16000, "system": self.system,
                        "messages": history, "cache_control": {"type": "ephemeral"}}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.monotonic()
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"API connection error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        usage = Usage(input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                      cache_read_input_tokens=u.cache_read_input_tokens or 0,
                      cache_creation_input_tokens=u.cache_creation_input_tokens or 0)
        self.messages = [*history, {"role": "assistant", "content": resp.content}]
        return Reply(text=text, usage=usage, cost_usd=turn_cost(self.model, usage),
                     wall_s=time.monotonic() - t0)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class AnthropicBackend:
    model: str = DEFAULT_MODEL
    effort: str | None = "medium"

    def open(self, system: str) -> AnthropicSession:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("anthropic SDK not installed: uv sync --extra api") from exc
        return AnthropicSession(client=anthropic.Anthropic(), model=self.model,
                                system=system, effort=self.effort)
