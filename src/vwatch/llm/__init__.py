"""Model backends. `default_backend()` picks the SDK when an API key exists, else the Claude Code CLI."""

from __future__ import annotations

import os

from vwatch.llm.base import DEFAULT_MODEL, Backend, LLMError, Reply, Session, extract_json, text_block


def default_backend(model: str = DEFAULT_MODEL, effort: str | None = "medium") -> Backend:
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("VWATCH_BACKEND", "") != "cli":
        from vwatch.llm.anthropic_api import AnthropicBackend

        return AnthropicBackend(model=model, effort=effort)
    from vwatch.llm.claude_cli import ClaudeCLIBackend

    return ClaudeCLIBackend(model=model, effort=effort)


__all__ = ["DEFAULT_MODEL", "Backend", "LLMError", "Reply", "Session", "default_backend",
           "extract_json", "text_block"]
