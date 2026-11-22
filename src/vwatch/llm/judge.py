"""LLM judge for the causal retell task (T3). One fresh session per judgement."""

from __future__ import annotations

from collections.abc import Callable

from vwatch.llm import default_backend, text_block

JUDGE_SYSTEM = ("You grade answers strictly against a gold list. Output only the JSON object "
                "requested, no prose.")


def make_judge(model: str = "claude-sonnet-5-5") -> Callable[[str], str]:
    backend = default_backend(model=model, effort="low")

    def judge(prompt: str) -> str:
        session = backend.open(JUDGE_SYSTEM)
        try:
            return session.send([text_block(prompt)]).text
        finally:
            session.close()

    return judge
