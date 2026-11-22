"""T5 (public multiple-choice benchmark) scoring: letter normalisation, per-item correctness, chance level."""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import ascii_uppercase

from vwatch.schemas import MCQReport, MCQTruth

# optional "option/choice/answer" word, optional opening bracket, ONE letter, then a delimiter or the end:
#   "B", "b", "(B)", "B) text", "B. text", "Answer: B", "option b", "**B**"
# "A man walks" is NOT a letter answer (the letter is followed by a space and a word).
_CHOICE_RE = re.compile(
    r"""^[\s"'`*_]*(?:(?:the\s+)?(?:final\s+)?(?:option|choice|answer)\s*(?:is)?\s*[:\-]?\s*)?
        [\(\[]?\s*([A-Za-z])\s*(?:[\)\]\.:,;\-–—*_"'`!]|$)""",
    re.IGNORECASE | re.VERBOSE,
)


def normalize_choice(raw: object, n_options: int) -> int | None:
    """Option index for a model's choice string, None if it is not a valid letter within range."""
    if not isinstance(raw, str):
        return None
    m = _CHOICE_RE.match(raw)
    if not m:
        return None
    idx = ascii_uppercase.find(m.group(1).upper())
    return idx if 0 <= idx < n_options else None


def option_letter(index: int) -> str:
    return ascii_uppercase[index]


@dataclass(frozen=True)
class MCQRow:
    id: str
    category: str
    correct: bool
    answered: bool  # a valid letter was given
    n_options: int


@dataclass(frozen=True)
class MCQScore:
    correct: int
    total: int
    chance: float  # expected number correct when guessing uniformly: sum(1/len(options))
    rows: tuple[MCQRow, ...]


def score_mcq(truth: MCQTruth, report: MCQReport | None) -> MCQScore:
    """Per-item correctness. Missing, duplicate-after-first, invalid or out-of-range choices are wrong."""
    answers: dict[str, str] = {}
    for a in report.answers if report else ():
        answers.setdefault(a.id, a.choice)
    rows = []
    for item in truth.items:
        got = normalize_choice(answers.get(item.id), len(item.options))
        rows.append(MCQRow(item.id, item.category, got == item.answer_index, got is not None, len(item.options)))
    return MCQScore(
        correct=sum(r.correct for r in rows),
        total=len(rows),
        chance=sum(1 / r.n_options for r in rows),
        rows=tuple(rows),
    )
