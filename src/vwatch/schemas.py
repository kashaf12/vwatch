"""Shared data contracts. Every boundary (truth files, model output, run logs) is validated here.

All models are frozen: build new objects instead of mutating.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- tasks & arms


class Task(StrEnum):
    DEFECTS = "T1"  # defect detection on Shorts
    EVENTS = "T2"  # short-event recall
    CAUSAL = "T3"  # "because" retell
    QA = "T4"  # targeted QA
    MCQ = "T5"  # public benchmark (TVBench) multiple choice
    FREE = "free"  # ad-hoc question from the CLI (not scored)


class Arm(StrEnum):
    B0 = "B0"  # uniform frames, no audio
    B1 = "B1"  # uniform frames + transcript (main baseline)
    B2 = "B2"  # B1 at 2x budget (set by runner, same code as B1)
    B1H = "B1H"  # B1 + the same low-level signal hints the watcher gets (method-only baseline)
    W = "W"  # watcher
    W_NOQ = "W-noQ"  # watcher without the question-first plan
    W_NOP = "W-noP"  # watcher without model prediction-error (signals only)
    W_NOCROP = "W-noCrop"  # watcher without crops
    W_NOS = "W-noS"  # watcher without low-level signal hints (attention from the model alone)
    CC = "CC"  # plain Claude Code agent with Bash + Read: extracts frames itself (how CC watches today)
    CVV = "CVV"  # Claude Code + the claude-video-vision plugin (third party)
    POP = "POP"  # Claude Code + the popcorn MCP server (third party)
    VWP = "VWP"  # Claude Code + the vwatch plugin (our MCP tools + watch-video skill)


# ---------------------------------------------------------------- ground truth


class DefectType(StrEnum):
    FLASH_FRAME = "flash_frame"  # 1-3 foreign frames
    FROZEN_FRAME = "frozen_frame"  # picture stops while audio continues
    JUMP_CUT = "jump_cut"  # a chunk of audio+video removed mid-sentence
    AUDIO_DROPOUT = "audio_dropout"  # sound goes silent mid-speech, picture continues
    CAPTION_BLOCKED = "caption_blocked"  # caption band covered by an opaque box
    WRONG_NUMBER = "wrong_number"  # on-screen figure contradicts the spoken one


DEFECT_DEFINITIONS: dict[DefectType, str] = {
    DefectType.FLASH_FRAME: "1-3 frames (under 0.1 s) of an unrelated or black image flash on screen.",
    DefectType.FROZEN_FRAME: "the picture freezes (no movement at all, captions stop updating) while narration continues.",
    DefectType.JUMP_CUT: "a short piece of the video and audio is missing: the picture jumps and a word is clipped.",
    DefectType.AUDIO_DROPOUT: "the narration goes silent in the middle of speech while captions keep showing words.",
    DefectType.CAPTION_BLOCKED: "an opaque box covers the subtitle/caption area so the caption can't be read.",
    DefectType.WRONG_NUMBER: "a number shown on screen (e.g. a rupee amount) contradicts the number being spoken or shown in captions at that moment.",
}


class TruthDefect(Frozen):
    type: DefectType
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    detail: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> TruthDefect:
        if self.end < self.start:
            raise ValueError(f"end {self.end} < start {self.start}")
        return self


class DefectTruth(Frozen):
    video_id: str
    source: str
    duration: float = Field(gt=0)
    defects: tuple[TruthDefect, ...] = ()  # empty = clean video


class TruthEvent(Frozen):
    id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    kind: Literal["appear", "color_change", "move", "flash"]
    subject: str  # e.g. "green star"
    keywords: tuple[str, ...]  # all must appear (case-insensitive) in a description to match


class EventTruth(Frozen):
    video_id: str
    duration: float = Field(gt=0)
    events: tuple[TruthEvent, ...]


class CausalLink(Frozen):
    id: str
    cause: str
    effect: str
    time: float = Field(ge=0)


class CausalTruth(Frozen):
    video_id: str
    duration: float = Field(gt=0)
    links: tuple[CausalLink, ...]
    distractors: tuple[str, ...] = ()  # things that happen but cause nothing


class QAItem(Frozen):
    id: str
    question: str
    answer: str
    answer_type: Literal["number", "text"]
    time: float | None = None


class QATruth(Frozen):
    video_id: str
    duration: float = Field(gt=0)
    items: tuple[QAItem, ...]


class MCQItem(Frozen):
    id: str
    question: str
    options: tuple[str, ...] = Field(min_length=2)
    answer_index: int = Field(ge=0)
    category: str = ""

    @model_validator(mode="after")
    def _answer_in_range(self) -> MCQItem:
        if self.answer_index >= len(self.options):
            raise ValueError(f"answer_index {self.answer_index} out of range for {len(self.options)} options")
        return self


class MCQTruth(Frozen):
    video_id: str
    duration: float = Field(gt=0)
    source: str = ""  # benchmark + task, e.g. "TVBench/action_count"
    items: tuple[MCQItem, ...]


class DatasetItem(Frozen):
    video_id: str
    task: Task
    video_path: str  # relative to the dataset root
    truth_path: str  # relative to the dataset root
    split: Literal["pilot", "test"]
    bucket: str = ""  # free label, e.g. "clean" / "defect"


# ---------------------------------------------------------------- model outputs


class ReportedDefect(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str
    start: float
    end: float | None = None
    evidence: str = ""

    @field_validator("type")
    @classmethod
    def _norm_type(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_").replace("-", "_")


class DefectReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    defects: tuple[ReportedDefect, ...] = ()


class ReportedEvent(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    start: float
    end: float | None = None
    description: str


class EventReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    events: tuple[ReportedEvent, ...] = ()


class CausalReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    retell: tuple[str, ...] = ()


class QAAnswer(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str
    answer: str


class QAReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    answers: tuple[QAAnswer, ...] = ()


class Moment(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    t: float
    what: str = ""


class FreeReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    answer: str
    moments: tuple[Moment, ...] = ()


class MCQAnswer(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str
    choice: str  # option letter, "A", "B", ...


class MCQReport(Frozen):
    model_config = ConfigDict(frozen=True, extra="ignore")
    answers: tuple[MCQAnswer, ...] = ()


REPORT_MODEL: dict[Task, type[BaseModel]] = {
    Task.MCQ: MCQReport,
    Task.FREE: FreeReport,
    Task.DEFECTS: DefectReport,
    Task.EVENTS: EventReport,
    Task.CAUSAL: CausalReport,
    Task.QA: QAReport,
}


# ---------------------------------------------------------------- run log


class Observation(Frozen):
    """Evidence the agent actually received. Used for the unseen-frame-claims check."""

    kind: Literal["frame", "crop", "audio", "transcript", "signal"]
    t0: float
    t1: float


class Usage(Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def plus(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


class RunRecord(Frozen):
    run_id: str
    arm: Arm
    budget_tokens: int
    repeat: int
    video_id: str
    task: Task
    model: str
    raw_output: str = ""
    parsed: dict | None = None
    parse_error: str | None = None
    content_tokens_est: int = 0  # tokens of evidence we sent (images + text), the budgeted quantity
    usage: Usage = Usage()
    cost_usd: float = 0.0
    wall_s: float = 0.0
    turns: int = 0
    observations: tuple[Observation, ...] = ()
    tool_calls: tuple[dict, ...] = ()  # external agent arms only
    error: str | None = None
