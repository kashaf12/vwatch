"""Task prompts and answer parsing. Every arm gets the same task text and the same answer schema."""

from __future__ import annotations

from pydantic import ValidationError

from vwatch.llm.base import extract_json
from vwatch.schemas import DEFECT_DEFINITIONS, REPORT_MODEL, MCQTruth, QATruth, Task

_DEFECT_LIST = "\n".join(f"- {t.value}: {d}" for t, d in DEFECT_DEFINITIONS.items())

_T1 = f"""TASK: You are QA-checking a short vertical video (a Hindi YouTube Short) before it is uploaded.
Find every production defect of these types:
{_DEFECT_LIST}
NOT defects: normal scene changes, natural pauses between sentences, motion graphics that hold still while
the narrator talks, normal caption changes. Only report something you have evidence for.
Answer with JSON only:
{{"defects": [{{"type": "<one of: {', '.join(t.value for t in DEFECT_DEFINITIONS)}>", "start": <seconds>, "end": <seconds>, "evidence": "<what you saw/heard>"}}]}}
Use an empty list if the video has no defects."""

_T2 = """TASK: List every distinct visual event in this video: a shape appearing (and later disappearing), an existing
shape changing colour, a shape moving across the screen, or the whole screen flashing a colour.
Ignore the slow continuous drift of background elements. Describe each event with basic colour words
(red, green, blue, yellow, orange, purple, pink, white, black, brown) and basic shape words
(circle, square, triangle, star, diamond, heart, ring, cross, arrow); for a whole-screen flash write
"<colour> flash".
Answer with JSON only:
{"events": [{"start": <seconds>, "end": <seconds>, "description": "<e.g. green star appears>"}]}"""

_T3 = """TASK: Retell what happens in this video as a chain of cause and effect. Write one sentence per link and
put the word "because" in every sentence (e.g. "The box fell because the ball hit it."). Only state
causal links you actually observed; things that happen without causing anything should not be described
as causes. Name objects with basic colour and object words.
Answer with JSON only:
{"retell": ["<sentence>", "..."]}"""

_T4_HEAD = """TASK: Answer these questions about the video. Answers must be short: plain digits for numbers (e.g. "80"),
lowercase basic words for text.
Questions:
{questions}
Answer with JSON only:
{{"answers": [{{"id": "<question id>", "answer": "<short answer>"}}]}}"""

_T5_HEAD = """TASK: Answer these multiple-choice questions about the video. Exactly one option per question is correct.
Questions:
{questions}
Answer with letters only (no option text). JSON only:
{{"answers": [{{"id": "<question id>", "choice": "<letter>"}}]}}"""


def mcq_questions(truth: MCQTruth) -> str:
    """Question ids, text and lettered options; the answer index is never read."""
    blocks = []
    for q in truth.items:
        opts = "\n".join(f"  {chr(ord('A') + i)}. {o}" for i, o in enumerate(q.options))
        blocks.append(f"- {q.id}: {q.question}\n{opts}")
    return "\n".join(blocks)


def free_prompt(question: str) -> str:
    q = question.strip()
    if not q:
        raise ValueError("question is empty")
    return (f"TASK: {q}\nAnswer with JSON only: "
            '{"answer": "<your answer>", "moments": [{"t": <seconds>, "what": "<what happens there>"}]}')


def task_prompt(task: Task, truth: object | None = None) -> str:
    if task is Task.DEFECTS:
        return _T1
    if task is Task.EVENTS:
        return _T2
    if task is Task.CAUSAL:
        return _T3
    if task is Task.QA:
        if not isinstance(truth, QATruth):
            raise ValueError("T4 needs the QATruth to read its questions (answers are not shown)")
        qs = "\n".join(f"- {q.id}: {q.question}" for q in truth.items)
        return _T4_HEAD.format(questions=qs)
    if task is Task.MCQ:
        if not isinstance(truth, MCQTruth):
            raise ValueError("T5 needs the MCQTruth to read its questions (answers are not shown)")
        return _T5_HEAD.format(questions=mcq_questions(truth))
    raise ValueError(f"unknown task {task}")


def parse_report(task: Task, text: str) -> tuple[dict | None, str | None]:
    """(parsed dict, None) on success, (None, error message) on failure."""
    try:
        obj = extract_json(text)
    except ValueError as exc:
        return None, str(exc)
    if "final" in obj and isinstance(obj["final"], dict):
        obj = obj["final"]
    try:
        model = REPORT_MODEL[task].model_validate(obj)
    except ValidationError as exc:
        return None, f"schema mismatch: {exc.errors()[:3]}"
    return model.model_dump(mode="json"), None


REPAIR_PROMPT = ("Your last reply was not valid JSON in the required schema ({error}). "
                 "Reply again with only the JSON object, nothing else.")
