"""Pure scoring functions for T1-T4 plus the unseen-claims check."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vwatch.schemas import (
    REPORT_MODEL,
    CausalReport,
    CausalTruth,
    DefectReport,
    DefectTruth,
    DefectType,
    EventReport,
    EventTruth,
    QAReport,
    QATruth,
    RunRecord,
    Task,
)

_EPS = 1e-9
_DEFECT_PAD = 0.5
OVERLAP_LEN_FACTOR = 2.0
OVERLAP_LEN_SLACK = 1.0  # seconds
UNSEEN_TOL = 0.5
SEEN_KINDS = frozenset({"frame", "crop", "audio"})
EVENT_BUCKETS = ("<0.5s", "0.5-1s", "1-3s", ">3s")
SHORT_BUCKETS = ("<0.5s", "0.5-1s")
_VALID_TYPES = frozenset(t.value for t in DefectType)


# ---------------------------------------------------------------- shared helpers


def _norm_type(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _greedy(cands: Sequence[tuple[float, int, int]]) -> dict[int, tuple[int, float]]:
    """One-to-one greedy matching, smallest error first. Returns truth_idx -> (rep_idx, err)."""
    used_r: set[int] = set()
    out: dict[int, tuple[int, float]] = {}
    for err, ti, rj in sorted(cands):
        if ti in out or rj in used_r:
            continue
        out[ti] = (rj, err)
        used_r.add(rj)
    return out


def _end(start: float, end: float | None) -> float:
    return end if end is not None and end >= start else start


# ---------------------------------------------------------------- T1


@dataclass(frozen=True)
class DefectRow:
    index: int
    type: str
    start: float
    end: float
    found: bool
    start_error: float | None
    loc_hit: bool


@dataclass(frozen=True)
class DefectScore:
    tp: int
    fp: int
    fn: int
    rows: tuple[DefectRow, ...]
    false_alarm: bool
    n_reported: int
    unknown_types: int


def score_defects(
    truth: DefectTruth,
    report: DefectReport | None,
    *,
    match_tol: float = 1.5,
    loc_tol: float = 0.5,
) -> DefectScore:
    if match_tol < 0 or loc_tol < 0:
        raise ValueError("tolerances must be >= 0")
    reps = report.defects if report else ()
    cands: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth.defects):
        for rj, r in enumerate(reps):
            if _norm_type(r.type) != t.type.value:
                continue
            err = abs(r.start - t.start)
            r_end = _end(r.start, r.end)
            # Overlap only counts for a reported interval that is not much longer than
            # the truth, otherwise a whole-video interval would "hit" every defect.
            short_enough = (r_end - r.start) <= OVERLAP_LEN_FACTOR * (t.end - t.start) + OVERLAP_LEN_SLACK + _EPS
            overlap = (
                short_enough
                and r.start - _DEFECT_PAD <= t.end + _DEFECT_PAD + _EPS
                and t.start - _DEFECT_PAD <= r_end + _DEFECT_PAD + _EPS
            )
            if err <= match_tol + _EPS or overlap:
                cands.append((err, ti, rj))
    matches = _greedy(cands)
    rows = []
    for ti, t in enumerate(truth.defects):
        hit = matches.get(ti)
        rows.append(
            DefectRow(
                index=ti,
                type=t.type.value,
                start=t.start,
                end=t.end,
                found=hit is not None,
                start_error=hit[1] if hit else None,
                loc_hit=bool(hit) and hit[1] <= loc_tol + _EPS,
            )
        )
    tp = len(matches)
    return DefectScore(
        tp=tp,
        fp=len(reps) - tp,
        fn=len(truth.defects) - tp,
        rows=tuple(rows),
        false_alarm=not truth.defects and len(reps) >= 1,
        n_reported=len(reps),
        unknown_types=sum(1 for r in reps if _norm_type(r.type) not in _VALID_TYPES),
    )


def failed_defect_score(truth: DefectTruth) -> DefectScore:
    """Score for a run that produced no usable answer (error / unparsable output).

    Defect video: every truth defect is a miss. Clean video: counted as ONE false alarm
    (conservative; the same rule for every arm), never as a correct "no alarm".
    """
    base = score_defects(truth, None)
    if truth.defects:
        return base
    return DefectScore(tp=0, fp=1, fn=0, rows=(), false_alarm=True, n_reported=0, unknown_types=0)


# ---------------------------------------------------------------- T2


def duration_bucket(duration: float) -> str:
    d = round(duration, 6)
    if d < 0.5:
        return "<0.5s"
    if d < 1.0:
        return "0.5-1s"
    if d <= 3.0:
        return "1-3s"
    return ">3s"


def _keyword_forms(kw: str) -> set[str]:
    k = kw.strip().lower().replace("-", " ")
    forms = {k}
    if k.endswith("ies"):
        forms.add(k[:-3] + "y")
    elif k.endswith("es"):
        forms.update({k[:-2], k[:-1]})
    elif k.endswith("s"):
        forms.add(k[:-1])
    else:
        forms.update({k + "s", k + "es"})
        if k.endswith("y"):
            forms.add(k[:-1] + "ies")
    return forms


def keyword_in(description: str, kw: str) -> bool:
    """Case-insensitive match allowing plural and hyphen/space/joined variants."""
    for form in _keyword_forms(kw):
        words = [re.escape(w) for w in form.split()]
        if not words:
            continue
        if re.search(r"(?<![a-z0-9])" + r"[\s\-]*".join(words) + r"(?![a-z0-9])", description, re.I):
            return True
    return False


@dataclass(frozen=True)
class EventRow:
    index: int
    id: str
    bucket: str
    start: float
    found: bool


@dataclass(frozen=True)
class EventScore:
    tp: int
    fp: int
    fn: int
    rows: tuple[EventRow, ...]


def score_events(truth: EventTruth, report: EventReport | None, *, tol: float = 1.0) -> EventScore:
    if tol < 0:
        raise ValueError("tol must be >= 0")
    reps = report.events if report else ()
    cands: list[tuple[float, int, int]] = []
    for ti, t in enumerate(truth.events):
        for rj, r in enumerate(reps):
            if not all(keyword_in(r.description, kw) for kw in t.keywords):
                continue
            err = abs(r.start - t.start)
            r_end = _end(r.start, r.end)
            overlap = r.start <= t.end + tol + _EPS and r_end >= t.start - tol - _EPS
            if err <= tol + _EPS or overlap:
                cands.append((err, ti, rj))
    matches = _greedy(cands)
    rows = tuple(
        EventRow(ti, t.id, duration_bucket(t.end - t.start), t.start, ti in matches)
        for ti, t in enumerate(truth.events)
    )
    tp = len(matches)
    return EventScore(tp=tp, fp=len(reps) - tp, fn=len(truth.events) - tp, rows=rows)


# ---------------------------------------------------------------- T4

_WORD_NUMS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUM_RE = re.compile(
    r"(?<!\w)-?\d[\d,]*(?:\.\d+)?|\b(?:" + "|".join(_WORD_NUMS) + r")\b", re.I
)


def _num_value(tok: str) -> float:
    tok = tok.lower()
    if tok in _WORD_NUMS:
        return float(_WORD_NUMS[tok])
    return float(tok.replace(",", ""))


def extract_numbers(text: str) -> list[float]:
    """All numbers in text, in order: digits (commas allowed) or words zero..twenty."""
    return [_num_value(m.group(0)) for m in _NUM_RE.finditer(text)]


def extract_number(text: str) -> float | None:
    """First number in text, or None."""
    nums = extract_numbers(text)
    return nums[0] if nums else None


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _contains_seq(hay: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


_ALT_RE = re.compile(r"\bor\b|/", re.I)
MAX_TEXT_REPLY_TOKENS = 3


def _text_correct(reply: str, truth: str) -> bool:
    """Exact normalized match, or a short (<=3 tokens) reply containing the truth with no alternatives."""
    r, t = _tokens(reply), _tokens(truth)
    if not t:
        return False
    if r == t:
        return True
    return len(r) <= MAX_TEXT_REPLY_TOKENS and _contains_seq(r, t) and not _ALT_RE.search(reply)


@dataclass(frozen=True)
class QARow:
    id: str
    correct: bool
    answered: bool


@dataclass(frozen=True)
class QAScore:
    correct: int
    total: int
    rows: tuple[QARow, ...]


def score_qa(truth: QATruth, report: QAReport | None) -> QAScore:
    answers: dict[str, str] = {}
    for a in report.answers if report else ():
        answers.setdefault(a.id, a.answer)
    rows = []
    for item in truth.items:
        reply = answers.get(item.id)
        ok = False
        if reply is not None:
            if item.answer_type == "number":
                want = extract_number(item.answer)
                if want is None:
                    raise ValueError(f"QA item {item.id}: truth answer {item.answer!r} has no number")
                got = {round(x, 9) for x in extract_numbers(reply)}
                # a hedged reply naming several numbers is wrong
                ok = len(got) == 1 and abs(next(iter(got)) - want) < _EPS
            else:
                ok = _text_correct(reply, item.answer)
        rows.append(QARow(item.id, ok, reply is not None))
    return QAScore(correct=sum(r.correct for r in rows), total=len(rows), rows=tuple(rows))


# ---------------------------------------------------------------- T3


class JudgeError(ValueError):
    """The judge output could not be parsed or validated."""


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    matched: list[str]
    invented: int = Field(ge=0)


@dataclass(frozen=True)
class CausalScore:
    matched: tuple[str, ...]
    n_links: int
    recall: float
    invented: int


def build_causal_prompt(truth: CausalTruth, report: CausalReport) -> str:
    links = "\n".join(f"{l.id}: {l.cause} -> {l.effect}" for l in truth.links)
    distract = "\n".join(f"D{i}: {d}" for i, d in enumerate(truth.distractors, 1)) or "(none)"
    retell = "\n".join(f"{i}. {s}" for i, s in enumerate(report.retell, 1))
    return (
        "You are grading a retelling of a video against a gold list of causal links.\n\n"
        "GOLD CAUSAL LINKS (id: cause -> effect):\n"
        f"{links}\n\n"
        "DISTRACTORS (things that happen in the video but cause nothing):\n"
        f"{distract}\n\n"
        "RETELLING (one sentence per line):\n"
        f"{retell}\n\n"
        "RUBRIC:\n"
        "- A gold link is matched ONLY if the retelling states that the cause led to the effect "
        "(paraphrase is fine). Merely listing both events without the causal relation does not count.\n"
        "- invented = the number of causal claims in the retelling that are not supported by any gold "
        "link. This includes any claim that makes a distractor a cause or an effect, and any claim of "
        "causation between events with no gold link.\n"
        "- Count each distinct unsupported causal claim once.\n\n"
        'Reply with strict JSON only, no prose: {"matched": ["L1", ...], "invented": <integer>}'
    )


def parse_judge_output(text: str, valid_ids: Sequence[str]) -> JudgeVerdict:
    decoder = json.JSONDecoder()
    obj = None
    for m in re.finditer(r"\{", text):
        try:
            cand, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict):
            obj = cand
            break
    if obj is None:
        raise JudgeError(f"judge output contains no JSON object: {text[:200]!r}")
    try:
        verdict = JudgeVerdict.model_validate(obj)
    except ValidationError as e:
        raise JudgeError(f"judge JSON failed validation: {e}") from e
    # Ids are matched case-insensitively (the prompt's example says "L1"; gold ids may be "l1").
    canon = {v.strip().lower(): v for v in valid_ids}
    unknown = sorted(m for m in verdict.matched if m.strip().lower() not in canon)
    if unknown:
        raise JudgeError(f"judge matched unknown link ids {unknown}; valid: {list(valid_ids)}")
    matched = list(dict.fromkeys(canon[m.strip().lower()] for m in verdict.matched))
    return verdict.model_copy(update={"matched": matched})


def score_causal(
    truth: CausalTruth,
    report: CausalReport | None,
    judge: Callable[[str], str],
) -> CausalScore:
    n = len(truth.links)
    if report is None or not any(s.strip() for s in report.retell):
        return CausalScore((), n, 0.0, 0)  # empty answer: nothing to judge
    ids = [l.id for l in truth.links]
    verdict = parse_judge_output(judge(build_causal_prompt(truth, report)), ids)
    matched = tuple(i for i in ids if i in set(verdict.matched))
    return CausalScore(matched, n, len(matched) / n if n else 0.0, verdict.invented)


# ---------------------------------------------------------------- run-level helpers


def parse_report(record: RunRecord):
    """Validate record.parsed against its task's report model; None if absent or invalid."""
    if record.parsed is None:
        return None
    try:
        return REPORT_MODEL[record.task].model_validate(record.parsed)
    except ValidationError:
        return None


def unseen_claims(record: RunRecord, *, tol: float = UNSEEN_TOL) -> int:
    """T1/T2 reported items whose start is not within tol of a frame/crop/audio observation.

    Signal hints and transcript text do not count as having "seen" the moment: a claim
    supported only by those is unverified and counted here.
    """
    if record.task not in (Task.DEFECTS, Task.EVENTS):
        return 0
    rep = parse_report(record)
    if rep is None:
        return 0
    items = rep.defects if record.task == Task.DEFECTS else rep.events
    spans = [(o.t0, o.t1) for o in record.observations if o.kind in SEEN_KINDS]
    return sum(1 for it in items if not any(t0 - tol <= it.start <= t1 + tol for t0, t1 in spans))
