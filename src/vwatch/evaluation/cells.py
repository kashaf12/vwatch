"""Per-record scoring and per-(arm, budget) aggregate cells."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from vwatch.evaluation.runs import is_failed
from vwatch.evaluation.scoring import (
    EVENT_BUCKETS,
    SHORT_BUCKETS,
    CausalScore,
    DefectScore,
    EventScore,
    QAScore,
    failed_defect_score,
    parse_report,
    score_causal,
    score_defects,
    score_events,
    score_qa,
    unseen_claims,
)
from vwatch.evaluation.scoring_mcq import MCQScore, score_mcq
from vwatch.evaluation.stats import f1
from vwatch.schemas import RunRecord, Task

Judge = Callable[[str], str]
Truths = Mapping[tuple[str, Task], object]


@dataclass(frozen=True)
class RecScore:
    run: RunRecord
    duration: float
    failed: bool
    cluster: str = ""  # T1: truth source clip; otherwise the video itself
    t1: DefectScore | None = None
    t2: EventScore | None = None
    t3: CausalScore | None = None
    t4: QAScore | None = None
    t5: MCQScore | None = None
    unseen: int = 0


def score_records(runs: Sequence[RunRecord], truths: Truths, judge: Judge | None) -> list[RecScore]:
    out = []
    for r in runs:
        truth = truths.get((r.video_id, r.task))
        if truth is None:
            raise ValueError(f"run {r.run_id}: no truth for video {r.video_id!r} task {r.task.value}")
        rep = parse_report(r)
        failed = rep is None or bool(r.parse_error) or is_failed(r)
        kw: dict = {}
        cluster = r.video_id
        if r.task == Task.DEFECTS:
            cluster = truth.source
            # a failed run is never a correct "no alarm" on a clean video
            t1 = failed_defect_score(truth) if failed else score_defects(truth, rep)
            kw = {"t1": t1, "unseen": unseen_claims(r)}
        elif r.task == Task.EVENTS:
            kw = {"t2": score_events(truth, rep), "unseen": unseen_claims(r)}
        elif r.task == Task.CAUSAL and judge is not None:
            kw = {"t3": score_causal(truth, rep, judge)}
        elif r.task == Task.QA:
            kw = {"t4": score_qa(truth, rep)}
        elif r.task == Task.MCQ:  # a failed run answers nothing: every item wrong
            kw = {"t5": score_mcq(truth, None if failed else rep)}
        out.append(RecScore(r, truth.duration, failed, cluster, **kw))
    return out


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def _t1(recs: list[RecScore]) -> dict | None:
    t1 = [r for r in recs if r.t1]
    if not t1:
        return None
    tp, fp, fn = (sum(getattr(r.t1, k) for r in t1) for k in ("tp", "fp", "fn"))
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    found = loc = n_truth = 0
    for r in t1:
        for row in r.t1.rows:
            by_type[row.type][1] += 1
            by_type[row.type][0] += row.found
            found += row.found
            loc += row.loc_hit
            n_truth += 1
    clean = [r for r in t1 if not r.t1.rows]
    per_repeat = defaultdict(lambda: [0, 0, 0])
    per_source = defaultdict(lambda: [0, 0, 0])
    for r in t1:
        for acc in (per_repeat[r.run.repeat], per_source[r.cluster]):
            acc[0] += r.t1.tp
            acc[1] += r.t1.fp
            acc[2] += r.t1.fn
    rep_f1 = [f1(*v) for v in per_repeat.values()]
    strict = _rate(loc, n_truth)  # found within +-0.5 s / ALL truth defects (a miss is a failure)
    return {
        "n": len(t1),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "f1": f1(tp, fp, fn),
        "f1_by_source": {k: f1(*v) for k, v in sorted(per_source.items())},
        "recall_by_type": {k: _rate(v[0], v[1]) for k, v in sorted(by_type.items())},
        "loc_rate": strict,
        "strict_recall": strict,
        "loc_given_found": _rate(loc, found),
        "false_alarm_rate": _rate(sum(r.t1.false_alarm for r in clean), len(clean)),
        "n_clean": len(clean),
        "f1_std_repeats": float(np.std(rep_f1, ddof=1)) if len(rep_f1) > 1 else None,
    }


def _t2(recs: list[RecScore]) -> dict | None:
    t2 = [r for r in recs if r.t2]
    if not t2:
        return None
    cnt = {b: [0, 0] for b in EVENT_BUCKETS}
    for r in t2:
        for row in r.t2.rows:
            cnt[row.bucket][1] += 1
            cnt[row.bucket][0] += row.found
    short = [sum(cnt[b][i] for b in SHORT_BUCKETS) for i in (0, 1)]
    allc = [sum(v[i] for v in cnt.values()) for i in (0, 1)]
    tp, fp = sum(r.t2.tp for r in t2), sum(r.t2.fp for r in t2)
    return {
        "n": len(t2),
        "by_bucket": {b: {"found": v[0], "total": v[1], "recall": _rate(*v)} for b, v in cnt.items()},
        "short_recall": _rate(*short),
        "recall": _rate(*allc),
        "fp": fp,
        "precision": _rate(tp, tp + fp),
        "mean_listed": (tp + fp) / len(t2),
    }


def _t3(recs: list[RecScore]) -> dict | None:
    t3 = [r for r in recs if r.t3]
    if not t3:
        return None
    m = sum(len(r.t3.matched) for r in t3)
    n = sum(r.t3.n_links for r in t3)
    return {"n": len(t3), "recall": _rate(m, n), "invented_total": sum(r.t3.invented for r in t3),
            "invented_mean": sum(r.t3.invented for r in t3) / len(t3)}


def _t4(recs: list[RecScore]) -> dict | None:
    t4 = [r for r in recs if r.t4]
    if not t4:
        return None
    return {"n": len(t4), "accuracy": _rate(sum(r.t4.correct for r in t4), sum(r.t4.total for r in t4))}


def _t5(recs: list[RecScore]) -> dict | None:
    """Public-benchmark accuracy, chance level (mean 1/len(options)) and per-category breakdown."""
    t5 = [r for r in recs if r.t5]
    if not t5:
        return None
    cat: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])  # correct, total, chance
    for r in t5:
        for row in r.t5.rows:
            c = cat[row.category or "all"]
            c[0] += row.correct
            c[1] += 1
            c[2] += 1 / row.n_options
    n = sum(v[1] for v in cat.values())
    return {
        "n_videos": len({r.run.video_id for r in t5}),
        "n_items": n,
        "accuracy": _rate(sum(v[0] for v in cat.values()), n),
        "chance": _rate(sum(v[2] for v in cat.values()), n),
        "by_category": {k: {"correct": v[0], "total": v[1], "accuracy": _rate(v[0], v[1]),
                            "chance": _rate(v[2], v[1])} for k, v in sorted(cat.items())},
    }


def make_cell(arm: str, budget: int, recs: list[RecScore]) -> dict:
    # per-minute cost is over the core tasks only: T5 clips are short public-benchmark videos and
    # would skew S6 (a cell holding only T5 records, e.g. an external arm, falls back to all)
    core = [r for r in recs if r.t5 is None and r.run.task != Task.MCQ] or recs
    minutes = sum(r.duration for r in core) / 60
    n = len(recs)
    t1 = [r for r in recs if r.t1]
    t2 = [r for r in recs if r.t2]
    failed = sum(r.failed for r in recs)
    return {
        "arm": arm,
        "budget": budget,
        "n_records": n,
        "n_videos": len({r.run.video_id for r in recs}),
        "repeats": len({r.run.repeat for r in recs}),
        "models": sorted({r.run.model for r in recs}),
        "t1": _t1(recs),
        "t2": _t2(recs),
        "t3": _t3(recs),
        "t4": _t4(recs),
        "t5": _t5(recs),
        "mean_input_tokens": sum(r.run.usage.total_input for r in recs) / n,
        "mean_content_tokens": sum(r.run.content_tokens_est for r in recs) / n,
        "usd_per_min": _rate(sum(r.run.cost_usd for r in core), minutes),
        "wall_s_per_min": _rate(sum(r.run.wall_s for r in core), minutes),
        "unseen_t1": sum(r.unseen for r in t1) if t1 else None,
        "unseen_t2": sum(r.unseen for r in t2) if t2 else None,
        "parse_errors": failed,
        "failure_rate": failed / n,
    }
