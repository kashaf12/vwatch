"""Paired, cluster-bootstrapped comparisons between two arms' scored records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping

from vwatch.evaluation.cells import RecScore
from vwatch.evaluation.scoring import SHORT_BUCKETS
from vwatch.evaluation.stats import (
    Estimate,
    f1,
    majority_found,
    mcnemar_from_flags,
    mean_by_key,
    paired_diff,
    safe_div,
)


def _units(recs: list[RecScore], pick: Callable[[RecScore], tuple | None]) -> dict:
    return mean_by_key((r.run.video_id, v) for r in recs if (v := pick(r)) is not None)


def _t1_units(recs):
    return _units(recs, lambda r: (r.t1.tp, r.t1.fp, r.t1.fn) if r.t1 else None)


def _short_units(recs):
    def pick(r):
        if not r.t2:
            return None
        rows = [x for x in r.t2.rows if x.bucket in SHORT_BUCKETS]
        return (sum(x.found for x in rows), len(rows))
    return _units(recs, pick)


def _t2_prec_units(recs):
    return _units(recs, lambda r: (r.t2.tp, r.t2.fp) if r.t2 else None)


def _t4_units(recs):
    return _units(recs, lambda r: (r.t4.correct, r.t4.total) if r.t4 else None)


def _t5_units(recs):
    return _units(recs, lambda r: (r.t5.correct, r.t5.total) if r.t5 else None)


def compare_t5(a: list[RecScore], b: list[RecScore], *, n_boot: int, seed: int) -> dict | None:
    """a minus b on T5 accuracy, bootstrap CI over videos (None when no video is shared)."""
    return est_dict(paired_diff(_t5_units(a), _t5_units(b), _ratio, n=n_boot, seed=seed))


def _causal_units(recs):
    return _units(recs, lambda r: (len(r.t3.matched), r.t3.n_links) if r.t3 else None)


def _flags(recs: list[RecScore], rows_of: Callable[[RecScore], list | None]) -> dict:
    acc: dict = defaultdict(list)
    for r in recs:
        for row in rows_of(r) or ():
            acc[(r.run.video_id, row.index)].append(row.found)
    return majority_found(acc)


def _mcnemar(a: list[RecScore], b: list[RecScore], rows_of) -> dict | None:
    fa, fb = _flags(a, rows_of), _flags(b, rows_of)
    if not set(fa) & set(fb):
        return None
    m = mcnemar_from_flags(fa, fb)
    return {"w_only": m["a_only"], "base_only": m["b_only"], "n": m["n"], "p": m["p"]}


def est_dict(e: Estimate | None) -> dict | None:
    return None if e is None else {"point": e.point, "lo": e.lo, "hi": e.hi}


def to_est(d: dict | None) -> Estimate | None:
    return None if d is None else Estimate(d["point"], d["lo"], d["hi"])


def _ratio(v) -> float:
    return safe_div(v[0], v[1])


def _prec(v) -> float:
    return safe_div(v[0], v[0] + v[1])


def compare(
    a: list[RecScore],
    b: list[RecScore],
    *,
    n_boot: int,
    seed: int,
    sources: Mapping[str, str] | None = None,
) -> dict:
    """a minus b (a is typically W). Paired by video; repeats averaged within arm.

    T1 is cluster-bootstrapped by source clip (``sources``: video -> source); every other
    task uses the video as its own cluster.
    """
    kw = {"n": n_boot, "seed": seed}
    return {
        "t1_f1": est_dict(paired_diff(_t1_units(a), _t1_units(b), lambda v: f1(*v), cluster_of=sources, **kw)),
        "t2_short_recall": est_dict(paired_diff(_short_units(a), _short_units(b), _ratio, **kw)),
        "t2_precision": est_dict(paired_diff(_t2_prec_units(a), _t2_prec_units(b), _prec, **kw)),
        "t3_recall": est_dict(paired_diff(_causal_units(a), _causal_units(b), _ratio, **kw)),
        "t4_accuracy": est_dict(paired_diff(_t4_units(a), _t4_units(b), _ratio, **kw)),
        "t5_accuracy": compare_t5(a, b, n_boot=n_boot, seed=seed),
        "mcnemar_t1": _mcnemar(a, b, lambda r: r.t1.rows if r.t1 else None),
        "mcnemar_t2_short": _mcnemar(
            a, b, lambda r: [x for x in r.t2.rows if x.bucket in SHORT_BUCKETS] if r.t2 else None
        ),
    }
