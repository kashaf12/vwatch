"""Build the scorecard dict from run records and truths (no I/O)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict

from vwatch.evaluation import criteria as C
from vwatch.evaluation.cells import Judge, RecScore, Truths, make_cell, score_records
from vwatch.evaluation.compare import compare, compare_t5, to_est
from vwatch.evaluation.runs import dedupe_runs, select_model
from vwatch.schemas import Arm, RunRecord, Task

__all__ = ["RecScore", "build_scorecard", "key", "score_records"]


def key(arm: str, budget: int) -> str:
    return f"{arm}@{budget}"


def _get(cells: dict, arm: str, budget: int) -> dict | None:
    return cells.get(key(arm, budget))


def _path(cell: dict | None, *path):
    for p in path:
        if cell is None:
            return None
        cell = cell.get(p)
    return cell


def _double_cell(cells: dict, budget: int) -> tuple[str, int] | None:
    for arm in ("B2", "B1"):  # convention: B2 records carry the 2X budget
        if key(arm, 2 * budget) in cells:
            return arm, 2 * budget
    return None


def _public_benchmark(cells: dict, groups: dict, X: int, n_boot: int, seed: int) -> dict | None:
    """T5 (TVBench subset): per-cell accuracy vs chance, plus W@X minus every other cell. Informational."""
    t5_cells = {k: c for k, c in cells.items() if c.get("t5")}
    if not t5_cells:
        return None
    w = groups.get((Arm.W.value, X))
    vs_w = {}
    if w and any(r.t5 for r in w):
        for k, c in t5_cells.items():
            if (c["arm"], c["budget"]) != (Arm.W.value, X):
                vs_w[k] = compare_t5(w, groups[(c["arm"], c["budget"])], n_boot=n_boot, seed=seed)
    cats = sorted({cat for c in t5_cells.values() for cat in c["t5"]["by_category"]})
    return {"cells": {k: c["t5"] | {"arm": c["arm"], "budget": c["budget"]} for k, c in t5_cells.items()},
            "categories": cats, "vs_W": vs_w}


def build_scorecard(
    runs: Sequence[RunRecord],
    truths: Truths,
    *,
    matched_budget: int,
    judge: Judge | None = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    if matched_budget <= 0:
        raise ValueError("matched_budget must be > 0")
    runs = select_model(dedupe_runs(runs), None)  # one record per (run, video, task); one model only
    scored = score_records(runs, truths, judge)
    groups: dict[tuple[str, int], list[RecScore]] = defaultdict(list)
    for s in scored:
        groups[(s.run.arm.value, s.run.budget_tokens)].append(s)  # single model => (model, arm, budget)
    cells = {key(a, b): make_cell(a, b, g) for (a, b), g in sorted(groups.items())}
    sources = {v: t.source for (v, task), t in truths.items() if task == Task.DEFECTS}
    cmp_kw = {"n_boot": n_boot, "seed": seed, "sources": sources}
    X = matched_budget
    w, b1 = groups.get((Arm.W.value, X)), groups.get((Arm.B1.value, X))

    comparisons: dict = {}
    if w and b1:
        comparisons["W_vs_B1"] = compare(w, b1, **cmp_kw)
    dbl = _double_cell(cells, X)
    if w and dbl:
        c = compare(w, groups[dbl], **cmp_kw)
        comparisons["W_vs_B_at_2X"] = {**c, "baseline": key(*dbl)}
    ablations = []
    for arm in C.ABLATION_ARMS + tuple(a for a in C.EXTRA_ARMS if (a, X) in groups):
        a = groups.get((arm, X))
        cmp = compare(w, a, **cmp_kw) if w and a else None
        ablations.append({"arm": arm, "cell": _get(cells, arm, X), "gain_vs_W": cmp})
    by_arm = {a["arm"]: a for a in ablations}
    method_only = {  # W - B1H: what the tools add beyond the same signal hints given to plain frames
        "W_vs_B1H": by_arm.get("B1H", {}).get("gain_vs_W"),
        "signal_contribution": by_arm.get("W-noS", {}).get("gain_vs_W"),  # W - W-noS
    }
    method_only["cells"] = {a: _get(cells, a, X) for a in ("W", "B1", "B1H", "W-noS")}

    cw, cb = _get(cells, "W", X), _get(cells, "B1", X)
    vs = comparisons.get("W_vs_B1", {})
    s7 = comparisons.get("W_vs_B_at_2X", {})
    cost_b = _path(cb, "usd_per_min")
    gaps = {}
    for a in ablations:
        if a["arm"] not in C.ABLATION_ARMS:
            continue
        d = to_est(_path(a["gain_vs_W"], "t1_f1"))
        gaps[a["arm"]] = d.point if d else None
    inputs = C.CriteriaInputs(
        f1_w=_path(cw, "t1", "f1"),
        t1_diff=to_est(vs.get("t1_f1")),
        t2_short_diff=to_est(vs.get("t2_short_recall")),
        loc_rate_w=_path(cw, "t1", "loc_rate"),
        false_alarm_w=_path(cw, "t1", "false_alarm_rate"),
        t2_precision_w=_path(cw, "t2", "precision"),
        t2_precision_b1=_path(cb, "t2", "precision"),
        m1_diff=to_est(_path(method_only["W_vs_B1H"], "t1_f1")),
        causal_diff=to_est(vs.get("t3_recall")),
        invented_w=_path(cw, "t3", "invented_mean"),
        invented_b1=_path(cb, "t3", "invented_mean"),
        cost_ratio=(cw["usd_per_min"] / cost_b) if cw and cost_b and cw["usd_per_min"] is not None else None,
        s7_diff=to_est(s7.get("t1_f1")),
        unseen_t1_w=_path(cw, "unseen_t1"),
        ablation_gaps=gaps,
    )
    results = C.evaluate(inputs)

    arms_in_data = sorted({a for a, _ in groups})
    budgets = sorted({b for a, b in groups if a in ("B1", "W")})
    pareto = [
        {
            "budget": bud,
            "B1_f1": _path(_get(cells, "B1", bud), "t1", "f1"),
            "W_f1": _path(_get(cells, "W", bud), "t1", "f1"),
            "B1_short_recall": _path(_get(cells, "B1", bud), "t2", "short_recall"),
            "W_short_recall": _path(_get(cells, "W", bud), "t2", "short_recall"),
        }
        for bud in budgets
    ]
    duration_table = {}
    for arm in ("B0", "B1", "B1H", "B2", "W", "W-noS"):
        if arm == "B2":
            ks = [k for k, c in cells.items() if c["arm"] == "B2"]
            cell = cells[key("B2", 2 * X)] if key("B2", 2 * X) in cells else (
                cells[max(ks, key=lambda k: cells[k]["budget"])] if ks else None)
        else:
            cell = _get(cells, arm, X)
        t2 = _path(cell, "t2")
        if t2:
            duration_table[arm] = {"budget": cell["budget"],
                                   **{bk: v["recall"] for bk, v in t2["by_bucket"].items()}}
    return {
        "matched_budget": X,
        "arms": arms_in_data,
        "models": sorted({r.model for r in runs}),
        "cells": cells,
        "comparisons": comparisons,
        "ablations": ablations,
        "method_only": method_only,
        "pareto": pareto,
        "public_benchmark": _public_benchmark(cells, groups, X, n_boot, seed),
        "duration_table": duration_table,
        "criteria": [asdict(r) for r in results],
        "verdict": str(C.verdict(results)),
        "verdict_label": C.verdict(results).label,
        "cost_ratio": inputs.cost_ratio,
        "invented": {"W": inputs.invented_w, "B1": inputs.invented_b1},
        "judge_used": judge is not None,
        "n_boot": n_boot,
    }
