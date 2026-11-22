"""Extra markdown sections: method-only comparison, per-source F1, T2 precision."""

from __future__ import annotations

from collections.abc import Mapping

from vwatch.evaluation.render_fmt import NA, f2, get, pct, pp, pt, table, ci
from vwatch.evaluation.scorecard import key

_ROWS = (  # (label, cell path, comparison key)
    ("T1 F1", ("t1", "f1"), "t1_f1", f2),
    ("T2 <1 s recall", ("t2", "short_recall"), "t2_short_recall", pct),
    ("T2 precision", ("t2", "precision"), "t2_precision", pct),
    ("T4 accuracy", ("t4", "accuracy"), "t4_accuracy", pct),
)


def method_only_section(sc: Mapping) -> list[str]:
    mo = sc.get("method_only") or {}
    cells = mo.get("cells") or {}
    out = ["## Method-only comparison\n"]
    if not mo.get("W_vs_B1H") and not mo.get("signal_contribution"):
        return out + ["No B1H / W-noS cells at the matched budget (n/a).\n"]
    out.append("W vs B1H (same signal hint text, plain frames) isolates the method from the hints. "
               "W - W-noS is the contribution of the signal hints/candidates. "
               "CIs are cluster bootstraps (T1 by source clip, others by video).\n")
    for title, other, cmp_key in (("W vs B1H (method-only)", "B1H", "W_vs_B1H"),
                                  ("Signal contribution: W - W-noS", "W-noS", "signal_contribution")):
        cmp = mo.get(cmp_key)
        out.append(f"### {title}\n")
        rows = []
        for label, path, ck, fmt in _ROWS:
            d = get(cmp, ck)
            rows.append([label, fmt(get(cells.get("W"), *path)), fmt(get(cells.get(other), *path)),
                         pp(pt(d)), ci(d)])
        out.append(table(["Metric", "W", other, "Δ", "95% CI"], rows))
        out.append("")
    return out


def per_source_section(sc: Mapping) -> list[str]:
    X = sc["matched_budget"]
    cells = {c["arm"]: c for c in sc["cells"].values() if c["budget"] == X and c.get("t1")}
    sources = sorted({s for c in cells.values() for s in c["t1"].get("f1_by_source", {})})
    out = ["## T1 F1 per source clip (matched budget)\n"]
    if not cells:
        return out + [f"{NA}\n"]
    arms = list(cells)
    out.append(table(["Source"] + arms, [[s] + [f2(cells[a]["t1"]["f1_by_source"].get(s)) for a in arms]
                                         for s in sources]))
    out.append("")
    return out


def t2_precision_rows(sc: Mapping) -> list[str]:
    X = sc["matched_budget"]
    rows = [[k, pct(get(c, "t2", "precision")), f2(get(c, "t2", "mean_listed")),
             pct(get(c, "t1", "strict_recall")), pct(get(c, "t1", "loc_given_found")),
             pct(c["failure_rate"])]
            for k, c in sc["cells"].items() if c["budget"] == X or c["arm"] == "B2"]
    return ["## Precision, strictness and failures\n",
            table(["Arm@budget", "T2 precision", "T2 events listed/video", "T1 strict recall (±0.5 s)",
                   "T1 loc | found", "Failure rate"], rows), ""]
