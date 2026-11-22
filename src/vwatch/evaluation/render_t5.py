"""Markdown section for the public benchmark subset (T5, TVBench). Informational, never gating."""

from __future__ import annotations

from collections.abc import Mapping

from vwatch.evaluation.render_fmt import NA, ci, pct, pp, pt, table


def public_benchmark_section(sc: Mapping) -> list[str]:
    pb = sc.get("public_benchmark")
    if not pb:
        return []
    cells, cats, vs_w = pb["cells"], pb["categories"], pb.get("vs_W") or {}
    out = ["## Public benchmark (TVBench subset)\n",
           "Multiple-choice video QA from TVBench (CC-BY-4.0, FunAILab); informational, not part of the "
           "S/K criteria. Accuracy = correct items / items; a missing or invalid letter is wrong. "
           "Δ = W@matched budget minus the row's arm, 95% CI is a paired bootstrap over videos.\n"]
    rows = []
    for k, c in cells.items():
        d = vs_w.get(k)
        rows.append([k, str(c["n_videos"])]
                    + [pct((c["by_category"].get(cat) or {}).get("accuracy")) for cat in cats]
                    + [pct(c["accuracy"]), pp(pt(d)) if k in vs_w else NA, ci(d) if k in vs_w else NA])
    chance = [c["chance"] for c in cells.values() if c.get("chance") is not None]
    by_cat_chance = []
    for cat in cats:
        vals = [c["by_category"][cat]["chance"] for c in cells.values() if cat in c["by_category"]]
        by_cat_chance.append(pct(vals[0]) if vals else NA)
    rows.append(["chance (guessing)", ""] + by_cat_chance + [pct(chance[0]) if chance else NA, "", ""])
    out.append(table(["Arm@budget", "Videos"] + cats + ["Overall", "Δ vs W", "95% CI"], rows))
    out.append("")
    return out
