"""Markdown rendering of a scorecard, following eval/results/TEMPLATE.md."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vwatch.evaluation.render_extra import method_only_section, per_source_section, t2_precision_rows
from vwatch.evaluation.render_fmt import NA, ci, f2, get, pct, pp, pt, table
from vwatch.evaluation.render_t5 import public_benchmark_section
from vwatch.evaluation.scorecard import key
from vwatch.evaluation.scoring import EVENT_BUCKETS
from vwatch.schemas import DefectType

_TYPE_LABELS = {
    DefectType.FLASH_FRAME: "Flash frame",
    DefectType.FROZEN_FRAME: "Frozen frame",
    DefectType.JUMP_CUT: "Jump cut",
    DefectType.AUDIO_DROPOUT: "Audio dropout",
    DefectType.CAPTION_BLOCKED: "Caption blocked",
    DefectType.WRONG_NUMBER: "Wrong number",
}


def _budget_label(budget: int) -> str:
    return f"{budget // 1000}k" if budget % 1000 == 0 else str(budget)


def headline(sc: Mapping) -> str:
    X = sc["matched_budget"]
    cw, cb = sc["cells"].get(key("W", X)), sc["cells"].get(key("B1", X))
    vs = get(sc, "comparisons", "W_vs_B1") or {}
    d1, d2 = vs.get("t1_f1"), vs.get("t2_short_recall")
    cost = sc.get("cost_ratio")
    return (
        f"W vs B1 @{_budget_label(X)} tokens: T1 F1 {f2(get(cw, 't1', 'f1'))} vs {f2(get(cb, 't1', 'f1'))} "
        f"({pp(pt(d1))}, CI {ci(d1)}), <1s recall {pp(pt(d2))}, "
        f"cost {NA if cost is None else f'{cost:.2f}×'}"
    )


def _criteria_rows(sc: Mapping) -> list[list[str]]:
    X = sc["matched_budget"]
    cw, cb = sc["cells"].get(key("W", X)), sc["cells"].get(key("B1", X))
    vs = get(sc, "comparisons", "W_vs_B1") or {}
    s7 = get(sc, "comparisons", "W_vs_B_at_2X") or {}
    cd = sc["cells"].get(s7.get("baseline", ""))
    st = {c["id"]: c["status"] for c in sc["criteria"]}
    tgt = {c["id"]: c["target"] for c in sc["criteria"]}

    def delta(a, b):
        return None if a is None or b is None else a - b

    rows = [
        ("S1", f2(get(cb, "t1", "f1")), f2(get(cw, "t1", "f1")), pp(pt(vs.get("t1_f1"))), ci(vs.get("t1_f1"))),
        ("S2", pct(get(cb, "t2", "short_recall")), pct(get(cw, "t2", "short_recall")),
         pp(pt(vs.get("t2_short_recall"))), ci(vs.get("t2_short_recall"))),
        ("S3", pct(get(cb, "t1", "loc_rate")), pct(get(cw, "t1", "loc_rate")),
         pp(delta(get(cw, "t1", "loc_rate"), get(cb, "t1", "loc_rate"))), NA),
        ("S4", pct(get(cb, "t1", "false_alarm_rate")), pct(get(cw, "t1", "false_alarm_rate")),
         pp(delta(get(cw, "t1", "false_alarm_rate"), get(cb, "t1", "false_alarm_rate"))), NA),
        ("S5", pct(get(cb, "t3", "recall")), pct(get(cw, "t3", "recall")),
         pp(pt(vs.get("t3_recall"))), ci(vs.get("t3_recall"))),
        ("S6", f"${get(cb, 'usd_per_min') or 0:.4f}/min" if cb else NA,
         f"${get(cw, 'usd_per_min') or 0:.4f}/min" if cw else NA,
         NA if sc.get("cost_ratio") is None else f"{sc['cost_ratio']:.2f}×", NA),
        ("S7", f2(get(cd, "t1", "f1")), f2(get(cw, "t1", "f1")), pp(pt(s7.get("t1_f1"))), ci(s7.get("t1_f1"))),
        ("S8", str(get(cb, "unseen_t1") if cb and get(cb, "unseen_t1") is not None else NA),
         str(get(cw, "unseen_t1") if cw and get(cw, "unseen_t1") is not None else NA), NA, NA),
    ]
    m1 = get(sc, "method_only", "W_vs_B1H", "t1_f1")
    b1h = get(sc, "method_only", "cells", "B1H")
    rows.append(("M1", f2(get(b1h, "t1", "f1")), f2(get(cw, "t1", "f1")), pp(pt(m1)), ci(m1)))
    names = {c["id"]: c["name"] for c in sc["criteria"]}
    return [[f"{i} {names[i]}", tgt[i], b, w, d, c, st[i]] for i, b, w, d, c in rows]


def render_markdown(scorecard: Mapping, meta: Mapping | None = None) -> str:
    meta = meta or {}
    sc = scorecard
    X = sc["matched_budget"]
    out: list[str] = []
    out.append(f"# Eval result: {meta.get('date', 'undated')}\n")
    out.append(f"- Model + version: {meta.get('model') or ', '.join(sc['models']) or NA}")
    out.append(f"- Watcher commit: {meta.get('commit', NA)}")
    out.append(f"- Thresholds locked on: {meta.get('locked_on', NA)}")
    reps = sorted({c["repeats"] for c in sc["cells"].values()})
    out.append(f"- Repeats per arm: {meta.get('repeats') or ', '.join(map(str, reps)) or NA}\n")
    out.append("## Headline\n")
    out.append(f"`{headline(sc)}`\n")
    out.append(f"Verdict: {sc['verdict']}\n")

    out.append("## Scorecard (matched budget)\n")
    out.append(table(["Criterion", "Target", "B1 (M1: B1H)", "W", "Δ", "95% CI", "Pass?"], _criteria_rows(sc)))
    out.append("\nS3 = localized (±0.5 s) defects ÷ ALL truth defects; S2 also needs W T2 precision >= B1 - 10 pp.")
    out.append("")
    out.append(table(
        ["Kill rule", "Condition", "Status"],
        [[f"{c['id']} {c['name']}", c["target"], c["status"]] for c in sc["criteria"] if c["id"].startswith("K")],
    ))
    out.append("")

    out.append("## Recall by event duration\n")
    dt = sc["duration_table"]
    arms = [a for a in ("B0", "B1", "B1H", "B2", "W", "W-noS") if a in dt]
    hdr = ["Bucket"] + [f"{a} (@{_budget_label(dt[a]['budget'])})" for a in arms]
    out.append(table(hdr, [[b] + [pct(dt[a][b]) for a in arms] for b in EVENT_BUCKETS]))
    if meta.get("recall_png"):
        out.append(f"\n![recall by duration]({meta['recall_png']})")
    out.append("")

    out.append("## Pareto (T1 F1 / <1 s event recall vs token cap)\n")
    out.append(table(
        ["Cap", "B1 F1", "W F1", "B1 <1s recall", "W <1s recall"],
        [[_budget_label(p["budget"]), f2(p["B1_f1"]), f2(p["W_f1"]),
          pct(p["B1_short_recall"]), pct(p["W_short_recall"])] for p in sc["pareto"]],
    ))
    if meta.get("pareto_png"):
        out.append(f"\n![pareto]({meta['pareto_png']})")
    out.append("")

    out.append("## Ablations\n")
    cw = sc["cells"].get(key("W", X))
    rows = [["W", f2(get(cw, "t1", "f1")), pct(get(cw, "t2", "short_recall")), pct(get(cw, "t3", "recall")), "—"]]
    for a in sc["ablations"]:
        c, g = a["cell"], a["gain_vs_W"]
        gain = get(g, "t1_f1")
        rows.append([a["arm"], f2(get(c, "t1", "f1")), pct(get(c, "t2", "short_recall")),
                     pct(get(c, "t3", "recall")),
                     NA if gain is None else f"{pp(gain['point'])} CI {ci(gain)}"])
    out.append(table(["Arm", "T1 F1", "<1 s recall", "Causal recall", "W minus this arm (T1 F1)"], rows))
    out.append("")

    out += method_only_section(sc)
    out += per_source_section(sc)
    out += t2_precision_rows(sc)
    out += public_benchmark_section(sc)

    out.append("## Per-defect-type recall\n")
    cb = sc["cells"].get(key("B1", X))
    out.append(table(
        ["Type", "B1", "W"],
        [[lab, pct(get(cb, "t1", "recall_by_type", t.value)), pct(get(cw, "t1", "recall_by_type", t.value))]
         for t, lab in _TYPE_LABELS.items()],
    ))
    out.append("")

    out.append("## All arms: cost and honesty\n")
    out.append(table(
        ["Arm@budget", "Records", "Videos", "T1 F1", "T1 prec", "T1 rec", "T2 rec", "T3 rec", "T4 acc",
         "In tok", "Content tok", "$/min", "Wall s/min", "Unseen T1", "T1 F1 std", "Failed runs", "Fail rate"],
        [[k, str(c["n_records"]), str(c["n_videos"]), f2(get(c, "t1", "f1")), f2(get(c, "t1", "precision")),
          f2(get(c, "t1", "recall")), f2(get(c, "t2", "recall")), f2(get(c, "t3", "recall")),
          f2(get(c, "t4", "accuracy")), f"{c['mean_input_tokens']:.0f}", f"{c['mean_content_tokens']:.0f}",
          NA if c["usd_per_min"] is None else f"{c['usd_per_min']:.4f}",
          NA if c["wall_s_per_min"] is None else f"{c['wall_s_per_min']:.1f}",
          NA if c["unseen_t1"] is None else str(c["unseen_t1"]), f2(get(c, "t1", "f1_std_repeats")),
          str(c["parse_errors"]), pct(c["failure_rate"])] for k, c in sc["cells"].items()],
    ))
    out.append("")

    out.append("## Notes / failures worth reading\n")
    notes = []
    for k, c in sc["cells"].items():
        if c["parse_errors"]:
            notes.append(f"{k}: {c['parse_errors']} of {c['n_records']} records failed (error / no parsed output): "
                         "T1 clean videos count as a false alarm, defect videos as all-missed, other tasks as empty.")
    na = [c["id"] for c in sc["criteria"] if c["status"] == NA]
    if na:
        notes.append(f"Criteria n/a (missing data): {', '.join(na)}.")
    if not sc["judge_used"]:
        notes.append("T3 not scored: no judge supplied.")
    for name, mc in (("T1 found/missed", get(sc, "comparisons", "W_vs_B1", "mcnemar_t1")),
                     ("T2 <1 s found/missed", get(sc, "comparisons", "W_vs_B1", "mcnemar_t2_short"))):
        if mc:
            notes.append(f"McNemar W vs B1, {name}: W-only {mc['w_only']}, B1-only {mc['base_only']}, "
                         f"n={mc['n']}, p={mc['p']:.4f}.")
    if sc["judge_used"] and sc["invented"]["W"] is not None:
        notes.append(f"Invented causal links per video: W {f2(sc['invented']['W'])}, B1 {f2(sc['invented']['B1'])}.")
    notes.append(f"Bootstrap: {sc['n_boot']} resamples, 95% percentile CI, repeats averaged per video. "
                 "T1 resamples whole source clips (cluster bootstrap); T2/T3/T4 resample videos.")
    notes.append("McNemar tests are per-item and not cluster-adjusted: descriptive only.")
    notes.append("Unseen claims count only frame/crop/audio observations within 0.5 s; hint- or transcript-only claims are unverified.")
    out += [f"- {n}" for n in notes]
    return "\n".join(out) + "\n"
