import json
import sys
import types

import pytest

from vwatch.evaluation import report as R
from vwatch.evaluation.runs import select_model
from vwatch.evaluation.scorecard import key, score_records
from vwatch.schemas import (
    CausalLink, CausalTruth, DatasetItem, DefectTruth, EventTruth, Observation, QAItem, QATruth,
    RunRecord, Task, TruthDefect, TruthEvent, Usage,
)

VIDS = [f"v{i}" for i in range(8)]
CLEAN = {"v6", "v7"}
X = 1000


def build_truths():
    t = {}
    for i, v in enumerate(VIDS):
        defects = () if v in CLEAN else (
            TruthDefect(type="flash_frame", start=5, end=5.1), TruthDefect(type="jump_cut", start=12, end=12.2))
        t[(v, Task.DEFECTS)] = DefectTruth(video_id=v, source=f"s{i % 4}", duration=30, defects=defects)
        t[(v, Task.EVENTS)] = EventTruth(video_id=v, duration=30, events=(
            TruthEvent(id="e0", start=2, end=2.3, kind="appear", subject="star", keywords=("star",)),
            TruthEvent(id="e1", start=8, end=12, kind="move", subject="ball", keywords=("ball",))))
        t[(v, Task.CAUSAL)] = CausalTruth(video_id=v, duration=30, links=(
            CausalLink(id="L1", cause="a", effect="b", time=1), CausalLink(id="L2", cause="c", effect="d", time=2)))
        t[(v, Task.QA)] = QATruth(video_id=v, duration=30, items=(
            QAItem(id="q1", question="?", answer="80", answer_type="number"),))
    return t


def rec(arm, budget, repeat, vid, task, parsed, *, w=False, **kw):
    obs = (Observation(kind="frame", t0=0, t1=30),) if w else ()
    return RunRecord(run_id=f"{arm}-{budget}-{repeat}-{vid}-{task.value}", arm=arm, budget_tokens=budget,
                     repeat=repeat, video_id=vid, task=task, model="m", parsed=parsed, observations=obs,
                     usage=Usage(input_tokens=budget // 2), content_tokens_est=budget // 3, **kw)


def runs_for(arm, budget, *, strong, repeats=(0, 1), cost=0.01):
    out = []
    for r in repeats:
        for v in VIDS:
            clean = v in CLEAN
            d = [] if clean else [{"type": "flash_frame", "start": 5.2}]
            if strong and not clean:
                d.append({"type": "jump_cut", "start": 12.1})
            if not strong and v == "v6":
                d.append({"type": "flash_frame", "start": 3})  # false alarm
            ev = [{"start": 8.5, "description": "ball moves"}]
            if strong:
                ev.append({"start": 2.1, "description": "a star appears"})
            retell = [("watcher: A because B" if strong else "plain: nothing")]
            kw = dict(w=strong, cost_usd=cost, wall_s=3.0)
            out += [
                rec(arm, budget, r, v, Task.DEFECTS, {"defects": d}, **kw),
                rec(arm, budget, r, v, Task.EVENTS, {"events": ev}, **kw),
                rec(arm, budget, r, v, Task.CAUSAL, {"retell": retell}, **kw),
                rec(arm, budget, r, v, Task.QA, {"answers": [{"id": "q1", "answer": "80" if strong else "7"}]}, **kw),
            ]
    return out


def judge(prompt):
    return '{"matched": ["L1", "L2"], "invented": 0}' if "watcher:" in prompt else '{"matched": [], "invented": 2}'


@pytest.fixture(scope="module")
def runs():
    rs = runs_for("W", X, strong=True, cost=0.012) + runs_for("B1", X, strong=False)
    rs += runs_for("B2", 2 * X, strong=False, cost=0.02) + runs_for("B0", X, strong=False)
    rs += runs_for("W", X // 2, strong=True) + runs_for("B1", X // 2, strong=False)
    rs += runs_for("W-noQ", X, strong=True) + runs_for("W-noP", X, strong=False) + runs_for("W-noCrop", X, strong=True)
    rs = [r for r in rs if not (r.arm.value == "B1" and r.budget_tokens == X and r.repeat == 0
                                and r.video_id == "v0" and r.task == Task.DEFECTS)]
    rs.append(rec("B1", X, 0, "v0", Task.DEFECTS, None, parse_error="bad json", cost_usd=0.01, wall_s=3.0))
    return rs


@pytest.fixture(scope="module")
def sc(runs):
    return R.build_scorecard(runs, build_truths(), matched_budget=X, judge=judge, n_boot=300)


def test_scorecard_cells(sc):
    w, b1 = sc["cells"][key("W", X)], sc["cells"][key("B1", X)]
    assert w["t1"]["f1"] == pytest.approx(1.0 - 0.0) or w["t1"]["f1"] > 0.99
    assert w["t1"]["recall"] == 1.0 and w["t1"]["precision"] == 1.0
    assert b1["t1"]["recall"] == pytest.approx(11 / 24) and b1["t1"]["false_alarm_rate"] == pytest.approx(0.5)
    assert w["t1"]["false_alarm_rate"] == 0.0 and w["t1"]["n_clean"] == 4
    assert w["t1"]["recall_by_type"] == {"flash_frame": 1.0, "jump_cut": 1.0}
    assert w["t1"]["loc_rate"] == 1.0
    assert w["t2"]["by_bucket"]["<0.5s"]["recall"] == 1.0 and b1["t2"]["by_bucket"]["<0.5s"]["recall"] == 0.0
    assert b1["t2"]["by_bucket"][">3s"]["recall"] == 1.0
    assert w["t2"]["short_recall"] == 1.0 and b1["t2"]["short_recall"] == 0.0
    assert w["t3"]["recall"] == 1.0 and b1["t3"]["invented_mean"] == 2
    assert w["t4"]["accuracy"] == 1.0 and b1["t4"]["accuracy"] == 0.0
    assert w["mean_input_tokens"] == X / 2 and w["mean_content_tokens"] == X // 3
    assert w["usd_per_min"] == pytest.approx(0.012 / 0.5)  # 30 s videos
    assert w["wall_s_per_min"] == pytest.approx(6.0)
    assert w["unseen_t1"] == 0 and b1["unseen_t1"] > 0
    assert b1["parse_errors"] == 1 and w["parse_errors"] == 0
    assert w["t1"]["f1_std_repeats"] == 0.0
    assert w["repeats"] == 2


def test_comparisons_and_verdict(sc):
    vs = sc["comparisons"]["W_vs_B1"]
    assert vs["t1_f1"]["point"] > 0.35 and vs["t1_f1"]["lo"] > 0
    assert vs["t2_short_recall"]["point"] == 1.0
    assert vs["mcnemar_t2_short"]["w_only"] == 8 and vs["mcnemar_t2_short"]["base_only"] == 0
    assert vs["mcnemar_t2_short"]["p"] == pytest.approx(0.0078125)
    assert vs["mcnemar_t1"]["w_only"] > 0
    assert sc["comparisons"]["W_vs_B_at_2X"]["baseline"] == "B2@2000"
    assert sc["cost_ratio"] == pytest.approx(1.2)
    st = {c["id"]: c["status"] for c in sc["criteria"]}
    assert st["S1"] == "pass" and st["S2"] == "pass" and st["S5"] == "pass" and st["S6"] == "pass"
    assert st["S8"] == "pass" and st["S4"] == "pass" and st["S7"] == "pass"
    assert st["K3"] == "clear" and st["K1"] == "clear" and st["K2"] == "clear"
    assert sc["verdict"] == "SHIP"
    abl = {a["arm"]: a for a in sc["ablations"]}
    assert abl["W-noP"]["gain_vs_W"]["t1_f1"]["point"] > 0.3
    assert abl["W-noQ"]["gain_vs_W"]["t1_f1"]["point"] == pytest.approx(0.0)


def test_pareto_and_duration_table(sc):
    assert [p["budget"] for p in sc["pareto"]] == [X // 2, X]
    assert sc["pareto"][1]["W_f1"] > sc["pareto"][1]["B1_f1"]
    dt = sc["duration_table"]
    assert dt["W"]["<0.5s"] == 1.0 and dt["B1"]["<0.5s"] == 0.0 and dt["B2"]["budget"] == 2 * X


def test_no_judge_gives_na_and_incomplete(runs):
    sc = R.build_scorecard(runs, build_truths(), matched_budget=X, n_boot=100)
    st = {c["id"]: c["status"] for c in sc["criteria"]}
    assert st["S5"] == "n/a" and sc["cells"][key("W", X)]["t3"] is None
    assert sc["verdict"] == "INCOMPLETE"
    md = R.render_markdown(sc, {})
    assert "T3 not scored" in md and "Criteria n/a" in md and "S5" in md


def test_missing_arms_na(runs):
    only_w = [r for r in runs if r.arm.value == "W" and r.budget_tokens == X]
    sc = R.build_scorecard(only_w, build_truths(), matched_budget=X, n_boot=50)
    assert sc["comparisons"] == {} and sc["verdict"] == "INCOMPLETE"
    assert R.render_markdown(sc, {}).count("n/a") > 5
    sc = R.build_scorecard(only_w, build_truths(), matched_budget=777, n_boot=50)
    assert sc["verdict"] == "INCOMPLETE" and sc["cost_ratio"] is None


def test_missing_truth_and_bad_budget(runs):
    with pytest.raises(ValueError, match="no truth"):
        R.build_scorecard(runs, {}, matched_budget=X)
    with pytest.raises(ValueError):
        R.build_scorecard(runs, build_truths(), matched_budget=0)


def test_kill_when_b2_matches(runs):
    strong_b2 = [r for r in runs if not (r.arm.value == "B2")] + runs_for("B2", 2 * X, strong=True)
    sc = R.build_scorecard(strong_b2, build_truths(), matched_budget=X, judge=judge, n_boot=100)
    assert sc["verdict"].startswith("KILL") and "K2" in sc["verdict"]


def test_render_markdown_headline(sc):
    md = R.render_markdown(sc, {"date": "2026-11-29", "model": "m-1", "commit": "abc", "locked_on": "d",
                                "recall_png": "a.png", "pareto_png": "b.png"})
    line = next(l for l in md.splitlines() if l.startswith("`W vs B1 @"))
    assert line.startswith("`W vs B1 @1k tokens: T1 F1 1.00 vs 0.")
    assert "pp, CI [" in line and "<1s recall +100.0 pp, cost 1.20×`" in line
    for h in ("## Headline", "## Scorecard (matched budget)", "## Recall by event duration", "## Pareto",
              "## Ablations", "## Per-defect-type recall", "## Notes", "Verdict: SHIP", "![pareto](b.png)",
              "# Eval result: 2026-11-29", "1 of "):
        assert h in md
    assert "S1 Defect F1" in md and "Flash frame" in md


def test_budget_label_fallback():
    from vwatch.evaluation.render import _budget_label
    assert _budget_label(25000) == "25k" and _budget_label(1500) == "1500"


def test_plots(sc, tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    R.plot_recall_by_duration(sc, a)
    R.plot_pareto(sc, b)
    assert a.stat().st_size > 1000 and b.stat().st_size > 1000


# ---------------------------------------------------------------- I/O


def write_dataset(tmp_path, runs):
    root = tmp_path / "data"
    (root / "truth").mkdir(parents=True)
    lines = []
    for (v, task), truth in build_truths().items():
        p = f"truth/{v}_{task.value}.json"
        (root / p).write_text(truth.model_dump_json())
        lines.append(DatasetItem(video_id=v, task=task, video_path=f"{v}.mp4", truth_path=p, split="test").model_dump_json())
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "a.jsonl").write_text("\n\n".join(r.model_dump_json() for r in runs[:50]) + "\n\n")
    (runs_dir / "b.jsonl").write_text("\n".join(r.model_dump_json() for r in runs[50:]) + "\n")
    return root, runs_dir


def test_load_roundtrip(tmp_path, runs):
    root, runs_dir = write_dataset(tmp_path, runs)
    assert len(R.load_runs(sorted(runs_dir.glob("*.jsonl")))) == len(runs)
    truths = R.load_truths(root)
    assert len(truths) == 32 and truths[("v0", Task.DEFECTS)].defects[0].type.value == "flash_frame"


def test_load_errors(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('\n{"nope": 1}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        R.load_runs([bad])
    bad.write_text("not json\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        R.load_runs([bad])
    with pytest.raises(FileNotFoundError):
        R.load_truths(tmp_path / "nowhere")
    root = tmp_path / "d"
    root.mkdir()
    item = DatasetItem(video_id="v", task=Task.QA, video_path="v.mp4", truth_path="t.json", split="test")
    (root / "manifest.jsonl").write_text(item.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="cannot load truth"):
        R.load_truths(root)
    (root / "t.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot load truth"):
        R.load_truths(root)
    (root / "manifest.jsonl").write_text(item.model_dump_json() + "\n" + item.model_dump_json() + "\n")
    (root / "t.json").write_text(QATruth(video_id="v", duration=1, items=()).model_dump_json())
    with pytest.raises(ValueError, match="duplicate"):
        R.load_truths(root)
    (root / "manifest.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="invalid manifest"):
        R.load_truths(root)


def test_main_end_to_end(tmp_path, runs, capsys):
    root, runs_dir = write_dataset(tmp_path, runs)
    out = tmp_path / "res" / "pilot.md"
    rc = R.main(["--runs", str(runs_dir / "*.jsonl"), "--root", str(root), "--budget", str(X),
                 "--out", str(out), "--n-boot", "100", "--date", "2026-11-29"])
    assert rc == 0
    assert out.exists() and (out.parent / "pilot_pareto.png").exists() and (out.parent / "pilot_recall_by_duration.png").exists()
    md = out.read_text()
    assert "# Eval result: 2026-11-29" in md and "pilot_pareto.png" in md
    assert "wrote" in capsys.readouterr().out


def test_main_judge_and_errors(tmp_path, runs, monkeypatch):
    root, runs_dir = write_dataset(tmp_path, runs)
    argv = ["--runs", str(runs_dir / "*.jsonl"), "--root", str(root), "--budget", str(X),
            "--out", str(tmp_path / "o.md"), "--n-boot", "50"]
    mod = types.ModuleType("vwatch.llm.judge")
    mod.make_judge = lambda model: judge
    monkeypatch.setitem(sys.modules, "vwatch.llm.judge", mod)
    assert R.main(argv + ["--judge", "sonnet"]) == 0
    assert "Invented causal links" in (tmp_path / "o.md").read_text()
    monkeypatch.setitem(sys.modules, "vwatch.llm.judge", None)
    with pytest.raises(RuntimeError, match="make_judge"):
        R.main(argv + ["--judge"])
    with pytest.raises(SystemExit):
        R.main(["--runs", str(tmp_path / "zzz*.jsonl"), "--budget", "1", "--out", "x.md"])
    empty = tmp_path / "e.jsonl"
    empty.write_text("\n")
    with pytest.raises(SystemExit):
        R.main(["--runs", str(empty), "--budget", "1", "--out", "x.md"])


def test_score_records_flags_failures(runs):
    s = score_records([r for r in runs if r.parse_error], build_truths(), None)
    assert s[0].failed and s[0].t1.tp == 0 and s[0].t1.fn == 2


# ---------------------------------------------------------------- validity fixes


def _all_runs(runs):
    rs = list(runs) + runs_for("B1H", X, strong=False) + runs_for("W-noS", X, strong=True)
    return rs


def test_failed_run_on_clean_video_is_false_alarm(runs):
    clean_fail = rec("W", X, 0, "v6", Task.DEFECTS, None, error="boom")
    ok = [r for r in runs if r.arm.value == "W" and r.budget_tokens == X
          and not (r.video_id == "v6" and r.task == Task.DEFECTS and r.repeat == 0)]
    s = score_records([clean_fail], build_truths(), None)[0]
    assert s.failed and s.t1.false_alarm and s.t1.fp == 1
    sc = R.build_scorecard(ok + [clean_fail], build_truths(), matched_budget=X, n_boot=50)
    cell = sc["cells"][key("W", X)]
    assert cell["t1"]["false_alarm_rate"] == pytest.approx(1 / 4)  # 1 of 4 clean records (2 clean videos x 2 repeats)
    assert cell["failure_rate"] == pytest.approx(1 / cell["n_records"])


def test_dedupe_last_wins_success_preferred():
    good = rec("W", X, 0, "v0", Task.DEFECTS, {"defects": []})
    bad = good.model_copy(update={"parsed": None, "error": "x"})
    bad2 = good.model_copy(update={"parsed": None, "error": "y"})
    late_good = good.model_copy(update={"raw_output": "retry"})
    assert R.dedupe_runs([bad, late_good]) == [late_good]
    assert R.dedupe_runs([good, bad]) == [good]  # a later failure never replaces a success
    assert R.dedupe_runs([bad, bad2]) == [bad2]
    assert R.dedupe_runs([good, late_good]) == [late_good]
    other = rec("W", X, 0, "v1", Task.DEFECTS, {"defects": []})
    assert R.dedupe_runs([bad, other, late_good]) == [late_good, other]


def test_scorecard_dedupes_retries(runs):
    w = [r for r in runs if r.arm.value == "W" and r.budget_tokens == X]
    first = w[0]
    failed = first.model_copy(update={"parsed": None, "error": "boom"})
    sc = R.build_scorecard([failed] + w, build_truths(), matched_budget=X, n_boot=50)
    assert sc["cells"][key("W", X)]["n_records"] == len(w)
    assert sc["cells"][key("W", X)]["failure_rate"] == 0


def test_multiple_models_require_selection(runs):
    w = [r for r in runs if r.arm.value == "W" and r.budget_tokens == X]
    other = [r.model_copy(update={"model": "m2", "run_id": r.run_id + "x"}) for r in w]
    with pytest.raises(ValueError, match="--model"):
        R.build_scorecard(w + other, build_truths(), matched_budget=X, n_boot=20)
    assert len(select_model(w + other, "m2")) == len(other)
    with pytest.raises(ValueError, match="no run records"):
        select_model(w, "zzz")
    assert select_model(w, None) == w


def test_main_model_flag(tmp_path, runs):
    root, runs_dir = write_dataset(tmp_path, runs)
    extra = [r.model_copy(update={"model": "m2", "run_id": r.run_id + "x"}) for r in runs if r.arm.value == "W"]
    (runs_dir / "c.jsonl").write_text("\n".join(r.model_dump_json() for r in extra) + "\n")
    argv = ["--runs", str(runs_dir / "*.jsonl"), "--root", str(root), "--budget", str(X),
            "--out", str(tmp_path / "o.md"), "--n-boot", "30"]
    with pytest.raises(SystemExit, match="--model"):
        R.main(argv)
    assert R.main(argv + ["--model", "m"]) == 0
    assert "Model + version: m" in (tmp_path / "o.md").read_text()
    with pytest.raises(SystemExit, match="no run records"):
        R.main(argv + ["--model", "nope"])


def test_strict_recall_and_loc_over_all_truth(sc):
    w, b1 = sc["cells"][key("W", X)]["t1"], sc["cells"][key("B1", X)]["t1"]
    assert w["strict_recall"] == w["loc_rate"] == 1.0
    # B1 finds only flash_frame at 0.2 s error (hit) and misses jump_cuts: hits / ALL truth defects
    assert b1["loc_rate"] == pytest.approx(b1["recall"] * (b1["loc_given_found"] or 0))
    assert b1["loc_rate"] <= b1["recall"]


def test_t2_precision_and_listed(sc):
    w, b1 = sc["cells"][key("W", X)]["t2"], sc["cells"][key("B1", X)]["t2"]
    assert w["precision"] == 1.0 and w["mean_listed"] == 2.0
    assert b1["mean_listed"] == 1.0 and b1["precision"] == 1.0
    assert sc["comparisons"]["W_vs_B1"]["t2_precision"]["point"] == 0.0
    assert sc["comparisons"]["W_vs_B1"]["t4_accuracy"]["point"] == 1.0


def test_s2_gated_by_precision(runs):
    spammy = []
    for r in runs:
        if r.arm.value == "W" and r.budget_tokens == X and r.task == Task.EVENTS:
            ev = list(r.parsed["events"]) + [{"start": 20 + i, "description": f"junk {i}"} for i in range(8)]
            r = r.model_copy(update={"parsed": {"events": ev}})
        spammy.append(r)
    sc = R.build_scorecard(spammy, build_truths(), matched_budget=X, judge=judge, n_boot=100)
    assert sc["cells"][key("W", X)]["t2"]["precision"] < 0.3
    st = {c["id"]: c["status"] for c in sc["criteria"]}
    assert st["S2"] == "fail" and sc["verdict"] == "ITERATE"


def test_clustered_ci_wider_than_per_video(runs):
    truths = build_truths()
    solo = {k: (t.model_copy(update={"source": "one"}) if k[1] == Task.DEFECTS else t) for k, t in truths.items()}
    sc1 = R.build_scorecard(runs, solo, matched_budget=X, n_boot=100)
    d = sc1["comparisons"]["W_vs_B1"]["t1_f1"]
    assert d["lo"] == float("-inf") and d["hi"] == float("inf")  # one cluster: uninformative CI
    assert next(c for c in sc1["criteria"] if c["id"] == "S1")["status"] == "fail"
    md = R.render_markdown(sc1, {})
    assert "n/a (<2 clusters)" in md


def test_per_source_f1(sc):
    by = sc["cells"][key("W", X)]["t1"]["f1_by_source"]
    assert sorted(by) == ["s0", "s1", "s2", "s3"] and all(v == 1.0 for v in by.values())
    md = R.render_markdown(sc, {})
    assert "## T1 F1 per source clip" in md and "| s0 |" in md


def test_method_only_section_and_m1(runs):
    rs = _all_runs(runs)
    sc = R.build_scorecard(rs, build_truths(), matched_budget=X, judge=judge, n_boot=100)
    mo = sc["method_only"]
    assert mo["W_vs_B1H"]["t1_f1"]["point"] > 0.3
    assert abs(mo["signal_contribution"]["t1_f1"]["point"]) < 1e-9  # W-noS fixture == W
    assert set(mo["W_vs_B1H"]) >= {"t1_f1", "t2_short_recall", "t2_precision", "t4_accuracy"}
    m1 = next(c for c in sc["criteria"] if c["id"] == "M1")
    assert m1["status"] == "info" and "W - B1H" in m1["detail"]
    assert sc["verdict"] == "SHIP"
    assert {a["arm"] for a in sc["ablations"]} >= {"B1H", "W-noS"}
    assert "B1H" in sc["duration_table"] and "W-noS" in sc["duration_table"]
    md = R.render_markdown(sc, {})
    assert "## Method-only comparison" in md and "Signal contribution: W - W-noS" in md
    assert "M1 W vs B1H on T1 F1 (method-only)" in md and "| info |" in md
    assert "## Precision, strictness and failures" in md
    R.plot_recall_by_duration(sc, "/dev/null.png") if False else None


def test_method_only_absent_is_na(sc):
    assert sc["method_only"]["W_vs_B1H"] is None
    assert next(c for c in sc["criteria"] if c["id"] == "M1")["status"] == "info"
    assert "No B1H / W-noS cells" in R.render_markdown(sc, {})


def test_plots_with_extra_arms(runs, tmp_path):
    sc = R.build_scorecard(_all_runs(runs), build_truths(), matched_budget=X, n_boot=30)
    R.plot_recall_by_duration(sc, tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 1000
