import pytest

from vwatch.evaluation import report as R
from vwatch.evaluation.render import render_markdown
from vwatch.evaluation.scorecard import build_scorecard
from vwatch.schemas import DatasetItem, MCQTruth, RunRecord, Task

X = 1000
CATS = ("scene_transition", "object_count")
VIDS = [(f"t5_{c}_{i:03d}", c) for c in CATS for i in range(6)]


def truths():
    t = {}
    for v, c in VIDS:
        n = 2 if c == "scene_transition" else 4
        t[(v, Task.MCQ)] = MCQTruth(video_id=v, duration=10, source=f"TVBench/{c}", items=(
            {"id": "q1", "question": "?", "options": tuple(f"o{i}" for i in range(n)), "answer_index": 1, "category": c},))
    return t


def rec(arm, budget, vid, choice, **kw):
    parsed = None if choice is None else {"answers": [{"id": "q1", "choice": choice}]}
    return RunRecord(run_id=f"{arm}-{budget}", arm=arm, budget_tokens=budget, repeat=0, video_id=vid,
                     task=Task.MCQ, model="m", parsed=parsed, **kw)


def runs():
    out = []
    for i, (v, c) in enumerate(VIDS):
        out.append(rec("W", X, v, "B"))  # always right
        out.append(rec("B1", X, v, "B" if i % 2 else "(A)"))  # half right
        out.append(rec("CC", 0, v, None if i == 0 else "b. text"))  # one failed run counts wrong
    return out


@pytest.fixture(scope="module")
def sc():
    return build_scorecard(runs(), truths(), matched_budget=X, n_boot=300, seed=1)


def test_cells_accuracy_chance_and_categories(sc):
    w = sc["cells"]["W@1000"]["t5"]
    assert w["accuracy"] == 1.0 and w["n_items"] == 12
    assert w["chance"] == pytest.approx((6 * 0.5 + 6 * 0.25) / 12)
    assert w["by_category"]["scene_transition"]["chance"] == 0.5
    assert w["by_category"]["object_count"]["chance"] == 0.25
    assert sc["cells"]["B1@1000"]["t5"]["accuracy"] == 0.5
    assert sc["cells"]["CC@0"]["t5"]["accuracy"] == pytest.approx(11 / 12)


def test_paired_comparisons_vs_every_other_arm(sc):
    pb = sc["public_benchmark"]
    assert set(pb["vs_W"]) == {"B1@1000", "CC@0"}
    d = pb["vs_W"]["B1@1000"]
    assert d["point"] == pytest.approx(0.5) and d["lo"] <= d["point"] <= d["hi"]
    assert sc["comparisons"]["W_vs_B1"]["t5_accuracy"]["point"] == pytest.approx(0.5)


def test_markdown_section(sc):
    md = render_markdown(sc, {})
    assert "## Public benchmark (TVBench subset)" in md
    sec = md.split("## Public benchmark (TVBench subset)")[1].split("\n## ")[0]
    assert "object_count" in sec and "scene_transition" in sec and "chance (guessing)" in sec
    assert "| W@1000 |" in sec and "100.0%" in sec and "+50.0 pp" in sec


def test_load_truths_reads_t5(tmp_path):
    (tmp_path / "t5").mkdir()
    (tmp_path / "t5" / "v.truth.json").write_text(truths()[(VIDS[0][0], Task.MCQ)].model_dump_json())
    item = DatasetItem(video_id=VIDS[0][0], task=Task.MCQ, video_path="t5/v.mp4", truth_path="t5/v.truth.json",
                       split="test", bucket="scene_transition")
    (tmp_path / "manifest.jsonl").write_text(item.model_dump_json() + "\n")
    got = R.load_truths(tmp_path)
    assert isinstance(got[(VIDS[0][0], Task.MCQ)], MCQTruth)
