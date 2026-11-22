from __future__ import annotations

import json

import pytest
from test_dataset_fixtures import SPEC, make_source

from vwatch.dataset import build
from vwatch.dataset._common import assign_splits
from vwatch.schemas import DatasetItem, Task


def test_assign_splits_deterministic_and_proportional():
    ids = [f"v{i}" for i in range(60)]
    a, b = assign_splits(ids, 7, Task.DEFECTS), assign_splits(ids, 7, Task.DEFECTS)
    assert a == b and sum(v == "pilot" for v in a.values()) == 9  # round(0.15 * 60)
    assert a != assign_splits(ids, 8, Task.DEFECTS)
    assert set(assign_splits(["only"], 7, Task.QA).values()) == {"test"}
    assert sum(v == "pilot" for v in assign_splits(["a", "b"], 7, Task.QA).values()) == 1


def test_manifest_roundtrip_and_layout(tmp_path):
    src = make_source()
    items = build.build_dataset(tmp_path, scale="tiny", seed=7, sources=[SPEC], loader=lambda _s: src)
    assert len(items) == 7 + 1 + 1 + 1
    assert build.read_manifest(tmp_path) == items
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == len(items) and all(DatasetItem.model_validate_json(x) for x in lines)
    for it in items:
        assert not it.video_path.startswith("/") and (tmp_path / it.video_path).exists()
        assert (tmp_path / it.truth_path).exists()
        assert it.video_path.startswith({"T1": "t1", "T2": "t2", "T3": "t3", "T4": "t4"}[it.task.value] + "/")
        assert it.truth_path == it.video_path.replace(".mp4", ".truth.json")
        json.loads((tmp_path / it.truth_path).read_text())
    assert {i.task for i in items} == {Task.DEFECTS, Task.EVENTS, Task.CAUSAL, Task.QA}
    assert {i.split for i in items} <= {"pilot", "test"}
    assert any(i.split == "pilot" for i in items if i.task is Task.DEFECTS)


def test_main_cli(tmp_path, monkeypatch):
    seen = {}

    def fake(root, *, scale, seed, tasks, **_):
        seen.update(root=root, scale=scale, seed=seed, tasks=tasks)
        return []
    monkeypatch.setattr(build, "build_dataset", fake)
    assert build.main(["--root", str(tmp_path), "--scale", "pilot", "--seed", "7", "--tasks", "T2", "T3"]) == 0
    assert seen == {"root": tmp_path, "scale": "pilot", "seed": 7, "tasks": (Task.EVENTS, Task.CAUSAL)}


def test_bad_scale():
    with pytest.raises(ValueError, match="unknown scale"):
        build.build_dataset(__import__("pathlib").Path("/nonexistent-root-never-created"), scale="x", seed=1)


def test_missing_sources_fail_clearly_for_t1(tmp_path):
    from vwatch.dataset._source import SourceSpec

    ghost = SourceSpec("ghost", tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError, match="VWATCH_SHORTS_DIR"):
        build.build_dataset(tmp_path, scale="tiny", seed=7, sources=[ghost], tasks=[Task.DEFECTS])


def test_merge_manifest_keeps_other_tasks_and_t1x(tmp_path):
    def item(vid, task):
        return DatasetItem(video_id=vid, task=task, video_path="v.mp4", truth_path="t.json", split="test")

    (tmp_path / "manifest.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in [
        item("t1_old", Task.DEFECTS), item("t1x_keep", Task.DEFECTS), item("t5_keep", Task.MCQ),
        item("t2_old", Task.EVENTS)]))
    build.merge_manifest(tmp_path, [item("t1_new", Task.DEFECTS)], [Task.DEFECTS])
    ids = [json.loads(x)["video_id"] for x in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert ids == ["t1x_keep", "t5_keep", "t2_old", "t1_new"]
