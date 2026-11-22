import json
import shutil

import pytest

from vwatch import transcribe
from vwatch.evaluation import runner
from vwatch.llm.base import LLMError
from vwatch.llm.fake import FakeBackend, content_text
from vwatch.schemas import Arm, DatasetItem, RunRecord, Task

DATE = "2026-01-01"
OK = '{"defects": []}'


@pytest.fixture(autouse=True)
def no_whisper(monkeypatch):
    monkeypatch.setattr(transcribe, "transcribe", lambda *a, **k: transcribe.EMPTY)


def item(vid, task=Task.DEFECTS, split="pilot", path=None):
    return DatasetItem(video_id=vid, task=task, video_path=path or f"videos/{vid}.mp4",
                       truth_path=f"truth/{vid}.json", split=split)


@pytest.fixture
def dataset(tmp_path, silent_video):
    root = tmp_path / "data"
    (root / "videos").mkdir(parents=True)
    (root / "truth").mkdir()
    items = [item("v1"), item("v2"), item("v3", split="test"), item("q1", Task.QA)]
    for it in items:
        shutil.copy(silent_video, root / it.video_path)
    for vid in ("v1", "v2", "v3"):
        (root / f"truth/{vid}.json").write_text(json.dumps(
            {"video_id": vid, "source": "synthetic", "duration": 4.0,
             "defects": [{"type": "flash_frame", "start": 1.0, "end": 1.1}]}))
    (root / "truth/q1.json").write_text(json.dumps(
        {"video_id": "q1", "duration": 4.0, "items": [
            {"id": "a", "question": "What is the price?", "answer": "98765", "answer_type": "number"}]}))
    (root / "manifest.jsonl").write_text("\n".join(it.model_dump_json() for it in items) + "\n")
    return root


def quiet(msgs=None):
    return (msgs if msgs is not None else []).append


class TestManifest:
    def test_load(self, dataset):
        items = runner.load_manifest(dataset)
        assert [i.video_id for i in items] == ["v1", "v2", "v3", "q1"]
        assert items[2].split == "test" and items[3].task is Task.QA

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="vw dataset build"):
            runner.load_manifest(tmp_path)

    def test_invalid_line_reports_file_and_line(self, tmp_path):
        (tmp_path / "manifest.jsonl").write_text(item("a").model_dump_json() + "\n\n{not json}\n")
        with pytest.raises(ValueError, match=r"manifest\.jsonl:3: invalid manifest line"):
            runner.load_manifest(tmp_path)

    def test_schema_violation_line(self, tmp_path):
        bad = json.loads(item("a").model_dump_json())
        bad["split"] = "weird"
        (tmp_path / "manifest.jsonl").write_text(item("ok").model_dump_json() + "\n" + json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match=r"manifest\.jsonl:2:"):
            runner.load_manifest(tmp_path)

    def test_blank_lines_skipped(self, tmp_path):
        (tmp_path / "manifest.jsonl").write_text("\n" + item("a").model_dump_json() + "\n   \n")
        assert len(runner.load_manifest(tmp_path)) == 1


class TestPlanJobs:
    def test_run_id(self):
        assert runner.run_id_for(DATE, Arm.W_NOQ, 25000, 2) == "2026-01-01_W-noQ_25k_r2"

    def test_b2_doubles_budget(self):
        jobs = runner.plan_jobs([item("a")], [Arm.B1, Arm.B2], [10000], 1, DATE)
        by_arm = {j.arm: j for j in jobs}
        assert by_arm[Arm.B1].budget == 10000 and by_arm[Arm.B2].budget == 20000
        assert by_arm[Arm.B2].run_id == "2026-01-01_B2_20k_r1" and by_arm[Arm.B1].run_id == "2026-01-01_B1_10k_r1"

    def test_cross_product_and_repeats(self):
        jobs = runner.plan_jobs([item("a"), item("b")], [Arm.B0, Arm.B2, Arm.W], [10000, 25000], 3, DATE)
        assert len(jobs) == 2 * 3 * 2 * 3
        assert {j.repeat for j in jobs} == {1, 2, 3}
        assert {j.run_id for j in jobs if j.arm is Arm.B2} == {
            f"{DATE}_B2_{b}k_r{r}" for b in (20, 50) for r in (1, 2, 3)}

    def test_empty(self):
        assert runner.plan_jobs([], [Arm.B0], [1000], 1, DATE) == []


class TestDoneKeys:
    def test_missing_file(self, tmp_path):
        assert runner.done_keys(tmp_path, "nope") == set()

    def test_ignores_errors_and_blanks(self, tmp_path):
        lines = [json.dumps({"video_id": "a", "task": "T1", "error": None}), "",
                 json.dumps({"video_id": "b", "task": "T1", "error": "LLMError: x"}),
                 json.dumps({"video_id": "c", "task": "T4"})]
        (tmp_path / "r.jsonl").write_text("\n".join(lines) + "\n")
        assert runner.done_keys(tmp_path, "r") == {("a", "T1"), ("c", "T4")}


def read_records(out, run_id):
    return [RunRecord.model_validate_json(l) for l in (out / f"{run_id}.jsonl").read_text().splitlines()]


class TestRunAll:
    def test_writes_jsonl(self, dataset, tmp_path):
        out, msgs = tmp_path / "runs", []
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:2], [Arm.B1], [4000], 1, DATE)
        be = FakeBackend(lambda *a: OK, model="fake-x")
        recs = runner.run_all(jobs, dataset, out, be, workers=2, log=msgs.append)
        assert len(recs) == 2
        saved = read_records(out, "2026-01-01_B1_4k_r1")
        assert sorted(r.video_id for r in saved) == ["v1", "v2"]
        r = saved[0]
        assert r.model == "fake-x" and r.parsed == {"defects": []} and r.error is None
        assert r.arm is Arm.B1 and r.budget_tokens == 4000 and r.task is Task.DEFECTS
        assert r.content_tokens_est > 0 and r.turns == 1 and r.cost_usd == pytest.approx(0.001)
        assert any("2 jobs, 0 already done, 2 to run" in m for m in msgs)
        assert any("prepared v1" in m for m in msgs) and any("ok" in m for m in msgs)

    def test_one_file_per_run_id(self, dataset, tmp_path):
        out = tmp_path / "runs"
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:1], [Arm.B0, Arm.B2], [4000], 1, DATE)
        runner.run_all(jobs, dataset, out, FakeBackend(lambda *a: OK), workers=1, log=quiet())
        assert sorted(p.name for p in out.glob("*.jsonl")) == ["2026-01-01_B0_4k_r1.jsonl", "2026-01-01_B2_8k_r1.jsonl"]

    def test_resume_skips_done(self, dataset, tmp_path):
        out, msgs = tmp_path / "runs", []
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:2], [Arm.B0], [4000], 1, DATE)
        runner.run_all(jobs, dataset, out, FakeBackend(lambda *a: OK), log=quiet())
        be2 = FakeBackend(lambda *a: OK)
        recs = runner.run_all(jobs, dataset, out, be2, log=msgs.append)
        assert recs == [] and be2.sessions == []
        assert "2 jobs, 2 already done, 0 to run" in msgs[0]
        assert len(read_records(out, "2026-01-01_B0_4k_r1")) == 2

    def test_partial_resume_only_runs_missing(self, dataset, tmp_path):
        out = tmp_path / "runs"
        items = runner.load_manifest(dataset)[:2]
        runner.run_all(runner.plan_jobs(items[:1], [Arm.B0], [4000], 1, DATE), dataset, out,
                       FakeBackend(lambda *a: OK), log=quiet())
        recs = runner.run_all(runner.plan_jobs(items, [Arm.B0], [4000], 1, DATE), dataset, out,
                              FakeBackend(lambda *a: OK), log=quiet())
        assert [r.video_id for r in recs] == ["v2"]

    def test_errored_runs_are_retried(self, dataset, tmp_path):
        out, msgs = tmp_path / "runs", []
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:2], [Arm.B0], [4000], 1, DATE)

        def boom(*a):
            raise LLMError("overloaded")

        first = runner.run_all(jobs, dataset, out, FakeBackend(boom), log=msgs.append)
        assert all(r.error and "overloaded" in r.error for r in first)
        assert any("ERROR LLMError: overloaded" in m for m in msgs)
        assert runner.done_keys(out, "2026-01-01_B0_4k_r1") == set()
        second = runner.run_all(jobs, dataset, out, FakeBackend(lambda *a: OK), log=quiet())
        assert len(second) == 2 and all(r.error is None for r in second)
        assert len(read_records(out, "2026-01-01_B0_4k_r1")) == 4  # history kept: 2 errors + 2 ok
        assert runner.run_all(jobs, dataset, out, FakeBackend(lambda *a: OK), log=quiet()) == []

    def test_parse_failure_logged_and_kept(self, dataset, tmp_path):
        msgs = []
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:1], [Arm.B0], [4000], 1, DATE)
        recs = runner.run_all(jobs, dataset, tmp_path / "r", FakeBackend(lambda *a: "garbage"), log=msgs.append)
        assert recs[0].parsed is None and recs[0].parse_error and recs[0].turns == 2
        assert any("parse-fail" in m for m in msgs)

    def test_context_prepared_once_per_video(self, dataset, tmp_path, monkeypatch):
        loads = []
        orig = runner.VideoContext.load

        def counting(path, language=None):
            loads.append((path.name, language))
            return orig(path, language=language)

        monkeypatch.setattr(runner.VideoContext, "load", staticmethod(counting))
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:1], [Arm.B0, Arm.B1, Arm.B2], [4000], 2, DATE)
        runner.run_all(jobs, dataset, tmp_path / "r", FakeBackend(lambda *a: OK), log=quiet())
        assert loads == [("v1.mp4", "hi")]  # T1 -> Hindi hint

    def test_t4_prompt_has_questions_not_answers(self, dataset, tmp_path):
        be = FakeBackend(lambda *a: '{"answers": [{"id": "a", "answer": "1"}]}')
        qa = [i for i in runner.load_manifest(dataset) if i.task is Task.QA]
        recs = runner.run_all(runner.plan_jobs(qa, [Arm.B0], [4000], 1, DATE), dataset, tmp_path / "r", be, log=quiet())
        sent = content_text(be.sessions[0].log[0])
        assert "a: What is the price?" in sent and "98765" not in sent
        assert recs[0].parsed == {"answers": [{"id": "a", "answer": "1"}]}

    def test_watcher_arm_through_runner(self, dataset, tmp_path):
        def resp(system, content, turn):
            return '{"final": {"defects": []}}' if "You can request" in content_text(content) else "{}"

        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:1], [Arm.W], [6000], 1, DATE)
        recs = runner.run_all(jobs, dataset, tmp_path / "r", FakeBackend(resp), log=quiet())
        assert recs[0].error is None and recs[0].turns >= 3 and recs[0].observations

    def test_missing_video_raises_before_any_work(self, dataset, tmp_path):
        (dataset / "videos/v1.mp4").unlink()
        jobs = runner.plan_jobs(runner.load_manifest(dataset)[:1], [Arm.B0], [4000], 1, DATE)
        with pytest.raises(Exception):
            runner.run_all(jobs, dataset, tmp_path / "r", FakeBackend(lambda *a: OK), log=quiet())


class TestMain:
    @pytest.fixture
    def patched(self, monkeypatch):
        made = {}

        def fake_default(model, effort):
            made.update(model=model, effort=effort)
            return FakeBackend(lambda *a: OK, model=model)

        monkeypatch.setattr(runner, "default_backend", fake_default)
        return made

    def test_csv(self):
        assert runner._csv("a, b,,c ") == ["a", "b", "c"]

    def test_main_ok(self, dataset, tmp_path, patched, capsys):
        out = tmp_path / "o"
        rc = runner.main(["--root", str(dataset), "--out", str(out), "--tasks", "T1", "--arms", "B0,B2",
                          "--budget", "4000", "--date", DATE, "--model", "m1", "--effort", "low"])
        assert rc == 0 and patched == {"model": "m1", "effort": "low"}
        assert sorted(p.name for p in out.glob("*.jsonl")) == ["2026-01-01_B0_4k_r1.jsonl", "2026-01-01_B2_8k_r1.jsonl"]
        # split defaults to pilot -> v1, v2 only (v3 is test)
        assert len(read_records(out, "2026-01-01_B0_4k_r1")) == 2
        assert "done: 4 runs, 0 errors" in capsys.readouterr().out

    def test_main_split_all_and_limit(self, dataset, tmp_path, patched):
        out = tmp_path / "o"
        runner.main(["--root", str(dataset), "--out", str(out), "--tasks", "T1", "--arms", "B0", "--budget", "4000",
                     "--split", "all", "--limit", "2", "--date", DATE])
        assert len(read_records(out, "2026-01-01_B0_4k_r1")) == 2
        out2 = tmp_path / "o2"
        runner.main(["--root", str(dataset), "--out", str(out2), "--tasks", "T1", "--arms", "B0", "--budget", "4000",
                     "--split", "all", "--date", DATE])
        assert len(read_records(out2, "2026-01-01_B0_4k_r1")) == 3

    def test_main_no_items(self, dataset, tmp_path, patched, capsys):
        rc = runner.main(["--root", str(dataset), "--out", str(tmp_path / "o"), "--tasks", "T3"])
        assert rc == 1 and "no dataset items match" in capsys.readouterr().err

    def test_main_errors_return_2(self, dataset, tmp_path, monkeypatch):
        def boom(*a):
            raise LLMError("x")

        monkeypatch.setattr(runner, "default_backend", lambda model, effort: FakeBackend(boom))
        rc = runner.main(["--root", str(dataset), "--out", str(tmp_path / "o"), "--tasks", "T1", "--arms", "B0",
                          "--budget", "4000", "--date", DATE])
        assert rc == 2


class TestCliSmoke:
    def test_help(self, capsys):
        from vwatch import cli

        assert cli.main(["--help"]) == 0
        assert "vw watch" in capsys.readouterr().out

    def test_unknown_command(self, capsys):
        from vwatch import cli

        assert cli.main(["bogus"]) == 2
        assert "unknown command" in capsys.readouterr().err


def test_limit_per_bucket_keeps_n_per_task_bucket():
    from vwatch.evaluation.runner import limit_per_bucket
    from vwatch.schemas import DatasetItem, Task

    def item(i, task, bucket):
        return DatasetItem(video_id=f"v{i}", task=task, video_path="x.mp4", truth_path="x.json",
                           split="pilot", bucket=bucket)

    items = [item(0, Task.DEFECTS, "clean"), item(1, Task.DEFECTS, "clean"), item(2, Task.DEFECTS, "clean"),
             item(3, Task.DEFECTS, "flash_frame"), item(4, Task.EVENTS, "clean")]
    kept = limit_per_bucket(items, 2)
    assert [k.video_id for k in kept] == ["v0", "v1", "v3", "v4"]


def test_language_hint_only_for_original_hindi_shorts():
    from vwatch.evaluation.runner import language_for
    from vwatch.schemas import DatasetItem, Task

    def item(vid, task):
        return DatasetItem(video_id=vid, task=task, video_path="v.mp4", truth_path="t.json", split="test")

    assert language_for(item("t1_flash_frame_000", Task.DEFECTS)) == "hi"
    assert language_for(item("t1x_prague_clean_0", Task.DEFECTS)) is None
    assert language_for(item("t5_object_count_000", Task.MCQ)) is None
