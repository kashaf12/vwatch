import io
import json
import threading
import zipfile

import pytest

from vwatch.dataset import tvbench as T
from vwatch.evaluation import scoring_mcq as M
from vwatch.schemas import DatasetItem, MCQReport, MCQTruth, Task
from vwatch.tasks import parse_report, task_prompt


def row(video, answer="Up.", cands=("Down.", "Up."), **kw):
    return {"video": video, "question": f"Q about {video}?", "answer": answer, "candidates": list(cands), **kw}


# ---------------------------------------------------------------- answer mapping


def test_answer_index_text_letter_int_and_errors():
    c = ["Down and left.", "Up and right.", "Down."]
    assert T.answer_index(c, "Up and right.") == 1
    assert T.answer_index(c, " up  and RIGHT ") == 1  # whitespace, case, trailing period
    assert T.answer_index(c, "C") == 2  # letter fallback only when no option text matches
    assert T.answer_index(c, 0) == 0
    with pytest.raises(ValueError):
        T.answer_index(c, "Sideways")
    with pytest.raises(ValueError):
        T.answer_index(["a", "A."], "a")  # ambiguous


def test_answer_text_wins_over_letter():
    assert T.answer_index(["A", "B", "C"], "B") == 1
    assert T.answer_index(["x", "A"], "A") == 1


def test_clip_trims_only_for_interval_categories():
    r = row("v.mp4", start=2.0, end=5.0)
    assert (T.clip_from_row("action_sequence", r).start, T.clip_from_row("action_sequence", r).end) == (2.0, 5.0)
    assert T.clip_from_row("scene_transition", r).start is None
    with pytest.raises(ValueError):
        T.clip_from_row("action_sequence", row("v.mp4", start=5.0, end=2.0))


def test_sampling_seeded_distinct_and_split():
    rows = [row(f"v{i % 12}.mp4", start=float(i % 2), end=9.0) for i in range(30)]
    a = T.sample_clips("action_sequence", rows, 10, seed=7)
    assert a == T.sample_clips("action_sequence", rows, 10, seed=7)
    assert a != T.sample_clips("action_sequence", rows, 10, seed=8)
    assert len({(c.video, c.start, c.end) for c in a}) == 10
    with pytest.raises(ValueError):
        T.sample_clips("action_sequence", rows, 50, seed=7)
    assert [T.split_of(i, 10) for i in range(10)] == ["pilot"] * 4 + ["test"] * 6


# ---------------------------------------------------------------- prompt


def test_prompt_lettered_options_and_never_the_answer():
    item_opts = ("Down and to the left.", "Up and to the right.", "Down and to the right.", "Up and to the left.")
    t = MCQTruth(video_id="t5_x_000", duration=3, source="TVBench/x", items=(
        {"id": "q1", "question": "Which way?", "options": item_opts, "answer_index": 2, "category": "SECRETCAT"},))
    p = task_prompt(Task.MCQ, t)
    for letter, o in zip("ABCD", item_opts):
        assert f"  {letter}. {o}" in p
    assert "q1: Which way?" in p and "letters only" in p and '"choice"' in p
    assert "answer_index" not in p and "SECRETCAT" not in p
    # the prompt is identical whatever the answer index is
    other = t.model_copy(update={"items": (t.items[0].model_copy(update={"answer_index": 0}),)})
    assert task_prompt(Task.MCQ, other) == p
    with pytest.raises(ValueError):
        task_prompt(Task.MCQ)


def test_parse_report_t5():
    parsed, err = parse_report(Task.MCQ, '{"answers": [{"id": "q1", "choice": "B"}]}')
    assert err is None and parsed["answers"][0]["choice"] == "B"


# ---------------------------------------------------------------- scoring


@pytest.mark.parametrize("raw,want", [
    ("B", 1), ("b", 1), ("(B)", 1), ("[b]", 1), ("B) Up and right", 1), ("B. text", 1), ("b.", 1),
    (" Answer: C ", 2), ("option d", 3), ("**A**", 0), ("Choice B", 1),
    ("E", None), ("Z", None), ("", None), ("A man walks", None), ("1", None), ("Up and right", None),
    (None, None), (3, None),
])
def test_normalize_choice(raw, want):
    assert M.normalize_choice(raw, 4) == want


def test_score_mcq():
    truth = MCQTruth(video_id="v", duration=5, source="s", items=(
        {"id": "q1", "question": "?", "options": ("a", "b"), "answer_index": 1, "category": "c1"},
        {"id": "q2", "question": "?", "options": ("a", "b", "c", "d"), "answer_index": 0, "category": "c2"},
        {"id": "q3", "question": "?", "options": ("a", "b", "c", "d"), "answer_index": 3, "category": "c2"},
        {"id": "q4", "question": "?", "options": ("a", "b"), "answer_index": 0, "category": "c1"}))
    rep = MCQReport.model_validate({"answers": [
        {"id": "q1", "choice": "(B)"}, {"id": "q2", "choice": "a"}, {"id": "q2", "choice": "d"},  # first wins
        {"id": "q3", "choice": "E"}]})  # invalid; q4 missing
    s = M.score_mcq(truth, rep)
    assert (s.correct, s.total) == (2, 4)
    assert [r.correct for r in s.rows] == [True, True, False, False]
    assert [r.answered for r in s.rows] == [True, True, False, False]
    assert s.chance == pytest.approx(0.5 + 0.25 + 0.25 + 0.5)
    none = M.score_mcq(truth, None)
    assert none.correct == 0 and none.total == 4


# ---------------------------------------------------------------- manifest


def _item(vid, task=Task.MCQ):
    return DatasetItem(video_id=vid, task=task, video_path=f"x/{vid}.mp4", truth_path=f"x/{vid}.json",
                       split="test", bucket="b")


def test_manifest_replace_is_idempotent_and_keeps_other_lines(tmp_path):
    t1 = '{"video_id":"t1_a","task":"T1","video_path":"t1/a.mp4","truth_path":"t1/a.json","split":"test","bucket":"x"}'
    (tmp_path / "manifest.jsonl").write_text(t1 + "\n")
    for _ in range(2):
        T.replace_t5_in_manifest(tmp_path, [_item("t5_a"), _item("t5_b")])
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert lines[0] == t1 and len(lines) == 3
    T.replace_t5_in_manifest(tmp_path, [_item("t5_c")])
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert [json.loads(x)["video_id"] for x in lines] == ["t1_a", "t5_c"]
    assert not list(tmp_path.glob(".manifest.jsonl.*"))  # no temp leftovers


def test_manifest_lock_serialises_concurrent_writers(tmp_path):
    """A writer holding the lock blocks replace_t5_in_manifest until released."""
    started, done = threading.Event(), threading.Event()
    with T.manifest_lock(tmp_path):
        def work():
            started.set()
            T.replace_t5_in_manifest(tmp_path, [_item("t5_a")])
            done.set()
        th = threading.Thread(target=work)
        th.start()
        started.wait(2)
        assert not done.wait(0.3)  # blocked while we hold the lock
    th.join(5)
    assert done.is_set()
    assert (tmp_path / "manifest.jsonl").exists()


def test_sources_block_replaced_not_duplicated(tmp_path):
    (tmp_path / "SOURCES.md").write_text("# Dataset sources\n\nother stuff\n")
    T.write_sources(tmp_path, {"a": 3})
    T.write_sources(tmp_path, {"a": 4, "b": 2})
    s = (tmp_path / "SOURCES.md").read_text()
    assert s.count("tvbench:start") == 1 and "other stuff" in s
    assert "CC-BY-4.0" in s and "`b` (2)" in s and "`a` (3)" not in s


# ---------------------------------------------------------------- end to end with a mocked downloader


class FakeFetcher:
    def __init__(self, video, rows_by_cat):
        self.video, self.rows = video, rows_by_cat
        self.calls = []

    def size(self, path):
        return 1000

    def text(self, path):
        self.calls.append(path)
        return json.dumps(self.rows[path.split("/")[-1][:-5]])

    def _build(self, target, path):
        with zipfile.ZipFile(target, "w") as zf:
            for i in range(4):
                zf.write(self.video, f"{path.split('/')[-1][:-4]}/vid{i}.mp4")

    def file(self, path, dest):
        self.calls.append(path)
        self._build(dest, path)

    def range_zip(self, path):
        self.calls.append("range:" + path)
        buf = io.BytesIO()
        self._build(buf, path)
        return zipfile.ZipFile(buf)


def test_build_end_to_end_mocked(tmp_path, tone_video):
    rows = {
        "scene_transition": [row(f"vid{i}.mp4", answer="B2.", cands=("B1.", "B2.")) for i in range(4)],
        "action_sequence": [row(f"vid{i}.mp4", start=1.0, end=3.0) for i in range(4)],
    }
    f = FakeFetcher(tone_video, rows)
    (tmp_path / "manifest.jsonl").write_text(
        '{"video_id":"t1_a","task":"T1","video_path":"t1/a.mp4","truth_path":"t1/a.json","split":"test","bucket":"x"}\n')
    cats = ("scene_transition", "action_sequence")
    items = T.build_tvbench(tmp_path, per_category=4, seed=1, categories=cats, fetcher=f, log=lambda _: None)
    again = T.build_tvbench(tmp_path, per_category=4, seed=1, categories=cats, fetcher=f, log=lambda _: None)
    assert items == again and len(items) == 8
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 9 and sum('"T5"' in ln for ln in lines) == 8
    assert sorted({i.split for i in items}) == ["pilot", "test"]
    assert [i.split for i in items if i.bucket == "scene_transition"] == ["pilot", "pilot", "test", "test"]
    truth = MCQTruth.model_validate_json((tmp_path / items[0].truth_path).read_text())
    assert truth.source == "TVBench/scene_transition" and truth.items[0].answer_index == 1
    assert truth.duration == pytest.approx(4.0, abs=0.3)
    trimmed = next(i for i in items if i.bucket == "action_sequence")
    assert MCQTruth.model_validate_json((tmp_path / trimmed.truth_path).read_text()).duration == pytest.approx(2.0, abs=0.3)
    assert (tmp_path / items[0].video_path).exists()
    assert not list((tmp_path / "_cache" / "tvbench").glob("*.zip"))  # zips deleted
    assert "TVBench" in (tmp_path / "SOURCES.md").read_text()


@pytest.mark.parametrize("full", [False, True])
def test_build_range_vs_full_download(tmp_path, tone_video, full):
    f = FakeFetcher(tone_video, {"scene_transition": [row(f"vid{i}.mp4") for i in range(4)]})
    T.build_tvbench(tmp_path, per_category=2, seed=1, categories=("scene_transition",), fetcher=f,
                    full_download=full, log=lambda _: None)
    assert ("video/scene_transition.zip" in f.calls) is full
    assert ("range:video/scene_transition.zip" in f.calls) is (not full)
    assert not list((tmp_path / "_cache" / "tvbench").glob("*.zip"))


def test_range_file_serves_zip_from_ranged_reads():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a/x.bin", b"x" * 3000)
        zf.writestr("a/y.bin", bytes(range(256)) * 40)
    data, reads = buf.getvalue(), []

    def read_range(a, b):
        reads.append((a, b))
        return data[a:b]

    rf = T.RangeFile(len(data), read_range)
    rf.BLOCK = 64
    with zipfile.ZipFile(rf) as zf:
        assert zf.read("a/y.bin") == bytes(range(256)) * 40
        assert zf.read("a/x.bin") == b"x" * 3000
    assert all(0 <= a < b <= len(data) for a, b in reads)


def test_build_refuses_oversized_download(tmp_path):
    class Big:
        def size(self, p):
            return 10**10
    with pytest.raises(ValueError, match="cap"):
        T.build_tvbench(tmp_path, per_category=1, seed=1, categories=("object_count",), fetcher=Big(),
                        full_download=True, log=lambda _: None)
