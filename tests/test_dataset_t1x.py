"""T1x: Commons-sourced generic defects. No network: the Commons client is a fake and the sources are tiny."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from test_dataset_fixtures import make_audio, make_frames

from vwatch import media
from vwatch.dataset import _commons, _segment, t1x
from vwatch.dataset._common import FPS, SPF
from vwatch.dataset._segment import CommonsSource
from vwatch.dataset._source import SourceSpec, analyse, load_source
from vwatch.schemas import DatasetItem, DefectTruth, DefectType
from vwatch.transcribe import Segment, Transcript

SECS = 9


def fake_info(title: str) -> _commons.FileInfo:
    return _commons.FileInfo(title=title, url=f"https://example.invalid/{title}.mp4",
                             page_url=f"https://commons.example.invalid/{title}", author="Some Author",
                             license="CC BY-SA 4.0", size=1, duration=float(SECS), width=160, height=90)


class FakeClient:
    """Writes a tiny 160x90 landscape clip (moving block, voiced tone with short pauses) instead of downloading."""

    def __init__(self) -> None:
        self.downloads: list[str] = []

    def info(self, title: str) -> _commons.FileInfo:
        return fake_info(title)

    def download(self, url: str, dest: Path) -> None:
        self.downloads.append(url)
        n = SECS * FPS
        fr = np.zeros((n, 90, 160, 3), np.uint8)
        for i in range(n):
            fr[i] = (40 + i % 50, 90, 140)
            x = (i * 3) % 130
            fr[i, 30:50, x:x + 20] = (230, 200, 40)
        t = np.arange(n * SPF) / 44100
        a = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        media.write_video(fr, FPS, dest, audio=a)


def fake_transcriber(path: Path) -> Transcript:
    dur = media.probe(path).duration
    return Transcript("en", (Segment(0.0, dur * 0.9, "hello"),), ())


@pytest.fixture(autouse=True)
def tiny_segments(monkeypatch):
    monkeypatch.setattr(_segment, "MIN_SEG", 4.0)
    monkeypatch.setattr(_segment, "LEAD_IN", 0.5)


SRCS = (CommonsSource("alpha", "File:Alpha.mp4", "English", seg_len=6.0),
        CommonsSource("beta", "File:Beta.mp4", "Hindi", seg_len=7.0))


def synth_loader(spec: SourceSpec):
    return analyse(spec, make_frames(), make_audio())  # 8 s, 108x192 with a cut and a pause


# ------------------------------------------------------------------ commons helpers


def test_license_gate_and_parse():
    for ok in ("CC BY 3.0", "CC BY-SA 4.0", "CC0", "Public domain", "PD-USGov"):
        assert _commons.license_ok(ok)
    for bad in ("CC BY-NC 4.0", "CC BY-ND 2.0", "All rights reserved", "", "GFDL"):
        assert not _commons.license_ok(bad)
    page = {"title": "File:X.webm", "imageinfo": [{"url": "https://u/x.webm?utm=1", "size": 5, "width": 2,
            "height": 4, "duration": 3.5, "descriptionurl": "https://p",
            "extmetadata": {"LicenseShortName": {"value": "CC BY 3.0"},
                            "Artist": {"value": '<a href="x">Jane &amp; Co</a>'}}}]}
    info = _commons.parse_info(page)
    assert info.url == "https://u/x.webm" and info.author == "Jane & Co" and info.duration == 3.5
    page["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = "Fair use"
    with pytest.raises(_commons.CommonsError):
        _commons.parse_info(page)
    with pytest.raises(_commons.CommonsError):
        _commons.parse_info({"title": "File:Missing.webm"})


# ------------------------------------------------------------------ segment selection


def test_rank_windows_prefers_dense_speech_and_is_non_overlapping():
    mask = np.zeros(100 * 100, bool)  # 100 s
    mask[3000:8000] = True  # speech 30-80 s
    starts = _segment.rank_windows(mask, 100.0, 40.0)
    assert 30 <= starts[0] <= 40 and all(abs(a - b) >= 40 for a in starts for b in starts if a != b)
    assert _segment.rank_windows(mask, 3.0, 40.0) == []  # shorter than the minimum segment


def test_coverage_union_of_segments():
    tr = Transcript("en", (Segment(0, 10, "a"), Segment(5, 15, "b"), Segment(30, 100, "c")), ())
    assert _segment.coverage(tr, 40.0) == pytest.approx((15 + 10) / 40)
    assert _segment.coverage(Transcript("", (), ()), 40.0) == 0.0


def test_prepare_source_cuts_normalises_and_caches(tmp_path):
    client = FakeClient()
    p = _segment.prepare_source(client, SRCS[0], tmp_path, fake_transcriber)
    info = media.probe(p.segment_path)
    assert info.width == 540 and info.height % 2 == 0 and info.height == 304
    assert info.fps == pytest.approx(30) and info.has_audio
    assert p.duration == pytest.approx(6.0) and p.coverage >= _segment.MIN_COVERAGE
    assert abs(info.duration - p.duration) < 0.1
    again = _segment.prepare_source(client, SRCS[0], tmp_path, lambda _p: pytest.fail("transcribed twice"))
    assert again.start == p.start and len(client.downloads) == 1


def test_prepare_source_rejects_low_speech(tmp_path):
    silent = lambda _p: Transcript("en", (Segment(0.0, 1.0, "x"),), ())  # noqa: E731
    with pytest.raises(_segment.SegmentError, match="speech coverage"):
        _segment.prepare_source(FakeClient(), SRCS[0], tmp_path, silent)
    assert not (tmp_path / "alpha.segment.mp4").exists()


def test_generic_source_loads_without_captions(tmp_path):
    p = _segment.prepare_source(FakeClient(), SRCS[0], tmp_path, fake_transcriber)
    src = load_source(SourceSpec("alpha", p.segment_path))
    assert src.frames.shape[1:3] == (304, 540) and not src.cap_on.any() and not src.cap_diff.any()
    assert src.n == pytest.approx(6 * FPS, abs=2) and src.speech.mean() > 0.8
    assert not SourceSpec("alpha", p.segment_path).has_captions


# ------------------------------------------------------------------ jobs, splits, manifest


def test_plan_jobs_ids_and_counts():
    jobs = t1x.plan_source_jobs("alpha", 0, 7)
    assert len(jobs) == 10 and len({j.video_id for j in jobs}) == 10
    assert all(j.video_id.startswith("t1x_alpha_") for j in jobs)
    kinds = [j.defect for j in jobs]
    assert {k for k in kinds if k} == set(t1x.GENERIC_TYPES) and kinds.count(None) == 2
    assert DefectType.CAPTION_BLOCKED not in kinds and DefectType.WRONG_NUMBER not in kinds
    assert jobs == t1x.plan_source_jobs("alpha", 0, 7) and jobs != t1x.plan_source_jobs("alpha", 0, 8)
    flash = sorted(j.frac for j in jobs if j.defect is DefectType.FLASH_FRAME)
    assert flash[0] < 0.5 < flash[1]  # stratified positions


def test_splits_two_pilot_per_source_distinct_buckets():
    for si in range(6):
        jobs = t1x.plan_source_jobs(f"s{si}", si, 7)
        sp = t1x.assign_t1x_splits(jobs, 7, si)
        pilot = [j for j in jobs if sp[j.video_id] == "pilot"]
        assert len(pilot) == 2 and len({j.defect for j in pilot}) == 2
        assert sp == t1x.assign_t1x_splits(jobs, 7, si)


def item(vid: str) -> DatasetItem:
    return DatasetItem(video_id=vid, task="T1", video_path=f"t1/{vid}.mp4", truth_path=f"t1/{vid}.truth.json",
                       split="test", bucket="clean")


def test_manifest_replaces_only_t1x_lines_and_is_idempotent(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(item("t1_a").model_dump_json() + "\n" + item("t1x_old_1").model_dump_json()
                                             + "\n" + item("t2_b").model_dump_json() + "\n")
    new = [item("t1x_new_0"), item("t1x_new_1")]
    t1x.update_manifest(tmp_path, new)
    t1x.update_manifest(tmp_path, new)
    ids = [json.loads(ln)["video_id"] for ln in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert ids == ["t1_a", "t2_b", "t1x_new_0", "t1x_new_1"]
    assert not list(tmp_path.glob(".manifest*tmp"))


def test_manifest_concurrent_writers_keep_foreign_lines(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(item("t1_a").model_dump_json() + "\n")
    errs: list[BaseException] = []

    def foreign():  # another process appending under the same lock protocol
        import fcntl
        try:
            for k in range(20):
                with open(tmp_path / ".manifest.lock", "a+") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    with open(tmp_path / "manifest.jsonl", "a") as f:
                        f.write(item(f"t5_{k}").model_dump_json() + "\n")
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    th = threading.Thread(target=foreign)
    th.start()
    for _ in range(10):
        t1x.update_manifest(tmp_path, [item("t1x_x_0")])
    th.join()
    ids = [json.loads(ln)["video_id"] for ln in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert not errs and ids.count("t1x_x_0") == 1 and sum(i.startswith("t5_") for i in ids) == 20


def test_sources_md_append_only_and_deduplicated(tmp_path):
    p = _segment.prepare_source(FakeClient(), SRCS[0], tmp_path / "c", fake_transcriber)
    (tmp_path / "SOURCES.md").write_text("# Sources\n\nexisting text\n")
    assert t1x.append_sources_md(tmp_path, [p]) is True
    assert t1x.append_sources_md(tmp_path, [p]) is False
    text = (tmp_path / "SOURCES.md").read_text()
    assert text.startswith("# Sources\n\nexisting text\n") and text.count("T1 extension (t1x)") == 1
    assert "Some Author" in text and "CC BY-SA 4.0" in text and "Alpha.mp4" in text


# ------------------------------------------------------------------ end to end


def test_build_t1x_end_to_end(tmp_path):
    items, checks = t1x.build_t1x(tmp_path, 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader,
                                  log=lambda _m: None)
    assert len(items) == 10 and len(checks) == 10
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert sorted({i.bucket for i in items}) == ["audio_dropout", "clean", "flash_frame", "frozen_frame", "jump_cut"]
    assert sum(i.split == "pilot" for i in items) == 2
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert [DatasetItem.model_validate_json(x).video_id for x in lines] == [i.video_id for i in items]
    for it in items:
        truth = DefectTruth.model_validate_json((tmp_path / it.truth_path).read_text())
        assert truth.source == "alpha" and len(truth.defects) == (0 if it.bucket == "clean" else 1)
        assert (tmp_path / it.video_path).exists() and it.video_id.startswith("t1x_alpha_")
    assert "T1 extension (t1x)" in (tmp_path / "SOURCES.md").read_text()
    before = (tmp_path / "manifest.jsonl").read_text()
    t1x.build_t1x(tmp_path, 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader, log=lambda _m: None)
    assert (tmp_path / "manifest.jsonl").read_text() == before


def test_build_t1x_does_not_publish_when_a_check_fails(tmp_path, monkeypatch):
    from vwatch.dataset import _t1x_verify

    monkeypatch.setattr(_t1x_verify, "verify_video", lambda path, truth, src, small: _t1x_verify.Check(truth.video_id, False, "x"))
    items, checks = t1x.build_t1x(tmp_path, 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader,
                                  log=lambda _m: None)
    assert not any(c.ok for c in checks) and not (tmp_path / "manifest.jsonl").exists()


def test_generic_source_with_height_not_divisible_by_four(tmp_path):
    """540x298 (e.g. a 870x480 original) used to break the 4x gray downscale."""
    class Odd(FakeClient):
        def download(self, url, dest):
            fr = np.zeros((SECS * FPS, 88, 160, 3), np.uint8)
            fr[:, 20:40, ::2] = 200
            media.write_video(fr, FPS, dest, audio=np.zeros(SECS * FPS * SPF, np.float32) + 0.1)

    p = _segment.prepare_source(Odd(), SRCS[0], tmp_path, fake_transcriber)
    src = load_source(SourceSpec("odd", p.segment_path))
    assert src.frames.shape[1:3] == (298, 540) and src.gray.shape[1:] == (75, 135)


def test_verify_only_reuses_rendered_files(tmp_path):
    first, _ = t1x.build_t1x(tmp_path, 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader,
                             log=lambda _m: None)
    (tmp_path / "manifest.jsonl").unlink()
    mtimes = {i.video_id: (tmp_path / i.video_path).stat().st_mtime_ns for i in first}
    again, checks = t1x.build_t1x(tmp_path, 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader,
                                  render=False, log=lambda _m: None)
    assert again == first and all(c.ok for c in checks) and (tmp_path / "manifest.jsonl").exists()
    assert mtimes == {i.video_id: (tmp_path / i.video_path).stat().st_mtime_ns for i in again}
    with pytest.raises(FileNotFoundError):
        t1x.build_t1x(tmp_path / "empty", 7, SRCS[:1], FakeClient(), fake_transcriber, loader=synth_loader,
                      render=False, log=lambda _m: None)
