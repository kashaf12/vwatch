"""The fresh-benchmark breakdown helpers in eval/bench/score_fresh.py."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "eval" / "bench" / "score_fresh.py"
_spec = importlib.util.spec_from_file_location("score_fresh", _PATH)
assert _spec and _spec.loader
score_fresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_fresh)


def test_false_alarms_by_type_counts_only_unmatched_reports():
    parsed = {"defects": [{"type": "Flash Frame"}, {"type": "flash_frame"}, {"type": "jump-cut"}]}
    assert score_fresh.false_alarms_by_type(parsed, ["flash_frame"]) == Counter({"flash_frame": 1, "jump_cut": 1})


def test_false_alarms_by_type_is_empty_when_every_report_matched():
    parsed = {"defects": [{"type": "frozen_frame"}]}
    assert score_fresh.false_alarms_by_type(parsed, ["frozen_frame"]) == Counter()


def test_false_alarms_by_type_handles_unparsed_output():
    assert score_fresh.false_alarms_by_type(None, []) == Counter()
    assert score_fresh.false_alarms_by_type({"defects": None}, []) == Counter()


def test_paired_compares_only_shared_videos():
    a = {"v1": 2, "v2": 0, "v3": 1, "only_a": 5}
    b = {"v1": 1, "v2": 1, "v3": 1, "only_b": 0}
    assert score_fresh.paired(a, b) == (1, 1, 1)
