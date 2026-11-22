"""Shared constants and helpers for the dataset generators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vwatch.schemas import DatasetItem, Task

FPS = 30
WIDTH = 540
HEIGHT = 960
SR = 44100
SPF = SR // FPS  # audio samples per video frame (1470)
PILOT_FRACTION = 0.15

TASK_KEY = {Task.DEFECTS: 1, Task.EVENTS: 2, Task.CAUSAL: 3, Task.QA: 4}
TASK_DIR = {Task.DEFECTS: "t1", Task.EVENTS: "t2", Task.CAUSAL: "t3", Task.QA: "t4"}


@dataclass(frozen=True)
class ScaleSpec:
    t1_per_type: int
    t1_clean: int
    t2: int
    t3: int
    t4: int


SCALES: dict[str, ScaleSpec] = {
    "tiny": ScaleSpec(1, 1, 1, 1, 1),
    "pilot": ScaleSpec(2, 3, 3, 2, 2),
    "full": ScaleSpec(30, 60, 20, 12, 12),
}


def scale_spec(name: str) -> ScaleSpec:
    try:
        return SCALES[name]
    except KeyError:
        raise ValueError(f"unknown scale {name!r}; expected one of {sorted(SCALES)}") from None


def item_rng(seed: int, *keys: int) -> np.random.Generator:
    """Independent, reproducible RNG stream per (seed, keys...)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(k) for k in keys]]))


def assign_splits(video_ids: Sequence[str], seed: int, task: Task) -> dict[str, str]:
    """~15% pilot (at least one when there are 2+ items), rest test. Deterministic."""
    ids = list(video_ids)
    n = len(ids)
    k = 0 if n < 2 else max(1, round(PILOT_FRACTION * n))
    rng = item_rng(seed, TASK_KEY[task], 9999)
    pilot = set(rng.permutation(n)[:k].tolist())
    return {vid: ("pilot" if i in pilot else "test") for i, vid in enumerate(ids)}


def silent_audio(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SR)), dtype=np.float32)


def finish_item(root: Path, task: Task, video_id: str, truth: BaseModel, bucket: str) -> DatasetItem:
    """Write `<task dir>/<id>.truth.json` and return the manifest item (split set later)."""
    tdir = TASK_DIR[task]
    rel_truth = f"{tdir}/{video_id}.truth.json"
    path = root / rel_truth
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return DatasetItem(
        video_id=video_id,
        task=task,
        video_path=f"{tdir}/{video_id}.mp4",
        truth_path=rel_truth,
        split="test",
        bucket=bucket,
    )


def video_out(root: Path, task: Task, video_id: str) -> Path:
    return root / TASK_DIR[task] / f"{video_id}.mp4"
