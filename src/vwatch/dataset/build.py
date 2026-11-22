"""Build the whole dataset (T1-T4) plus `manifest.jsonl`.

    python -m vwatch.dataset.build --root eval/data --scale pilot --seed 7
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from vwatch.dataset._common import assign_splits, scale_spec
from vwatch.dataset._source import DEFAULT_SOURCES, SourceClip, SourceSpec, load_source
from vwatch.dataset.inject import build_t1
from vwatch.dataset.qa import build_t4
from vwatch.dataset.synth_causal import build_t3
from vwatch.dataset.synth_events import build_t2
from vwatch.schemas import DatasetItem, Task

MANIFEST = "manifest.jsonl"
BUILDABLE = (Task.DEFECTS, Task.EVENTS, Task.CAUSAL, Task.QA)  # T1x and T5 have their own builders


def with_splits(items: list[DatasetItem], seed: int, task: Task) -> list[DatasetItem]:
    splits = assign_splits([i.video_id for i in items], seed, task)
    return [i.model_copy(update={"split": splits[i.video_id]}) for i in items]


def write_manifest(root: Path, items: Sequence[DatasetItem]) -> Path:
    path = root / MANIFEST
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    return path


def merge_manifest(root: Path, items: Sequence[DatasetItem], built: Sequence[Task]) -> Path:
    """Replace the lines of the tasks just built and keep everything else (T5, `t1x_` items, other tasks).

    Runs under the same exclusive flock as the T1x and TVBench builders and writes atomically.
    """
    path = root / MANIFEST
    built_values = {t.value for t in built}
    with open(root / ".manifest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            kept = []
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    old = json.loads(line)
                    if old["task"] not in built_values or old["video_id"].startswith("t1x_"):
                        kept.append(line + "\n")
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(kept) + "".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return path


def available_sources(sources: Sequence[SourceSpec]) -> list[SourceSpec]:
    """Sources whose video file exists; warns about the rest."""
    present = [s for s in sources if Path(s.path).exists()]
    for s in sources:
        if s not in present:
            print(f"[build] source missing, skipped: {s.name} ({s.path})", file=sys.stderr)
    return present


def read_manifest(root: Path) -> list[DatasetItem]:
    lines = (root / MANIFEST).read_text(encoding="utf-8").splitlines()
    return [DatasetItem.model_validate_json(line) for line in lines if line.strip()]


def build_dataset(root: Path, *, scale: str, seed: int, sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
                  tasks: Sequence[Task] = BUILDABLE,
                  loader: Callable[[SourceSpec], SourceClip] = load_source) -> list[DatasetItem]:
    """Build every task's videos + truth files under `root` and write the manifest."""
    spec = scale_spec(scale)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items: list[DatasetItem] = []
    # A custom loader (tests) doesn't read files; only check the disk for the real loader.
    needs = Task.DEFECTS in tasks or Task.QA in tasks
    present = (available_sources(sources) if loader is load_source else list(sources)) if needs else []
    if Task.DEFECTS in tasks:
        if not present:
            raise FileNotFoundError("T1 needs source Shorts: none of the configured files exist. Put your own "
                                    "videos in VWATCH_SHORTS_DIR (see docs/DATA.md) or build other tasks with --tasks.")
        items += build_t1(root, scale, seed, present, loader)  # splits assigned inside
    if Task.EVENTS in tasks:
        items += with_splits(build_t2(root, spec.t2, seed), seed, Task.EVENTS)
    if Task.CAUSAL in tasks:
        items += with_splits(build_t3(root, spec.t3, seed), seed, Task.CAUSAL)
    if Task.QA in tasks:
        real = next((s for s in present if s.props_path is not None
                     and (loader is not load_source or Path(s.props_path).exists())), None)
        items += with_splits(build_t4(root, spec.t4, seed, real, loader), seed, Task.QA)
    merge_manifest(root, items, [t for t in tasks if t in BUILDABLE])
    return items


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vwatch.dataset.build", description=__doc__)
    p.add_argument("--root", type=Path, default=Path("eval/data"))
    p.add_argument("--scale", choices=["tiny", "pilot", "full"], default="pilot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tasks", nargs="*", choices=[t.value for t in BUILDABLE], default=[t.value for t in BUILDABLE],
                   help="subset of tasks to build (the manifest then only lists those)")
    args = p.parse_args(argv)
    items = build_dataset(args.root, scale=args.scale, seed=args.seed, tasks=tuple(Task(t) for t in args.tasks))
    counts: dict[str, int] = {}
    for it in items:
        key = f"{it.task.value}/{it.split}"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(items)} items to {args.root / MANIFEST}: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
