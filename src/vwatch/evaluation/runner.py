"""Runs arms over the dataset and appends RunRecords to eval/runs/<run_id>.jsonl.

Resumable: a (video, task) already present in a run file is skipped, so an interrupted run
(rate limit, laptop sleep) continues where it stopped.

    python -m vwatch.evaluation.runner --root eval/data --split pilot --arms B0,B1,B2,W \
        --tasks T1,T2,T4 --budget 25000 --repeats 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from vwatch.arms import VideoContext, run_arm
from vwatch.llm import default_backend
from vwatch.llm.base import Backend
from vwatch.schemas import (Arm, CausalTruth, DatasetItem, DefectTruth, EventTruth, MCQTruth, QATruth, RunRecord,
                             Task)
from vwatch.tasks import task_prompt

TRUTH_MODEL = {Task.DEFECTS: DefectTruth, Task.EVENTS: EventTruth, Task.CAUSAL: CausalTruth, Task.QA: QATruth,
               Task.MCQ: MCQTruth}
TASK_LANGUAGE = {Task.DEFECTS: "hi"}  # the original T1 videos are Hindi Shorts


def language_for(item: DatasetItem) -> str | None:
    """Whisper language hint: Hindi for the original Shorts, auto-detect for everything else (T1x is multilingual)."""
    if item.video_id.startswith("t1x_"):
        return None
    return TASK_LANGUAGE.get(item.task)


@dataclass(frozen=True)
class Job:
    item: DatasetItem
    arm: Arm
    budget: int  # effective budget for this arm (B2 = 2x)
    repeat: int
    run_id: str


def load_manifest(root: Path) -> list[DatasetItem]:
    path = root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: build the dataset first (vw dataset build)")
    items = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                items.append(DatasetItem.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: invalid manifest line: {exc}") from exc
    return items


def run_id_for(date: str, arm: Arm, budget: int, repeat: int) -> str:
    return f"{date}_{arm.value}_{budget // 1000}k_r{repeat}"


def plan_jobs(items: Iterable[DatasetItem], arms: list[Arm], budgets: list[int], repeats: int,
              date: str) -> list[Job]:
    jobs = []
    for item in items:
        for budget in budgets:
            for arm in arms:
                eff = budget * 2 if arm is Arm.B2 else budget
                for r in range(1, repeats + 1):
                    jobs.append(Job(item, arm, eff, r, run_id_for(date, arm, eff, r)))
    return jobs


def done_keys(out_dir: Path, run_id: str) -> set[tuple[str, str]]:
    path = out_dir / f"{run_id}.jsonl"
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("error"):  # failed runs are retried
                keys.add((rec["video_id"], rec["task"]))
    return keys


def execute(job: Job, ctx: VideoContext, root: Path, backend: Backend) -> RunRecord:
    item = job.item
    truth = TRUTH_MODEL[item.task].model_validate_json((root / item.truth_path).read_text())
    prompt = task_prompt(item.task, truth)
    res = run_arm(job.arm, ctx, item.task, prompt, job.budget, backend)
    return RunRecord(run_id=job.run_id, arm=job.arm, budget_tokens=job.budget, repeat=job.repeat,
                     video_id=item.video_id, task=item.task, model=backend.model,
                     raw_output=res.raw_output, parsed=res.parsed, parse_error=res.parse_error,
                     content_tokens_est=res.content_tokens_est, usage=res.usage, cost_usd=res.cost_usd,
                     wall_s=res.wall_s, turns=res.turns, observations=res.observations,
                     tool_calls=res.tool_calls, error=res.error)


def run_all(jobs: list[Job], root: Path, out_dir: Path, backend: Backend, *, workers: int = 2,
            log: Callable[[str], None] = print) -> list[RunRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {rid: done_keys(out_dir, rid) for rid in {j.run_id for j in jobs}}
    todo = [j for j in jobs if (j.item.video_id, j.item.task.value) not in done[j.run_id]]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    contexts: dict[str, VideoContext] = {}
    for j in todo:  # perception is CPU/GPU bound: do it up front, serially
        vid = j.item.video_id
        if vid not in contexts:
            contexts[vid] = VideoContext.load(root / j.item.video_path, language=language_for(j.item))
            log(f"  prepared {vid} ({len(contexts)}/{len({x.item.video_id for x in todo})})")
    lock = threading.Lock()
    records: list[RunRecord] = []
    total_cost = 0.0

    def one(job: Job) -> RunRecord:
        return execute(job, contexts[job.item.video_id], root, backend)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, j): j for j in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            rec = fut.result()
            with lock:
                with open(out_dir / f"{job.run_id}.jsonl", "a") as f:
                    f.write(rec.model_dump_json() + "\n")
                records.append(rec)
                total_cost += rec.cost_usd
            status = "ERROR " + rec.error[:120] if rec.error else ("parse-fail" if rec.parsed is None else "ok")
            log(f"[{k}/{len(todo)}] {job.arm.value:8s} {job.budget // 1000}k {rec.video_id} {rec.task.value}: "
                f"{status} | {rec.content_tokens_est} est tok, {rec.turns} turns, ${rec.cost_usd:.3f} "
                f"(total ${total_cost:.2f})")
    return records


def limit_per_bucket(items: list[DatasetItem], n: int) -> list[DatasetItem]:
    """Keep at most n items per (task, bucket), in manifest order."""
    seen: dict[tuple[str, str], int] = {}
    kept = []
    for i in items:
        key = (i.task.value, i.bucket)
        if seen.get(key, 0) < n:
            kept.append(i)
            seen[key] = seen.get(key, 0) + 1
    return kept


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vw eval run", description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="eval/data", type=Path)
    ap.add_argument("--out", default="eval/runs", type=Path)
    ap.add_argument("--split", default="pilot", choices=["pilot", "test", "all"])
    ap.add_argument("--tasks", default="T1,T2,T4")
    ap.add_argument("--arms", default="B0,B1,B2,W")
    ap.add_argument("--budget", default="25000", help="comma-separated token caps, e.g. 10000,25000,50000")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="max items per task (0 = all)")
    ap.add_argument("--ids-file", type=Path, default=None, help="only video_ids listed in this file (one per line)")
    ap.add_argument("--id-prefix", default="", help="only items whose video_id starts with one of these (comma-separated)")
    ap.add_argument("--per-bucket", type=int, default=0,
                    help="max items per (task, bucket), e.g. per defect type (0 = all)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    a = ap.parse_args(argv)

    tasks = [Task(t) for t in _csv(a.tasks)]
    arms = [Arm(x) for x in _csv(a.arms)]
    budgets = [int(b) for b in _csv(a.budget)]
    items = [i for i in load_manifest(a.root) if i.task in tasks and (a.split == "all" or i.split == a.split)]
    if a.limit:
        per: dict[Task, int] = {}
        kept = []
        for i in items:
            if per.get(i.task, 0) < a.limit:
                kept.append(i)
                per[i.task] = per.get(i.task, 0) + 1
        items = kept
    if a.ids_file:
        wanted = {ln.strip() for ln in a.ids_file.read_text().splitlines() if ln.strip()}
        items = [i for i in items if i.video_id in wanted]
    if a.id_prefix:
        prefixes = tuple(_csv(a.id_prefix))
        items = [i for i in items if i.video_id.startswith(prefixes)]
    if a.per_bucket:
        items = limit_per_bucket(items, a.per_bucket)
    if not items:
        print("no dataset items match the filters", file=sys.stderr)
        return 1
    jobs = plan_jobs(items, arms, budgets, a.repeats, a.date)
    backend = default_backend(model=a.model, effort=a.effort)
    recs = run_all(jobs, a.root, a.out, backend, workers=a.workers)
    errors = sum(1 for r in recs if r.error)
    print(f"done: {len(recs)} runs, {errors} errors, ${sum(r.cost_usd for r in recs):.2f}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
