"""Report entry point: load runs/truths, build scorecard, write markdown + PNGs.

    python -m vwatch.evaluation.report --runs 'eval/runs/*.jsonl' --root eval/data \
        --budget 25000 --out eval/results/pilot.md [--judge [MODEL]] [--model M]

Records are deduplicated per (run_id, video, task) (last wins, a success beats a failure) and
arms are never pooled across models: several models in the inputs require --model.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from vwatch.evaluation.plots import plot_pareto, plot_recall_by_duration
from vwatch.evaluation.render import render_markdown
from vwatch.evaluation.runs import dedupe_runs, select_model
from vwatch.evaluation.scorecard import build_scorecard
from vwatch.schemas import (
    CausalTruth,
    DatasetItem,
    DefectTruth,
    EventTruth,
    MCQTruth,
    QATruth,
    RunRecord,
    Task,
)

__all__ = [
    "build_scorecard", "dedupe_runs", "load_runs", "load_truths", "main",
    "plot_pareto", "plot_recall_by_duration", "render_markdown",
]

_TRUTH_MODEL = {
    Task.DEFECTS: DefectTruth,
    Task.EVENTS: EventTruth,
    Task.CAUSAL: CausalTruth,
    Task.QA: QATruth,
    Task.MCQ: MCQTruth,
}


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                yield n, line


def load_runs(paths: Iterable[str | Path]) -> list[RunRecord]:
    out = []
    for p in map(Path, paths):
        for n, line in _jsonl(p):
            try:
                out.append(RunRecord.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"{p}:{n}: invalid run record: {e}") from e
    return out


def load_truths(root: str | Path) -> dict[tuple[str, Task], object]:
    root = Path(root)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    truths: dict[tuple[str, Task], object] = {}
    for n, line in _jsonl(manifest):
        try:
            item = DatasetItem.model_validate_json(line)
        except ValidationError as e:
            raise ValueError(f"{manifest}:{n}: invalid manifest item: {e}") from e
        k = (item.video_id, item.task)
        if k in truths:
            raise ValueError(f"{manifest}:{n}: duplicate manifest entry for {k}")
        tpath = root / item.truth_path
        try:
            truths[k] = _TRUTH_MODEL[item.task].model_validate_json(tpath.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as e:
            raise ValueError(f"{tpath}: cannot load truth for {k}: {e}") from e
    return truths


def _load_judge(model: str) -> Callable[[str], str]:
    try:
        mod = importlib.import_module("vwatch.llm.judge")
        return mod.make_judge(model)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "--judge needs vwatch.llm.judge.make_judge(model) -> Callable[[str], str]; "
            f"import failed: {e}"
        ) from e


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vwatch.evaluation.report")
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) of run JSONL files")
    ap.add_argument("--root", default="eval/data")
    ap.add_argument("--budget", type=int, required=True, help="matched token budget X")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", nargs="?", const="sonnet", default=None, metavar="MODEL",
                    help="score T3 with vwatch.llm.judge.make_judge(MODEL); omit for T3 n/a")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default=None)
    ap.add_argument("--model", default=None,
                    help="select this model's records; required when the run files hold several models")
    ap.add_argument("--commit", default="n/a")
    ap.add_argument("--locked-on", default="n/a")
    args = ap.parse_args(argv)

    files = sorted({f for pat in args.runs for f in glob.glob(pat)})
    if not files:
        raise SystemExit(f"no run files match {args.runs}")
    runs = load_runs(files)
    if not runs:
        raise SystemExit("run files contained no records")
    try:
        runs = select_model(dedupe_runs(runs), args.model)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    judge = _load_judge(args.judge) if args.judge else None
    sc = build_scorecard(runs, load_truths(args.root), matched_budget=args.budget,
                         judge=judge, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recall_png, pareto_png = f"{out.stem}_recall_by_duration.png", f"{out.stem}_pareto.png"
    plot_recall_by_duration(sc, out.parent / recall_png)
    plot_pareto(sc, out.parent / pareto_png)
    meta = {"date": args.date or "undated", "model": args.model or runs[0].model, "commit": args.commit,
            "locked_on": args.locked_on, "recall_png": recall_png, "pareto_png": pareto_png}
    out.write_text(render_markdown(sc, meta), encoding="utf-8")
    print(f"wrote {out} (verdict: {sc['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
