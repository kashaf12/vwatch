"""Score the fresh held-out benchmark exactly as docs/fresh-bench-protocol.md fixes it.

  uv run python eval/bench/score_fresh.py [--runs 'eval/runs/2026-11-24fresh_*.jsonl'] [--root eval/data/fresh]

Primary metric: defect F1 over all 48 defect items (40 short + 8 long), scored as labeled, failed runs count.
Paired differences: cluster bootstrap over sources (10,000 resamples, seed 0).
Claim rule: "best" only if every paired 95% CI is above 0; "tied" if any includes 0; "behind" if one is below.
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

from vwatch.evaluation.cells import score_records
from vwatch.evaluation.compare import compare
from vwatch.evaluation.report import load_runs, load_truths
from vwatch.evaluation.runs import dedupe_runs
from vwatch.evaluation.stats import f1
from vwatch.schemas import Task

ARMS = ("VWP", "CC", "CVV")
NAMES = {"VWP": "Claude Code + vwatch plugin", "CC": "Plain Claude Code", "CVV": "claude-video-vision"}


def arm_of(r) -> str:
    return r.run.arm.value if hasattr(r.run.arm, "value") else str(r.run.arm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="eval/runs/2026-11-24fresh_*.jsonl")
    ap.add_argument("--root", default="eval/data/fresh")
    args = ap.parse_args()
    runs = dedupe_runs(load_runs(sorted(glob.glob(args.runs))))
    truths = load_truths(args.root)
    recs = score_records(runs, truths, None)
    by: dict[str, dict[Task, list]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[arm_of(r)][r.run.task].append(r)
    sources = {v: t.source for (v, task), t in truths.items() if task == Task.DEFECTS}
    n_def = sum(1 for (_, task) in truths if task == Task.DEFECTS)

    out = ["| Tool | Items | Defect F1 | Precision | Recall | Clean videos with a false alarm | Failed runs | "
           "Code word | $/video-min short | $/video-min long | s/video-min short | s/video-min long |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    types: dict[str, dict[str, list[int]]] = {}
    for arm in ARMS:
        d = by[arm][Task.DEFECTS]
        tp, fp, fn = (sum(getattr(x.t1, k) for x in d) for k in ("tp", "fp", "fn"))
        clean = [x for x in d if not truths[(x.run.video_id, Task.DEFECTS)].defects]
        fa = sum(1 for x in clean if x.t1.false_alarm)
        failed = sum(1 for x in d + by[arm][Task.QA] if x.failed)
        qa = by[arm][Task.QA]
        cw = sum(x.t4.correct for x in qa if x.t4)
        cost: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # $ , s, minutes
        for x in d + qa:
            k = "long" if x.run.video_id.startswith("fxl") else "short"
            cost[k][0] += x.run.cost_usd or 0
            cost[k][1] += x.run.wall_s or 0
            cost[k][2] += x.duration / 60
        def per(k: str, i: int) -> str:
            c = cost[k]
            return f"{c[i] / c[2]:.2f}" if c[2] else "n/a"
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(f"| {NAMES[arm]} | {len(d)}/{n_def} | **{f1(tp, fp, fn):.2f}** | {prec:.2f} | {rec:.2f} | "
                   f"{fa}/{len(clean)} | {failed} | {cw}/{len(qa)} | ${per('short', 0)} | ${per('long', 0)} | "
                   f"{float(per('short', 1)):.0f} | {float(per('long', 1)):.0f} |")
        t: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for x in d:
            for row in x.t1.rows:
                t[row.type][1] += 1
                t[row.type][0] += int(bool(row.found))
        types[arm] = t
    print("\n".join(out))
    print("\n| Defect type | " + " | ".join(NAMES[a] for a in ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for ty in sorted({k for t in types.values() for k in t}):
        print(f"| {ty} | " + " | ".join(f"{types[a][ty][0]}/{types[a][ty][1]}" for a in ARMS) + " |")

    print("\nPaired defect-F1 differences (vwatch plugin minus each tool), cluster bootstrap over sources:\n")
    verdicts = []
    for b in ("CC", "CVV"):
        est = compare(by["VWP"][Task.DEFECTS], by[b][Task.DEFECTS], n_boot=10000, seed=0, sources=sources)["t1_f1"]
        lo, hi = est["lo"] * 100, est["hi"] * 100
        v = "ahead" if lo > 0 else "behind" if hi < 0 else "tied"
        verdicts.append(v)
        print(f"- vs {NAMES[b]}: {est['point'] * 100:+.1f} pp, 95% CI [{lo:+.1f}, {hi:+.1f}] -> {v}")
    claim = ("BEST" if all(v == "ahead" for v in verdicts) else
             "BEHIND" if any(v == "behind" for v in verdicts) else "TIED")
    print(f"\nProtocol claim on the primary metric: **{claim}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
