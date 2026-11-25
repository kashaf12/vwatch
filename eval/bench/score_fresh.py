"""Score the fresh held-out benchmark exactly as docs/fresh-bench-protocol.md fixes it.

  uv run python eval/bench/score_fresh.py [--runs 'eval/runs/2026-11-24fresh_*.jsonl'] [--root eval/data/fresh]

Primary metric: defect F1 over all 48 defect items (40 short + 8 long), scored as labeled, failed runs count.
Paired differences: cluster bootstrap over sources (10,000 resamples, seed 0).
Claim rule: "best" only if every paired 95% CI is above 0; "tied" if any includes 0; "behind" if one is below.

Also prints a breakdown that the protocol did not pre-register (added 2026-11-25 after a reader asked for it):
false alarms by reported type and per video, and a paired per-video comparison. Same runs, same scoring, no
exclusions; it describes the result above and makes no new claim.
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path

from vwatch.evaluation.cells import score_records
from vwatch.evaluation.compare import compare
from vwatch.evaluation.report import load_runs, load_truths
from vwatch.evaluation.runs import dedupe_runs
from vwatch.evaluation.scoring import _norm_type
from vwatch.evaluation.stats import f1
from vwatch.schemas import Task

ARMS = ("VWP", "CC", "CVV")
NAMES = {"VWP": "Claude Code + vwatch plugin", "CC": "Plain Claude Code", "CVV": "claude-video-vision"}


def arm_of(r) -> str:
    return r.run.arm.value if hasattr(r.run.arm, "value") else str(r.run.arm)


def false_alarms_by_type(parsed: object, found_types: list[str]) -> Counter:
    """Reported defects left unmatched, by type. Matching is same-type only, so per type:
    unmatched = reported - matched, and matched = truth rows of that type that were found."""
    reps = (parsed.get("defects") or []) if isinstance(parsed, dict) else []
    reported = Counter(_norm_type(str(r.get("type", ""))) for r in reps if isinstance(r, dict))
    found = Counter(found_types)
    return Counter({t: n - found[t] for t, n in reported.items() if n > found[t]})


def paired(a: dict[str, int], b: dict[str, int]) -> tuple[int, int, int]:
    """Per-video comparison over the videos both have: (a higher, a lower, equal)."""
    keys = a.keys() & b.keys()
    return (sum(a[k] > b[k] for k in keys), sum(a[k] < b[k] for k in keys), sum(a[k] == b[k] for k in keys))


def print_breakdown(by: dict) -> None:
    print("\n## Breakdown (not pre-registered; describes the same runs)\n")
    print("| Tool | False alarms | Per video | Videos with a false alarm | By reported type |\n|---|---|---|---|---|")
    hits: dict[str, dict[str, int]] = {}
    fps: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        d = by[arm][Task.DEFECTS]
        types: Counter = Counter()
        for x in d:
            types += false_alarms_by_type(x.run.parsed, [row.type for row in x.t1.rows if row.found])
        hits[arm] = {x.run.video_id: x.t1.tp for x in d}
        fps[arm] = {x.run.video_id: x.t1.fp for x in d}
        total = sum(fps[arm].values())
        with_fa = sum(1 for v in fps[arm].values() if v)
        by_type = ", ".join(f"{t} {n}" for t, n in types.most_common())
        print(f"| {NAMES[arm]} | {total} | {total / len(d):.2f} | {with_fa}/{len(d)} | {by_type} |")
    print("\nPaired per video (vwatch plugin vs each tool):\n")
    for b in ("CC", "CVV"):
        more, fewer, same = paired(hits["VWP"], hits[b])
        fmore, ffewer, fsame = paired(fps["VWP"], fps[b])
        print(f"- vs {NAMES[b]}: found more on {more}, fewer on {fewer}, same on {same}; "
              f"false alarms fewer on {ffewer}, more on {fmore}, same on {fsame}")


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
    print_breakdown(by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
