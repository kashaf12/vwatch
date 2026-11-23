"""Model-free evaluation of the signal candidates (no model calls).

For each defect type: detector recall (a truth defect is "hit" when a candidate of the matching kind
overlaps it within +-0.5 s), per-kind precision (share of a kind's candidates that land on a truth
defect of the matching type), and candidates per minute on clean videos (false-alarm load).

Sets:
  dev    t1x_* videos NOT in eval/bench/t1x_ids.txt (tuning set)
  bench  the 40 held-out t1x ids of the head-to-head benchmark (report once, never tune on it)
  t1     the original t1_* videos (extra dev set)
  fdev   fresh frozen-frame injections into the T1x sources (eval/bench/make_freeze_dev.py; tuning set)

Usage:
  uv run python eval/bench/eval_signals.py dev [--json out.json] [--workers 8]
  uv run python eval/bench/eval_signals.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "eval" / "data"
BENCH_IDS = ROOT / "eval" / "bench" / "t1x_ids.txt"
TOL = 0.5

# defect type -> candidate kinds that count as detecting it
MATCH: dict[str, tuple[str, ...]] = {
    "flash_frame": ("flash",),
    "frozen_frame": ("freeze",),
    "jump_cut": ("jump_cut",),
    "audio_dropout": ("silence_in_speech",),
    "caption_blocked": ("local_change",),
    "wrong_number": ("local_change",),
}
# a weaker "proxy" kind the agent could also have used before a dedicated kind existed
PROXY: dict[str, tuple[str, ...]] = {"jump_cut": ("cut_in_speech", "cut")}
KIND_TO_TYPE = {"flash": "flash_frame", "freeze": "frozen_frame", "jump_cut": "jump_cut",
                "silence_in_speech": "audio_dropout"}


def select(which: str) -> list[dict]:
    bench = set(BENCH_IDS.read_text().split())
    items = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if which == "dev":
        return [m for m in items if m["video_id"].startswith("t1x_") and m["video_id"] not in bench]
    if which == "bench":
        return [m for m in items if m["video_id"] in bench]
    if which == "t1":
        return [m for m in items if m["video_id"].startswith("t1_")]
    if which == "fdev":
        return [{"video_id": p.stem, "video_path": f"fdev/t1/{p.name}", "truth_path": f"fdev/t1/{p.stem}.truth.json",
                 "bucket": "clean" if "_clean_" in p.stem else "frozen_frame"}
                for p in sorted((DATA / "fdev" / "t1").glob("*.mp4"))]
    raise SystemExit(f"unknown set {which!r} (dev | bench | t1 | fdev)")


def run_one(item: dict) -> dict:
    from vwatch import signals

    path = DATA / item["video_path"]
    t = time.perf_counter()
    track = signals.compute(path)
    cands = signals.find_candidates(track)
    took = time.perf_counter() - t
    truth = json.loads((DATA / item["truth_path"]).read_text())
    return {"video_id": item["video_id"], "bucket": item["bucket"], "duration": truth["duration"],
            "defects": [{"type": d["type"], "start": d["start"], "end": d["end"]} for d in truth["defects"]],
            "cands": [{"t0": c.t0, "t1": c.t1, "kind": c.kind, "score": c.score} for c in cands],
            "secs": took}


def _overlaps(c: dict, d: dict) -> bool:
    return c["t0"] <= d["end"] + TOL and c["t1"] >= d["start"] - TOL


def score(rows: list[dict]) -> dict:
    hit, proxy_hit, total = Counter(), Counter(), Counter()
    tp, n_kind = Counter(), Counter()
    clean_min, clean_counts = 0.0, Counter()
    for r in rows:
        for d in r["defects"]:
            total[d["type"]] += 1
            kinds = MATCH.get(d["type"], ())
            hit[d["type"]] += any(c["kind"] in kinds and _overlaps(c, d) for c in r["cands"])
            pk = PROXY.get(d["type"], ())
            proxy_hit[d["type"]] += any(c["kind"] in kinds + pk and _overlaps(c, d) for c in r["cands"])
        for c in r["cands"]:
            n_kind[c["kind"]] += 1
            want = KIND_TO_TYPE.get(c["kind"])
            tp[c["kind"]] += any(d["type"] == want and _overlaps(c, d) for d in r["defects"])
        if not r["defects"]:
            clean_min += r["duration"] / 60
            clean_counts.update(c["kind"] for c in r["cands"])
    secs = [r["secs"] / (r["duration"] / 60) for r in rows]
    return {
        "n_videos": len(rows),
        "recall": {k: [hit[k], total[k]] for k in total},
        "proxy_recall": {k: [proxy_hit[k], total[k]] for k in PROXY if k in total},
        "precision": {k: [tp[k], n_kind[k]] for k in KIND_TO_TYPE},
        "clean_per_min": {k: clean_counts[k] / clean_min if clean_min else 0.0 for k in sorted(n_kind)},
        "clean_minutes": clean_min,
        "secs_per_video_min": {"mean": sum(secs) / len(secs), "max": max(secs)} if secs else {},
    }


def _frac(p: list[int]) -> str:
    return f"{p[0]}/{p[1]} ({100 * p[0] / p[1]:.0f}%)" if p[1] else "n/a"


def render(res: dict, title: str) -> str:
    lines = [f"### {title} ({res['n_videos']} videos, {res['clean_minutes']:.1f} clean min)", "",
             "| Defect type | Recall (matching kind) | Recall incl. proxy kind |", "|---|---|---|"]
    for k in sorted(res["recall"]):
        prox = res["proxy_recall"].get(k)
        lines.append(f"| {k} | {_frac(res['recall'][k])} | {_frac(prox) if prox else '-'} |")
    lines += ["", "| Kind | Precision | Clean candidates / min |", "|---|---|---|"]
    kinds = sorted(set(res["precision"]) | set(res["clean_per_min"]))
    for k in kinds:
        p = res["precision"].get(k)
        lines.append(f"| {k} | {_frac(p) if p else '-'} | {res['clean_per_min'].get(k, 0.0):.2f} |")
    s = res["secs_per_video_min"]
    if s:
        lines += ["", f"Runtime: {s['mean']:.2f} s per video-minute (max {s['max']:.2f})."]
    return "\n".join(lines)


def compare(before: dict, after: dict) -> str:
    lines = ["| Defect type | Recall before | Recall after |", "|---|---|---|"]
    for k in sorted(after["recall"]):
        lines.append(f"| {k} | {_frac(before['recall'].get(k, [0, 0]))} | {_frac(after['recall'][k])} |")
    lines += ["", "| Kind | Precision before | Precision after | Clean/min before | Clean/min after |",
              "|---|---|---|---|---|"]
    for k in sorted(set(after["clean_per_min"]) | set(before["clean_per_min"]) | set(after["precision"])):
        pb, pa = before["precision"].get(k), after["precision"].get(k)
        lines.append(f"| {k} | {_frac(pb) if pb else '-'} | {_frac(pa) if pa else '-'} | "
                     f"{before['clean_per_min'].get(k, 0.0):.2f} | {after['clean_per_min'].get(k, 0.0):.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", nargs="?", choices=("dev", "bench", "t1", "fdev"))
    ap.add_argument("--json", type=Path, help="write raw per-video rows + scores here")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args(argv)
    if a.compare:
        b, f = (json.loads(p.read_text())["score"] for p in a.compare)
        print(compare(b, f))
        return 0
    if not a.set:
        ap.error("give a set (dev | bench | t1 | fdev) or --compare")
    items = select(a.set)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, items))
    res = score(rows)
    print(render(res, a.set))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"set": a.set, "score": res, "rows": rows}, indent=1))
    misses = defaultdict(list)
    for r in rows:
        for d in r["defects"]:
            if not any(c["kind"] in MATCH.get(d["type"], ()) and _overlaps(c, d) for c in r["cands"]):
                misses[d["type"]].append(f"{r['video_id']}@{d['start']:.2f}")
    for k, v in sorted(misses.items()):
        print(f"missed {k}: {', '.join(v)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
