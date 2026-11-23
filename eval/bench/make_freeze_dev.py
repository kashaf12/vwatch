"""Fresh freeze dev set for detector tuning, disjoint from the benchmark videos.

Re-injects frozen frames into the 10 cached T1x source segments with a different seed and spread of
positions than the benchmark build, plus one clean copy per source, under eval/data/fdev/ (gitignored).
The 40 benchmark ids in eval/bench/t1x_ids.txt are never read or written here.

  uv run python eval/bench/make_freeze_dev.py [--seed 1001] [--per-source 6]
  uv run python eval/bench/make_freeze_dev.py --seed 2002 --out eval/data/fdev2 --prefix fdev2   # validation set
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from vwatch.dataset._source import SourceSpec, load_source
from vwatch.dataset.inject import T1Job, write_job
from vwatch.schemas import DefectType

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "eval" / "data" / "_cache" / "commons"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--per-source", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "data" / "fdev")
    ap.add_argument("--prefix", default="fdev", help="video id prefix (use a different one per set)")
    args = ap.parse_args()
    segments = sorted(CACHE.glob("*.segment.mp4"))
    if not segments:
        raise SystemExit(f"no cached T1x segments in {CACHE}; build t1x first")
    for si, seg in enumerate(segments):
        name = seg.name.removesuffix(".segment.mp4")
        src = load_source(SourceSpec(name=name, path=seg))
        n = args.per_source
        jobs = [T1Job(f"{args.prefix}_{name}_frozen_frame_{k}", si, DefectType.FROZEN_FRAME, k, (k + 0.5) / n) for k in range(n)]
        jobs.append(T1Job(f"{args.prefix}_{name}_clean_0", si, None, 0, 0.0))
        for job in jobs:
            write_job(args.out, src, job, args.seed + si)
            print(job.video_id, flush=True)
        del src
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
