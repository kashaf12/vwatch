"""Quick per-arm TVBench accuracy for one run date tag. Usage: python eval/bench/quick_t5.py 2026-11-22c"""
import collections, glob, json, sys
from pathlib import Path
from vwatch.schemas import MCQReport, MCQTruth, RunRecord
from vwatch.evaluation.scoring_mcq import score_mcq

tag = sys.argv[1]
man = {json.loads(l)["video_id"]: json.loads(l) for l in open("eval/data/manifest.jsonl")}
acc, cost = collections.defaultdict(list), collections.defaultdict(float)
cat = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"eval/runs/{tag}_*.jsonl"):
    for line in open(f):
        r = RunRecord.model_validate_json(line); m = man[r.video_id]
        t = MCQTruth.model_validate_json(Path("eval/data", m["truth_path"]).read_text())
        s = score_mcq(t, MCQReport.model_validate(r.parsed) if r.parsed else None)
        c = s.correct if hasattr(s, "correct") else s
        v = sum(c) / len(c) if isinstance(c, (list, tuple)) else float(c)
        acc[r.arm.value].append(v); cost[r.arm.value] += r.cost_usd; cat[r.arm.value][m["bucket"]].append(v)
for a in sorted(acc, key=lambda a: -sum(acc[a]) / len(acc[a])):
    n = len(acc[a])
    print(f"{a:6s} n={n} acc={sum(acc[a]) / n:.3f} $/vid={cost[a] / n:.3f}",
          {k: round(sum(v) / len(v), 2) for k, v in sorted(cat[a].items())})
