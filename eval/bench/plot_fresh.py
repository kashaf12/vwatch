"""Fresh-benchmark chart for the README, computed from the run files (eval/runs/2026-11-24fresh_*.jsonl)."""
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("VWP", "Claude Code + vwatch plugin"), ("CC", "Plain Claude Code"), ("CVV", "claude-video-vision")]
F1 = {"VWP": 0.62, "CC": 0.68, "CVV": 0.64}  # from eval/results/bench-fresh-2026-11-24.md (score_fresh.py)
COLORS = ["#1f5fae", "#8a96a6", "#8a96a6"]

last: dict = {}
for f in glob.glob("eval/runs/2026-11-24fresh_*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        last[(r["arm"], r["video_id"], r["task"])] = r
cost = {a: sum(r["cost_usd"] or 0 for (x, _, _), r in last.items() if x == a) / 56 for a, _ in ARMS}
secs = {a: sum(r["wall_s"] or 0 for (x, _, _), r in last.items() if x == a) / 56 for a, _ in ARMS}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True, gridspec_kw={"wspace": 0.18})
for ax, vals, title, fmt, lim in [
    (axes[0], F1, "Defect F1 (higher is better)\n48 fresh videos; differences are within noise", "{:.2f}", 1.0),
    (axes[1], cost, "Cost per video, $\n(lower is better)", "${:.2f}", 0.45),
    (axes[2], secs, "Time per video, seconds\n(lower is better)", "{:.0f}s", 140),
]:
    y = list(range(len(ARMS)))[::-1]
    v = [vals[a] for a, _ in ARMS]
    ax.barh(y, v, color=COLORS, height=0.6)
    ax.set_yticks(y, [n for _, n in ARMS], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, lim * 1.12)
    ax.set_title(title, fontsize=10, loc="left")
    for yi, x in zip(y, v):
        ax.text(x + lim * 0.015, yi, fmt.format(x), va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
fig.text(0.01, -0.04, "Pre-registered fresh benchmark, 2026-11-24: 40 short clips + 8 long videos (12-29 min), one run per tool, "
         "claude-opus-5-5. Cost and time differences are significant; F1 differences are not.", fontsize=8, color="#566170")
fig.savefig("docs/assets/bench-fresh-2026-11-24.png", dpi=150, bbox_inches="tight", facecolor="white")
print({a: (round(cost[a], 3), round(secs[a])) for a, _ in ARMS})
