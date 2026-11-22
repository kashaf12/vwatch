"""Head-to-head chart for the README, from the 2026-11-22 benchmark reports (numbers copied from
eval/results/bench-defects-t1x-2026-11-22.md and bench-tvbench*-2026-11-22.md)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["Claude Code", "claude-video-vision", "popcorn", "vwatch (no predict)", "vwatch", "frames + transcript"]
F1 = [0.86, 0.85, 0.85, 0.72, 0.69, 0.37]
COST = [0.30, 0.37, 0.46, 0.18, 0.28, 0.28]       # $ per video-minute, defect set
TVB = [0.72, 0.81, 0.75, 0.61, 0.69, 0.81]        # TVBench subset, 36 questions (vwatch v0.1)
COLORS = ["#8a96a6", "#8a96a6", "#8a96a6", "#1f5fae", "#1f5fae", "#c9d1db"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"wspace": 0.55})
for ax, vals, title, fmt, lim in [
    (axes[0], F1, "Defect detection F1\n40 real videos, 10 sources", "{:.2f}", 1.0),
    (axes[1], COST, "Cost, $ per video-minute\n(defect set)", "${:.2f}", 0.55),
    (axes[2], TVB, "TVBench subset accuracy\n36 questions, 6 categories", "{:.0%}", 1.0),
]:
    y = range(len(ARMS))[::-1]
    ax.barh(list(y), vals, color=COLORS, height=0.62)
    ax.set_yticks(list(y), ARMS, fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_title(title, fontsize=10, loc="left")
    for yi, v in zip(y, vals):
        ax.text(v + lim * 0.015, yi, fmt.format(v), va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
fig.text(0.01, 0.01, "claude-opus-5-5, effort medium, 25k evidence-token cap for vwatch and frames; "
         "Claude Code tools uncapped. Grey = Claude Code tools, blue = vwatch.", fontsize=8, color="#566170")
fig.savefig("docs/assets/bench-2026-11-22.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote docs/assets/bench-2026-11-22.png")
