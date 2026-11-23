"""Plugin head-to-head chart for the README (numbers copied from eval/results/bench-plugin-v023-2026-11-23.md,
bench-plugin-2026-11-23.md and bench-plugin-tvbench-2026-11-23.md)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["Claude Code + vwatch plugin", "Claude Code", "claude-video-vision", "popcorn", "vwatch loop (v0.2)",
        "frames + transcript"]
F1 = [0.85, 0.83, 0.86, 0.85, 0.79, 0.37]  # 3 runs for plugin, CC, CVV; 1 for the rest
COST = [0.17, 0.30, 0.39, 0.46, 0.28, 0.28]       # $ per video-minute, defect set
WALL = [33, 66, 61, 102, 36, 8]                   # wall seconds per video-minute, defect set
TVB = [0.78, 0.72, 0.81, 0.75, 0.69, 0.81]        # TVBench subset, 36 questions
COLORS = ["#1f5fae", "#8a96a6", "#8a96a6", "#8a96a6", "#6f9bd1", "#c9d1db"]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True, gridspec_kw={"wspace": 0.18})
for ax, vals, title, fmt, lim in [
    (axes[0], F1, "Defect detection F1\n40 real videos, 10 sources", "{:.2f}", 1.0),
    (axes[1], COST, "Cost, $ per video-minute\n(defect set, lower is better)", "${:.2f}", 0.55),
    (axes[2], WALL, "Wall time, s per video-minute\n(defect set, lower is better)", "{:.0f}s", 120),
    (axes[3], TVB, "TVBench subset accuracy\n36 questions, 6 categories", "{:.0%}", 1.0),
]:
    y = range(len(ARMS))[::-1]
    ax.barh(list(y), vals, color=COLORS, height=0.62)
    ax.set_yticks(list(y), ARMS, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, lim * 1.12)
    ax.set_title(title, fontsize=10, loc="left")
    for yi, v in zip(y, vals):
        ax.text(v + lim * 0.015, yi, fmt.format(v), va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
fig.text(0.01, 0.01, "claude-opus-5-5, effort medium. Claude Code arms uncapped; vwatch loop and frames under a 25k "
         "evidence-token cap. Dark blue = Claude Code + vwatch plugin, grey = other Claude Code tools.",
         fontsize=8, color="#566170")
fig.savefig("docs/assets/bench-2026-11-23.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote docs/assets/bench-2026-11-23.png")
