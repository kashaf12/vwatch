"""PNG plots for the report (matplotlib, Agg backend)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vwatch.evaluation.scoring import EVENT_BUCKETS  # noqa: E402

_MARKERS = {"B0": "s", "B1": "o", "B1H": "v", "B2": "^", "W": "D", "W-noS": "P"}


def plot_recall_by_duration(scorecard: Mapping, path: str | Path) -> None:
    table = scorecard["duration_table"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ("B0", "B1", "B1H", "B2", "W", "W-noS"):
        if arm not in table:
            continue
        ys = [table[arm][b] for b in EVENT_BUCKETS]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], marker=_MARKERS[arm], label=f"{arm} (@{table[arm]['budget'] // 1000}k)")
    ax.set_xticks(range(len(EVENT_BUCKETS)), EVENT_BUCKETS)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Event duration")
    ax.set_ylabel("Event recall (T2)")
    ax.set_title("Recall by event duration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pareto(scorecard: Mapping, path: str | Path) -> None:
    rows = scorecard["pareto"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, suffix, ylabel in ((axes[0], "f1", "T1 defect F1"), (axes[1], "short_recall", "T2 <1 s event recall")):
        for arm in ("B1", "W"):
            pts = [(r["budget"], r[f"{arm}_{suffix}"]) for r in rows if r[f"{arm}_{suffix}"] is not None]
            if pts:
                ax.plot(*zip(*pts), marker=_MARKERS[arm], label=arm)
        ax.set_xlabel("Input-token cap per video")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Accuracy vs token budget (up and to the left is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
