"""Formatting helpers shared by the markdown renderers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

NA = "n/a"


def f2(x: float | None) -> str:
    return NA if x is None else f"{x:.2f}"


def pct(x: float | None) -> str:
    return NA if x is None else f"{x * 100:.1f}%"


def pp(x: float | None) -> str:
    return NA if x is None else f"{x * 100:+.1f} pp"


def ci(d: Mapping | None) -> str:
    if d is None:
        return NA
    if not (math.isfinite(d["lo"]) and math.isfinite(d["hi"])):
        return "n/a (<2 clusters)"
    return f"[{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"


def pt(d: Mapping | None) -> float | None:
    return None if d is None else d["point"]


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


