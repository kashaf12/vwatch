"""Statistics helpers: exact McNemar, paired bootstrap, F1, and video-level pairing.

Pairing choice: comparisons are paired by ``video_id``. A video's repeats are
averaged within each arm first (per-unit counts become means over repeats), so
every video is one unit, regardless of how many repeats each arm has.

Clustering choice: videos are NOT always independent (all T1 videos derive from a
few source clips). ``paired_diff`` therefore does a one-stage cluster bootstrap:
resample *clusters* with replacement, and keep every unit inside each drawn
cluster. Default cluster = the video itself (the old per-video bootstrap). With
fewer than 2 clusters the CI is uninformative, so it is reported as (-inf, +inf)
and can never "exclude zero". Only videos present in both arms are used. For McNemar
(per-truth-item found/missed) an item counts as found when it was found in at
least half of that arm's repeats (majority vote).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple, TypeVar

import numpy as np

K = TypeVar("K")


class Estimate(NamedTuple):
    point: float
    lo: float
    hi: float


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(tp: float, fp: float, fn: float) -> float:
    """F1 from counts; 0.0 when there is nothing to score."""
    denom = 2 * tp + fp + fn
    return safe_div(2 * tp, denom)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    units: Sequence[K],
    stat: Callable[[Sequence[K]], float],
    *,
    n: int = 10000,
    seed: int = 0,
) -> Estimate:
    """Resample units with replacement; return (point, 2.5th, 97.5th percentile)."""
    if not units:
        raise ValueError("paired_bootstrap needs at least one unit")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n, len(units)))
    draws = np.fromiter(
        (stat([units[i] for i in row]) for row in idx), dtype=float, count=n
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Estimate(float(stat(units)), float(lo), float(hi))


def cluster_bootstrap(
    units: Sequence[K],
    clusters: Sequence[str],
    stat: Callable[[Sequence[K]], float],
    *,
    n: int = 10000,
    seed: int = 0,
) -> Estimate:
    """One-stage cluster bootstrap: draw clusters with replacement, keep all their units."""
    if not units:
        raise ValueError("cluster_bootstrap needs at least one unit")
    if len(clusters) != len(units):
        raise ValueError("clusters must have one label per unit")
    if n < 1:
        raise ValueError("n must be >= 1")
    groups: dict[str, list[K]] = {}
    for u, c in zip(units, clusters, strict=True):
        groups.setdefault(c, []).append(u)
    point = float(stat(units))
    if len(groups) < 2:
        return Estimate(point, -math.inf, math.inf)
    members = list(groups.values())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(members), size=(n, len(members)))
    draws = np.fromiter(
        (stat([u for c in row for u in members[c]]) for row in idx), dtype=float, count=n
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi))


def mean_by_key(items: Iterable[tuple[str, Sequence[float]]]) -> dict[str, tuple[float, ...]]:
    """Average the vectors that share a key (e.g. a video's repeats)."""
    acc: dict[str, list[Sequence[float]]] = {}
    for key, vec in items:
        acc.setdefault(key, []).append(vec)
    return {k: tuple(float(x) for x in np.mean(np.asarray(v, dtype=float), axis=0)) for k, v in acc.items()}


def paired_diff(
    a: Mapping[str, Sequence[float]],
    b: Mapping[str, Sequence[float]],
    metric: Callable[[Sequence[float]], float],
    *,
    n: int = 10000,
    seed: int = 0,
    cluster_of: Mapping[str, str] | None = None,
) -> Estimate | None:
    """Cluster-bootstrap CI for metric(sum of a's columns) - metric(sum of b's columns).

    Videos present in both arms are the units; ``cluster_of`` maps video -> cluster
    (missing videos are their own cluster). Returns None when the arms share no video.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    common = sorted(set(a) & set(b))
    if not common:
        return None
    wa = len(a[common[0]])
    units = np.asarray([tuple(a[v]) + tuple(b[v]) for v in common], dtype=float)
    labels = [(cluster_of or {}).get(v, v) for v in common]

    def stat(sums: np.ndarray) -> float:
        return metric(sums[:wa]) - metric(sums[wa:])

    point = float(stat(units.sum(axis=0)))
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return Estimate(point, -math.inf, math.inf)
    csum = np.asarray([units[[l == u for l in labels]].sum(axis=0) for u in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    boot = csum[idx].sum(axis=1)  # (n, width): the statistic only needs summed counts
    draws = np.fromiter((stat(row) for row in boot), dtype=float, count=n)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi))


def majority_found(per_repeat: Mapping[K, Sequence[bool]]) -> dict[K, bool]:
    """Collapse repeats to one found/missed flag per item (found in >= half)."""
    return {k: sum(v) * 2 >= len(v) for k, v in per_repeat.items() if v}


def mcnemar_from_flags(a: Mapping[K, bool], b: Mapping[K, bool]) -> dict[str, float | int]:
    """McNemar between arm a and arm b on shared items. b_only = a found & b missed."""
    common = set(a) & set(b)
    only_a = sum(1 for k in common if a[k] and not b[k])
    only_b = sum(1 for k in common if b[k] and not a[k])
    return {"a_only": only_a, "b_only": only_b, "n": len(common), "p": mcnemar_exact(only_a, only_b)}
