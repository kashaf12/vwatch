"""Pass/kill criteria from docs/eval-protocol.md section 5.

All values are fractions (0.15 = 15 pp). Missing data evaluates to "n/a", never a pass.
A difference "counts" only when its 95% bootstrap CI excludes 0 (lower bound > 0), and
the point estimate must also meet the threshold (S1, S2, S5, S7).

Definitions chosen where the spec is open:
- K2 triggers when the point estimate of (W F1 - B2@2X F1) on T1 is below K2_MIN_GAP (2 pp).
  The same difference drives S7 (which additionally needs CI lower bound > 0).
- K3 triggers when |W F1 - ablation F1| <= 3 pp for all three ablations (all must be present).
- S2 additionally requires W's T2 precision >= B1's T2 precision - 10 pp (guards against
  recall bought by listing everything). Missing precision data makes S2 n/a, never a pass.
- M1 (W vs B1H on T1 F1) is informational (status "info"); it never affects the verdict.
- Verdict: KILL if any K triggered; else INCOMPLETE if a core criterion (S1, S2, K1) is n/a;
  else ITERATE if any S fails; else INCOMPLETE if anything is still n/a; else SHIP.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from vwatch.evaluation.stats import Estimate

S1_MIN_F1 = 0.85
S1_MIN_GAIN = 0.15
S2_MIN_GAIN = 0.25
S2_MAX_PRECISION_DROP = 0.10  # W T2 precision must be >= B1 precision - 10 pp
S3_MIN_LOC = 0.90
S4_MAX_FALSE_ALARM = 0.05
S5_MIN_GAIN = 0.15
S6_MAX_COST_RATIO = 1.5
S8_MAX_UNSEEN = 0
K1_MAX_GAIN = 0.05
K2_MIN_GAP = 0.02
K3_BAND = 0.03
ABLATION_ARMS = ("W-noQ", "W-noP", "W-noCrop")
EXTRA_ARMS = ("B1H", "W-noS")  # method-only baseline / signal-contribution ablation (informational)

PASS, FAIL, NA, TRIGGERED, CLEAR, INFO = "pass", "fail", "n/a", "triggered", "clear", "info"
_CORE = ("S1", "S2", "K1")


@dataclass(frozen=True)
class CriteriaInputs:
    f1_w: float | None = None
    t1_diff: Estimate | None = None  # W - B1, T1 F1
    t2_short_diff: Estimate | None = None  # W - B1, recall on events < 1 s
    t2_precision_w: float | None = None  # S2 guard: W T2 precision
    t2_precision_b1: float | None = None
    m1_diff: Estimate | None = None  # W - B1H, T1 F1 (informational, never gates)
    loc_rate_w: float | None = None
    false_alarm_w: float | None = None
    causal_diff: Estimate | None = None  # W - B1, causal recall
    invented_w: float | None = None  # mean invented links per video
    invented_b1: float | None = None
    cost_ratio: float | None = None  # W $/min over B1 $/min
    s7_diff: Estimate | None = None  # W@X - B@2X, T1 F1
    unseen_t1_w: int | None = None
    ablation_gaps: Mapping[str, float | None] = field(default_factory=dict)  # W F1 - ablation F1


@dataclass(frozen=True)
class CriterionResult:
    id: str
    name: str
    target: str
    status: str
    detail: str = ""


def _counts(est: Estimate | None, threshold: float) -> bool | None:
    if est is None:
        return None
    return est.point >= threshold and est.lo > 0


def _res(cid: str, name: str, target: str, ok: bool | None, detail: str = "") -> CriterionResult:
    status = NA if ok is None else (PASS if ok else FAIL)
    return CriterionResult(cid, name, target, status, detail)


def _kres(cid: str, name: str, target: str, hit: bool | None, detail: str = "") -> CriterionResult:
    status = NA if hit is None else (TRIGGERED if hit else CLEAR)
    return CriterionResult(cid, name, target, status, detail)


def _all(*flags: bool | None) -> bool | None:
    if any(f is None for f in flags):
        return None
    return all(flags)


def _all_strict(*flags: bool | None) -> bool | None:
    """AND where a definite False beats an unknown (None)."""
    if any(f is False for f in flags):
        return False
    return None if any(f is None for f in flags) else True


def _m1(d: Estimate | None) -> CriterionResult:
    if d is None:
        detail = "n/a (needs W and B1H at the matched budget)"
    elif math.isfinite(d.lo):
        detail = f"W - B1H = {d.point * 100:+.1f} pp, CI [{d.lo * 100:+.1f}, {d.hi * 100:+.1f}]"
    else:
        detail = f"W - B1H = {d.point * 100:+.1f} pp, CI n/a (<2 clusters)"
    return CriterionResult("M1", "W vs B1H on T1 F1 (method-only)", "informational, non-gating", INFO, detail)


def evaluate(x: CriteriaInputs) -> list[CriterionResult]:
    s1 = _all(
        None if x.f1_w is None else x.f1_w >= S1_MIN_F1,
        _counts(x.t1_diff, S1_MIN_GAIN),
    )
    s5 = _all(
        _counts(x.causal_diff, S5_MIN_GAIN),
        None if x.invented_w is None or x.invented_b1 is None else x.invented_w <= x.invented_b1,
    )
    s7 = None if x.s7_diff is None else (x.s7_diff.point > 0 and x.s7_diff.lo > 0)
    prec_ok = (
        None
        if x.t2_precision_w is None or x.t2_precision_b1 is None
        else x.t2_precision_w >= x.t2_precision_b1 - S2_MAX_PRECISION_DROP - 1e-9
    )
    s2 = _all_strict(_counts(x.t2_short_diff, S2_MIN_GAIN), prec_ok)
    k1 = (
        None
        if x.t1_diff is None or x.t2_short_diff is None
        else x.t1_diff.point < K1_MAX_GAIN and x.t2_short_diff.point < K1_MAX_GAIN
    )
    k2 = None if x.s7_diff is None else x.s7_diff.point < K2_MIN_GAP
    gaps = [x.ablation_gaps.get(a) for a in ABLATION_ARMS]
    k3 = None if any(g is None for g in gaps) else all(abs(g) <= K3_BAND for g in gaps)
    return [
        _res("S1", "Defect F1", ">=0.85 abs and >=+15pp over B1 (CI>0)", s1),
        _res("S2", "<1 s event recall", ">=+25pp over B1 (CI>0) and W precision >= B1 - 10pp", s2),
        _res("S3", "Localization +-0.5 s", ">=90%", None if x.loc_rate_w is None else x.loc_rate_w >= S3_MIN_LOC),
        _res("S4", "False alarms (clean)", "<=5%", None if x.false_alarm_w is None else x.false_alarm_w <= S4_MAX_FALSE_ALARM),
        _res("S5", "Causal link recall", ">=+15pp over B1 (CI>0), invented <= B1", s5),
        _res("S6", "Cost vs B1", "<=1.5x $/video-minute", None if x.cost_ratio is None else x.cost_ratio <= S6_MAX_COST_RATIO),
        _res("S7", "W@X vs B1@2X", "W wins on T1 F1 (CI>0)", s7),
        _res("S8", "Unseen-frame claims", "0 in T1", None if x.unseen_t1_w is None else x.unseen_t1_w <= S8_MAX_UNSEEN),
        _kres("K1", "Gain < 5pp on T1 and T2", "kill if both gains < 5pp", k1),
        _kres("K2", "B2 matches W on T1", "kill if W - B2@2X F1 < 2pp", k2),
        _kres("K3", "Ablations within 3pp", "kill if all three within 3pp of W", k3),
        _m1(x.m1_diff),
    ]


@dataclass(frozen=True)
class Verdict:
    label: str  # SHIP | KILL | ITERATE | INCOMPLETE
    killed_by: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"KILL ({', '.join(self.killed_by)})" if self.label == "KILL" else self.label


def verdict(results: list[CriterionResult]) -> Verdict:
    killed = tuple(r.id for r in results if r.status == TRIGGERED)
    if killed:
        return Verdict("KILL", killed)
    if any(r.status == NA for r in results if r.id in _CORE):
        return Verdict("INCOMPLETE")
    if any(r.status == FAIL for r in results):
        return Verdict("ITERATE")
    if any(r.status == NA for r in results):
        return Verdict("INCOMPLETE")
    return Verdict("SHIP")
