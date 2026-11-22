import pytest

from vwatch.evaluation import criteria as C
from vwatch.evaluation.stats import (
    Estimate, f1, majority_found, mcnemar_exact, mcnemar_from_flags, mean_by_key,
    paired_bootstrap, paired_diff, safe_div,
)


def test_f1_zero_safe():
    assert f1(0, 0, 0) == 0.0
    assert f1(2, 0, 0) == 1.0
    assert f1(1, 1, 1) == pytest.approx(0.5)
    assert safe_div(1, 0) == 0.0


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(2, 8) == pytest.approx(0.109375)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_bootstrap_deterministic_and_ordered():
    units = [1.0, 2.0, 3.0, 4.0, 10.0]
    stat = lambda us: sum(us) / len(us)
    a = paired_bootstrap(units, stat, n=500, seed=3)
    b = paired_bootstrap(units, stat, n=500, seed=3)
    c = paired_bootstrap(units, stat, n=500, seed=4)
    assert a == b and a != c
    assert a.point == pytest.approx(4.0)
    assert a.lo <= a.point <= a.hi


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        paired_bootstrap([], sum)
    with pytest.raises(ValueError):
        paired_bootstrap([1], sum, n=0)


def test_mean_by_key_and_paired_diff():
    a = mean_by_key([("v1", (1, 0)), ("v1", (3, 0)), ("v2", (2, 0))])
    assert a == {"v1": (2.0, 0.0), "v2": (2.0, 0.0)}
    b = {"v1": (0.0, 0.0), "v2": (0.0, 0.0), "v3": (9.0, 9.0)}
    est = paired_diff(a, b, lambda v: v[0], n=200)
    assert est.point == pytest.approx(4.0)
    assert est.lo > 0
    assert paired_diff(a, {"zz": (1, 1)}, lambda v: v[0]) is None


def test_flags_and_mcnemar_from_flags():
    maj = majority_found({"x": [True, True, False], "y": [True, False], "z": [False, False, True]})
    assert maj == {"x": True, "y": True, "z": False}
    m = mcnemar_from_flags({"a": True, "b": True, "c": False}, {"a": False, "b": True, "c": False, "d": True})
    assert m["a_only"] == 1 and m["b_only"] == 0 and m["n"] == 3 and m["p"] == 1.0


# ---------------------------------------------------------------- criteria


def est(p, lo=0.01, hi=None):
    return Estimate(p, lo, hi if hi is not None else p + 0.05)


def full_inputs(**over):
    base = dict(
        f1_w=0.9, t1_diff=est(0.2), t2_short_diff=est(0.3), loc_rate_w=0.95, false_alarm_w=0.0,
        causal_diff=est(0.2), t2_precision_w=0.8, t2_precision_b1=0.8, m1_diff=est(0.1), invented_w=0.5, invented_b1=1.0, cost_ratio=1.2, s7_diff=est(0.1),
        unseen_t1_w=0, ablation_gaps={"W-noQ": 0.1, "W-noP": 0.08, "W-noCrop": 0.09},
    )
    base.update(over)
    return C.CriteriaInputs(**base)


def status(results, cid):
    return next(r.status for r in results if r.id == cid)


def test_all_pass_ships():
    res = C.evaluate(full_inputs())
    assert all(r.status in (C.PASS, C.CLEAR, C.INFO) for r in res)
    assert C.verdict(res).label == "SHIP"


def test_missing_data_is_na_not_pass():
    res = C.evaluate(C.CriteriaInputs())
    assert all(r.status == C.NA for r in res if r.id != "M1")  # M1 is informational
    assert status(res, "M1") == C.INFO
    assert C.verdict(res).label == "INCOMPLETE"


def test_ci_must_exclude_zero():
    res = C.evaluate(full_inputs(t1_diff=est(0.2, lo=-0.01)))
    assert status(res, "S1") == C.FAIL
    res = C.evaluate(full_inputs(t2_short_diff=est(0.2)))  # point below 25pp
    assert status(res, "S2") == C.FAIL
    assert C.verdict(res).label == "ITERATE"


def test_s5_invented_and_partial_na():
    assert status(C.evaluate(full_inputs(invented_w=2.0)), "S5") == C.FAIL
    assert status(C.evaluate(full_inputs(invented_w=None)), "S5") == C.NA
    assert status(C.evaluate(full_inputs(f1_w=0.8)), "S1") == C.FAIL


def test_simple_thresholds():
    res = C.evaluate(full_inputs(loc_rate_w=0.85, false_alarm_w=0.06, cost_ratio=1.6, unseen_t1_w=1,
                                 s7_diff=est(0.05, lo=-0.01)))
    for cid in ("S3", "S4", "S6", "S7", "S8"):
        assert status(res, cid) == C.FAIL


def test_kill_rules():
    res = C.evaluate(full_inputs(t1_diff=est(0.03), t2_short_diff=est(0.04)))
    assert status(res, "K1") == C.TRIGGERED
    v = C.verdict(res)
    assert v.label == "KILL" and "K1" in v.killed_by and str(v).startswith("KILL (")
    res = C.evaluate(full_inputs(s7_diff=est(0.01)))
    assert status(res, "K2") == C.TRIGGERED
    res = C.evaluate(full_inputs(ablation_gaps={"W-noQ": 0.01, "W-noP": -0.02, "W-noCrop": 0.03}))
    assert status(res, "K3") == C.TRIGGERED
    res = C.evaluate(full_inputs(ablation_gaps={"W-noQ": 0.01, "W-noP": None, "W-noCrop": 0.0}))
    assert status(res, "K3") == C.NA


def test_verdict_priority():
    # core n/a beats ITERATE; other n/a with all else passing is INCOMPLETE
    res = C.evaluate(full_inputs(t1_diff=None, loc_rate_w=0.1))
    assert C.verdict(res).label == "INCOMPLETE"
    res = C.evaluate(full_inputs(causal_diff=None))
    assert C.verdict(res).label == "INCOMPLETE"


def test_s2_precision_guard_and_m1_info():
    assert status(C.evaluate(full_inputs(t2_precision_w=0.70)), "S2") == C.PASS  # exactly -10 pp
    res = C.evaluate(full_inputs(t2_precision_w=0.69))
    assert status(res, "S2") == C.FAIL and C.verdict(res).label == "ITERATE"
    assert status(C.evaluate(full_inputs(t2_precision_w=None)), "S2") == C.NA
    # a definite recall failure is not hidden by missing precision
    assert status(C.evaluate(full_inputs(t2_precision_w=None, t2_short_diff=est(0.1))), "S2") == C.FAIL
    m1 = next(r for r in C.evaluate(full_inputs()) if r.id == "M1")
    assert m1.status == C.INFO and "+10.0 pp" in m1.detail
    assert "n/a (needs" in next(r for r in C.evaluate(full_inputs(m1_diff=None)) if r.id == "M1").detail
    inf = Estimate(0.1, float("-inf"), float("inf"))
    assert "CI n/a" in next(r for r in C.evaluate(full_inputs(m1_diff=inf)) if r.id == "M1").detail
    # M1 never changes the verdict
    assert C.verdict(C.evaluate(full_inputs(m1_diff=est(-0.5, lo=-0.6)))).label == "SHIP"


def test_cluster_bootstrap_and_clustered_paired_diff():
    from vwatch.evaluation.stats import cluster_bootstrap
    units = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    clusters = ["a", "a", "b", "b", "c", "c"]
    e1 = cluster_bootstrap(units, clusters, lambda u: sum(u) / len(u), n=300, seed=1)
    assert e1 == cluster_bootstrap(units, clusters, lambda u: sum(u) / len(u), n=300, seed=1)
    assert e1.point == pytest.approx(3.5) and e1.lo < e1.point < e1.hi
    solo = cluster_bootstrap(units, ["x"] * 6, lambda u: sum(u), n=10)
    assert solo.point == 21 and solo.lo == float("-inf") and solo.hi == float("inf")
    with pytest.raises(ValueError):
        cluster_bootstrap([], [], sum)
    with pytest.raises(ValueError):
        cluster_bootstrap([1], ["a", "b"], sum)
    with pytest.raises(ValueError):
        cluster_bootstrap([1], ["a"], sum, n=0)
    # clustering widens the CI: 8 videos, strong within-cluster correlation
    a = {f"v{i}": (1.0 if i < 4 else 0.0,) for i in range(8)}
    b = {f"v{i}": (0.0,) for i in range(8)}
    per_video = paired_diff(a, b, lambda v: v[0], n=2000)
    clustered = paired_diff(a, b, lambda v: v[0], n=2000, cluster_of={f"v{i}": "s0" if i < 4 else "s1" for i in range(8)})
    assert per_video.point == clustered.point == 4.0
    assert (clustered.hi - clustered.lo) > (per_video.hi - per_video.lo) or clustered.lo == 0.0
    one = paired_diff(a, b, lambda v: v[0], cluster_of={f"v{i}": "s" for i in range(8)})
    assert one.lo == float("-inf")
    with pytest.raises(ValueError):
        paired_diff(a, b, lambda v: v[0], n=0)
