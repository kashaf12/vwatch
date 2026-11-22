# Eval result: <run date>

- Model + version:
- Watcher commit:
- Thresholds locked on: (date / commit)
- Repeats per arm: 3

## Headline

`W vs B1 @__k tokens: T1 F1 __ vs __ (+__ pp, CI [__, __]), <1s recall +__ pp, cost __×`

Verdict: SHIP / KILL (K_) / ITERATE

## Scorecard (matched budget)

| Criterion | Target | B1 | W | Δ | 95% CI | Pass? |
|---|---|---|---|---|---|---|
| S1 Defect F1 | ≥0.85, +15pp | | | | | |
| S2 <1 s event recall | +25pp | | | | | |
| S3 Localization ±0.5 s | ≥90% | | | | | |
| S4 False alarms (clean) | ≤5% | | | | | |
| S5 Causal link recall | +15pp | | | | | |
| S6 Cost ×B1 | ≤1.5× | | | | | |
| S7 W@X vs B1@2X | W wins | | | | | |
| S8 Unseen-frame claims | 0 | | | | | |

## Recall by event duration

| Bucket | B0 | B1 | B2 | W |
|---|---|---|---|---|
| <0.5 s | | | | |
| 0.5–1 s | | | | |
| 1–3 s | | | | |
| >3 s | | | | |

## Pareto (T1 F1 vs token cap)

| Cap | B1 | W |
|---|---|---|
| 10k | | |
| 25k | | |
| 50k | | |
| 100k | | |

## Ablations

| Arm | T1 F1 | <1 s recall | Causal recall | Gain from the removed idea |
|---|---|---|---|---|
| W | | | | — |
| W-noQ | | | | |
| W-noP | | | | |
| W-noCrop | | | | |

## Per-defect-type recall

| Type | B1 | W |
|---|---|---|
| Caption over face | | |
| Number mismatch | | |
| Cut mid-word | | |
| Flash frame | | |
| Audio desync | | |
| Wrong subtitle text | | |
| Frozen frame | | |

## Notes / failures worth reading

-
