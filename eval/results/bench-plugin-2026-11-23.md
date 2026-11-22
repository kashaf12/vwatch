# Eval result: 2026-11-23

- Model + version: claude-opus-5-5
- Watcher commit: n/a
- Thresholds locked on: n/a
- Repeats per arm: 1

## Headline

`W vs B1 @25k tokens: T1 F1 0.77 vs n/a (n/a, CI n/a), <1s recall n/a, cost n/a`

Verdict: INCOMPLETE

## Scorecard (matched budget)

| Criterion | Target | B1 (M1: B1H) | W | Δ | 95% CI | Pass? |
|---|---|---|---|---|---|---|
| S1 Defect F1 | >=0.85 abs and >=+15pp over B1 (CI>0) | n/a | 0.77 | n/a | n/a | n/a |
| S2 <1 s event recall | >=+25pp over B1 (CI>0) and W precision >= B1 - 10pp | n/a | n/a | n/a | n/a | n/a |
| S3 Localization +-0.5 s | >=90% | n/a | 76.7% | n/a | n/a | fail |
| S4 False alarms (clean) | <=5% | n/a | 10.0% | n/a | n/a | fail |
| S5 Causal link recall | >=+15pp over B1 (CI>0), invented <= B1 | n/a | n/a | n/a | n/a | n/a |
| S6 Cost vs B1 | <=1.5x $/video-minute | n/a | $0.2753/min | n/a | n/a | n/a |
| S7 W@X vs B1@2X | W wins on T1 F1 (CI>0) | n/a | 0.77 | n/a | n/a | n/a |
| S8 Unseen-frame claims | 0 in T1 | n/a | 0 | n/a | n/a | pass |
| M1 W vs B1H on T1 F1 (method-only) | informational, non-gating | n/a | 0.77 | n/a | n/a | info |

S3 = localized (±0.5 s) defects ÷ ALL truth defects; S2 also needs W T2 precision >= B1 - 10 pp.

| Kill rule | Condition | Status |
|---|---|---|
| K1 Gain < 5pp on T1 and T2 | kill if both gains < 5pp | n/a |
| K2 B2 matches W on T1 | kill if W - B2@2X F1 < 2pp | n/a |
| K3 Ablations within 3pp | kill if all three within 3pp of W | n/a |

## Recall by event duration

| Bucket |
|---|
| <0.5s |
| 0.5-1s |
| 1-3s |
| >3s |

![recall by duration](bench-plugin-2026-11-23_recall_by_duration.png)

## Pareto (T1 F1 / <1 s event recall vs token cap)

| Cap | B1 F1 | W F1 | B1 <1s recall | W <1s recall |
|---|---|---|---|---|
| 25k | n/a | 0.77 | n/a | n/a |

![pareto](bench-plugin-2026-11-23_pareto.png)

## Ablations

| Arm | T1 F1 | <1 s recall | Causal recall | W minus this arm (T1 F1) |
|---|---|---|---|---|
| W | 0.77 | n/a | n/a | — |
| W-noQ | n/a | n/a | n/a | n/a |
| W-noP | n/a | n/a | n/a | n/a |
| W-noCrop | n/a | n/a | n/a | n/a |

## Method-only comparison

No B1H / W-noS cells at the matched budget (n/a).

## T1 F1 per source clip (matched budget)

| Source | VWP | W |
|---|---|---|
| bnreel | 0.80 | 0.80 |
| boston | 0.80 | 0.80 |
| gujarat | 0.80 | 0.80 |
| jharkhand | 0.50 | 0.50 |
| khizr | 0.80 | 0.80 |
| linagora | 0.80 | 0.80 |
| pietra | 1.00 | 1.00 |
| prague | 1.00 | 0.67 |
| sem | 0.67 | 0.60 |
| steel | 1.00 | 1.00 |

## Precision, strictness and failures

| Arm@budget | T2 precision | T2 events listed/video | T1 strict recall (±0.5 s) | T1 loc | found | Failure rate |
|---|---|---|---|---|---|
| VWP@25000 | n/a | n/a | 76.7% | 100.0% | 0.0% |
| W@25000 | n/a | n/a | 76.7% | 100.0% | 0.0% |

## Per-defect-type recall

| Type | B1 | W |
|---|---|---|
| Flash frame | n/a | 100.0% |
| Frozen frame | n/a | 57.1% |
| Jump cut | n/a | 50.0% |
| Audio dropout | n/a | 100.0% |
| Caption blocked | n/a | n/a |
| Wrong number | n/a | n/a |

## All arms: cost and honesty

| Arm@budget | Records | Videos | T1 F1 | T1 prec | T1 rec | T2 rec | T3 rec | T4 acc | In tok | Content tok | $/min | Wall s/min | Unseen T1 | T1 F1 std | Failed runs | Fail rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VWP@25000 | 40 | 40 | 0.82 | 0.88 | 0.77 | n/a | n/a | n/a | 86736 | 0 | 0.1775 | 36.3 | 26 | n/a | 0 | 0.0% |
| W@25000 | 40 | 40 | 0.77 | 0.77 | 0.77 | n/a | n/a | n/a | 35767 | 14310 | 0.2753 | 36.3 | 0 | n/a | 0 | 0.0% |

## Notes / failures worth reading

- Criteria n/a (missing data): S1, S2, S5, S6, S7, K1, K2, K3.
- T3 not scored: no judge supplied.
- Bootstrap: 10000 resamples, 95% percentile CI, repeats averaged per video. T1 resamples whole source clips (cluster bootstrap); T2/T3/T4 resample videos.
- McNemar tests are per-item and not cluster-adjusted: descriptive only.
- Unseen claims count only frame/crop/audio observations within 0.5 s; hint- or transcript-only claims are unverified.
