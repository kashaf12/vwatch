# Eval result: 2026-11-22

- Model + version: claude-opus-5-5
- Watcher commit: n/a
- Thresholds locked on: n/a
- Repeats per arm: 1

## Headline

`W vs B1 @25k tokens: T1 F1 n/a vs n/a (n/a, CI n/a), <1s recall n/a, cost 0.52×`

Verdict: INCOMPLETE

## Scorecard (matched budget)

| Criterion | Target | B1 (M1: B1H) | W | Δ | 95% CI | Pass? |
|---|---|---|---|---|---|---|
| S1 Defect F1 | >=0.85 abs and >=+15pp over B1 (CI>0) | n/a | n/a | n/a | n/a | n/a |
| S2 <1 s event recall | >=+25pp over B1 (CI>0) and W precision >= B1 - 10pp | n/a | n/a | n/a | n/a | n/a |
| S3 Localization +-0.5 s | >=90% | n/a | n/a | n/a | n/a | n/a |
| S4 False alarms (clean) | <=5% | n/a | n/a | n/a | n/a | n/a |
| S5 Causal link recall | >=+15pp over B1 (CI>0), invented <= B1 | n/a | n/a | n/a | n/a | n/a |
| S6 Cost vs B1 | <=1.5x $/video-minute | $0.8845/min | $0.4599/min | 0.52× | n/a | pass |
| S7 W@X vs B1@2X | W wins on T1 F1 (CI>0) | n/a | n/a | n/a | n/a | n/a |
| S8 Unseen-frame claims | 0 in T1 | n/a | n/a | n/a | n/a | n/a |
| M1 W vs B1H on T1 F1 (method-only) | informational, non-gating | n/a | n/a | n/a | n/a | info |

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

![recall by duration](bench-tvbench-2026-11-22_recall_by_duration.png)

## Pareto (T1 F1 / <1 s event recall vs token cap)

| Cap | B1 F1 | W F1 | B1 <1s recall | W <1s recall |
|---|---|---|---|---|
| 25k | n/a | n/a | n/a | n/a |

![pareto](bench-tvbench-2026-11-22_pareto.png)

## Ablations

| Arm | T1 F1 | <1 s recall | Causal recall | W minus this arm (T1 F1) |
|---|---|---|---|---|
| W | n/a | n/a | n/a | — |
| W-noQ | n/a | n/a | n/a | n/a |
| W-noP | n/a | n/a | n/a | n/a |
| W-noCrop | n/a | n/a | n/a | n/a |

## Method-only comparison

No B1H / W-noS cells at the matched budget (n/a).

## T1 F1 per source clip (matched budget)

n/a

## Precision, strictness and failures

| Arm@budget | T2 precision | T2 events listed/video | T1 strict recall (±0.5 s) | T1 loc | found | Failure rate |
|---|---|---|---|---|---|
| B1@25000 | n/a | n/a | n/a | n/a | 0.0% |
| CC@25000 | n/a | n/a | n/a | n/a | 0.0% |
| CVV@25000 | n/a | n/a | n/a | n/a | 0.0% |
| POP@25000 | n/a | n/a | n/a | n/a | 0.0% |
| W@25000 | n/a | n/a | n/a | n/a | 0.0% |
| W-noP@25000 | n/a | n/a | n/a | n/a | 0.0% |

## Public benchmark (TVBench subset)

Multiple-choice video QA from TVBench (CC-BY-4.0, FunAILab); informational, not part of the S/K criteria. Accuracy = correct items / items; a missing or invalid letter is wrong. Δ = W@matched budget minus the row's arm, 95% CI is a paired bootstrap over videos.

| Arm@budget | Videos | action_localization | action_sequence | moving_direction | object_count | scene_transition | unexpected_action | Overall | Δ vs W | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| B1@25000 | 36 | 50.0% | 100.0% | 83.3% | 100.0% | 83.3% | 66.7% | 80.6% | -11.1 pp | [-22.2, -2.8] |
| CC@25000 | 36 | 83.3% | 100.0% | 66.7% | 83.3% | 66.7% | 33.3% | 72.2% | -2.8 pp | [-19.4, +13.9] |
| CVV@25000 | 36 | 50.0% | 100.0% | 83.3% | 100.0% | 83.3% | 66.7% | 80.6% | -11.1 pp | [-22.2, -2.8] |
| POP@25000 | 36 | 66.7% | 100.0% | 66.7% | 83.3% | 66.7% | 66.7% | 75.0% | -5.6 pp | [-19.4, +8.3] |
| W@25000 | 36 | 50.0% | 100.0% | 83.3% | 50.0% | 83.3% | 50.0% | 69.4% | n/a | n/a |
| W-noP@25000 | 36 | 50.0% | 100.0% | 83.3% | 33.3% | 66.7% | 33.3% | 61.1% | +8.3 pp | [-2.8, +19.4] |
| chance (guessing) |  | 25.0% | 50.0% | 25.0% | 25.0% | 50.0% | 25.0% | 33.3% |  |  |

## Per-defect-type recall

| Type | B1 | W |
|---|---|---|
| Flash frame | n/a | n/a |
| Frozen frame | n/a | n/a |
| Jump cut | n/a | n/a |
| Audio dropout | n/a | n/a |
| Caption blocked | n/a | n/a |
| Wrong number | n/a | n/a |

## All arms: cost and honesty

| Arm@budget | Records | Videos | T1 F1 | T1 prec | T1 rec | T2 rec | T3 rec | T4 acc | In tok | Content tok | $/min | Wall s/min | Unseen T1 | T1 F1 std | Failed runs | Fail rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 23504 | 20916 | 0.8845 | 18.5 | n/a | n/a | 0 | 0.0% |
| CC@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 22975 | 0 | 0.2122 | 53.7 | n/a | n/a | 0 | 0.0% |
| CVV@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 46671 | 0 | 0.4571 | 38.2 | n/a | n/a | 0 | 0.0% |
| POP@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 29185 | 0 | 0.3145 | 56.5 | n/a | n/a | 0 | 0.0% |
| W@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 11903 | 6376 | 0.4599 | 66.9 | n/a | n/a | 0 | 0.0% |
| W-noP@25000 | 36 | 36 | n/a | n/a | n/a | n/a | n/a | n/a | 7838 | 4523 | 0.2575 | 43.7 | n/a | n/a | 0 | 0.0% |

## Notes / failures worth reading

- Criteria n/a (missing data): S1, S2, S3, S4, S5, S7, S8, K1, K2, K3.
- T3 not scored: no judge supplied.
- Bootstrap: 10000 resamples, 95% percentile CI, repeats averaged per video. T1 resamples whole source clips (cluster bootstrap); T2/T3/T4 resample videos.
- McNemar tests are per-item and not cluster-adjusted: descriptive only.
- Unseen claims count only frame/crop/audio observations within 0.5 s; hint- or transcript-only claims are unverified.
