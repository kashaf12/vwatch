# Plugin benchmark, 3 runs per arm: defects (T1x), 2026-11-23

40 held-out T1x videos (`eval/bench/t1x_ids.txt`). VWP = Claude Code + vwatch plugin at v0.2.2 (cell-level freeze detection, digital-silence dropout hint, skill guidance on strong hints), 3 runs (`2026-11-23k_VWP_r1-3`). CC and CVV: 3 runs each (`2026-11-22d_*_r1-3`; r2/r3 added 2026-11-23, they do not use vwatch code). POP and B1: 1 run. Repeats are averaged per video before bootstrapping.

Note: most of VWP's wrong reports are jump cuts at one moment in the `sem` source lecture that looks like an edit in the original footage. It is scored as labeled (clean). No re-score excluding it is reported, because such a rule would be chosen after seeing the results.

- Model + version: claude-opus-5-5
- Watcher commit: n/a
- Thresholds locked on: n/a
- Repeats per arm: 1, 3

## Headline

`W vs B1 @25k tokens: T1 F1 n/a vs 0.37 (n/a, CI n/a), <1s recall n/a, cost n/a`

Verdict: INCOMPLETE

## Scorecard (matched budget)

| Criterion | Target | B1 (M1: B1H) | W | Δ | 95% CI | Pass? |
|---|---|---|---|---|---|---|
| S1 Defect F1 | >=0.85 abs and >=+15pp over B1 (CI>0) | 0.37 | n/a | n/a | n/a | n/a |
| S2 <1 s event recall | >=+25pp over B1 (CI>0) and W precision >= B1 - 10pp | n/a | n/a | n/a | n/a | n/a |
| S3 Localization +-0.5 s | >=90% | 23.3% | n/a | n/a | n/a | n/a |
| S4 False alarms (clean) | <=5% | 0.0% | n/a | n/a | n/a | n/a |
| S5 Causal link recall | >=+15pp over B1 (CI>0), invented <= B1 | n/a | n/a | n/a | n/a | n/a |
| S6 Cost vs B1 | <=1.5x $/video-minute | $0.2788/min | n/a | n/a | n/a | n/a |
| S7 W@X vs B1@2X | W wins on T1 F1 (CI>0) | n/a | n/a | n/a | n/a | n/a |
| S8 Unseen-frame claims | 0 in T1 | 0 | n/a | n/a | n/a | n/a |
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

![recall by duration](bench-plugin-3runs-2026-11-23_recall_by_duration.png)

## Pareto (T1 F1 / <1 s event recall vs token cap)

| Cap | B1 F1 | W F1 | B1 <1s recall | W <1s recall |
|---|---|---|---|---|
| 25k | 0.37 | n/a | n/a | n/a |

![pareto](bench-plugin-3runs-2026-11-23_pareto.png)

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

| Source | B1 | CC | CVV | POP | VWP |
|---|---|---|---|---|---|
| bnreel | 0.00 | 0.74 | 0.80 | 0.80 | 0.88 |
| boston | 0.80 | 0.88 | 1.00 | 1.00 | 0.80 |
| gujarat | 0.00 | 1.00 | 0.94 | 0.86 | 0.88 |
| jharkhand | 0.50 | 0.55 | 0.80 | 0.75 | 0.80 |
| khizr | 0.50 | 0.80 | 0.80 | 0.80 | 0.62 |
| linagora | 0.00 | 1.00 | 0.88 | 0.80 | 0.80 |
| pietra | 0.50 | 0.94 | 0.88 | 0.80 | 1.00 |
| prague | 0.50 | 0.80 | 0.80 | 0.80 | 0.90 |
| sem | 0.50 | 0.75 | 0.80 | 0.86 | 0.64 |
| steel | 0.00 | 0.94 | 0.88 | 1.00 | 0.94 |

## Precision, strictness and failures

| Arm@budget | T2 precision | T2 events listed/video | T1 strict recall (±0.5 s) | T1 loc | found | Failure rate |
|---|---|---|---|---|---|
| B1@25000 | n/a | n/a | 23.3% | 100.0% | 0.0% |
| CC@25000 | n/a | n/a | 77.8% | 97.2% | 0.0% |
| CVV@25000 | n/a | n/a | 74.4% | 98.5% | 0.0% |
| POP@25000 | n/a | n/a | 83.3% | 100.0% | 0.0% |
| VWP@25000 | n/a | n/a | 78.9% | 100.0% | 0.0% |

## Per-defect-type recall

| Type | B1 | W |
|---|---|---|
| Flash frame | 14.3% | n/a |
| Frozen frame | 71.4% | n/a |
| Jump cut | 12.5% | n/a |
| Audio dropout | 0.0% | n/a |
| Caption blocked | n/a | n/a |
| Wrong number | n/a | n/a |

## All arms: cost and honesty

| Arm@budget | Records | Videos | T1 F1 | T1 prec | T1 rec | T2 rec | T3 rec | T4 acc | In tok | Content tok | $/min | Wall s/min | Unseen T1 | T1 F1 std | Failed runs | Fail rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1@25000 | 40 | 40 | 0.37 | 0.88 | 0.23 | n/a | n/a | n/a | 27234 | 24489 | 0.2788 | 8.3 | 0 | n/a | 0 | 0.0% |
| CC@25000 | 120 | 40 | 0.83 | 0.87 | 0.80 | n/a | n/a | n/a | 160443 | 0 | 0.3050 | 66.1 | 83 | 0.05 | 0 | 0.0% |
| CVV@25000 | 120 | 40 | 0.86 | 1.00 | 0.76 | n/a | n/a | n/a | 243475 | 0 | 0.3907 | 61.3 | 2 | 0.01 | 0 | 0.0% |
| POP@25000 | 40 | 40 | 0.85 | 0.86 | 0.83 | n/a | n/a | n/a | 305874 | 0 | 0.4583 | 101.9 | 7 | n/a | 0 | 0.0% |
| VWP@25000 | 120 | 40 | 0.82 | 0.86 | 0.79 | n/a | n/a | n/a | 82358 | 0 | 0.1695 | 32.4 | 83 | 0.03 | 0 | 0.0% |

## Notes / failures worth reading

- Criteria n/a (missing data): S1, S2, S3, S4, S5, S6, S7, S8, K1, K2, K3.
- T3 not scored: no judge supplied.
- Bootstrap: 10000 resamples, 95% percentile CI, repeats averaged per video. T1 resamples whole source clips (cluster bootstrap); T2/T3/T4 resample videos.
- McNemar tests are per-item and not cluster-adjusted: descriptive only.
- Unseen claims count only frame/crop/audio observations within 0.5 s; hint- or transcript-only claims are unverified.
