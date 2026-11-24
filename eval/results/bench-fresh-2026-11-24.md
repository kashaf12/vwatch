# Fresh held-out benchmark: results (2026-11-24)

Run under [docs/fresh-bench-protocol.md](../../docs/fresh-bench-protocol.md). The protocol was committed before any data was built or any tool was run, and its three deviations are listed there. The data is 48 never-before-used Wikimedia Commons videos:
- **Short set:** 40 clips of 50 s from 10 sources.
- **Long set:** 8 whole videos of 12–29 min, each with a 2 s code-word card.

Tools, each run once with the same model (`claude-opus-5-5`, effort medium) and no token cap:
- **Claude Code + vwatch plugin v0.2.5**;
- **plain Claude Code**;
- **Claude Code + claude-video-vision**.

There were 168 runs plus 8 re-runs after the auditor fix (deviation 3), costing $52.42 in total. Scored as labeled, with no exclusions. Reproduce with `uv run python eval/bench/score_fresh.py`.

## Result

| Tool | Items | Defect F1 | Precision | Recall | Clean videos with a false alarm | Failed runs | Code word | $/video-min short | $/video-min long | s/video-min short | s/video-min long |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Code + vwatch plugin | 48/48 | **0.62** | 0.52 | 0.78 | 4/12 | 0 | 8/8 | $0.20 | $0.01 | 39 | 6 |
| Plain Claude Code | 48/48 | **0.68** | 0.59 | 0.81 | 3/12 | 0 | 8/8 | $0.32 | $0.02 | 69 | 7 |
| claude-video-vision | 48/48 | **0.64** | 0.58 | 0.72 | 3/12 | 0 | 8/8 | $0.32 | $0.03 | 64 | 13 |

| Defect type | Claude Code + vwatch plugin | Plain Claude Code | claude-video-vision |
|---|---|---|---|
| audio_dropout | 9/9 | 9/9 | 9/9 |
| flash_frame | 7/9 | 9/9 | 8/9 |
| frozen_frame | 8/10 | 9/10 | 8/10 |
| jump_cut | 4/8 | 2/8 | 1/8 |

Paired defect-F1 differences (vwatch plugin minus each tool), cluster bootstrap over sources:

- vs Plain Claude Code: -6.0 pp, 95% CI [-18.5, +4.0] -> tied
- vs claude-video-vision: -2.0 pp, 95% CI [-15.2, +8.2] -> tied

Protocol claim on the primary metric: **TIED**

## Cost and time (secondary claims; same CI rule)

| | Short: $ / run | Short: s / run | Long: $ / run | Long: s / run |
|---|---|---|---|---|
| Claude Code + vwatch plugin | **$0.17** | **32** | **$0.27** | **122** |
| Plain Claude Code | $0.26 | 57 | $0.41 | 143 |
| claude-video-vision | $0.26 | 53 | $0.57 | 272 |

Totals over all 56 items, vwatch ÷ other tool. Cluster bootstrap over sources, 10,000 resamples, seed 0:

| | vs plain Claude Code | vs claude-video-vision |
|---|---|---|
| Cost | **0.64×** [0.58, 0.71] | **0.56×** [0.47, 0.66] |
| Wall time | **0.71×** [0.61, 0.82] | **0.50×** [0.39, 0.63] |

Every interval lies entirely below 1, so "cheaper" and "faster" both meet the protocol's claim rule.

## What this says

- **Accuracy is a tie.** vwatch's defect F1 (0.62) is the lowest point estimate of the three tools, but both paired 95% intervals include zero. Under the protocol that is "tied", not "best" and not "behind".
- **Every tool scores much lower than on the old set** (0.62–0.68 here against 0.82–0.86 there).
  - The new sources include old archive films (1941, 1961) and news footage with real splices and film damage. All three tools reported those as flash frames (14–16 wrong flash reports each), and they count as false alarms because the clips are labeled clean.
  - On the old set, vwatch's hint wording had been revised after reading its failures there. Here it had not.
- **vwatch found the most jump cuts:** 4 of 8, against 2 and 1. All three found every audio dropout (9 of 9) and every code word (8 of 8).
- **vwatch's new weaknesses**, named for the roadmap and not fixed against this set:
  - **Flash frames on bright slides:** it missed both flash frames in the two long lecture videos, where a white flash lands on a white slide. Its free detector barely registers the change there (a weak 0.3 "local change"). Plain Claude Code caught both.
  - **A freeze in a 1984 NASA film** that its detector did not flag.
- **It costs about a third less and takes about 30–50% less time** than either other tool, on short and long videos alike.

## Limitations

- One run per tool, so differences of a few points are noise; the intervals above say how much.
- Two of the eight long videos are lectures from the CopyrightX series. vwatch's still-slide fix (v0.2.4) was developed on a different lecture of that series. The protocol allows them because they are different files, but they look similar.
- 12 clean videos is a small base for false-alarm rates.
- Now that this set has been looked at, it is no longer held out for vwatch. Fixes to the weaknesses above need yet another fresh set to be measured cleanly.
