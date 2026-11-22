# Eval protocol: watcher vs plain Claude

Assume the watcher tools are built and working. This document says how we prove they are worth it, with numbers: the arms, tasks, metrics, scoring, ship/kill thresholds and run protocol. How the videos are made is in [dataset-spec.md](dataset-spec.md).

Rule: thresholds are fixed **before** the first full run. If the pilot shows a threshold is impossible or trivial, change it once, write down why in the Changelog section at the end, and then lock it.

---

## 1. What we compare (the arms)

All arms use the **same model, the same prompt for the task, and the same videos**. Only how the video reaches the model changes.

| Arm | Name | How the video gets to the model |
|---|---|---|
| B0 | Plain frames | N frames evenly spaced, no audio. This is how Claude handles video by default today. |
| B1 | Frames + transcript | B0 plus a timestamped Whisper transcript. **This is the main baseline** because it's the fair, strong version of the "normal" way. |
| B2 | Dense frames | B1 at 2× the frame budget. Answers the question: "is the watcher just paying for more frames?" |
| W | Watcher | Question-first plan, then prediction-error attention plus seek/crop/listen tools. |

Ablations of W, to see which idea earns the gain:

| Arm | What is removed |
|---|---|
| W-noQ | No question-first plan. The agent watches generically, then answers. |
| W-noP | No model prediction step: attention comes only from the low-level signals (frame difference, loudness). |
| W-noCrop | Seek allowed, crop disabled. |
| W-noS | No low-level signal hints at all: attention comes from the model alone. |

External reference arms (not part of ship/kill, reported alongside): **CC** = plain headless Claude Code with Bash (ffmpeg) + Read, **CVV** = CC + the claude-video-vision plugin, **POP** = CC + the popcorn MCP server. Same task prompts; they are **uncapped** (no evidence budget, `content_tokens_est` = 0), so compare them on accuracy, $ and wall time only. Runs whose tool calls touch anything outside their temp workdir are recorded as errors (failures).

Method-only control: **B1H** = B1 plus exactly the same signal-hint text W receives, inside the same budget. W vs B1H shows what the watching method adds beyond the detectors; W − W-noS shows what the detectors add. Note W-noQ also loses planned crops (no plan, so no regions), so the ablation gains overlap and should not be summed.

**Matched budget is the headline comparison.** W and B1 get the same cap on *evidence tokens* per video: the estimated tokens of every image and text block we send (images at w×h/750). Billed tokens (which include re-sent conversation history in W's multi-turn loop) are reported separately as cost. If W wins only when it is allowed more evidence, the win is money, not method.

---

## 2. Tasks (what the agents are asked to do)

| ID | Task | Why it matters | Ground truth source |
|---|---|---|---|
| T1 | **Defect detection** on our own Shorts | The real job. | Defects we inject ourselves, so we know their exact positions (see [dataset-spec.md](dataset-spec.md)). |
| T2 | **Short-event recall**: "list every event in this clip" | Uniform sampling misses fast events, and this task shows where the baseline breaks. | Procedurally rendered clips; the generator writes the exact event list. |
| T3 | **Causal retell**: retell the video using "because" in each sentence | Tests whether it followed the video, not just saw it. | Procedurally rendered cause→effect chains; the generator writes the gold links. |
| T4 | **Targeted QA**: specific questions with one right answer ("what price is on the board when he says 'pachaas'?") | Tests question-first attention directly. | Generated with the clips (and from `props.json` for a real Short), 5 per video. |
| T5 (optional) | Public benchmark subset: 60 multiple-choice clips from TVBench (originally planned: NExT-QA, EgoSchema) | An outside sanity check, so we aren't only grading on our own homework. | Benchmark labels. |

---

## 3. Metrics (how we put numbers on it)

### 3.1 Accuracy

| Metric | Task | Definition |
|---|---|---|
| **Defect recall** | T1 | Injected defects found ÷ injected defects. Reported per defect type. |
| **Defect precision** | T1 | Real defects reported ÷ all defects reported. |
| **Defect F1** | T1 | Harmonic mean of the two above. |
| **Localization hit rate** | T1 | Share of found defects whose reported time is within ±0.5 s of the truth. |
| **False alarm rate** | T1 | Share of clean videos where at least one defect is reported. |
| **Recall by event duration** | T2 | Event recall bucketed by event length: <0.5 s, 0.5–1 s, 1–3 s, >3 s. **This is the key chart.** The baseline should collapse on the short buckets, and the watcher should stay flat. |
| **Causal link score** | T3 | Correct causal links ÷ gold causal links (recall), plus an invented-links count (hallucinations). |
| **QA accuracy** | T4, T5 | Exact or rubric match. |

### 3.2 Cost

| Metric | Definition |
|---|---|
| Input tokens per video-minute | Including every tool call. |
| $ per video-minute | Tokens × current price. |
| Wall-clock per video-minute | Including preprocessing (Whisper, decode). |
| Frames actually looked at | Shows where attention went. |

### 3.3 Efficiency (the single number to quote)

- **Accuracy per 1k tokens** for each task.
- **Pareto curve**: run B1 and W at token caps of 10k, 25k, 50k and 100k per video, then plot accuracy against tokens. The watcher wins if its curve sits **above and to the left** of B1's.

### 3.4 Honesty

| Metric | Definition |
|---|---|
| Unseen-frame claims | Claims about moments the agent never looked at (checked from the tool-call log). Should be zero. |
| Run-to-run spread | Standard deviation across 3 repeated runs per arm. |

---

## 4. How results are scored

- **T1 / T2 / T4**: scored automatically against the labels by script. No judging needed.
- **T3**: an LLM judge with a fixed rubric (one point per gold link matched, one penalty per invented link). A human re-scores a random 20%. If the judge and the human agree less than 85% of the time, fix the rubric before trusting any T3 number.
- **Statistics**: every item is scored for both arms, so comparisons are paired.
  - Binary outcomes (found / missed): McNemar test.
  - F1 and score differences: paired bootstrap, 10,000 resamples, 95% confidence interval. For T1 the bootstrap is **clustered by source clip** (all Shorts come from 4 source videos), so the CI reflects 4 clusters, not 240 independent videos. McNemar is reported as descriptive only.
  - Failed or unparseable runs count against the arm: a false alarm on a clean video, all misses on a defect video. Retries are de-duplicated (last record wins).
  - A difference counts only if its 95% CI excludes zero.
- **Sample size**: about 200 injected defects and 60 clean videos. Because they come from only 4 source clips, the honest CI is wider than ±6 pp; adding new source Shorts is the cheapest way to tighten it.

---

## 5. Pass / kill thresholds (starting targets; lock after the pilot)

All compared with **B1 at a matched token budget** unless stated otherwise.

### Ship (all must hold)

| # | Criterion | Target |
|---|---|---|
| S1 | Defect F1 | ≥ 0.85 absolute, and ≥ +15 pp over B1 |
| S2 | Recall on events < 1 s | ≥ +25 pp over B1, **and** W's T2 precision ≥ B1's − 10 pp (no winning by listing guesses) |
| S3 | Localization hit rate (±0.5 s), over **all** truth defects (a miss is a failure) | ≥ 90% |
| S4 | False alarm rate on clean videos | ≤ 5% |
| S5 | Causal link recall | ≥ +15 pp over B1, with invented links no higher than B1 |
| S6 | Cost | ≤ 1.5× B1's $ per video-minute at the operating point |
| S7 | Beats B2 | W at budget X beats B1 at budget 2X on T1 F1 (it isn't just buying frames) |
| S8 | Unseen-frame claims: reported times with no frame/crop/audio evidence within 0.5 s (hints and transcript don't count as seeing) | 0 in the T1 runs |

### Kill (any one is enough)

| # | Condition | Meaning |
|---|---|---|
| K1 | W beats B1 by < 5 pp on T1 and T2 at matched budget | The tools add complexity for nothing. Use B2 (just more frames). |
| K2 | B2 matches W on T1 | The gain was frame count, not method. |
| K3 | All three ablations score within 3 pp of full W | None of the three ideas does the work, so the result is noise or prompt luck. |

Informational (non-gating): **M1** = W vs B1H on T1 F1. If W's lead over B1 vanishes against B1H, the gain is the detectors, not the watching method.

### Attribution (from the ablations)

Report how much of the gain each idea contributes: (W − W-noQ), (W − W-noP) and (W − W-noCrop). Any idea that adds < 3 pp gets cut from the product to keep it simple.

---

## 6. Run protocol

1. **Pilot**: 10 videos, all arms. Check that the scripts score correctly and that the thresholds aren't impossible. Adjust once and log the change.
2. **Lock**: commit this file. A threshold changed after this point is reported as a changed threshold, not a pass.
3. **Full run**: all arms × all videos × 3 repeats. Fixed model version, temperature 0 where the API allows it.
4. **Report**: fill in [`eval/results/TEMPLATE.md`](../eval/results/TEMPLATE.md) (`vw eval report` renders it). Headline line format:
   `W vs B1 @25k tokens: T1 F1 0.xx vs 0.xx (+xx pp, CI [a, b]), <1s recall +xx pp, cost 1.xx×`
5. **Regression**: after any watcher change, re-run the locked set. A drop of more than 3 pp on any S-metric blocks the change.

### Run IDs

`YYYY-MM-DD_<arm>_<budget>k_r<repeat>`, e.g. `2026-11-29_W_25k_r2`. One JSONL file per run goes in `eval/runs/`, one line per (video, task) with the raw output, the score, the tokens and the full tool-call log.

---

## Changelog

- 2026-11-22: first draft, thresholds unlocked.
- 2026-11-22 (build): implementation decisions, thresholds unchanged and still unlocked.
  - Defect types adapted to what the real Shorts contain (motion graphics, no filmed faces): `caption over face` → `caption_blocked`; `cut mid-word` → `jump_cut` (A/V chunk removed mid-speech); `wrong subtitle text` dropped (Devanagari re-rendering not reliable); `audio desync` replaced by `audio_dropout`. `wrong_number` and `flash_frame`, `frozen_frame` kept.
  - T2/T3/T4 ground truth is procedural instead of hand-labelled, so no labelling time is needed; T5 not built.
  - W-noP redefined as "signals only" (see the ablation table).
  - Budget = evidence tokens (see section 1).
- 2026-11-22 (review): an independent validity review added B1H and W-noS, clustered bootstrap by source, failed-run scoring, a stricter S3/S8, an S2 precision guard, tighter T1 overlap matching (reported span ≤ 2× truth + 1 s) and stricter T4 matching. Known dataset artefacts to disclose: injected defects are cleaner than real faults (digital-zero dropouts, solid overlays, odd-number badges), each video has 0 or 1 defect, and all T1 videos derive from 4 source clips.
- 2026-11-22 (external arms): added CC / CVV / POP reference arms (uncapped, leakage-audited); thresholds unchanged.
- 2026-11-22 (release prep): document moved from `SUCCESS_CRITERIA.md` to `docs/eval-protocol.md`; T5 is a TVBench subset. Thresholds unchanged and still unlocked.
