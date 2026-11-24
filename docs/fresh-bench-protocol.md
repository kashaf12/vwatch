# Fresh benchmark protocol (pre-registered)

This file is committed before any fresh-benchmark data is built or any tool is run on it. Everything it fixes
(sources, tasks, scoring, the claims we may make) stays as written until the results are published. Any later
deviation is listed under "Deviations" with the reason, next to the results.

## Why

The plugin's hint and skill wording was revised after reading its failures on the original 40 T1x benchmark
videos, so those numbers are optimistic for vwatch. A clean comparison needs videos that no vwatch decision has
seen, run once for every tool, under rules fixed in advance.

## Tools (arms)

| Arm | What runs |
|---|---|
| VWP | Claude Code + vwatch plugin **v0.2.5**, prompted `/vwatch:watch <video> <task>`. The harness runs the plugin's MCP server from this repository's checkout; its `src/vwatch` and `plugins/` are identical to tag v0.2.5 except the harness-only files changed with this protocol (`tasks.py`, `evaluation/runner.py`). |
| CC | Plain Claude Code (Bash with ffmpeg/ffprobe/python, Read) |
| CVV | Claude Code + claude-video-vision (pinned commit in `scripts/setup_external.sh`) |

Same model for all: `claude-opus-5-5`, effort medium, one run per arm and item, no token cap. Each run is
sandboxed in a fresh temp dir with the existing tool-call audit. A run the audit flags is inspected; if the flag
is a false positive of the auditor (not a real escape), the auditor is fixed and that run is repeated for that
arm only, and this is listed under Deviations.

No vwatch code, prompt, skill or threshold changes between committing this file and publishing the results.

## Data

All videos come from Wikimedia Commons with a CC BY, CC BY-SA, CC0 or public-domain license, and none of them
is a source used in any earlier vwatch set (T1, T1x, fdev, fdev2, TVBench, the acceptance clips, or the
21-minute CopyrightX lecture).

**Source selection is mechanical**, by `eval/bench/build_fresh.py`, committed with this file:

- It runs fixed Commons searches (the query list is in the script), keeps files that meet the criteria below,
  sorts by Commons page id and takes the first ones in that order. It never sees any tool's output.
- It records every candidate it rejects, with the reason, in `eval/data/fresh/SELECTION.md`.

| Set | Count | Criteria |
|---|---|---|
| Short | 10 sources × 4 videos = 40 | 60 s – 30 min source file, ≤ 300 MB; audio with speech (Whisper speech coverage ≥ 60% in the chosen 50 s segment, the existing T1x segment picker's rule) |
| Long | 8 sources × 1 video = 8 | 10 – 30 min, ≤ 500 MB, with an audio track that has sound on ≥ 50% of the file (energy gate); the whole file, re-encoded to 720p30 H.264/AAC |

**Short set:** each source gives 3 defect videos and 1 clean one. Defect types rotate across sources over flash
frame, frozen frame, jump cut and audio dropout, so each type appears 7 or 8 times. The injectors and positions
are the existing T1x ones with seed 4242.

**Long set:** each video gets:
- a 2.0 s on-screen code-word card at a random time between 20% and 80% of its length. The word is drawn with
  seed 4242 from a fixed list of 50 animal names; the card is white with red text, in the lower third.
- one injected defect at a random time (types: freeze ×2, flash ×2, audio dropout ×2, none ×2).

Each long video becomes two items: a defect task and a question task.

The defect task uses a neutral prompt: "You are QA-checking a video…". It lists the four generic defect types
only, without the old prompt's "Hindi YouTube Short" wording, and is identical for every arm.

The question task is: "A code word is shown on screen for about two seconds at some point. What is the code
word?"

## Scoring (fixed now)

- **Defects:** the existing T1 scorer. A report is a hit if its type matches and it overlaps the truth
  window within ±0.5 s. Precision, recall and F1 are computed over all 48 defect items (40 short + 8 long).
- **Scored as labeled.** No item, moment or report is excluded after scoring, including edits that may already
  be in the source footage. Moments like that are described in the report, never removed.
- **Code word:** correct if the answer contains the word, case-insensitive. Accuracy is over the 8 long videos.
- **Failures:** a failed or unparseable run counts as a miss on defect videos, a false alarm on clean ones, and
  a wrong answer on the question.
- **Cost and time:** billed $ per video-minute and wall seconds per video-minute, reported separately for the
  short and long sets.
- **Uncertainty:** paired differences with a cluster bootstrap over sources (10,000 resamples, seed 0).

## Claims we may make

- **Primary metric:** defect F1 over all 48 items.
- **"vwatch is best"** only if the 95% CI of VWP minus each other arm's F1 is entirely above 0.
- **"Tied"** if any CI includes 0. **"Behind"** if a CI is entirely below 0.
- **Secondary claims** need the same CI rule on their own metric: code-word accuracy, cost, time, the long set
  alone.
- **All numbers are published,** including the ones where vwatch loses.

## Deviations

1. **2026-11-24, data build:** the first build stopped on the first long video because the freeze injection
   used one ffmpeg stream twice without a `split` (an ffmpeg error, exit 234). Fixed by adding the split; source
   selection, injection parameters, seeds and scoring are unchanged. No tool had been run on any fresh item.
2. **2026-11-24, source selection:**
   - **The problem:** Commons search results vary between calls, so two build runs picked different short
     sources: 20 distinct files over 2 runs instead of the same 10. The first long video (the JFK
     inauguration speech) was also the third short source.
   - **The fix:**
     - the selection is now frozen in `eval/bench/fresh_sources.json` on first use and reused after;
     - a frozen source that fails stops the build instead of being replaced;
     - long-video selection skips any file already in the short set.
   - **What was kept:** the short set is the most recent completed selection (run 2). The long set is selected
     once by the same rule and then frozen.
   - **Unchanged:** the queries, criteria, seeds, injection and scoring. No tool had been run on any fresh
     item, and the run-1 outputs were deleted.
