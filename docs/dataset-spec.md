# Dataset spec

How the eval videos are made and labelled. Every label is either injected by script or written by the generator that rendered the clip, so no hand-labelling is needed and the ground truth is exact. Licenses and attribution for third-party footage are in [DATA.md](DATA.md).

The first draft of this spec (below, "planned") assumed hand-labelled real clips. What was actually built is marked "as built"; the reasons are logged in the [eval protocol changelog](eval-protocol.md#changelog).

## A. Defect set (task T1): real videos, defects injected by script

Start from clean source videos. A script injects at most one defect per copy and writes the exact ground truth: type, start time, end time and (for overlays) region. Because we caused each defect, we know the answer with certainty.

Two source pools:

- **T1, own Shorts.** Four clean renders from an autonomous pipeline that renders Hindi YouTube Shorts (motion graphics, burned-in captions, Hindi narration). These are not redistributed; see [DATA.md](DATA.md) for how to plug in your own.
- **T1x, Wikimedia Commons speech videos** (in progress). Ten openly licensed clips in six languages, each cut to one 35–60 s segment. Generic defects only (no caption or number defects, since these clips have no burned-in captions).

| Defect type (as built) | How it's injected | Typical length | Pools |
|---|---|---|---|
| `flash_frame` | Insert 1–3 black or foreign frames | 33–100 ms | T1, T1x |
| `frozen_frame` | Hold one frame while audio continues | 0.5–2 s | T1, T1x |
| `jump_cut` | Remove a 0.3–0.6 s chunk of audio and video mid-speech | instant | T1, T1x |
| `audio_dropout` | Silence the audio mid-speech, picture continues | 0.4–2 s | T1, T1x |
| `caption_blocked` | Cover the caption band with an opaque box | 1–3 s | T1 |
| `wrong_number` | Overlay an on-screen figure that contradicts the spoken one | 2–4 s | T1 |

Planned but changed: "caption over face" became `caption_blocked` (the Shorts have no filmed faces), "cut mid-word" became `jump_cut`, "audio desync" became `audio_dropout`, and "wrong subtitle text" was dropped because re-rendering Devanagari reliably was not possible.

Counts at `--scale full`: 30 per type plus 60 untouched clean videos for the false-alarm rate (240 T1 videos). T1x adds 2 per generic type plus 2 clean per source (100 videos).

The **short defects** (flash frame, jump cut, frozen frame) are the ones uniform 1 fps sampling should mostly miss. They drive criterion S2.

Known artefacts (disclose with any result): injected defects are cleaner than real faults (digital-zero dropouts, solid overlays), each video has 0 or 1 defect, and all T1 videos derive from 4 source clips, so T1 confidence intervals are bootstrapped by source clip.

## B. Event, causal and QA sets (T2, T3, T4): procedural clips

Planned: 30 hand-labelled real clips. As built: procedurally rendered clips whose generator writes the labels.

| Task | Clip | Ground truth written by the generator |
|---|---|---|
| T2 | Calm scene (gradient, static furniture, one drifting element) with 8–12 non-overlapping events, durations spread over the recall buckets (<0.5 s, 0.5–1 s, 1–3 s, >3 s) | Event list with start and end times |
| T3 | 12–25 s chain of 2–4 cause→effect links plus 1–2 distractor events; five template variants | Gold causal links |
| T4 | Number sign, word card, bell icon, corner shapes and blinking squares that change between events | 5 questions per video, one right answer each |

Counts at `--scale full`: T2 20, T3 12, T4 12 videos.

## C. Public benchmark subset (T5)

60 multiple-choice clips from [TVBench](https://huggingface.co/datasets/FunAILab/TVBench) (10 each from six temporal categories), as an outside check so we aren't only grading our own homework. See [DATA.md](DATA.md) for license and citation.

## D. Split

- Pilot: about 15% of each task (at least one item), used for tuning prompts and thresholds.
- Locked test: everything else. **Never tune on it.**

Splits are deterministic given the seed (`--seed 7`).

## Storage

```
eval/
  data/                        # gitignored: rebuilt locally, see DATA.md
    manifest.jsonl             # one DatasetItem per video: task, paths, split, bucket
    SOURCES.md                 # attribution written by the builders
    t1/<video_id>.mp4 + .truth.json
    t2/ t3/ t4/ t5/            # same layout
    _cache/                    # downloads and decoded segments
  bench/*.txt                  # tracked: fixed id lists for benchmark runs
  runs/<run_id>.jsonl          # gitignored: raw run records
  results/                     # tracked: reports and charts
```

Video files are never committed.
