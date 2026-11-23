# The `vw` command line

Most people want vwatch inside their agent: install the [Claude Code plugin](../README.md#install-in-claude-code) or the [MCP server](mcp.md). The `vw` CLI is for using vwatch on its own: it runs vwatch's own watcher loop (which calls a model), prints the free signal hints, and drives the [eval harness](eval-harness.md).

| Command | What it does | Calls a model? |
|---|---|---|
| `vw watch VIDEO -q "question"` | Watch a video with vwatch's watcher and answer a question | yes |
| `vw signals VIDEO` | Print the model-free signal hints | no |
| `vw mcp` | Run the MCP server (stdio), same as `vwatch-mcp` | no |
| `vw dataset build [...]` | Build the eval dataset | no |
| `vw eval run [...]` | Run arms over the dataset | yes |
| `vw eval report [...]` | Score runs and write the report | only with `--judge` |

## Install

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`. For `vw watch` and `vw eval run` you also need either an `ANTHROPIC_API_KEY` or a logged-in [Claude Code](https://claude.com/claude-code) (see [model backends](eval-harness.md#model-backends)).

```bash
git clone https://github.com/kashaf12/vwatch.git
cd vwatch
uv sync --all-extras
uv run vw --help
```

## `vw watch`

Ask a question about any local video:

```bash
uv run vw watch path/to/video.mp4 -q "Is there any moment where the on-screen price contradicts the narration?" --language hi
```

It prints the answer, then one line per moment it cites (`mm:ss.ss  what happened`), and on stderr how many frames and crops it looked at, the evidence tokens, the cost and the wall time. The first run downloads the Whisper `small` model; transcripts are cached in `~/.cache/vwatch/`.

Compare against plain frames on the same question, or get everything as JSON:

```bash
uv run vw watch path/to/video.mp4 -q "..." --arm B1      # uniform frames + transcript, same budget
uv run vw watch path/to/video.mp4 -q "..." --json        # answer, raw output, cost, turns, every frame/crop it saw
uv run vw watch path/to/video.mp4 -q "..." --budget 10000 --effort low
```

| Option | Default | Meaning |
|---|---|---|
| `-q`, `--question` | required | what you want to know about the video |
| `--arm` | `W` | how the video reaches the model: `W`, `W-noQ`, `W-noP`, `W-noCrop`, `W-noS`, `B0`, `B1`, `B1H` (see [arms](eval-harness.md#arms)) |
| `--budget` | `25000` | evidence-token cap per video |
| `--model` | `claude-opus-5-5` | model id |
| `--effort` | `medium` | reasoning effort, e.g. `low`, `medium`, `high` |
| `--language` | auto-detect | speech language hint for whisper, e.g. `hi`, `en` |
| `--json` | off | print the full result as JSON |

On the [benchmark](benchmark.md), a plain Claude Code agent with ffmpeg beat this watcher loop on defect detection. If you already use an agent, the [plugin](../README.md#install-in-claude-code) is the better route.

## `vw signals`

The model-free signal hints on their own (free, no model call):

```console
$ uv run vw signals clip.mp4        # a public-domain speech clip with a 3-frame black flash injected at 44.27 s
- 00:24.77–00:24.83 local_change (score 0.30): change confined to one region
- 00:44.27–00:44.37 flash (score 1.00): 3 frame(s) differ from both neighbours
```

Hints are not verdicts: a held frame before an edit looks like a freeze, an edit in the source looks like a jump cut, and a sentence pause looks like an audio dropout. Deciding which is which is the model's job. `eval/bench/eval_signals.py` measures the detectors alone (recall, precision, false alarms per clean minute) with no model calls.

## `vw mcp`

Runs the MCP server over stdio from a clone (`uv run vw mcp` or `uv run vwatch-mcp`). Configuration for each client is in [mcp.md](mcp.md#from-a-clone).

## `vw dataset build`, `vw eval run`, `vw eval report`

```bash
uv run vw dataset build --root eval/data --scale full --seed 7 --tasks T2 T3 T4
uv run vw eval run --split pilot --tasks T2,T3 --arms B0,B1,W --budget 25000
uv run vw eval report --runs 'eval/runs/*.jsonl' --root eval/data --budget 25000 --out eval/results/pilot.md
```

The full walkthrough, including the T1x and TVBench builders and the external arms, is in [eval-harness.md](eval-harness.md#reproduce).
