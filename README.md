<div align="center">

# vwatch

**Let your agent watch local videos the way a person does: skim, notice what's off, look closer, cite the time.**

[![CI](https://github.com/kashaf12/vwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/kashaf12/vwatch/actions/workflows/ci.yml)
[![Fresh install](https://github.com/kashaf12/vwatch/actions/workflows/fresh-install.yml/badge.svg)](https://github.com/kashaf12/vwatch/actions/workflows/fresh-install.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

<img src="docs/assets/hero.svg" width="800" alt="Illustration: on a 40-second video with three short defects, 16 evenly spaced frames catch only the 0.8 s freeze; vwatch skims, flags three surprising moments, samples them densely and finds all three.">

</div>

vwatch gives your agent eyes and ears for video files on your machine: timestamped frame grids, dense seeks, crops, a transcript, a loudness meter and model-free hints about where something looks wrong. A skill teaches it how to use them without burning tokens on frames where nothing happens.

## Install in Claude Code

```
/plugin marketplace add kashaf12/vwatch
/plugin install vwatch@vwatch
```

That's it on macOS and Linux. On first start the plugin fetches [uv](https://docs.astral.sh/uv/), Python, vwatch and a static ffmpeg by itself (one-time download: about 170 MB on Apple Silicon, 350 MB on Linux; a system ffmpeg is used instead if you have one). Windows: see [the Windows notes](docs/mcp.md#windows).

**Other agents** (Cursor, Claude Desktop, Windsurf, any MCP client): add the `vwatch` MCP server with `uvx`; copy-paste configs are in [docs/mcp.md](docs/mcp.md).

## Usage

```
/vwatch:watch ~/renders/promo-v3.mp4 Any flash frames, freezes, jump cuts or audio dropouts?
/vwatch:watch talk.mp4 What is on screen when the speaker says "quarterly revenue"?
/vwatch:watch lecture.mp4 Summarize this with a timestamp for each section
/vwatch:watch bug-report.mov Read the error message that appears near the end
```

Leave out the question and you get a description of what happens, with times.

Or just ask about a video in plain words; the `watch-video` skill loads on its own:

> "Check ~/Desktop/final-cut.mp4 for glitches before I upload it."
>
> "When does the presenter first show the pricing slide in demo.mp4?"

## What you get

| | What it does |
|---|---|
| `video_info` | Duration, fps, size, audio and speech, and what an overview or seek will cost. Call it first. |
| `video_transcript` | Timestamped speech from a local whisper model, optionally word by word or for one window. |
| `video_signals` | Free hints from every frame and the audio: flash, freeze, cut, jump cut, silence in speech, local change. |
| `video_overview` | The whole video as tiled frame grids with times burned in; short clips come back whole. |
| `video_seek` | Up to 16 consecutive frames in a window, up to 30 fps, for a close look at one moment. |
| `video_crop` | One frame cropped to a region at full resolution, for small text, faces and logos. |
| `video_listen` | Loudness every 0.1 s and the words heard in a window, for audio glitches. |
| `watch-video` skill | The method below, with a cheat sheet of which signal to check for which problem and a token budget. |
| `/vwatch:watch` | `/vwatch:watch <path> [question]` runs the method on one file. |

The tools make **no model calls and need no API key**: your agent does the looking, so the only cost is its own usage. Every response starts with a line saying which times it shows and roughly how many tokens it costs, so your agent can budget. For example, `video_signals` on a 50 s clip:

```
SIGNALS: 31 of 31 candidate moments (top 40 by score, in time order). Model-free hints ...
- 00:13.30–00:13.70 local_change (score 0.30): change confined to one region
- 00:13.43–00:13.53 flash (score 1.00): 3 frame(s) differ from both neighbours
- 00:15.73–00:15.73 cut_in_speech (score 0.75): picture changes abruptly while sound is on
```

Hints are not verdicts (a sentence pause looks like an audio dropout); the skill has your agent confirm each one by looking or listening. Full reference: [docs/mcp.md](docs/mcp.md).

## How it watches

Uniform frame sampling misses anything shorter than the gap between frames: a one-frame flash, a half-second freeze, a dropped word. Sampling densely everywhere costs a fortune. The skill has your agent work like a person instead:

1. **Plan.** Decide what would answer the question, where on screen it would be, and which signals matter.
2. **Take the free evidence.** Video facts, the transcript and the signal hints are cheap text, and the hints are computed from every frame.
3. **Skim once.** One overview of the whole video, noting anything that breaks expectation.
4. **Zoom on surprises.** Dense seeks, crops and listens only where the skim, the hints or the question point.
5. **Cite times.** Answer first, then each moment with its timestamp and the tool that showed it; say what was only skimmed.

## Results

<!-- RESULTS:START -->
(results landing: Claude Code + vwatch plugin vs plain Claude Code and other video plugins)
<!-- RESULTS:END -->

Earlier, on 40 real videos with injected defects, vwatch's own watcher loop beat frame sampling by +32 pp F1, but plain Claude Code with ffmpeg scored 0.86 F1 against the watcher's 0.69. That is why vwatch now ships as tools for your agent. Full tables, charts and caveats: [docs/benchmark.md](docs/benchmark.md).

## Standalone CLI

vwatch also runs on its own from a clone, with its own watcher loop (this one calls a model: an `ANTHROPIC_API_KEY` or a logged-in Claude Code):

```bash
uv run vw watch clip.mp4 -q "Is there any moment where the on-screen price contradicts the narration?"
uv run vw signals clip.mp4     # the free signal hints only, no model call
```

Setup, options and the eval commands: [docs/cli.md](docs/cli.md).

## FAQ

**What does the first run download?** uv, Python, vwatch and a static ffmpeg: about 170 MB (~35 s) on Apple Silicon, about 350 MB on Linux x86_64. The first transcript also fetches a whisper model (~480 MB for the default `small`). Later starts take about a second. [Details](docs/mcp.md#install).

**Is my video uploaded anywhere?** vwatch reads local files only, refuses URLs, and runs ffmpeg and whisper on your machine. The frames and text it returns become part of your agent's conversation, so they go wherever your agent's model runs, like any image you paste.

**Can I use my own ffmpeg?** Yes. If `ffmpeg` and `ffprobe` are on `PATH`, or `VWATCH_FFMPEG` / `VWATCH_FFPROBE` point at them, nothing is downloaded.

**Windows?** The plugin's launcher is a shell script and is untested on native Windows. Install uv (`winget install astral-sh.uv`) and add the server with `claude mcp add`; see [docs/mcp.md](docs/mcp.md#windows). WSL works like Linux.

**`/mcp` shows vwatch as failed after install.** On a slow connection the first download can outlast Claude Code's MCP startup timeout. Reconnect once it finishes, or start Claude Code with `MCP_TIMEOUT=120000`.

**Which videos work?** Any local file ffmpeg can read (.mp4, .mov, .mkv, .webm, ...). Pass absolute paths when you can.

## Learn more

[Benchmark](docs/benchmark.md) · [Eval harness](docs/eval-harness.md) · [Datasets](docs/DATA.md) · [MCP reference](docs/mcp.md) · [CLI](docs/cli.md) · [Roadmap](docs/roadmap.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## License

Code: [MIT](LICENSE). Datasets keep their own licenses and are not redistributed here; see [docs/DATA.md](docs/DATA.md). To cite vwatch, use [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).

## Acknowledgements

[FFmpeg](https://ffmpeg.org/) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) do the media work; [static-ffmpeg](https://github.com/zackees/static_ffmpeg) and [uv](https://docs.astral.sh/uv/) make the zero-setup install possible. The benchmark uses [TVBench](https://huggingface.co/datasets/FunAILab/TVBench) (Cores et al., BMVC 2025) and openly licensed videos from [Wikimedia Commons](https://commons.wikimedia.org/) ([full list](docs/DATA.md#t1x-wikimedia-commons-speech-videos)), and compares against [claude-video-vision](https://github.com/jordanrendric/claude-video-vision) (with [whisper.cpp](https://github.com/ggml-org/whisper.cpp)) and [popcorn](https://github.com/haithamelmengad/popcorn).
