# vwatch MCP server and Claude Code plugin

vwatch's perception primitives as [MCP](https://modelcontextprotocol.io) tools, for any agent that can
call tools and look at images. The tools make **no model calls and need no API key**: they turn a local
video file into evidence (timestamped frame grids, crops, a transcript, a loudness meter, model-free
signal hints). Your agent does the watching; the only cost is its own usage.

Why tools and not the watcher: in the [head-to-head](../README.md#head-to-head-with-claude-code-video-tools)
a Claude Code agent steering ffmpeg beat vwatch's own nested watcher loop. Running a second model inside
your agent adds cost and a layer of summarisation; the plugin instead gives your agent the primitives and
teaches it the method through the `watch-video` skill.

## Install

**macOS and Linux: nothing to install first.** The plugin starts the server through a small launcher
([`plugins/vwatch/bin/vwatch-mcp`](../plugins/vwatch/bin/vwatch-mcp)) that:

1. uses `uvx` from `PATH`, `~/.local/bin` or `~/.cargo/bin`, or else installs [uv](https://docs.astral.sh/uv/)
   into `~/.local/bin` with the official installer (no sudo, shell profiles untouched; set
   `VWATCH_UV_INSTALL=0` to forbid this and get an error instead);
2. runs `uvx --python ">=3.11,<3.13" --from "vwatch[mcp] @ git+...@vX.Y.Z" vwatch-mcp`; uv downloads a
   Python if no 3.11/3.12 is installed (newer Pythons are skipped: onnxruntime, used by faster-whisper,
   has no wheels for 3.15 yet);
3. vwatch then finds ffmpeg/ffprobe: `VWATCH_FFMPEG`/`VWATCH_FFPROBE` if set, else the system ones on
   `PATH`, else a static build from [static-ffmpeg](https://github.com/zackees/static_ffmpeg), downloaded
   once in the background when the server starts to `~/.cache/vwatch/ffmpeg/<platform>` (`VWATCH_CACHE`).

One-time first-run download, measured from an empty `HOME` with no uv, Python 3.11+ or ffmpeg:

| Platform | uv | Python | vwatch + deps | static ffmpeg + ffprobe | Total | Time |
|---|---|---|---|---|---|---|
| macOS arm64 | 15 MB | 25 MB | 86 MB | 42 MB | ~170 MB | ~35 s on a home connection (21 s to a running server, then ffmpeg in the background); ~10 s on a GitHub runner |
| Linux x86_64 | 17 MB | 35 MB | ~155 MB | 142 MB | ~350 MB | ~11 s on a GitHub runner; budget ~1 min on 50 Mbit/s |

The first tool call after a fresh install takes ~15 s more on macOS (one-time Python bytecode compilation and
the OS's first-run check of the new native binaries). Later starts take about a second. The first start can be slower than Claude Code's MCP startup timeout on a
slow connection; if `/mcp` shows vwatch as failed, reconnect it once the download is done (or start Claude
Code with `MCP_TIMEOUT=120000`). The launcher prints its progress to stderr only (stdout is the MCP
channel).

Also downloaded on first use: the whisper model, by the first `video_transcript` (or `video_listen`) call
(`VWATCH_WHISPER_MODEL`, default `small`, ~480 MB); transcripts are cached in `~/.cache/vwatch/`
(`VWATCH_CACHE`).

**Use your own ffmpeg**: put `ffmpeg` and `ffprobe` on `PATH` (`brew install ffmpeg`,
`sudo apt install ffmpeg`), or point `VWATCH_FFMPEG` / `VWATCH_FFPROBE` at the binaries. Then nothing is
downloaded. The static builds are GPL-licensed FFmpeg builds (BtbN on Linux, gyan.dev on Windows,
evermeet.cx / osxexperts on macOS) mirrored by static-ffmpeg on GitHub; vwatch does not redistribute them.

**Pin a local checkout** (testing): `VWATCH_SPEC='vwatch[mcp] @ file:///path/to/vwatch'` in the environment
replaces the plugin's `--from` requirement.

### Windows

The launcher is a POSIX `sh` script and plugin MCP configs cannot choose a command per OS, so on native
Windows the plugin's server only starts if `sh` is on `PATH` (for example Git for Windows'
`C:\Program Files\Git\bin`) and uv is installed; this is untested. The reliable route on Windows:

```powershell
winget install astral-sh.uv          # then open a new terminal
claude mcp add vwatch -s user -- uvx --python 3.12 --from "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0" vwatch-mcp
```

ffmpeg is fetched automatically (static-ffmpeg has Windows x64 builds) unless `ffmpeg`/`ffprobe` are on
`PATH` (`winget install ffmpeg`). Under WSL everything works as on Linux.

### Claude Code plugin

```
/plugin marketplace add kashaf12/vwatch
/plugin install vwatch@vwatch
```

From a shell: `claude plugin marketplace add kashaf12/vwatch && claude plugin install vwatch@vwatch`.

The plugin contains:

- the MCP server `vwatch` (tools appear as `mcp__plugin_vwatch_vwatch__video_info`, ...);
- the skill `watch-video`, loaded automatically when a task involves a video file;
- the command `/vwatch:watch <path> [question]`.

### Claude Code without the plugin

```bash
claude mcp add vwatch -- uvx --python 3.12 --from "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0" vwatch-mcp
```

(tools appear as `mcp__vwatch__video_info`, ...; add `-s user` to make it available in every project).

### Claude Desktop, Cursor, Windsurf and other MCP clients

Add to the client's MCP config (`claude_desktop_config.json` for Claude Desktop, `~/.cursor/mcp.json` or
`.cursor/mcp.json` for Cursor):

```json
{
  "mcpServers": {
    "vwatch": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0", "vwatch-mcp"]
    }
  }
}
```

GUI apps often start with a minimal `PATH`; if the client cannot find `uvx`, use its absolute path
(`which uvx`). Without ffmpeg on that `PATH` vwatch fetches its static build (see above); to use a system
ffmpeg add `"env": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}` or
`"env": {"VWATCH_FFMPEG": "/opt/homebrew/bin/ffmpeg", "VWATCH_FFPROBE": "/opt/homebrew/bin/ffprobe"}`.
Or use the launcher from a clone: `"command": "sh", "args": ["/path/to/vwatch/plugins/vwatch/bin/vwatch-mcp",
"--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0", "vwatch-mcp"]`.

### From a clone

```bash
uv sync --all-extras
uv run vwatch-mcp            # or: uv run vw mcp
```

In an MCP config: `"command": "uv", "args": ["run", "--project", "/path/to/vwatch", "vwatch-mcp"]`.
Use `--project`, not `--directory`: `--directory` changes the server's working directory, so relative
video paths would resolve against the clone.

## Paths

Every tool takes `path`: a local video file ffmpeg can read. Absolute paths are safest. Relative paths
resolve against the **server's** working directory, which is not always the agent's. URLs are refused;
the server reads local files only and never fetches anything.

## Tools

Times are seconds from the start of the video. Every image has its time burned in as `mm:ss.ss`
(yellow label, top-left). Every result begins with a text line saying what is shown and an estimated
token cost (`[~N tokens]`, using ~750 pixels per image token and ~3 characters per text token).
Sizes below are measured on a 50 s 540x960 clip.

### `video_info(path)`

Duration, frame count and fps, size, whether there is audio, and whether there is speech (silero
voice-activity check, a second or two; no whisper). Also the token cost of the default overview and of a
1 s seek. Call it first. ~100 tokens.

```
file: /videos/clip.mp4
duration: 50.00 s (00:50.00), 1500 frames at 30 fps
size: 540x960
audio: yes; speech: yes, ~48.3 s of voice (use video_transcript)
video_overview default: low-res skim, up to ~10000 tokens
video_seek cost: ~965 tokens per 1 s at 8 fps (max 16 frames per call)
```

### `video_transcript(path, language=None, t0=None, t1=None, words=False)`

Timestamped speech segments from a local [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
model, cached on disk by file hash and in-process. `language` is a code such as `en` or `hi` (default:
auto-detect). `t0`/`t1` restrict the output to a window; `words=true` adds every word with its start time.
Output is capped at 48,000 characters (narrow it with `t0`/`t1`). ~300 tokens per 50 s of speech.

```
TRANSCRIPT 00:10.00–00:20.00 (language en; [start–end] mm:ss.ss) [~95 tokens]
[00:06.80–00:14.84] Urban plants, urban plants, with a proud Pennsylvania patriot toward the heart and soul of U.S. Steel.
[00:15.64–00:20.74] A strong steel industry is not just a matter of dignity or prosperity and pride.
```

### `video_signals(path, kinds=None, top=40)`

Model-free candidate moments computed from every frame (64 px grey) and the audio loudness:

| kind | meaning |
|---|---|
| `flash` | 1-4 frames that differ from both neighbours |
| `cut` | abrupt picture change |
| `cut_in_speech` | abrupt picture change while sound is on (jump-cut candidate) |
| `freeze` | picture perfectly still while sound continues, between motion |
| `jump_cut` | a one-frame picture jump well above the nearby motion plus an audio splice click, inside speech |
| `silence_in_speech` | near-digital silence with sound on both sides (dropout candidate) |
| `local_change` | change confined to one region (overlay, caption, UI) |

Returns the `top` highest-scoring (1-200), optionally only of `kinds`, in time order. Hints, not
verdicts: a scene cut and a jump cut look alike, so do a sentence pause and a dropout. ~1k tokens for 30.

```
SIGNALS: 31 of 31 candidate moments (top 40 by score, in time order). Model-free hints ...
- 00:13.30–00:13.70 local_change (score 0.30): change confined to one region
- 00:13.43–00:13.53 flash (score 1.00): 3 frame(s) differ from both neighbours
- 00:15.73–00:15.73 cut_in_speech (score 0.75): picture changes abruptly while sound is on
```

### `video_overview(path, fps=None, tile_width=None)`

The whole video as JPEG grids of up to 12 tiles (4 columns), left-to-right, top-to-bottom. By default a
short clip is shown **whole** at watchable size (360 or 270 px tiles at 4-8 fps, if that fits in ~10k
tokens); a longer one is skimmed at 1 fps with 135 px tiles, halving the rate (down to 0.25 fps) to stay
near 6k tokens. Hard caps: 96 frames and ~10k tokens per call; the summary says when a cap applied.
`fps` (0-30) and `tile_width` (90-360) override the automatic choice.
50 s clip: 50 frames in 5 images, ~2.3k tokens.

### `video_seek(path, t0, t1, fps=8)`

Consecutive frames in `[t0, t1]` at `fps` (0.5-30), 180 px tiles, **at most 16 frames per call**; a longer
window gets 16 frames spread evenly (the summary says so). Windows are capped at 60 s. For one-frame
events use 30 fps over <= 0.5 s. ~1k tokens per call.

### `video_crop(path, t, box)`

The frame nearest `t`, cropped to `box = [x0, y0, x1, y1]` (normalized 0..1, (0,0) = top-left), decoded at
source resolution (up to 1920 px wide) and scaled down only if the crop's long edge exceeds 1024 px. For
small text, faces, logos, details. ~100-800 tokens.

### `video_listen(path, t0, t1)`

Loudness (peak dB per 0.1 s; `sil` below -55 dB) and the words heard in the window, with their times.
Windows are capped at 30 s. If whisper is unavailable the meter is still returned. ~80 tokens per 1.5 s.

```
LISTEN 00:13.00–00:14.50: loudness in dB every 0.1 s ('sil' = silence): -7 -11 -13 -10 -10 -9 ...
  words heard (word@seconds): and@12.90 soul@13.54 of@13.94 U@14.18 .S.@14.40
```

## Errors

Bad input never crashes the server: a missing file, a non-video, a window past the end, an invalid box
or an out-of-range argument comes back as a tool error with a message saying what to fix.

## Caching

Per video, keyed by (absolute path, mtime, size): the probe, the signal track, the voice-activity result
and transcripts stay in memory for the 8 most recently used files. Transcripts are also cached on disk.
Editing or replacing the file invalidates the entry.

## The method (what the skill teaches)

1. Plan: what would answer the question, where on screen, which signal kinds matter.
2. Free evidence first: `video_info`, `video_transcript` if there is speech, `video_signals`.
3. Skim once with `video_overview`, predicting what comes next; note what breaks the expectation.
4. Zoom only on surprises, relevant hints and moments the question names: `video_seek`, `video_crop`,
   `video_listen`. Confirm or reject every hint.
5. Report: answer first, then each moment with its time and the tool that showed it; never claim a
   moment you did not look at, and say what was only skimmed.

The full text is in [`plugins/vwatch/skills/watch-video/SKILL.md`](../plugins/vwatch/skills/watch-video/SKILL.md).

## Releasing

`plugins/vwatch/.claude-plugin/plugin.json` holds the plugin `version` and the git tag in the
`--from ...@vX.Y.Z` argument the launcher passes to `uvx` (the launcher itself has no version). Bump both together with `pyproject.toml` and `src/vwatch/__init__.py`
(`tests/test_mcp_server.py::TestPlugin::test_versions_agree` fails if they disagree), then push the tag
`vX.Y.Z`: until the tag exists, the plugin's server cannot start.
