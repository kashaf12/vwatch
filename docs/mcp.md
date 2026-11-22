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

Requirements: [uv](https://docs.astral.sh/uv/) and `ffmpeg` / `ffprobe` on `PATH`. The server is launched
with `uvx` from a pinned git tag, so no clone is needed. The first `video_transcript` (or `video_listen`)
call downloads the local whisper model (`VWATCH_WHISPER_MODEL`, default `small`); transcripts are cached in
`~/.cache/vwatch/` (`VWATCH_CACHE`).

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
claude mcp add vwatch -- uvx --from "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0" vwatch-mcp
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
      "args": ["--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.0", "vwatch-mcp"]
    }
  }
}
```

GUI apps often start with a minimal `PATH`; if the client cannot find `uvx` or `ffmpeg`, use the absolute
path to `uvx` (`which uvx`) and add `"env": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}`.

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
`uvx --from ...@vX.Y.Z` argument. Bump both together with `pyproject.toml` and `src/vwatch/__init__.py`
(`tests/test_mcp_server.py::TestPlugin::test_versions_agree` fails if they disagree), then push the tag
`vX.Y.Z`: until the tag exists, the plugin's server cannot start.
