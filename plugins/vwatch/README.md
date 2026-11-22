# vwatch plugin for Claude Code

Gives your agent eyes and ears for local video files, and teaches it to watch like a person: plan,
take the free signals, skim once, zoom only on surprises, cite timestamps.

```
/plugin marketplace add kashaf12/vwatch
/plugin install vwatch@vwatch
```

Then: `/vwatch:watch path/to/video.mp4 Are there any flash frames or audio dropouts?`
(or just ask about a video; the `watch-video` skill loads on its own).

**What you get**

- MCP server `vwatch` with seven tools: `video_info`, `video_transcript`, `video_signals`,
  `video_overview`, `video_seek`, `video_crop`, `video_listen`.
- Skill `watch-video`: the vwatch method, step by step, with budget advice.
- Command `/vwatch:watch <path> [question]`.

The tools make no model calls and need no API key; the only cost is your agent's own usage.

**Requirements**: none on macOS/Linux. On first start the launcher (`bin/vwatch-mcp`) installs
[uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is missing, uv fetches Python and vwatch, and
vwatch fetches a static ffmpeg/ffprobe if none is on `PATH` (one-time, ~170 MB on macOS, ~300 MB on Linux).
A system ffmpeg, or `VWATCH_FFMPEG`/`VWATCH_FFPROBE`, is used when present. The first `video_transcript`
call downloads a local whisper model. Windows: see docs/mcp.md (install uv with `winget install astral-sh.uv`).

Full tool reference: [docs/mcp.md](https://github.com/kashaf12/vwatch/blob/main/docs/mcp.md).

**Bumping the version**: `.claude-plugin/plugin.json` holds both `version` and the git tag in the
`--from ...@vX.Y.Z` argument it passes to `bin/vwatch-mcp`; `tests/test_mcp_server.py` checks they match
`pyproject.toml`. The launcher has no version of its own.
