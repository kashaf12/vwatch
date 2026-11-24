# Security policy

## Supported versions

Only the latest release gets fixes. The Claude Code plugin always installs the release it pins, so update
the plugin (`/plugin marketplace update vwatch`) to pick up a fix.

## Reporting a vulnerability

Please don't open a public issue. Use
[GitHub's private vulnerability reporting](https://github.com/kashaf12/vwatch/security/advisories/new)
instead. You should get a reply within a week.

## What vwatch touches

- It reads local video files you point it at, refuses URLs, and runs `ffmpeg`, `ffprobe` and whisper
  locally.
- The MCP tools make no network calls and no model calls. The only downloads are uv, Python, vwatch and a
  static ffmpeg on first start, and the whisper model the first time a transcript is requested.
- Frames and text returned by the tools go into your agent's conversation, so they reach whatever model your
  agent uses.

Path handling, ffmpeg argument building and anything that could make the server read or run something the
user did not ask for are in scope.
