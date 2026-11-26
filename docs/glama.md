# Glama listing

vwatch is listed at <https://glama.ai/mcp/servers/kashaf12/vwatch>. `glama.json` in the repo root
lets the maintainer claim that listing. Until Glama has a working build of the server, the listing
shows no score and "This server cannot be deployed".

## One-time setup on Glama

1. Merge this file and `glama.json` to `main`.
2. Sign in to Glama with GitHub as `kashaf12` and press **Claim** on the listing page.
3. Open the listing's admin page, then **Dockerfile**, and fill in the form with the values below.
4. Deploy, wait for the build test to pass, press **Make Release**, then publish the release.
5. Check the listing's score page: the server should now show its 7 tools and a score.

## Build settings

| Field | Value |
| --- | --- |
| Python version | `3.12` (server.json pins `>=3.11,<3.13`) |
| Build steps | `["pip install ."]` |
| CMD arguments | `["vwatch"]` |
| Environment variables | none (`{"type": "object", "properties": {}, "required": []}`) |

Notes:

- The MCP tools need no API key. `ANTHROPIC_API_KEY` is only for the standalone CLI, so leave it out.
- ffmpeg is not needed to start the server or list its tools. `static-ffmpeg` fetches it in the
  background on first start, and the server keeps running if that fails.
- The whisper model is only downloaded on the first `video_transcript` call, so the build stays small.

Checked locally on Python 3.12: `pip install .` followed by `vwatch` answers `initialize` and lists
`video_info`, `video_transcript`, `video_signals`, `video_overview`, `video_seek`, `video_crop` and
`video_listen` over stdio.
