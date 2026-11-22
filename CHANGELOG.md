# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Jump-cut detector** (`jump_cut` signal kind): a single-frame picture jump well above the nearby motion plus an audio splice click on the same frame boundary, inside speech (`vwatch.detectors`). The watcher's plan can select it.
- `eval/bench/eval_signals.py`: model-free evaluation of the signal candidates per defect type (recall at +-0.5 s, per-kind precision, candidates per clean minute) on the dev split, the original T1 set or the held-out benchmark ids.

### Changed

- **Freeze detector** rewritten: near-zero frame difference confirmed cell by cell at mid resolution, sound on, motion around, and judged by how the hold ends (catch-up jump 0.9, scene cut 0.55, gentle restart dropped as a natural pause). The old rule stays as `signals.detect_freezes` for reference.
- `signals.compute` also decodes a 128 px gray track and 44.1 kHz audio for these detectors (about 0.8 s per video-minute in total).

## [0.2.0] - unreleased (draft)

vwatch in your own agent. The head-to-head showed a capable agent steering its own tools beats vwatch's nested watcher loop, so this release gives agents vwatch's perception primitives instead of a second model.

### Added

- **MCP server** (`vwatch-mcp`, `vw mcp`; `mcp` extra, MCP Python SDK 2.x, stdio): seven read-only tools, `video_info`, `video_transcript`, `video_signals`, `video_overview`, `video_seek`, `video_crop`, `video_listen`. No model calls, no API key. Images come back as JPEG with burned-in timestamps. Every result states the timestamps shown and an estimated token cost. Responses are capped: overview at 96 frames and about 10k tokens, seek at 16 frames and a 60 s window, listen at 30 s, text at 48k characters. Per-video work (probe, signal track, voice activity, transcripts) is cached in-process by (path, mtime, size). Bad files and arguments return tool errors and never crash the server.
- **Claude Code plugin** `plugins/vwatch/` and marketplace `.claude-plugin/marketplace.json`: `/plugin marketplace add kashaf12/vwatch`, then `/plugin install vwatch@vwatch`. The plugin bundles the MCP server, launched with `uvx` from the pinned git tag, so no clone is needed. It also adds the `watch-video` skill (plan, then free signals, then skim, then zoom on surprises, then cite timestamps) and the `/vwatch:watch <path> [question]` command.
- `docs/mcp.md`: tool reference, and config snippets for `claude mcp add`, Claude Desktop and Cursor.
- `vwatch.mcp_tools`: the same tools as plain Python functions.
- **Zero-prerequisite plugin start on macOS/Linux**: the plugin launches the server through `plugins/vwatch/bin/vwatch-mcp`, which installs uv into `~/.local/bin` when it is missing (official installer, no sudo, stdout kept clean for MCP) and pins `--python ">=3.11,<3.13"`. `VWATCH_SPEC` overrides the install spec.
- **ffmpeg without a system install**: `vwatch.media` resolves ffmpeg/ffprobe from `VWATCH_FFMPEG`/`VWATCH_FFPROBE`, then `PATH`, then a static build fetched once by [static-ffmpeg](https://github.com/zackees/static_ffmpeg) (new dependency of the `mcp` extra) into `~/.cache/vwatch/ffmpeg/`. The MCP server prefetches it in the background at startup.
- `video_signals` accepts and documents the `jump_cut` kind and rejects unknown kinds with the list of valid ones (`signals.KINDS`).
- `.github/workflows/fresh-install.yml` + `scripts/smoke_mcp.py`: installs and runs the plugin on Ubuntu and macOS runners with no uv and no ffmpeg, and calls the tools over MCP.

## [0.1.0] - 2026-11-22

First public release.

### Added

- **Watcher (arm W)**: question-first plan, low-resolution overview with model prediction error plus model-free signal hints, dense `seek` / `crop` / `listen` on the surprising windows, all within an evidence-token budget.
- **Signals** (`vw signals`): model-free detectors for flashes, cuts, cuts during speech, freezes, silences during speech and local changes.
- **Baselines**: B0 (uniform frames), B1 (frames + Whisper transcript), B2 (B1 at 2x budget), B1H (B1 + the same signal hints).
- **Ablations**: W-noQ, W-noP, W-noCrop, W-noS.
- **External reference arms**: plain headless Claude Code (CC), Claude Code + claude-video-vision (CVV), Claude Code + popcorn (POP), sandboxed per run with a tool-call audit; `scripts/setup_external.sh` installs pinned versions into `vendor/`.
- **Model backends**: Anthropic SDK when `ANTHROPIC_API_KEY` is set, otherwise the logged-in Claude Code CLI.
- **Eval harness**: dataset builders for T1 (defects injected into real videos), T1x (Wikimedia Commons speech videos), T2-T4 (procedural clips) and T5 (TVBench subset); resumable runner; scoring, clustered bootstrap CIs, McNemar, ship/kill scorecard, Markdown report and charts.
- **Pilot result** (`eval/results/pilot-2026-11-22.md`): verdict ITERATE.
- `vw` CLI: `watch`, `signals`, `dataset build`, `eval run`, `eval report`.

[Unreleased]: https://github.com/kashaf12/vwatch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kashaf12/vwatch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kashaf12/vwatch/releases/tag/v0.1.0
