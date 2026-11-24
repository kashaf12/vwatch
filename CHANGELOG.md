# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.6] - 2026-11-24

### Added

- **On PyPI and in the official MCP Registry** (`io.github.kashaf12/vwatch`). `uvx --python 3.12 vwatch` starts the MCP server: the server's dependencies (`mcp`, `static-ffmpeg`) are now part of the base install, and a new `vwatch` command runs the server. `vwatch[mcp]` still works. Releases are published from a tag by `.github/workflows/publish.yml` with PyPI trusted publishing and the registry's GitHub OIDC login, so no tokens are stored.

- **Fresh held-out benchmark results** (`eval/results/bench-fresh-2026-11-24.md`, `eval/bench/score_fresh.py`): on 48 new videos, under rules committed before any video was chosen, the plugin is tied on accuracy with plain Claude Code and claude-video-vision, and significantly cheaper (0.64× and 0.56× the cost) and faster (0.71× and 0.50× the time).
- **Setup for other agents** in `docs/agents.md`: Codex (checked end to end: it found the same glitches as Claude Code), Gemini CLI, Cursor, VS Code, opencode, Windsurf and Claude Desktop.
- **The repository is also a Gemini CLI extension:** `gemini extensions install https://github.com/kashaf12/vwatch` gives you the server, the skill and a short context file. Tests keep its pinned version in step with the package.

### Fixed

- External-arm audit: a run reading its own Claude Code session folder (where oversized tool outputs are saved) is no longer flagged; other project folders still are.

### Changed

- The README explains how vwatch decides where to look, and what the benchmark numbers mean, in plain language.

## [0.2.5] - 2026-11-24

### Fixed

- **No speech-model download unless you ask for words.** In 0.2.4 the background prefetch started a full transcription for every video with sound. A first-time user who only wanted frames therefore downloaded the whisper model (~480 MB) and paid for a transcription they never asked for. The transcript is now prefetched only when the model is already on disk (checked offline); otherwise the first `video_transcript` call downloads it, as documented. The signal track and voice check are still prefetched; the voice-check model ships with the package. `VWATCH_PREFETCH_TRANSCRIPT=0` turns transcript prefetch off entirely.

## [0.2.4] - 2026-11-23

Long videos, and real footage instead of injected test sets.

### Changed

- **Seeks and crops no longer decode from the start of the file.** Frames near a few spots come from an input seek plus a short decode, pixel-identical to the full decode. On a 21-minute video that takes 0.1 s instead of 4.5–6.6 s per call. Frame sets spread over the whole file keep the one-pass decode, which now stops after its last wanted frame.
- **The MCP server prefetches** a video's signal track, voice check and transcript in the background on first touch; `VWATCH_PREFETCH=0` turns it off. Parallel tool calls share one computation. `video_info` waits at most 2 s for the voice check.
- **Batched whisper decoding:** 2.2× faster on Apple Silicon with the same words. Segments are rebuilt from word times, so transcript timestamps stay sentence-sized. `VWATCH_WHISPER_BATCH=0` restores one window at a time.

### Fixed

- A slow tool call could overwrite a newer cache entry and drop the signal track, so the next call recomputed it. Results now merge into the latest entry.
- **No freeze alarms on still slides.** On a real lecture, a 38 s narrated slide between two talking-head shots was reported as a freeze with score 0.9. A hold that starts with a cut, or ends with a cut to a different-looking shot, is no longer a cell-level freeze. A hold with no motion before it and a jump after (a pause, then a pan) now scores 0.55 with a "check the frames" note.

## [0.2.3] - 2026-11-23

### Added

- **More freezes found at cell level:**
  - a catch-up jump measured on the cell track (after fast motion, e.g. animation, the 64 px jump looks ordinary);
  - a single small refresh blip no longer splits a hold;
  - a repeated frame with constant encoder noise counts as an exact repeat.
- **Results:**
  - Freeze recall on a fresh validation set (`fdev2`, never used for tuning): 56/60 → 59/60 at 100% precision.
  - Held-out benchmark detector recall: 5/7 → 6/7.
  - Plugin over 3 runs: 18/21 freezes (from 15/21), F1 0.85, recall 0.83.
- `make_freeze_dev.py --prefix`, and `eval_signals.py fdev2`.

### Changed

- Freeze hints say the frames are identical at mid resolution (an exact repeat, not a dissolve or slow move). The skill tells the agent to check it is seeking inside the hinted span.

### Notes

- The hint and skill wording in 0.2.2 and 0.2.3 was revised after reading the plugin's failures on the benchmark videos, so its benchmark numbers are optimistic. See `docs/benchmark.md`.

## [0.2.2] - 2026-11-23

### Added

- **Freezes in talking heads.** A second freeze pass on the mid-resolution track finds holds where every cell repeats while the cells around move and speech continues. The 64 px average barely moves on a speaking face, so the main rule used to drop these as still shots. A catch-up jump scores 0.9; an exact hold of 1.2 s or more with speech nearly throughout scores 0.6. On a fresh freeze dev set (`eval/bench/make_freeze_dev.py`, 60 new injections disjoint from the benchmark) freeze recall went from 43/60 to 55/60 at 100% precision; on the held-out benchmark detector recall went from 4/7 to 5/7, still with no false alarms.
- `eval/bench/eval_signals.py fdev` scores the detectors on that set.

### Changed

- `silence_in_speech`: sound dropping to exact digital zero mid-speech, in a clip that otherwise keeps room tone, scores 0.9 and the hint says so. On real recordings every injected dropout is exact zero and every natural pause keeps room tone. Clips where every pause is exact zero (synthetic narration) get a note instead.
- The `flash` hint says the picture returns to where it was, so a spliced-in frame showing the same scene is not mistaken for movement.
- `watch-video` skill: how to confirm flashes and dropouts, and reject a hint scored 0.9 or more only with direct evidence.
- Benchmark with 3 runs per tool: the plugin goes from F1 0.79 to 0.82 (recall 0.73 to 0.79), within noise of plain Claude Code (0.83) and claude-video-vision (0.86), at about half their cost. See `docs/benchmark.md`.

### Fixed

- External-arm audit: numpy's `~` in inline code, sed substitutions and `/dev/stdin` no longer count as leaving the sandbox. 4 plain Claude Code runs had been wrongly failed; they were re-run.

## [0.2.1] - 2026-11-23

### Fixed

- **Audio out of sync on files whose audio stream starts at a different time from the video** (common in WebM/Ogg exports and screen recordings). `media.extract_audio` now pads or trims to the first video frame, so silences, transcript words, loudness and `video_listen` line up with the frames. Before this, a file with audio starting 4 s late reported every audio event 4 s early. Found by the post-release install test on a new video.
- The T1x builder inherits the fix; see the known issue in `docs/DATA.md` for sets built earlier.

## [0.2.0] - 2026-11-23

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
- **Jump-cut detector** (`jump_cut` signal kind): a single-frame picture jump well above the nearby motion plus an audio splice click on the same frame boundary, inside speech (`vwatch.detectors`). The watcher's plan can select it.
- `eval/bench/eval_signals.py`: model-free evaluation of the signal candidates per defect type (recall at +-0.5 s, per-kind precision, candidates per clean minute) on the dev split, the original T1 set or the held-out benchmark ids.
- **VWP benchmark arm** (Claude Code + the vwatch plugin, sandboxed like CC/CVV/POP) and `eval/results/bench-plugin-2026-11-23.md`.

### Changed

- **Freeze detector** rewritten: near-zero frame difference confirmed cell by cell at mid resolution, sound on, motion around, and judged by how the hold ends (catch-up jump 0.9, scene cut 0.55, gentle restart dropped as a natural pause). The old rule stays as `signals.detect_freezes` for reference.
- `signals.compute` also decodes a 128 px gray track and 44.1 kHz audio for these detectors (about 0.8 s per video-minute in total).

### Fixed

- Freezes in silent clips: without sound as context, a hold inside a moving shot that ends in a same-shot catch-up jump is reported at 0.9, and one that ends in a cut-sized jump (a catch-up far into a fast shot looks like a new scene) at 0.55 for the agent to check.
- Freezes in repeated-frame sources (15 fps stretched to 30 fps, pulldown): motion around a hold is measured ignoring the duplicate frames.
- Local plugin installs (`VWATCH_SPEC='vwatch[mcp] @ file:///...'`) rebuild when the source changes (`tool.uv.cache-keys`), instead of uvx serving a stale build.

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

[Unreleased]: https://github.com/kashaf12/vwatch/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/kashaf12/vwatch/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/kashaf12/vwatch/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/kashaf12/vwatch/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/kashaf12/vwatch/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/kashaf12/vwatch/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kashaf12/vwatch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kashaf12/vwatch/releases/tag/v0.1.0
