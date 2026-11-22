# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Jump-cut detector** (`jump_cut` signal kind): a single-frame picture jump well above the nearby motion plus an audio splice click on the same frame boundary, inside speech (`vwatch.detectors`). The watcher's plan can select it.
- `eval/bench/eval_signals.py`: model-free evaluation of the signal candidates per defect type (recall at +-0.5 s, per-kind precision, candidates per clean minute) on the dev split, the original T1 set or the held-out benchmark ids.

### Changed

- **Freeze detector** rewritten: near-zero frame difference confirmed cell by cell at mid resolution, sound on, motion around, and judged by how the hold ends (catch-up jump 0.9, scene cut 0.55, gentle restart dropped as a natural pause). The old rule stays as `signals.detect_freezes` for reference.
- `signals.compute` also decodes a 128 px gray track and 44.1 kHz audio for these detectors (about 0.8 s per video-minute in total).

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

[Unreleased]: https://github.com/kashaf12/vwatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kashaf12/vwatch/releases/tag/v0.1.0
