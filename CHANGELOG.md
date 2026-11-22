# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
