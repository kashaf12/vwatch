# Roadmap

From the pilot verdict and the head-to-head ([benchmark.md](benchmark.md)), in rough order:

- [x] **Head-to-head** with plain Claude Code, claude-video-vision and popcorn (2026-11-22).
- [x] **Ship the primitives, not the watcher**: MCP server and Claude Code plugin (v0.2, see [mcp.md](mcp.md)).
- [x] **Measure the plugin**: tied with plain Claude Code and the other video plugins on accuracy, about 40% cheaper and 2x faster on 50 s videos ([benchmark.md](benchmark.md#claude-code--vwatch-plugin-2026-11-23)).
- [ ] **Close the gap to Claude Code on defects.** Add full-pass detectors like the ones the agents improvised (ffmpeg `freezedetect`, `signalstats`, audio/video seam checks for jump cuts), since freezes and jump cuts are where all of the gap is. The new freeze and jump-cut detectors in `src/vwatch/detectors.py` are the start.
- [ ] **Rework or drop the prediction step.** It adds cost without measurable gain over signals-only attention (W-noP), on both the pilot and the defect benchmark.
- [ ] **More sources.** T1 rests on 4 source clips; more (the T1x Commons set is the start) narrows the confidence intervals more than anything else.
- [ ] **Full run**: all arms × all videos × 3 repeats, thresholds locked first.
- [ ] **A Gemini-backed comparison**, since Gemini takes video natively.
