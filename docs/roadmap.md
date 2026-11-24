# Roadmap

From the pilot verdict and the head-to-head ([benchmark.md](benchmark.md)), in rough order:

- [x] **Head-to-head** with plain Claude Code, claude-video-vision and popcorn (2026-11-22).
- [x] **Ship the primitives, not the watcher**: MCP server and Claude Code plugin (v0.2, see [mcp.md](mcp.md)).
- [x] **Measure the plugin**: tied with plain Claude Code and the other video plugins on accuracy, about 40% cheaper and 2x faster on 50 s videos ([benchmark.md](benchmark.md#claude-code--vwatch-plugin-2026-11-23)).
- [ ] **Close the gap to Claude Code on defects.** Add full-pass detectors like the ones the agents improvised (ffmpeg `freezedetect`, `signalstats`, audio/video seam checks for jump cuts), since freezes and jump cuts are where all of the gap is. The new freeze and jump-cut detectors in `src/vwatch/detectors.py` are the start.
- [ ] **Rework or drop the prediction step.** It adds cost without measurable gain over signals-only attention (W-noP), on both the pilot and the defect benchmark.
- [ ] **Audit sources for edits already in the footage**, with a rule fixed before scoring (e.g. the jump-cut detector on each raw segment, every hit excluded for every tool), then re-run all tools. One such moment in the `sem` lecture accounts for most of the plugin's wrong reports.
- [x] **Long videos** (v0.2.4): seek instead of decoding from the start, background prefetch, batched whisper; no freeze alarms on still slides. A 21-minute lecture went from 313 s to 138–159 s.
- [x] **A fresh held-out test set** ([protocol](fresh-bench-protocol.md), [results](../eval/results/bench-fresh-2026-11-24.md)): tied on accuracy, significantly cheaper and faster.
- [ ] **Flash frames on bright slides:** a white flash on a white slide barely moves the frame-difference signal. Missed twice in the fresh long set.
- [ ] **Archive footage:** old film with splices and damage triggers flash and jump-cut reports in every tool. A "film artefact versus defect" hint could cut those false alarms.
- [ ] **Another fresh set** to measure fixes to the above. The 2026-11-24 set has now been looked at.
- [ ] **More sources.** T1 rests on 4 source clips; more (the T1x Commons set is the start) narrows the confidence intervals more than anything else.
- [ ] **Full run**: all arms × all videos × 3 repeats, thresholds locked first.
- [ ] **A Gemini-backed comparison**, since Gemini takes video natively.
