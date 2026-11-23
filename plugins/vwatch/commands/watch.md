---
description: Watch a local video with the vwatch tools and answer a question about it, citing timestamps
argument-hint: <path/to/video> [question]
allowed-tools: mcp__plugin_vwatch_vwatch__video_info, mcp__plugin_vwatch_vwatch__video_transcript, mcp__plugin_vwatch_vwatch__video_signals, mcp__plugin_vwatch_vwatch__video_overview, mcp__plugin_vwatch_vwatch__video_seek, mcp__plugin_vwatch_vwatch__video_crop, mcp__plugin_vwatch_vwatch__video_listen
---

Watch a video and answer: $ARGUMENTS

The first argument is the path to a local video file; the rest is the question. If no question is given,
describe what happens in the video, with times. Pass the tools an absolute path (resolve a relative one
against your working directory).

Follow the watch-video skill (vwatch method) with the vwatch MCP tools:

1. Plan in two or three lines: what would answer the question, where on screen, which signal kinds matter.
2. `video_info`, then `video_transcript` if there is speech, then `video_signals` with the kinds from the plan.
3. `video_overview` once; note anything that breaks your expectation.
4. `video_seek` / `video_crop` / `video_listen` only on surprises, signal hints relevant to the question, and
   moments the question names. Confirm or reject each hint by looking or listening.
5. Answer first, then the moments as seconds (or mm:ss.ss) with what you saw and which tool showed it. Only
   claim what you looked at; say what was only skimmed.
