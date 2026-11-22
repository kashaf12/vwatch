---
name: watch-video
description: Watch, check or answer questions about a local video file (.mp4, .mov, .mkv, .webm, ...) with the vwatch MCP tools - timestamped frame grids, dense seeks, crops, transcript, loudness meter and model-free signal hints.
when_to_use: Any task that needs understanding, checking or answering questions about a video file - what happens or is said and when, finding a moment, QA for production defects (flash frames, freezes, jump cuts, audio dropouts), reading on-screen text, verifying an edit or a render.
---

# Watch a video like a person

The vwatch tools (`video_info`, `video_transcript`, `video_signals`, `video_overview`, `video_seek`,
`video_crop`, `video_listen`; in Claude Code they appear as `mcp__plugin_vwatch_vwatch__<name>`) turn a
local video into evidence you can look at. They make no model calls: you do the watching. Every image has
its time burned in as `mm:ss.ss`; every result starts with a line saying what it shows and roughly how many
tokens it cost. Always pass an **absolute** `path` (resolve the user's path against your working directory
first): the server may run in a different directory.

A person does not read every frame. They know what they are looking for, skim, notice what breaks their
expectation, and look closely only there. Do the same:

## 1. Plan before any tool call

Write two or three lines: what would answer the question, where on screen it would be, and which
signal kinds matter.

| Looking for | Signal kinds | Confirm with |
|---|---|---|
| flash frame / glitch frame | `flash` | `video_seek` at 30 fps over <= 0.5 s |
| frozen picture | `freeze` | `video_seek` across the span: identical frames while audio continues |
| jump cut / bad edit | `cut_in_speech`, `cut` | `video_seek` +-0.5 s, plus `video_listen` (a cut mid-word) |
| audio dropout / mute | `silence_in_speech` | `video_listen`: a run of `sil` while words continue around it |
| overlay, caption, UI change | `local_change` | `video_crop` on that region |
| what is said | - | `video_transcript` (add `words: true` for exact word times) |
| what happens / when | all | `video_overview`, then `video_seek` |

## 2. Take the free evidence first

1. `video_info` - duration, audio, speech, what an overview and a seek will cost.
2. `video_transcript` if there is speech and it can matter (it usually does).
3. `video_signals` with the kinds from your plan (omit `kinds` to see everything). These are computed
   from every frame and the audio, so they catch one-frame events a skim misses. They are hints, not
   verdicts: a scene cut and a jump cut look alike here, so do a sentence pause and a dropout.

## 3. Skim once, and predict

Call `video_overview` once. A short clip comes back whole at watchable size; a long one as a low-res skim.
Go through it in order and, for each gap between frames, ask what should happen next. Anything that breaks
the expectation, any signal hint relevant to the question, and any moment the question names is a
**surprise** worth a closer look. Everything else stays at skim level.

## 4. Zoom only on surprises

- `video_seek(t0, t1, fps)`: a window around the moment, about +-0.5-1 s at 8-12 fps. One-frame events
  (flashes) need every frame: 30 fps over <= 0.5 s. Max 16 frames per call.
- `video_crop(t, [x0, y0, x1, y1])`: small text, a face, a logo, a detail. Box is normalized, (0,0) top-left.
- `video_listen(t0, t1)`: loudness every 0.1 s and the words heard, for audio events.
- Confirm or reject each hint by looking or listening. Drop hints that turn out to be ordinary edits
  (a clean scene change between shots, a natural pause between sentences) unless the question asks for them.

## 5. Budget

Rough costs: `video_info`, `video_signals`, `video_listen`, `video_transcript` are text (tens to a few
hundred tokens; a long transcript more - narrow it with `t0`/`t1`). `video_overview` ~1-10k tokens.
`video_seek` ~1k tokens per call. `video_crop` ~100-800 tokens. For a clip under a minute, about
10-25k tokens of evidence is plenty; stop as soon as the question is answered. Never seek through the whole
video densely: that is what the overview and the signals are for.

## 6. Report

- Lead with the answer.
- Give every moment as seconds from the start (or `mm:ss.ss`, matching the burned-in stamps), a range
  `t0-t1` for things with duration, with what you saw or heard and which tool showed it.
- Only claim what you looked at or heard. If a part of the video was only skimmed, say so ("checked at
  1 frame/s only"). If you found nothing, say what you checked and at what density.
- Do not invent times between frames you saw; if an event lies between two frames, give the bracket.
