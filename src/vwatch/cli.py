"""`vw` command line.

    vw watch VIDEO -q "question"        watch any video with the watcher and answer a question
    vw watch VIDEO --arm B1 -q "..."    same question through a baseline, for comparison
    vw signals VIDEO                    print the low-level signal hints
    vw mcp                              run the MCP server (stdio) with the video tools for your agent
    vw dataset build [...]              build the eval dataset
    vw eval run [...]                   run arms over the dataset
    vw eval report [...]                score runs and write the report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vwatch.imaging import fmt_time
from vwatch.llm.base import LLMError
from vwatch.media import MediaError

WATCH_ARMS = "W (default), W-noQ, W-noP, W-noCrop, W-noS, B0, B1, B1H"


def _video_arg(path: str) -> str:
    if not Path(path).is_file():
        raise argparse.ArgumentTypeError(f"no such video file: {path}")
    return path


def _watch(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vw watch", description="Watch a video and answer a question about it.")
    ap.add_argument("video", type=_video_arg, help="path to a video file ffmpeg can read")
    ap.add_argument("-q", "--question", required=True, help="what you want to know about the video")
    ap.add_argument("--arm", default="W", choices=WATCH_ARMS.replace(" (default)", "").split(", "),
                    metavar="ARM", help=f"how the video reaches the model: {WATCH_ARMS}")
    ap.add_argument("--budget", type=int, default=25000, help="evidence-token cap per video (default 25000)")
    ap.add_argument("--model", default="claude-opus-5-5", help="model id (default claude-opus-5-5)")
    ap.add_argument("--effort", default="medium", help="reasoning effort, e.g. low, medium, high (default medium)")
    ap.add_argument("--language", default=None, help="speech language hint for whisper, e.g. hi, en")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)

    from vwatch.arms import VideoContext, run_arm
    from vwatch.llm import default_backend
    from vwatch.schemas import Arm, Task
    from vwatch.tasks import free_prompt

    print(f"preparing {a.video} (transcript + signals)...", file=sys.stderr)
    ctx = VideoContext.load(a.video, language=a.language)
    backend = default_backend(model=a.model, effort=a.effort)
    print(f"watching with {a.arm} ({backend.model}, budget {a.budget} tokens)...", file=sys.stderr)
    res = run_arm(Arm(a.arm), ctx, Task.FREE, free_prompt(a.question), a.budget, backend)
    if a.json:
        print(json.dumps({"parsed": res.parsed, "raw": res.raw_output, "error": res.error,
                          "tokens_est": res.content_tokens_est, "cost_usd": res.cost_usd,
                          "turns": res.turns, "observations": [o.model_dump() for o in res.observations]},
                         ensure_ascii=False, indent=2))
        return 0 if not res.error else 1
    if res.error:
        print(f"error: {res.error}", file=sys.stderr)
        return 1
    if res.parsed is None:
        print(res.raw_output)
        print(f"(could not parse the answer: {res.parse_error})", file=sys.stderr)
        return 1
    print(res.parsed["answer"])
    for m in res.parsed.get("moments", []):
        print(f"  {fmt_time(m['t'])}  {m.get('what', '')}")
    frames = sum(1 for o in res.observations if o.kind == "frame")
    crops = sum(1 for o in res.observations if o.kind == "crop")
    print(f"\nlooked at {frames} frames, {crops} crops in {res.turns} turns; "
          f"~{res.content_tokens_est} evidence tokens, ${res.cost_usd:.3f}, {res.wall_s:.0f}s", file=sys.stderr)
    return 0


def _signals(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vw signals", description="Print the model-free signal hints for a video.")
    ap.add_argument("video", type=_video_arg, help="path to a video file ffmpeg can read")
    a = ap.parse_args(argv)
    from vwatch import signals

    print(signals.describe(signals.find_candidates(signals.compute(a.video))))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return _dispatch(argv)
    except (MediaError, LLMError) as exc:  # expected failures: say what went wrong, no traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "watch":
        return _watch(rest)
    if cmd == "signals":
        return _signals(rest)
    if cmd == "mcp":
        from vwatch.mcp_server import main as mcp_main

        return mcp_main(rest)
    if cmd == "dataset" and rest[:1] == ["build"]:
        from vwatch.dataset.build import main as build_main

        return build_main(rest[1:]) or 0
    if cmd == "eval" and rest[:1] == ["run"]:
        from vwatch.evaluation.runner import main as run_main

        return run_main(rest[1:])
    if cmd == "eval" and rest[:1] == ["report"]:
        from vwatch.evaluation.report import main as report_main

        return report_main(rest[1:]) or 0
    print(f"unknown command: {' '.join(argv[:2])}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
