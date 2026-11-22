"""Arms: the different ways of showing a video to the model."""

from __future__ import annotations

from vwatch.arms.baseline import run_baseline
from vwatch.arms.common import ArmResult, VideoContext
from vwatch.arms.external import EXTERNAL_ARMS, run_external
from vwatch.arms.watcher import VARIANTS, run_watcher
from vwatch.llm.base import Backend
from vwatch.schemas import Arm, Task


def run_arm(arm: Arm, ctx: VideoContext, task: Task, prompt: str, budget: int, backend: Backend) -> ArmResult:
    if arm is Arm.B0:
        return run_baseline(ctx, task, prompt, budget, backend, with_transcript=False)
    if arm in (Arm.B1, Arm.B2):
        return run_baseline(ctx, task, prompt, budget, backend, with_transcript=True)
    if arm is Arm.B1H:
        return run_baseline(ctx, task, prompt, budget, backend, with_transcript=True, with_hints=True)
    if arm in VARIANTS:
        return run_watcher(ctx, task, prompt, budget, backend, arm=arm)
    if arm in EXTERNAL_ARMS:  # uncapped: budget is ignored (see arms/external.py)
        return run_external(arm, ctx, task, prompt, backend)
    raise ValueError(f"unknown arm {arm}")


__all__ = ["ArmResult", "VideoContext", "run_arm"]
