"""Run-record hygiene: dedupe retries, pick one model. Pure functions, no I/O."""

from __future__ import annotations

from collections.abc import Sequence

from vwatch.schemas import RunRecord


def is_failed(r: RunRecord) -> bool:
    """A run with an error or no parsed output produced no usable answer."""
    return bool(r.error) or r.parsed is None


def dedupe_runs(runs: Sequence[RunRecord]) -> list[RunRecord]:
    """Keep ONE record per (run_id, video_id, task).

    The runner retries errored pairs by appending a new record to the same run file, so
    the last record wins; but a successful record is preferred over a failed one even if
    the failure came later. Output order = first appearance of each key.
    """
    order: list[tuple[str, str, str]] = []
    best: dict[tuple[str, str, str], RunRecord] = {}
    for r in runs:
        k = (r.run_id, r.video_id, r.task.value)
        if k not in best:
            order.append(k)
            best[k] = r
        elif not is_failed(r) or is_failed(best[k]):
            best[k] = r
    return [best[k] for k in order]


def models_in(runs: Sequence[RunRecord]) -> list[str]:
    return sorted({r.model for r in runs})


def select_model(runs: Sequence[RunRecord], model: str | None) -> list[RunRecord]:
    """Filter to one model (if given) and require that exactly one model remains."""
    chosen = [r for r in runs if r.model == model] if model else list(runs)
    if model and not chosen:
        raise ValueError(f"no run records for model {model!r}; models present: {models_in(runs)}")
    ms = models_in(chosen)
    if len(ms) > 1:
        raise ValueError(
            f"run files contain several models {ms}; pass --model to pick one "
            "(arms must never be pooled across models)"
        )
    return chosen
