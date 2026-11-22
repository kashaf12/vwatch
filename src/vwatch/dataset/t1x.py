"""T1 extension: generic defects injected into openly licensed Wikimedia Commons speech videos.

    python -m vwatch.dataset.t1x --root eval/data --seed 7

Per source: 2 videos for each of flash_frame / frozen_frame / jump_cut / audio_dropout plus 2 clean.
Idempotent: re-running replaces only the `t1x_` manifest lines (under an flock) and reuses the
download / segment cache in `<root>/_cache/commons/`.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from vwatch.dataset import _commons, _segment, _t1x_verify
from vwatch.dataset._common import item_rng
from vwatch.dataset._segment import CommonsSource, Prepared
from vwatch.dataset._source import SourceClip, SourceSpec, load_source
from vwatch.dataset.inject import T1Job, write_job
from vwatch.schemas import DatasetItem, DefectTruth, DefectType, Task

PREFIX = "t1x_"
GENERIC_TYPES = (DefectType.FLASH_FRAME, DefectType.FROZEN_FRAME, DefectType.JUMP_CUT, DefectType.AUDIO_DROPOUT)
PER_TYPE = 2
N_CLEAN = 2
PILOT_PER_SOURCE = 2  # of 10 videos = 20 %, drawn from two different buckets
MANIFEST = "manifest.jsonl"
LOCK = ".manifest.lock"
SOURCES_MD = "SOURCES.md"

SOURCES: tuple[CommonsSource, ...] = (
    CommonsSource("steel", "File:President Donald J. Trump is Protecting American Steel (short).webm", "English"),
    CommonsSource("boston", "File:President Obama speaks on explosions in Boston (2013-04-15).ogv", "English"),
    CommonsSource("sem", "File:Scanning Electron Microscope (SEM) Lecture 1.webm", "English"),
    CommonsSource("khizr", "File:Khizr Khan Trumps TheDonald.webm", "English"),
    CommonsSource("jharkhand", "File:Unheard Voice of India, Jharkhand, 5.ogv", "Hindi"),
    CommonsSource("gujarat", "File:Unheard Voice of India, 7, Gujarat.ogv", "Hindi"),
    CommonsSource("bnreel", "File:Can I create a Wikipedia article about myself - Bengali explainer reel.webm", "Bengali", 36.0),
    CommonsSource("prague", "File:Novoroční přání pražského primátora Bohuslava Svobody do roku 2013.ogv", "Czech"),
    CommonsSource("linagora", "File:Benjamin (linagora).webm", "French"),
    CommonsSource("pietra", "File:Conheça o segredo do sucesso da historiadora, confeiteira e empresária Pietra Diwan.webm",
                  "Portuguese"),
)


# ------------------------------------------------------------------ jobs and splits


def plan_source_jobs(short: str, si: int, seed: int) -> list[T1Job]:
    jobs: list[T1Job] = []
    for ti, kind in enumerate(GENERIC_TYPES):
        rng = item_rng(seed, 5, si, ti)
        fracs = (rng.permutation(PER_TYPE) + rng.uniform(0.15, 0.85, PER_TYPE)) / PER_TYPE
        for k in range(PER_TYPE):
            jobs.append(T1Job(f"{PREFIX}{short}_{kind.value}_{k}", si, kind, k, float(fracs[k])))
    for k in range(N_CLEAN):
        jobs.append(T1Job(f"{PREFIX}{short}_clean_{k}", si, None, k, 0.0))
    return jobs


def render_seed(seed: int, si: int) -> int:
    """Per-source seed so the injectors' random choices differ between sources."""
    return seed * 1000 + 100 + si


def assign_t1x_splits(jobs: Sequence[T1Job], seed: int, si: int) -> dict[str, str]:
    """~20 % pilot, stratified by source: PILOT_PER_SOURCE videos from distinct buckets."""
    rng = item_rng(seed, 5, si, 9999)
    buckets: dict[str, list[T1Job]] = {}
    for j in jobs:
        buckets.setdefault(j.defect.value if j.defect else "clean", []).append(j)
    names = sorted(buckets)
    pilot: set[str] = set()
    for b in rng.permutation(len(names))[:PILOT_PER_SOURCE]:
        group = buckets[names[int(b)]]
        pilot.add(group[int(rng.integers(len(group)))].video_id)
    return {j.video_id: ("pilot" if j.video_id in pilot else "test") for j in jobs}


# ------------------------------------------------------------------ manifest + docs


def update_manifest(root: Path, items: Sequence[DatasetItem]) -> None:
    """Replace every `t1x_` line with `items` under an exclusive flock; atomic temp + os.replace."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST
    with open(root / LOCK, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            kept = []
            if path.exists():
                kept = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                        if ln.strip() and not DatasetItem.model_validate_json(ln).video_id.startswith(PREFIX)]
            lines = kept + [i.model_dump_json() for i in items]
            tmp = path.with_name(f".{MANIFEST}.t1x.{os.getpid()}.tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def sources_section(prepared: Sequence[Prepared]) -> str:
    rows = []
    for p in prepared:
        i = p.info
        rows.append(
            f"| `{p.source.short}` | {i.title.removeprefix('File:')} | {p.source.language} | {i.author} | {i.license} "
            f"| {i.page_url} | {p.start:.1f}-{p.start + p.duration:.1f} s ({p.duration:.1f} s) | {p.coverage:.0%} |")
    body = ("| source | Commons file | language | author | license | page | segment used | Whisper speech coverage |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    tag = hashlib.sha1(body.encode()).hexdigest()[:10]
    head = (f"\n## T1 extension (t1x): Wikimedia Commons speech videos <!-- t1x:{tag} -->\n\n"
            "Real, openly licensed videos with speech. Each is cut to one 35-60 s segment, normalised to 540 px wide, "
            "30 fps, AAC 44.1 kHz mono, and used to build 10 videos (`t1x_<source>_<type>_<k>`): 2 each of flash_frame, "
            "frozen_frame, jump_cut, audio_dropout, plus 2 clean. Attribution below; modified (cut/re-encoded/defect-injected) "
            "derivatives are redistributed under the same licenses.\n\n")
    return head + body


def append_sources_md(root: Path, prepared: Sequence[Prepared]) -> bool:
    """Append-only (another process may write the file too); skipped if this exact section is present."""
    section = sources_section(prepared)
    tag = section.split("<!-- ")[1].split(" -->")[0]
    path = root / SOURCES_MD
    with open(root / LOCK, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if path.exists() and f"<!-- {tag} -->" in path.read_text(encoding="utf-8"):
                return False
            with open(path, "a", encoding="utf-8") as f:
                f.write(section)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return True


# ------------------------------------------------------------------ build


def _existing_item(root: Path, job: T1Job) -> DatasetItem:
    """Manifest item for an already rendered video (`--verify-only`); fails if the files are missing."""
    video, truth = f"t1/{job.video_id}.mp4", f"t1/{job.video_id}.truth.json"
    if not (root / video).exists() or not (root / truth).exists():
        raise FileNotFoundError(f"{job.video_id}: not rendered yet; run without --verify-only")
    return DatasetItem(video_id=job.video_id, task=Task.DEFECTS, video_path=video, truth_path=truth, split="test",
                       bucket=job.defect.value if job.defect else "clean")


def build_t1x(root: Path, seed: int, sources: Sequence[CommonsSource], client: _commons.CommonsClient,
              transcriber: Callable[[Path], object], *, verify: bool = True, render: bool = True,
              loader: Callable[[SourceSpec], SourceClip] = load_source,
              log: Callable[[str], None] = lambda m: print(m, flush=True)) -> tuple[list[DatasetItem], list[_t1x_verify.Check]]:
    cache = root / "_cache" / "commons"
    prepared = [_segment.prepare_source(client, s, cache, transcriber) for s in sources]  # type: ignore[arg-type]
    for p in prepared:
        log(f"[t1x] {p.source.short}: {p.duration:.1f}s @ {p.start:.1f}s, speech {p.coverage:.0%}, "
            f"{p.language_detected}, {p.info.license}")
    items: list[DatasetItem] = []
    checks: list[_t1x_verify.Check] = []
    for si, p in enumerate(prepared):
        spec = SourceSpec(name=p.source.short, path=p.segment_path)
        src = loader(spec)
        jobs = plan_source_jobs(p.source.short, si, seed)
        splits = assign_t1x_splits(jobs, seed, si)
        small = _t1x_verify.source_small(src) if verify else None
        for job in jobs:
            item = (write_job(root, src, job, render_seed(seed, si)) if render else _existing_item(root, job)
                    ).model_copy(update={"split": splits[job.video_id]})
            items.append(item)
            if verify:
                truth = DefectTruth.model_validate_json((root / item.truth_path).read_text(encoding="utf-8"))
                chk = _t1x_verify.verify_video(root / item.video_path, truth, src, small)
                checks.append(chk)
                log(f"[t1x] {'ok  ' if chk.ok else 'FAIL'} {job.video_id}: {chk.detail}")
            else:
                log(f"[t1x] {job.video_id}")
        del src, small
        gc.collect()
    if all(c.ok for c in checks):
        update_manifest(root, items)
        append_sources_md(root, prepared)
    else:
        log("[t1x] sanity checks failed: manifest and SOURCES.md NOT updated")
    return items, checks


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vwatch.dataset.t1x", description=__doc__)
    p.add_argument("--root", type=Path, default=Path("eval/data"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-verify", action="store_true", help="skip the post-render sanity checks")
    p.add_argument("--verify-only", action="store_true", help="re-check already rendered videos, do not re-render")
    args = p.parse_args(argv)
    from vwatch.transcribe import transcribe

    items, checks = build_t1x(args.root, args.seed, SOURCES, _commons.CurlCommonsClient(), transcribe,
                              verify=not args.no_verify, render=not args.verify_only)
    counts: dict[str, int] = {}
    for it in items:
        counts[f"{it.bucket}/{it.split}"] = counts.get(f"{it.bucket}/{it.split}", 0) + 1
    print(f"built {len(items)} t1x videos: {dict(sorted(counts.items()))}")
    bad = [c for c in checks if not c.ok]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
