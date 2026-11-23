"""Add a public video benchmark subset (TVBench, task T5) to the dataset.

    python -m vwatch.dataset.tvbench --root eval/data --per-category 10 --seed 7

TVBench (Cores et al., BMVC 2025; https://huggingface.co/datasets/FunAILab/TVBench, CC-BY-4.0).
Per category: download the json + the video zip into `<root>/_cache/tvbench/`, sample clips (seeded),
cut each to its permitted interval, re-encode to mp4, write an `MCQTruth`, delete the zip. The manifest's
T5 lines are replaced (idempotent) under an exclusive flock so a concurrent builder can append safely.
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import random
import re
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from vwatch import media
from vwatch.schemas import DatasetItem, MCQItem, MCQTruth, Task

REPO = "FunAILab/TVBench"
HF = "https://huggingface.co"
RESOLVE = f"{HF}/datasets/{REPO}/resolve/main"
TREE = f"{HF}/api/datasets/{REPO}/tree/main"
LICENSE = "CC-BY-4.0"
CITATION = ("Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees G. M. Snoek, Yuki M. Asano. "
            "Lost in Time: A New Temporal Benchmark for VideoLLMs. BMVC 2025. https://arxiv.org/abs/2410.07752")

# Temporal-understanding categories with small zips (egocentric_sequence is split in many zips, action_count
# 4.3 GB, object_shuffle 3.2 GB and action_antonym needs NTU RGB+D: all skipped).
CATEGORIES = ("object_count", "moving_direction", "scene_transition",
              "action_sequence", "unexpected_action", "action_localization")
TRIMMED = frozenset({"action_sequence", "action_localization"})  # json start/end = the permitted interval
PILOT_FRACTION = 0.4
MAX_DOWNLOAD_BYTES = int(3.2e9)
MANIFEST = "manifest.jsonl"
LOCK = ".manifest.lock"
SOURCES = "SOURCES.md"
_MARK = ("<!-- tvbench:start -->", "<!-- tvbench:end -->")


class Fetcher(Protocol):
    def size(self, path: str) -> int: ...
    def text(self, path: str) -> str: ...
    def file(self, path: str, dest: Path) -> None: ...
    def range_zip(self, path: str) -> zipfile.ZipFile: ...


class RangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file via a `read_range(start, end_exclusive) -> bytes` callable,
    so `zipfile` can pull just the sampled members out of a multi-GB archive."""

    BLOCK = 4 << 20

    def __init__(self, size: int, read_range: Callable[[int, int], bytes]):
        self._size, self._read, self._pos = size, read_range, 0
        self._blk_start, self._blk = 0, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self._size - self._pos))
        if n == 0:
            return 0
        if not (self._blk_start <= self._pos and self._pos + n <= self._blk_start + len(self._blk)):
            self._blk_start = self._pos
            self._blk = self._read(self._pos, min(self._size, self._pos + max(n, self.BLOCK)))
        off = self._pos - self._blk_start
        b[:n] = self._blk[off:off + n]
        self._pos += n
        return n


def _ssl_context() -> ssl.SSLContext:
    """Default verification; fall back to the system bundle when python ships without CA certificates."""
    paths = ssl.get_default_verify_paths()
    if not (paths.cafile and os.path.exists(paths.cafile)) and os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    return ssl.create_default_context()


class HttpFetcher:
    """Plain HTTPS against the public dataset repo (no login); downloads resume from `<dest>.part`."""

    def _open(self, url: str, headers: dict | None = None):
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vwatch-eval", **(headers or {})}),
                                      timeout=60, context=_ssl_context())

    def size(self, path: str) -> int:
        folder, _, name = path.rpartition("/")
        with self._open(f"{TREE}/{folder}") as r:
            for f in json.load(r):
                if f["path"] == path:
                    return int(f["size"])
        raise FileNotFoundError(path)

    def text(self, path: str) -> str:
        with self._open(f"{RESOLVE}/{path}") as r:
            return r.read().decode("utf-8")

    def range_zip(self, path: str) -> zipfile.ZipFile:
        size = self.size(path)

        def read_range(start: int, end: int) -> bytes:
            for attempt in range(4):
                try:
                    with self._open(f"{RESOLVE}/{path}", {"Range": f"bytes={start}-{end - 1}"}) as r:
                        data = r.read()
                    if len(data) == end - start:
                        return data
                except OSError:
                    if attempt == 3:
                        raise
            raise OSError(f"short read of {path} bytes {start}-{end}")

        return zipfile.ZipFile(RangeFile(size, read_range))

    def file(self, path: str, dest: Path) -> None:
        want = self.size(path)
        if dest.exists() and dest.stat().st_size == want:
            return
        part = dest.with_suffix(dest.suffix + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            have = part.stat().st_size if part.exists() else 0
            if have == want:
                break
            try:
                with self._open(f"{RESOLVE}/{path}", {"Range": f"bytes={have}-"} if have else None) as r, \
                        part.open("ab" if have else "wb") as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
            except OSError:
                continue
        if part.stat().st_size != want:
            raise OSError(f"incomplete download of {path}: {part.stat().st_size}/{want} bytes")
        os.replace(part, dest)


# ---------------------------------------------------------------- question mapping


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()


def answer_index(candidates: Sequence[str], answer: object) -> int:
    """Index of the correct option. TVBench gives the answer as option text; a letter or an int index is
    accepted too. Raises unless it identifies exactly one option."""
    cands = [str(c) for c in candidates]
    hits = [i for i, c in enumerate(cands) if _norm(c) == _norm(answer)]
    if not hits and isinstance(answer, str) and re.fullmatch(r"\(?[A-Za-z]\)?\.?", answer.strip()):
        letter = ord(answer.strip().strip("().").upper()) - ord("A")
        hits = [letter] if 0 <= letter < len(cands) else []
    if not hits and isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(cands):
        hits = [answer]
    if len(hits) != 1:
        raise ValueError(f"answer {answer!r} matches {len(hits)} of the options {cands}")
    if len({_norm(c) for c in cands}) != len(cands):
        raise ValueError(f"duplicate options in {cands}")
    return hits[0]


@dataclass(frozen=True)
class Clip:
    category: str
    video: str  # file name inside the zip
    start: float | None
    end: float | None
    question: str
    options: tuple[str, ...]
    answer_index: int


def clip_from_row(category: str, row: dict) -> Clip:
    start, end = (row.get("start"), row.get("end")) if category in TRIMMED else (None, None)
    if (start is None) != (end is None) or (start is not None and not end > start):
        raise ValueError(f"{category}: bad interval {start}..{end} for {row.get('video')}")
    cands = tuple(str(c) for c in row["candidates"])
    return Clip(category, row["video"], start, end, row["question"].strip(), cands, answer_index(cands, row["answer"]))


def sample_clips(category: str, rows: Sequence[dict], n: int, seed: int) -> list[Clip]:
    """Seeded sample of n distinct clips (video + interval), one question each; order = sampling order."""
    rng = random.Random(f"{seed}:{category}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    out: list[Clip] = []
    seen: set[tuple] = set()
    for i in order:
        clip = clip_from_row(category, rows[i])
        key = (clip.video, clip.start, clip.end)
        if key not in seen:
            seen.add(key)
            out.append(clip)
        if len(out) == n:
            return out
    raise ValueError(f"{category}: only {len(out)} distinct clips, wanted {n}")


def split_of(index: int, n: int) -> str:
    return "pilot" if index < round(PILOT_FRACTION * n) else "test"


def truth_for(video_id: str, clip: Clip, duration: float) -> MCQTruth:
    item = MCQItem(id="q1", question=clip.question, options=clip.options,
                   answer_index=clip.answer_index, category=clip.category)
    return MCQTruth(video_id=video_id, duration=duration, source=f"TVBench/{clip.category}", items=(item,))


# ---------------------------------------------------------------- video conversion


def probe_duration(path: Path) -> float:
    out = subprocess.run([media.ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


# long side <= 1280, short side <= 720 (720p class), never upscaled, even dimensions
_SCALE = ("scale=w='min(iw,if(gt(iw,ih),1280,720))':h='min(ih,if(gt(iw,ih),720,1280))'"
          ":force_original_aspect_ratio=decrease:force_divisible_by=2")


def convert(src: Path, dst: Path, start: float | None = None, end: float | None = None) -> None:
    cmd = [media.ffmpeg(), "-y", "-v", "error"]
    if start is not None and end is not None:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}"]
    cmd += ["-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-vf", _SCALE, "-c:v", "libx264", "-crf", "23",
            "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_member(zf: zipfile.ZipFile, name: str, dest_dir: Path) -> Path:
    """Copy one archive member to a path we choose (never trust member paths from the archive)."""
    matches = [m for m in zf.namelist() if m.rsplit("/", 1)[-1] == name and not m.endswith("/")]
    if not matches:
        raise FileNotFoundError(f"{name} not in archive")
    dest = dest_dir / f"src_{abs(hash(name))}{Path(name).suffix}"
    with zf.open(matches[0]) as r, dest.open("wb") as w:
        while chunk := r.read(1 << 20):
            w.write(chunk)
    return dest


# ---------------------------------------------------------------- manifest / sources (locked, atomic)


@contextmanager
def manifest_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / LOCK).open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def replace_t5_in_manifest(root: Path, items: Sequence[DatasetItem]) -> None:
    """Drop every previous T5 line, keep all other lines byte-for-byte, append `items`."""
    path = root / MANIFEST
    with manifest_lock(root):
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        keep = [ln for ln in lines if ln.strip() and json.loads(ln).get("task") != Task.MCQ.value]
        keep += [i.model_dump_json() for i in items]
        _atomic_write(path, "".join(ln + "\n" for ln in keep))


def sources_block(counts: dict[str, int]) -> str:
    cats = ", ".join(f"`{c}` ({n})" for c, n in counts.items())
    return "\n".join([
        _MARK[0],
        "## T5: TVBench subset",
        "",
        "- Dataset: TVBench, https://huggingface.co/datasets/FunAILab/TVBench",
        f"- License: {LICENSE} (annotations/videos; the repo's `license_notice.txt` adds that part of the video "
        "content comes from Apache-2.0 source datasets: Perception Test, CLEVRER, STAR and others listed in its README).",
        f"- Attribution: {CITATION}",
        f"- Categories used (clips): {cats}; {sum(counts.values())} clips in total.",
        "- Changes: clips re-encoded to H.264/AAC mp4 (long side <= 1280, short side <= 720, audio kept); "
        "action_sequence and action_localization cut to the benchmark's permitted `start`..`end` interval; "
        "one question per clip; options keep their original order.",
        "- Reproduce: `python -m vwatch.dataset.tvbench --root eval/data --per-category 10 --seed 7`",
        _MARK[1],
        "",
    ])


def write_sources(root: Path, counts: dict[str, int]) -> None:
    path = root / SOURCES
    with manifest_lock(root):
        text = path.read_text(encoding="utf-8") if path.exists() else "# Dataset sources\n\n"
        block = sources_block(counts)
        pat = re.compile(re.escape(_MARK[0]) + r".*?" + re.escape(_MARK[1]) + r"\n?", re.S)
        text = pat.sub("", text).rstrip("\n") + "\n\n" + block
        _atomic_write(path, text)


# ---------------------------------------------------------------- build


def build_tvbench(root: Path, *, per_category: int, seed: int, categories: Sequence[str] = CATEGORIES,
                  fetcher: Fetcher | None = None, keep_cache: bool = False, full_download: bool = False,
                  log: Callable[[str], None] = print) -> list[DatasetItem]:
    if per_category < 1:
        raise ValueError("per_category must be >= 1")
    root = Path(root)
    fetcher = fetcher or HttpFetcher()
    cache = root / "_cache" / "tvbench"
    cache.mkdir(parents=True, exist_ok=True)
    sizes = {c: fetcher.size(f"video/{c}.zip") for c in categories}
    total = sum(sizes.values())
    if full_download and total > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"zips total {total / 1e9:.2f} GB > cap {MAX_DOWNLOAD_BYTES / 1e9:.2f} GB")
    log(f"archives: {total / 1e9:.2f} GB for {len(categories)} categories "
        f"({'full download' if full_download else 'only the sampled members are fetched (HTTP range), unless already cached'})")
    t5dir = root / "t5"
    for stale in list(t5dir.glob("t5_*")) if t5dir.exists() else []:
        stale.unlink()
    items: list[DatasetItem] = []
    counts: dict[str, int] = {}
    for cat in categories:
        rows = json.loads(fetcher.text(f"json/{cat}.json"))
        clips = sample_clips(cat, rows, per_category, seed)
        zpath = cache / f"{cat}.zip"
        cached = zpath.exists() and zpath.stat().st_size == sizes[cat]
        if full_download and not cached:
            fetcher.file(f"video/{cat}.zip", zpath)
            cached = True
        with (zipfile.ZipFile(zpath) if cached else fetcher.range_zip(f"video/{cat}.zip")) as zf, \
                tempfile.TemporaryDirectory(dir=cache) as tmp:
            for n, clip in enumerate(clips):
                vid = f"t5_{cat}_{n:03d}"
                src = _extract_member(zf, clip.video, Path(tmp))
                dst = t5dir / f"{vid}.mp4"
                convert(src, dst, clip.start, clip.end)
                src.unlink()
                truth = truth_for(vid, clip, probe_duration(dst))
                (t5dir / f"{vid}.truth.json").write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
                items.append(DatasetItem(video_id=vid, task=Task.MCQ, video_path=f"t5/{vid}.mp4",
                                         truth_path=f"t5/{vid}.truth.json", split=split_of(n, len(clips)),
                                         bucket=cat))
        counts[cat] = len(clips)
        if not keep_cache:
            zpath.unlink(missing_ok=True)
        log(f"{cat}: {len(clips)} clips")
    replace_t5_in_manifest(root, items)
    write_sources(root, counts)
    return items


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vwatch.dataset.tvbench", description=__doc__)
    p.add_argument("--root", type=Path, default=Path("eval/data"))
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--categories", nargs="*", choices=CATEGORIES, default=list(CATEGORIES))
    p.add_argument("--keep-cache", action="store_true", help="keep downloaded zips (they are deleted by default)")
    p.add_argument("--full-download", action="store_true",
                   help="download whole zips (<= 3.2 GB total) instead of range-reading the sampled members")
    args = p.parse_args(argv)
    try:
        items = build_tvbench(args.root, per_category=args.per_category, seed=args.seed,
                              categories=args.categories, keep_cache=args.keep_cache,
                              full_download=args.full_download)
    except (ValueError, OSError, ValidationError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    counts: dict[str, int] = {}
    for it in items:
        counts[f"{it.bucket}/{it.split}"] = counts.get(f"{it.bucket}/{it.split}", 0) + 1
    print(f"wrote {len(items)} T5 items to {args.root / MANIFEST}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
