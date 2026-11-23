"""Fresh held-out benchmark builder (see docs/fresh-bench-protocol.md; committed before any scoring).

Selects sources mechanically from fixed Commons searches (sorted by page id, first that qualify), builds:
  short: 10 sources x 4 videos (3 injected defects rotating over 4 types + 1 clean), 50 s segments, ids fxs_*
  long:  8 whole 10-30 min videos, each with a 2 s code-word card and one injected defect (or none), ids fxl_*
         (two items per long video: T1 defects and T4 code-word question)
under eval/data/fresh/ (gitignored) with manifest.jsonl, SOURCES.md and SELECTION.md (every rejection + reason).

  uv run python eval/bench/build_fresh.py [--root eval/data/fresh]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vwatch import media, transcribe
from vwatch.dataset import _commons, _segment
from vwatch.dataset._common import item_rng
from vwatch.dataset._segment import CommonsSource, SegmentError
from vwatch.dataset._source import SourceSpec, load_source
from vwatch.dataset.inject import T1Job, write_job
from vwatch.schemas import DatasetItem, DefectTruth, DefectType, QAItem, QATruth, Task, TruthDefect

SEED = 4242
N_SHORT, N_LONG = 10, 8
SHORT_QUERIES = ("interview", "lecture", "tutorial", "press conference", "documentary", "speech",
                 "presentation", "explainer", "news report", "vlog")
LONG_QUERIES = ("lecture", "conference talk", "presentation", "seminar", "keynote", "workshop")
GENERIC = (DefectType.FLASH_FRAME, DefectType.FROZEN_FRAME, DefectType.JUMP_CUT, DefectType.AUDIO_DROPOUT)
LONG_DEFECTS = ("freeze", "freeze", "flash", "flash", "dropout", "dropout", "none", "none")
WORDS = ("pelican", "otter", "lynx", "heron", "badger", "falcon", "walrus", "gecko", "bison", "marmot",
         "osprey", "jackal", "ibex", "tapir", "puffin", "beaver", "coyote", "ferret", "gazelle", "hyena",
         "iguana", "koala", "lemur", "mantis", "narwhal", "ocelot", "panda", "quokka", "raven", "salmon",
         "toucan", "urchin", "vulture", "wombat", "yak", "zebra", "alpaca", "bobcat", "condor", "dingo",
         "egret", "flamingo", "gibbon", "hornet", "impala", "jaguar", "kestrel", "llama", "moose", "newt")
# every source used by an earlier vwatch set, never reused here
USED = {"File:President Donald J. Trump is Protecting American Steel (short).webm",
        "File:President Obama speaks on explosions in Boston (2013-04-15).ogv",
        "File:Scanning Electron Microscope (SEM) Lecture 1.webm", "File:Khizr Khan Trumps TheDonald.webm",
        "File:Unheard Voice of India, Jharkhand, 5.ogv", "File:Unheard Voice of India, 7, Gujarat.ogv",
        "File:Can I create a Wikipedia article about myself - Bengali explainer reel.webm",
        "File:Novoroční přání pražského primátora Bohuslava Svobody do roku 2013.ogv",
        "File:Benjamin (linagora).webm",
        "File:Conheça o segredo do sucesso da historiadora, confeiteira e empresária Pietra Diwan.webm",
        "File:William Fisher, CopyrightX- Lecture 3.1, The Subject Matter of Copyright- Literature (and software).webm",
        "File:COMREL Cooking Demonstration (1015886).webm", "File:Freeride Skateboarding & Downhill Slides.webm",
        "File:Curry Chicken.webmhd.webm"}
API = "https://commons.wikimedia.org/w/api.php"


def search(client: _commons.CurlCommonsClient, query: str, limit: int = 50) -> list[dict]:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"filetype:video {query}", "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|size|mediatype|extmetadata"})
    data = json.loads(client._get(f"{API}?{q}"))
    return list(data.get("query", {}).get("pages", {}).values())


def candidates(client, queries, lo_s: float, hi_s: float, max_bytes: int, log: list[str]) -> list[dict]:
    seen: dict[int, dict] = {}
    for q in queries:
        for page in search(client, q):
            seen.setdefault(int(page["pageid"]), page)
    out = []
    for pid in sorted(seen):
        page = seen[pid]
        title = page["title"]
        try:
            info = _commons.parse_info(page)
        except Exception as exc:  # noqa: BLE001
            log.append(f"| {title} | unreadable info: {exc} |")
            continue
        ii = page.get("imageinfo", [{}])[0]
        dur = float(ii.get("duration") or 0)
        if title in USED:
            log.append(f"| {title} | used in an earlier vwatch set |")
        elif not _commons.license_ok(info.license):
            log.append(f"| {title} | license {info.license} |")
        elif not lo_s <= dur <= hi_s:
            log.append(f"| {title} | duration {dur:.0f} s outside {lo_s:.0f}-{hi_s:.0f} s |")
        elif int(ii.get("size") or 0) > max_bytes:
            log.append(f"| {title} | size {int(ii.get('size') or 0) / 1e6:.0f} MB > {max_bytes / 1e6:.0f} MB |")
        else:
            out.append(page)
    return out


def short_name(title: str, k: int) -> str:
    base = "".join(c for c in title.removeprefix("File:").lower() if c.isalnum())[:10] or "src"
    return f"{base}{k}"


# ------------------------------------------------------------------ short set


def build_short(root: Path, client, log: list[str]) -> tuple[list[DatasetItem], list[str]]:
    cache = root / "_cache"
    items: list[DatasetItem] = []
    sources_md: list[str] = []
    picked = 0
    for page in candidates(client, SHORT_QUERIES, 60, 1800, 300_000_000, log):
        if picked == N_SHORT:
            break
        src = CommonsSource(short_name(page["title"], picked), page["title"], "auto")
        try:
            prep = _segment.prepare_source(client, src, cache, lambda p: transcribe.transcribe(p))
        except (SegmentError, _commons.CommonsError, media.MediaError, ValueError) as exc:
            log.append(f"| {page['title']} | short: {str(exc)[:160]} |")
            continue
        spec = SourceSpec(name=src.short, path=prep.segment_path)
        clip = load_source(spec)
        types = [GENERIC[(picked + k) % 4] for k in range(3)]
        rng = item_rng(SEED, 11, picked, 0)
        jobs = [T1Job(f"fxs_{src.short}_{t.value}", picked, t, 0, float(rng.uniform(0.15, 0.85))) for t in types]
        jobs.append(T1Job(f"fxs_{src.short}_clean", picked, None, 0, 0.0))
        for job in jobs:
            items.append(write_job(root, clip, job, SEED + picked))
        i = prep.info
        sources_md.append(f"| short | `{src.short}` | {i.title.removeprefix('File:')} | {i.author} | {i.license} | "
                          f"{i.page_url} | {prep.start:.1f}-{prep.start + prep.duration:.1f} s | {prep.coverage:.0%} |")
        picked += 1
        del clip
    if picked < N_SHORT:
        raise SystemExit(f"only {picked} short sources qualified; widen the queries in the protocol first")
    return items, sources_md


# ------------------------------------------------------------------ long set


def card_png(word: str, w: int, h: int, out: Path) -> Path:
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", h // 16)
    except OSError:
        font = ImageFont.load_default()
    text = f"CODE WORD: {word.upper()}"
    tw = d.textlength(text, font=font)
    x0, y0 = (w - tw) / 2 - 30, h * 0.76
    d.rectangle([x0, y0, x0 + tw + 60, y0 + h / 16 + 40], fill=(255, 255, 255, 240))
    d.text((x0 + 30, y0 + 20), text, fill=(200, 0, 0, 255), font=font)
    im.save(out)
    return out


def speech_time(raw: Path, lo: float, hi: float, rng: np.random.Generator, need_s: float = 1.2) -> float:
    """A random time in [lo, hi] where the energy gate shows continuous sound for need_s seconds."""
    audio = media.extract_audio(raw, sr=_segment.ENERGY_SR)
    mask = _segment.rough_speech(audio)  # per 10 ms
    k = int(need_s * 100)
    run = np.convolve(mask.astype(int), np.ones(k, int), mode="valid") == k
    ok = [i / 100 for i in np.flatnonzero(run) if lo <= i / 100 <= hi]
    if not ok:
        raise ValueError("no continuous speech for a dropout")
    return float(ok[int(rng.integers(0, len(ok)))])


def render_long(raw: Path, out: Path, card: Path, t_card: float, kind: str, t_def: float) -> tuple[float, float]:
    """720p30 re-encode with the card overlay and one defect; returns the defect's (start, end) in output time."""
    vf = "[0:v]scale=1280:720,fps=30,setsar=1[v0];[v0][1:v]overlay=0:0:enable='between(t,{a},{b})'[v1]".format(
        a=t_card, b=t_card + 2.0)
    af = "[0:a]aresample=44100[a]"
    span = (t_def, t_def)
    if kind == "freeze":
        hold = 1.2
        vf += (f";[v1]trim=0:{t_def},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={hold}[p];"
               f"[v1]trim={t_def + hold},setpts=PTS-STARTPTS[q];[p][q]concat=n=2:v=1[v]")
        span = (t_def, t_def + hold)
    elif kind == "flash":
        n0 = int(round(t_def * 30))
        vf += f";[v1]drawbox=enable='between(n,{n0},{n0 + 1})':x=0:y=0:w=iw:h=ih:color=white:t=fill[v]"
        span = (n0 / 30, (n0 + 2) / 30)
    else:
        vf += ";[v1]null[v]"
    if kind == "dropout":
        af = f"[0:a]aresample=44100,volume=enable='between(t,{t_def},{t_def + 0.8})':volume=0[a]"
        span = (t_def, t_def + 0.8)
    subprocess.run([media.ffmpeg(), "-v", "error", "-y", "-i", str(raw), "-i", str(card), "-filter_complex",
                    f"{vf};{af}", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-shortest", str(out)], check=True)
    return span


def build_long(root: Path, client, log: list[str]) -> tuple[list[DatasetItem], list[str]]:
    cache = root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    items: list[DatasetItem] = []
    sources_md: list[str] = []
    picked = 0
    for page in candidates(client, LONG_QUERIES, 600, 1800, 500_000_000, log):
        if picked == N_LONG:
            break
        info = _commons.parse_info(page)
        raw = cache / f"long{picked}{Path(info.url).suffix.lower()}"
        try:
            if not raw.exists():
                client.download(info.url, raw)
            probe = media.probe(raw)
            if not probe.has_audio:
                raise ValueError("no audio track")
            speech = float(_segment.rough_speech(media.extract_audio(raw, sr=_segment.ENERGY_SR)).mean())
            if speech < 0.5:
                raise ValueError(f"sound on only {speech:.0%} of the file")
        except (ValueError, media.MediaError, _commons.CommonsError) as exc:
            log.append(f"| {page['title']} | long: {str(exc)[:160]} |")
            raw.unlink(missing_ok=True)
            continue
        rng = item_rng(SEED, 13, picked, 0)
        dur = probe.duration
        word = WORDS[int(rng.integers(0, len(WORDS)))]
        t_card = round(float(rng.uniform(0.2, 0.8)) * dur, 2)
        kind = LONG_DEFECTS[picked]
        t_def = 0.0
        for _ in range(50):
            t_def = round(float(rng.uniform(0.15, 0.85)) * dur, 2)
            if abs(t_def - t_card) > 30:
                break
        if kind == "dropout":
            t_def = round(speech_time(raw, max(0.15 * dur, t_card + 30 if t_card < 0.5 * dur else 0.15 * dur),
                                      0.85 * dur, rng), 2)
        vid = f"fxl_{short_name(page['title'], picked)}"
        out = root / "t1" / f"{vid}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        span = render_long(raw, out, card_png(word, 1280, 720, cache / f"{vid}.card.png"), t_card, kind, t_def)
        out_dur = media.probe(out).duration
        defects = () if kind == "none" else (TruthDefect(
            type={"freeze": DefectType.FROZEN_FRAME, "flash": DefectType.FLASH_FRAME,
                  "dropout": DefectType.AUDIO_DROPOUT}[kind], start=span[0], end=span[1], detail=f"injected {kind}"),)
        dt = DefectTruth(video_id=vid, source=vid, duration=out_dur, defects=defects)
        qt = QATruth(video_id=vid, duration=out_dur, items=(QAItem(
            id="q1", question="A code word is shown on screen for about two seconds at some point. What is the code word?",
            answer=word, answer_type="text", time=t_card),))
        for task, truth, bucket in ((Task.DEFECTS, dt, kind if kind != "none" else "clean"), (Task.QA, qt, "needle")):
            rel = f"t1/{vid}.{task.value}.truth.json"
            (root / rel).write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
            items.append(DatasetItem(video_id=vid, task=task, video_path=f"t1/{vid}.mp4", truth_path=rel,
                                     split="test", bucket=bucket))
        sources_md.append(f"| long | `{vid}` | {info.title.removeprefix('File:')} | {info.author} | {info.license} | "
                          f"{info.page_url} | whole file, {out_dur:.0f} s | card {t_card:.1f} s, {kind} {span[0]:.2f} s |")
        picked += 1
    if picked < N_LONG:
        raise SystemExit(f"only {picked} long sources qualified; widen the queries in the protocol first")
    return items, sources_md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "eval" / "data" / "fresh")
    ap.add_argument("--only", choices=("short", "long"), default=None)
    args = ap.parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    client = _commons.CurlCommonsClient()
    log: list[str] = []
    items: list[DatasetItem] = []
    rows: list[str] = []
    if args.only in (None, "short"):
        a, b = build_short(root, client, log)
        items += a
        rows += b
    if args.only in (None, "long"):
        a, b = build_long(root, client, log)
        items += a
        rows += b
    (root / "manifest.jsonl").write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    (root / "SOURCES.md").write_text("| set | id | file | author | license | page | used | detail |\n"
                                     "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (root / "SELECTION.md").write_text("| candidate | rejected because |\n|---|---|\n" + "\n".join(log) + "\n",
                                       encoding="utf-8")
    print(f"{len(items)} items, {len(rows)} sources, {len(log)} rejections -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
