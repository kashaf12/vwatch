"""Minimal Wikimedia Commons client (MediaWiki API + file download) for the T1x sources.

HTTP goes through `curl` (honours the system trust store; polite User-Agent, no personal data).
Everything the pipeline needs from the network sits behind the `CommonsClient` protocol so tests
can swap in a fake.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

USER_AGENT = "vwatch-dataset/0.1 (research eval)"
API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_LICENSES = ("cc by", "cc0", "public domain", "pd")  # prefix match, case-insensitive; CC BY covers BY-SA


class CommonsError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileInfo:
    title: str  # "File:Foo.webm"
    url: str  # direct media URL
    page_url: str  # human-readable Commons page
    author: str
    license: str
    size: int
    duration: float
    width: int
    height: int

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> FileInfo:
        return cls(**d)


class CommonsClient(Protocol):
    def info(self, title: str) -> FileInfo: ...

    def download(self, url: str, dest: Path) -> None: ...


def license_ok(name: str) -> bool:
    n = name.strip().lower()
    if "-nc" in n or "-nd" in n:  # we modify and redistribute the clips: no NonCommercial / NoDerivatives
        return False
    return any(n.startswith(p) for p in ALLOWED_LICENSES)


def clean_text(raw: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", raw or ""))).strip()


def parse_info(page: dict) -> FileInfo:
    """Turn one `query` page (prop=imageinfo, extmetadata) into a validated FileInfo."""
    title = page.get("title", "?")
    if not page.get("imageinfo"):
        raise CommonsError(f"{title}: no imageinfo (file missing?)")
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    lic = clean_text(meta.get("LicenseShortName", {}).get("value", ""))
    if not license_ok(lic):
        raise CommonsError(f"{title}: license {lic!r} is not CC-BY/CC-BY-SA/CC0/public domain")
    author = clean_text(meta.get("Artist", {}).get("value", "")) or "unknown"
    url = ii["url"].split("?")[0]
    return FileInfo(
        title=title, url=url, page_url=ii.get("descriptionurl", ""), author=author, license=lic,
        size=int(ii.get("size", 0)), duration=float(ii.get("duration") or 0.0),
        width=int(ii.get("width", 0)), height=int(ii.get("height", 0)),
    )


class CurlCommonsClient:
    def __init__(self, retries: int = 5, pause: float = 1.0) -> None:
        self.retries = retries
        self.pause = pause

    def _get(self, url: str) -> bytes:
        out = b""
        for k in range(self.retries):
            time.sleep(self.pause)
            out = subprocess.run(["curl", "-sL", "--max-time", "90", "-A", USER_AGENT, url],
                                 capture_output=True).stdout
            if out:
                return out
            time.sleep(3 * (k + 1))  # empty body = throttled / transient
        raise CommonsError(f"no response from {url}")

    def info(self, title: str) -> FileInfo:
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": title, "prop": "imageinfo",
            "iiprop": "url|size|mediatype|extmetadata"})
        data = json.loads(self._get(f"{API}?{q}"))
        pages = list(data.get("query", {}).get("pages", {}).values())
        if not pages:
            raise CommonsError(f"{title}: not found")
        return parse_info(pages[0])

    def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        for k in range(self.retries):
            time.sleep(self.pause)
            r = subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--max-time", "900", "-A", USER_AGENT, "-o", str(part), url],
                               capture_output=True)
            if part.exists() and part.stat().st_size > 0 and r.returncode in (0, 33):  # 33: already complete
                part.replace(dest)
                return
            time.sleep(5 * (k + 1))
        part.unlink(missing_ok=True)
        raise CommonsError(f"download failed: {url}")
