"""Leakage audit for the external-agent arms.

Every tool call the agent made (taken from the stream-json `tool_use` blocks, including calls the
permission system denied) is checked for references to anything outside its temp workdir: absolute
paths elsewhere, `~`, `..` escapes, and markers of the ground truth (`truth`, `eval/data`, `manifest`,
the repo path). Any hit turns the run into an error, which scores as a failure.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_MARKERS: tuple[str, ...] = ("truth", "eval/data", "manifest")
ALLOWED_SYSTEM_PATHS: tuple[str, ...] = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/zero")
_SKIP_KEYS = frozenset({"description"})  # free-text explanation fields, not paths
_ABS_PATH = re.compile(r"""(?:^|(?<=[\s'"=:,(<>|]))(/[A-Za-z0-9._~\-][^\s'"|;&<>(),]*)""")
_FS_ROOT = re.compile(r"""(?:^|(?<=[\s'"=]))/(?=$|[\s'"|;&<>)])""")  # a bare "/" (e.g. grep -r x /)
_HOME_PATH = re.compile(r"""(?:^|(?<=[\s'"=:,(<>|]))(~[^\s'"|;&<>(),]*)""")
_PARENT = re.compile(r"""(?:^|(?<=[\s'"=:,(<>|]))(\.\.(?:/[^\s'"|;&<>(),]*)?)(?=$|[\s'"|;&<>(),])""")
_REL_WITH_PARENT = re.compile(r"""(?:^|(?<=[\s'"=:,(<>|]))([A-Za-z0-9._\-][^\s'"|;&<>(),]*/\.\.(?:/[^\s'"|;&<>(),]*)?)""")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict

    @property
    def summary(self) -> str:
        if self.name == "Bash":
            text = str(self.input.get("command", ""))
        elif self.name == "Read":
            text = str(self.input.get("file_path", ""))
        else:
            text = json.dumps(self.input, ensure_ascii=False, sort_keys=True)
        return text if len(text) <= 240 else text[:237] + "..."

    def to_log(self) -> dict:
        return {"name": self.name, "input": self.summary}


def tool_calls(events: list[dict]) -> tuple[ToolCall, ...]:
    """All tool_use blocks from assistant events, in order, de-duplicated by id."""
    seen: set[str] = set()
    calls: list[ToolCall] = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            bid = str(block.get("id", ""))
            if bid and bid in seen:
                continue
            seen.add(bid)
            inp = block.get("input")
            calls.append(ToolCall(id=bid, name=str(block.get("name", "")),
                                  input=inp if isinstance(inp, dict) else {"value": inp}))
    return tuple(calls)


def _strings(value: object, key: str = "") -> list[str]:
    if key in _SKIP_KEYS:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(v, str(k))]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def _keyed_strings(inp: dict) -> list[tuple[str, str]]:
    return [(str(k), s) for k, v in inp.items() for s in _strings(v, str(k))]


def _roots(workdir: Path) -> tuple[str, ...]:
    raw = str(workdir)
    real = os.path.realpath(raw)
    variants = {raw, real}
    for p in (raw, real):  # macOS: /var/... and /private/var/... are the same place
        if p.startswith("/private/"):
            variants.add(p[len("/private"):])
        else:
            variants.add("/private" + p)
    return tuple(sorted(variants))


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    norm = os.path.normpath(path)
    if norm in ALLOWED_SYSTEM_PATHS:
        return True
    return any(norm == r or norm.startswith(r.rstrip("/") + "/") for r in roots)


def _path_problems(text: str, workdir: Path, roots: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for m in _ABS_PATH.finditer(text):
        p = m.group(1).rstrip(".:")
        if not _inside(p, roots):
            found.append(f"path outside workdir: {p}")
    if _FS_ROOT.search(text):
        found.append("path outside workdir: /")
    for m in _HOME_PATH.finditer(text):
        found.append(f"home path: {m.group(1)}")
    for rx in (_PARENT, _REL_WITH_PARENT):
        for m in rx.finditer(text):
            if not _inside(os.path.join(str(workdir), m.group(1)), roots):
                found.append(f"relative path escapes workdir: {m.group(1)}")
    return found


_SHELL_SEGMENTS = re.compile(r"&&|\|\||;|\n")


def _bash_problems(command: str, cwd: Path, roots: tuple[str, ...]) -> tuple[list[str], Path]:
    """Check each `&&`/`;` segment against the shell cwd at that point; follow `cd` (it persists
    between Claude Code Bash calls). Returns the problems and the cwd after the command."""
    found: list[str] = []
    for seg in _SHELL_SEGMENTS.split(command):
        found += _path_problems(seg, cwd, roots)
        try:
            argv = shlex.split(seg)
        except ValueError:
            continue
        if len(argv) == 2 and argv[0] == "cd":
            target = Path(os.path.normpath(os.path.join(str(cwd), argv[1])))
            cwd = target if _inside(str(target), roots) else cwd  # escapes were flagged above
    return found, cwd


def audit(calls: tuple[ToolCall, ...], workdir: Path, *, repo_root: Path | None = None,
          markers: tuple[str, ...] = FORBIDDEN_MARKERS) -> tuple[str, ...]:
    """Return one human-readable problem per offending tool call (empty = clean)."""
    roots = _roots(workdir)
    repo = str(repo_root.resolve()) if repo_root else None
    problems: list[str] = []
    cwd = workdir
    for call in calls:
        issues: list[str] = []
        for key, s in _keyed_strings(call.input):
            low = s.lower()
            issues += [f"mentions '{mk}'" for mk in markers if mk in low]
            if repo and repo in s:
                issues.append("mentions the repo path")
            if call.name == "Bash" and key == "command":
                found, cwd = _bash_problems(s, cwd, roots)
                issues += found
            else:
                issues += _path_problems(s, workdir, roots)
        if issues:
            uniq = list(dict.fromkeys(issues))
            problems.append(f"{call.name}({call.summary[:120]}): {'; '.join(uniq[:3])}")
    return tuple(problems)
