"""External-tool arms: Claude Code as people actually use it to watch video.

CC   plain headless Claude Code (its normal system prompt) with Bash (read-only media commands) + Read.
CVV  CC + the claude-video-vision plugin (jordanrendric/claude-video-vision), loaded with --plugin-dir.
POP  CC + the popcorn MCP server (haithamelmengad/popcorn), attached with --mcp-config.
VWP  CC + the vwatch plugin from this repo (plugins/vwatch: MCP tools + watch-video skill), run from the checkout.

Each run: a fresh temp dir outside the repo holding only `video.mp4` (neutral name: dataset names
encode the defect), `claude -p` with cwd there, no user settings / CLAUDE.md / hooks / plugins / MCP
of the user, permission mode dontAsk with an allowlist, a hard timeout that kills the process tree,
an audit of every tool call, then the temp dir is deleted.

These arms are NOT budget-capped: `budget` is ignored and content_tokens_est is 0, because that is
how these tools are used in practice. Compare them on accuracy and on billed cost / wall time.
Third-party code is pinned (PINS) and built by scripts/setup_external.sh into vendor/ (gitignored).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from vwatch.arms.common import ArmResult, VideoContext
from vwatch.arms.external_audit import audit, tool_calls
from vwatch.arms.external_obs import infer_observations
from vwatch.arms.external_proc import AgentProcess, AgentTimeout
from vwatch.llm.base import DEFAULT_MODEL, LLMError
from vwatch.llm.claude_cli import find_claude_binary, parse_usage
from vwatch.schemas import Arm, Task, Usage
from vwatch.tasks import REPAIR_PROMPT, parse_report

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ARMS = (Arm.CC, Arm.CVV, Arm.POP, Arm.VWP)
DEFAULT_TIMEOUT_S = 900.0
PINS = {
    "claude-video-vision": "4a4f99016f249a24018cd19b66531ebbcd58024b",  # v1.4.0
    "popcorn": "e6a2b2887c235f5157bb47fdbd3ca8ae9f112cc1",  # 0.4.0
    "whisper.cpp": "5eaee0cc9024eb9e9a99469199920c8b7f5cb9ec",  # whisper-cli for the CVV local backend
}
# Shell commands the agent may run (what a user would approve for "look at this video"). No network
# tools (curl, wget), no rm/mv/sed/awk/xargs. python3 can do anything; leaks are caught by the audit.
BASH_COMMANDS = ("ffmpeg", "ffprobe", "python3", "python", "ls", "mkdir", "cd", "pwd", "echo", "which", "cat",
                 "head", "tail", "wc", "sort", "uniq", "cut", "tr", "grep", "bc", "file", "stat", "du")
CVV_MCP = "mcp__claude-video-vision"
POP_MCP = "mcp__popcorn"
# Named like the plugin-provided server so the plugin's /vwatch:watch allowed-tools match unchanged.
VWP_SERVER = "plugin_vwatch_vwatch"
VWP_MCP = f"mcp__{VWP_SERVER}"
CVV_WHISPER_MODEL = "small"  # same size as our own transcriber; the wizard's "auto" would pick by RAM
_ENV_KEEP = ("HOME", "USER", "LOGNAME", "SHELL", "LANG", "TERM", "TMPDIR", "PATH", "CLAUDE_CONFIG_DIR")
_SANDBOX_NOTE = ("Sandbox: keep every file you create inside the current directory. Shell commands allowed here: "
                 + ", ".join(BASH_COMMANDS) + " (anything else is blocked); no network. Commands that use shell "
                 "variables, loops or $(...) are blocked too: write each command out literally (&& and pipes are fine).")


@dataclass(frozen=True)
class ExternalConfig:
    model: str = DEFAULT_MODEL
    effort: str | None = "medium"
    timeout_s: float = DEFAULT_TIMEOUT_S
    binary: str | None = None  # default: find_claude_binary()
    vendor_dir: Path = REPO_ROOT / "vendor"
    log_dir: Path | None = REPO_ROOT / "eval" / "runs" / "external"  # None = no sidecar log
    temp_root: Path | None = None  # default $TMPDIR/vwatch-ext
    check_pins: bool = True
    base_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))

    @classmethod
    def from_backend(cls, backend: object) -> ExternalConfig:
        env = os.environ
        return cls(model=getattr(backend, "model", DEFAULT_MODEL), effort=getattr(backend, "effort", "medium"),
                   timeout_s=float(env.get("VWATCH_EXT_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
                   vendor_dir=Path(env.get("VWATCH_VENDOR_DIR", REPO_ROOT / "vendor")),
                   log_dir=Path(env["VWATCH_EXT_LOG_DIR"]) if env.get("VWATCH_EXT_LOG_DIR")
                   else REPO_ROOT / "eval" / "runs" / "external")


@dataclass(frozen=True)
class RunDirs:
    root: Path  # everything for this run; deleted afterwards
    work: Path  # the agent's cwd: only video.mp4 at start
    video: Path


class SetupError(RuntimeError):
    """A third-party tool is not installed/built as pinned. Run scripts/setup_external.sh."""


# ---------------------------------------------------------------- setup


def make_run_dirs(src_video: Path, temp_root: Path | None) -> RunDirs:
    base = temp_root or Path(tempfile.gettempdir()) / "vwatch-ext"
    root = base / uuid.uuid4().hex
    work = root / "work"
    work.mkdir(parents=True)
    (root / "tmp").mkdir()
    video = work / "video.mp4"
    shutil.copyfile(src_video, video)
    return RunDirs(root=root, work=work, video=video)


def agent_env(base: Mapping[str, str], repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Minimal environment for claude: nothing that points into the repo (e.g. uv's VIRTUAL_ENV)."""
    repo = str(repo_root)
    env = {k: v for k, v in base.items()
           if (k in _ENV_KEEP or k.startswith(("LC_", "ANTHROPIC_"))) and repo not in v}
    env["PATH"] = os.pathsep.join(p for p in base.get("PATH", "").split(os.pathsep) if p and repo not in p)
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def _pinned(vendor: Path, name: str, check: bool) -> Path:
    repo = vendor / name
    if not repo.is_dir():
        raise SetupError(f"{repo} missing: run scripts/setup_external.sh")
    if check:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout.strip() != PINS[name]:
            raise SetupError(f"{repo} is at {head.stdout.strip() or '?'} but PINS says {PINS[name]}")
    return repo


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        raise SetupError(f"{what} not found at {path}: run scripts/setup_external.sh")
    return path


def _node(env: Mapping[str, str]) -> str:
    node = shutil.which("node", path=env.get("PATH"))
    if not node:
        raise SetupError("node not found on PATH (needed by the MCP servers)")
    return node


def setup_cvv(cfg: ExternalConfig, dirs: RunDirs, env: Mapping[str, str]) -> list[str]:
    """Per-run copy of the plugin whose MCP server runs our pinned build (not `npx ...@latest`)."""
    repo = _pinned(cfg.vendor_dir, "claude-video-vision", cfg.check_pins)
    server = _need(repo / "mcp-server" / "dist" / "index.js", "claude-video-vision MCP build")
    wcpp = _pinned(cfg.vendor_dir, "whisper.cpp", cfg.check_pins)
    whisper_bin = _need(wcpp / "build" / "bin" / "whisper-cli", "whisper-cli").parent
    plugin = dirs.root / "plugin"
    for part in (".claude-plugin", "commands", "skills", "agents"):
        shutil.copytree(repo / part, plugin / part)
    home = dirs.root / "home"
    conf_dir = home / ".claude-video-vision"
    conf_dir.mkdir(parents=True)
    (conf_dir / "config.json").write_text(json.dumps(
        {"backend": "local", "whisper_engine": "cpp", "whisper_model": CVV_WHISPER_MODEL}))
    models = cfg.vendor_dir / "cvv-models"
    models.mkdir(exist_ok=True)
    (conf_dir / "models").symlink_to(models, target_is_directory=True)
    server_env = {"HOME": str(home), "TMPDIR": str(dirs.root / "tmp"),
                  "PATH": f"{whisper_bin}{os.pathsep}{env.get('PATH', '')}"}
    # --strict-mcp-config (which keeps the user's MCP servers out) also drops plugin-declared servers,
    # so the plugin's own server is attached explicitly under the same name.
    (plugin / ".mcp.json").write_text("{}")
    conf = dirs.root / "mcp.json"
    conf.write_text(json.dumps({"mcpServers": {"claude-video-vision": {
        "command": _node(env), "args": [str(server)], "env": server_env}}}, indent=2))
    return ["--plugin-dir", str(plugin), "--mcp-config", str(conf)]


def setup_pop(cfg: ExternalConfig, dirs: RunDirs, env: Mapping[str, str]) -> list[str]:
    repo = _pinned(cfg.vendor_dir, "popcorn", cfg.check_pins)
    server = _need(repo / "dist" / "index.js", "popcorn build")
    venv_bin = _need(cfg.vendor_dir / "popcorn-venv" / "bin" / "python3", "popcorn transcription venv").parent
    server_env = {"PATH": f"{venv_bin}{os.pathsep}{env.get('PATH', '')}", "TMPDIR": str(dirs.root / "tmp"),
                  "HF_HOME": str(cfg.vendor_dir / "hf-cache")}
    conf = dirs.root / "mcp.json"
    conf.write_text(json.dumps(
        {"mcpServers": {"popcorn": {"command": _node(env), "args": [str(server)], "env": server_env}}}, indent=2))
    return ["--mcp-config", str(conf)]


def setup_vwp(cfg: ExternalConfig, dirs: RunDirs, env: Mapping[str, str], repo_root: Path = REPO_ROOT) -> list[str]:
    """Per-run copy of this repo's plugin; its MCP server runs from the local checkout (not the release tag)."""
    src = _need(repo_root / "plugins" / "vwatch", "vwatch plugin directory")
    plugin = dirs.root / "plugin"
    shutil.copytree(src, plugin)
    (plugin / ".mcp.json").write_text("{}")
    manifest = plugin / ".claude-plugin" / "plugin.json"
    data = json.loads(manifest.read_text())
    data.pop("mcpServers", None)  # attached below; --strict-mcp-config drops plugin-declared servers anyway
    manifest.write_text(json.dumps(data, indent=2))
    uv = shutil.which("uv", path=env.get("PATH"))
    if not uv:
        raise SetupError("uv not found on PATH (needed to run the vwatch MCP server from the checkout)")
    server_env = {"TMPDIR": str(dirs.root / "tmp")}
    conf = dirs.root / "mcp.json"
    conf.write_text(json.dumps({"mcpServers": {VWP_SERVER: {
        "command": uv, "args": ["run", "--project", str(repo_root), "--extra", "mcp", "vwatch-mcp"],
        "env": server_env}}}, indent=2))
    return ["--plugin-dir", str(plugin), "--mcp-config", str(conf)]


# ---------------------------------------------------------------- command and prompt


def allowed_tools(arm: Arm) -> tuple[str, ...]:
    bash = tuple(r for c in BASH_COMMANDS for r in (f"Bash({c})", f"Bash({c} *)"))
    base = ("Read(./**)",) + bash
    if arm is Arm.CVV:
        return base + ("Skill", CVV_MCP)
    if arm is Arm.POP:
        return base + (POP_MCP,)
    if arm is Arm.VWP:
        return base + ("Skill", VWP_MCP)
    return base


def builtin_tools(arm: Arm) -> str:
    return "Bash,Read,Skill" if arm in (Arm.CVV, Arm.VWP) else "Bash,Read"


def build_command(arm: Arm, cfg: ExternalConfig, binary: str, extra: list[str]) -> list[str]:
    cmd = [binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", cfg.model, "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--permission-mode", "dontAsk", "--permission-prompts", "none",
           "--tools", builtin_tools(arm), "--allowedTools", *allowed_tools(arm),
           "--disallowedTools", "WebFetch", "WebSearch"]
    if cfg.effort:
        cmd += ["--effort", cfg.effort]
    return cmd + extra


def agent_prompt(arm: Arm, task_text: str) -> str:
    if arm is Arm.CVV:
        return f"/claude-video-vision:watch-video ./video.mp4 {task_text}\n\n{_SANDBOX_NOTE}"
    if arm is Arm.POP:
        return f"The video is ./video.mp4. Use the popcorn MCP tools to watch it.\n{task_text}\n\n{_SANDBOX_NOTE}"
    if arm is Arm.VWP:
        return f"/vwatch:watch ./video.mp4 {task_text}\n\n{_SANDBOX_NOTE}"
    return f"The video is ./video.mp4. {task_text}\n\n{_SANDBOX_NOTE}"


# ---------------------------------------------------------------- run


def _turns(events: list[dict]) -> int:
    """Model turns = distinct assistant message ids (stream-json emits one event per content block)."""
    ids = set()
    for e in events:
        if e.get("type") == "assistant":
            msg = e.get("message")
            ids.add((msg.get("id") if isinstance(msg, dict) else None) or id(e))
    return len(ids)


def _usage(results: list[dict]) -> tuple[Usage, float]:
    usage = Usage()
    for r in results:
        usage = usage.plus(parse_usage(r.get("usage")))
    cost = max((float(r.get("total_cost_usd") or 0.0) for r in results), default=0.0)  # cumulative
    return usage, cost


def _converse(agent: AgentProcess, task: Task, prompt: str) -> tuple[list[dict], str, dict | None, str | None]:
    results = [agent.ask(prompt)]
    if results[-1].get("is_error"):
        raise LLMError(f"claude returned an error: {str(results[-1].get('result'))[:300]}")
    text = str(results[-1].get("result") or "")
    parsed, err = parse_report(task, text)
    if parsed is None:
        results.append(agent.ask(REPAIR_PROMPT.format(error=err)))
        if not results[-1].get("is_error"):
            text = str(results[-1].get("result") or "")
            parsed, err = parse_report(task, text)
    return results, text, parsed, err


def _write_log(cfg: ExternalConfig, arm: Arm, src: Path, record: dict) -> None:
    if cfg.log_dir is None:
        return
    path = cfg.log_dir / f"{arm.value}_{src.stem}_{uuid.uuid4().hex[:8]}.json"
    try:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    except OSError as exc:
        log.warning("could not write external-arm log %s: %s", path, exc)


def _cleanup(root: Path) -> None:
    try:
        shutil.rmtree(root)
    except OSError as exc:
        log.warning("could not delete run dir %s: %s", root, exc)


def _launch(arm: Arm, cfg: ExternalConfig, dirs: RunDirs, task: Task, prompt: str,
            box: list[AgentProcess]) -> tuple[list[dict], str, dict | None, str | None]:
    env = agent_env(cfg.base_env)
    setups = {Arm.CVV: setup_cvv, Arm.POP: setup_pop, Arm.VWP: setup_vwp}
    extra = setups[arm](cfg, dirs, env) if arm in setups else []
    cmd = build_command(arm, cfg, cfg.binary or find_claude_binary(), extra)
    box.append(AgentProcess.start(cmd, cwd=str(dirs.work), env=env, timeout_s=cfg.timeout_s))
    return _converse(box[0], task, agent_prompt(arm, prompt))


def run_external(arm: Arm, ctx: VideoContext, task: Task, prompt: str, backend: object,
                 cfg: ExternalConfig | None = None) -> ArmResult:
    """Run one external agent on one video. Never raises for run failures: they land in `error`."""
    if arm not in EXTERNAL_ARMS:
        raise ValueError(f"{arm} is not an external arm")
    cfg = cfg or ExternalConfig.from_backend(backend)
    started = time.monotonic()
    dirs: RunDirs | None = None
    box: list[AgentProcess] = []  # the agent, once started
    results: list[dict] = []
    text, parsed, perr, error = "", None, None, None
    try:
        try:
            dirs = make_run_dirs(Path(ctx.path), cfg.temp_root)
            results, text, parsed, perr = _launch(arm, cfg, dirs, task, prompt, box)
        except AgentTimeout as exc:
            error = f"timeout after {cfg.timeout_s:.0f}s: {exc}"
        except (LLMError, SetupError, OSError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if box:
                box[0].close()
        events = box[0].events if box else []
        results = results or [e for e in events if e.get("type") == "result"]
        calls = tool_calls(events)
        flags = audit(calls, dirs.work, repo_root=REPO_ROOT) if dirs else ()
        if flags:
            error = "audit: " + " | ".join(flags)[:1500] + (f" (also: {error})" if error else "")
        usage, cost = _usage(results)
        obs = infer_observations(events, dirs.work, ctx.info.duration) if dirs else ()
        res = ArmResult(raw_output=text, parsed=parsed, parse_error=perr, content_tokens_est=0, usage=usage,
                        cost_usd=cost, wall_s=time.monotonic() - started, turns=_turns(events),
                        observations=obs, error=error, tool_calls=tuple(c.to_log() for c in calls))
        _write_log(cfg, arm, Path(ctx.path), {
            "arm": arm.value, "video": str(ctx.path), "task": task.value, "model": cfg.model,
            "workdir": str(dirs.work) if dirs else None, "error": error, "audit": list(flags),
            "parse_error": perr, "raw_output": text, "cost_usd": cost, "wall_s": res.wall_s, "turns": res.turns,
            "usage": usage.model_dump(), "tool_calls": list(res.tool_calls),
            "permission_denials": [d.get("tool_input") for r in results for d in r.get("permission_denials") or []],
            "observations": [o.model_dump() for o in obs], "stderr_tail": box[0].stderr_tail if box else ""})
        return res
    finally:
        if dirs is not None:
            _cleanup(dirs.root)
