# Use vwatch with any agent

vwatch has two parts:

- **The tools:** an MCP server with seven video tools (`video_info`, `video_transcript`, `video_signals`,
  `video_overview`, `video_seek`, `video_crop`, `video_listen`). Any agent that speaks MCP can use them.
- **The method:** the `watch-video` skill ([`SKILL.md`](../plugins/vwatch/skills/watch-video/SKILL.md)). It
  tells the agent to plan, take the free evidence, skim once, zoom on surprises and cite times. Agents that
  read Agent Skills load it on their own; for the others, paste it into their rules or instructions.

Every setup below starts the same server: `uvx` fetches vwatch from the release tag. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first if you don't have it; ffmpeg is fetched
automatically if it's missing. The first start downloads about 170–350 MB, so allow it a couple of minutes.

The frames come back as images, so **the agent's model must accept images**, and the agent must pass image
results from MCP tools through to it. The last column says whether we have checked that for each agent.

| Agent | Setup | Skill | Checked end to end |
|---|---|---|---|
| Claude Code | plugin (one command) | built in | yes, every release |
| Codex (OpenAI) | `config.toml` | `~/.agents/skills` | yes (2026-11-24, gpt-5.6-terra): found the flash, the dropout and the jump cut on the test video |
| Gemini CLI | extension (one command) | built in | not yet |
| Cursor | `mcp.json` | paste into a rule | not yet |
| VS Code (Copilot) | `.vscode/mcp.json` | paste into instructions | not yet |
| opencode | `opencode.json` | `~/.config/opencode/skills` | not yet |
| Windsurf, Claude Desktop, others | `mcpServers` JSON | paste into instructions | not yet |

"Not yet" means the config follows that agent's documented format but we haven't run it. If you try one,
please [open an issue](https://github.com/kashaf12/vwatch/issues) either way.

## Claude Code

```
/plugin marketplace add kashaf12/vwatch
/plugin install vwatch@vwatch
```

You get the server, the skill and `/vwatch:watch`. More options are in [mcp.md](mcp.md#claude-code-plugin).

## Codex (OpenAI)

Add the server to `~/.codex/config.toml`, or to `.codex/config.toml` in a trusted project:

```toml
[mcp_servers.vwatch]
command = "uvx"
args = ["--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.6", "vwatch-mcp"]
startup_timeout_sec = 180   # the first start downloads Python and vwatch
tool_timeout_sec = 600      # transcribing a long video can take a few minutes
```

Install the skill. Codex reads Agent Skills from `~/.agents/skills`; some versions use `~/.codex/skills`, so
copy it there too if it doesn't show up:

```bash
mkdir -p ~/.agents/skills/watch-video
curl -fsSL https://raw.githubusercontent.com/kashaf12/vwatch/v0.2.6/plugins/vwatch/skills/watch-video/SKILL.md \
  -o ~/.agents/skills/watch-video/SKILL.md
```

Then ask: *"Check ~/Desktop/final-cut.mp4 for glitches using the vwatch tools."*

## Gemini CLI

This repository is also a Gemini CLI extension. The extension bundles the server, the skill and a short
context file:

```bash
gemini extensions install https://github.com/kashaf12/vwatch
```

Update it later with `gemini extensions update vwatch`. If you would rather add only the server, put it in
`~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "vwatch": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.6", "vwatch-mcp"],
      "timeout": 600000
    }
  }
}
```

## Cursor

Add the server to `~/.cursor/mcp.json`, or to `.cursor/mcp.json` in a project:

```json
{
  "mcpServers": {
    "vwatch": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.6", "vwatch-mcp"]
    }
  }
}
```

For the method, add the body of `SKILL.md` as a project rule. Use the agent in the editor: there are
reports that the Cursor CLI does not pass image results from MCP tools to the model.

## VS Code (GitHub Copilot agent mode)

Add the server to `.vscode/mcp.json`:

```json
{
  "servers": {
    "vwatch": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.6", "vwatch-mcp"]
    }
  }
}
```

For the method, paste the body of `SKILL.md` into `.github/copilot-instructions.md`.

## opencode

Add the server to `opencode.json` (project) or `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "vwatch": {
      "type": "local",
      "command": ["uvx", "--python", "3.12", "--from", "vwatch[mcp] @ git+https://github.com/kashaf12/vwatch@v0.2.6", "vwatch-mcp"],
      "enabled": true,
      "timeout": 600000
    }
  }
}
```

Install the skill into `~/.config/opencode/skills/watch-video/SKILL.md`, using the same `curl` command as
for Codex with that destination.

## Windsurf, Claude Desktop and other MCP clients

Most clients take the same `mcpServers` block as Cursor:
- **Windsurf:** `~/.codeium/windsurf/mcp_config.json`.
- **Claude Desktop:** `claude_desktop_config.json`.

GUI apps often start with a minimal `PATH`. If the client can't find `uvx`, use its absolute path, which
`which uvx` prints. More client notes, Windows and running from a clone are covered in
[mcp.md](mcp.md#claude-desktop-cursor-windsurf-and-other-mcp-clients).
