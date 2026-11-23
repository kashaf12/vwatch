# Contributing to vwatch

Thanks for helping. Bug reports, small fixes, new signal detectors, new eval tasks and results on your own videos are all welcome.

## Dev setup

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`.

```bash
git clone https://github.com/kashaf12/vwatch.git
cd vwatch
uv sync --all-extras          # runtime deps, the Anthropic SDK extra and the dev group (pytest)
uv run pytest -q              # ~740 tests, about 1-2 minutes, no network or model calls
uv run pytest --cov=vwatch    # with coverage
```

On Linux, install a font so frame timestamps render with a real typeface (`apt install fonts-dejavu-core`); without one, Pillow's built-in font is used and everything still works.

## Ground rules

- **Tests never touch the network or a model.** Use `vwatch.llm.fake` for model replies, the tiny synthetic videos in `tests/conftest.py` for media, and fake clients for downloads (see `tests/test_tvbench.py`). A test that needs `ANTHROPIC_API_KEY`, the `claude` binary or the internet will be rejected.
- **Write the test first.** Red, green, then tidy. Aim for at least 80% coverage on new logic.
- **Don't change an expected value to make a test pass** without explaining why in the PR.
- **Immutability.** Return new objects; don't mutate inputs or shared state. Dataclasses are `frozen=True`, pydantic models are copied with `model_copy(update=...)`.
- **Small files.** 200–400 lines is typical, 800 is the ceiling. Organise by feature (`arms/`, `evaluation/`, `dataset/`), not by type.
- **Validate at the edges.** External input (model JSON, truth files, CLI args) is parsed with a schema and fails fast with a clear message. Never swallow an error silently: user-facing messages stay short, logs and `--json` keep the detail.
- **Keep the comparison fair.** Every arm gets the same task prompt (`vwatch/tasks.py`) and the same video. If you change how evidence reaches the model in one arm, say what that does to the matched-budget comparison.

## Changing the watcher or a baseline

The eval protocol in [docs/eval-protocol.md](docs/eval-protocol.md) is the contract. Before you open a PR that changes `arms/`, `signals.py` or `tasks.py`:

1. Run the pilot split for the arms you touched and for `B1` at the same budget (this spends model tokens; see the README's cost notes).
2. Put the before/after headline line in the PR description.
3. A drop of more than 3 pp on any S-metric blocks the change.

Thresholds are fixed before a full run. If you think one is wrong, open an issue; changes go in the protocol's changelog.

## Commits and PRs

- Commit messages: `<type>: <description>` with types `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
- One topic per PR; fill in the PR template.
- Don't commit videos, `eval/data/`, `eval/runs/`, `vendor/` or anything with keys in it. `.gitignore` covers the usual cases.

## Data you add

Only add sources you have the right to use, and record their license in [docs/DATA.md](docs/DATA.md). Openly licensed sources (CC BY, CC BY-SA, CC0, public domain) are preferred so others can rebuild the set.

## License

By contributing you agree that your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
