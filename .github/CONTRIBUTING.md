# Contributing

This is a research repo accompanying an arXiv preprint. Contributions that improve reproducibility, fix bugs, or extend the analysis are welcome.

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/) (Python package manager).
2. Clone and install:
   ```bash
   git clone https://github.com/kusp-dev/retrieval-weight-experiment.git
   cd retrieval-weight-experiment
   uv sync --all-extras
   ```
3. Pre-commit hooks are already configured. Install them:
   ```bash
   uv run pre-commit install
   ```

## Running Tests

```bash
uv run pytest                      # run all tests
uv run pytest tests/ --cov=src     # with coverage report
```

## Linting and Formatting

```bash
uv run ruff check src/ tests/      # lint
uv run ruff format src/ tests/     # auto-format
```

## Type Checking

```bash
uv run mypy src/
```

## Code Style

- Enforced by [ruff](https://docs.astral.sh/ruff/) via pre-commit hooks
- Line length: 100
- Python 3.11+
- See `pyproject.toml` `[tool.ruff]` and `[tool.mypy]` sections for exact configurations

## Pull Request Guidelines

- Keep changes focused -- one concern per PR.
- Include tests for new functionality.
- Make sure `uv run pytest`, `uv run ruff check`, and `uv run mypy src/` all pass before submitting.
- Reference any related issues in the PR description.

## Reporting Issues

Use the provided issue templates for bug reports and feature requests.
