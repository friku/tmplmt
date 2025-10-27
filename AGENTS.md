# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally lightweight; add only the folders you need. Keep runtime code inside `src/`, grouping modules by feature (`src/data/`, `src/services/`). Mirror that layout inside `tests/` so every module has a matching test file. Place reusable notebooks or exploratory scripts in `notebooks/` and keep sample inputs, fixtures, or UI assets in `assets/`. Shared developer utilities belong in `scripts/` (PowerShell or Bash) so automation stays discoverable.

## Build, Test, and Development Commands
Create an isolated environment before installing anything:\
`python -m venv .venv`\
`.venv\Scripts\activate`

Install and refresh dependencies from the pinned manifest:\
`pip install -r requirements.txt`

Run the full test suite locally before pushing:\
`pytest -q`

Package a release artifact when you need to validate distribution metadata:\
`python -m build`

Add comparable shell scripts in `scripts/` (for example `scripts/dev.ps1`) to wrap these commands for teammates.

## Coding Style & Naming Conventions
Follow Black’s default formatting (4‑space indentation, 88 character line width) and lint with Ruff focusing on the `pycodestyle`, `pyflakes`, and `isort` rule sets. Prefer descriptive module names (`src/services/job_runner.py`) and PascalCase for classes, snake_case for functions and variables. Keep top-level functions short; extract helpers into private modules named `_utils.py` when they are not part of the public API. Document public functions with concise docstrings that state purpose, inputs, and side effects.

## Testing Guidelines
Author tests with `pytest`, mirroring the module path (`tests/services/test_job_runner.py`). Use parametrized tests to cover boundary cases and include regression tests for discovered bugs. Maintain ≥90% branch coverage; check locally with `pytest --cov=src --cov-branch`. Store golden data under `tests/fixtures/` and keep it minimal. When introducing external services, guard them behind feature flags and replace them with fakes in tests.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) so automated changelog tooling can infer release notes. Keep commits focused, scoped to a single concern, and include a one-sentence body explaining the “why” when motivation is not obvious. For pull requests, supply a concise summary, testing evidence (`pytest -q` output or screenshots), and link any tracking tickets. Request review once the branch is rebased onto `main` and CI passes without warnings.

