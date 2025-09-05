# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entry point for running, optimizing, monitoring, and analysis.
- `berghain/core`: API client and domain types.
- `berghain/solvers`: Strategy implementations (conservative, greedy, adaptive, etc.).
- `berghain/runner`: Orchestration and parallel execution.
- `berghain/monitoring`: TUI dashboard and file watcher utilities.
- `berghain/analysis`: Analytics and statistical comparison tools.
- `berghain/config/scenarios`: Scenario YAMLs (`scenario_<n>.yaml`).
- `berghain/config/strategies`: Strategy YAMLs, including `evolved/` variants.
- `game_logs/`: Run outputs and logs.

## Build, Test, and Development Commands
- Install deps: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Run strategies: `python main.py run --scenario 1 --strategy conservative --count 5`.
- Optimize strategies: `python main.py optimize --scenario 1 --workers 4 --generations 5`.
- Monitor/TUI: `python main.py monitor` (or `--file-watch`).
- API rate-limit test: `python test_api_limits.py` (writes `api_rate_limit_test_results.json`).

## Coding Style & Naming Conventions
- Python 3; follow PEP 8 with 4‑space indentation and type hints where practical.
- Naming: modules `snake_case`, classes `PascalCase`, functions/variables `snake_case`, constants `UPPER_SNAKE_CASE`.
- Keep functions small and single‑purpose; prefer clear docstrings and `logging` over prints (except CLI UX).
- Maintain consistency with existing modules; avoid introducing new dependencies without discussion.

## Testing Guidelines
- Current script-based test: `test_api_limits.py` for concurrency/rate limits.
- When adding tests, prefer `unittest` (baseline) or `pytest` if added to requirements.
- Place tests under `tests/` and name files `test_*.py`; target solvers, runner behaviors, and config parsing.
- Aim for meaningful coverage on strategy decisions and failure handling.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject line in Title Case (e.g., "Improve TUI Dashboard: Fix stale games").
- PRs: include summary (what/why), testing steps with sample commands, affected configs, performance impact, and screenshots/GIFs for TUI.
- Keep changes scoped; update/readme snippets and config examples when behavior or flags change.

## Security & Configuration Tips
- Do not commit secrets or personal `player_id`s. The default lives in `berghain/core/api_client.py`—override locally when testing.
- Be considerate of live API limits; use `--workers` conservatively and leverage built‑in backoff. Between heavy runs, pause to avoid 429s.
- Add new scenarios/strategies via YAML files following the existing naming patterns.
