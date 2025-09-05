# ABOUTME: README for the refactored Berghain Challenge bot system
# ABOUTME: Documentation for clean architecture implementation

# Berghain Challenge Bot - Refactored

Clean, modular implementation of the Berghain Challenge game bot using domain-driven design principles.

## Architecture

```
berghain/
├── core/           # Domain models and API client
├── config/         # YAML configuration files
├── solvers/        # Strategy pattern implementations
├── runner/         # Parallel game execution
├── monitoring/     # TUI dashboard and streaming
├── logging/        # Structured game logging
└── analysis/       # Post-game analysis tools
```

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Run games:
```bash
python main.py run --scenario 1 --strategy conservative --count 5
```

Monitor games:
```bash
python main.py monitor
```

Analyze results:
```bash
python main.py analyze --limit 20
```

## Commands

### Run Games
```bash
python main.py run [options]
  --scenario 1         # Scenario ID (default: 1)
  --strategy conservative  # Strategy name (default: conservative)
  --count 10           # Number of games (default: 10)
  --workers 3          # Parallel workers (default: 3)
```

### Monitor
```bash
python main.py monitor [options]
  --file-watch         # Use file watching instead of TUI
```

### Analysis
```bash
python main.py analyze [options]
  --limit 20           # Number of recent games (default: 20)
  --scenario 1         # Filter by scenario
  --compare "conservative,aggressive"  # Compare strategies
  --parameter "threshold"  # Analyze parameter impact
  --strategy "rarity"  # Strategy for parameter analysis
```

### Reports
```bash
python main.py report [options]
  --days 7             # Days to include (default: 7)
```

## Configuration

### Scenarios
Edit files in `berghain/config/scenarios/`:
- `scenario_1.yaml` - Young & well-dressed constraints
- `scenario_2.yaml` - Creative constraint  
- `scenario_3.yaml` - Custom scenarios

### Strategies
Edit files in `berghain/config/strategies/`.

Supported strategy names (use with `--strategy`):
- `conservative` → Rarity-weighted, safe baseline
- `aggressive` → Rarity-weighted, stricter early game
- `adaptive` → Learns acceptance rates from state
- `balanced` → Blends rarity and shortage signals
- `greedy` → Accepts any unmet-constraint person
- `diversity` → Balances underrepresented constraints

Strategy config schema:
- `name`: Human-readable name (string)
- `description`: Short description (string)
- `parameters`: Dict of strategy parameters used by the implementation
- `scenario_adjustments`: Dict keyed by scenario ID (int or string) with param overrides

Common parameter keys recognized across strategies:
- `ultra_rare_threshold`: Frequency threshold treated as ultra-rare
- `rare_accept_rate`: Acceptance prob for ultra-rare cases
- `common_reject_rate`: Baseline prob for no-constraint fillers
- `early_game_threshold`, `mid_game_threshold`: Phase boundaries (0–1 capacity)
- `deficit_panic_threshold`: Progress threshold to consider a constraint critical

Notes:
- Scenario adjustments accept keys as `1:` or `'1':`; both are supported.
- Additional strategy-specific keys are documented in each YAML.

## Game Logs

All games are logged to `game_logs/` with structured logs:
- `game_*.json`: Final summary per game with metadata, params, and last 1000 decisions.
- `live_{solver_id}.json`: Live snapshot, overwritten on each update.
- `events_{solver_id}_*.jsonl`: Append-only NDJSON; one row per API response (post-decision state).

The summary JSON includes:
- Game metadata and timing
- Strategy parameters used
- Constraint satisfaction details
- Decision samples
- Final statistics

## Analysis Features

- Success rate and performance metrics
- Strategy comparison with statistical testing
- Parameter impact analysis
- Constraint satisfaction patterns
- Performance trends over time

## Monitoring

Real-time monitoring available through:
- Interactive TUI dashboard
- File-based log watching
- WebSocket streaming (for external integrations)

## Legacy Support

The system can read and analyze old game logs in addition to the new structured format.
