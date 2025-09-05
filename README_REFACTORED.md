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
Edit files in `berghain/config/strategies/`:
- `conservative.yaml` - Safe approach
- `aggressive.yaml` - High acceptance rate
- `adaptive.yaml` - Dynamic parameter adjustment

## Game Logs

All games are logged to `game_logs/` with structured JSON format including:
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