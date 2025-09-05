# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Berghain Challenge bot system - a clean, modular implementation using domain-driven design to solve the Berghain nightclub admission optimization game. The system uses strategy patterns to implement different decision-making approaches across multiple game scenarios.

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Run Games
```bash
# Basic game run
python main.py run --scenario 1 --strategy conservative --count 5

# Parallel execution
python main.py run --scenario 1 --strategy aggressive --count 10 --workers 4
```

### Analysis and Monitoring
```bash
# Real-time TUI monitoring
python main.py monitor

# Strategy comparison analysis
python main.py analyze --compare "conservative,aggressive" --scenario 1

# Parameter impact analysis
python main.py analyze --parameter "ultra_rare_threshold" --strategy "rarity"

# Performance reporting
python main.py report --days 7
```

### Legacy Ultimate Solver
```bash
# Run the standalone ultimate solver (legacy)
python berghain_solver_ultimate.py
```

## Architecture Overview

### Domain-Driven Design Structure
- **Core Domain** (`berghain/core/`): Game state, constraints, decisions, and API client
- **Strategy Pattern** (`berghain/solvers/`): Pluggable decision-making implementations
- **Configuration** (`berghain/config/`): YAML-based scenarios and strategies
- **Analysis Engine** (`berghain/analysis/`): Statistical analysis and performance metrics
- **Monitoring** (`berghain/monitoring/`): Real-time TUI dashboard and streaming
- **Parallel Execution** (`berghain/runner/`): Multi-game coordination and batch processing

### Key Domain Models
- `GameState`: Tracks constraints, attribute counts, and game progress
- `Decision`: Records person evaluation with reasoning
- `Constraint`: Defines minimum requirements (e.g., 600 young people)
- `AttributeStatistics`: Frequency and correlation data for decision-making

### Strategy Architecture
Solvers implement clean separation between strategy logic and game execution:
- `BaseSolver`: Handles API interaction and game flow
- `DecisionStrategy`: Pure strategy logic without I/O concerns
- Configuration-driven parameter tuning via YAML files

## Game Mechanics

### Scenarios
- **Scenario 1**: 600 young + 600 well_dressed (32.3% frequency each)
- **Scenario 2**: Creative constraint (very rare attribute)
- **Scenario 3**: Multiple rare attributes

### Success Criteria
- Meet all attribute constraints
- Stay under 1000 admitted capacity
- Stay under 20,000 rejection limit
- Game status must be "completed"

## Configuration System

### Strategy Parameters (`berghain/config/strategies/`)
Key tunable parameters across all strategies:
- `ultra_rare_threshold`: Frequency threshold for ultra-rare attributes
- `phase1_multi_attr_only`: Early game selectivity
- `deficit_panic_threshold`: When to enter emergency mode
- `multi_attr_bonus`: Preference for people with multiple desired attributes

### Scenario Definitions (`berghain/config/scenarios/`)
Each scenario defines:
- Constraint requirements
- Expected attribute frequencies
- Strategy hints and difficulty assessment

## Development Patterns

### Adding New Strategies
1. Create YAML config in `berghain/config/strategies/`
2. Implement `DecisionStrategy` subclass in `berghain/solvers/`
3. Register in solver factory

### Decision Reasoning
All strategies must provide human-readable reasoning for decisions (used in analysis and debugging).

### Streaming Integration
Solvers support real-time streaming via callback mechanism for TUI integration.

## Analysis Features

### Statistical Methods
- Strategy comparison with significance testing
- Parameter impact analysis with correlation metrics
- Constraint satisfaction pattern analysis
- Performance trend tracking

### Game Logs
Structured JSON format in `game_logs/` directory containing:
- Complete decision history with reasoning
- Strategy parameters used
- Constraint satisfaction details
- Final performance metrics

## Testing Requirements

Currently no test framework exists. Future development should include:
- Unit tests for core domain models
- Integration tests for strategy implementations
- End-to-end tests for complete game scenarios
- Performance regression tests

## Key Dependencies

- `requests`: Berghain API communication
- `rich`: TUI dashboard rendering
- `pyyaml`: Configuration management
- `numpy/scipy`: Statistical analysis
- `websockets`: Real-time streaming (monitoring)