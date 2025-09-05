# Berghain Game Monitoring & Solver Specification

## Overview

This specification defines a comprehensive real-time monitoring and analysis system for the Berghain nightclub simulation game. The system enables parallel game execution, live streaming visualization, and automated solver development.

## Core Architecture

### Game Flow
```
[Bot Solver] → [Berghain API] → [Data Collector] → [JSON Logs]
                                      ↓
[Stream Monitor] → [Multi-Game Dashboard] → [Analytics Engine]
```

### Key Components

1. **Game Dashboard** - Multi-view TUI for real-time monitoring
2. **Stream Monitor** - Real-time data extraction and streaming
3. **Session Manager** - Parallel game coordination 
4. **Analytics Engine** - Cross-game pattern analysis
5. **Success Highlighter** - Automated best-performance detection

## Game Rules & Constraints

### Success Criteria
- **Primary Goal**: Meet all attribute constraints (e.g., 600 young + 600 well_dressed)
- **Capacity Limit**: Maximum 1000 people admitted to venue
- **Rejection Limit**: Maximum 20,000 people rejected
- **Status**: Game must complete as "completed" (not "failed")

### Scenarios
1. **Scenario 1 (Friday Night)**: 600 young + 600 well_dressed (32.25% frequency each)
2. **Scenario 2 (Saturday Night)**: TBD
3. **Scenario 3 (Sunday Night)**: TBD

### Performance Metrics
- **Efficiency**: Acceptance rate (admitted / (admitted + rejected))
- **Constraint Fill Rate**: % of each constraint requirement met
- **Time to Complete**: Game duration
- **Success Rate**: % of games completed successfully

## Streaming Architecture

### Option A: File Watching + Partial JSON
**Best for**: No bot modifications needed
```python
# Monitor file system for JSON changes
watcher = FileSystemWatcher("game_logs/")
watcher.on_modified = parse_partial_json

# Parse incomplete JSON with recovery
def parse_partial_json(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Try to parse, recover from incomplete JSON
            return json.loads(content + '}]') if content.endswith(',') else json.loads(content)
    except: return None
```

### Option B: Socket Sidecar (Recommended)
**Best for**: Real-time streaming with minimal bot changes
```python
# Lightweight HTTP proxy/interceptor
class GameStreamProxy:
    def intercept_api_response(self, response):
        # Parse game state from API responses
        # Broadcast to TUI dashboard via WebSocket
        self.broadcast_update(self.extract_game_state(response))

# Minimal bot modification needed:
# self.stream_proxy.log_decision(person, decision) 
```

### Option C: Log Tailing
**Best for**: Detailed decision-by-decision streaming
```python
# Bots write to .stream files during gameplay
bot.stream_log("person_123: young=True, decision=accept")
# TUI tails these files for real-time updates
tailer = LogTailer("*.stream")
```

## Multi-Game Dashboard Architecture

### View Modes

#### 1. Grid View (Default)
```
┌─────────────────┬─────────────────┬─────────────────┐
│   Game 1 (S1)   │   Game 2 (S1)   │   Game 3 (S2)   │
│  ▶ Running      │  ● Completed    │  ✗ Failed       │
│  Young: 45%     │  Young: 100% ✓  │  Young: 82%     │
│  Dressed: 32%   │  Dressed: 100%✓ │  Dressed: 74%   │
│  Capacity: 456  │  Capacity: 1000  │  Capacity: 1000 │
│  Rejected: 2.3k │  Rejected: 8.5k │  Rejected: 966  │
├─────────────────┼─────────────────┼─────────────────│
│   Game 4 (S3)   │   Game 5 (S1)   │   Game 6 (S2)   │
│  ▶ Running      │  ⏸ Paused       │  🏆 BEST        │
│  ...            │  ...            │  SUCCESS RATE:  │
│                 │                 │  92% (11/12)    │
└─────────────────┴─────────────────┴─────────────────┘
```

#### 2. Detail View 
Full-screen focus on single game (similar to current game-stats.png)

#### 3. Leaderboard View
```
🏆 TOP PERFORMING STRATEGIES

Rank │ Strategy      │ Scenario │ Success Rate │ Avg Rejections
━━━━━┿━━━━━━━━━━━━━━━┿━━━━━━━━━━┿━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━
  1  │ OptimalBot    │    S1    │    94.2%     │     6,432
  2  │ DynamicBot    │    S1    │    87.8%     │     7,891  
  3  │ OptimalBot    │    S2    │    76.5%     │    12,334
  4  │ SimpleBot     │    S1    │    23.1%     │    19,456
```

#### 4. Analytics View
Cross-game insights, constraint correlation patterns, success predictors

#### 5. Split-Screen View
Compare two specific games side-by-side

### Color Coding System
- **Green**: Success/Completed/Above threshold
- **Yellow**: Running/In progress/Warning
- **Red**: Failed/Critical/Below threshold  
- **Blue**: Capacity/Secondary metrics
- **Purple**: Best performance highlights
- **Dim**: Inactive/Historical data

## Session Management

### Parallel Game Coordination
```python
class SessionManager:
    def run_parallel_session(self, config):
        # Config example:
        {
            "scenarios": [1, 2, 3],
            "bots": ["optimal", "dynamic", "simple"],
            "games_per_combination": 5,
            "max_concurrent": 6,
            "timeout_minutes": 30
        }
        
    def manage_game_lifecycle(self):
        # Queue management
        # Resource allocation  
        # Crash recovery
        # Results aggregation
```

### Process Pool Architecture
- **Worker Processes**: Execute individual games
- **Monitor Thread**: Health checking and recovery
- **Results Collector**: Aggregate statistics
- **Stream Broadcaster**: Real-time updates to TUI

## Success Detection & Highlighting

### Real-time Success Detection
```python
def is_game_successful(game_state):
    return (
        all(constraint.current >= constraint.required for constraint in game_state.constraints) and
        game_state.admitted_count <= 1000 and
        game_state.rejected_count <= 20000 and
        game_state.status == "completed"
    )

def highlight_success(game_id):
    # Visual celebration in TUI
    # Sound notification (optional)
    # Save replay data
    # Extract strategy parameters
```

### Performance Ranking Algorithm
```python
def calculate_game_score(game_result):
    if not game_result.successful:
        return 0
        
    efficiency = game_result.admitted / (game_result.admitted + game_result.rejected)
    constraint_bonus = sum(min(1.0, achieved/required) for achieved, required in constraints)
    time_bonus = max(0, (max_time - game_result.duration) / max_time)
    
    return efficiency * 0.4 + constraint_bonus * 0.4 + time_bonus * 0.2
```

## Solver Integration Protocol

### Bot Interface Standard
All bots must implement:
```python
class BerghainSolver:
    def start_game(self, scenario: int) -> GameState
    def make_decision(self, person: Person, game_state: GameState) -> bool
    def get_strategy_params(self) -> Dict[str, Any]
    def set_streaming_callback(self, callback: Callable)  # Optional
```

### Registration System
```python
# Bots register themselves with the system
REGISTERED_SOLVERS = {
    "optimal": BerghainBotOptimal,
    "dynamic": BerghainBot, 
    "simple": BerghainBotSimple,
    "ml_v1": MLBerghainBot,  # Future ML solvers
    "genetic": GeneticBot,   # Future genetic algorithm
}
```

### Strategy Parameter Extraction
```python
def extract_strategy(successful_game):
    return {
        "decision_thresholds": analyze_decision_patterns(successful_game),
        "constraint_priorities": analyze_constraint_focus(successful_game),
        "timing_patterns": analyze_decision_timing(successful_game),
        "attribute_correlations": analyze_attribute_usage(successful_game)
    }
```

## Communication Protocols

### TUI ↔ Stream Monitor
```python
# WebSocket messages
{
    "type": "game_update",
    "game_id": "abc123",
    "timestamp": 1694123456,
    "data": {
        "constraints": [{"attribute": "young", "current": 234, "required": 600}],
        "admitted": 456,
        "rejected": 2340,
        "status": "running"
    }
}
```

### Session Manager ↔ Bots
```python
# IPC message queue
{
    "command": "start_game",
    "bot_type": "optimal",
    "scenario": 1,
    "game_id": "game_001",
    "streaming_port": 8001
}
```

## File Structure

```
├── SPEC.md                    # This specification
├── game_dashboard.py          # Main TUI application
├── stream_monitor.py          # Real-time streaming system
├── session_manager.py         # Parallel game coordinator
├── analytics.py              # Cross-game analysis
├── success_highlighter.py    # Best performance detection
├── solver_registry.py        # Bot registration system
├── config/
│   ├── dashboard_layouts.yaml
│   ├── streaming_config.yaml
│   └── session_presets.yaml
├── game_logs/                 # JSON game logs (existing)
├── streaming_logs/            # Real-time stream data
└── replays/                   # Best game replays
```

## Performance Requirements

### Real-time Constraints
- **Update Frequency**: 2-5 Hz for live games
- **Display Latency**: < 500ms from decision to TUI update
- **Concurrent Games**: Support up to 12 simultaneous games
- **Memory Usage**: < 500MB for full dashboard

### Scalability Targets
- **Game History**: Store up to 10,000 games
- **Analytics Period**: Real-time analysis of last 100 games
- **Solver Capacity**: Support 50+ registered bot types

## Future Extensions

### Machine Learning Integration
- **Feature Extraction**: Automatic pattern detection from successful games
- **Neural Network Bots**: Train bots on historical game data
- **Reinforcement Learning**: Online learning from game outcomes

### Advanced Analytics
- **Predictive Models**: Success probability estimation during gameplay
- **Strategy Evolution**: Track how strategies improve over time
- **Meta-Strategy Analysis**: Optimal strategy selection based on scenario

### Web Interface
- **Remote Monitoring**: Web dashboard for game monitoring
- **API Server**: REST API for external integrations
- **Data Export**: CSV/JSON export for external analysis

## Configuration Examples

### Single Bot Testing
```yaml
session_type: "single_bot_test"
bot: "optimal"
scenarios: [1, 2, 3]
runs_per_scenario: 10
display_mode: "detail_view"
streaming: true
```

### Strategy Comparison
```yaml
session_type: "bot_comparison"
bots: ["optimal", "dynamic", "simple"]
scenario: 1
runs_per_bot: 5
display_mode: "split_screen"
save_replays: true
```

### Marathon Testing
```yaml
session_type: "marathon"
bots: ["optimal"]
scenario: 1
total_runs: 100
max_concurrent: 4
display_mode: "grid_view"
analytics: "real_time"
```

---

*This specification is designed to be extensible and should evolve as new solver strategies and monitoring requirements emerge.*