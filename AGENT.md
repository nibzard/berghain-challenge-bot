# AGENT.md

## Build/Test Commands
```bash
# Environment setup
pip install -r requirements.txt

# Basic game execution
python main.py run --scenario 1 --strategy conservative --count 5

# Run multiple strategies with high score checking disabled
python main.py run --scenario 1 --strategy "conservative,aggressive" --no-high-score-check

# Optimize strategies with genetic evolution
python main.py optimize --scenario 1 --workers 4 --generations 5

# Legacy solver execution
python berghain_solver_ultimate.py

# No formal tests exist yet - use game runs for validation
python main.py analyze --limit 10
```

## Code Style Guidelines
- **Imports**: Use absolute imports from berghain package, relative imports within modules
- **Types**: Use type hints (List, Optional, Callable) from typing module
- **Naming**: snake_case for functions/variables, PascalCase for classes, ALL_CAPS for constants
- **Comments**: Use `# ABOUTME:` comments at top of files for module descriptions
- **Error handling**: Use custom exceptions (e.g., BerghainAPIError) and proper logging
- **Structure**: Domain-driven design with clean separation (core/, solvers/, config/, analysis/)
- **Strategy pattern**: Implement DecisionStrategy interface for new solver strategies
- **Logging**: Use module-level loggers: `logger = logging.getLogger(__name__)`
