# Berghain Challenge Bot 🎯

**Intelligent strategy optimization system for the Berghain nightclub admission game**

A sophisticated, AI-powered bot that uses genetic algorithms, real-time performance monitoring, and dynamic resource allocation to discover optimal strategies for the Berghain Challenge.

## ✨ Key Features

### 🧬 **Genetic Strategy Evolution**
- **Automatic strategy generation** through mutation and crossover
- **Real-time performance monitoring** with early termination of underperformers
- **Multi-generational optimization** with fitness-based selection
- **Continuous learning** from successful strategies

### ⚡ **Smart Resource Management** 
- **High-score early termination** saves computation time
- **Dynamic worker allocation** based on performance
- **API rate limiting** with exponential backoff retry logic
- **Parallel execution** across multiple strategies

### 📊 **Advanced Analytics**
- **Statistical strategy comparison** with significance testing
- **Parameter impact analysis** with correlation metrics
- **Real-time TUI dashboard** for live monitoring
- **Comprehensive performance reporting** with trend analysis

### 🎮 **Multi-Strategy Execution**
- Run **all available strategies** simultaneously
- **Strategy comparison** mode for head-to-head analysis  
- **Custom parameter tuning** per strategy and scenario
- **Legacy solver integration** for backwards compatibility

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd berghain-challenge-bot
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run single strategy (local simulator by default)
python main.py run --scenario 1 --strategy conservative --count 5 --mode local

# Run all available strategies in parallel (local)
python main.py run --scenario 1 --strategy all --workers 6 --mode local

# Compare specific strategies
python main.py run --scenario 1 --strategy "conservative,aggressive,adaptive" --mode local

# Use the live API (slow/flaky) sparingly
python main.py run --scenario 1 --strategy rbcr --count 10 --workers 2 --mode api
```

## 🧬 Genetic Optimization

**Automatically discover superior strategies through evolution:**

```bash
# Quick optimization (recommended)
python main.py optimize --scenario 1 --workers 4 --generations 5

# Intensive optimization 
python main.py optimize --scenario 1 --workers 6 --generations 10 --target-success-rate 90

# Rapid prototyping
python main.py optimize --scenario 1 --workers 2 --games-per-strategy 1 --generations 3
```

### How Evolution Works:
1. **Population Initialization**: Starts with base strategies (conservative, aggressive, adaptive, etc.)
2. **Performance Monitoring**: Real-time tracking of efficiency and success rates
3. **Smart Termination**: Kills underperforming strategies after 60 seconds
4. **Genetic Operations**: 
   - **Mutation**: Tweaks successful strategy parameters
   - **Crossover**: Combines traits from top performers
   - **Selection**: Fitness-based survival of the fittest
5. **Continuous Evolution**: Each generation builds on the previous best

## 💻 Local Simulator vs API

- The CLI supports two backends via `--mode`:
  - `--mode local` (default): fast, offline simulator that uses Scenario YAML expected frequencies/correlation. It can also load a calibration profile to match recent API behavior.
  - `--mode api`: live API; use only for small canary batches due to slowness and flakiness.

### Calibrate the Simulator
```bash
# Estimate freqs + correlation from recent API event logs
python -m berghain.analysis.calibrate_sim > sim_profile.json

# Point the simulator at this profile (or place at berghain/config/feasibility/scenario_1.json)
export BERGHAIN_SIM_PROFILE=sim_profile.json

# Train large batches locally
python main.py run --scenario 1 --strategy rbcr --count 500 --workers 8 --mode local

# Freeze learning and validate on API
# (set dual_eta: 0.0 and dual_decay: 1.0 under scenario_1 in rbcr.yaml)
python main.py run --scenario 1 --strategy rbcr --count 10 --workers 2 --mode api
```

## 🚀 Evolution → Production Pipeline

### **Step 1: Run Optimization**
```bash
python main.py optimize --scenario 1 --workers 4 --generations 5
```

### **Step 2: Discover Evolved Strategies**
```bash
# Check what was created
ls berghain/config/strategies/evolved/

# Example evolved strategies:
# greedy_gen2_1847.yaml - Mutated greedy with optimized parameters
# cross_greedy_adaptive_gen3.yaml - Hybrid combining best traits
# balanced_mutant_gen4.yaml - Novel parameter combinations
```

### **Step 3: Test Evolved Strategies**
```bash
# Test single evolved strategy
python main.py run --scenario 1 --strategy greedy_gen2_1847 --count 5

# Compare evolved vs original
python main.py run --scenario 1 --strategy "greedy,greedy_gen2_1847" --count 10

# Tournament mode - all strategies compete
python main.py run --scenario 1 --strategy all --count 10
```

### **Step 4: Performance Analysis**
```bash
# Statistical comparison
python main.py analyze --compare "greedy_gen2_1847,greedy" --scenario 1

# Multi-generation tracking
python main.py report --days 7
```

### **Step 5: Production Deployment**
```bash
# Deploy champion strategy for production runs
python main.py run --scenario 1 --strategy your_champion_strategy --count 50 --workers 6
```

## 🔄 Advanced Evolution Workflows

### **Continuous Evolution Cycle**
```bash
# Multi-cycle optimization for maximum performance
for i in {1..5}; do
    echo "🧬 Evolution Cycle $i"
    python main.py optimize --scenario 1 --workers 4 --generations 3
    python main.py run --scenario 1 --strategy all --count 5
    python main.py analyze --limit 20
done
```

### **Strategy Integration Methods**

**Auto-Discovery (Recommended):**
```bash
# Automatically includes evolved strategies
python main.py run --scenario 1 --strategy all --count 20
```

**Manual Selection:**
```bash
# Cherry-pick specific evolved strategies  
python main.py run --scenario 1 --strategy "greedy_gen3_2847,balanced_mutant_gen2_1543"
```

**Head-to-Head Comparison:**
```bash
# Compare original vs evolved
python main.py analyze --compare "original_strategy,evolved_strategy" --scenario 1
```

### **Expected Evolution Results**

Evolved strategies typically show improvements like:
```yaml
# greedy_gen2_1847.yaml - Optimized parameters
parameters:
  ultra_rare_threshold: 0.048        # Fine-tuned from data
  constraint_urgency_multiplier: 2.3 # Learned parameter
  early_game_aggression: 0.85        # Evolved trait

# cross_greedy_adaptive_gen3.yaml - Hybrid strategy
parameters:
  constraint_focus: 2.1    # Inherited from greedy parent
  learning_rate: 0.15      # Inherited from adaptive parent
  hybrid_threshold: 0.7    # Novel crossover trait
```

## 📊 Analytics & Monitoring

### Real-Time Dashboard
```bash
# Interactive TUI dashboard
python main.py monitor

# File-based monitoring
python main.py monitor --file-watch
```

### Strategy Analysis
```bash
# Compare two strategies statistically
python main.py analyze --compare "conservative,aggressive" --scenario 1

# Analyze parameter impact
python main.py analyze --parameter "ultra_rare_threshold" --strategy "rarity"

# Recent performance summary
python main.py analyze --limit 20

# Weekly performance report
python main.py report --days 7
```

## ⚙️ Configuration

### High Score System
Automatically configured in `berghain/config/high_scores.yaml`:
```yaml
high_scores:
  scenario_1: 716   # Current best rejection count
  scenario_2: 3140  
  scenario_3: 4003  

settings:
  buffer_percentage: 1.35  # Allow 35% beyond record to finish under guard
  enabled: true            # Enable early termination
```

### Strategy Configuration
Located in `berghain/config/strategies/`:

**Available Strategies:**
- `conservative` → Safe baseline with rarity weighting
- `aggressive` → Stricter early-game filtering  
- `adaptive` → Learns optimal acceptance rates
- `balanced` → Balances multiple signals
- `greedy` → Constraint-focused approach
- `diversity` → Ensures balanced attribute distribution
- `quota` → Gap-aware quota tracker (dual first, singles by lag, minimal filler)
- `dual` → DualDeficit controller (urgency-based singles + rate-floor PI control)
- `rbcr` → Re-solving Bid-Price with Confidence Reserves (near-optimal control)
- `rbcr2` → Enhanced RBCR with LP-optimized dual prices and joint probability handling

**Custom Parameters:**
```yaml
parameters:
  ultra_rare_threshold: 0.05
  deficit_panic_threshold: 0.8
  multi_attr_bonus: 2.0
  rarity_multiplier: 3.0
```

## 🎯 Advanced Usage Examples

### Multi-Strategy Comparison
```bash
# Run comprehensive comparison
python main.py run --scenario 1 --strategy all --count 10 --workers 8

# Disable high-score checking for full runs
python main.py run --scenario 1 --strategy "conservative,aggressive" --no-high-score-check
```

### Optimization Workflows
```bash
# Conservative optimization (API-friendly)
python main.py optimize --scenario 1 --workers 2 --games-per-strategy 2

# Production optimization
python main.py optimize --scenario 1 --workers 6 --generations 8 --target-success-rate 85

# Quick strategy discovery
python main.py optimize --scenario 1 --workers 4 --generations 3
```

### Analysis Workflows
```bash
# Strategy head-to-head
python main.py analyze --compare "best_evolved,conservative" --scenario 1

# Parameter optimization
python main.py analyze --parameter "rarity_multiplier" --strategy "adaptive"

# Performance trends
python main.py report --days 30
```

## 🏗️ Architecture

```
berghain/
├── core/              # Domain models & API client
│   ├── domain.py      # Game state, constraints, decisions
│   ├── api_client.py  # HTTP client with retry logic
│   ├── local_simulator.py  # Offline simulator backend (default)
│   └── high_score_checker.py # Early termination system
├── config/            # YAML configuration files
│   ├── scenarios/     # Game scenario definitions
│   ├── strategies/    # Strategy parameter configs
│   └── high_scores.yaml # Performance thresholds
├── solvers/           # Strategy implementations
│   ├── base_solver.py # Common game execution logic
│   ├── quota_solver.py       # QuotaTracker
│   ├── dual_deficit_solver.py# DualDeficit
│   └── rbcr_solver.py        # RBCR controller
├── optimization/      # Genetic algorithm system
│   ├── strategy_monitor.py    # Real-time performance tracking
│   ├── strategy_evolution.py # Genetic operations
│   └── dynamic_runner.py     # Evolution orchestration  
├── runner/            # Parallel execution
│   ├── parallel_runner.py # Multi-game coordination
│   └── game_executor.py   # Single game execution
├── monitoring/        # Real-time dashboards
│   └── tui_dashboard.py # Interactive terminal UI
└── analysis/          # Post-game analytics & utilities
    ├── game_analyzer.py           # Performance analysis
    ├── statistical_analyzer.py    # Statistical testing
    ├── calibrate_sim.py           # Build sim profile from logs
    └── feasibility_table.py       # Monte Carlo feasibility oracle
```

## 📈 Performance Optimization

### API Rate Limiting
- **Automatic retry** with exponential backoff
- **Request delays** to prevent 429 errors  
- **Keep API batches small** (use `--mode api` with low `--workers`) and do large runs offline with `--mode local`

### Resource Efficiency  
- **High score early termination** (stops at 95% of best known score)
- **Real-time underperformer detection** (kills inefficient strategies in 60s)
- **Smart worker allocation** (adapts to available resources)

### Genetic Optimization
- **Mutation rates** automatically adjusted based on performance
- **Crossover strategies** preserve successful traits
- **Fitness scoring** based on success rate and efficiency
- **Population diversity** maintained through selection pressure

## 🎮 Game Scenarios

### Scenario 1: Friday Night
- **Constraints**: 600 young + 600 well_dressed
- **Frequency**: ~32.3% each attribute  
- **Current Record**: 716 rejections
- **Strategy Focus**: Rarity-weighted selection

### Scenario 2: Creative Scene  
- **Constraints**: Rare creative attribute
- **Current Record**: 3140 rejections
- **Strategy Focus**: Ultra-rare detection

### Scenario 3: Multi-Rare
- **Constraints**: Multiple rare attributes
- **Current Record**: 4003 rejections  
- **Strategy Focus**: Balanced rare collection

## 🔧 Development

### Adding New Strategies
1. Create YAML config in `berghain/config/strategies/`
2. Implement `DecisionStrategy` subclass
3. Add to strategy factory in solvers module

### Custom Analysis
```python
from berghain.analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer("game_logs")
result = analyzer.compare_strategies("strategy_a", "strategy_b")
```

### Monitoring Integration
```python
from berghain.optimization import StrategyPerformanceMonitor

monitor = StrategyPerformanceMonitor()
monitor.add_termination_callback(my_callback)
```

## 📝 Game Logs

**Structured logging** in `game_logs/`:
- `game_*.json` - Complete game summaries with decisions
- `live_*.json` - Real-time status snapshots  
- Performance metrics and constraint satisfaction data

## 🏆 Records & Achievements

- **Best Scenario 1**: 716 rejections (Maksim)
- **Best Scenario 2**: 3140 rejections  
- **Best Scenario 3**: 4003 rejections
- **Total Challenge Score**: 7859

## 📊 Algorithm Performance Benchmarks

*Complete results from 21 strategies (5 runs each, 20 workers, scenario 1, September 2025):*

| Strategy | Success Rate | Avg Rejections | Performance | Status |
|----------|-------------|---------------|-------------|--------|
| **perfect** | **100.0%** (5/5) | **932** | 🥇 **Best Consistent** | ✅ Production Ready |
| **ultimate3h** | **100.0%** (5/5) | **922** | 🥈 **Excellent** | ✅ Reliable |
| **dual** | 60.0% (3/5) | 925 | 🥉 **Good** | ⚠️ Moderate Success |
| **adaptive** | 40.0% (2/5) | 942 | Fair | ⚠️ Inconsistent |
| **ultimate2** | 40.0% (2/5) | 952 | Fair | ⚠️ Inconsistent |
| **ultimate3** | 40.0% (2/5) | 909 | Fair | ⚠️ Inconsistent |
| **rbcr** | 20.0% (1/5) | 845 | Poor | ❌ **Best Single Run** |
| **optimal_safe** | 20.0% (1/5) | 934 | Poor | ❌ Unreliable |
| aggressive | 0.0% (0/5) | - | Failed | ❌ Not Working |
| balanced | 0.0% (0/5) | - | Failed | ❌ Not Working |
| diversity | 0.0% (0/5) | - | Failed | ❌ Not Working |
| conservative | 0.0% (0/5) | - | Failed | ❌ Not Working |
| greedy | 0.0% (0/5) | - | Failed | ❌ Not Working |
| quota | 0.0% (0/5) | - | Failed | ❌ Not Working |
| dvo | 0.0% (0/5) | - | Failed | ❌ Not Working |
| ramanujan | 0.0% (0/5) | - | Failed | ❌ Not Working |
| ultimate | 0.0% (0/5) | - | Failed | ❌ Not Working |
| rbcr2 | 0.0% (0/5) | - | Failed | ❌ Not Working |
| mec | 0.0% (0/5) | - | Failed | ❌ Not Working |
| optimal | 0.0% (0/5) | - | Failed | ❌ Not Working |
| optimal_final | 0.0% (0/5) | - | Failed | ❌ Not Working |

**🏆 Overall Results:** 21 successful games out of 105 total (20.0% success rate)  
**⚡ Best Single Result:** 845 rejections (rbcr strategy)  
**🎯 Target:** 716 rejections (current record)

### 🎯 **Performance Summary:**
- **Target:** 716 rejections (Maksim's record) 
- **Best Production Algorithm:** Perfect (932 avg rejections, 100% success)
- **Most Reliable:** Perfect & Ultimate3H (100% success rate)
- **Best Single Run:** RBCR (845 rejections, but only 20% success rate)

### 🧠 **Key Findings:**
- **Top Tier Algorithms:** Perfect and Ultimate3H deliver consistent 100% success
- **Moderate Performers:** Dual (60%), Adaptive/Ultimate2/Ultimate3 (40% each)
- **Struggling Algorithms:** 13 strategies failed completely (0% success)
- **Success Rate Crisis:** Only 20% overall success across all runs

### ⚡ **Strategic Insights:**
- **Reliability vs Performance:** Perfect/Ultimate3h trade slightly higher rejection counts for guaranteed success
- **High-Risk/High-Reward:** RBCR achieved the best single result (845) but failed 80% of the time
- **Algorithm Maturity:** Most strategies appear to need tuning or debugging
- **Production Recommendation:** Use Perfect strategy for reliable results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request

## 📄 License

[Add your license here]

---

## 🎯 Complete Workflow Examples

### **Scenario Optimization Workflow**
```bash
# 1. Initial optimization discovery
python main.py optimize --scenario 1 --workers 4 --generations 5

# 2. Monitor real-time evolution (in separate terminal)
python main.py monitor

# 3. Test all strategies including evolved ones
python main.py run --scenario 1 --strategy all --count 10

# 4. Statistical analysis to find winners
python main.py analyze --limit 50 --scenario 1

# 5. Compare best evolved vs original
python main.py analyze --compare "champion_evolved,original_baseline"

# 6. Production deployment
python main.py run --scenario 1 --strategy champion_evolved --count 100
```

### **Multi-Scenario Campaign**
```bash
# Optimize each scenario independently
for scenario in 1 2 3; do
    python main.py optimize --scenario $scenario --workers 4 --generations 5
    python main.py analyze --limit 20 --scenario $scenario
done

# Cross-scenario strategy testing
python main.py run --scenario 1 --strategy "$(ls berghain/config/strategies/evolved/*.yaml | head -5 | xargs -I {} basename {} .yaml | tr '\n' ',')"
```

### **Continuous Improvement Pipeline**
```bash
# Weekly optimization cycles
while true; do
    # Run optimization
    python main.py optimize --scenario 1 --workers 6 --generations 8
    
    # Performance testing
    python main.py run --scenario 1 --strategy all --count 20
    
    # Generate reports
    python main.py report --days 7
    
    # Wait a week
    sleep 604800
done
```

## 🔬 RBCR2 Algorithm

RBCR2 represents the latest advancement in the RBCR family, implementing cutting-edge algorithmic improvements based on detailed performance analysis and optimization theory.

### **Key Enhancements**
- **LP-Optimized Dual Computation**: Uses CVXPY to solve small 4-variable LP for optimal dual multipliers λ_y, λ_w, μ
- **Joint Probability Handling**: Computes proper joint probabilities (p11, p10, p01, p00) using Gaussian copula approximation
- **Enhanced Feasibility Oracle**: Uses Bonferroni bounds with joint probabilities instead of naive independent tests  
- **Principled Dual Learning**: Implements proper subgradient updates with step size decay and projection
- **Improved Parameter Tuning**: More aggressive filler acceptance and tighter oracle confidence intervals

### **Usage Examples**
```bash
# Basic RBCR2 usage
python main.py run --scenario 1 --strategy rbcr2 --count 10

# Compare RBCR vs RBCR2 performance
python main.py run --scenario 1 --strategy "rbcr,rbcr2" --count 50

# Analyze RBCR2 improvements
python main.py analyze --compare "rbcr,rbcr2" --scenario 1
```

### **Performance Characteristics**
- **Best Case**: 787 rejections (35 fewer than standard RBCR)
- **Efficiency**: 4.3% better rejection count when successful
- **Trade-off**: More selective success rate (10%) vs standard RBCR (30%) due to conservative exploration
- **Potential**: With CVXPY installed, expected 25-35% success rate with consistent performance

### **Installation for Full Performance**
```bash
# Install CVXPY for LP-optimized duals
pip install cvxpy

# Enable LP solver in config
# Set use_lp_duals: true in berghain/config/strategies/rbcr2.yaml
```

## 🏆 Strategy Evolution Tracking

### **Performance Genealogy**
```bash
# Track strategy lineage
python main.py analyze --compare "greedy,greedy_gen2_1847" --scenario 1
python main.py analyze --compare "greedy_gen2_1847,greedy_gen3_4821" --scenario 1

# Measure evolution progress
python main.py analyze --parameter "efficiency_improvement" --strategy "evolved"
```

### **Champion Selection Process**
```bash
# 1. Tournament selection
python main.py run --scenario 1 --strategy all --count 15

# 2. Statistical validation
python main.py analyze --compare "top_candidate,current_champion" --scenario 1

# 3. Production validation
python main.py run --scenario 1 --strategy top_candidate --count 30

# 4. Crown new champion if superior
# Manual decision based on statistical significance
```

## 💡 Pro Tips & Best Practices

### **Optimization Strategy**
- **Start with optimization mode** to discover best strategies for your scenarios
- **Use monitoring dashboard** to watch strategies compete in real-time  
- **Enable high-score checking** to save compute time on poor runs
- **Evolution works best** with 5-10 generations for meaningful improvement

### **Production Deployment**
- **Compare strategies statistically** before committing to production runs
- **Test evolved strategies** on small batches before full deployment
- **Maintain strategy genealogy** to understand performance lineage
- **Run continuous evolution cycles** to stay ahead of the competition

### **Performance Optimization**
- **Use appropriate worker counts** based on API stability (start with 2-4)
- **Monitor real-time efficiency** to spot winners early
- **Combine multiple optimization cycles** for maximum improvement
- **Archive champion strategies** for future baseline comparisons

### **API Management**
- **Respect rate limits** with proper worker scaling
- **Use high-score termination** to avoid wasted computation
- **Monitor 429 errors** and adjust concurrency accordingly
- **Implement request delays** for sustained operation

**Happy evolving! Your strategies will get smarter every generation! 🧬🚀**
