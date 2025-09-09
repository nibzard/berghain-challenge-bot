# MASTERPLAN: Ultra-Optimized Training Data Generation & Transformer Improvement

**Goal**: Beat the current record of 716 rejections by generating superior training data and improving transformer architecture.

## Current Situation Analysis

- **Current Record**: 716 rejections (Maksim)
- **Transformer Performance**: 843-988 rejections (avg ~880)
- **Best Algorithms**: RBCR achieving 761-777 rejections in best runs
- **Training Data Issue**: Only 119 elite games, mostly 800+ rejections
- **Core Problem**: Transformer can't learn to be better than its training data

## Phase 1: Generate Superior Training Data

### 1A. Mass Run Top Algorithms (16,000+ games total)

Execute the mass generation script:
```bash
./generate_training_data.sh
```

This will run:
- **RBCR**: 5,000 games (best early game performance)
- **RBCR2**: 5,000 games (consistent performer) 
- **Ultimate3**: 2,000 games (hybrid approach)
- **Perfect**: 2,000 games (balanced strategy)
- **Apex**: 2,000 games (advanced logic)

**Expected**: 500+ games with <750 rejections, 2000+ games with <800 rejections

### 1B. Filter for Ultra-Elite Games

Create multi-tier filtering system:
- **Tier 1 (Ultra-Elite)**: < 750 rejections 
- **Tier 2 (Elite)**: 750-800 rejections
- **Tier 3 (Good)**: 800-850 rejections

Implementation:
```python
# Use existing filter_ultra_elite_games.py
python filter_ultra_elite_games.py --max-rejections 750 --output ultra_elite_games
python filter_ultra_elite_games.py --max-rejections 800 --output elite_games_filtered
```

### 1C. Create Synthetic Optimal Games

Analyze patterns from sub-750 games and generate synthetic "perfect" games:
- Identify optimal decision patterns from best games
- Create variations with slight randomization  
- Generate 1000+ synthetic games with 700-750 rejections

## Phase 2: Advanced Data Augmentation

### 2A. Decision Sequence Splicing
- Take best decision sequences from different games
- Splice together optimal segments
- Create hybrid games that combine best decisions from multiple strategies

### 2B. Rejection Minimization Analysis
```python
def analyze_rejection_patterns():
    # Find common rejection points across strategies
    # Identify avoidable rejections in near-optimal games
    # Create counter-factual better decisions
```

### 2C. Generate Counter-Factual Training Data
For each 800+ rejection game:
- Identify suboptimal decisions that led to excess rejections
- Generate alternative decision sequences
- Create "what-if" games with better outcomes

## Phase 3: Transformer Architecture Improvements

### 3A. Dual-Head Architecture
```python
class DualHeadTransformer(nn.Module):
    def __init__(self):
        # Head 1: Constraint satisfaction (focus on meeting requirements)
        # Head 2: Efficiency (focus on minimizing rejections)
        # Combine both heads for final decision
```

Benefits:
- Explicit optimization for both success AND efficiency
- Can learn different strategies for different phases
- Better handling of constraint vs efficiency tradeoffs

### 3B. Reward-Conditioned Training
Train transformer with explicit rejection targets:
- **Input**: Game state + target rejection count
- **Output**: Decision that achieves target
- **Training**: Use games labeled with final rejection counts

```python
# Example: "Play to achieve <750 rejections"
decision = transformer(game_state, target_rejections=750)
```

### 3C. Ensemble Approach
Train multiple specialized transformers:
- **Early Game Specialist** (0-300 admits): Focus on efficient constraint gathering
- **Mid Game Specialist** (300-700 admits): Balance efficiency and constraints
- **End Game Specialist** (700-1000 admits): Optimize final decisions
- **Panic Mode Specialist**: Handle critical constraint deficits

## Phase 4: Hybrid Algorithm-Transformer System

### 4A. Create Meta-Controller
```python
class MetaController:
    def decide_strategy(self, game_state):
        if game_state.admitted < 100:
            return "rbcr"  # Best early game performance
        elif constraint_deficit_critical():
            return "panic_transformer"
        elif game_state.admitted > 850:
            return "endgame_transformer"  
        else:
            return "main_transformer"
```

### 4B. Algorithm-Guided Transformer
Use algorithmic strategies as "teachers":
- Run RBCR/Perfect/Apex in parallel as advisors
- Transformer learns when to follow which algorithm
- Combines best aspects of all strategies
- Can override algorithms when it has better insights

## Phase 5: Iterative Improvement Loop

### 5A. Self-Play Evolution
```python
while current_best > 720:  # Target: beat 716 record
    # 1. Generate 1000 games with current best model
    # 2. Select top 10% (lowest rejections) 
    # 3. Train new transformer on these elite games
    # 4. Test new transformer against benchmarks
    # 5. If better, update current_best and repeat
```

### 5B. Genetic Algorithm for Strategy Parameters
- Mutate strategy parameters (temperature, thresholds, etc.)
- Run tournaments between parameter variations
- Breed best performers using crossover
- Generate training data from tournament winners

## Phase 6: Implementation Timeline

### Week 1: Data Generation
1. **Run mass generation script** (48-72 hours execution time)
2. **Filter and analyze results** 
3. **Create data augmentation pipeline**

### Week 2: Architecture Development  
4. **Implement dual-head transformer**
5. **Create ensemble training pipeline**
6. **Develop meta-controller system**

### Week 3: Training & Testing
7. **Train improved models on new data**
8. **Test ensemble approach**
9. **Implement and test meta-controller**

### Week 4: Evolution & Optimization
10. **Run self-play evolution loop**
11. **Genetic algorithm optimization**
12. **Final benchmarking and analysis**

## Expected Outcomes

### Short Term (After Phase 1-2)
- **Training Data**: 10,000+ high-quality games
- **Ultra-Elite Games**: 500+ games with <750 rejections  
- **Transformer Performance**: 780-820 rejections (improvement from 880)
- **Success Rate**: Maintained at 95%+

### Medium Term (After Phase 3-4)
- **Hybrid System**: 740-780 rejections consistently
- **Success Rate**: 90%+ with better rejection counts
- **Architecture**: Multi-specialist ensemble system
- **Meta-Controller**: Intelligent strategy selection

### Long Term (After Phase 5-6)
- **Record Performance**: Approach or beat 716 record
- **Self-Improving**: System that continues to evolve
- **Optimal Discovery**: Novel strategies beyond current algorithms
- **Benchmark**: New state-of-the-art for Berghain Challenge

## Key Scripts to Create

1. **`mass_generate_elite_data.py`** - Enhanced parallel mass game generation
2. **`augment_training_data.py`** - Create synthetic optimal games  
3. **`train_ensemble_transformer.py`** - Multi-model training pipeline
4. **`meta_controller.py`** - Strategy orchestration system
5. **`self_play_evolution.py`** - Iterative improvement loop
6. **`genetic_optimizer.py`** - Parameter evolution system

## Success Metrics

- **Primary Goal**: Achieve <720 rejections consistently  
- **Secondary Goal**: Beat 716 record at least once
- **Tertiary Goal**: 95%+ success rate maintained
- **Research Goal**: Discover novel algorithmic insights

## Risk Mitigation

- **Data Quality**: Multiple filtering and validation steps
- **Training Stability**: Ensemble approach reduces single-point failures
- **Performance Regression**: Always maintain baseline comparisons
- **Resource Management**: Staged approach allows stopping at any successful phase

---

## Execution Command

Start the masterplan with:
```bash
./generate_training_data.sh
```

Then proceed through phases based on results analysis.

**Status**: Ready for Phase 1 execution
**Estimated Total Time**: 4-6 weeks
**Resource Requirements**: 16,000+ game generations, significant compute for training
**Success Probability**: High (based on proven algorithmic foundations)