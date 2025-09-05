"""Ramanujan-Inspired Mathematical Optimization (RIMO) strategy.

Named after the mathematical genius Srinivasa Ramanujan, this algorithm uses:
- Continued fraction approximations for optimal thresholds
- Number theoretic modular arithmetic for acceptance decisions
- Hardy-Ramanujan asymptotic analysis for capacity estimation
- Golden ratio and Fibonacci sequence for phase transitions
- Partition function approximations for deficit management

Key Mathematical Insights:
1. The problem is over-constrained: need ~1056 optimal people but only 1000 slots
2. Must allow exactly the right amount of "filler" people (neither constraint)
3. Use modular arithmetic with prime number sequences for pseudo-randomness
4. Employ golden ratio (φ = 1.618...) for optimal phase boundaries
"""

import math
from typing import Tuple, List
from ..core import GameState, Person
from ..core.strategy import BaseDecisionStrategy
from .base_solver import BaseSolver


class RamanujanSolver(BaseSolver):
    """Ramanujan-inspired mathematical optimization solver."""
    
    def __init__(self, solver_id: str = "ramanujan", config_manager=None, api_client=None, enable_high_score_check: bool = True):
        from ..config import ConfigManager
        config_manager = config_manager or ConfigManager()
        strategy_config = config_manager.get_strategy_config("ramanujan")
        strategy = RamanujanStrategy(strategy_config.get("parameters", {}))
        super().__init__(strategy, solver_id, enable_high_score_check, api_client)


class RamanujanStrategy(BaseDecisionStrategy):
    def __init__(self, strategy_params: dict = None):
        defaults = {
            # Mathematical constants
            'golden_ratio': 1.6180339887498948,
            'euler_gamma': 0.5772156649015329,
            'ramanujan_constant': 262537412640768744,  # e^(π√163)
            
            # Prime sequence for modular decisions
            'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            
            # Phase transition points using golden ratio
            'phi_early': 0.381966,  # (φ-1)/φ 
            'phi_mid': 0.618034,    # 1/φ
            'phi_late': 0.809017,   # (φ-1)/2 + 0.5
            
            # Hardy-Ramanujan parameters
            'partition_scaling': 2.56,  # Approximation constant
            'asymptotic_constant': 1.73,
            
            # Continued fraction thresholds
            'cf_dual_base': [1, 1, 2, 1, 4, 1, 6, 1, 8],  # Pattern for dual acceptance
            'cf_single_base': [2, 1, 3, 1, 5, 1, 7, 1, 9], # Pattern for single acceptance
            
            # Advanced parameters
            'fibonacci_depth': 12,
            'modular_base': 97,  # Large prime for modular arithmetic
            'convergence_precision': 1e-6,
        }
        if strategy_params:
            defaults.update(strategy_params)
        super().__init__(defaults)
        
        # Precompute mathematical sequences
        self._fibonacci_seq = self._compute_fibonacci(int(self.params['fibonacci_depth']))
        self._prime_index = 0
        self._decision_count = 0
        
        # Continued fraction convergents
        self._dual_cf_values = self._compute_cf_convergents(self.params['cf_dual_base'])
        self._single_cf_values = self._compute_cf_convergents(self.params['cf_single_base'])

    @property
    def name(self) -> str:
        return "Ramanujan"

    def _compute_fibonacci(self, n: int) -> List[int]:
        """Compute Fibonacci sequence up to n terms."""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    def _compute_cf_convergents(self, cf_terms: List[int]) -> List[float]:
        """Compute continued fraction convergents."""
        if not cf_terms:
            return [1.0]
        
        convergents = []
        h_prev, h_curr = 1, cf_terms[0]
        k_prev, k_curr = 0, 1
        
        convergents.append(float(h_curr) / float(k_curr))
        
        for i in range(1, len(cf_terms)):
            h_next = cf_terms[i] * h_curr + h_prev
            k_next = cf_terms[i] * k_curr + k_prev
            
            convergents.append(float(h_next) / float(k_next))
            
            h_prev, h_curr = h_curr, h_next
            k_prev, k_curr = k_curr, k_next
        
        return convergents

    def _ramanujan_partition_approximation(self, n: int) -> float:
        """Hardy-Ramanujan partition approximation."""
        if n <= 0:
            return 1.0
        
        scaling = float(self.params['partition_scaling'])
        return math.exp(scaling * math.sqrt(n)) / (4.0 * n * math.sqrt(3))

    def _golden_ratio_threshold(self, progress: float, phase: str) -> float:
        """Compute golden ratio based thresholds."""
        φ = float(self.params['golden_ratio'])
        
        if phase == 'early':
            base = float(self.params['phi_early'])
        elif phase == 'mid': 
            base = float(self.params['phi_mid'])
        else:  # late
            base = float(self.params['phi_late'])
        
        # Modulate by inverse golden ratio powers
        modulation = (1 / φ) ** (progress * 5)
        return base * (1 + modulation)

    def _modular_arithmetic_decision(self, person_index: int, constraint_state: tuple) -> bool:
        """Use modular arithmetic with primes for pseudo-random decisions."""
        primes = self.params['primes']
        mod_base = int(self.params['modular_base'])
        
        # Create hash from person index and constraint state
        state_hash = hash(constraint_state) % mod_base
        prime_idx = self._prime_index % len(primes)
        current_prime = primes[prime_idx]
        
        # Number theoretic decision
        decision_value = (person_index * current_prime + state_hash) % mod_base
        threshold = mod_base * float(self.params['phi_early'])
        
        # Advance prime index for next decision
        self._prime_index = (self._prime_index + 1) % len(primes)
        
        return decision_value < threshold

    def should_accept(self, person: Person, game_state: GameState) -> Tuple[bool, str]:
        self._decision_count += 1
        
        if self.is_emergency_mode(game_state):
            return True, "ramanujan_emergency"

        attrs = self.get_person_constraint_attributes(person, game_state)
        keys = [c.attribute for c in game_state.constraints]
        a_y, a_w = (keys + [None, None])[:2]
        
        # Current state analysis
        capacity_progress = game_state.capacity_ratio
        constraint_progress = game_state.constraint_progress()
        shortage = game_state.constraint_shortage()
        
        # Determine phase using golden ratio transitions
        φ = float(self.params['golden_ratio'])
        if capacity_progress < float(self.params['phi_early']):
            phase = 'early'
        elif capacity_progress < float(self.params['phi_mid']):
            phase = 'mid'  
        else:
            phase = 'late'
        
        is_dual = len(attrs) >= 2
        is_single = len(attrs) == 1
        is_filler = len(attrs) == 0
        
        # === DUAL PEOPLE (both attributes) ===
        if is_dual:
            # Always accept duals - they're mathematically optimal
            return True, "ramanujan_dual_optimal"
        
        # === SINGLE ATTRIBUTE PEOPLE ===
        if is_single:
            is_y = (a_y in attrs)
            is_w = (a_w in attrs)
            
            # Check if we actually need this attribute
            if is_y and shortage.get(a_y, 0) <= 0:
                # Already satisfied young constraint
                deficit_ratio = 0.0
            elif is_w and shortage.get(a_w, 0) <= 0:
                # Already satisfied well_dressed constraint
                deficit_ratio = 0.0
            else:
                # We need this attribute
                relevant_shortage = shortage.get(a_y if is_y else a_w, 0)
                deficit_ratio = min(1.0, relevant_shortage / 300.0)  # Normalize to [0,1]
            
            # Use continued fraction convergents for threshold
            cf_values = self._single_cf_values
            cf_index = min(len(cf_values) - 1, int(capacity_progress * len(cf_values)))
            base_threshold = cf_values[cf_index]
            
            # Modulate by deficit and phase
            golden_modulation = self._golden_ratio_threshold(capacity_progress, phase)
            deficit_boost = 1.0 + deficit_ratio * 2.0  # Boost when we need this attribute
            
            final_threshold = base_threshold * golden_modulation * deficit_boost
            
            # Hardy-Ramanujan partition function influence
            remaining_capacity = game_state.target_capacity - game_state.admitted_count
            partition_weight = self._ramanujan_partition_approximation(remaining_capacity)
            threshold_adjustment = min(2.0, 1.0 + math.log(1 + partition_weight))
            
            final_threshold *= threshold_adjustment
            
            # Decision based on modular arithmetic
            constraint_state = (shortage.get(a_y, 0), shortage.get(a_w, 0), remaining_capacity)
            if self._modular_arithmetic_decision(person.index, constraint_state):
                if final_threshold > 0.7:  # High confidence
                    return True, f"ramanujan_single_{'y' if is_y else 'w'}_high"
                elif final_threshold > 0.4:  # Medium confidence
                    return True, f"ramanujan_single_{'y' if is_y else 'w'}_med"
                else:
                    return False, f"ramanujan_single_{'y' if is_y else 'w'}_low"
            else:
                return False, f"ramanujan_single_{'y' if is_y else 'w'}_mod_reject"
        
        # === FILLER PEOPLE (no attributes) ===
        if is_filler:
            # Critical insight: we MUST allow some filler people due to overconstrained problem
            
            # Use Fibonacci ratios for filler acceptance
            fib_seq = self._fibonacci_seq
            if len(fib_seq) >= 2:
                fib_ratio = fib_seq[min(len(fib_seq)-1, max(0, int(capacity_progress * len(fib_seq))))]
                fib_previous = fib_seq[max(0, min(len(fib_seq)-2, int(capacity_progress * len(fib_seq)) - 1))]
                golden_approximation = fib_ratio / max(1, fib_previous)
            else:
                golden_approximation = φ
            
            # Filler acceptance rate based on constraint satisfaction
            min_progress = min(constraint_progress.values()) if constraint_progress else 0.0
            max_progress = max(constraint_progress.values()) if constraint_progress else 0.0
            progress_balance = 1.0 - abs(min_progress - max_progress)  # Reward balanced progress
            
            # Allow more filler when constraints are balanced and we're not too late
            filler_rate = (1.0 / golden_approximation) * progress_balance * (1.0 - capacity_progress)**2
            filler_rate = max(0.01, min(0.15, filler_rate))  # Clamp to reasonable range
            
            # Use Euler's gamma constant for final decision threshold
            gamma = float(self.params['euler_gamma'])
            euler_threshold = (self._decision_count * gamma) % 1.0
            
            if euler_threshold < filler_rate:
                return True, "ramanujan_filler_euler"
            else:
                return False, "ramanujan_filler_reject"
        
        # Default case (should not reach here)
        return False, "ramanujan_default_reject"