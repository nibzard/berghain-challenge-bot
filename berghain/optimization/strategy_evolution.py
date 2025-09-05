# ABOUTME: Strategy evolution and generation system
# ABOUTME: Creates new strategies by mutating successful ones

import logging
import random
import copy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """Genetic representation of a strategy."""
    name: str
    base_strategy: str  # conservative, aggressive, adaptive, etc.
    parameters: Dict[str, Any]
    generation: int = 0
    parent_ids: List[str] = None
    fitness_score: float = 0.0
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class StrategyEvolution:
    """Evolves strategies using genetic algorithm principles."""
    
    def __init__(self):
        self.population: Dict[str, StrategyGenome] = {}
        self.generation = 0
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        
        # Parameter mutation ranges
        self.parameter_ranges = {
            "ultra_rare_threshold": (0.01, 0.1),
            "phase1_multi_attr_only": (True, False),
            "deficit_panic_threshold": (0.6, 0.9),
            "multi_attr_bonus": (1.1, 3.0),
            "young_weight": (0.5, 2.0),
            "well_dressed_weight": (0.5, 2.0),
            "rarity_multiplier": (1.0, 5.0),
            "admission_threshold": (0.3, 0.8),
        }
    
    def create_base_population(self, base_strategies: List[Dict[str, Any]]) -> List[StrategyGenome]:
        """Create initial population from base strategies."""
        genomes = []
        
        for i, strategy_config in enumerate(base_strategies):
            strategy_name = strategy_config.get('name', f'base_{i}')
            base_type = strategy_config.get('type', 'conservative')
            parameters = strategy_config.get('parameters', {})
            
            genome = StrategyGenome(
                name=strategy_name,
                base_strategy=base_type,
                parameters=copy.deepcopy(parameters),
                generation=0
            )
            
            self.population[strategy_name] = genome
            genomes.append(genome)
            
        logger.info(f"🧬 Created base population: {len(genomes)} strategies")
        return genomes
    
    def mutate_strategy(self, parent: StrategyGenome, mutation_intensity: float = 1.0) -> StrategyGenome:
        """Create a mutated version of a strategy."""
        new_params = copy.deepcopy(parent.parameters)
        mutation_count = 0
        
        # Mutate parameters
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if random.random() < self.mutation_rate * mutation_intensity:
                if isinstance(min_val, bool):
                    # Boolean parameter
                    new_params[param_name] = random.choice([True, False])
                else:
                    # Numeric parameter
                    if param_name in new_params:
                        current_val = new_params[param_name]
                        # Add gaussian noise
                        noise = random.gauss(0, (max_val - min_val) * 0.1)
                        new_val = max(min_val, min(max_val, current_val + noise))
                        new_params[param_name] = new_val
                        mutation_count += 1
                    else:
                        # Add new parameter
                        new_params[param_name] = random.uniform(min_val, max_val)
                        mutation_count += 1
        
        # Create new strategy name
        self.generation += 1
        new_name = f"{parent.base_strategy}_gen{self.generation}_{random.randint(1000, 9999)}"
        
        mutant = StrategyGenome(
            name=new_name,
            base_strategy=parent.base_strategy,
            parameters=new_params,
            generation=self.generation,
            parent_ids=[parent.name]
        )
        
        logger.info(f"🧬 Mutated {parent.name} -> {new_name} ({mutation_count} changes)")
        return mutant
    
    def crossover_strategies(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Create offspring by crossing two strategies."""
        new_params = {}
        
        # Combine parameters from both parents
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        for param in all_params:
            # Randomly choose parameter from either parent
            if param in parent1.parameters and param in parent2.parameters:
                if random.random() < 0.5:
                    new_params[param] = parent1.parameters[param]
                else:
                    new_params[param] = parent2.parameters[param]
            elif param in parent1.parameters:
                new_params[param] = parent1.parameters[param]
            else:
                new_params[param] = parent2.parameters[param]
        
        # Choose dominant base strategy
        base_strategy = random.choice([parent1.base_strategy, parent2.base_strategy])
        
        self.generation += 1
        new_name = f"cross_{base_strategy}_gen{self.generation}_{random.randint(1000, 9999)}"
        
        offspring = StrategyGenome(
            name=new_name,
            base_strategy=base_strategy,
            parameters=new_params,
            generation=self.generation,
            parent_ids=[parent1.name, parent2.name]
        )
        
        logger.info(f"🧬 Crossed {parent1.name} + {parent2.name} -> {new_name}")
        return offspring
    
    def evolve_population(self, performance_data: Dict[str, float], 
                         population_size: int = 10) -> List[StrategyGenome]:
        """Evolve the population based on performance."""
        # Update fitness scores
        for strategy_id, fitness in performance_data.items():
            if strategy_id in self.population:
                self.population[strategy_id].fitness_score = fitness
        
        # Select best performers
        sorted_population = sorted(
            self.population.values(), 
            key=lambda x: x.fitness_score, 
            reverse=True
        )
        
        elite_count = max(2, population_size // 4)  # Keep top 25%
        elite = sorted_population[:elite_count]
        
        new_generation = []
        
        # Keep elite
        for genome in elite:
            new_generation.append(genome)
        
        # Generate new offspring
        while len(new_generation) < population_size:
            if random.random() < self.crossover_rate and len(elite) >= 2:
                # Crossover
                parent1, parent2 = random.sample(elite, 2)
                offspring = self.crossover_strategies(parent1, parent2)
                
                # Possibly mutate the offspring
                if random.random() < self.mutation_rate:
                    offspring = self.mutate_strategy(offspring, 0.5)
                    
            else:
                # Mutation
                parent = random.choice(elite)
                offspring = self.mutate_strategy(parent)
            
            new_generation.append(offspring)
            self.population[offspring.name] = offspring
        
        logger.info(f"🧬 Evolved generation {self.generation}: {len(new_generation)} strategies")
        return new_generation
    
    def get_strategy_config(self, genome: StrategyGenome) -> Dict[str, Any]:
        """Convert genome to strategy configuration."""
        return {
            "name": genome.name,
            "type": genome.base_strategy,
            "parameters": genome.parameters,
            "generation": genome.generation,
            "parent_ids": genome.parent_ids
        }
