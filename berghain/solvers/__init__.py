# ABOUTME: Solver module exports
# ABOUTME: Clean strategy implementations for the Berghain Challenge

from .base_solver import BaseSolver
from .rarity_solver import RarityWeightedStrategy
from .adaptive_solver import AdaptiveStrategy
from .balanced_solver import BalancedStrategy
from .greedy_solver import GreedyConstraintStrategy
from .diversity_solver import DiversityFirstStrategy
from .quota_solver import QuotaTrackerStrategy
from .dual_deficit_solver import DualDeficitController
from .rbcr_solver import RBCRStrategy
from .rbcr2_solver import RBCR2Strategy
from .dvo_solver import DVOSolver, DVOStrategy
from .ramanujan_solver import RamanujanSolver, RamanujanStrategy
from .ultimate_solver import UltimateSolver, UltimateStrategy
from .ultimate2_solver import Ultimate2Solver, Ultimate2Strategy
from .ultimate3_solver import Ultimate3Solver, Ultimate3Strategy
from .ultimate3h_solver import Ultimate3HSolver, Ultimate3HStrategy
from .optimal_control_solver import OptimalControlSolver, OptimalControlStrategy
from .optimal_control_final import OptimalControlFinalSolver, OptimalControlFinalStrategy
from .optimal_control_safe import OptimalControlSafeSolver, OptimalControlSafeStrategy
from .perfect_solver import PerfectSolver, PerfectStrategy
from .mec_solver import MecSolver, MecStrategy

__all__ = [
    "BaseSolver",
    "RarityWeightedStrategy", 
    "AdaptiveStrategy",
    "BalancedStrategy",
    "GreedyConstraintStrategy",
    "DiversityFirstStrategy",
    "QuotaTrackerStrategy",
    "DualDeficitController",
    "RBCRStrategy",
    "RBCR2Strategy",
    "DVOSolver",
    "DVOStrategy",
    "RamanujanSolver",
    "RamanujanStrategy",
    "UltimateSolver",
    "UltimateStrategy",
    "Ultimate2Solver",
    "Ultimate2Strategy",
    "Ultimate3Solver", 
    "Ultimate3Strategy",
    "Ultimate3HSolver",
    "Ultimate3HStrategy",
    "OptimalControlSolver",
    "OptimalControlStrategy",
    "OptimalControlFinalSolver",
    "OptimalControlFinalStrategy",
    "OptimalControlSafeSolver",
    "OptimalControlSafeStrategy",
    "PerfectSolver",
    "PerfectStrategy",
    "MecSolver",
    "MecStrategy",
]
