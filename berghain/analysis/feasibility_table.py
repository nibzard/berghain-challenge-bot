import os
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np


class FeasibilityOracle:
    """Monte-Carlo feasibility oracle with on-demand caching.

    Models arrivals as i.i.d. draws over four categories with probabilities:
      - both (y=1,w=1) : p11
      - young-only     : p10
      - well-only      : p01
      - neither        : p00

    For a state (dy, dw, S_remaining), returns True if with probability >= 1-delta
    we can still satisfy both deficits by the end (i.e., future helpful counts
    meet dy and dw), otherwise False.
    """

    def __init__(self, p11: float, p10: float, p01: float, p00: float,
                 delta: float = 0.01, samples: int = 5000,
                 cache_dir: str = "berghain/config/feasibility",
                 cache_key: str = "scenario_1"):
        probs = np.array([p11, p10, p01, p00], dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / max(1e-12, probs.sum())
        self.probs = probs
        self.delta = float(delta)
        self.samples = int(samples)
        self.cache: Dict[Tuple[int, int, int], bool] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{cache_key}.pkl"
        self._load_cache()

    def _load_cache(self):
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    self.cache.update(obj)
        except Exception:
            pass

    def _save_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def is_feasible(self, dy: int, dw: int, s: int) -> bool:
        key = (int(dy), int(dw), int(s))
        if key in self.cache:
            return self.cache[key]
        if dy <= 0 and dw <= 0:
            self.cache[key] = True
            return True
        if s <= 0:
            self.cache[key] = False
            return False
        # Vectorized multinomial simulation
        draws = np.random.multinomial(s, self.probs, size=self.samples)
        both = draws[:, 0]
        y_only = draws[:, 1]
        w_only = draws[:, 2]
        help_y = both + y_only
        help_w = both + w_only
        ok = np.logical_and(help_y >= dy, help_w >= dw)
        prob_ok = ok.mean()
        feasible = prob_ok >= (1.0 - self.delta)
        self.cache[key] = bool(feasible)
        # Periodically save cache
        if len(self.cache) % 200 == 0:
            self._save_cache()
        return feasible

