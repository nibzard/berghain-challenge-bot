# ABOUTME: Local simulator client to run strategies without the live API
# ABOUTME: Mimics BerghainAPIClient interface for start/get/submit/close

import uuid
import random
import logging
from typing import Dict, Optional, Any, Tuple

from .domain import GameState, Person, Constraint, AttributeStatistics, GameStatus
from ..config import ConfigManager


logger = logging.getLogger(__name__)


class LocalSimulatorClient:
    """Local simulator that mirrors the BerghainAPIClient interface.

    Generates people based on scenario expected_frequencies and (optionally)
    expected_correlations. For two-attribute scenarios, uses the requested
    correlation to construct a joint distribution; otherwise falls back to
    independent sampling by marginal frequencies.
    """

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.config = ConfigManager()

    # --- Public API surface (mirror BerghainAPIClient) ---
    def start_new_game(self, scenario: int) -> GameState:
        scenario_config = self.config.get_scenario_config(scenario)
        if not scenario_config:
            raise ValueError(f"Scenario config not found for id={scenario}")

        name = scenario_config.get('name', f'Scenario {scenario}')
        ef = scenario_config.get('expected_frequencies', {})
        ec = scenario_config.get('expected_correlations', {})
        constraints_cfg = scenario_config.get('constraints', [])

        # Build constraints and statistics
        constraints = [Constraint(c['attribute'], int(c['min_count'])) for c in constraints_cfg]
        statistics = AttributeStatistics(
            frequencies={k: float(v) for k, v in ef.items()},
            correlations={a: {b: float(ec.get(a, {}).get(b, 0.0)) for b in ef.keys()} for a in ef.keys()}
        )

        game_id = str(uuid.uuid4())
        game_state = GameState(
            game_id=game_id,
            scenario=scenario,
            constraints=constraints,
            statistics=statistics
        )

        # Prepare generator for persons
        joint_sampler = self._build_joint_sampler(statistics)
        self._sessions[game_id] = {
            'next_index': 0,
            'sampler': joint_sampler,
        }

        logger.info(f"[Local] Started new game {game_id[:8]} for scenario {scenario} ({name})")
        return game_state

    def get_next_person(self, game_state: GameState, person_index: int) -> Optional[Person]:
        sess = self._sessions.get(game_state.game_id)
        if not sess or game_state.status != GameStatus.RUNNING:
            return None
        idx = sess['next_index']
        attrs = sess['sampler']()
        person = Person(index=idx, attributes=attrs)
        sess['next_index'] = idx + 1
        return person

    def submit_decision(self, game_state: GameState, person: Person, accept: bool) -> Dict[str, Any]:
        """Submit decision and provide next person if running.

        This method intentionally does not mutate game_state counts; BaseSolver
        updates the GameState after this call. We only set 'status' and
        optionally 'nextPerson' to match the live API contract used by the
        executor.
        """
        sess = self._sessions.get(game_state.game_id)
        if not sess or game_state.status != GameStatus.RUNNING:
            return {"status": "failed"}

        # Predict post-decision state to detect completion earlier
        will_admit = 1 if accept else 0
        projected_admitted = game_state.admitted_count + will_admit

        # Projected admitted attributes
        projected_attrs = dict(game_state.admitted_attributes)
        if accept:
            for a, has in person.attributes.items():
                if has:
                    projected_attrs[a] = projected_attrs.get(a, 0) + 1

        # Check completion conditions (constraints met or capacity reached)
        constraints_met = all(
            projected_attrs.get(c.attribute, 0) >= c.min_count for c in game_state.constraints
        )
        capacity_reached = projected_admitted >= game_state.target_capacity

        if constraints_met or capacity_reached:
            # Mark completed; the solver will also mark completed when mapping status
            return {"status": "completed"}

        # Otherwise, continue with next person
        idx = sess['next_index']
        attrs = sess['sampler']()
        next_person = {"personIndex": idx, "attributes": attrs}
        sess['next_index'] = idx + 1
        return {
            "status": "running",
            "nextPerson": next_person,
            # Optionally include counters for parity with API
            "admittedCount": game_state.admitted_count,
            "rejectedCount": game_state.rejected_count,
        }

    def play_turn(self, game_state: GameState, person_index: int, accept: bool) -> Optional[Person]:
        resp = self.submit_decision(game_state, Person(person_index, {}), accept)
        if resp.get("status") == "running" and "nextPerson" in resp:
            np = resp["nextPerson"]
            return Person(np["personIndex"], np["attributes"])
        return None

    def close(self):
        # Nothing to close for local
        pass

    # --- Internal helpers ---
    def _build_joint_sampler(self, stats: AttributeStatistics):
        keys = list(stats.frequencies.keys())
        if len(keys) == 2:
            a, b = keys
            pa = float(stats.frequencies.get(a, 0.0))
            pb = float(stats.frequencies.get(b, 0.0))
            r = float(stats.correlations.get(a, {}).get(b, 0.0))
            return self._sampler_two_attrs(a, b, pa, pb, r)
        # Fallback: independent Bernoulli for each attribute
        return self._sampler_independent(keys, stats.frequencies)

    def _sampler_independent(self, keys, freqs):
        def sample() -> Dict[str, bool]:
            return {k: (self.random.random() < float(freqs.get(k, 0.0))) for k in keys}
        return sample

    def _sampler_two_attrs(self, a: str, b: str, pa: float, pb: float, corr: float):
        # Compute joint P11 from correlation (Pearson for Bernoulli)
        import math
        denom = math.sqrt(max(pa*(1-pa), 1e-9) * max(pb*(1-pb), 1e-9))
        p11 = pa*pb + corr * denom
        # Clamp to feasible region
        p11 = max(0.0, min(p11, min(pa, pb)))
        p10 = max(0.0, pa - p11)
        p01 = max(0.0, pb - p11)
        p00 = 1.0 - (p11 + p10 + p01)
        if p00 < 0:
            # Renormalize minimally
            total = max(1e-9, p11 + p10 + p01)
            p11, p10, p01 = (p11/total*0.999, p10/total*0.999, p01/total*0.999)
            p00 = 1.0 - (p11 + p10 + p01)
        cdf = [p00, p00+p10, p00+p10+p01, 1.0]

        def sample() -> Dict[str, bool]:
            u = self.random.random()
            if u < cdf[0]:
                return {a: False, b: False}
            elif u < cdf[1]:
                return {a: True, b: False}
            elif u < cdf[2]:
                return {a: False, b: True}
            else:
                return {a: True, b: True}

        return sample

